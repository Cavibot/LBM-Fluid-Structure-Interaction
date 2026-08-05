# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P6 acceptance tests for authoritative geometry and surface tension."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    LbmDomain,
    LbmModel,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, OPPOSITE, W
from wanphys._src.fluid.fluid_grid.lbm.vof import VofCellType, VofGridState
from wanphys._src.fluid.fluid_grid.lbm.vof.solver.geometry import (
    VofInterfaceGeometry,
    plic_cube_volume,
    plic_plane_offset,
    validate_authoritative_geometry,
)
from wanphys._src.fluid.fluid_grid.lbm.vof.solver.kernels.geometry import (
    plic_cube_offset,
)


@wp.kernel
def _plic_offset_test_kernel(
    fill: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    offset: wp.array(dtype=float),
) -> None:
    index = wp.tid()
    offset[index] = plic_cube_offset(fill[index], normal[index])


def _model(
    shape: tuple[int, int, int],
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    pressure: float = 1.0 / 3.0,
    surface_tension: float = 0.0,
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding="fullf",
        collision="srt",
        interface_model="vof",
        bc_periodic=periodic,
        vof_atmosphere_pressure=pressure,
        vof_surface_tension=surface_tension,
        enforce_population_positivity=False,
    )


def _layered_phi(shape: tuple[int, int, int], value: float = 0.5) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    split = shape[0] // 2
    result[:split] = 1.0
    result[split] = np.float32(value)
    return result


def _equilibrium(
    q: int,
    rho: float,
    velocity: tuple[float, float, float],
) -> float:
    ux, uy, uz = velocity
    cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz
    u2 = ux * ux + uy * uy + uz * uz
    return W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


class TestVofP6StateAndEpoch(unittest.TestCase):
    def test_geometry_storage_clear_clone_and_epoch_lifecycle(self) -> None:
        domain = LbmDomain(_model((7, 3, 3)))
        empty = domain.create_state()
        assert empty.vof is not None
        self.assertEqual(empty.vof.normal.dtype, wp.vec3)
        self.assertEqual(empty.vof.plic_offset.dtype, wp.float32)
        self.assertEqual(empty.vof.curvature.dtype, wp.float32)
        self.assertEqual((empty.vof.epoch, empty.vof.geometry_epoch), (-1, -1))

        state = domain.initialize_vof(_layered_phi((7, 3, 3)))
        assert state.vof is not None
        self.assertEqual((state.vof.epoch, state.vof.geometry_epoch), (0, 0))
        clone = state.clone()
        assert clone.vof is not None
        for name in (
            "normal",
            "plic_offset",
            "curvature",
        ):
            source = getattr(state.vof, name)
            target = getattr(clone.vof, name)
            self.assertNotEqual(int(source.ptr), int(target.ptr))
            np.testing.assert_array_equal(source.numpy(), target.numpy())
        self.assertEqual((clone.vof.epoch, clone.vof.geometry_epoch), (0, 0))
        clone.clear()
        self.assertEqual((clone.vof.epoch, clone.vof.geometry_epoch), (-1, -1))
        self.assertTrue(np.all(clone.vof.normal.numpy() == 0.0))

    def test_integrated_step_advances_vof_and_geometry_epoch_once(self) -> None:
        domain = LbmDomain(_model((7, 3, 3)))
        domain.initialize_vof(_layered_phi((7, 3, 3)))
        for expected in range(1, 4):
            domain.step(1.0)
            assert domain.state.vof is not None
            self.assertEqual(domain.state.vof.epoch, expected)
            self.assertEqual(domain.state.vof.geometry_epoch, expected)
            validate_authoritative_geometry(domain.state.vof)

    def test_stale_geometry_fails_before_state_or_population_mutation(self) -> None:
        domain = LbmDomain(_model((7, 3, 3), surface_tension=0.01))
        state = domain.initialize_vof(_layered_phi((7, 3, 3)))
        assert state.vof is not None
        state.vof.geometry_epoch = -1
        populations_before = state.f_post.numpy().copy()
        with self.assertRaisesRegex(ValueError, "geometry is stale"):
            domain.solver.compute_vof_surface_populations(state)
        np.testing.assert_array_equal(state.f_post.numpy(), populations_before)


class TestVofP6NormalPlicCurvature(unittest.TestCase):
    def test_axis_plane_has_liquid_to_gas_normal_exact_plic_and_zero_curvature(
        self,
    ) -> None:
        shape = (9, 5, 5)
        domain = LbmDomain(_model(shape))
        state = domain.initialize_vof(_layered_phi(shape, value=0.25))
        assert state.vof is not None
        interface = state.vof.cell_type.numpy() == int(VofCellType.INTERFACE)
        expected_normal = np.zeros((*shape, 3), dtype=np.float32)
        expected_normal[interface, 0] = 1.0
        np.testing.assert_allclose(
            state.vof.normal.numpy()[interface],
            expected_normal[interface],
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            state.vof.plic_offset.numpy()[interface],
            -0.25,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            state.vof.curvature.numpy()[interface],
            0.0,
            atol=1.0e-7,
        )

    def test_degenerate_isolated_interface_has_finite_canonical_geometry(
        self,
    ) -> None:
        shape = (3, 3, 3)
        phi = np.zeros(shape, dtype=np.float32)
        phi[1, 1, 1] = 0.5
        state = LbmDomain(_model(shape)).initialize_vof(phi)
        assert state.vof is not None
        center = (1, 1, 1)
        np.testing.assert_array_equal(
            state.vof.normal.numpy()[center],
            np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(float(state.vof.plic_offset.numpy()[center]), 0.0)
        self.assertEqual(float(state.vof.curvature.numpy()[center]), 0.0)

    def test_random_plic_planes_close_requested_unit_cube_volume(self) -> None:
        rng = np.random.default_rng(601)
        count = 96
        normal = rng.normal(size=(count, 3)).astype(np.float32)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        fill = rng.uniform(1.0e-4, 1.0 - 1.0e-4, size=count).astype(np.float32)
        fill[:5] = np.array([0.0001, 0.01, 0.5, 0.99, 0.9999], np.float32)
        normal[:3] = np.eye(3, dtype=np.float32)
        fill_device = wp.array(fill, dtype=float, device="cpu")
        normal_device = wp.array(normal, dtype=wp.vec3, device="cpu")
        offset_device = wp.zeros(count, dtype=float, device="cpu")
        wp.launch(
            _plic_offset_test_kernel,
            dim=count,
            inputs=[fill_device, normal_device, offset_device],
            device="cpu",
        )
        offsets = offset_device.numpy()
        reconstructed = np.array(
            [
                plic_cube_volume(vector, float(offset))
                for vector, offset in zip(normal, offsets, strict=True)
            ]
        )
        np.testing.assert_allclose(reconstructed, fill, atol=2.0e-5, rtol=0.0)
        self.assertAlmostEqual(float(offsets[2]), 0.0, places=7)

    def test_near_axis_plic_inverse_avoids_float32_cancellation(self) -> None:
        normal = np.array(
            [
                [1.12466514e-4, 1.12436712e-4, 1.0],
                [1.35063205e-4, -5.32898689e-1, -8.46179041e-1],
            ],
            dtype=np.float32,
        )
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        fill = np.array([0.4994719923, 0.4773713218], dtype=np.float32)
        fill_device = wp.array(fill, dtype=float, device="cpu")
        normal_device = wp.array(normal, dtype=wp.vec3, device="cpu")
        offset_device = wp.zeros(len(fill), dtype=float, device="cpu")

        wp.launch(
            _plic_offset_test_kernel,
            dim=len(fill),
            inputs=[fill_device, normal_device, offset_device],
            device="cpu",
        )

        offsets = offset_device.numpy()
        host_offsets = np.array(
            [
                plic_plane_offset(float(value), vector)
                for value, vector in zip(fill, normal, strict=True)
            ]
        )
        reconstructed = np.array(
            [
                plic_cube_volume(vector, float(offset))
                for vector, offset in zip(normal, offsets, strict=True)
            ]
        )
        np.testing.assert_allclose(reconstructed, fill, atol=2.0e-5, rtol=0.0)
        np.testing.assert_allclose(offsets, host_offsets, atol=2.0e-7, rtol=0.0)
        self.assertLess(abs(float(offsets[0])), 1.0e-3)

    def test_convex_sphere_curvature_is_negative_and_lattice_convergent(
        self,
    ) -> None:
        mean_errors: list[float] = []
        for radius in (4, 8, 12):
            size = 2 * radius + 7
            center = 0.5 * (size - 1)
            coordinates = np.indices((size, size, size), dtype=np.float32)
            distance = np.sqrt(
                sum((axis - center) ** 2 for axis in coordinates)
            )
            phi = np.clip(radius + 0.5 - distance, 0.0, 1.0).astype(
                np.float32
            )
            cell_type = np.where(
                phi <= 0.0,
                int(VofCellType.GAS),
                np.where(
                    phi >= 1.0,
                    int(VofCellType.LIQUID),
                    int(VofCellType.INTERFACE),
                ),
            ).astype(np.uint8)
            state = VofGridState((size, size, size), wp.get_device("cpu"))
            state.phi.assign(phi)
            state.cell_type.assign(cell_type)
            state.epoch = 0
            VofInterfaceGeometry(
                (size, size, size),
                wp.get_device("cpu"),
                (0, 0, 0),
            ).compute(state)
            values = state.curvature.numpy()[
                cell_type == int(VofCellType.INTERFACE)
            ]
            self.assertLess(float(np.mean(values)), 0.0)
            mean_errors.append(abs(float(np.mean(values)) + 1.0 / radius))
        self.assertLess(mean_errors[1], mean_errors[0])
        self.assertLess(mean_errors[2], mean_errors[1])


class TestVofP6SurfaceTension(unittest.TestCase):
    def test_equation12_link_matches_independent_laplace_density(self) -> None:
        shape = (5, 3, 3)
        phi = _layered_phi(shape)
        center = (2, 1, 1)
        gamma = 0.02
        curvature = -0.25
        pressure = 1.0 / 3.0
        velocity = (0.04, -0.02, 0.01)
        domain = LbmDomain(
            _model(shape, pressure=pressure, surface_tension=gamma)
        )
        state = domain.initialize_vof(phi, u0=velocity)
        assert state.vof is not None
        curvature_field = state.vof.curvature.numpy()
        curvature_field[center] = np.float32(curvature)
        state.vof.curvature.assign(curvature_field)
        populations = state.f_post.numpy().reshape((19, *shape)).copy()
        for q in range(19):
            populations[(q,) + center] = np.float32(0.013 + 0.002 * q)
        state.f_post.assign(populations.reshape(-1))

        actual = (
            domain.solver.compute_vof_surface_populations(state)
            .numpy()
            .reshape((19, *shape))
        )
        rho_g = 3.0 * (pressure - 2.0 * gamma * curvature)
        self.assertGreater(rho_g, 1.0)
        for q in range(1, 19):
            if CX[q] != -1:
                continue
            opposite = OPPOSITE[q]
            expected = (
                _equilibrium(q, rho_g, velocity)
                + _equilibrium(opposite, rho_g, velocity)
                - float(populations[(opposite,) + center])
            )
            self.assertAlmostEqual(
                float(actual[(q,) + center]),
                expected,
                places=6,
            )

    def test_gamma_zero_is_bitwise_identical_for_any_finite_curvature(self) -> None:
        shape = (5, 3, 3)
        phi = _layered_phi(shape)
        center = (2, 1, 1)
        domain = LbmDomain(_model(shape, surface_tension=0.0))
        state = domain.initialize_vof(phi)
        assert state.vof is not None
        baseline = (
            domain.solver.compute_vof_surface_populations(state).numpy().copy()
        )
        curvature = state.vof.curvature.numpy()
        curvature[center] = np.float32(0.875)
        state.vof.curvature.assign(curvature)
        actual = domain.solver.compute_vof_surface_populations(state).numpy()
        np.testing.assert_array_equal(actual, baseline)

    def test_nonpositive_laplace_density_fails_before_state_mutation(self) -> None:
        shape = (5, 3, 3)
        phi = _layered_phi(shape)
        center = (2, 1, 1)
        domain = LbmDomain(_model(shape, surface_tension=1.0))
        state = domain.initialize_vof(phi)
        assert state.vof is not None
        curvature = state.vof.curvature.numpy()
        curvature[center] = np.float32(0.5)
        state.vof.curvature.assign(curvature)
        state_before = state.f_post.numpy().copy()
        with self.assertRaisesRegex(ValueError, "positive rho_g"):
            domain.solver.compute_vof_surface_populations(state)
        np.testing.assert_array_equal(state.f_post.numpy(), state_before)

    def test_nonzero_gamma_planar_interface_remains_mass_conservative(self) -> None:
        shape = (9, 4, 4)
        domain = LbmDomain(_model(shape, surface_tension=0.02))
        initial = domain.initialize_vof(_layered_phi(shape))
        assert initial.vof is not None
        initial_mass = float(np.sum(initial.vof.mass.numpy(), dtype=np.float64))
        for _ in range(5):
            domain.step(1.0)
        final = domain.state
        assert final.vof is not None
        self.assertEqual(final.vof.epoch, 5)
        self.assertEqual(final.vof.geometry_epoch, 5)
        self.assertAlmostEqual(
            float(np.sum(final.vof.mass.numpy(), dtype=np.float64)),
            initial_mass,
            places=5,
        )
        self.assertTrue(np.all(np.isfinite(final.f_post.numpy())))


if __name__ == "__main__":
    unittest.main()
