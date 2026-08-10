# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bulk momentum-strain reconstruction for full Bogner FSL closure."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm.constants import D3Q27_DIRECTIONS
from wanphys._src.fluid.fluid_grid.home_lbm.model import HomeLbmModel
from wanphys._src.fluid.fluid_grid.home_lbm.state import HomeLbmState
from wanphys._src.fluid.fluid_grid.home_lbm.vof.fsl_stress import (
    HomeFreeFslBulkStrainBuilder,
    HomeFreeFslStressClosure,
    fsl_normal_strain_targets,
    reconstruct_fsl_bulk_strain,
)


def _affine_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (9, 8, 7)
    active = np.zeros(shape, dtype=bool)
    active[:6] = True
    base = np.asarray((0.021, -0.014, 0.009), dtype=np.float64)
    gradient = np.asarray(
        (
            (0.004, -0.002, 0.001),
            (0.003, -0.005, 0.002),
            (-0.0015, 0.0005, 0.006),
        ),
        dtype=np.float64,
    )
    coordinates = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0)
    momentum = base + np.einsum("ab,...b->...a", gradient, coordinates)
    rho = 0.98 + 0.002 * coordinates[..., 0] + 0.001 * coordinates[..., 1]
    velocity = momentum / rho[..., None]
    moments = np.empty(shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = momentum
    moments[..., 4] = rho * velocity[..., 0] ** 2
    moments[..., 5] = rho * velocity[..., 1] ** 2
    moments[..., 6] = rho * velocity[..., 2] ** 2
    moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
    moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
    return moments, active, 0.5 * (gradient + gradient.T)


def _upload(state: HomeLbmState, moments: np.ndarray) -> None:
    state.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    )


def _download_strain(builder: HomeFreeFslBulkStrainBuilder) -> np.ndarray:
    components = np.moveaxis(
        builder.bulk_strain.numpy().reshape(6, *builder.res), 0, -1
    ).astype(np.float64)
    result = np.zeros(builder.res + (3, 3), dtype=np.float64)
    result[..., 0, 0] = components[..., 0]
    result[..., 1, 1] = components[..., 1]
    result[..., 2, 2] = components[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = components[..., 3]
    result[..., 0, 2] = result[..., 2, 0] = components[..., 4]
    result[..., 1, 2] = result[..., 2, 1] = components[..., 5]
    return result


class TestHomeFreeFslBulkStrain(unittest.TestCase):
    def test_numpy_affine_momentum_gradient_is_exact_at_active_boundaries(self) -> None:
        moments, active, expected = _affine_case()

        result = reconstruct_fsl_bulk_strain(moments, active)

        np.testing.assert_allclose(
            result.strain[active],
            np.broadcast_to(expected, result.strain[active].shape),
            rtol=0.0,
            atol=4.0e-17,
        )
        self.assertGreaterEqual(int(np.min(result.donor_count[active])), 27)
        self.assertLess(result.maximum_condition_number, 10.0)
        np.testing.assert_array_equal(result.strain[~active], 0.0)

    def _warp_matches_reference(self, device: str) -> None:
        moments, active, _ = _affine_case()
        expected = reconstruct_fsl_bulk_strain(moments, active)
        model = HomeLbmModel(
            fluid_grid_res=active.shape,
            periodic=(False, False, False),
            device=device,
        )
        state = HomeLbmState(model)
        _upload(state, moments)
        active_device = wp.array(
            active.astype(np.int32), dtype=wp.int32, device=device
        )
        builder = HomeFreeFslBulkStrainBuilder(model)

        builder.build(state, active_device)
        wp.synchronize_device(device)

        assert builder.last_diagnostics is not None
        self.assertEqual(
            builder.last_diagnostics.reconstructed_cell_count,
            int(np.count_nonzero(active)),
        )
        np.testing.assert_array_equal(builder.donor_count.numpy(), expected.donor_count)
        np.testing.assert_allclose(
            _download_strain(builder)[active],
            expected.strain[active],
            rtol=5.0e-5,
            atol=5.0e-7,
        )
        np.testing.assert_array_equal(_download_strain(builder)[~active], 0.0)

    def test_warp_matches_reference_on_cpu(self) -> None:
        self._warp_matches_reference("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_matches_reference_on_cuda(self) -> None:
        self._warp_matches_reference("cuda:0")

    def test_constant_momentum_produces_exact_zero_strain(self) -> None:
        shape = (6, 6, 6)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.0
        moments[..., 1:4] = (0.02, -0.01, 0.004)
        active = np.ones(shape, dtype=bool)
        expected = reconstruct_fsl_bulk_strain(moments, active)
        np.testing.assert_array_equal(expected.strain, 0.0)

        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        state = HomeLbmState(model)
        _upload(state, moments)
        builder = HomeFreeFslBulkStrainBuilder(model)
        builder.build(
            state,
            wp.array(active.astype(np.int32), dtype=wp.int32, device="cpu"),
        )
        np.testing.assert_array_equal(builder.bulk_strain.numpy(), 0.0)

    def test_rank_deficient_active_sheet_fails(self) -> None:
        shape = (5, 6, 6)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.0
        coordinates = np.indices(shape, dtype=np.float64)
        moments[..., 1] = 0.01 + 0.002 * coordinates[1]
        active = np.zeros(shape, dtype=bool)
        active[2] = True
        with self.assertRaisesRegex(RuntimeError, "rank deficient"):
            reconstruct_fsl_bulk_strain(moments, active)

        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        state = HomeLbmState(model)
        _upload(state, moments)
        builder = HomeFreeFslBulkStrainBuilder(model)
        with self.assertRaisesRegex(RuntimeError, "rank-deficient"):
            builder.build(
                state,
                wp.array(active.astype(np.int32), dtype=wp.int32, device="cpu"),
            )


class TestHomeFreeFslStressClosure(unittest.TestCase):
    @staticmethod
    def _planar_case():
        shape = (6, 5, 4)
        active = np.zeros(shape, dtype=bool)
        active[:5] = True
        coordinates = np.indices(shape, dtype=np.float64)
        rho = 1.01 + 0.003 * coordinates[0]
        momentum = np.empty(shape + (3,), dtype=np.float64)
        momentum[..., 0] = 0.02 + 0.004 * coordinates[0]
        momentum[..., 1] = -0.01 + 0.002 * coordinates[0]
        momentum[..., 2] = 0.006 - 0.001 * coordinates[0]
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = rho
        moments[..., 1:4] = momentum
        velocity = momentum / rho[..., None]
        moments[..., 4] = rho * velocity[..., 0] ** 2
        moments[..., 5] = rho * velocity[..., 1] ** 2
        moments[..., 6] = rho * velocity[..., 2] ** 2
        moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
        moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
        moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
        q = int(np.flatnonzero(np.all(D3Q27_DIRECTIONS == (1, 0, 0), axis=1))[0])
        status = np.zeros(shape + (27,), dtype=np.int32)
        status[1:3, ..., q] = 1
        fraction = np.zeros(shape + (27,), dtype=np.float64)
        fraction[1:3, ..., q] = 0.35
        owner = np.zeros(shape + (27,), dtype=np.int32)
        owner[1:3, ..., q] = 1
        normal = np.zeros(shape + (3,), dtype=np.float64)
        normal[..., 0] = 1.0
        gas_density = np.full(shape, 0.997, dtype=np.float64)
        owner[2, ..., q] = 2
        gas_density[1] = 0.994
        return (
            moments,
            active,
            status,
            fraction,
            owner,
            normal,
            gas_density,
            q,
        )

    def test_normal_target_matches_free_surface_stress_equation(self) -> None:
        moments, active, status, fraction, owner, _, gas_density, q = self._planar_case()
        viscosity = 0.2

        result = fsl_normal_strain_targets(
            moments,
            active,
            status,
            owner,
            fraction,
            gas_density,
            lattice_viscosity=viscosity,
            periodic=(False, True, True),
        )

        for i in range(1, 3):
            expected_rho = 1.01 + 0.003 * (i - 0.35)
            owner_i = i if owner[i, 0, 0, q] == 1 else i - 1
            expected_gas = gas_density[owner_i, 0, 0]
            expected = (expected_rho - expected_gas) / (6.0 * viscosity)
            np.testing.assert_allclose(
                result.normal_strain[i, ..., q], expected, rtol=0.0, atol=2.0e-16
            )
        self.assertEqual(result.no_support_link_count, 0)

    def _warp_full_closure_matches_equations(self, device: str) -> None:
        (
            moments,
            active,
            status,
            fraction,
            owner,
            normal,
            gas_density,
            q,
        ) = self._planar_case()
        model = HomeLbmModel(
            fluid_grid_res=active.shape,
            periodic=(False, True, True),
            kinematic_viscosity=0.2,
            device=device,
        )
        state = HomeLbmState(model)
        _upload(state, moments)
        closure = HomeFreeFslStressClosure(model)
        active_device = wp.array(active.astype(np.int32), dtype=wp.int32, device=device)
        status_device = wp.array(
            np.ascontiguousarray(status.reshape(-1)), dtype=wp.int32, device=device
        )
        fraction_device = wp.array(
            np.ascontiguousarray(fraction.reshape(-1), dtype=np.float32),
            dtype=float,
            device=device,
        )
        owner_device = wp.array(
            np.ascontiguousarray(owner.reshape(-1)), dtype=wp.int32, device=device
        )
        directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device
        )

        closure.build(
            state,
            active_device,
            status_device,
            owner_device,
            fraction_device,
            wp.array(normal.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(gas_density.astype(np.float32), dtype=float, device=device),
            directions,
        )
        wp.synchronize_device(device)

        assert closure.last_diagnostics is not None
        self.assertEqual(closure.last_diagnostics.invalid_normal_strain_count, 0)
        self.assertEqual(closure.last_diagnostics.invalid_boundary_strain_count, 0)
        link_stride = 27 * int(np.prod(active.shape))
        components = np.moveaxis(
            closure.boundary_strain.numpy().reshape(6, *active.shape, 27),
            0,
            -1,
        )
        target = closure.normal_strain.numpy().reshape(active.shape + (27,))
        applicable = status == 1
        np.testing.assert_allclose(
            components[..., 0][applicable], target[applicable], rtol=3.0e-5, atol=3.0e-7
        )
        np.testing.assert_allclose(components[..., 3][applicable], 0.0, atol=3.0e-7)
        np.testing.assert_allclose(components[..., 4][applicable], 0.0, atol=3.0e-7)
        np.testing.assert_allclose(components[..., 1][applicable], 0.0, atol=3.0e-7)
        np.testing.assert_allclose(components[..., 2][applicable], 0.0, atol=3.0e-7)
        np.testing.assert_allclose(components[..., 5][applicable], 0.0, atol=3.0e-7)
        self.assertEqual(closure.boundary_strain.size, 6 * link_stride)
        self.assertEqual(q, int(np.flatnonzero(status[1, 0, 0] == 1)[0]))

    def test_full_closure_matches_equations_on_cpu(self) -> None:
        self._warp_full_closure_matches_equations("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_full_closure_matches_equations_on_cuda(self) -> None:
        self._warp_full_closure_matches_equations("cuda:0")

    def test_no_support_is_rejected_by_default(self) -> None:
        moments, active, status, fraction, owner, _, gas_density, q = self._planar_case()
        status[0, 0, 0, q] = -7
        with self.assertRaisesRegex(RuntimeError, "NO_SUPPORT"):
            fsl_normal_strain_targets(
                moments,
                active,
                status,
                owner,
                fraction,
                gas_density,
                lattice_viscosity=0.2,
                periodic=(False, True, True),
            )


if __name__ == "__main__":
    unittest.main()
