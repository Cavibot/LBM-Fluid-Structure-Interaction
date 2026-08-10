# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""PLIC curvature to Laplace-pressure density tests for paper Eq. (12)."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeMassAdvector,
    HomeFreeState,
    HomeFreeSurfaceTension,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    discrete_capillary_pressure_link_momentum,
    plic_capillary_pressure_momentum,
)


def _spherical_fill(
    shape: tuple[int, int, int], *, liquid_inside: bool
) -> np.ndarray:
    center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    radius = 3.25
    fill = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        distance = float(np.linalg.norm(np.asarray(index, dtype=np.float64) - center))
        inside = np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0)
        fill[index] = inside if liquid_inside else 1.0 - inside
    return fill


class TestHomeFreeSurfaceTension(unittest.TestCase):
    def _state(self, model: HomeLbmModel, fill: np.ndarray) -> HomeFreeState:
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)
        return state

    def test_eq12_density_matches_curvature_and_physical_scaling(self) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=0.02,
            time_step=0.001,
            reference_density=1000.0,
            kinematic_viscosity=4.0e-4,
            periodic=(False, False, False),
            device="cpu",
        )
        lattice_gamma = 0.02
        physical_gamma = model.scaling.surface_tension_to_physical(lattice_gamma)
        surface = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=1.0,
            surface_tension=physical_gamma,
        )
        state = self._state(model, _spherical_fill(shape, liquid_inside=True))

        diagnostics = surface.update(state)

        interface = state.flags.numpy() == 1
        curvature = surface.geometry.curvature.numpy()[interface]
        expected = 1.0 - 6.0 * lattice_gamma * curvature
        np.testing.assert_allclose(
            surface.gas_density.numpy()[interface], expected, rtol=2.0e-6, atol=2.0e-7
        )
        self.assertAlmostEqual(surface.lattice_surface_tension, lattice_gamma, places=7)
        self.assertEqual(diagnostics.invalid_gas_density_count, 0)
        self.assertIsNone(surface.capillary_momentum_correction)
        self.assertGreater(float(np.mean(expected)), 1.0)

    def test_excessive_laplace_pressure_is_rejected_not_clamped(self) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        state = self._state(model, _spherical_fill(shape, liquid_inside=False))
        surface = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=1.0,
            surface_tension=1.0,
        )

        with self.assertRaisesRegex(FloatingPointError, "nonpositive"):
            surface.update(state)

    def test_plic_traction_removes_small_fill_gas_link_area_amplification(self) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            max_lattice_speed=0.2,
            device="cpu",
        )
        state = self._state(model, _spherical_fill(shape, liquid_inside=True))
        surface = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=1.0,
            surface_tension=model.scaling.surface_tension_to_physical(0.001),
        )
        surface.enable_plic_capillary_momentum()
        surface.update(state)
        flags = state.flags.numpy()
        interface = flags == int(HomeFreeCellFlag.INTERFACE)
        staircase = discrete_capillary_pressure_link_momentum(
            surface.gas_density.numpy(),
            surface.ambient_gas_density,
            flags,
            periodic=model.periodic,
        )
        traction = plic_capillary_pressure_momentum(
            surface.geometry.normal.numpy(),
            surface.geometry.plane_offset.numpy(),
            surface.gas_density.numpy(),
            surface.ambient_gas_density,
            flags,
        )
        expected_correction = traction - staircase
        actual_correction = np.moveaxis(
            surface.capillary_momentum_correction.numpy().reshape(3, *shape),
            0,
            -1,
        )
        np.testing.assert_allclose(
            actual_correction, expected_correction, rtol=4.0e-5, atol=4.0e-8
        )
        mass = state.mass.numpy()[interface]
        staircase_acceleration = np.linalg.norm(staircase[interface], axis=-1) / mass
        traction_acceleration = np.linalg.norm(traction[interface], axis=-1) / mass

        self.assertGreater(
            float(np.max(staircase_acceleration)), model.max_lattice_speed
        )
        self.assertLess(float(np.max(traction_acceleration)), 0.04)
        self.assertGreater(
            float(np.max(staircase_acceleration))
            / float(np.max(traction_acceleration)),
            8.0,
        )

    def _assert_corrected_home_momentum_target(self, device: str) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.3,
            periodic=(False, False, False),
            max_lattice_speed=0.2,
            device=device,
        )
        source = HomeLbmState(model)
        destination = HomeLbmState(model)
        solver = HomeLbmSolver(model)
        solver.initialize_uniform_lattice(source)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(
            source, _spherical_fill(shape, liquid_inside=True)
        )
        surface = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=1.0,
            surface_tension=model.scaling.surface_tension_to_physical(0.001),
        )
        surface.enable_plic_capillary_momentum()
        surface.update(state)
        advector = HomeFreeMassAdvector(model)
        advector.enable_conservative_momentum_transport()
        advector.advect(source, state, pressure_reference_density=1.0)
        solver.step(
            source,
            destination,
            model.time_step,
            free_surface_flags=state.flags,
            gas_density=1.0,
            gas_density_field=surface.gas_density,
        )
        raw = np.moveaxis(
            advector.finalize_momentum_target(destination)
            .numpy()
            .reshape(3, *shape),
            0,
            -1,
        ).copy()
        corrected = np.moveaxis(
            advector.finalize_momentum_target(
                destination,
                capillary_momentum_correction=surface.capillary_momentum_correction,
            )
            .numpy()
            .reshape(3, *shape),
            0,
            -1,
        )
        flags = state.flags.numpy()
        interface = flags == int(HomeFreeCellFlag.INTERFACE)
        mass = advector.advected_mass.numpy()[interface]
        raw_speed = np.linalg.norm(raw[interface], axis=-1) / mass
        corrected_speed = np.linalg.norm(corrected[interface], axis=-1) / mass
        correction = np.moveaxis(
            surface.capillary_momentum_correction.numpy().reshape(3, *shape),
            0,
            -1,
        )

        self.assertGreater(float(np.max(raw_speed)), model.max_lattice_speed)
        self.assertLess(float(np.max(corrected_speed)), 0.04)
        self.assertGreater(
            float(np.max(raw_speed)) / float(np.max(corrected_speed)), 8.0
        )
        self.assertLess(
            float(np.linalg.norm(np.sum(correction[interface], axis=0))), 2.0e-8
        )

    def _assert_full_cell_capillary_application(self, device: str) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.3,
            periodic=(False, False, False),
            device=device,
        )
        source = HomeLbmState(model)
        destination = HomeLbmState(model)
        solver = HomeLbmSolver(model)
        solver.initialize_uniform_lattice(source)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(
            source, _spherical_fill(shape, liquid_inside=True)
        )
        surface = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=1.0,
            surface_tension=model.scaling.surface_tension_to_physical(0.001),
        )
        surface.enable_plic_capillary_momentum()
        surface.update(state)
        solver.step(
            source,
            destination,
            model.time_step,
            free_surface_flags=state.flags,
            gas_density=1.0,
            gas_density_field=surface.gas_density,
        )
        before = (
            destination.moments.numpy()
            .reshape(10, -1)
            .T.reshape(shape + (10,))
        ).copy()
        correction = np.moveaxis(
            surface.capillary_momentum_correction.numpy().reshape(3, *shape),
            0,
            -1,
        )
        surface.apply_momentum_correction(destination, state)
        after = (
            destination.moments.numpy()
            .reshape(10, -1)
            .T.reshape(shape + (10,))
        )
        interface = state.flags.numpy() == int(HomeFreeCellFlag.INTERFACE)

        np.testing.assert_array_equal(after[..., 0], before[..., 0])
        np.testing.assert_array_equal(after[~interface], before[~interface])
        np.testing.assert_allclose(
            after[..., 1:4][interface],
            before[..., 1:4][interface] + correction[interface],
            rtol=3.0e-6,
            atol=3.0e-8,
        )

        def central_second(moments: np.ndarray) -> np.ndarray:
            rho = moments[..., 0]
            momentum = moments[..., 1:4]
            return np.stack(
                (
                    moments[..., 4] - momentum[..., 0] ** 2 / rho,
                    moments[..., 5] - momentum[..., 1] ** 2 / rho,
                    moments[..., 6] - momentum[..., 2] ** 2 / rho,
                    moments[..., 7]
                    - momentum[..., 0] * momentum[..., 1] / rho,
                    moments[..., 8]
                    - momentum[..., 0] * momentum[..., 2] / rho,
                    moments[..., 9]
                    - momentum[..., 1] * momentum[..., 2] / rho,
                ),
                axis=-1,
            )

        np.testing.assert_allclose(
            central_second(after)[interface],
            central_second(before)[interface],
            rtol=2.0e-5,
            atol=2.0e-8,
        )

    def test_cpu_corrected_home_momentum_target_uses_plic_area(self) -> None:
        self._assert_corrected_home_momentum_target("cpu")

    def test_cpu_full_cell_capillary_application_preserves_central_stress(self) -> None:
        self._assert_full_cell_capillary_application("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_corrected_home_momentum_target_uses_plic_area(self) -> None:
        self._assert_corrected_home_momentum_target("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_full_cell_capillary_application_preserves_central_stress(self) -> None:
        self._assert_full_cell_capillary_application("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_eq12_density_matches_curvature(self) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cuda:0",
        )
        state = self._state(model, _spherical_fill(shape, liquid_inside=True))
        surface = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=1.0,
            surface_tension=model.scaling.surface_tension_to_physical(0.02),
        )
        surface.enable_plic_capillary_momentum()

        surface.update(state)

        interface = state.flags.numpy() == 1
        expected = 1.0 - 0.12 * surface.geometry.curvature.numpy()[interface]
        np.testing.assert_allclose(
            surface.gas_density.numpy()[interface], expected, rtol=2.0e-6, atol=2.0e-7
        )
        staircase = discrete_capillary_pressure_link_momentum(
            surface.gas_density.numpy(),
            surface.ambient_gas_density,
            state.flags.numpy(),
            periodic=model.periodic,
        )
        traction = plic_capillary_pressure_momentum(
            surface.geometry.normal.numpy(),
            surface.geometry.plane_offset.numpy(),
            surface.gas_density.numpy(),
            surface.ambient_gas_density,
            state.flags.numpy(),
        )
        actual_correction = np.moveaxis(
            surface.capillary_momentum_correction.numpy().reshape(3, *shape),
            0,
            -1,
        )
        np.testing.assert_allclose(
            actual_correction, traction - staircase, rtol=4.0e-5, atol=4.0e-8
        )


if __name__ == "__main__":
    unittest.main()
