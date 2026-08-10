# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional scheduling and conservation tests for HomeFreeDomain."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeFreeMassAdvector,
    HomeFreeState,
    HomeFreeTopologyUpdater,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)


def _layered_fill(shape: tuple[int, int, int]) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    fill[:3] = 1.0
    fill[3] = 0.5
    return fill


def _spherical_fill(shape: tuple[int, int, int]) -> np.ndarray:
    center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    fill = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        distance = float(np.linalg.norm(np.asarray(index, dtype=np.float64) - center))
        fill[index] = np.clip(0.5 + (3.25 - distance) / 2.0, 0.0, 1.0)
    return fill


class TestHomeFreeDomain(unittest.TestCase):
    def _assert_ordered_step(self, device: str) -> None:
        shape = (8, 3, 2)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, True, True),
            kinematic_viscosity=0.2,
            device=device,
        )
        fill = _layered_fill(shape)
        velocity = (0.018, -0.004, 0.003)
        domain = HomeFreeLegacyDomain(model, gas_density=0.99)
        domain.initialize_uniform_lattice(fill, velocity=velocity)
        original_front = domain.state

        manual_in = HomeLbmState(model)
        manual_out = HomeLbmState(model)
        manual_solver = HomeLbmSolver(model)
        manual_solver.initialize_uniform_lattice(manual_in, velocity=velocity)
        manual_free_in = HomeFreeState(model)
        manual_free_in.initialize_from_fill_level(manual_in, fill)
        manual_free_out = HomeFreeState(model)
        advector = HomeFreeMassAdvector(model)
        topology = HomeFreeTopologyUpdater(model)
        advected = advector.advect(manual_in, manual_free_in)
        manual_solver.step(
            manual_in,
            manual_out,
            model.time_step,
            free_surface_flags=manual_free_in.flags,
            gas_density=0.99,
        )
        advector.apply_queue_momentum_correction(manual_out)
        topology.classify(manual_out, manual_free_in, advected)
        topology.commit(manual_out, manual_free_in, advected, manual_free_out)

        domain.step(model.time_step)
        wp.synchronize_device(device)

        self.assertIsNot(domain.state, original_front)
        np.testing.assert_allclose(
            domain.fluid_state.moments.numpy(),
            manual_out.moments.numpy(),
            rtol=3.0e-6,
            atol=3.0e-7,
        )
        for actual, expected in (
            (domain.free_surface_state.mass, manual_free_out.mass),
            (domain.free_surface_state.fill_level, manual_free_out.fill_level),
            (domain.free_surface_state.excess_mass, manual_free_out.excess_mass),
            (
                domain.free_surface_state.excess_momentum,
                manual_free_out.excess_momentum,
            ),
        ):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=3.0e-6, atol=3.0e-7)
        np.testing.assert_array_equal(
            domain.free_surface_state.flags.numpy(), manual_free_out.flags.numpy()
        )
        self.assertIsNotNone(domain.last_diagnostics)
        self.assertEqual(domain.last_diagnostics.invalid_cell_count, 0)
        self.assertEqual(domain.last_diagnostics.direct_liquid_gas_link_count, 0)

    def test_cpu_executes_paper_order_into_back_buffer(self) -> None:
        self._assert_ordered_step("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_executes_paper_order_into_back_buffer(self) -> None:
        self._assert_ordered_step("cuda:0")

    def test_multiple_steps_conserve_liquid_mass_including_queued_excess(self) -> None:
        shape = (9, 4, 3)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, True, True),
            kinematic_viscosity=0.18,
            device="cpu",
        )
        fill = _layered_fill(shape)
        domain = HomeFreeLegacyDomain(model)
        domain.initialize_uniform_lattice(fill, velocity=(0.025, 0.0, 0.0))
        initial_mass = float(np.sum(domain.free_surface_state.mass.numpy(), dtype=np.float64))

        for _ in range(8):
            domain.step(model.time_step)
            free = domain.free_surface_state
            represented = free.mass.numpy() + free.excess_mass.numpy() * domain.topology.active_neighbor_count.numpy()
            self.assertAlmostEqual(
                float(np.sum(represented, dtype=np.float64)), initial_mass, places=4
            )
            free.validate(
                domain.fluid_state,
                allow_interface_endpoints=True,
            )

    def test_failed_step_does_not_swap_committed_state(self) -> None:
        shape = (3, 2, 2)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 1.0
        domain = HomeFreeLegacyDomain(model)
        domain.initialize_uniform_lattice(fill, validate_topology=False)
        committed = domain.state
        moments_before = domain.fluid_state.moments.numpy().copy()

        with self.assertRaisesRegex(RuntimeError, "direct liquid-gas"):
            domain.step(model.time_step)

        self.assertIs(domain.state, committed)
        np.testing.assert_array_equal(domain.fluid_state.moments.numpy(), moments_before)

    def test_surface_tension_domain_uses_local_laplace_density(self) -> None:
        shape = (11, 11, 11)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        fill = _spherical_fill(shape)
        baseline = HomeFreeLegacyDomain(model)
        capillary = HomeFreeLegacyDomain(
            model,
            surface_tension=model.scaling.surface_tension_to_physical(0.001),
        )
        baseline.initialize_uniform_lattice(fill)
        capillary.initialize_uniform_lattice(fill)

        baseline.step(model.time_step)
        capillary.step(model.time_step)

        self.assertIsNotNone(capillary.last_surface_tension_diagnostics)
        self.assertEqual(
            capillary.last_surface_tension_diagnostics.invalid_gas_density_count,
            0,
        )
        self.assertGreater(
            capillary.last_surface_tension_diagnostics.min_interface_gas_density,
            1.0,
        )
        interface = capillary.free_surface_state.flags.numpy() == 1
        baseline_moments = baseline.fluid_state.moments.numpy().reshape(10, -1)
        capillary_moments = capillary.fluid_state.moments.numpy().reshape(10, -1)
        flat_interface = interface.reshape(-1)
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        capillary_moments[:, flat_interface]
                        - baseline_moments[:, flat_interface]
                    )
                )
            ),
            1.0e-5,
        )


if __name__ == "__main__":
    unittest.main()
