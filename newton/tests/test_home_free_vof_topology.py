# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic topology-transition tests for HOME-FREE VOF."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeState,
    HomeFreeTopologyUpdater,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof import (
    HomeFreeCellFlag,
    HomeFreeTransition,
    apply_topology_transitions,
    classify_topology_transitions,
)


class TestHomeFreeVofTopology(unittest.TestCase):
    def _layered_case(self, device: str = "cpu") -> tuple:
        shape = (5, 3, 3)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(
            fluid, rho=1.0, velocity=(0.02, -0.01, 0.03)
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 1.0
        fill[1] = 0.55
        free = HomeFreeState(model)
        free.initialize_from_fill_level(fluid, fill)
        return model, fluid, free

    def _assert_matches_reference(
        self,
        model: HomeLbmModel,
        fluid: HomeLbmState,
        free: HomeFreeState,
        advected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rho = fluid.moments.numpy()[: free.cell_count].reshape(free.res)
        expected_transitions = classify_topology_transitions(
            rho,
            advected,
            free.flags.numpy(),
            periodic=model.periodic,
        )
        expected_flags = apply_topology_transitions(
            free.flags.numpy(), expected_transitions
        )
        device_mass = wp.array(advected, dtype=float, device=model._device)
        actual_transitions, actual_flags = HomeFreeTopologyUpdater(model).classify(
            fluid, free, device_mass
        )
        np.testing.assert_array_equal(actual_transitions.numpy(), expected_transitions)
        np.testing.assert_array_equal(actual_flags.numpy(), expected_flags)
        return expected_transitions, expected_flags

    def test_liquid_growth_creates_interface_layer_in_former_gas(self) -> None:
        model, fluid, free = self._layered_case()
        advected = free.mass.numpy()
        advected[1] = 1.02

        transitions, flags = self._assert_matches_reference(model, fluid, free, advected)

        np.testing.assert_array_equal(
            transitions[1], int(HomeFreeTransition.INTERFACE_TO_LIQUID)
        )
        np.testing.assert_array_equal(
            transitions[2], int(HomeFreeTransition.GAS_TO_INTERFACE)
        )
        np.testing.assert_array_equal(flags[1], int(HomeFreeCellFlag.LIQUID))
        np.testing.assert_array_equal(flags[2], int(HomeFreeCellFlag.INTERFACE))

    def test_commit_uses_final_topology_and_preserves_queued_mass(self) -> None:
        model, fluid, free = self._layered_case()
        advected = free.mass.numpy()
        advected[1] = 1.02
        device_mass = wp.array(advected, dtype=float, device=model._device)
        updater = HomeFreeTopologyUpdater(model)
        updater.classify(fluid, free, device_mass)
        destination = HomeFreeState(model)

        updater.commit(fluid, free, device_mass, destination)

        destination.validate(fluid, allow_interface_endpoints=True)
        counts = updater.active_neighbor_count.numpy()
        represented_mass = destination.mass.numpy() + destination.excess_mass.numpy() * counts
        self.assertAlmostEqual(float(np.sum(represented_mass)), float(np.sum(advected)), places=5)
        excess_momentum = np.moveaxis(
            destination.excess_momentum.numpy().reshape(3, *destination.res),
            0,
            -1,
        )
        expected_excess_momentum = destination.excess_mass.numpy()[..., None] * np.asarray(
            (0.02, -0.01, 0.03), dtype=np.float32
        )
        np.testing.assert_allclose(
            excess_momentum,
            expected_excess_momentum,
            rtol=3.0e-6,
            atol=3.0e-7,
        )
        np.testing.assert_array_equal(
            destination.flags.numpy()[2], int(HomeFreeCellFlag.INTERFACE)
        )
        np.testing.assert_allclose(destination.fill_level.numpy()[2], 0.0)

    def test_fresh_interface_moments_are_equilibrium_of_active_neighbor_average(self) -> None:
        model, fluid, free = self._layered_case()
        stride = free.cell_count
        moments = fluid.moments.numpy().reshape(10, stride)
        rho = np.linspace(0.98, 1.02, stride, dtype=np.float32)
        moments[0] = rho
        moments[1] = rho * 0.03
        moments[2] = rho * -0.01
        moments[3] = rho * 0.02
        fluid.moments.assign(moments.reshape(-1))
        advected = free.mass.numpy()
        advected[1] = 1.2
        device_mass = wp.array(advected, dtype=float, device=model._device)
        updater = HomeFreeTopologyUpdater(model)
        updater.classify(fluid, free, device_mass)

        updater.commit(fluid, free, device_mass, HomeFreeState(model))

        result = fluid.moments.numpy().reshape(10, stride)
        for index in np.ndindex((3, 3)):
            cell = 2 * 9 + index[0] * 3 + index[1]
            donor_cells = []
            for y in range(max(0, index[0] - 1), min(3, index[0] + 2)):
                for z in range(max(0, index[1] - 1), min(3, index[1] + 2)):
                    donor_cells.append(1 * 9 + y * 3 + z)
            expected_rho = float(np.mean(rho[donor_cells]))
            self.assertAlmostEqual(float(result[0, cell]), expected_rho, places=6)
            self.assertAlmostEqual(float(result[1, cell]), expected_rho * 0.03, places=6)
            self.assertAlmostEqual(float(result[4, cell]), expected_rho * 0.03**2, places=6)

    def _assert_joint_commit_closes_mass_momentum(self, device: str) -> None:
        model, fluid, free = self._layered_case(device=device)
        shape = free.res
        advected = free.mass.numpy().astype(np.float64)
        advected[1] = 1.02
        target_velocity = np.zeros(shape + (3,), dtype=np.float64)
        coordinates = np.indices(shape, dtype=np.float64)
        target_velocity[..., 0] = 0.025 + 0.001 * coordinates[1]
        target_velocity[..., 1] = -0.012 + 0.001 * coordinates[2]
        target_velocity[..., 2] = 0.018 - 0.0005 * coordinates[1]
        source_active = np.isin(
            free.flags.numpy(),
            (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
        )
        target = np.zeros(shape + (3,), dtype=np.float64)
        target[source_active] = (
            advected[source_active, None] * target_velocity[source_active]
        )
        device_mass = wp.array(advected, dtype=float, device=model._device)
        device_momentum = wp.array(
            np.moveaxis(target, -1, 0).reshape(-1),
            dtype=float,
            device=model._device,
        )
        updater = HomeFreeTopologyUpdater(model)
        updater.classify(fluid, free, device_mass)
        destination = HomeFreeState(model)

        updater.commit(
            fluid,
            free,
            device_mass,
            destination,
            advected_momentum=device_momentum,
        )

        flags = destination.flags.numpy()
        active = np.isin(
            flags,
            (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
        )
        moments = np.moveaxis(
            fluid.moments.numpy().reshape(10, *shape), 0, -1
        ).astype(np.float64)
        velocity = np.zeros(shape + (3,), dtype=np.float64)
        velocity[active] = moments[..., 1:4][active] / moments[..., 0][active, None]
        excess_j = np.moveaxis(
            destination.excess_momentum.numpy().reshape(3, *shape), 0, -1
        ).astype(np.float64)
        represented_mass = (
            destination.mass.numpy()
            + updater.active_neighbor_count.numpy() * destination.excess_mass.numpy()
        )
        represented_momentum = destination.mass.numpy()[..., None] * velocity
        represented_momentum += (
            updater.active_neighbor_count.numpy()[..., None] * excess_j
        )
        np.testing.assert_allclose(
            represented_mass, advected, rtol=4.0e-6, atol=4.0e-7
        )
        np.testing.assert_allclose(
            represented_momentum, target, rtol=5.0e-6, atol=5.0e-7
        )
        central = moments[..., 4:10].copy()
        central[..., 0] -= moments[..., 1] ** 2 / moments[..., 0]
        central[..., 1] -= moments[..., 2] ** 2 / moments[..., 0]
        central[..., 2] -= moments[..., 3] ** 2 / moments[..., 0]
        central[..., 3] -= moments[..., 1] * moments[..., 2] / moments[..., 0]
        central[..., 4] -= moments[..., 1] * moments[..., 3] / moments[..., 0]
        central[..., 5] -= moments[..., 2] * moments[..., 3] / moments[..., 0]
        np.testing.assert_allclose(central[active], 0.0, rtol=0.0, atol=3.0e-7)

    def test_joint_commit_closes_mass_momentum_on_cpu(self) -> None:
        self._assert_joint_commit_closes_mass_momentum("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_joint_commit_closes_mass_momentum_on_cuda(self) -> None:
        self._assert_joint_commit_closes_mass_momentum("cuda:0")

    def test_gas_growth_converts_adjacent_liquid_to_interface(self) -> None:
        model, fluid, free = self._layered_case()
        advected = free.mass.numpy()
        advected[1] = -0.02

        transitions, flags = self._assert_matches_reference(model, fluid, free, advected)

        np.testing.assert_array_equal(
            transitions[0], int(HomeFreeTransition.LIQUID_TO_INTERFACE)
        )
        np.testing.assert_array_equal(
            transitions[1], int(HomeFreeTransition.INTERFACE_TO_GAS)
        )
        np.testing.assert_array_equal(flags[0], int(HomeFreeCellFlag.INTERFACE))
        np.testing.assert_array_equal(flags[1], int(HomeFreeCellFlag.GAS))

    def test_competing_growth_cancels_adjacent_interface_to_gas(self) -> None:
        model, fluid, free = self._layered_case()
        fill = free.fill_level.numpy()
        fill[2] = 0.4
        free.fill_level.assign(fill)
        flags = free.flags.numpy()
        flags[2] = int(HomeFreeCellFlag.INTERFACE)
        free.flags.assign(flags)
        mass = free.mass.numpy()
        mass[2] = 0.4
        free.mass.assign(mass)
        advected = mass.copy()
        advected[1] = 1.02
        advected[2] = -0.02

        transitions, resolved_flags = self._assert_matches_reference(
            model, fluid, free, advected
        )

        np.testing.assert_array_equal(
            transitions[1], int(HomeFreeTransition.INTERFACE_TO_LIQUID)
        )
        np.testing.assert_array_equal(transitions[2], int(HomeFreeTransition.KEEP))
        np.testing.assert_array_equal(resolved_flags[2], int(HomeFreeCellFlag.INTERFACE))

    def test_epsilon_prevents_chatter_for_tiny_mass_overshoot(self) -> None:
        model, fluid, free = self._layered_case()
        advected = free.mass.numpy()
        advected[1] = 1.00005

        transitions, _ = self._assert_matches_reference(model, fluid, free, advected)

        np.testing.assert_array_equal(transitions[1], int(HomeFreeTransition.KEEP))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_topology_matches_reference(self) -> None:
        model, fluid, free = self._layered_case(device="cuda:0")
        advected = free.mass.numpy()
        advected[1] = 1.02
        self._assert_matches_reference(model, fluid, free, advected)


if __name__ == "__main__":
    unittest.main()
