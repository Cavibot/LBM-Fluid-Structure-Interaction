# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU moving-solid remap against the NumPy phase oracle and invariants."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeRigidTransitionRemapper,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    classify_moving_solid_phases,
    represented_liquid_momentum,
)


def _cube_mask(
    shape: tuple[int, int, int], lower: tuple[int, int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    i, j, k = lower
    mask[i : i + 2, j : j + 2, k : k + 3] = True
    return mask


def _create_state(
    shape: tuple[int, int, int],
    old_solid: np.ndarray,
    device: str,
) -> tuple[HomeLbmModel, HomeLbmState, HomeFreeState]:
    model = HomeLbmModel(
        fluid_grid_res=shape,
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        kinematic_viscosity=0.2,
        periodic=(False, False, False),
        device=device,
    )
    fluid = HomeLbmState(model)
    HomeLbmSolver(model).initialize_uniform_lattice(fluid)
    fluid.solid_phi.assign(np.where(old_solid, -1.0, 1.0).astype(np.float32))
    fill = np.zeros(shape, dtype=np.float32)
    fill[:, :, :4] = 1.0
    fill[:, :, 4] = 0.5
    free = HomeFreeState(model)
    free.initialize_from_fill_level(fluid, fill)
    return model, fluid, free


def _active_moment_sum(fluid: HomeLbmState, free: HomeFreeState) -> np.ndarray:
    moments = fluid.moments.numpy().reshape(10, -1)
    active = np.isin(free.flags.numpy().reshape(-1), (1, 2))
    return np.sum(moments[:, active], axis=1, dtype=np.float64)


class TestHomeFreeRigidTransition(unittest.TestCase):
    def _run_horizontal_oracle(self, device: str) -> None:
        shape = (10, 6, 8)
        old_solid = _cube_mask(shape, (3, 2, 2))
        new_solid = _cube_mask(shape, (4, 2, 2))
        model, fluid, free = _create_state(shape, old_solid, device)
        old_flags = free.flags.numpy().copy()
        old_fill = free.fill_level.numpy().copy()
        old_moments = _active_moment_sum(fluid, free)
        reference = classify_moving_solid_phases(
            old_solid,
            new_solid,
            old_flags,
            old_fill,
        )
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        fluid.solid_phi.assign(np.where(new_solid, -1.0, 1.0).astype(np.float32))

        diagnostics = remapper.remap(fluid, free)

        np.testing.assert_array_equal(free.flags.numpy(), reference.flags)
        np.testing.assert_array_equal(remapper.proposed_flags.numpy(), reference.flags)
        np.testing.assert_allclose(
            remapper.proposed_fill.numpy()[reference.fresh_mask],
            reference.fill_level[reference.fresh_mask],
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            _active_moment_sum(fluid, free), old_moments, rtol=2.0e-6, atol=2.0e-6
        )
        self.assertEqual(diagnostics.fresh_cell_count, 6)
        self.assertEqual(diagnostics.dead_cell_count, 6)
        self.assertEqual(diagnostics.fresh_liquid_cells, 2)
        self.assertEqual(diagnostics.fresh_interface_cells, 4)
        self.assertEqual(diagnostics.fresh_gas_cells, 0)
        self.assertLess(diagnostics.relative_mass_error, 2.0e-6)
        self.assertLess(diagnostics.queued_mass_ratio, 0.02)

    def test_cpu_horizontal_remap_matches_phase_oracle(self) -> None:
        self._run_horizontal_oracle("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_horizontal_remap_matches_phase_oracle(self) -> None:
        self._run_horizontal_oracle("cuda:0")

    def test_vertical_exit_conserves_moments_and_queues_interface_mass(self) -> None:
        shape = (8, 6, 9)
        old_solid = _cube_mask(shape, (3, 2, 2))
        new_solid = _cube_mask(shape, (3, 2, 3))
        model, fluid, free = _create_state(shape, old_solid, "cpu")
        old_moments = _active_moment_sum(fluid, free)
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        fluid.solid_phi.assign(np.where(new_solid, -1.0, 1.0).astype(np.float32))

        diagnostics = remapper.remap(fluid, free)

        np.testing.assert_allclose(
            _active_moment_sum(fluid, free), old_moments, rtol=2.0e-6, atol=2.0e-6
        )
        self.assertEqual(diagnostics.fresh_cell_count, 4)
        self.assertEqual(diagnostics.dead_cell_count, 4)
        self.assertEqual(diagnostics.fresh_liquid_cells, 4)
        self.assertLess(diagnostics.queued_mass, 0.0)
        self.assertLess(diagnostics.queued_mass_ratio, 0.03)
        self.assertLess(diagnostics.relative_mass_error, 2.0e-6)

    def test_vertical_exit_queue_remains_local_to_moving_solid(self) -> None:
        shape = (16, 12, 9)
        old_solid = _cube_mask(shape, (7, 5, 2))
        new_solid = _cube_mask(shape, (7, 5, 3))
        model, fluid, free = _create_state(shape, old_solid, "cpu")
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        fluid.solid_phi.assign(np.where(new_solid, -1.0, 1.0).astype(np.float32))

        diagnostics = remapper.remap(fluid, free)

        sources = np.argwhere(free.excess_mass.numpy() != 0.0)
        changed = np.argwhere(old_solid != new_solid)
        self.assertGreater(len(sources), 0)
        self.assertEqual(diagnostics.queue_source_count, len(sources))
        self.assertEqual(diagnostics.queue_locality_radius, 3)
        for source in sources:
            three_dimensional_distance = np.max(
                np.abs(changed - source[None, :]), axis=1
            )
            horizontal_distance = np.max(
                np.abs(changed[:, :2] - source[None, :2]), axis=1
            )
            self.assertTrue(
                int(np.min(three_dimensional_distance)) <= 3
                or int(np.min(horizontal_distance)) <= 3
            )
        interface_count = int(np.count_nonzero(free.flags.numpy() == 1))
        self.assertLess(len(sources), interface_count)
        np.testing.assert_array_equal(free.excess_mass.numpy()[0], 0.0)
        self.assertLess(diagnostics.relative_mass_error, 2.0e-6)

    def test_nonuniform_flow_conserves_mass_weighted_liquid_momentum(self) -> None:
        shape = (10, 6, 8)
        old_solid = _cube_mask(shape, (3, 2, 2))
        new_solid = _cube_mask(shape, (4, 2, 2))
        model, fluid, free = _create_state(shape, old_solid, "cpu")
        stride = free.cell_count
        moments = fluid.moments.numpy().reshape(10, stride)
        coordinates = np.indices(shape, dtype=np.float32)
        ux = 0.008 * coordinates[0] - 0.003 * coordinates[2]
        uy = 0.002 * coordinates[1]
        uz = -0.001 * coordinates[0]
        velocity = np.stack((ux, uy, uz), axis=-1).reshape(-1, 3)
        rho = moments[0].copy()
        moments[1:4] = (rho[:, None] * velocity).T
        moments[4] = rho * velocity[:, 0] * velocity[:, 0]
        moments[5] = rho * velocity[:, 1] * velocity[:, 1]
        moments[6] = rho * velocity[:, 2] * velocity[:, 2]
        moments[7] = rho * velocity[:, 0] * velocity[:, 1]
        moments[8] = rho * velocity[:, 0] * velocity[:, 2]
        moments[9] = rho * velocity[:, 1] * velocity[:, 2]
        fluid.moments.assign(moments.reshape(-1))
        old_flags = free.flags.numpy()
        active = np.isin(old_flags, (1, 2))
        old_expected = np.sum(
            free.mass.numpy()[..., None] * velocity.reshape(shape + (3,)) * active[..., None],
            axis=(0, 1, 2),
            dtype=np.float64,
        )
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        fluid.solid_phi.assign(np.where(new_solid, -1.0, 1.0).astype(np.float32))

        diagnostics = remapper.remap(fluid, free)

        counts = remapper.active_neighbor_count.numpy()
        final_moments = fluid.moments.numpy().reshape(10, shape[0], shape[1], shape[2])
        final_actual = represented_liquid_momentum(
            final_moments[0],
            np.moveaxis(final_moments[1:4], 0, -1),
            free.mass.numpy(),
            free.excess_mass.numpy(),
            np.moveaxis(
                free.excess_momentum.numpy().reshape(3, *shape), 0, -1
            ),
            counts,
        )
        np.testing.assert_allclose(
            np.asarray(diagnostics.old_represented_momentum),
            old_expected,
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(final_actual, old_expected, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(
            np.asarray(diagnostics.final_represented_momentum),
            old_expected,
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        self.assertLess(diagnostics.relative_momentum_error, 2.0e-6)

    def test_unchanged_solid_mask_is_bitwise_noop(self) -> None:
        shape = (8, 6, 9)
        solid = _cube_mask(shape, (3, 2, 2))
        model, fluid, free = _create_state(shape, solid, "cpu")
        excess = np.zeros(shape, dtype=np.float32)
        excess[0, 0, 4] = 0.001
        free.excess_mass.assign(excess)
        excess_momentum = np.zeros((3,) + shape, dtype=np.float32)
        excess_momentum[:, 0, 0, 4] = (2.0e-5, -1.0e-5, 3.0e-5)
        free.excess_momentum.assign(excess_momentum.reshape(3, -1).reshape(-1))
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        snapshots = (
            fluid.moments.numpy().copy(),
            free.flags.numpy().copy(),
            free.fill_level.numpy().copy(),
            free.mass.numpy().copy(),
            free.excess_mass.numpy().copy(),
            free.excess_momentum.numpy().copy(),
        )

        diagnostics = remapper.remap(fluid, free)

        current = (
            fluid.moments.numpy(),
            free.flags.numpy(),
            free.fill_level.numpy(),
            free.mass.numpy(),
            free.excess_mass.numpy(),
            free.excess_momentum.numpy(),
        )
        for before, after in zip(snapshots, current, strict=True):
            np.testing.assert_array_equal(after, before)
        self.assertEqual(diagnostics.fresh_cell_count, 0)
        self.assertEqual(diagnostics.dead_cell_count, 0)
        self.assertEqual(diagnostics.relative_mass_error, 0.0)

    def test_unresolved_remap_does_not_commit_state(self) -> None:
        shape = (5, 5, 5)
        old_solid = np.ones(shape, dtype=bool)
        model, fluid, free = _create_state(shape, old_solid, "cpu")
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        new_solid = old_solid.copy()
        new_solid[2, 2, 2] = False
        fluid.solid_phi.assign(np.where(new_solid, -1.0, 1.0).astype(np.float32))
        moments_before = fluid.moments.numpy().copy()
        flags_before = free.flags.numpy().copy()
        mass_before = free.mass.numpy().copy()
        excess_momentum_before = free.excess_momentum.numpy().copy()

        with self.assertRaisesRegex(RuntimeError, "no persistent phase donor"):
            remapper.remap(fluid, free)

        np.testing.assert_array_equal(fluid.moments.numpy(), moments_before)
        np.testing.assert_array_equal(free.flags.numpy(), flags_before)
        np.testing.assert_array_equal(free.mass.numpy(), mass_before)
        np.testing.assert_array_equal(
            free.excess_momentum.numpy(), excess_momentum_before
        )

    def test_device_validator_rejects_momentum_without_queued_mass(self) -> None:
        shape = (8, 6, 9)
        solid = _cube_mask(shape, (3, 2, 2))
        model, fluid, free = _create_state(shape, solid, "cpu")
        remapper = HomeFreeRigidTransitionRemapper(model, fluid, free)
        queued_momentum = np.zeros((3,) + shape, dtype=np.float32)
        queued_momentum[0, 0, 0, 4] = 1.0e-3
        free.excess_momentum.assign(queued_momentum.reshape(-1))

        with self.assertRaisesRegex(ValueError, "invalid HOME-Free cells"):
            remapper._validate_committed_state(fluid, free)


if __name__ == "__main__":
    unittest.main()
