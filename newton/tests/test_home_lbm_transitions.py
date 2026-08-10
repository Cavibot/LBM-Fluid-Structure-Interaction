# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservation tests for moving-solid HOME cell transitions."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    ConservativeCellRemapper,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)


def _sphere_phi(shape: tuple[int, int, int], center: np.ndarray, radius: float) -> np.ndarray:
    points = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0) + 0.5
    return (np.linalg.norm(points - center, axis=-1) - radius).astype(np.float32)


def _upload_phi(state: HomeLbmState, phi: np.ndarray) -> None:
    wp.copy(state.solid_phi, wp.array(phi, dtype=float, device=state.device))


def _fluid_totals(state: HomeLbmState, phi: np.ndarray) -> np.ndarray:
    moments = state.moments.numpy().reshape(10, -1)
    return moments[:, phi.reshape(-1) >= 0.0].sum(axis=1, dtype=np.float64)


class TestHomeLbmTransitions(unittest.TestCase):
    def test_unchanged_geometry_leaves_moments_bitwise_unchanged(self) -> None:
        shape = (12, 10, 8)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        state = HomeLbmState(model)
        solver = HomeLbmSolver(model)
        solver.initialize_uniform_lattice(state, rho=1.03, velocity=(0.02, 0.01, -0.01))
        phi = _sphere_phi(shape, np.array([6.0, 5.0, 4.0]), 2.4)
        _upload_phi(state, phi)
        remapper = ConservativeCellRemapper(state)
        moments_before = state.moments.numpy().copy()

        diagnostics = remapper.remap(state)

        self.assertEqual(diagnostics.fresh_cell_count, 0)
        self.assertEqual(diagnostics.dead_cell_count, 0)
        self.assertEqual(diagnostics.unresolved_cell_count, 0)
        np.testing.assert_array_equal(state.moments.numpy(), moments_before)

    def test_moving_solid_conserves_all_ten_home_moments(self) -> None:
        shape = (16, 14, 12)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        state = HomeLbmState(model)
        solver = HomeLbmSolver(model)
        solver.initialize_uniform_lattice(state, rho=1.0, velocity=(0.03, -0.01, 0.02))
        old_phi = _sphere_phi(shape, np.array([7.0, 7.0, 6.0]), 3.1)
        _upload_phi(state, old_phi)
        remapper = ConservativeCellRemapper(state)
        before = _fluid_totals(state, old_phi)

        new_phi = _sphere_phi(shape, np.array([7.8, 7.0, 6.0]), 3.1)
        _upload_phi(state, new_phi)
        diagnostics = remapper.remap(state)
        after = _fluid_totals(state, new_phi)

        self.assertGreater(diagnostics.fresh_cell_count, 0)
        self.assertGreater(diagnostics.dead_cell_count, 0)
        self.assertEqual(diagnostics.unresolved_cell_count, 0)
        np.testing.assert_allclose(after, before, atol=3.0e-5)
        fluid_density = state.moments.numpy().reshape(10, -1)[
            0, new_phi.reshape(-1) >= 0.0
        ]
        self.assertTrue(np.all(fluid_density > 0.0))
        self.assertLess(float(np.max(np.abs(fluid_density - 1.0))), 0.01)


if __name__ == "__main__":
    unittest.main()
