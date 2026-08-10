# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeGeometricQueueMaterializer,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)


class TestHomeFreeGeometricQueueMaterializer(unittest.TestCase):
    shape = (5, 5, 3)
    center = (2, 2, 1)

    def _run_signed_queue(
        self, device: str, queued_mass: float
    ) -> tuple[np.ndarray, np.ndarray, object]:
        model = HomeLbmModel(
            fluid_grid_res=self.shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.2,
            device=device,
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(
            fluid, velocity=(0.0, 0.0, 0.0)
        )
        source = HomeFreeState(model)
        source.initialize_from_fill_level(
            fluid,
            np.full(self.shape, 0.5, dtype=np.float32),
        )
        excess = np.zeros(self.shape, dtype=np.float32)
        excess[self.center] = queued_mass
        source.excess_mass.assign(excess)
        momentum_share = np.asarray((0.002, -0.001, 0.0005), dtype=np.float32)
        excess_momentum = np.zeros((3,) + self.shape, dtype=np.float32)
        excess_momentum[(slice(None),) + self.center] = momentum_share
        source.excess_momentum.assign(excess_momentum.reshape(3, -1).reshape(-1))

        destination = HomeFreeState(model)
        materializer = HomeFreeGeometricQueueMaterializer(model)
        diagnostics = materializer.materialize(fluid, source, destination)

        self.assertEqual(diagnostics.queue_source_count, 1)
        self.assertEqual(diagnostics.materialized_cell_count, 27)
        self.assertLessEqual(diagnostics.relative_mass_error, 4.0e-7)
        self.assertLessEqual(diagnostics.relative_momentum_error, 4.0e-7)
        self.assertIsNotNone(diagnostics.topology)
        np.testing.assert_array_equal(destination.excess_mass.numpy(), 0.0)
        np.testing.assert_array_equal(destination.excess_momentum.numpy(), 0.0)

        fill = destination.fill_level.numpy()
        expected_fill = 0.5 + 26.0 * queued_mass / 27.0
        neighbor_mask = np.zeros(self.shape, dtype=bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    neighbor_mask[
                        self.center[0] + di,
                        self.center[1] + dj,
                        self.center[2] + dk,
                    ] = True
        np.testing.assert_allclose(
            fill[neighbor_mask], expected_fill, rtol=0.0, atol=2.0e-7
        )
        self.assertAlmostEqual(float(fill[self.center]), expected_fill, places=7)
        untouched_mask = ~neighbor_mask
        untouched_mask[self.center] = False
        np.testing.assert_allclose(
            fill[untouched_mask], 0.5,
            rtol=0.0,
            atol=2.0e-7,
        )
        expected_mass = 0.5 * np.prod(self.shape) + 26.0 * queued_mass
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            expected_mass,
            places=5,
        )
        expected_momentum = 26.0 * momentum_share.astype(np.float64)
        np.testing.assert_allclose(
            diagnostics.committed_momentum,
            expected_momentum,
            rtol=0.0,
            atol=2.0e-7,
        )
        return fill.copy(), fluid.moments.numpy().copy(), diagnostics

    def test_cpu_positive_queue(self) -> None:
        self._run_signed_queue("cpu", 0.01)

    def test_cpu_negative_queue(self) -> None:
        self._run_signed_queue("cpu", -0.01)

    def test_overlapping_positive_sources_share_global_remaining_capacity(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=self.shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.2,
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        source = HomeFreeState(model)
        source.initialize_from_fill_level(
            fluid, np.full(self.shape, 0.99, dtype=np.float32)
        )
        excess = np.zeros(self.shape, dtype=np.float32)
        excess[2, 2, 1] = 0.01
        excess[3, 2, 1] = 0.01
        source.excess_mass.assign(excess)
        destination = HomeFreeState(model)

        diagnostics = HomeFreeGeometricQueueMaterializer(model).materialize(
            fluid, source, destination
        )

        self.assertEqual(diagnostics.queue_source_count, 2)
        self.assertLessEqual(diagnostics.relative_mass_error, 4.0e-7)
        self.assertLessEqual(diagnostics.relative_momentum_error, 4.0e-7)
        fill = destination.fill_level.numpy()
        self.assertGreaterEqual(float(np.min(fill)), 0.0)
        self.assertLessEqual(float(np.max(fill)), 1.0)
        expected_mass = 0.99 * np.prod(self.shape) + 52.0 * 0.01
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            expected_mass,
            places=5,
        )

    def test_positive_queue_without_local_capacity_fails(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=self.shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.2,
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        source = HomeFreeState(model)
        source.initialize_from_fill_level(
            fluid, np.ones(self.shape, dtype=np.float32)
        )
        excess = np.zeros(self.shape, dtype=np.float32)
        excess[self.center] = 0.01
        source.excess_mass.assign(excess)
        with self.assertRaisesRegex(RuntimeError, "invalid cells"):
            HomeFreeGeometricQueueMaterializer(model).materialize(
                fluid, source, HomeFreeState(model)
            )

    def test_negative_queue_is_absorbed_by_interface_band_only(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=self.shape,
            periodic=(True, True, False),
            kinematic_viscosity=0.2,
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        fill = np.zeros(self.shape, dtype=np.float32)
        fill[:, :, 0] = 1.0
        fill[:, :, 1] = 0.5
        source = HomeFreeState(model)
        source.initialize_from_fill_level(fluid, fill)
        excess = np.zeros(self.shape, dtype=np.float32)
        excess[self.center] = -0.01
        source.excess_mass.assign(excess)

        destination = HomeFreeState(model)
        diagnostics = HomeFreeGeometricQueueMaterializer(model).materialize(
            fluid, source, destination
        )

        result = destination.fill_level.numpy()
        np.testing.assert_array_equal(result[:, :, 0], 1.0)
        np.testing.assert_array_equal(result[:, :, 2], 0.0)
        self.assertLess(float(np.min(result[:, :, 1])), 0.5)
        self.assertEqual(diagnostics.topology.liquid_to_interface_count, 0)
        expected_mass = float(np.sum(fill, dtype=np.float64)) - 0.17
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            expected_mass,
            places=5,
        )

    def test_zero_mass_nonzero_momentum_fails(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=self.shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.2,
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        source = HomeFreeState(model)
        source.initialize_from_fill_level(
            fluid, np.full(self.shape, 0.5, dtype=np.float32)
        )
        excess_momentum = np.zeros((3,) + self.shape, dtype=np.float32)
        excess_momentum[(0,) + self.center] = 0.01
        source.excess_momentum.assign(excess_momentum.reshape(3, -1).reshape(-1))
        with self.assertRaisesRegex(RuntimeError, "invalid cells"):
            HomeFreeGeometricQueueMaterializer(model).materialize(
                fluid, source, HomeFreeState(model)
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_matches_cpu(self) -> None:
        for queued_mass in (0.01, -0.01):
            expected_fill, expected_moments, _ = self._run_signed_queue(
                "cpu", queued_mass
            )
            actual_fill, actual_moments, _ = self._run_signed_queue(
                "cuda:0", queued_mass
            )
            np.testing.assert_allclose(
                actual_fill, expected_fill, rtol=0.0, atol=3.0e-7
            )
            np.testing.assert_allclose(
                actual_moments, expected_moments, rtol=2.0e-6, atol=3.0e-7
            )


if __name__ == "__main__":
    unittest.main()
