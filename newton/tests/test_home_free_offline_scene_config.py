# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Contracts for final HOME-Free offline scene construction."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeFreeGeometricDomain
from wanphys.examples.lbm.home_free_offline import (
    OfflineSceneLevel,
    OfflineSceneName,
    build_rectangular_liquid_fill,
    build_offline_scene,
    make_scene_config,
)
from wanphys.examples.lbm.fluid_grid_home_free_column import run as run_column


class TestHomeFreeOfflineSceneConfig(unittest.TestCase):
    def test_all_l0_configs_round_trip_with_stable_hash(self) -> None:
        for name in OfflineSceneName:
            with self.subTest(scene=name.value):
                config = make_scene_config(name)
                restored = type(config).from_json(config.to_json())
                self.assertEqual(restored, config)
                self.assertEqual(restored.config_hash, config.config_hash)
                self.assertEqual(len(config.config_hash), 64)

    def test_unvalidated_l1_and_l2_presets_fail(self) -> None:
        for level in (OfflineSceneLevel.L1, OfflineSceneLevel.L2):
            with self.subTest(level=level.value):
                with self.assertRaisesRegex(ValueError, "not frozen"):
                    make_scene_config(OfflineSceneName.DAMBREAK, level=level)

    def test_invalid_sphere_domain_overlap_fails(self) -> None:
        config = make_scene_config(OfflineSceneName.SPHERE_ENTRY)
        assert config.sphere is not None
        invalid_sphere = dataclasses.replace(config.sphere, position=(1.0, 12.0, 14.8))
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            dataclasses.replace(config, sphere=invalid_sphere)

    def test_physical_column_volume_is_grid_independent(self) -> None:
        coarse = build_rectangular_liquid_fill(
            (24, 12, 16),
            cell_size=1.0,
            height=8.25,
            x_range=(0.0, 8.25),
        )
        fine = build_rectangular_liquid_fill(
            (48, 24, 32),
            cell_size=0.5,
            height=8.25,
            x_range=(0.0, 8.25),
        )
        coarse_volume = float(np.sum(coarse, dtype=np.float64))
        fine_volume = float(np.sum(fine, dtype=np.float64)) * 0.5**3
        self.assertAlmostEqual(coarse_volume, fine_volume, places=10)

    def test_acoustic_grid_scaling_records_dimensionless_contract(self) -> None:
        coarse = make_scene_config(OfflineSceneName.DAMBREAK)
        fine = dataclasses.replace(
            coarse,
            steps=2 * coarse.steps,
            output_every_steps=2 * coarse.output_every_steps,
            fluid=dataclasses.replace(
                coarse.fluid,
                resolution=tuple(2 * value for value in coarse.fluid.resolution),
                cell_size=0.5 * coarse.fluid.cell_size,
                time_step=0.5 * coarse.fluid.time_step,
            ),
        )
        coarse_derived = coarse.derived_parameters
        fine_derived = fine.derived_parameters
        self.assertEqual(
            fine_derived["physical_extent"], coarse_derived["physical_extent"]
        )
        self.assertAlmostEqual(fine_derived["reynolds"], coarse_derived["reynolds"])
        self.assertAlmostEqual(fine_derived["froude"], coarse_derived["froude"])
        self.assertAlmostEqual(
            fine_derived["lattice_acceleration"][2],
            0.5 * coarse_derived["lattice_acceleration"][2],
        )
        self.assertGreater(
            fine_derived["shear_relaxation_time"],
            coarse_derived["shear_relaxation_time"],
        )

    def test_all_l0_scenes_build_geometric_domain_on_cpu(self) -> None:
        for name in OfflineSceneName:
            with self.subTest(scene=name.value):
                scene = build_offline_scene(make_scene_config(name, device="cpu"))
                self.assertIsInstance(scene.fluid, HomeFreeGeometricDomain)
                self.assertGreater(scene.initial_mass, 0.0)
                self.assertEqual(scene.step_index, 0)
                scene.step()
                metrics = scene.measure()
                self.assertEqual(metrics.step, 1)
                self.assertLess(metrics.relative_mass_error, 2.0e-5)
                self.assertLess(metrics.projected_max_divergence, 2.0e-8)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_all_l0_scenes_build_and_step_on_cuda(self) -> None:
        for name in OfflineSceneName:
            with self.subTest(scene=name.value):
                scene = build_offline_scene(make_scene_config(name, device="cuda:0"))
                scene.step()
                metrics = scene.measure()
                self.assertLess(metrics.relative_mass_error, 2.0e-5)
                self.assertLess(metrics.projected_max_divergence, 2.0e-8)

    def test_public_column_example_uses_shared_scene(self) -> None:
        result = run_column(2, "cpu")
        self.assertEqual(result["steps"], 2.0)
        self.assertLess(result["relative_mass_error"], 2.0e-5)
        self.assertLessEqual(result["maximum_fill"], 1.0)


if __name__ == "__main__":
    unittest.main()
