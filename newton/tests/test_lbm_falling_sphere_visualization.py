# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math
import types
import unittest
from typing import Any

import numpy as np
import warp as wp

FALLING_MODULE_NAME: str = "wanphys.examples.lbm.fluid_grid_lbm_falling_sphere_twophase_visual"
DENSITY_MODULE_NAME: str = "wanphys.examples.lbm.fluid_grid_lbm_density_sphere_twophase_visual"


@wp.kernel
def _write_density_cell_kernel(density: wp.array3d(dtype=float), value: float) -> None:
    density[0, 0, 0] = value


def _import_module(module_name: str) -> types.ModuleType:
    return importlib.import_module(module_name)


class TestLbmFallingSphereVisualization(unittest.TestCase):
    def test_falling_sphere_visual_smoke_reports_finite_metrics(self) -> None:
        module: types.ModuleType = _import_module(FALLING_MODULE_NAME)

        metrics: Any = module.run_smoke(num_frames=4, grid_res=(24, 16, 20), tracer_count=0)

        self.assertTrue(np.all(np.isfinite(metrics.body_position)))
        self.assertTrue(math.isfinite(float(metrics.density_min)))
        self.assertTrue(math.isfinite(float(metrics.density_max)))
        self.assertGreater(int(metrics.solid_cells), 0)
        self.assertGreater(float(metrics.density_max), float(metrics.density_min))

    def test_falling_sphere_visual_prescribed_body_moves_downward(self) -> None:
        module: types.ModuleType = _import_module(FALLING_MODULE_NAME)

        metrics: Any = module.run_smoke(num_frames=8, grid_res=(24, 16, 20), tracer_count=0)

        self.assertLess(float(metrics.body_position[2]), float(metrics.initial_body_position[2]))
        self.assertGreater(float(metrics.water_cells), 0.0)

    def test_falling_sphere_visual_keeps_water_field_stable_after_startup(self) -> None:
        module: types.ModuleType = _import_module(FALLING_MODULE_NAME)

        metrics: Any = module.run_smoke(num_frames=30, grid_res=(32, 20, 28), tracer_count=0)

        self.assertTrue(metrics.density_all_finite)
        self.assertGreater(int(metrics.water_cells), 1000)
        self.assertLess(float(metrics.density_max), 5.0)

    def test_density_force_helper_light_sphere_stays_higher_without_mem(self) -> None:
        helpers: types.ModuleType = _import_module("wanphys.examples.lbm._lbm_falling_sphere_scene")

        def run_preset(preset: str) -> Any:
            cfg: Any = helpers.FallingSphereSceneConfig(
                grid_res=(24, 16, 20),
                tracer_count=0,
                sphere_density=helpers.SPHERE_DENSITY_PRESETS[preset],
                sphere_start_fraction=(0.5, 0.5, 0.45),
                analytic_force_scale=100.0,
            )
            scene: Any = helpers.build_falling_sphere_scene(cfg, advance_rigid=True, two_way=False)
            metrics: Any = helpers.collect_falling_sphere_metrics(scene)
            for _frame_index in range(18):
                metrics = helpers.step_free_density_sphere_scene(scene, 1.0 / 120.0)
            return metrics

        heavy: Any = run_preset("heavy")
        light: Any = run_preset("light")

        self.assertTrue(np.all(np.isfinite(heavy.body_position)))
        self.assertTrue(np.all(np.isfinite(light.body_position)))
        self.assertLess(float(heavy.body_position[2]), float(light.body_position[2]))

    def test_density_metrics_expose_nonfinite_density_cell(self) -> None:
        helpers: types.ModuleType = _import_module("wanphys.examples.lbm._lbm_falling_sphere_scene")
        cfg: Any = helpers.FallingSphereSceneConfig(grid_res=(24, 16, 20), tracer_count=0)
        scene: Any = helpers.build_falling_sphere_scene(cfg, advance_rigid=False, two_way=False)

        wp.launch(
            _write_density_cell_kernel,
            dim=1,
            inputs=[scene.fluid_domain.state.density, float("nan")],
            device=scene.fluid_domain.model._device,
        )

        metrics: Any = helpers.collect_falling_sphere_metrics(scene)

        self.assertFalse(metrics.density_all_finite)
        self.assertTrue(math.isfinite(float(metrics.density_min)))
        self.assertTrue(math.isfinite(float(metrics.density_max)))

    def test_density_sphere_visual_smoke_accepts_density_presets(self) -> None:
        module: types.ModuleType = _import_module(DENSITY_MODULE_NAME)

        for preset in ("heavy", "neutral", "light"):
            with self.subTest(preset=preset):
                metrics: Any = module.run_smoke(
                    num_frames=6,
                    grid_res=(24, 16, 20),
                    density_preset=preset,
                    tracer_count=0,
                )
                self.assertTrue(np.all(np.isfinite(metrics.body_position)))
                self.assertTrue(math.isfinite(float(metrics.submerged_fraction)))
                self.assertGreaterEqual(float(metrics.submerged_fraction), 0.0)
                self.assertLessEqual(float(metrics.submerged_fraction), 1.0)

    def test_density_sphere_visual_light_sphere_stays_higher_than_heavy_sphere(self) -> None:
        module: types.ModuleType = _import_module(DENSITY_MODULE_NAME)

        heavy: Any = module.run_smoke(
            num_frames=18,
            grid_res=(24, 16, 20),
            density_preset="heavy",
            tracer_count=0,
        )
        light: Any = module.run_smoke(
            num_frames=18,
            grid_res=(24, 16, 20),
            density_preset="light",
            tracer_count=0,
        )

        self.assertLess(float(heavy.body_position[2]), float(light.body_position[2]))

    def test_density_sphere_visual_keeps_water_field_stable_after_startup(self) -> None:
        module: types.ModuleType = _import_module(DENSITY_MODULE_NAME)

        metrics: Any = module.run_smoke(
            num_frames=30,
            grid_res=(32, 20, 28),
            density_preset="heavy",
            tracer_count=0,
        )

        self.assertTrue(metrics.density_all_finite)
        self.assertGreater(int(metrics.water_cells), 1000)
        self.assertLess(float(metrics.density_max), 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

