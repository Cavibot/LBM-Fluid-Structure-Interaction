# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the LBM dam-break falling-sphere visual example."""

from __future__ import annotations

import importlib
import math
import types
import unittest
from typing import Any

import numpy as np

MODULE_NAME: str = "wanphys.examples.lbm.fluid_grid_lbm_dambreak_falling_sphere_visual"


def _import_module() -> types.ModuleType:
    return importlib.import_module(MODULE_NAME)


class TestLbmDamBreakFallingSphereVisualization(unittest.TestCase):
    def test_smoke_reports_finite_density_and_rigid_metrics(self) -> None:
        module: types.ModuleType = _import_module()

        metrics: Any = module.run_smoke(
            num_frames=8,
            grid_res=(28, 20, 24),
            density_preset="light",
        )

        self.assertTrue(metrics.density_all_finite)
        self.assertTrue(np.all(np.isfinite(metrics.body_position)))
        self.assertTrue(np.all(np.isfinite(metrics.body_velocity)))
        self.assertTrue(math.isfinite(float(metrics.density_min)))
        self.assertTrue(math.isfinite(float(metrics.density_max)))
        self.assertGreater(int(metrics.water_cells), 0)
        self.assertGreater(int(metrics.solid_cells), 0)

    def test_keeps_water_field_stable_after_startup(self) -> None:
        module: types.ModuleType = _import_module()

        metrics: Any = module.run_smoke(
            num_frames=60,
            grid_res=(32, 24, 28),
            density_preset="light",
        )

        self.assertTrue(metrics.density_all_finite)
        self.assertGreater(int(metrics.water_cells), 1000)
        self.assertLess(float(metrics.density_max), 5.0)

    def test_light_sphere_stays_higher_than_heavy_sphere(self) -> None:
        module: types.ModuleType = _import_module()

        heavy: Any = module.run_smoke(
            num_frames=96,
            grid_res=(32, 24, 28),
            density_preset="heavy",
        )
        light: Any = module.run_smoke(
            num_frames=96,
            grid_res=(32, 24, 28),
            density_preset="light",
        )

        self.assertLess(float(heavy.body_position[2]) + 0.02, float(light.body_position[2]))
        self.assertLess(float(heavy.min_body_z) + 0.02, float(light.min_body_z))
        self.assertGreater(float(light.horizontal_displacement), 0.005)

    def test_local_flow_drag_pushes_sphere_with_observed_downstream_flow(self) -> None:
        module: types.ModuleType = _import_module()

        metrics: Any = module.run_smoke(
            num_frames=96,
            grid_res=(32, 24, 28),
            density_preset="neutral",
        )

        self.assertGreater(float(metrics.horizontal_displacement), 0.005)
        self.assertGreater(float(metrics.max_water_contact_fraction), 0.10)

    def test_showcase_starts_in_air_then_contacts_dambreak_water(self) -> None:
        module: types.ModuleType = _import_module()

        metrics: Any = module.run_smoke(
            num_frames=96,
            grid_res=(32, 24, 28),
            density_preset="light",
        )

        self.assertLess(float(metrics.initial_water_contact_fraction), 0.05)
        self.assertGreater(float(metrics.max_water_contact_fraction), 0.10)
        self.assertGreater(float(metrics.initial_body_position[2] - metrics.min_body_z), 0.03)
        self.assertGreater(float(metrics.horizontal_displacement), 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)

