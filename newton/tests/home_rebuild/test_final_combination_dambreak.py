# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np


from wanphys.examples.lbm.home_rebuild.final_combination import (
    CombinationBackend,
    build_combination_dambreak,
    build_combination_fill,
    make_combination_config,
)


class TestFinalCombinationDambreak(unittest.TestCase):
    def test_profiles_share_physics_and_only_strict_projects(self) -> None:
        fast = make_combination_config("fast")
        balanced = make_combination_config("balanced")
        strict = make_combination_config("strict")

        self.assertEqual(fast.resolution, balanced.resolution)
        self.assertEqual(fast.kinematic_viscosity, balanced.kinematic_viscosity)
        self.assertEqual(fast.gravity, balanced.gravity)
        self.assertEqual(balanced.periodic, (False, False, False))
        self.assertEqual(balanced.advection_axes, (0, 1, 2))
        self.assertAlmostEqual(fast.lattice_viscosity, 0.02)
        self.assertAlmostEqual(fast.lattice_gravity, -5.0e-5)
        self.assertEqual(fast.surface_tension, 0.0)
        self.assertAlmostEqual(balanced.lattice_surface_tension, 0.005)
        self.assertEqual(fast.relative_mass_tolerance, 5.0e-3)
        self.assertEqual(balanced.relative_mass_tolerance, 2.0e-5)
        self.assertFalse(balanced.project_courant)
        self.assertTrue(strict.project_courant)

    def test_fill_is_rectangular_and_touches_the_floor(self) -> None:
        config = make_combination_config(CombinationBackend.BALANCED)
        fill = build_combination_fill(config)

        self.assertEqual(fill.shape, config.resolution)
        self.assertAlmostEqual(
            float(np.sum(fill)),
            config.column_width_cells
            * config.resolution[1]
            * config.column_height_cells,
        )
        self.assertTrue(np.all(fill[:8, :, 0] == 1.0))
        self.assertTrue(np.all(fill[9:, :, 0] == 0.0))
        self.assertTrue(np.all(fill[:, :, -1] == 0.0))

    def test_fast_and_balanced_take_a_finite_cpu_step(self) -> None:
        for backend in (CombinationBackend.FAST, CombinationBackend.BALANCED):
            with self.subTest(backend=backend.value):
                scene = build_combination_dambreak(
                    make_combination_config(backend, device="cpu")
                )
                initial = scene.measure()
                scene.step()
                scene.synchronize()
                evolved = scene.measure()
                self.assertEqual(initial.floor_gap_columns, 0)
                self.assertEqual(initial.detached_bulk_columns, 0)
                self.assertEqual(evolved.invalid_fluid_cells, 0)
                self.assertEqual(evolved.direct_liquid_gas_links, 0)
                self.assertEqual(evolved.invalid_surface_tension_cells, 0)
                self.assertTrue(np.isfinite(evolved.maximum_speed))





if __name__ == "__main__":
    unittest.main()
