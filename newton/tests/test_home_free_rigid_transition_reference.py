# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Executable phase contract for moving-solid HOME-Free transitions."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_lbm import classify_moving_solid_phases
from wanphys._src.fluid.fluid_grid.home_lbm.vof.flags import HomeFreeCellFlag
from wanphys._src.fluid.fluid_grid.home_lbm.vof.reference import classify_fill_levels


def _cube_mask(shape: tuple[int, int, int], lower: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    i, j, k = lower
    mask[i : i + 2, j : j + 2, k : k + 3] = True
    return mask


class TestHomeFreeRigidTransitionReference(unittest.TestCase):
    def test_horizontal_motion_reconstructs_liquid_interface_and_gas_fresh_cells(self) -> None:
        shape = (10, 6, 8)
        fill = np.zeros(shape, dtype=np.float64)
        fill[:, :, :4] = 1.0
        fill[:, :, 4] = 0.5
        previous_solid = _cube_mask(shape, (3, 2, 2))
        current_solid = _cube_mask(shape, (4, 2, 2))
        flags = classify_fill_levels(fill, previous_solid)

        result = classify_moving_solid_phases(
            previous_solid,
            current_solid,
            flags,
            fill,
        )

        fresh_indices = np.argwhere(result.fresh_mask)
        self.assertEqual(len(fresh_indices), 6)
        for i, j, k in fresh_indices:
            if k == 2:
                self.assertEqual(result.flags[i, j, k], HomeFreeCellFlag.LIQUID)
                self.assertEqual(result.fill_level[i, j, k], 1.0)
            else:
                self.assertEqual(result.flags[i, j, k], HomeFreeCellFlag.INTERFACE)
                self.assertGreater(result.fill_level[i, j, k], 0.0)
                self.assertLess(result.fill_level[i, j, k], 1.0)
        self.assertEqual(result.fresh_liquid_cells, 2)
        self.assertEqual(result.fresh_interface_cells, 4)
        self.assertEqual(result.fresh_gas_cells, 0)
        np.testing.assert_array_equal(
            result.flags[current_solid], int(HomeFreeCellFlag.SOLID)
        )

    def test_vertical_exit_releases_liquid_and_covers_gas(self) -> None:
        shape = (8, 6, 9)
        fill = np.zeros(shape, dtype=np.float64)
        fill[:, :, :4] = 1.0
        fill[:, :, 4] = 0.5
        previous_solid = _cube_mask(shape, (3, 2, 2))
        current_solid = _cube_mask(shape, (3, 2, 3))
        flags = classify_fill_levels(fill, previous_solid)

        result = classify_moving_solid_phases(
            previous_solid,
            current_solid,
            flags,
            fill,
        )

        np.testing.assert_array_equal(
            result.flags[:, :, 2][result.fresh_mask[:, :, 2]],
            int(HomeFreeCellFlag.LIQUID),
        )
        np.testing.assert_array_equal(
            result.flags[:, :, 5][result.dead_mask[:, :, 5]],
            int(HomeFreeCellFlag.SOLID),
        )
        self.assertEqual(result.fresh_liquid_cells, 4)
        self.assertEqual(result.fresh_interface_cells, 0)
        self.assertEqual(result.fresh_gas_cells, 0)

    def test_mask_mismatch_and_unresolved_fresh_cell_are_hard_errors(self) -> None:
        shape = (5, 5, 5)
        fill = np.zeros(shape, dtype=np.float64)
        previous_solid = np.zeros(shape, dtype=bool)
        previous_solid[2, 2, 2] = True
        flags = classify_fill_levels(fill, previous_solid)
        bad_flags = flags.copy()
        bad_flags[2, 2, 2] = int(HomeFreeCellFlag.GAS)
        with self.assertRaisesRegex(ValueError, "disagree"):
            classify_moving_solid_phases(
                previous_solid,
                np.zeros(shape, dtype=bool),
                bad_flags,
                fill,
            )

        previous_solid = np.ones(shape, dtype=bool)
        current_solid = previous_solid.copy()
        current_solid[2, 2, 2] = False
        flags = classify_fill_levels(fill, previous_solid)
        with self.assertRaisesRegex(RuntimeError, "no persistent fluid donor"):
            classify_moving_solid_phases(
                previous_solid,
                current_solid,
                flags,
                fill,
            )


if __name__ == "__main__":
    unittest.main()
