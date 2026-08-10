# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    axis_aligned_wall_mask,
    closed_box_wall_mask,
    initialize_hydrostatic_fields,
    validate_closed_wall_mask,
    validate_wall_mask,
    wall_link_mass_exchange,
    wall_only_missing_stream_moments,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    _impact_audit,
    _trapped_gas_metrics,
)


def _rest_moments(shape: tuple[int, int, int]) -> np.ndarray:
    moments = np.zeros(shape + (10,), dtype=np.float64)
    moments[..., 0] = 1.0
    return moments


class TestFslWallReference(unittest.TestCase):
    def test_closed_box_requires_all_six_boundary_planes(self) -> None:
        solid = closed_box_wall_mask((8, 7, 6))
        self.assertEqual(int(np.count_nonzero(~solid)), 6 * 5 * 4)
        validate_closed_wall_mask(solid)
        solid[0, 3, 3] = False
        with self.assertRaisesRegex(ValueError, "lower axis-0"):
            validate_closed_wall_mask(solid)

    def test_stationary_uniform_liquid_is_unchanged_by_wall_bounce(self) -> None:
        shape = (8, 7, 6)
        solid = closed_box_wall_mask(shape)
        moments = _rest_moments(shape)
        flags = np.full(shape, int(FslCellFlag.LIQUID), dtype=np.int32)
        flags[solid] = int(FslCellFlag.GAS)
        result = wall_only_missing_stream_moments(moments, flags, solid)

        np.testing.assert_allclose(result.moments[~solid], moments[~solid], atol=2.0e-15)
        self.assertGreater(int(np.sum(result.wall_link_count)), 0)
        self.assertEqual(int(np.sum(result.gas_link_count)), 0)

    def test_periodic_depth_channel_closes_only_xy_planes(self) -> None:
        solid = axis_aligned_wall_mask((8, 7, 1), closed_axes=(True, True, False))
        self.assertEqual(int(np.count_nonzero(~solid)), 6 * 5)
        self.assertTrue(np.all(solid[0, :, :]))
        self.assertTrue(np.all(solid[:, -1, :]))
        self.assertFalse(solid[3, 3, 0])
        validate_wall_mask(solid, closed_axes=(True, True, False))
        with self.assertRaisesRegex(ValueError, "closed axes require interior cells"):
            validate_closed_wall_mask(solid)

    def test_periodic_depth_hydrostatic_initializer_accepts_single_layer(self) -> None:
        solid = axis_aligned_wall_mask((8, 8, 1), closed_axes=(True, True, False))
        fill = np.zeros(solid.shape, dtype=np.float64)
        fill[1:5, 1:4, :] = 1.0
        fill[1:6, 4, :] = 0.5
        fields = initialize_hydrostatic_fields(
            fill,
            solid,
            gas_density=1.0,
            gravity_axis=1,
            lattice_acceleration=-1.0e-4,
            surface_coordinate=4.0,
            closed_axes=(True, True, False),
        )
        self.assertEqual(fields.mass.shape, (8, 8, 1))
        np.testing.assert_array_equal(fields.mass[solid], 0.0)

    def test_wall_links_have_exactly_zero_mass_flux(self) -> None:
        shape = (8, 7, 6)
        solid = closed_box_wall_mask(shape)
        moments = _rest_moments(shape)
        flags = np.full(shape, int(FslCellFlag.LIQUID), dtype=np.int32)
        flags[solid] = int(FslCellFlag.GAS)
        mass = np.ones(shape, dtype=np.float64)
        mass[solid] = 0.0
        fill = mass.copy()

        advected = wall_link_mass_exchange(moments, mass, fill, flags, solid)

        np.testing.assert_array_equal(advected, mass)

    def test_hydrostatic_density_anchors_the_horizontal_surface(self) -> None:
        shape = (8, 8, 6)
        solid = closed_box_wall_mask(shape)
        fill = np.zeros(shape, dtype=np.float64)
        fill[:, 1:4, :] = 1.0
        fill[:, 4, :] = 0.5
        fill[solid] = 0.0
        fields = initialize_hydrostatic_fields(
            fill,
            solid,
            gas_density=1.0,
            gravity_axis=1,
            lattice_acceleration=-1.0e-4,
            surface_coordinate=4.0,
        )

        self.assertAlmostEqual(fields.moments[3, 4, 3, 0], 1.0, places=15)
        self.assertAlmostEqual(
            fields.moments[3, 1, 3, 0], np.exp(9.0e-4), places=15
        )
        self.assertGreater(fields.moments[3, 1, 3, 0], fields.moments[3, 3, 3, 0])
        np.testing.assert_array_equal(fields.mass, fields.moments[..., 0] * fill)
        np.testing.assert_array_equal(fields.mass[solid], 0.0)

    def test_trapped_gas_metric_excludes_the_connected_headspace(self) -> None:
        solid = closed_box_wall_mask((7, 7, 3))[:, :, 1]
        flags = np.full((7, 7), int(FslCellFlag.GAS), dtype=np.int32)
        for neighbor in ((2, 3), (4, 3), (3, 2), (3, 4)):
            flags[neighbor] = int(FslCellFlag.INTERFACE)

        self.assertEqual(_trapped_gas_metrics(flags, solid), (1, 1))
        flags[3, 4] = int(FslCellFlag.GAS)
        self.assertEqual(_trapped_gas_metrics(flags, solid), (0, 0))

    def test_impact_audit_requires_contact_reflection_and_conservation(self) -> None:
        metrics = {
            "0": {"right_wall_wet_cells": 0, "center_of_mass_x": 20.0, "front_x": 41, "relative_total_mass_drift": 0.0},
            "4000": {"right_wall_wet_cells": 20, "center_of_mass_x": 90.0, "front_x": 158, "relative_total_mass_drift": 2.0e-5},
            "8000": {"right_wall_wet_cells": 15, "center_of_mass_x": 75.0, "front_x": 158, "relative_total_mass_drift": -4.0e-5},
        }
        audit = _impact_audit(metrics, interior_right_x=158, max_active_speed=0.08)
        self.assertEqual(audit["first_sampled_wall_contact_step"], 4000)
        self.assertAlmostEqual(audit["reflected_center_of_mass_displacement"], 15.0)
        self.assertTrue(audit["reflection_observed"])
        self.assertTrue(audit["mass_drift_within_1e-3"])
        self.assertTrue(audit["no_solid_penetration"])


if __name__ == "__main__":
    unittest.main()
