# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid midpoint/end transform sampling tests for HOME-FSI."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import RigidMotionPredictor


class TestHomeLbmMotion(unittest.TestCase):
    def test_constant_twist_preserves_and_advances_offset_center_of_mass(self) -> None:
        body_q = wp.array(
            [wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
            device="cpu",
        )
        linear = np.array([0.4, -0.2, 0.0])
        angular_speed = np.pi
        body_qd = wp.array(
            [[*linear, 0.0, 0.0, angular_speed]],
            dtype=wp.spatial_vector,
            device="cpu",
        )
        center_local = np.array([1.0, 0.0, 0.0])
        body_com = wp.array([center_local], dtype=wp.vec3, device="cpu")
        predictor = RigidMotionPredictor(1, device="cpu")

        predictor.sample(body_q, body_qd, body_com, dt=0.5)
        wp.synchronize_device("cpu")

        for sample, sample_dt in (
            (predictor.body_q_half.numpy()[0], 0.25),
            (predictor.body_q_end.numpy()[0], 0.5),
        ):
            angle = angular_speed * sample_dt
            expected_rotation = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)])
            rotated_center = np.array([np.cos(angle), np.sin(angle), 0.0])
            expected_center = np.array([2.0, 0.0, 0.0]) + linear * sample_dt
            expected_position = expected_center - rotated_center
            np.testing.assert_allclose(sample[:3], expected_position, atol=2.0e-7)
            np.testing.assert_allclose(sample[3:], expected_rotation, atol=2.0e-7)

    def test_surface_displacement_combines_translation_and_rotation(self) -> None:
        predictor = RigidMotionPredictor(1, device="cpu")
        body_qd = wp.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 3.0]],
            dtype=wp.spatial_vector,
            device="cpu",
        )
        radius = wp.array([2.0], dtype=float, device="cpu")

        displacement = predictor.measure_lattice_displacement(
            body_qd, radius, cell_size=0.5, time_step=0.1
        )

        self.assertAlmostEqual(displacement, 1.4, places=6)
        with self.assertRaisesRegex(ValueError, "reduce the physical HOME time step"):
            predictor.validate_lattice_displacement(
                body_qd, radius, cell_size=0.5, time_step=0.1, limit=0.5
            )


if __name__ == "__main__":
    unittest.main()
