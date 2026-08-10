# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    active_divergence,
    build_face_courant_reference,
    project_face_courant_reference,
)


class TestCourantReference(unittest.TestCase):
    def test_home_velocity_builder_uses_shared_faces_and_blocks_solids(self) -> None:
        shape = (10, 8, 1)
        closed_axes = (True, True, False)
        solid = axis_aligned_wall_mask(shape, closed_axes=closed_axes)
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[1:6, 1:5, :] = int(FslCellFlag.LIQUID)
        flags[6, 1:5, :] = int(FslCellFlag.INTERFACE)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.0
        moments[..., 1] = 0.08
        moments[..., 2] = -0.03
        fields = build_face_courant_reference(
            moments, flags, solid, closed_axes=closed_axes
        )

        self.assertAlmostEqual(fields.face_courant[0][4, 3, 0], 0.08)
        self.assertAlmostEqual(fields.face_courant[0][7, 3, 0], 0.08)
        self.assertEqual(fields.face_courant[0][1, 3, 0], 0.0)
        self.assertEqual(fields.face_courant[1][3, 1, 0], 0.0)
        self.assertEqual(fields.maximum_courant, 0.08)

    def test_projection_removes_active_divergence_with_periodic_single_depth(self) -> None:
        shape = (12, 10, 1)
        closed_axes = (True, True, False)
        solid = axis_aligned_wall_mask(shape, closed_axes=closed_axes)
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[1:9, 1:7, :] = int(FslCellFlag.LIQUID)
        flags[9, 1:7, :] = int(FslCellFlag.INTERFACE)
        active = (flags != int(FslCellFlag.GAS)) & ~solid
        rng = np.random.default_rng(44021)
        faces = [
            rng.uniform(-0.05, 0.05, size=(13, 10, 1)),
            rng.uniform(-0.05, 0.05, size=(12, 11, 1)),
            np.zeros((12, 10, 2), dtype=np.float64),
        ]
        for axis, field in enumerate(faces[:2]):
            for face in np.ndindex(field.shape):
                coordinate = face[axis]
                if coordinate in (0, shape[axis]):
                    field[face] = 0.0
                    continue
                left = list(face)
                right = list(face)
                left[axis] -= 1
                if solid[tuple(left)] or solid[tuple(right)]:
                    field[face] = 0.0
        result = project_face_courant_reference(
            tuple(faces),
            flags,
            solid,
            closed_axes=closed_axes,
            absolute_tolerance=2.0e-10,
        )

        divergence = active_divergence(result.face_courant, active)
        self.assertGreater(result.initial_maximum_divergence, 1.0e-3)
        self.assertLess(float(np.max(np.abs(divergence[active]))), 2.0e-10)
        self.assertGreater(result.maximum_face_correction, 0.0)
        self.assertTrue(np.all(result.face_courant[0][1, :, :] == 0.0))
        self.assertTrue(np.all(result.face_courant[1][:, 1, :] == 0.0))


if __name__ == "__main__":
    unittest.main()
