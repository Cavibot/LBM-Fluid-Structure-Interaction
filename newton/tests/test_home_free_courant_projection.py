# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Hodge projection of frozen HOME-Free geometric face Courants."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeFreeCellFlag, HomeLbmModel
from wanphys._src.fluid.fluid_grid.home_lbm.vof.courant_projection import (
    HomeFreeCourantProjector,
    project_face_courant_reference,
)


class TestHomeFreeCourantProjection(unittest.TestCase):
    @staticmethod
    def _case() -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        shape = (5, 4, 3)
        rng = np.random.default_rng(32416190071)
        faces = []
        for axis in range(3):
            face_shape = list(shape)
            face_shape[axis] += 1
            field = rng.uniform(-0.012, 0.012, size=face_shape)
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = 0
            upper[axis] = shape[axis]
            field[tuple(upper)] = field[tuple(lower)]
            faces.append(field)
        flags = np.full(shape, int(HomeFreeCellFlag.GAS), dtype=np.int32)
        flags[1:4, 1:3, 1:3] = int(HomeFreeCellFlag.LIQUID)
        flags[1, 1:3, 1:3] = int(HomeFreeCellFlag.INTERFACE)
        flags[4, 1, 1] = int(HomeFreeCellFlag.SOLID)
        return tuple(faces), flags

    def _warp_matches_reference(self, device: str) -> None:
        faces, flags = self._case()
        expected = project_face_courant_reference(
            faces, flags, periodic=(True, True, True)
        )
        model = HomeLbmModel(
            fluid_grid_res=flags.shape,
            periodic=(True, True, True),
            device=device,
        )
        projector = HomeFreeCourantProjector(
            model,
            max_iterations=320,
            relative_tolerance=2.0e-6,
            absolute_divergence_tolerance=1.0e-7,
            check_interval=4,
        )
        actual = projector.project(
            tuple(
                wp.array(field.astype(np.float32), dtype=float, device=device)
                for field in faces
            ),
            wp.array(flags, dtype=wp.int32, device=device),
        )

        self.assertGreater(actual.diagnostics.iteration_count, 0)
        self.assertLess(actual.diagnostics.relative_residual, 2.0e-6)
        self.assertLess(actual.diagnostics.projected_max_divergence, 1.0e-7)
        self.assertAlmostEqual(
            actual.diagnostics.initial_max_divergence,
            expected.initial_max_divergence,
            delta=2.0e-8,
        )
        for actual_face, expected_face in zip(
            actual.face_courant, expected.face_courant, strict=True
        ):
            np.testing.assert_allclose(
                actual_face.numpy(), expected_face, rtol=2.0e-4, atol=2.0e-6
            )
        self.assertEqual(float(actual.face_courant[0].numpy()[4, 1, 1]), 0.0)

    def test_numpy_projection_closes_active_divergence(self) -> None:
        faces, flags = self._case()
        result = project_face_courant_reference(
            faces, flags, periodic=(True, True, True)
        )
        self.assertGreater(result.initial_max_divergence, 1.0e-3)
        self.assertLess(result.projected_max_divergence, 1.0e-14)
        self.assertEqual(float(result.face_courant[0][4, 1, 1]), 0.0)

    def test_warp_projection_matches_reference_on_cpu(self) -> None:
        self._warp_matches_reference("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_projection_matches_reference_on_cuda(self) -> None:
        self._warp_matches_reference("cuda:0")


if __name__ == "__main__":
    unittest.main()
