# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared face Courant numbers on Bogner-FSL hydrodynamic masks."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeLbmModel,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.fsl_courant import (
    HomeFreeFslFaceCourantBuilder,
    fsl_face_courant,
)


def _affine_case(axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (12, 11, 10)
    coordinates = np.indices(shape, dtype=np.float64)
    for component in range(3):
        coordinates[component] += 0.5
    velocity = (
        0.017
        + 0.0013 * coordinates[0]
        - 0.0007 * coordinates[1]
        + 0.0009 * coordinates[2]
    )
    rho = 0.98 + 0.0005 * coordinates[0] + 0.0003 * coordinates[1]
    moments = np.zeros(shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1 + axis] = rho * velocity
    fill = np.zeros(shape, dtype=np.float64)
    active = np.zeros(shape, dtype=bool)
    lower = [slice(None)] * 3
    lower[axis] = 2
    interior = [slice(None)] * 3
    interior[axis] = slice(3, 6)
    upper = [slice(None)] * 3
    upper[axis] = 6
    fill[tuple(lower)] = 0.7
    fill[tuple(interior)] = 1.0
    fill[tuple(upper)] = 0.3
    active_lower = [slice(None)] * 3
    active_lower[axis] = slice(2, 6)
    active[tuple(active_lower)] = True
    return moments, active, fill, velocity


class TestHomeFreeFslCourant(unittest.TestCase):
    def _warp_matches_affine_reference(self, device: str) -> None:
        for axis in range(3):
            moments, active, fill, _ = _affine_case(axis)
            expected = fsl_face_courant(
                moments,
                active,
                fill,
                axis=axis,
                maximum_fit_condition=1.0e3,
            )
            model = HomeLbmModel(
                fluid_grid_res=fill.shape,
                periodic=(False, False, False),
                device=device,
            )
            state = HomeLbmState(model)
            state.moments.assign(
                np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
            )
            builder = HomeFreeFslFaceCourantBuilder(model, axis=axis)
            actual = builder.build(
                state,
                wp.array(active.astype(np.int32), dtype=wp.int32, device=device),
                wp.array(fill.astype(np.float32), dtype=float, device=device),
                wp.zeros(fill.shape, dtype=wp.int32, device=device),
            )
            np.testing.assert_allclose(
                actual.numpy(), expected.face_courant, rtol=2.0e-5, atol=3.0e-7
            )
            np.testing.assert_array_equal(
                builder.donor_count.numpy(), expected.donor_count
            )
            assert builder.last_diagnostics is not None
            self.assertEqual(builder.last_diagnostics.invalid_moment_count, 0)
            self.assertEqual(builder.last_diagnostics.insufficient_donor_face_count, 0)
            self.assertEqual(builder.last_diagnostics.ill_conditioned_face_count, 0)
            self.assertEqual(builder.last_diagnostics.invalid_courant_face_count, 0)
            self.assertGreater(builder.last_diagnostics.extrapolated_face_count, 0)
            self.assertEqual(builder.last_diagnostics.solid_face_count, 0)

    def _warp_closes_solid_faces(self, device: str) -> None:
        shape = (8, 7, 6)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.0
        moments[..., 1] = 0.04
        active = np.ones(shape, dtype=bool)
        fill = np.ones(shape, dtype=np.float64)
        flags = np.full(shape, int(HomeFreeCellFlag.LIQUID), dtype=np.int32)
        flags[4, 2:5, 1:5] = int(HomeFreeCellFlag.SOLID)
        active[flags == int(HomeFreeCellFlag.SOLID)] = False
        fill[flags == int(HomeFreeCellFlag.SOLID)] = 0.0
        expected = fsl_face_courant(
            moments,
            active,
            fill,
            axis=0,
            flags=flags,
        )
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        state = HomeLbmState(model)
        state.moments.assign(
            np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
        )
        builder = HomeFreeFslFaceCourantBuilder(model, axis=0)
        actual = builder.build(
            state,
            wp.array(active.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(fill.astype(np.float32), dtype=float, device=device),
            wp.array(flags, dtype=wp.int32, device=device),
        )
        np.testing.assert_array_equal(actual.numpy()[4:6, 2:5, 1:5], 0.0)
        np.testing.assert_allclose(actual.numpy(), expected.face_courant, atol=1.0e-7)
        assert builder.last_diagnostics is not None
        self.assertEqual(
            builder.last_diagnostics.solid_face_count,
            expected.solid_face_count,
        )
        self.assertGreater(expected.solid_face_count, 0)

    def test_affine_extrapolation_is_exact_on_all_axes(self) -> None:
        for axis in range(3):
            moments, active, fill, _ = _affine_case(axis)
            result = fsl_face_courant(
                moments,
                active,
                fill,
                axis=axis,
                maximum_fit_condition=1.0e3,
            )
            face_shape = list(fill.shape)
            face_shape[axis] += 1
            face_coordinates = np.indices(face_shape, dtype=np.float64)
            for component in range(3):
                face_coordinates[component] += 0.5
            face_coordinates[axis] -= 0.5
            expected = (
                0.017
                + 0.0013 * face_coordinates[0]
                - 0.0007 * face_coordinates[1]
                + 0.0009 * face_coordinates[2]
            )
            adjacent_liquid = np.zeros(face_shape, dtype=bool)
            lower_faces = [slice(None)] * 3
            lower_faces[axis] = slice(2, 8)
            adjacent_liquid[tuple(lower_faces)] = True
            np.testing.assert_allclose(
                result.face_courant[adjacent_liquid],
                expected[adjacent_liquid],
                rtol=0.0,
                atol=2.0e-14,
            )
            pure_gas = [slice(None)] * 3
            pure_gas[axis] = 9
            np.testing.assert_array_equal(result.face_courant[tuple(pure_gas)], 0.0)
            self.assertGreater(result.extrapolated_face_count, 0)
            self.assertLess(result.maximum_condition_number, 1.0e3)

    def test_periodic_uniform_state_has_matching_duplicate_faces(self) -> None:
        shape = (6, 5, 4)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.03
        moments[..., 2] = -0.012 * 1.03
        active = np.ones(shape, dtype=bool)
        fill = np.ones(shape, dtype=np.float64)
        result = fsl_face_courant(
            moments,
            active,
            fill,
            axis=1,
            periodic=(True, True, True),
        )
        np.testing.assert_allclose(result.face_courant, -0.012, atol=1.0e-15)
        np.testing.assert_array_equal(result.face_courant[:, 0], result.face_courant[:, -1])

    def test_rank_deficient_extrapolation_fails(self) -> None:
        shape = (7, 5, 4)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.0
        active = np.zeros(shape, dtype=bool)
        active[3, 2, 2] = True
        fill = np.zeros(shape, dtype=np.float64)
        fill[3, 2, 2] = 0.4
        with self.assertRaisesRegex(RuntimeError, "active affine donors"):
            fsl_face_courant(moments, active, fill, axis=0)

    def test_super_courant_active_state_fails(self) -> None:
        shape = (4, 3, 2)
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = 1.0
        moments[..., 1] = 1.01
        with self.assertRaisesRegex(RuntimeError, "invalid Courant"):
            fsl_face_courant(
                moments,
                np.ones(shape, dtype=bool),
                np.ones(shape, dtype=np.float64),
                axis=0,
                periodic=(True, True, True),
            )

    def test_warp_cpu_matches_affine_reference(self) -> None:
        self._warp_matches_affine_reference("cpu")

    def test_warp_cpu_closes_solid_faces(self) -> None:
        self._warp_closes_solid_faces("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_matches_affine_reference(self) -> None:
        self._warp_matches_affine_reference("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_closes_solid_faces(self) -> None:
        self._warp_closes_solid_faces("cuda:0")


if __name__ == "__main__":
    unittest.main()
