# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometric PLIC volume flux for the Bogner-FSL research path."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeLbmModel
from wanphys._src.fluid.fluid_grid.home_lbm.vof.geometric_advection import (
    HomeFreeGeometricAxisAdvector,
    advect_plic_axis,
    plic_swept_slab_volume,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.plic import plic_plane_offset


def _planar_periodic_geometry(fill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros(fill.shape + (3,), dtype=np.float64)
    offsets = np.zeros(fill.shape, dtype=np.float64)
    for i in (0, 5):
        normal = np.asarray((-1.0, 0.0, 0.0) if i == 0 else (1.0, 0.0, 0.0))
        normals[i] = normal
        offsets[i] = plic_plane_offset(float(fill[i, 0, 0]), normal)
    return normals, offsets


def _center_side_active(
    fill: np.ndarray, normals: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    fractional = (fill > 0.0) & (fill < 1.0)
    valid = np.linalg.norm(normals, axis=-1) > 0.0
    return (fill == 1.0) | (fractional & valid & (offsets >= 0.0))


class TestHomeFreeGeometricAdvection(unittest.TestCase):
    def _warp_near_axis_sliver_uses_reconstruction_dimension(
        self, device: str
    ) -> None:
        shape = (3, 3, 3)
        donor = (1, 1, 1)
        face = (2, 1, 1)
        donor_fill = np.float32(6.894107400512439e-7)
        signed_courant = np.float32(0.01282152347266674)
        fill = np.zeros(shape, dtype=np.float32)
        normal = np.zeros(shape + (3,), dtype=np.float32)
        offset = np.zeros(shape, dtype=np.float32)
        courant = np.zeros((shape[0] + 1, shape[1], shape[2]), dtype=np.float32)
        fill[donor] = donor_fill
        normal[donor] = (-1.0, 3.72925860574469e-5, 6.123234262925839e-17)
        offset[donor] = np.float32(-0.4999993145465851)
        courant[face] = signed_courant
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        advector = HomeFreeGeometricAxisAdvector(model, axis=0)

        advector.advect(
            wp.array(fill, dtype=float, device=device),
            wp.array(normal, dtype=wp.vec3, device=device),
            wp.array(offset, dtype=float, device=device),
            wp.array(courant, dtype=float, device=device),
        )

        flux = float(advector.face_flux.numpy()[face])
        self.assertLessEqual(flux, float(donor_fill))
        self.assertAlmostEqual(flux, float(donor_fill), delta=2.0e-10)
        assert advector.last_diagnostics is not None
        self.assertEqual(advector.last_diagnostics.invalid_face_count, 0)

    def _warp_rejects_gross_flux_bound_correction(self, device: str) -> None:
        shape = (3, 3, 3)
        fill = np.zeros(shape, dtype=np.float32)
        normals = np.zeros(shape + (3,), dtype=np.float32)
        offsets = np.zeros(shape, dtype=np.float32)
        courant = np.zeros((shape[0], shape[1], shape[2] + 1), dtype=np.float32)
        donor = (1, 1, 1)
        fill[donor] = 0.9
        normals[donor] = (0.0, 0.0, 1.0)
        offsets[donor] = plic_plane_offset(0.1, normals[donor])
        courant[1, 1, 2] = 0.2
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        advector = HomeFreeGeometricAxisAdvector(model, axis=2)

        with self.assertRaisesRegex(RuntimeError, "maximum correction"):
            advector.advect(
                wp.array(fill, dtype=float, device=device),
                wp.array(normals, dtype=wp.vec3, device=device),
                wp.array(offsets, dtype=float, device=device),
                wp.array(courant, dtype=float, device=device),
            )
        assert advector.last_diagnostics is not None
        self.assertEqual(advector.last_diagnostics.bounded_face_count, 1)
        self.assertGreater(
            advector.last_diagnostics.maximum_bound_correction, 0.09
        )

    def _warp_oblique_near_full_flux_respects_intersection_bound(
        self, device: str
    ) -> None:
        shape = (3, 3, 3)
        fill_value = np.float32(0.9999994039535522)
        signed_courant = np.float32(0.0030099968425929546)
        normal = np.asarray(
            (-0.93517399, 0.01653305, 0.35380265), dtype=np.float32
        )
        fill = np.zeros(shape, dtype=np.float32)
        normals = np.zeros(shape + (3,), dtype=np.float32)
        offsets = np.zeros(shape, dtype=np.float32)
        courant = np.zeros((shape[0], shape[1], shape[2] + 1), dtype=np.float32)
        compression = np.zeros(shape, dtype=np.int32)
        donor = (1, 1, 1)
        incoming_donor = (1, 1, 0)
        incoming_face = (1, 1, 1)
        outgoing_face = (1, 1, 2)
        fill[donor] = fill_value
        fill[incoming_donor] = 1.0
        normals[donor] = normal
        offsets[donor] = np.float32(0.6500602960586548)
        courant[incoming_face] = np.float32(0.003010522)
        courant[outgoing_face] = signed_courant
        compression[donor] = 1
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        advector = HomeFreeGeometricAxisAdvector(model, axis=2)

        result = advector.advect(
            wp.array(fill, dtype=float, device=device),
            wp.array(normals, dtype=wp.vec3, device=device),
            wp.array(offsets, dtype=float, device=device),
            wp.array(courant, dtype=float, device=device),
            wp.array(compression, dtype=wp.int32, device=device),
        )

        flux = float(advector.face_flux.numpy()[outgoing_face])
        lower_bound = float(signed_courant - (1.0 - fill_value))
        self.assertGreaterEqual(flux, lower_bound)
        self.assertLessEqual(flux, float(signed_courant))
        self.assertLessEqual(float(result.numpy()[donor]), 1.0)
        assert advector.last_diagnostics is not None
        self.assertEqual(advector.last_diagnostics.invalid_cell_count, 0)

    def _warp_near_full_flux_respects_support_bound(self, device: str) -> None:
        shape = (3, 3, 3)
        fill_value = np.float32(0.99987644)
        signed_courant = np.float32(0.00019622757)
        normal = np.asarray((2.718711e-7, -1.0, 6.796778e-8), dtype=np.float32)
        fill = np.zeros(shape, dtype=np.float32)
        normals = np.zeros(shape + (3,), dtype=np.float32)
        offsets = np.zeros(shape, dtype=np.float32)
        courant = np.zeros((shape[0], shape[1] + 1, shape[2]), dtype=np.float32)
        donor = (1, 1, 1)
        face = (1, 2, 1)
        fill[donor] = fill_value
        normals[donor] = normal
        offsets[donor] = plic_plane_offset(float(fill_value), normal)
        courant[face] = signed_courant
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        advector = HomeFreeGeometricAxisAdvector(model, axis=1)

        advector.advect(
            wp.array(fill, dtype=float, device=device),
            wp.array(normals, dtype=wp.vec3, device=device),
            wp.array(offsets, dtype=float, device=device),
            wp.array(courant, dtype=float, device=device),
        )

        actual = float(advector.face_flux.numpy()[face])
        lower_bound = float(signed_courant - (1.0 - fill_value))
        self.assertGreaterEqual(actual, lower_bound)
        self.assertAlmostEqual(actual, float(signed_courant), places=10)

    def _warp_weymouth_yue_fixed_phases(self, device: str) -> None:
        shape = (5, 3, 2)
        normals = np.zeros(shape + (3,), dtype=np.float32)
        offsets = np.zeros(shape, dtype=np.float32)
        courant = np.zeros((shape[0] + 1, shape[1], shape[2]), dtype=np.float32)
        courant[1] = 0.05
        courant[2] = 0.02
        courant[3] = -0.01
        courant[4] = 0.03
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        advector = HomeFreeGeometricAxisAdvector(model, axis=0)
        for value, compression_value in ((1.0, 1), (0.0, 0)):
            fill = np.full(shape, value, dtype=np.float32)
            compression = np.full(shape, compression_value, dtype=np.int32)
            expected = advect_plic_axis(
                fill,
                normals,
                offsets,
                courant,
                axis=0,
                periodic=False,
                compression=compression,
            )
            result = advector.advect(
                wp.array(fill, dtype=float, device=device),
                wp.array(normals, dtype=wp.vec3, device=device),
                wp.array(offsets, dtype=float, device=device),
                wp.array(courant, dtype=float, device=device),
                wp.array(compression, dtype=wp.int32, device=device),
            )
            np.testing.assert_array_equal(expected.fill_level, value)
            np.testing.assert_array_equal(result.numpy(), value)

    def _warp_all_axes_match_reference(self, device: str) -> None:
        shape = (6, 7, 5)
        for axis, speed in enumerate((0.07, -0.06, 0.05)):
            fill = np.zeros(shape, dtype=np.float32)
            lower = [slice(None)] * 3
            lower[axis] = 0
            interior = [slice(None)] * 3
            interior[axis] = slice(1, 3)
            upper = [slice(None)] * 3
            upper[axis] = 3
            fill[tuple(lower)] = 0.7
            fill[tuple(interior)] = 1.0
            fill[tuple(upper)] = 0.3
            normals = np.zeros(shape + (3,), dtype=np.float32)
            lower_normal = np.zeros(3, dtype=np.float32)
            lower_normal[axis] = -1.0
            upper_normal = -lower_normal
            normals[tuple(lower)] = lower_normal
            normals[tuple(upper)] = upper_normal
            offsets = np.zeros(shape, dtype=np.float32)
            offsets[tuple(lower)] = plic_plane_offset(0.7, lower_normal)
            offsets[tuple(upper)] = plic_plane_offset(0.3, upper_normal)
            face_shape = list(shape)
            face_shape[axis] += 1
            face_courant = np.full(face_shape, speed, dtype=np.float32)
            expected = advect_plic_axis(
                fill,
                normals,
                offsets,
                face_courant,
                axis=axis,
                periodic=True,
                bound_tolerance=2.0e-6,
            )
            model = HomeLbmModel(
                fluid_grid_res=shape,
                periodic=(True, True, True),
                device=device,
            )
            advector = HomeFreeGeometricAxisAdvector(model, axis=axis)
            result = advector.advect(
                wp.array(fill, dtype=float, device=device),
                wp.array(normals, dtype=wp.vec3, device=device),
                wp.array(offsets, dtype=float, device=device),
                wp.array(face_courant, dtype=float, device=device),
            )
            np.testing.assert_allclose(
                result.numpy(), expected.fill_level, rtol=0.0, atol=3.0e-7
            )
            np.testing.assert_allclose(
                advector.face_flux.numpy(),
                expected.face_flux,
                rtol=2.0e-5,
                atol=3.0e-7,
            )

    def _warp_periodic_translation(self, device: str) -> None:
        shape = (12, 3, 2)
        speed = 0.02
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.75
        fill[1:5] = 1.0
        fill[5] = 0.25
        face_courant = np.full(
            (shape[0] + 1, shape[1], shape[2]), speed, dtype=np.float32
        )
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            device=device,
        )
        advector = HomeFreeGeometricAxisAdvector(model, axis=0)
        courant_device = wp.array(face_courant, dtype=float, device=device)
        initial_volume = float(np.sum(fill, dtype=np.float64))
        normals, offsets = _planar_periodic_geometry(fill)
        initial_active = _center_side_active(fill, normals, offsets)

        for step in range(1, 21):
            normals, offsets = _planar_periodic_geometry(fill)
            expected = advect_plic_axis(
                fill,
                normals,
                offsets,
                face_courant,
                axis=0,
                periodic=True,
                bound_tolerance=2.0e-6,
            )
            result = advector.advect(
                wp.array(fill, dtype=float, device=device),
                wp.array(normals.astype(np.float32), dtype=wp.vec3, device=device),
                wp.array(offsets.astype(np.float32), dtype=float, device=device),
                courant_device,
            )
            fill = result.numpy()
            np.testing.assert_allclose(fill, expected.fill_level, rtol=0.0, atol=2.5e-7)
            np.testing.assert_allclose(
                advector.face_flux.numpy(), expected.face_flux, rtol=2.0e-5, atol=2.5e-7
            )
            self.assertAlmostEqual(
                float(np.sum(fill, dtype=np.float64)), initial_volume, places=5
            )
            np.testing.assert_allclose(fill[0], 0.75 - step * speed, atol=3.0e-7)
            np.testing.assert_allclose(fill[5], 0.25 + step * speed, atol=3.0e-7)

        normals, offsets = _planar_periodic_geometry(fill)
        final_active = _center_side_active(fill, normals, offsets)
        self.assertEqual(int(np.count_nonzero(~initial_active & final_active)), 6)
        self.assertEqual(int(np.count_nonzero(initial_active & ~final_active)), 6)

    def test_axis_aligned_slab_has_exact_volume(self) -> None:
        normal = np.asarray((1.0, 0.0, 0.0))
        offset = plic_plane_offset(0.3, normal)

        self.assertEqual(
            plic_swept_slab_volume(
                0.3, normal, offset, axis=0, signed_courant=0.4
            ),
            0.0,
        )
        self.assertAlmostEqual(
            plic_swept_slab_volume(
                0.3, normal, offset, axis=0, signed_courant=-0.4
            ),
            0.3,
            places=14,
        )
        self.assertEqual(
            plic_swept_slab_volume(
                1.0, np.zeros(3), 0.0, axis=2, signed_courant=0.27
            ),
            0.27,
        )
        self.assertEqual(
            plic_swept_slab_volume(
                0.0, np.zeros(3), 0.0, axis=1, signed_courant=-0.81
            ),
            0.0,
        )

    def test_complementary_slabs_partition_random_plic_cells(self) -> None:
        rng = np.random.default_rng(72031)
        for _ in range(64):
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            fill = float(rng.uniform(0.02, 0.98))
            offset = plic_plane_offset(fill, normal)
            axis = int(rng.integers(0, 3))
            split = float(rng.uniform(0.05, 0.95))
            lower = plic_swept_slab_volume(
                fill,
                normal,
                offset,
                axis=axis,
                signed_courant=-split,
            )
            upper = plic_swept_slab_volume(
                fill,
                normal,
                offset,
                axis=axis,
                signed_courant=1.0 - split,
            )
            self.assertAlmostEqual(lower + upper, fill, places=12)

    def test_near_full_thin_slab_respects_weymouth_yue_flux_bounds(self) -> None:
        fill = 0.9998764395713806
        normal = np.asarray((2.71871102e-7, -1.0, 6.79677754e-8))
        offset = plic_plane_offset(fill, normal)
        courant = 0.00019622757099568844

        flux = plic_swept_slab_volume(
            fill,
            normal,
            offset,
            axis=1,
            signed_courant=courant,
        )

        self.assertGreaterEqual(flux, max(0.0, courant - (1.0 - fill)))
        self.assertLessEqual(flux, min(courant, fill))
        self.assertAlmostEqual(flux, courant, places=15)

    def test_reference_rejects_gross_flux_bound_correction(self) -> None:
        shape = (3, 3, 3)
        fill = np.zeros(shape, dtype=np.float64)
        normals = np.zeros(shape + (3,), dtype=np.float64)
        offsets = np.zeros(shape, dtype=np.float64)
        courant = np.zeros((shape[0], shape[1], shape[2] + 1), dtype=np.float64)
        donor = (1, 1, 1)
        fill[donor] = 0.9
        normals[donor] = (0.0, 0.0, 1.0)
        offsets[donor] = plic_plane_offset(0.1, normals[donor])
        courant[1, 1, 2] = 0.2

        with self.assertRaisesRegex(RuntimeError, "excessive bound correction"):
            advect_plic_axis(
                fill,
                normals,
                offsets,
                courant,
                axis=2,
                periodic=False,
            )

    def test_periodic_planar_translation_crosses_active_threshold_conservatively(
        self,
    ) -> None:
        shape = (12, 3, 2)
        speed = 0.02
        fill = np.zeros(shape, dtype=np.float64)
        fill[0] = 0.75
        fill[1:5] = 1.0
        fill[5] = 0.25
        face_courant = np.full((shape[0] + 1, shape[1], shape[2]), speed)
        normals, offsets = _planar_periodic_geometry(fill)
        initial_active = _center_side_active(fill, normals, offsets)
        initial_volume = float(np.sum(fill, dtype=np.float64))

        for step in range(1, 21):
            normals, offsets = _planar_periodic_geometry(fill)
            result = advect_plic_axis(
                fill,
                normals,
                offsets,
                face_courant,
                axis=0,
                periodic=True,
            )
            fill = result.fill_level
            np.testing.assert_allclose(fill[0], 0.75 - step * speed, atol=2.0e-14)
            np.testing.assert_allclose(fill[5], 0.25 + step * speed, atol=2.0e-14)
            self.assertAlmostEqual(result.volume_after, initial_volume, places=13)

        normals, offsets = _planar_periodic_geometry(fill)
        final_active = _center_side_active(fill, normals, offsets)
        self.assertEqual(int(np.count_nonzero(~initial_active & final_active)), 6)
        self.assertEqual(int(np.count_nonzero(initial_active & ~final_active)), 6)

    def test_closed_sweep_shares_each_internal_face_flux(self) -> None:
        fill = np.zeros((6, 2, 2), dtype=np.float64)
        fill[1] = 0.4
        fill[2:4] = 1.0
        fill[4] = 0.6
        normals = np.zeros(fill.shape + (3,), dtype=np.float64)
        normals[1] = (-1.0, 0.0, 0.0)
        normals[4] = (1.0, 0.0, 0.0)
        offsets = np.zeros_like(fill)
        offsets[1] = plic_plane_offset(0.4, normals[1, 0, 0])
        offsets[4] = plic_plane_offset(0.6, normals[4, 0, 0])
        face_courant = np.zeros((7, 2, 2), dtype=np.float64)
        face_courant[1:6] = 0.03

        result = advect_plic_axis(
            fill,
            normals,
            offsets,
            face_courant,
            axis=0,
            periodic=False,
        )

        self.assertAlmostEqual(result.volume_before, result.volume_after, places=14)
        self.assertEqual(float(np.max(result.fill_level)), 1.0)
        self.assertEqual(float(np.min(result.fill_level)), 0.0)

    def test_weymouth_yue_correction_preserves_pure_phases_under_stretching(
        self,
    ) -> None:
        shape = (5, 2, 2)
        fill = np.ones(shape, dtype=np.float64)
        normals = np.zeros(shape + (3,), dtype=np.float64)
        offsets = np.zeros(shape, dtype=np.float64)
        courant = np.zeros((shape[0] + 1, shape[1], shape[2]), dtype=np.float64)
        courant[1] = 0.05
        courant[2] = 0.02
        courant[3] = -0.01
        courant[4] = 0.03
        with self.assertRaisesRegex(RuntimeError, "physical fill bounds"):
            advect_plic_axis(
                fill,
                normals,
                offsets,
                courant,
                axis=0,
                periodic=False,
            )
        corrected = advect_plic_axis(
            fill,
            normals,
            offsets,
            courant,
            axis=0,
            periodic=False,
            compression=np.ones(shape, dtype=np.int32),
        )
        np.testing.assert_array_equal(corrected.fill_level, 1.0)

    def test_invalid_courant_and_inconsistent_periodic_faces_fail(self) -> None:
        fill = np.zeros((2, 2, 2), dtype=np.float64)
        normals = np.zeros(fill.shape + (3,), dtype=np.float64)
        offsets = np.zeros_like(fill)
        courant = np.zeros((3, 2, 2), dtype=np.float64)
        courant[0] = 0.1
        with self.assertRaisesRegex(ValueError, "must match"):
            advect_plic_axis(
                fill, normals, offsets, courant, axis=0, periodic=True
            )
        courant.fill(1.01)
        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            advect_plic_axis(
                fill, normals, offsets, courant, axis=0, periodic=True
            )

    def test_warp_cpu_matches_periodic_reference_through_active_transition(self) -> None:
        self._warp_periodic_translation("cpu")

    def test_warp_cpu_matches_reference_on_all_axes_and_flow_directions(self) -> None:
        self._warp_all_axes_match_reference("cpu")

    def test_warp_cpu_weymouth_yue_fixed_phases(self) -> None:
        self._warp_weymouth_yue_fixed_phases("cpu")

    def test_warp_cpu_near_full_flux_respects_support_bound(self) -> None:
        self._warp_near_full_flux_respects_support_bound("cpu")

    def test_warp_cpu_oblique_near_full_flux_respects_intersection_bound(self) -> None:
        self._warp_oblique_near_full_flux_respects_intersection_bound("cpu")

    def test_warp_cpu_rejects_gross_flux_bound_correction(self) -> None:
        self._warp_rejects_gross_flux_bound_correction("cpu")

    def test_warp_cpu_near_axis_sliver_uses_reconstruction_dimension(self) -> None:
        self._warp_near_axis_sliver_uses_reconstruction_dimension("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_matches_periodic_reference_through_active_transition(self) -> None:
        self._warp_periodic_translation("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_matches_reference_on_all_axes_and_flow_directions(self) -> None:
        self._warp_all_axes_match_reference("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_weymouth_yue_fixed_phases(self) -> None:
        self._warp_weymouth_yue_fixed_phases("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_near_full_flux_respects_support_bound(self) -> None:
        self._warp_near_full_flux_respects_support_bound("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_oblique_near_full_flux_respects_intersection_bound(
        self,
    ) -> None:
        self._warp_oblique_near_full_flux_respects_intersection_bound("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_rejects_gross_flux_bound_correction(self) -> None:
        self._warp_rejects_gross_flux_bound_correction("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_near_axis_sliver_uses_reconstruction_dimension(self) -> None:
        self._warp_near_axis_sliver_uses_reconstruction_dimension("cuda:0")


if __name__ == "__main__":
    unittest.main()
