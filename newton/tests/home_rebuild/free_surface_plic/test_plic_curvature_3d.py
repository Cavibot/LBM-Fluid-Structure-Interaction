# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicCurvatureEstimator3D,
    PlicCurvatureState,
    PlicDomain,
    PlicGeometryReconstructor,
    PlicGeometryState,
    plane_volume_fraction,
)


def _flags(fill: np.ndarray) -> np.ndarray:
    flags = np.zeros(fill.shape, dtype=np.int32)
    flags[fill >= 1.0] = 2
    flags[(fill > 0.0) & (fill < 1.0)] = 1
    return flags


def _sphere_fill(shape: tuple[int, int, int], radius: float) -> np.ndarray:
    center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    coordinates = np.indices(shape, dtype=np.float64)
    distance = np.sqrt(
        sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
    )
    return np.clip(0.5 + radius - distance, 0.0, 1.0).astype(np.float32)


def _reconstruct(
    domain: PlicDomain, fill: np.ndarray, *, strict_bulk: bool = True
):
    fill_device = wp.array(fill, dtype=float, device=domain._device)
    flags_device = wp.array(_flags(fill), dtype=wp.int32, device=domain._device)
    geometry = PlicGeometryState(domain)
    PlicGeometryReconstructor(domain).reconstruct_fill(fill_device, geometry)
    curvature = PlicCurvatureState(domain)
    diagnostics = PlicCurvatureEstimator3D(
        domain, strict_bulk=strict_bulk
    ).reconstruct(fill_device, flags_device, geometry, curvature)
    return diagnostics, curvature


class TestPlicCurvature3D(unittest.TestCase):
    def test_sphere_has_finite_consistent_mean_curvature(self) -> None:
        radius = 6.0
        domain = PlicDomain.closed_box((21, 21, 21), device="cpu")
        fill = _sphere_fill(domain.res, radius)

        diagnostics, state = _reconstruct(domain, fill)

        values = state.curvature.numpy()[state.valid.numpy() != 0]
        self.assertGreater(values.size, 100)
        self.assertEqual(diagnostics.wall_contact_skipped_count, 0)
        self.assertEqual(diagnostics.insufficient_neighbor_count, 0)
        self.assertEqual(diagnostics.ill_conditioned_count, 0)
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(float(np.median(values)), 0.0)
        self.assertAlmostEqual(float(np.median(values)), -1.0 / radius, delta=0.08)

    def test_planar_interface_has_near_zero_bulk_curvature(self) -> None:
        domain = PlicDomain.closed_box((19, 17, 15), device="cpu")
        normal = np.asarray((1.0, 0.25, 0.15), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        plane_offset = 10.0 / np.linalg.norm((1.0, 0.25, 0.15))
        fill = np.empty(domain.res, dtype=np.float32)
        for index in np.ndindex(domain.res):
            fill[index] = plane_volume_fraction(
                plane_offset - float(normal @ np.asarray(index)), normal
            )
        fill[domain.host] = 0.0
        flags = _flags(fill)
        fill_device = wp.array(fill, dtype=float, device=domain._device)
        flags_device = wp.array(flags, dtype=wp.int32, device=domain._device)
        geometry = PlicGeometryState(domain)
        normals = np.broadcast_to(normal, domain.res + (3,)).astype(np.float32)
        offsets = np.empty(domain.res, dtype=np.float32)
        for index in np.ndindex(domain.res):
            offsets[index] = plane_offset - float(normal @ np.asarray(index))
        geometry.normal.assign(normals)
        geometry.plane_offset.assign(offsets)
        geometry.valid.assign((flags == 1).astype(np.int32))
        state = PlicCurvatureState(domain)
        diagnostics = PlicCurvatureEstimator3D(domain).reconstruct(
            fill_device, flags_device, geometry, state
        )

        values = state.curvature.numpy()[state.valid.numpy() != 0]
        self.assertGreater(values.size, 20)
        self.assertGreater(diagnostics.wall_contact_skipped_count, 0)
        self.assertLess(float(np.max(np.abs(values))), 2.0e-4)

    def test_wall_contact_is_reported_not_fitted(self) -> None:
        domain = PlicDomain.closed_box((11, 11, 11), device="cpu")
        fill = np.zeros(domain.res, dtype=np.float32)
        fill[1:6, 1:6, 1:6] = 1.0
        fill[5, 1:6, 1:6] = 0.5
        fill[domain.host] = 0.0

        diagnostics, state = _reconstruct(domain, fill, strict_bulk=False)

        self.assertGreater(diagnostics.wall_contact_skipped_count, 0)
        skipped = (state.required.numpy() != 0) & (state.valid.numpy() == 0)
        self.assertGreater(int(np.count_nonzero(skipped)), 0)


if __name__ == "__main__":
    unittest.main()
