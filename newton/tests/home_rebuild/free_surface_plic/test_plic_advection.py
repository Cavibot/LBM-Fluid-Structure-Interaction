# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    advect_plic_axis,
    plane_volume_fraction,
    plic_plane_offset,
    plic_swept_slab_volume,
    reconstruct_plic_geometry,
)


def _periodic_block() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (16, 3, 3)
    fill = np.zeros(shape, dtype=np.float64)
    fill[3] = 0.7
    fill[4:8] = 1.0
    fill[8] = 0.3
    flags = np.zeros(shape, dtype=np.int32)
    flags[fill == 1.0] = 2
    flags[(fill > 0.0) & (fill < 1.0)] = 1
    solid = np.zeros(shape, dtype=bool)
    geometry = reconstruct_plic_geometry(
        fill, flags, solid, closed_axes=(False, True, True)
    )
    return fill, geometry.normal, geometry.plane_offset, solid


class TestPlicAdvectionReference(unittest.TestCase):
    def test_axis_aligned_slab_volume_matches_exact_overlap(self) -> None:
        normal = np.asarray((1.0, 0.0, 0.0))
        offset = plic_plane_offset(0.7, normal)
        self.assertAlmostEqual(
            plic_swept_slab_volume(
                0.7, normal, offset, axis=0, signed_courant=0.4
            ),
            0.1,
            places=14,
        )
        self.assertAlmostEqual(
            plic_swept_slab_volume(
                0.7, normal, offset, axis=0, signed_courant=-0.4
            ),
            0.4,
            places=14,
        )

    def test_periodic_block_translates_one_cell_in_five_sweeps(self) -> None:
        fill, _, _, solid = _periodic_block()
        initial = fill.copy()
        for _ in range(5):
            flags = np.zeros(fill.shape, dtype=np.int32)
            flags[fill == 1.0] = 2
            flags[(fill > 0.0) & (fill < 1.0)] = 1
            geometry = reconstruct_plic_geometry(
                fill, flags, solid, closed_axes=(False, True, True)
            )
            courant = np.full((17, 3, 3), 0.2, dtype=np.float64)
            result = advect_plic_axis(
                fill,
                geometry.normal,
                geometry.plane_offset,
                solid,
                courant,
                axis=0,
                periodic=True,
            )
            fill = result.fill_level
            self.assertAlmostEqual(result.volume_before, result.volume_after, places=13)
        np.testing.assert_allclose(fill, np.roll(initial, 1, axis=0), atol=2.0e-13)

    def test_oblique_thin_layer_sweep_conserves_volume_and_bounds(self) -> None:
        shape = (24, 20, 1)
        normal = np.asarray((0.8, 0.6, 0.0))
        fill = np.zeros(shape, dtype=np.float64)
        for index in np.ndindex(shape):
            center = np.asarray(index, dtype=np.float64)
            outer = plane_volume_fraction(12.0 - float(normal @ center), normal)
            inner = plane_volume_fraction(10.5 - float(normal @ center), normal)
            fill[index] = outer - inner
        flags = np.zeros(shape, dtype=np.int32)
        flags[fill == 1.0] = 2
        flags[(fill > 0.0) & (fill < 1.0)] = 1
        solid = np.zeros(shape, dtype=bool)
        geometry = reconstruct_plic_geometry(
            fill, flags, solid, closed_axes=(False, False, False)
        )
        courant = np.full((25, 20, 1), 0.17, dtype=np.float64)
        result = advect_plic_axis(
            fill,
            geometry.normal,
            geometry.plane_offset,
            solid,
            courant,
            axis=0,
            periodic=True,
        )
        self.assertAlmostEqual(result.volume_before, result.volume_after, places=12)
        self.assertGreaterEqual(float(np.min(result.fill_level)), 0.0)
        self.assertLessEqual(float(np.max(result.fill_level)), 1.0)


if __name__ == "__main__":
    unittest.main()
