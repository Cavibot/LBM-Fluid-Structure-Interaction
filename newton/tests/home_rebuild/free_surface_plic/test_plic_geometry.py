# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    plane_interface_area,
    plane_volume_fraction,
    plic_plane_offset,
    reconstruct_plic_geometry,
    youngs_interface_normal,
)


def _planar_stencil(normal: np.ndarray, offset: float = 0.0) -> np.ndarray:
    fill = np.empty((3, 3, 3), dtype=np.float64)
    for index in np.ndindex(fill.shape):
        center = np.asarray(index, dtype=np.float64) - 1.0
        fill[index] = plane_volume_fraction(offset - float(normal @ center), normal)
    return fill


class TestPlicGeometryReference(unittest.TestCase):
    def test_axis_aligned_cut_and_complement_are_exact(self) -> None:
        normal = np.asarray((2.0, 0.0, 0.0))
        self.assertAlmostEqual(plane_volume_fraction(-0.5, normal), 0.25, places=14)
        self.assertAlmostEqual(plic_plane_offset(0.25, normal), -0.5, places=14)
        self.assertAlmostEqual(plic_plane_offset(0.75, normal), 0.5, places=14)
        self.assertAlmostEqual(plane_interface_area(0.0, normal), 1.0, places=14)

    def test_random_offsets_invert_exact_cube_volume(self) -> None:
        rng = np.random.default_rng(20260722)
        for _ in range(100):
            normal = rng.normal(size=3)
            fill = float(rng.uniform(1.0e-6, 1.0 - 1.0e-6))
            offset = plic_plane_offset(fill, normal)
            self.assertAlmostEqual(
                plane_volume_fraction(offset, normal), fill, delta=3.0e-13
            )

    def test_interface_area_is_volume_derivative(self) -> None:
        normal = np.asarray((0.3, -0.8, 0.5))
        offset = plic_plane_offset(0.37, normal)
        epsilon = 3.0e-6
        derivative = (
            plane_volume_fraction(offset + epsilon, normal)
            - plane_volume_fraction(offset - epsilon, normal)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            plane_interface_area(offset, normal),
            float(np.linalg.norm(normal)) * derivative,
            delta=1.0e-8,
        )

    def test_youngs_normal_recovers_planar_orientation(self) -> None:
        expected = np.asarray((0.7, -0.3, 0.2), dtype=np.float64)
        expected /= np.linalg.norm(expected)
        actual = youngs_interface_normal(_planar_stencil(expected))
        self.assertGreater(float(actual @ expected), 0.999)

    def test_periodic_depth_reconstruction_is_valid_and_inverts_fill(self) -> None:
        shape = (18, 14, 1)
        solid = axis_aligned_wall_mask(shape, closed_axes=(True, True, False))
        normal = np.asarray((1.0, 0.35, 0.0), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        fill = np.zeros(shape, dtype=np.float64)
        for index in np.ndindex(shape):
            center = np.asarray(index, dtype=np.float64)
            fill[index] = plane_volume_fraction(8.0 - float(normal @ center), normal)
        fill[solid] = 0.0
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[fill == 1.0] = int(FslCellFlag.LIQUID)
        flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
        fields = reconstruct_plic_geometry(
            fill, flags, solid, closed_axes=(True, True, False)
        )

        interface = flags == int(FslCellFlag.INTERFACE)
        self.assertTrue(np.all(fields.valid[interface]))
        self.assertTrue(np.all(fields.normal[interface, 2] == 0.0))
        for index in zip(*np.nonzero(interface)):
            self.assertAlmostEqual(
                plane_volume_fraction(fields.plane_offset[index], fields.normal[index]),
                fill[index],
                delta=3.0e-13,
            )


if __name__ == "__main__":
    unittest.main()
