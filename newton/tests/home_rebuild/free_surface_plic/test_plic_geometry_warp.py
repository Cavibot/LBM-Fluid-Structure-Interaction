# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicGeometryReconstructor,
    PlicGeometryState,
    plane_volume_fraction,
    reconstruct_plic_geometry,
)


def _geometry_case(device: str) -> tuple[np.ndarray, ...]:
    shape = (18, 14, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    walls = FslWallMask.periodic_depth_channel(model)
    normal = np.asarray((1.0, 0.35, 0.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    fill = np.zeros(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        center = np.asarray(index, dtype=np.float64)
        fill[index] = plane_volume_fraction(8.0 - float(normal @ center), normal)
    fill[walls.host] = 0.0
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    fluid = HomeCoreState(model)
    fsl = FslState(model)
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[..., 0] = 1.0
    fluid.moments.assign(np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1)))
    fsl.fill_level.assign(fill)
    fsl.mass.assign(fill)
    fsl.flags.assign(flags)
    geometry = PlicGeometryState(model)
    diagnostics = PlicGeometryReconstructor(model, walls).reconstruct(fsl, geometry)
    reference = reconstruct_plic_geometry(
        fill, flags, walls.host, closed_axes=walls.closed_axes
    )
    self_count = int(np.count_nonzero(flags == int(FslCellFlag.INTERFACE)))
    if diagnostics.interface_cell_count != self_count:
        raise AssertionError("Warp reconstruction did not cover every interface")
    return (
        geometry.normal.numpy(),
        geometry.plane_offset.numpy(),
        geometry.interface_area.numpy(),
        reference.normal,
        reference.plane_offset,
        reference.interface_area,
        reference.valid,
    )


class TestPlicGeometryWarp(unittest.TestCase):
    def test_cpu_reconstructs_periodic_depth_plane(self) -> None:
        normal, offset, area, reference_normal, reference_offset, reference_area, valid = (
            _geometry_case("cpu")
        )
        self.assertTrue(np.isfinite(normal[valid]).all())
        self.assertTrue(np.isfinite(offset[valid]).all())
        self.assertTrue(np.all(area[valid] > 0.0))
        np.testing.assert_allclose(normal[valid], reference_normal[valid], atol=2.0e-6)
        np.testing.assert_allclose(offset[valid], reference_offset[valid], atol=2.0e-6)
        np.testing.assert_allclose(area[valid], reference_area[valid], atol=3.0e-6)

    def test_cuda_matches_cpu(self) -> None:
        cpu = _geometry_case("cpu")
        cuda = _geometry_case("cuda:0")
        valid = cpu[6]
        np.testing.assert_allclose(cuda[0][valid], cpu[0][valid], rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(cuda[1][valid], cpu[1][valid], rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(cuda[2][valid], cpu[2][valid], rtol=3.0e-6, atol=3.0e-6)

    def test_near_full_uniform_cell_uses_audited_endpoint_fallback(self) -> None:
        for device in ("cpu", "cuda:0"):
            shape = (8, 5, 1)
            model = HomeCoreModel(fluid_grid_res=shape, device=device)
            walls = FslWallMask.periodic_depth_channel(model)
            fill = np.ones(shape, dtype=np.float32)
            fill[walls.host] = 0.0
            fill[3, 2, 0] = np.float32(1.0 - 5.3e-6)
            geometry = PlicGeometryState(model)
            diagnostics = PlicGeometryReconstructor(
                model, walls
            ).reconstruct_fill(
                wp.array(fill, dtype=float, device=device),
                geometry,
                interface_epsilon=5.0e-6,
                endpoint_fallback_tolerance=2.0e-5,
            )

            self.assertEqual(diagnostics.invalid_interface_count, 0)
            self.assertEqual(diagnostics.endpoint_fallback_cell_count, 1)
            self.assertEqual(geometry.valid.numpy()[3, 2, 0], 0)

    def test_mid_fill_uniform_cell_remains_a_hard_failure(self) -> None:
        shape = (8, 5, 1)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask.periodic_depth_channel(model)
        fill = np.ones(shape, dtype=np.float32)
        fill[walls.host] = 0.0
        fill[3, 2, 0] = 0.5
        with self.assertRaisesRegex(
            FloatingPointError, "undefined interface normals"
        ):
            PlicGeometryReconstructor(model, walls).reconstruct_fill(
                wp.array(fill, dtype=float, device="cpu"),
                PlicGeometryState(model),
                interface_epsilon=5.0e-6,
                endpoint_fallback_tolerance=2.0e-5,
            )

    def test_planar_wall_contact_angles_match_configured_normals(self) -> None:
        shape = (9, 9, 1)
        for device in ("cpu", "cuda:0"):
            for angle in (60.0, 90.0, 120.0):
                model = HomeCoreModel(fluid_grid_res=shape, device=device)
                walls = FslWallMask.periodic_depth_channel(model)
                fill = np.zeros(shape, dtype=np.float32)
                fill[1:-1, 1:4, 0] = 1.0
                fill[1:-1, 4, 0] = 0.5
                geometry = PlicGeometryState(model)
                diagnostics = PlicGeometryReconstructor(
                    model,
                    walls,
                    contact_angle_degrees=angle,
                ).reconstruct_fill(
                    wp.array(fill, dtype=float, device=device),
                    geometry,
                )

                theta = np.deg2rad(angle)
                expected = np.asarray(
                    (np.cos(theta), np.sin(theta), 0.0),
                    dtype=np.float32,
                )
                np.testing.assert_allclose(
                    geometry.normal.numpy()[1, 4, 0],
                    expected,
                    rtol=2.0e-6,
                    atol=2.0e-6,
                )
                self.assertEqual(diagnostics.wetting_cell_count, 2)
                self.assertEqual(diagnostics.invalid_wall_normal_count, 0)
                self.assertEqual(
                    diagnostics.parallel_wall_interface_cell_count, 0
                )

    def test_contact_angle_configuration_is_strict(self) -> None:
        model = HomeCoreModel(fluid_grid_res=(5, 5, 1), device="cpu")
        walls = FslWallMask.periodic_depth_channel(model)
        for value in (0.0, 180.0, -1.0, 181.0, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "contact_angle_degrees"
            ):
                PlicGeometryReconstructor(
                    model,
                    walls,
                    contact_angle_degrees=value,
                )

    def test_parallel_wall_film_is_preserved_and_audited(self) -> None:
        shape = (9, 9, 1)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask.periodic_depth_channel(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[1, 1:-1, 0] = 0.5
        geometry = PlicGeometryState(model)
        diagnostics = PlicGeometryReconstructor(
            model,
            walls,
            contact_angle_degrees=120.0,
        ).reconstruct_fill(
            wp.array(fill, dtype=float, device="cpu"),
            geometry,
        )

        self.assertGreater(
            diagnostics.parallel_wall_interface_cell_count, 0
        )
        self.assertEqual(geometry.valid.numpy()[1, 4, 0], 1)
        np.testing.assert_allclose(
            geometry.normal.numpy()[1, 4, 0],
            (1.0, 0.0, 0.0),
            rtol=0.0,
            atol=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
