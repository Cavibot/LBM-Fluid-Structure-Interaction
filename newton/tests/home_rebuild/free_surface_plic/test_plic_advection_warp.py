# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicAxisAdvector,
    PlicGeometryReconstructor,
    PlicGeometryState,
    advect_plic_axis,
    reconstruct_plic_geometry,
)


def _run_axis_sweep(device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (16, 3, 3)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    closed_axes = (False, True, True)
    solid = axis_aligned_wall_mask(shape, closed_axes=closed_axes)
    walls = FslWallMask(model, solid, closed_axes=closed_axes)
    fill = np.zeros(shape, dtype=np.float32)
    fill[3, 1, 1] = 0.7
    fill[4:8, 1, 1] = 1.0
    fill[8, 1, 1] = 0.3
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    fsl = FslState(model)
    fsl.fill_level.assign(fill)
    fsl.mass.assign(fill)
    fsl.flags.assign(flags)
    geometry = PlicGeometryState(model)
    PlicGeometryReconstructor(model, walls).reconstruct(fsl, geometry)
    courant = np.zeros((17, 3, 3), dtype=np.float32)
    courant[:, 1, 1] = 0.2
    advector = PlicAxisAdvector(model, walls, axis=0)
    updated, diagnostics = advector.advect(
        fsl.fill_level,
        geometry,
        wp.array(courant, dtype=float, device=device),
    )
    reference_geometry = reconstruct_plic_geometry(
        fill, flags, solid, closed_axes=closed_axes
    )
    reference = advect_plic_axis(
        fill,
        reference_geometry.normal,
        reference_geometry.plane_offset,
        solid,
        courant,
        axis=0,
        periodic=True,
    )
    if abs(diagnostics.relative_volume_drift) > 2.0e-6:
        raise AssertionError("Warp PLIC sweep did not conserve volume")
    return updated.numpy(), advector.face_flux.numpy(), reference.fill_level


class TestPlicAdvectionWarp(unittest.TestCase):
    def test_cpu_matches_numpy_shared_face_oracle(self) -> None:
        updated, face_flux, reference = _run_axis_sweep("cpu")
        np.testing.assert_allclose(updated, reference, rtol=2.0e-6, atol=2.0e-6)
        self.assertEqual(int(np.count_nonzero(face_flux[:, 0, :])), 0)
        self.assertEqual(int(np.count_nonzero(face_flux[:, -1, :])), 0)

    def test_cuda_matches_cpu(self) -> None:
        cpu = _run_axis_sweep("cpu")
        cuda = _run_axis_sweep("cuda:0")
        np.testing.assert_allclose(cuda[0], cpu[0], rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(cuda[1], cpu[1], rtol=2.0e-6, atol=2.0e-6)

    def test_near_full_fallback_flux_is_bounded_and_conservative(self) -> None:
        for device in ("cpu", "cuda:0"):
            shape = (8, 5, 1)
            model = HomeCoreModel(fluid_grid_res=shape, device=device)
            closed_axes = (False, True, False)
            walls = FslWallMask(
                model,
                axis_aligned_wall_mask(shape, closed_axes=closed_axes),
                closed_axes=closed_axes,
            )
            fill = np.ones(shape, dtype=np.float32)
            fill[walls.host] = 0.0
            fill[3, 2, 0] = np.float32(1.0 - 5.3e-6)
            fill_device = wp.array(fill, dtype=float, device=device)
            geometry = PlicGeometryState(model)
            geometry_diagnostics = PlicGeometryReconstructor(
                model, walls
            ).reconstruct_fill(
                fill_device,
                geometry,
                interface_epsilon=5.0e-6,
                endpoint_fallback_tolerance=2.0e-5,
            )
            courant = np.zeros((shape[0] + 1, shape[1], shape[2]), dtype=np.float32)
            courant[:, 2, 0] = 0.05
            updated, diagnostics = PlicAxisAdvector(
                model, walls, axis=0
            ).advect(
                fill_device,
                geometry,
                wp.array(courant, dtype=float, device=device),
                interface_epsilon=5.0e-6,
                endpoint_fallback_tolerance=2.0e-5,
            )

            self.assertEqual(
                geometry_diagnostics.endpoint_fallback_cell_count, 1
            )
            self.assertEqual(diagnostics.endpoint_fallback_face_count, 1)
            self.assertAlmostEqual(
                diagnostics.volume_before, diagnostics.volume_after, places=6
            )
            np.testing.assert_allclose(
                updated.numpy(), fill, rtol=0.0, atol=2.0e-7
            )


if __name__ == "__main__":
    unittest.main()
