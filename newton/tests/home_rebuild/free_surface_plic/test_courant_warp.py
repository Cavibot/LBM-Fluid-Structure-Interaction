# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    HomeCourantProjector,
    HomeFaceCourantBuilder,
    active_divergence,
    build_face_courant_reference,
)


def _courant_case(device: str) -> tuple[tuple[np.ndarray, ...], object, object]:
    shape = (12, 10, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    walls = FslWallMask.periodic_depth_channel(model)
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[1:9, 1:7, :] = int(FslCellFlag.LIQUID)
    flags[9, 1:7, :] = int(FslCellFlag.INTERFACE)
    fill = np.zeros(shape, dtype=np.float32)
    fill[flags == int(FslCellFlag.LIQUID)] = 1.0
    fill[flags == int(FslCellFlag.INTERFACE)] = 0.5
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[..., 0] = 1.0
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(shape[1], dtype=np.float32)[None, :, None]
    moments[..., 1] = 0.002 * x + 0.0003 * y
    moments[..., 2] = -0.001 * y + 0.0002 * x
    moments[walls.host, 1:4] = 0.0
    fluid = HomeCoreState(model)
    fluid.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1))
    )
    fsl = FslState(model)
    fsl.fill_level.assign(fill)
    fsl.mass.assign(fill)
    fsl.flags.assign(flags)
    builder = HomeFaceCourantBuilder(model, walls)
    source = builder.build(fluid, fsl)
    source_numpy = tuple(field.numpy() for field in source)
    reference = build_face_courant_reference(
        moments, flags, walls.host, closed_axes=walls.closed_axes
    )
    for actual, expected in zip(source_numpy, reference.face_courant):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)
    projector = HomeCourantProjector(
        model,
        walls,
        maximum_iterations=160,
        relative_tolerance=2.0e-5,
        absolute_divergence_tolerance=2.0e-7,
    )
    projected = projector.project(source, fsl)
    assert projector.last_diagnostics is not None
    return tuple(field.numpy() for field in projected), builder.last_diagnostics, projector.last_diagnostics


def _zero_courant_case(device: str) -> object:
    shape = (8, 7, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    walls = FslWallMask.periodic_depth_channel(model)
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[1:-1, 1:-1, :] = int(FslCellFlag.LIQUID)
    fill = (flags == int(FslCellFlag.LIQUID)).astype(np.float32)
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[..., 0] = 1.0
    fluid = HomeCoreState(model)
    fluid.moments.assign(np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1)))
    fsl = FslState(model)
    fsl.fill_level.assign(fill)
    fsl.mass.assign(fill)
    fsl.flags.assign(flags)
    faces = HomeFaceCourantBuilder(model, walls).build(fluid, fsl)
    projector = HomeCourantProjector(model, walls)
    projected = projector.project(faces, fsl)
    for field in projected:
        np.testing.assert_array_equal(field.numpy(), np.zeros(field.shape, dtype=np.float32))
    return projector.last_diagnostics


class TestCourantWarp(unittest.TestCase):
    def test_zero_divergence_uses_zero_iteration_fast_path(self) -> None:
        for device in ("cpu", "cuda:0"):
            diagnostics = _zero_courant_case(device)
            self.assertEqual(diagnostics.iteration_count, 0)
            self.assertEqual(diagnostics.projected_maximum_divergence, 0.0)

    def test_cpu_builder_and_projector_satisfy_active_divergence(self) -> None:
        projected, builder, projector = _courant_case("cpu")
        shape = (12, 10, 1)
        active = np.zeros(shape, dtype=bool)
        active[1:10, 1:7, :] = True
        divergence = active_divergence(projected, active)
        self.assertGreater(builder.maximum_active_divergence, 1.0e-4)
        self.assertLess(float(np.max(np.abs(divergence[active]))), 2.0e-7)
        self.assertLess(projector.relative_residual, 2.0e-5)

    def test_cuda_matches_cpu_projected_faces(self) -> None:
        cpu = _courant_case("cpu")
        cuda = _courant_case("cuda:0")
        for actual, expected in zip(cuda[0], cpu[0]):
            np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-6)
        self.assertLess(cuda[2].projected_maximum_divergence, 2.0e-7)


if __name__ == "__main__":
    unittest.main()
