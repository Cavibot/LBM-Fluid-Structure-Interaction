# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    GeometricHomeFslStepper,
)


def _static_case(device: str) -> tuple[np.ndarray, np.ndarray, object]:
    shape = (18, 12, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=0.02,
    )
    walls = FslWallMask.periodic_depth_channel(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    moments = np.zeros((10, fluid_a.cell_count), dtype=np.float32)
    moments[0] = 1.0
    fluid_a.moments.assign(moments.reshape(-1))
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fill = np.zeros(shape, dtype=np.float32)
    fill[1:-1, 1:5, :] = 1.0
    fill[1:-1, 5, :] = 0.5
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    fsl_a.fill_level.assign(fill)
    fsl_a.mass.assign(fill)
    fsl_a.flags.assign(flags)
    stepper = GeometricHomeFslStepper(model, walls)
    initial_volume = float(np.sum(fill, dtype=np.float64))
    for _ in range(5):
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    self_volume = float(np.sum(fsl_a.fill_level.numpy(), dtype=np.float64))
    if abs(self_volume - initial_volume) > 2.0e-5:
        raise AssertionError("static geometric stepper changed liquid volume")
    return fsl_a.fill_level.numpy(), fluid_a.moments.numpy(), diagnostics


def _translation_case(
    device: str, *, steps: int = 1
) -> tuple[np.ndarray, np.ndarray, object]:
    shape = (24, 8, 1)
    closed_axes = (False, True, False)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=0.02,
    )
    walls = FslWallMask(
        model,
        axis_aligned_wall_mask(shape, closed_axes=closed_axes),
        closed_axes=closed_axes,
    )
    fluid_in = HomeCoreState(model)
    fluid_out = HomeCoreState(model)
    fsl_in = FslState(model)
    fsl_out = FslState(model)
    fill = np.zeros(shape, dtype=np.float32)
    fill[4:11, 1:7, :] = 0.5
    fill[5:10, 2:6, :] = 1.0
    fill[10, 2:6, :] = 0.98
    fill[walls.host] = 0.0
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    moments = np.zeros((10,) + shape, dtype=np.float32)
    moments[0] = 1.0
    active = flags != int(FslCellFlag.GAS)
    moments[1][active] = 0.05
    moments[4][active] = 0.05**2
    fluid_in.moments.assign(moments.reshape(10, -1).reshape(-1))
    fsl_in.fill_level.assign(fill)
    fsl_in.mass.assign(fill)
    fsl_in.flags.assign(flags)
    stepper = GeometricHomeFslStepper(model, walls)
    for _ in range(steps):
        diagnostics = stepper.step(
            fluid_in, fsl_in, fluid_out, fsl_out, model.time_step
        )
        fluid_in, fluid_out = fluid_out, fluid_in
        fsl_in, fsl_out = fsl_out, fsl_in
    return fsl_in.fill_level.numpy(), fsl_in.mass.numpy(), diagnostics


class TestGeometricHomeFslStepper(unittest.TestCase):
    def test_translation_remains_conservative_for_eight_steps(self) -> None:
        fill, mass, diagnostics = _translation_case("cpu", steps=8)
        self.assertTrue(np.isfinite(fill).all())
        self.assertTrue(np.isfinite(mass).all())
        self.assertAlmostEqual(float(np.sum(fill)), 32.92, delta=3.0e-4)
        self.assertAlmostEqual(float(np.sum(mass)), 32.92, delta=3.0e-4)
        self.assertEqual(diagnostics.topology.direct_liquid_gas_link_count, 0)

    def test_translation_pipeline_creates_a_fresh_leading_interface(self) -> None:
        fill, mass, diagnostics = _translation_case("cpu")
        self.assertGreater(diagnostics.topology.fresh_interface_count, 0)
        self.assertLess(abs(diagnostics.transport.volume_drift), 2.0e-5)
        self.assertLess(abs(diagnostics.transport.mass_drift), 2.0e-5)
        self.assertGreater(float(np.sum(fill[11:])), 0.0)
        self.assertAlmostEqual(float(np.sum(fill)), float(np.sum(mass)), places=5)

    def test_translation_cuda_matches_cpu(self) -> None:
        cpu = _translation_case("cpu")
        cuda = _translation_case("cuda:0")
        np.testing.assert_allclose(cuda[0], cpu[0], rtol=4.0e-5, atol=4.0e-6)
        np.testing.assert_allclose(cuda[1], cpu[1], rtol=4.0e-5, atol=4.0e-6)

    def test_cpu_static_layer_remains_at_rest(self) -> None:
        fill, moments, diagnostics = _static_case("cpu")
        self.assertEqual(diagnostics.projection.iteration_count, 0)
        self.assertEqual(diagnostics.topology.fresh_interface_count, 0)
        self.assertLess(diagnostics.fluid.max_speed, 2.0e-6)
        self.assertLess(diagnostics.collided_fluid.max_speed, 2.0e-6)
        self.assertGreater(float(np.sum(fill)), 0.0)
        self.assertTrue(np.isfinite(moments).all())

    def test_cuda_matches_cpu_static_layer(self) -> None:
        cpu = _static_case("cpu")
        cuda = _static_case("cuda:0")
        np.testing.assert_allclose(cuda[0], cpu[0], rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(cuda[1], cpu[1], rtol=3.0e-5, atol=3.0e-6)


if __name__ == "__main__":
    unittest.main()
