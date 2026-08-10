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
    GvofFslWallStepper,
)


def _run_static_layer(device: str) -> tuple[np.ndarray, np.ndarray, object]:
    shape = (18, 18, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=0.02,
    )
    walls = FslWallMask.periodic_depth_channel(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fill = np.zeros(shape, dtype=np.float32)
    fill[1:-1, 1:7, :] = 1.0
    fill[1:-1, 7, :] = 0.5
    moments = np.zeros((10, fluid_a.cell_count), dtype=np.float32)
    moments[0] = 1.0
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    fluid_a.moments.assign(moments.reshape(-1))
    fsl_a.fill_level.assign(fill)
    fsl_a.mass.assign(fill)
    fsl_a.flags.assign(flags)
    stepper = GvofFslWallStepper(model, walls)
    self_test = (
        hasattr(stepper, "projector"),
        hasattr(stepper.topology, "transported_momentum"),
    )
    if any(self_test):
        raise AssertionError("GVOF-only composition contains a disabled subsystem")
    for _ in range(8):
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    return fsl_a.fill_level.numpy(), fsl_a.mass.numpy(), diagnostics


class TestGvofFslWallStepper(unittest.TestCase):
    def test_static_layer_is_conservative_without_projection_or_momentum(self) -> None:
        fill, mass, diagnostics = _run_static_layer("cpu")
        self.assertTrue(np.isfinite(fill).all())
        self.assertTrue(np.isfinite(mass).all())
        self.assertAlmostEqual(float(np.sum(fill)), 104.0, delta=3.0e-4)
        self.assertAlmostEqual(float(np.sum(mass)), 104.0, delta=3.0e-4)
        self.assertLess(abs(diagnostics.transport.volume_drift), 2.0e-5)
        self.assertLess(abs(diagnostics.transport.mass_drift), 2.0e-5)
        self.assertEqual(diagnostics.topology.direct_liquid_gas_link_count, 0)
        self.assertLess(diagnostics.fluid.max_speed, 2.0e-6)

    def test_cuda_matches_cpu_for_static_layer(self) -> None:
        cpu = _run_static_layer("cpu")
        cuda = _run_static_layer("cuda:0")
        np.testing.assert_allclose(cuda[0], cpu[0], rtol=3.0e-6, atol=3.0e-7)
        np.testing.assert_allclose(cuda[1], cpu[1], rtol=3.0e-6, atol=3.0e-7)


if __name__ == "__main__":
    unittest.main()
