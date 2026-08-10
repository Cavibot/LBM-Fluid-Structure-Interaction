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
    ProjectedGvofFslWallStepper,
)


class TestProjectedGvofFslWallStepper(unittest.TestCase):
    def test_static_layer_uses_projection_but_not_geometric_topology(self) -> None:
        shape = (18, 18, 1)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask.periodic_depth_channel(model)
        fluid_a = HomeCoreState(model)
        fluid_b = HomeCoreState(model)
        fsl_a = FslState(model)
        fsl_b = FslState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[1:-1, 1:7, :] = 1.0
        fill[1:-1, 7, :] = 0.5
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[fill == 1.0] = int(FslCellFlag.LIQUID)
        flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
        moments = np.zeros((10, fluid_a.cell_count), dtype=np.float32)
        moments[0] = 1.0
        fluid_a.moments.assign(moments.reshape(-1))
        fsl_a.fill_level.assign(fill)
        fsl_a.mass.assign(fill)
        fsl_a.flags.assign(flags)
        stepper = ProjectedGvofFslWallStepper(model, walls)

        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )

        self.assertEqual(type(stepper.topology).__name__, "FslWallTopologyUpdater")
        self.assertFalse(hasattr(stepper, "initial_momentum"))
        self.assertEqual(diagnostics.projection.iteration_count, 0)
        self.assertLess(abs(diagnostics.transport.volume_drift), 2.0e-5)
        self.assertLess(abs(diagnostics.transport.mass_drift), 2.0e-5)
        self.assertAlmostEqual(float(np.sum(fsl_b.mass.numpy())), 104.0, places=4)


if __name__ == "__main__":
    unittest.main()
