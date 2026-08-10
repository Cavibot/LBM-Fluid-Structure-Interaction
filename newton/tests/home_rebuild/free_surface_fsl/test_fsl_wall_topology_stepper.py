# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
    FslWallTopologyUpdater,
    closed_box_wall_mask,
)


def _rest_state(model: HomeCoreModel) -> HomeCoreState:
    state = HomeCoreState(model)
    moments = np.zeros((10, state.cell_count), dtype=np.float32)
    moments[0] = 1.0
    state.moments.assign(moments.reshape(-1))
    return state


class TestFslWallTopologyStepper(unittest.TestCase):
    def test_wall_neighbor_is_not_treated_as_gas(self) -> None:
        shape = (8, 8, 6)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask(model, closed_box_wall_mask(shape))
        fluid = _rest_state(model)
        source = FslState(model)
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[:, 1, :] = int(FslCellFlag.INTERFACE)
        flags[:, 2, :] = int(FslCellFlag.LIQUID)
        flags[:, 3, :] = int(FslCellFlag.INTERFACE)
        flags[walls.host] = int(FslCellFlag.GAS)
        mass = np.zeros(shape, dtype=np.float32)
        mass[flags == int(FslCellFlag.LIQUID)] = 1.0
        mass[flags == int(FslCellFlag.INTERFACE)] = 0.5
        source.flags.assign(flags)
        source.mass.assign(mass)
        source.fill_level.assign(mass)
        fluid_out = HomeCoreState(model)
        destination = FslState(model)

        diagnostics = FslWallTopologyUpdater(model, walls).resolve(
            fluid, source, source.mass, fluid_out, destination
        )

        interior_lower = destination.flags.numpy()[1:-1, 1, 1:-1]
        np.testing.assert_array_equal(interior_lower, int(FslCellFlag.LIQUID))
        self.assertEqual(diagnostics.interface_to_liquid_count, 24)
        self.assertEqual(diagnostics.direct_liquid_gas_link_count, 0)
        np.testing.assert_array_equal(destination.mass.numpy()[walls.host], 0.0)

    def test_hydrostatic_horizontal_layer_remains_near_rest(self) -> None:
        shape = (18, 18, 6)
        model = HomeCoreModel(
            fluid_grid_res=shape,
            device="cpu",
            kinematic_viscosity=0.02,
            body_acceleration=(0.0, -1.0e-5, 0.0),
        )
        walls = FslWallMask.closed_box(model)
        fluid_a = HomeCoreState(model)
        fluid_b = HomeCoreState(model)
        fsl_a = FslState(model)
        fsl_b = FslState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, 1:7, :] = 1.0
        fill[:, 7, :] = 0.5
        fill[walls.host] = 0.0
        walls.initialize_hydrostatic(
            fluid_a,
            fsl_a,
            fill,
            gas_density=1.0,
            gravity_axis=1,
            surface_coordinate=7.0,
        )
        stepper = FslWallDynamicTopologyStepper(model, walls)
        initial_total = float(np.sum(fsl_a.mass.numpy(), dtype=np.float64))
        max_speed = 0.0

        for _ in range(20):
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
            max_speed = max(max_speed, diagnostics.fluid.max_speed)
            fluid_a, fluid_b = fluid_b, fluid_a
            fsl_a, fsl_b = fsl_b, fsl_a

        final_total = float(
            np.sum(fsl_a.mass.numpy(), dtype=np.float64)
            + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
        )
        self.assertLess(max_speed, 2.0e-3)
        self.assertAlmostEqual(final_total, initial_total, delta=3.0e-4)
        self.assertEqual(stepper.last_diagnostics.topology.interface_to_liquid_count, 0)
        self.assertEqual(stepper.last_diagnostics.topology.interface_to_gas_count, 0)
        np.testing.assert_array_equal(fsl_a.mass.numpy()[walls.host], 0.0)
        np.testing.assert_array_equal(fsl_a.fill_level.numpy()[walls.host], 0.0)
        wall_momenta = fluid_a.moments.numpy().reshape(10, *shape)[1:4]
        np.testing.assert_array_equal(wall_momenta[:, walls.host], 0.0)
        self.assertGreater(stepper.last_diagnostics.stream.wall_link_count, 0)

    def test_failed_direct_link_does_not_commit_outputs(self) -> None:
        shape = (10, 10, 6)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask.closed_box(model)
        fluid_in = _rest_state(model)
        fluid_out = _rest_state(model)
        fsl_in = FslState(model)
        fsl_out = FslState(model)
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[:, 1:4, :] = int(FslCellFlag.LIQUID)
        flags[walls.host] = int(FslCellFlag.GAS)
        mass = np.zeros(shape, dtype=np.float32)
        mass[flags == int(FslCellFlag.LIQUID)] = 1.0
        fsl_in.flags.assign(flags)
        fsl_in.mass.assign(mass)
        fsl_in.fill_level.assign(mass)
        fsl_out.mass.fill_(7.0)
        before_fluid = fluid_out.moments.numpy().copy()
        before_mass = fsl_out.mass.numpy().copy()

        with self.assertRaises(RuntimeError):
            FslWallDynamicTopologyStepper(model, walls).step(
                fluid_in, fsl_in, fluid_out, fsl_out, model.time_step
            )

        np.testing.assert_array_equal(fluid_out.moments.numpy(), before_fluid)
        np.testing.assert_array_equal(fsl_out.mass.numpy(), before_mass)


if __name__ == "__main__":
    unittest.main()
