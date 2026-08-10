# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreSolver,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslDynamicTopologyStepper,
    FslState,
)


def _layered_fill(shape: tuple[int, int, int]) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    fill[3] = 0.5
    fill[4:8] = 1.0
    fill[8] = 0.5
    return fill


class TestFslDynamicTopologyStepper(unittest.TestCase):
    def _states(self) -> tuple:
        shape = (12, 4, 4)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        solver = HomeCoreSolver(model)
        fluid_in = HomeCoreState(model)
        solver.initialize_uniform_lattice(fluid_in)
        fsl_in = FslState(model)
        fsl_in.initialize_from_fill_level(fluid_in, _layered_fill(shape))
        return model, fluid_in, fsl_in, HomeCoreState(model), FslState(model)

    def test_static_layer_executes_without_topology_change(self) -> None:
        model, fluid_in, fsl_in, fluid_out, fsl_out = self._states()
        diagnostics = FslDynamicTopologyStepper(model).step(
            fluid_in, fsl_in, fluid_out, fsl_out, model.time_step
        )

        np.testing.assert_array_equal(fsl_out.flags.numpy(), fsl_in.flags.numpy())
        self.assertEqual(diagnostics.topology.interface_to_liquid_count, 0)
        self.assertEqual(diagnostics.topology.interface_to_gas_count, 0)
        self.assertLess(abs(diagnostics.topology.relative_total_mass_drift), 2.0e-7)

    def test_growth_is_committed_only_after_full_validation(self) -> None:
        model, fluid_in, fsl_in, fluid_out, fsl_out = self._states()
        mass = fsl_in.mass.numpy()
        mass[3] = 1.1
        fsl_in.mass.assign(mass)
        diagnostics = FslDynamicTopologyStepper(model).step(
            fluid_in, fsl_in, fluid_out, fsl_out, model.time_step
        )

        np.testing.assert_array_equal(fsl_out.flags.numpy()[3], int(FslCellFlag.LIQUID))
        np.testing.assert_array_equal(fsl_out.flags.numpy()[2], int(FslCellFlag.INTERFACE))
        self.assertEqual(diagnostics.topology.interface_to_liquid_count, 16)
        self.assertEqual(diagnostics.topology.gas_to_interface_count, 16)

    def test_failure_does_not_modify_caller_outputs(self) -> None:
        model, fluid_in, fsl_in, fluid_out, fsl_out = self._states()
        bad_fill = np.zeros(fsl_in.res, dtype=np.float32)
        bad_fill[4:8] = 1.0
        fsl_in.initialize_from_fill_level(
            fluid_in, bad_fill, validate_topology=False
        )
        fluid_out.moments.fill_(7.0)
        fsl_out.mass.fill_(3.0)
        before_fluid = fluid_out.moments.numpy().copy()
        before_mass = fsl_out.mass.numpy().copy()

        with self.assertRaises(RuntimeError):
            FslDynamicTopologyStepper(model).step(
                fluid_in, fsl_in, fluid_out, fsl_out, model.time_step
            )

        np.testing.assert_array_equal(fluid_out.moments.numpy(), before_fluid)
        np.testing.assert_array_equal(fsl_out.mass.numpy(), before_mass)

    def test_translating_slab_crosses_a_lattice_plane(self) -> None:
        shape = (24, 4, 4)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        solver = HomeCoreSolver(model)
        fluid_a = HomeCoreState(model)
        fluid_b = HomeCoreState(model)
        solver.initialize_uniform_lattice(fluid_a, velocity=(0.05, 0.0, 0.0))
        fill = np.zeros(shape, dtype=np.float32)
        fill[4] = 0.5
        fill[5:12] = 1.0
        fill[12] = 0.5
        fsl_a = FslState(model)
        fsl_b = FslState(model)
        fsl_a.initialize_from_fill_level(fluid_a, fill)
        stepper = FslDynamicTopologyStepper(model)
        initial_mass = fsl_a.validate(fluid_a).total_mass
        transition_count = 0

        for _ in range(20):
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
            transition_count += (
                diagnostics.topology.interface_to_liquid_count
                + diagnostics.topology.interface_to_gas_count
            )
            fluid_a, fluid_b = fluid_b, fluid_a
            fsl_a, fsl_b = fsl_b, fsl_a

        final_mass = fsl_a.validate(
            fluid_a, allow_interface_endpoints=True
        ).total_mass
        self.assertGreater(transition_count, 0)
        self.assertAlmostEqual(final_mass, initial_mass, delta=2.0e-4)
        self.assertEqual(
            fsl_a.validate(
                fluid_a, allow_interface_endpoints=True
            ).direct_liquid_gas_link_count,
            0,
        )


if __name__ == "__main__":
    unittest.main()
