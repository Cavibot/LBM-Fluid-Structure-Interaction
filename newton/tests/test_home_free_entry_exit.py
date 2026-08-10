# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid sphere entry and exit through a HOME-Free interface."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeFreeRigidCoupling,
    HomeLbmModel,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


def _represented_mass(fluid: HomeFreeLegacyDomain) -> float:
    state = fluid.free_surface_state
    recipients = fluid.topology.active_neighbor_count.numpy()
    return float(
        np.sum(
            state.mass.numpy() + state.excess_mass.numpy() * recipients,
            dtype=np.float64,
        )
    )


class TestHomeFreeRigidEntryExit(unittest.TestCase):
    def _run_crossing(self, device: str, *, entering: bool) -> None:
        shape = (24, 24, 22)
        initial_speed = -0.15 if entering else 0.15
        initial_z = 14.8 if entering else 6.8
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            max_lattice_speed=0.2,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(
            position=(12.0, 12.0, initial_z),
            label="entering_sphere" if entering else "exiting_sphere",
        )
        builder.add_shape_sphere(
            body,
            radius=3.0,
            cfg=ShapeConfig(density=2.0, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        rigid.state._body_qd = wp.array(
            [[0.0, 0.0, initial_speed, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=10,
            strong_coupling_tolerance=2.0e-6,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :10] = 1.0
        fill[:, :, 10] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=10,
            gas_direction=1,
        )
        initial_mass = _represented_mass(fluid)
        wet_counts: list[int] = []
        dry_counts: list[int] = []
        fresh_total = 0
        dead_total = 0

        for _ in range(20):
            diagnostics = coupling.step()
            wet_counts.append(diagnostics.interface.load.wet_link_count)
            dry_counts.append(diagnostics.interface.load.dry_link_count)
            fresh_total += diagnostics.interface.transitions.fresh_cell_count
            dead_total += diagnostics.interface.transitions.dead_cell_count
            self.assertEqual(
                diagnostics.interface.fluid.direct_liquid_gas_link_count, 0
            )
            self.assertLess(
                diagnostics.strong_coupling_residual,
                coupling.strong_coupling_tolerance,
            )

        final_speed = float(rigid.state.body_qd.numpy()[body, 2])
        final_mass = _represented_mass(fluid)
        self.assertGreater(fresh_total, 0)
        self.assertGreater(dead_total, 0)
        self.assertLess(abs(final_speed), abs(initial_speed))
        self.assertEqual(np.sign(final_speed), np.sign(initial_speed))
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 2.0e-5)
        if entering:
            self.assertEqual(wet_counts[0], 0)
            self.assertGreater(wet_counts[-1], 0)
            self.assertGreater(dry_counts[0], 0)
        else:
            self.assertEqual(dry_counts[0], 0)
            self.assertGreater(dry_counts[-1], 0)
            self.assertGreater(wet_counts[0], 0)

    def test_cpu_sphere_enters_free_surface(self) -> None:
        self._run_crossing("cpu", entering=True)

    def test_cpu_sphere_exits_free_surface(self) -> None:
        self._run_crossing("cpu", entering=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_sphere_enters_free_surface(self) -> None:
        self._run_crossing("cuda:0", entering=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_sphere_exits_free_surface(self) -> None:
        self._run_crossing("cuda:0", entering=False)


if __name__ == "__main__":
    unittest.main()
