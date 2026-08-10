# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid entry and exit through the geometric HOME-Free interface."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeGeometricDomain,
    HomeFreeRigidCoupling,
    HomeLbmModel,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


class TestHomeFreeGeometricRigidEntryExit(unittest.TestCase):
    def _run_crossing(self, device: str, *, entering: bool) -> None:
        shape = (24, 24, 22)
        initial_speed = -0.15 if entering else 0.15
        initial_z = 14.8 if entering else 7.3
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            max_lattice_speed=0.2,
            periodic=(True, True, False),
            device=device,
        )
        fluid = HomeFreeGeometricDomain(
            model,
            contact_angle_degrees=90.0,
            project_courant=True,
        )
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
        initial_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        wet_counts: list[int] = []
        dry_counts: list[int] = []
        wetting_counts: list[int] = []
        fresh_total = 0
        dead_total = 0
        queue_source_total = 0

        step_count = 20 if entering else 48
        for step in range(step_count):
            with self.subTest(step=step):
                diagnostics = coupling.step()
            geometric = fluid.last_geometric_diagnostics
            assert geometric is not None and geometric.projection is not None
            wet_counts.append(diagnostics.interface.load.wet_link_count)
            dry_counts.append(diagnostics.interface.load.dry_link_count)
            wetting_counts.append(
                geometric.surface_tension.geometry.wetting_cell_count
            )
            fresh_total += diagnostics.interface.transitions.fresh_cell_count
            dead_total += diagnostics.interface.transitions.dead_cell_count
            queue_source_total += geometric.queue.queue_source_count
            self.assertEqual(
                diagnostics.interface.fluid.direct_liquid_gas_link_count, 0
            )
            self.assertEqual(diagnostics.interface.fluid.missing_cut_link_count, 0)
            self.assertLessEqual(
                geometric.projection.projected_max_divergence,
                2.0e-8,
                f"step {step}",
            )
            self.assertLess(
                diagnostics.strong_coupling_residual,
                coupling.strong_coupling_tolerance,
            )
            self.assertLessEqual(geometric.queue.relative_mass_error, 4.0e-6)
            self.assertLessEqual(geometric.queue.relative_momentum_error, 4.0e-6)
            np.testing.assert_array_equal(
                fluid.free_surface_state.excess_mass.numpy(), 0.0
            )
            np.testing.assert_array_equal(
                fluid.free_surface_state.excess_momentum.numpy(), 0.0
            )
            current_mass = float(
                np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
            )
            self.assertLess(
                abs(current_mass - initial_mass) / initial_mass,
                2.0e-5,
                f"step {step}",
            )

        final_speed = float(rigid.state.body_qd.numpy()[body, 2])
        final_z = float(rigid.state.body_q.numpy()[body, 2])
        final_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        self.assertGreater(fresh_total, 0)
        self.assertGreater(dead_total, 0)
        self.assertGreater(queue_source_total, 0)
        self.assertGreater(max(wetting_counts), 0)
        self.assertLess(abs(final_speed), abs(initial_speed))
        self.assertEqual(np.sign(final_speed), np.sign(initial_speed))
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 2.0e-5)
        if entering:
            self.assertEqual(wet_counts[0], 0)
            self.assertGreater(wet_counts[-1], 0)
            self.assertGreater(dry_counts[0], 0)
        else:
            self.assertEqual(dry_counts[0], 0)
            self.assertGreater(
                max(dry_counts),
                0,
                f"sphere did not expose a dry link; final center z={final_z:.6f}",
            )
            self.assertGreater(wet_counts[0], 0)

    def test_cpu_sphere_enters_geometric_free_surface(self) -> None:
        self._run_crossing("cpu", entering=True)

    def test_cpu_sphere_exits_geometric_free_surface(self) -> None:
        self._run_crossing("cpu", entering=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_sphere_enters_geometric_free_surface(self) -> None:
        self._run_crossing("cuda:0", entering=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_sphere_exits_geometric_free_surface(self) -> None:
        self._run_crossing("cuda:0", entering=False)


if __name__ == "__main__":
    unittest.main()
