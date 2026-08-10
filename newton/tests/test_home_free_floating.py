# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Half-submerged sphere equilibrium for strong HOME-Free FSI."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
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


def _represented_mass(fluid: HomeFreeLegacyDomain) -> float:
    state = fluid.free_surface_state
    if getattr(fluid, "uses_independent_geometric_mass", False):
        return float(np.sum(state.mass.numpy(), dtype=np.float64))
    recipients = fluid.topology.active_neighbor_count.numpy()
    return float(
        np.sum(
            state.mass.numpy() + state.excess_mass.numpy() * recipients,
            dtype=np.float64,
        )
    )


class TestHomeFreeFloatingEquilibrium(unittest.TestCase):
    def _run_half_submerged_sphere(
        self, device: str, *, geometric: bool = False
    ) -> None:
        shape = (32, 32, 28)
        radius = 6.0
        gravity = -2.0e-4
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=(0.0, 0.0, gravity),
            max_lattice_speed=0.1,
            periodic=(False, False, False),
            device=device,
        )
        fluid = (
            HomeFreeGeometricDomain(model, contact_angle_degrees=90.0)
            if geometric
            else HomeFreeLegacyDomain(model)
        )
        builder = RigidModelBuilder(gravity=gravity)
        body = builder.add_body(
            position=(16.0, 16.0, 14.5), label="half_submerged_sphere"
        )
        builder.add_shape_sphere(
            body,
            radius=radius,
            cfg=ShapeConfig(density=0.5, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=12,
            strong_coupling_tolerance=2.0e-6,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :14] = 1.0
        fill[:, :, 14] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=14,
            gas_direction=1,
        )
        initial_mass = _represented_mass(fluid)
        initial_position = rigid.state.body_q.numpy()[body, :3].astype(np.float64)

        static = coupling.interface.step_static()
        force, torque = coupling.interface.wrench.physical_numpy(model.scaling)
        weight = rigid.model.get_body_mass(body) * abs(gravity)
        relative_balance_error = abs(float(force[body, 2]) - weight) / weight
        self.assertGreater(static.load.wet_link_count, 0)
        self.assertGreater(static.load.dry_link_count, 0)
        self.assertLess(relative_balance_error, 0.20)
        self.assertLess(float(np.linalg.norm(force[body, :2])) / weight, 2.0e-4)
        self.assertLess(
            float(np.linalg.norm(torque[body])) / (weight * radius),
            2.0e-4,
        )

        maximum_residual = 0.0
        for _ in range(8):
            diagnostics = coupling.step()
            maximum_residual = max(
                maximum_residual, diagnostics.strong_coupling_residual
            )
            self.assertGreater(diagnostics.interface.load.wet_link_count, 0)
            self.assertGreater(diagnostics.interface.load.dry_link_count, 0)
            self.assertEqual(
                diagnostics.interface.fluid.direct_liquid_gas_link_count, 0
            )

        final_position = rigid.state.body_q.numpy()[body, :3].astype(np.float64)
        final_velocity = rigid.state.body_qd.numpy()[body].astype(np.float64)
        final_mass = _represented_mass(fluid)
        self.assertLess(maximum_residual, 2.0e-6)
        self.assertLess(abs(final_position[2] - initial_position[2]), 2.0e-3)
        self.assertLess(float(np.linalg.norm(final_position[:2] - initial_position[:2])), 2.0e-5)
        self.assertLess(abs(final_velocity[2]), 5.0e-4)
        self.assertLess(float(np.linalg.norm(final_velocity[:2])), 2.0e-5)
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 2.0e-5)

    def test_cpu_half_submerged_sphere_remains_in_equilibrium(self) -> None:
        self._run_half_submerged_sphere("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_half_submerged_sphere_remains_in_equilibrium(self) -> None:
        self._run_half_submerged_sphere("cuda:0")

    def test_cpu_geometric_half_submerged_sphere_remains_in_equilibrium(self) -> None:
        self._run_half_submerged_sphere("cpu", geometric=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_geometric_half_submerged_sphere_remains_in_equilibrium(self) -> None:
        self._run_half_submerged_sphere("cuda:0", geometric=True)


if __name__ == "__main__":
    unittest.main()
