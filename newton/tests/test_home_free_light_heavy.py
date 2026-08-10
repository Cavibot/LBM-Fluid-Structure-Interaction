# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Opposite buoyant motion of light and heavy submerged spheres."""

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


class TestHomeFreeLightHeavySpheres(unittest.TestCase):
    def _run_opposite_buoyancy(
        self, device: str, *, geometric: bool = False
    ) -> None:
        shape = (24, 16, 24)
        gravity = -2.0e-3
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=(0.0, 0.0, gravity),
            max_lattice_speed=0.15,
            periodic=(False, False, False),
            device=device,
        )
        fluid = (
            HomeFreeGeometricDomain(model, contact_angle_degrees=90.0)
            if geometric
            else HomeFreeLegacyDomain(model)
        )
        builder = RigidModelBuilder(gravity=gravity)
        light = builder.add_body(position=(7.0, 8.0, 10.0), label="light_sphere")
        builder.add_shape_sphere(
            light,
            radius=3.0,
            cfg=ShapeConfig(density=0.5, has_shape_collision=False),
        )
        heavy = builder.add_body(position=(17.0, 8.0, 10.0), label="heavy_sphere")
        builder.add_shape_sphere(
            heavy,
            radius=3.0,
            cfg=ShapeConfig(density=1.5, has_shape_collision=False),
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
        fill[:, :, :18] = 1.0
        fill[:, :, 18] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=18,
            gas_direction=1,
        )
        initial_positions = rigid.state.body_q.numpy()[:, :3].astype(np.float64)
        initial_mass = _represented_mass(fluid)

        for _ in range(20):
            diagnostics = coupling.step()
            self.assertEqual(diagnostics.interface.load.dry_link_count, 0)
            self.assertEqual(
                diagnostics.interface.fluid.direct_liquid_gas_link_count, 0
            )
            self.assertLess(
                diagnostics.strong_coupling_residual,
                coupling.strong_coupling_tolerance,
            )

        final_positions = rigid.state.body_q.numpy()[:, :3].astype(np.float64)
        final_velocities = rigid.state.body_qd.numpy().astype(np.float64)
        light_displacement = final_positions[light, 2] - initial_positions[light, 2]
        heavy_displacement = final_positions[heavy, 2] - initial_positions[heavy, 2]
        self.assertGreater(light_displacement, 0.08)
        self.assertLess(heavy_displacement, -0.015)
        self.assertGreater(light_displacement, abs(heavy_displacement))
        self.assertGreater(final_velocities[light, 2], 0.0)
        self.assertLess(final_velocities[heavy, 2], 0.0)
        self.assertLess(
            abs(_represented_mass(fluid) - initial_mass) / initial_mass,
            2.0e-5,
        )

    def test_cpu_light_sphere_rises_while_heavy_sphere_sinks(self) -> None:
        self._run_opposite_buoyancy("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_light_sphere_rises_while_heavy_sphere_sinks(self) -> None:
        self._run_opposite_buoyancy("cuda:0")

    def test_cpu_geometric_light_sphere_rises_while_heavy_sphere_sinks(self) -> None:
        self._run_opposite_buoyancy("cpu", geometric=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_geometric_light_sphere_rises_while_heavy_sphere_sinks(self) -> None:
        self._run_opposite_buoyancy("cuda:0", geometric=True)


if __name__ == "__main__":
    unittest.main()
