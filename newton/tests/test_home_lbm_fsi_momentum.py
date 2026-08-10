# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenario-level momentum accounting for moving-solid HOME coupling."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.collision.pipeline import CollisionPipeline
from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidCoupling,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    create_semiimplicit_solver,
)


def _fluid_physical_momentum(fluid: HomeLbmDomain) -> np.ndarray:
    state = fluid.state
    moments = state.moments.numpy().reshape(10, -1)
    fluid_mask = state.solid_phi.numpy().reshape(-1) >= 0.0
    lattice_momentum = moments[1:4, fluid_mask].sum(axis=1, dtype=np.float64)
    return lattice_momentum * fluid.model.scaling.momentum_unit


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fluid_physical_angular_momentum(fluid: HomeLbmDomain) -> np.ndarray:
    state = fluid.state
    shape = state.res
    moments = state.moments.numpy().reshape(10, -1)
    fluid_mask = state.solid_phi.numpy().reshape(-1) >= 0.0
    coordinates = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0)
    positions = ((coordinates + 0.5) * fluid.model.fluid_grid_cell_size).reshape(-1, 3)
    physical_momentum = moments[1:4].T * fluid.model.scaling.momentum_unit
    return np.cross(positions[fluid_mask], physical_momentum[fluid_mask]).sum(axis=0)


def _rigid_physical_angular_momentum(
    rigid: RigidDomain,
    body: int,
) -> np.ndarray:
    transform = rigid.state.body_q.numpy()[body]
    velocity = rigid.state.body_qd.numpy()[body]
    rotation = _quaternion_matrix(transform[3:])
    center = transform[:3] + rotation @ rigid.model.body_com.numpy()[body]
    mass = rigid.model.get_body_mass(body)
    inertia_body = rigid.model.body_inertia.numpy()[body]
    inertia_world = rotation @ inertia_body @ rotation.T
    return np.cross(center, mass * velocity[:3]) + inertia_world @ velocity[3:]


class TestHomeLbmFsiMomentum(unittest.TestCase):
    def test_periodic_moving_sphere_conserves_total_linear_momentum(self) -> None:
        cell_size = 0.1
        time_step = 0.01
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        fluid_model = HomeLbmModel(
            fluid_grid_res=(18, 18, 18),
            fluid_grid_cell_size=cell_size,
            time_step=time_step,
            kinematic_viscosity=0.1,
            periodic=(True, True, True),
            device=device,
        )
        fluid = HomeLbmDomain(fluid_model)
        fluid.create_state()
        fluid.solver.initialize_uniform_lattice(fluid.state)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.805, 0.9, 0.9), label="momentum_sphere")
        builder.add_shape_sphere(body, radius=0.24)
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model),
        )
        rigid.create_state()
        initial_velocity = np.array([0.08, 0.015, -0.01], dtype=np.float32)
        rigid.state._body_qd = wp.array(
            [[*initial_velocity, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeLbmRigidCoupling(fluid, rigid, rigid_substeps=2)
        contacts = CollisionPipeline.collide_rigid(rigid)
        body_mass = rigid.model.get_body_mass(body)
        initial_total = _fluid_physical_momentum(fluid) + body_mass * initial_velocity

        totals = []
        transition_count = 0
        for _ in range(24):
            diagnostics = coupling.step(contacts=contacts)
            transition_count += (
                diagnostics.transitions.fresh_cell_count
                + diagnostics.transitions.dead_cell_count
            )
            rigid_momentum = body_mass * rigid.state.body_qd.numpy()[body, :3]
            totals.append(_fluid_physical_momentum(fluid) + rigid_momentum)

        totals_array = np.asarray(totals)
        drift = np.linalg.norm(totals_array - initial_total, axis=1)
        momentum_scale = np.linalg.norm(initial_total)
        self.assertGreater(transition_count, 0)
        self.assertLess(float(drift.max() / momentum_scale), 2.0e-5)
        self.assertLess(rigid.state.body_qd.numpy()[body, 0], initial_velocity[0])

    def test_rotating_sphere_conserves_total_angular_momentum(self) -> None:
        cell_size = 0.1
        time_step = 0.01
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        fluid_model = HomeLbmModel(
            fluid_grid_res=(32, 32, 32),
            fluid_grid_cell_size=cell_size,
            time_step=time_step,
            kinematic_viscosity=0.1,
            periodic=(True, True, True),
            device=device,
        )
        fluid = HomeLbmDomain(fluid_model)
        fluid.create_state()
        fluid.solver.initialize_uniform_lattice(fluid.state)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(1.6, 1.6, 1.6), label="rotating_sphere")
        builder.add_shape_sphere(body, radius=0.28)
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        initial_angular_velocity = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        rigid.state._body_qd = wp.array(
            [[0.0, 0.0, 0.0, *initial_angular_velocity]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeLbmRigidCoupling(fluid, rigid, rigid_substeps=2)
        contacts = CollisionPipeline.collide_rigid(rigid)
        initial_total = _rigid_physical_angular_momentum(rigid, body)

        totals = []
        for _ in range(10):
            coupling.step(contacts=contacts)
            totals.append(
                _fluid_physical_angular_momentum(fluid)
                + _rigid_physical_angular_momentum(rigid, body)
            )

        totals_array = np.asarray(totals)
        drift = np.linalg.norm(totals_array - initial_total, axis=1)
        angular_momentum_scale = np.linalg.norm(initial_total)
        fluid_angular_momentum = _fluid_physical_angular_momentum(fluid)
        self.assertLess(float(drift.max() / angular_momentum_scale), 3.0e-5)
        self.assertGreater(fluid_angular_momentum[2], 0.0)
        self.assertLess(
            rigid.state.body_qd.numpy()[body, 5],
            initial_angular_velocity[2],
        )


if __name__ == "__main__":
    unittest.main()
