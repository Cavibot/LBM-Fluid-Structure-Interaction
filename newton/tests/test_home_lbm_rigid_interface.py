# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end boundary/load exchange test for the new HOME rigid interface."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidInterface,
    RigidMotionPredictor,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder


class TestHomeLbmRigidInterface(unittest.TestCase):
    def test_midpoint_moving_sphere_drives_fluid_and_receives_physical_load(self) -> None:
        cell_size = 0.1
        time_step = 0.02
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        fluid_model = HomeLbmModel(
            fluid_grid_res=(14, 14, 14),
            fluid_grid_cell_size=cell_size,
            time_step=time_step,
            kinematic_viscosity=0.1,
            device=device,
        )
        fluid = HomeLbmDomain(fluid_model)
        fluid.create_state()
        fluid.solver.initialize_uniform_lattice(fluid.state)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.7, 0.7, 0.7), label="home_moving_sphere")
        builder.add_shape_sphere(body, radius=0.28)
        rigid = RigidDomain(builder.finalize(device=device))
        rigid.create_state()
        rigid.state._body_qd = wp.array(
            [[0.12, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )

        interface = HomeLbmRigidInterface(fluid, rigid)
        predictor = RigidMotionPredictor(rigid.model.body_count, device=device)
        predictor.validate_lattice_displacement(
            rigid.state.body_qd,
            interface.geometry.radius_bound,
            cell_size,
            time_step,
        )
        predictor.sample(rigid.state.body_q, rigid.state.body_qd, rigid.model.body_com, time_step)
        transitions = interface.prepare_step(predictor.body_q_half, rigid.state.body_qd)

        fluid.step(time_step)
        diagnostics = fluid.solver.validate_state(fluid.state)
        interface.collect_wrench(predictor.body_q_half, rigid.state.body_f)
        wp.synchronize_device(device)

        moments = fluid.state.moments.numpy().reshape(10, -1)
        fluid_mask = fluid.state.solid_phi.numpy().reshape(-1) >= 0.0
        total_fluid_momentum = moments[1:4, fluid_mask].sum(axis=1)
        rigid_force = rigid.state.body_f.numpy()[body, :3]
        self.assertGreater(interface.cut_links.link_count, 0)
        self.assertEqual(diagnostics.missing_cut_link_count, 0)
        self.assertGreaterEqual(transitions.fresh_cell_count, 0)
        self.assertGreater(total_fluid_momentum[0], 0.0)
        self.assertLess(rigid_force[0], 0.0)
        self.assertTrue(np.all(np.isfinite(rigid.state.body_f.numpy())))


if __name__ == "__main__":
    unittest.main()
