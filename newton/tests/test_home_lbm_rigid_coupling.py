# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Verified substep ordering for the independent HOME rigid coupling."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidCoupling,
)
from wanphys._src.collision.pipeline import CollisionPipeline
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    create_semiimplicit_solver,
)


class TestHomeLbmRigidCoupling(unittest.TestCase):
    @staticmethod
    def _make_strong_case(
        device: str,
        maximum_iterations: int,
        tolerance: float,
    ) -> tuple[HomeLbmDomain, RigidDomain, HomeLbmRigidCoupling, object, int]:
        model = HomeLbmModel(
            fluid_grid_res=(14, 14, 14),
            fluid_grid_cell_size=0.1,
            time_step=0.01,
            kinematic_viscosity=0.1,
            device=device,
        )
        fluid = HomeLbmDomain(model)
        fluid.create_state()
        fluid.solver.initialize_uniform_lattice(fluid.state)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.7, 0.7, 0.7), label="strong_sphere")
        builder.add_shape_sphere(body, radius=0.24)
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model),
        )
        rigid.create_state()
        rigid.state._body_qd = wp.array(
            [[0.08, 0.01, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeLbmRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=maximum_iterations,
            strong_coupling_tolerance=tolerance,
            strong_coupling_relaxation=0.8,
        )
        return fluid, rigid, coupling, CollisionPipeline.collide_rigid(rigid), body

    def test_midpoint_fluid_wrench_is_held_across_rigid_substeps(self) -> None:
        cell_size = 0.1
        time_step = 0.01
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        fluid_model = HomeLbmModel(
            fluid_grid_res=(18, 18, 18),
            fluid_grid_cell_size=cell_size,
            time_step=time_step,
            kinematic_viscosity=0.1,
            device=device,
        )
        fluid = HomeLbmDomain(fluid_model)
        fluid.create_state()
        fluid.solver.initialize_uniform_lattice(fluid.state)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.9, 0.9, 0.9), label="coupled_sphere")
        builder.add_shape_sphere(body, radius=0.24)
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model),
        )
        rigid.create_state()
        initial_velocity = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        rigid.state._body_qd = wp.array(
            [[*initial_velocity, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        external_wrench = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wp.copy(
            rigid.state.body_f,
            wp.array([external_wrench], dtype=wp.spatial_vector, device=device),
        )

        coupling = HomeLbmRigidCoupling(fluid, rigid, rigid_substeps=3)
        contact_calls = 0

        def provide_contacts(_domain: RigidDomain) -> object:
            nonlocal contact_calls
            contact_calls += 1
            return CollisionPipeline.collide_rigid(_domain)

        diagnostics = coupling.step(contact_provider=provide_contacts)
        wp.synchronize_device(device)

        final_velocity = rigid.state.body_qd.numpy()[body, :3]
        fluid_force, _ = coupling.interface.wrench.physical_numpy(fluid_model.scaling)
        inverse_mass = np.float32(rigid.model.body_inv_mass.numpy()[body])
        total_force = np.asarray(
            fluid_force[body] + external_wrench[:3], dtype=np.float32
        )
        expected_velocity = initial_velocity.copy()
        rigid_dt = np.float32(time_step / 3.0)
        for _ in range(3):
            expected_velocity = np.asarray(
                expected_velocity + total_force * inverse_mass * rigid_dt,
                dtype=np.float32,
            )
        np.testing.assert_allclose(
            final_velocity,
            expected_velocity,
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            rigid.state.body_f.numpy()[body], external_wrench, atol=1.0e-7
        )
        self.assertLess(fluid_force[body, 0], 0.0)
        self.assertLess(final_velocity[0], initial_velocity[0])
        self.assertGreater(final_velocity[1], 0.0)
        self.assertEqual(contact_calls, 3)
        self.assertEqual(diagnostics.rigid_substeps, 3)
        self.assertEqual(diagnostics.fluid.invalid_cell_count, 0)
        self.assertLess(diagnostics.maximum_lattice_displacement, 0.5)

    def test_ambiguous_contact_ownership_is_rejected(self) -> None:
        model = HomeLbmModel(fluid_grid_res=(4, 4, 4), device="cpu")
        fluid = HomeLbmDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(2.0, 2.0, 2.0), label="contact_contract")
        builder.add_shape_sphere(body, radius=0.25)
        rigid_model = builder.finalize(device="cpu")
        rigid = RigidDomain(rigid_model)
        coupling = HomeLbmRigidCoupling(fluid, rigid)

        with self.assertRaisesRegex(ValueError, "either fixed contacts"):
            coupling.step(contacts=object(), contact_provider=lambda _domain: object())

    def test_strong_coupling_converges_and_restores_external_forces(self) -> None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        fluid, rigid, coupling, contacts, body = self._make_strong_case(
            device,
            maximum_iterations=12,
            tolerance=1.0e-6,
        )
        external_wrench = np.array(
            [0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        rigid.state.body_f.assign([external_wrench])

        diagnostics = coupling.step(contacts=contacts)

        self.assertGreaterEqual(diagnostics.strong_coupling_iterations, 1)
        self.assertLessEqual(diagnostics.strong_coupling_residual, 1.0e-6)
        np.testing.assert_allclose(
            rigid.state.body_f.numpy()[body],
            external_wrench,
            atol=1.0e-7,
        )
        self.assertEqual(diagnostics.fluid.invalid_cell_count, 0)
        self.assertIs(fluid.solver.cut_links, coupling.interface.cut_links)

    def test_failed_strong_coupling_rolls_back_both_domains(self) -> None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        fluid, rigid, coupling, contacts, _ = self._make_strong_case(
            device,
            maximum_iterations=2,
            tolerance=1.0e-20,
        )
        moments_before = fluid.state.moments.numpy().copy()
        phi_before = fluid.state.solid_phi.numpy().copy()
        body_q_before = rigid.state.body_q.numpy().copy()
        body_qd_before = rigid.state.body_qd.numpy().copy()

        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            coupling.step(contacts=contacts)

        np.testing.assert_array_equal(fluid.state.moments.numpy(), moments_before)
        np.testing.assert_array_equal(fluid.state.solid_phi.numpy(), phi_before)
        np.testing.assert_array_equal(rigid.state.body_q.numpy(), body_q_before)
        np.testing.assert_array_equal(rigid.state.body_qd.numpy(), body_qd_before)
        self.assertIs(fluid.solver.cut_links, coupling.interface.cut_links)


if __name__ == "__main__":
    unittest.main()
