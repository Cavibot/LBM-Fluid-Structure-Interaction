# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit midpoint HOME-Free rigid schedule and horizontal impulse ledger."""

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


def _liquid_momentum_x(fluid: HomeFreeLegacyDomain) -> float:
    moments = fluid.fluid_state.moments.numpy().reshape(10, -1)
    mass = fluid.free_surface_state.mass.numpy().reshape(-1)
    committed = mass * moments[1] / moments[0]
    recipients = fluid.topology.active_neighbor_count.numpy().reshape(-1)
    queued = fluid.free_surface_state.excess_momentum.numpy().reshape(3, -1)[0]
    lattice = float(np.sum(committed + recipients * queued, dtype=np.float64))
    return lattice * fluid.model.scaling.momentum_unit


class TestHomeFreeRigidCoupling(unittest.TestCase):
    @staticmethod
    def _make_strong_case(
        device: str,
        maximum_iterations: int,
        tolerance: float,
    ) -> tuple[HomeFreeLegacyDomain, RigidDomain, HomeFreeRigidCoupling, int]:
        shape = (16, 12, 16)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=0.1,
            time_step=0.01,
            reference_density=1.0,
            kinematic_viscosity=0.1,
            max_lattice_speed=0.2,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.8, 0.6, 0.65), label="strong_free_sphere")
        builder.add_shape_sphere(
            body,
            radius=0.22,
            cfg=ShapeConfig(density=1.2, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        rigid.state._body_qd = wp.array(
            [[0.06, 0.01, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=maximum_iterations,
            strong_coupling_tolerance=tolerance,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :12] = 1.0
        fill[:, :, 12] = 0.5
        fluid.initialize_uniform_lattice(fill)
        return fluid, rigid, coupling, body

    def _run_submerged_horizontal_impulse(self, device: str) -> None:
        shape = (20, 12, 18)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=0.1,
            time_step=0.02,
            reference_density=1.0,
            kinematic_viscosity=0.1,
            max_lattice_speed=0.2,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.8627, 0.6, 0.6), label="submerged_sphere")
        builder.add_shape_sphere(
            body,
            radius=0.2,
            cfg=ShapeConfig(density=1.2, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        initial_velocity = 0.05
        rigid.state._body_qd = wp.array(
            [[initial_velocity, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeFreeRigidCoupling(fluid, rigid, rigid_substeps=2)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :13] = 1.0
        fill[:, :, 13] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=13,
            gas_direction=1,
        )
        body_mass = rigid.model.get_body_mass(body)
        initial_total_x = _liquid_momentum_x(fluid) + body_mass * initial_velocity
        initial_liquid_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )

        diagnostics = coupling.step()

        final_velocity = float(rigid.state.body_qd.numpy()[body, 0])
        final_total_x = _liquid_momentum_x(fluid) + body_mass * final_velocity
        recipients = fluid.topology.active_neighbor_count.numpy()
        final_liquid_mass = float(
            np.sum(
                fluid.free_surface_state.mass.numpy()
                + fluid.free_surface_state.excess_mass.numpy() * recipients,
                dtype=np.float64,
            )
        )
        self.assertGreater(diagnostics.interface.transitions.dead_cell_count, 0)
        self.assertLess(
            abs(final_total_x - initial_total_x) / abs(initial_total_x),
            3.0e-4,
        )
        self.assertLess(
            abs(final_liquid_mass - initial_liquid_mass) / initial_liquid_mass,
            2.0e-5,
        )
        self.assertLess(final_velocity, initial_velocity)
        self.assertGreater(final_velocity, 0.0)
        self.assertEqual(diagnostics.interface.load.dry_link_count, 0)
        self.assertEqual(diagnostics.interface.fluid.missing_cut_link_count, 0)
        self.assertEqual(diagnostics.interface.fluid.direct_liquid_gas_link_count, 0)
        np.testing.assert_array_equal(rigid.state.body_f.numpy(), 0.0)

    def test_cpu_submerged_horizontal_impulse_is_action_reaction_consistent(self) -> None:
        self._run_submerged_horizontal_impulse("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_submerged_horizontal_impulse_is_action_reaction_consistent(self) -> None:
        self._run_submerged_horizontal_impulse("cuda:0")

    def _assert_strong_coupling_converges(self, device: str) -> None:
        fluid, rigid, coupling, body = self._make_strong_case(
            device,
            maximum_iterations=12,
            tolerance=1.0e-6,
        )
        external_wrench = np.asarray(
            [0.0, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        rigid.state.body_f.assign([external_wrench])

        diagnostics = coupling.step()

        self.assertGreaterEqual(diagnostics.strong_coupling_iterations, 1)
        self.assertLessEqual(diagnostics.strong_coupling_residual, 1.0e-6)
        np.testing.assert_allclose(
            rigid.state.body_f.numpy()[body], external_wrench, atol=1.0e-7
        )
        self.assertEqual(diagnostics.interface.fluid.invalid_cell_count, 0)
        self.assertIs(fluid.solver.cut_links, coupling.interface.cut_links)
        self.assertIsNone(coupling.interface._prepared)

    def test_cpu_strong_coupling_converges_and_restores_external_forces(self) -> None:
        self._assert_strong_coupling_converges("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_strong_coupling_converges_and_restores_external_forces(self) -> None:
        self._assert_strong_coupling_converges("cuda:0")

    def test_failed_strong_coupling_rolls_back_full_free_surface_state(self) -> None:
        fluid, rigid, coupling, _ = self._make_strong_case(
            "cpu",
            maximum_iterations=2,
            tolerance=1.0e-20,
        )
        fluid_before = (
            fluid.fluid_state.moments.numpy().copy(),
            fluid.fluid_state.solid_phi.numpy().copy(),
            fluid.fluid_state.solid_body_id.numpy().copy(),
            fluid.free_surface_state.mass.numpy().copy(),
            fluid.free_surface_state.fill_level.numpy().copy(),
            fluid.free_surface_state.excess_mass.numpy().copy(),
            fluid.free_surface_state.excess_momentum.numpy().copy(),
            fluid.free_surface_state.flags.numpy().copy(),
        )
        rigid_before = (
            rigid.state.body_q.numpy().copy(),
            rigid.state.body_qd.numpy().copy(),
            rigid.state.body_f.numpy().copy(),
        )

        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            coupling.step()

        fluid_after = (
            fluid.fluid_state.moments.numpy(),
            fluid.fluid_state.solid_phi.numpy(),
            fluid.fluid_state.solid_body_id.numpy(),
            fluid.free_surface_state.mass.numpy(),
            fluid.free_surface_state.fill_level.numpy(),
            fluid.free_surface_state.excess_mass.numpy(),
            fluid.free_surface_state.excess_momentum.numpy(),
            fluid.free_surface_state.flags.numpy(),
        )
        rigid_after = (
            rigid.state.body_q.numpy(),
            rigid.state.body_qd.numpy(),
            rigid.state.body_f.numpy(),
        )
        for before, after in zip(fluid_before, fluid_after, strict=True):
            np.testing.assert_array_equal(after, before)
        for before, after in zip(rigid_before, rigid_after, strict=True):
            np.testing.assert_array_equal(after, before)
        self.assertIs(fluid.solver.cut_links, coupling.interface.cut_links)
        self.assertIsNone(coupling.interface._prepared)
        assert coupling.interface._remapper is not None
        np.testing.assert_array_equal(
            coupling.interface._remapper.previous_phi.numpy(),
            fluid.fluid_state.solid_phi.numpy(),
        )

    def test_strong_coupling_rolls_back_contact_provider_exception(self) -> None:
        fluid, rigid, coupling, _ = self._make_strong_case(
            "cpu",
            maximum_iterations=8,
            tolerance=1.0e-6,
        )
        moments_before = fluid.fluid_state.moments.numpy().copy()
        mass_before = fluid.free_surface_state.mass.numpy().copy()
        flags_before = fluid.free_surface_state.flags.numpy().copy()
        body_q_before = rigid.state.body_q.numpy().copy()
        body_qd_before = rigid.state.body_qd.numpy().copy()

        def fail_contacts(_domain: RigidDomain) -> object:
            raise ValueError("intentional contact failure")

        with self.assertRaisesRegex(ValueError, "intentional contact failure"):
            coupling.step(contact_provider=fail_contacts)

        np.testing.assert_array_equal(
            fluid.fluid_state.moments.numpy(), moments_before
        )
        np.testing.assert_array_equal(fluid.free_surface_state.mass.numpy(), mass_before)
        np.testing.assert_array_equal(fluid.free_surface_state.flags.numpy(), flags_before)
        np.testing.assert_array_equal(rigid.state.body_q.numpy(), body_q_before)
        np.testing.assert_array_equal(rigid.state.body_qd.numpy(), body_qd_before)
        self.assertIs(fluid.solver.cut_links, coupling.interface.cut_links)
        self.assertIsNone(coupling.interface._prepared)


if __name__ == "__main__":
    unittest.main()
