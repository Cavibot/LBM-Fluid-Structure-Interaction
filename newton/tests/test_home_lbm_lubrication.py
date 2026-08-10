# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic and coupling tests for HOME sphere-wall lubrication."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidCoupling,
    SphereWallLubrication,
    SphereWallLubricationConfig,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


class TestHomeLbmLubrication(unittest.TestCase):
    @staticmethod
    def _make_sphere(
        model: HomeLbmModel,
        position: tuple[float, float, float] = (4.0, 4.0, 1.5),
        density: float = 1.0,
    ) -> tuple[RigidDomain, int]:
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=position, label="lubricated_sphere")
        builder.add_shape_sphere(
            body,
            radius=1.0,
            cfg=ShapeConfig(density=density, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=model._device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        return rigid, body

    @staticmethod
    def _model(periodic: tuple[bool, bool, bool] = (False, False, False)) -> HomeLbmModel:
        return HomeLbmModel(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=1.0,
            time_step=0.1,
            reference_density=1.0,
            kinematic_viscosity=0.02,
            periodic=periodic,
            device="cuda:0" if wp.is_cuda_available() else "cpu",
        )

    def test_matches_ten_cate_cutoff_correction(self) -> None:
        model = self._model()
        rigid, body = self._make_sphere(model)
        rigid.state.body_qd.assign([[0.0, 0.0, -0.1, 0.0, 0.0, 0.0]])
        rigid.state.body_f.zero_()
        lubrication = SphereWallLubrication(
            model,
            rigid,
            SphereWallLubricationConfig(cutoff_cells=1.0, contact_gap_cells=0.05),
        )

        lubrication.add_to_rigid_forces(
            rigid.state.body_q,
            rigid.state.body_qd,
            rigid.state.body_f,
        )

        expected_force = -6.0 * math.pi * 0.02 * 1.0**2 * (1.0 / 0.5 - 1.0) * -0.1
        np.testing.assert_allclose(
            rigid.state.body_f.numpy()[body, :3],
            [0.0, 0.0, expected_force],
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        self.assertEqual(int(lubrication.active_wall_count.numpy()[body]), 1)
        self.assertAlmostEqual(float(lubrication.minimum_gap.numpy()[body]), 0.5)

    def test_periodic_axis_has_no_lubrication_wall(self) -> None:
        model = self._model(periodic=(False, False, True))
        rigid, body = self._make_sphere(model)
        rigid.state.body_qd.assign([[0.0, 0.0, -0.1, 0.0, 0.0, 0.0]])
        rigid.state.body_f.zero_()
        lubrication = SphereWallLubrication(
            model,
            rigid,
            SphereWallLubricationConfig(),
        )

        lubrication.add_to_rigid_forces(
            rigid.state.body_q,
            rigid.state.body_qd,
            rigid.state.body_f,
        )

        np.testing.assert_array_equal(rigid.state.body_f.numpy()[body], 0.0)
        self.assertEqual(int(lubrication.active_wall_count.numpy()[body]), 0)

    def test_static_pressure_baseline_is_replaced_by_archimedes_buoyancy(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=1.0,
            time_step=0.1,
            reference_density=1.0,
            kinematic_viscosity=0.02,
            body_acceleration=(0.0, 0.0, -0.1),
            periodic=(False, False, False),
            device="cuda:0" if wp.is_cuda_available() else "cpu",
        )
        rigid, body = self._make_sphere(model, position=(4.0, 4.0, 4.0))
        rigid.state.body_qd.zero_()
        rigid.state.body_f.zero_()
        lubrication = SphereWallLubrication(
            model,
            rigid,
            SphereWallLubricationConfig(),
        )

        lubrication.add_to_rigid_forces(
            rigid.state.body_q,
            rigid.state.body_qd,
            rigid.state.body_f,
        )

        expected_buoyancy = 4.0 * math.pi * 0.1 / 3.0
        np.testing.assert_allclose(
            rigid.state.body_f.numpy()[body, :3],
            [0.0, 0.0, expected_buoyancy],
            rtol=2.0e-6,
            atol=1.0e-7,
        )

    def test_roughness_contact_stops_inward_normal_motion(self) -> None:
        model = self._model()
        rigid, body = self._make_sphere(model, position=(4.0, 4.0, 1.02))
        rigid.state.body_qd.assign([[0.03, 0.0, -0.1, 0.0, 0.0, 0.0]])
        lubrication = SphereWallLubrication(
            model,
            rigid,
            SphereWallLubricationConfig(cutoff_cells=1.0, contact_gap_cells=0.05),
        )

        lubrication.project_contacts(rigid.state.body_q, rigid.state.body_qd)

        self.assertAlmostEqual(float(rigid.state.body_q.numpy()[body, 2]), 1.05, places=6)
        np.testing.assert_allclose(
            rigid.state.body_qd.numpy()[body, :3],
            [0.03, 0.0, 0.0],
            atol=1.0e-7,
        )
        self.assertEqual(int(lubrication.contact_count.numpy()[body]), 1)

    def test_non_sphere_geometry_is_rejected(self) -> None:
        model = self._model()
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(4.0, 4.0, 1.5), label="box")
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        rigid_model = builder.finalize(device=model._device)
        rigid = RigidDomain(rigid_model)

        with self.assertRaisesRegex(ValueError, "exactly one sphere"):
            SphereWallLubrication(model, rigid, SphereWallLubricationConfig())

    def test_unstable_explicit_lubrication_is_rejected(self) -> None:
        model = self._model()
        rigid, _ = self._make_sphere(model, density=1.0e-3)

        with self.assertRaisesRegex(ValueError, "explicitly unstable"):
            SphereWallLubrication(
                model,
                rigid,
                SphereWallLubricationConfig(cutoff_cells=1.0, contact_gap_cells=0.01),
            )

    def test_coupling_restores_external_force_ownership(self) -> None:
        model = self._model()
        fluid = HomeLbmDomain(model)
        fluid.create_state()
        fluid.solver.initialize_uniform_lattice(fluid.state)
        rigid, body = self._make_sphere(model)
        rigid.state.body_qd.assign([[0.0, 0.0, -0.02, 0.0, 0.0, 0.0]])
        external_wrench = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rigid.state.body_f.assign([external_wrench])
        coupling = HomeLbmRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            lubrication=SphereWallLubricationConfig(
                cutoff_cells=1.0,
                contact_gap_cells=0.05,
            ),
        )

        coupling.step()

        assert coupling.lubrication is not None
        self.assertGreater(float(coupling.lubrication.correction_force.numpy()[body, 2]), 0.0)
        diagnostics = coupling.fluid_domain.solver.collect_diagnostics(fluid.state)
        self.assertGreater(diagnostics.wall_confined_interpolation_count, 0)
        np.testing.assert_array_equal(rigid.state.body_f.numpy()[body], external_wrench)


if __name__ == "__main__":
    unittest.main()
