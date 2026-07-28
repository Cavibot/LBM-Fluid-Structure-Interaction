# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P1 tests: HOME-FREE VOF IC/BC factory, cylinder SDF, feedback mode enum."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling, LbmFeedbackMode
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    CELL_INTERFACE,
    CELL_LIQUID,
    HomeDomainBC,
    HomeFaceKind,
    apply_home_domain_bc_to_model,
    make_home_vof_model,
    seed_droplet,
    seed_pool,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import (
    BC_BOUNCE_BACK,
    BC_PERIODIC,
    BC_VELOCITY_INLET,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder


class TestHomeVofIcFactory(unittest.TestCase):
    def test_seed_pool_has_liquid_and_interface(self) -> None:
        st = seed_pool((12, 10, 12), fill_z=5, rho_liquid=1.0)
        self.assertGreater(int((st.cell_type == CELL_LIQUID).sum()), 0)
        self.assertGreater(int((st.cell_type == CELL_INTERFACE).sum()), 0)
        self.assertTrue(np.all(st.phi[st.cell_type == CELL_LIQUID] == 1.0))
        self.assertTrue(np.allclose(st.phi[st.cell_type == CELL_INTERFACE], 0.5))

    def test_seed_droplet_centered(self) -> None:
        n = 16
        c = (n * 0.5, n * 0.5, n * 0.5)
        st = seed_droplet((n, n, n), center=c, radius=3.0)
        self.assertGreater(int((st.cell_type == CELL_LIQUID).sum()), 5)
        self.assertGreater(int((st.cell_type == CELL_INTERFACE).sum()), 0)
        # Droplet should not fill the whole box.
        self.assertLess(int((st.cell_type != 0).sum()), n * n * n // 2)


class TestHomeVofBcPresets(unittest.TestCase):
    def test_open_top_zmax_is_zou_he(self) -> None:
        bc = HomeDomainBC.open_top()
        self.assertEqual(bc.zmax.kind, HomeFaceKind.ZOU_HE)
        self.assertEqual(bc.zmin.kind, HomeFaceKind.WALL)
        self.assertEqual(bc.xmin.kind, HomeFaceKind.WALL)

    def test_channel_x_inlet_outlet(self) -> None:
        bc = HomeDomainBC.channel_x(0.05, periodic_y=True, periodic_z=False)
        self.assertEqual(bc.xmin.kind, HomeFaceKind.ZOU_HE)
        self.assertEqual(bc.xmax.kind, HomeFaceKind.ZOU_HE)
        self.assertEqual(bc.ymin.kind, HomeFaceKind.PERIODIC)
        self.assertEqual(bc.zmin.kind, HomeFaceKind.WALL)

    def test_apply_home_domain_bc_to_model(self) -> None:
        model = make_home_vof_model(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.02,
        )
        apply_home_domain_bc_to_model(model, HomeDomainBC.open_top())
        self.assertEqual(model.bc_types[5], BC_VELOCITY_INLET)  # zmax
        self.assertEqual(model.bc_types[4], BC_BOUNCE_BACK)  # zmin
        apply_home_domain_bc_to_model(
            model, HomeDomainBC.channel_x(0.1, periodic_y=True, periodic_z=True)
        )
        self.assertEqual(model.bc_types[0], BC_VELOCITY_INLET)
        self.assertTrue(model.bc_periodic[1])
        self.assertTrue(model.bc_periodic[2])
        self.assertEqual(model.bc_types[2], BC_PERIODIC)


class TestLbmFeedbackMode(unittest.TestCase):
    def test_feedback_mode_none_and_enum(self) -> None:
        model = LbmModel(
            fluid_grid_res=(10, 10, 10),
            fluid_grid_cell_size=0.1,
            tau=0.55,
            phase_mode="none",
            lbm_backend="dist",
        )
        fluid = LbmDomain(model)
        fluid.create_state()
        builder = RigidModelBuilder(gravity=0.0)
        bid = builder.add_body(position=(0.5, 0.5, 0.5), label="s")
        builder.add_shape_sphere(bid, radius=0.2)
        rigid = RigidDomain(builder.finalize(device=model._device))
        rigid.create_state()
        coupling = GridLbmRigidCoupling(fluid, rigid)
        coupling.add_body_sphere(bid, radius=0.2)
        coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
        coupling.set_feedback_mode(LbmFeedbackMode.NONE)
        self.assertEqual(coupling.feedback_mode, "none")
        coupling.set_feedback_mode("approx")
        self.assertEqual(coupling.feedback_mode, LbmFeedbackMode.APPROX.value)
        with self.assertRaises(ValueError):
            coupling.set_feedback_mode("not_a_mode")


class TestCylinderSdfRaster(unittest.TestCase):
    def test_add_body_cylinder_marks_solid_phi(self) -> None:
        try:
            wp.init()
            if wp.get_cuda_device_count() <= 0:
                raise RuntimeError("no CUDA")
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"CUDA unavailable: {exc}")

        nx = 24
        dh = 0.05
        model = LbmModel(
            fluid_grid_res=(nx, nx, nx),
            fluid_grid_cell_size=dh,
            tau=0.55,
            phase_mode="none",
            lbm_backend="dist",
        )
        fluid = LbmDomain(model)
        fluid.create_state()
        builder = RigidModelBuilder(gravity=0.0)
        center = (0.5 * nx * dh, 0.5 * nx * dh, 0.5 * nx * dh)
        bid = builder.add_body(position=center, label="cyl")
        # Collision shape can be capsule; LBM coupling uses cylinder SDF.
        builder.add_shape_capsule(bid, radius=0.15, half_height=0.25)
        rigid = RigidDomain(builder.finalize(device=model._device))
        rigid.create_state()
        coupling = GridLbmRigidCoupling(fluid, rigid)
        coupling.add_body_cylinder(bid, radius=0.15, half_height=0.25)
        coupling.set_rigid_dynamics_enabled(False)
        coupling.set_two_way_feedback_enabled(False)
        coupling.set_feedback_mode(LbmFeedbackMode.NONE)
        coupling.step(1.0 / 60.0)
        wp.synchronize_device(model._device)
        solid = fluid.state.solid_phi.numpy()
        self.assertTrue(np.any(solid < 0.0))
        # Cylinder should occupy fewer cells than a same-radius sphere of
        # equivalent bounding sphere — just check a connected solid core.
        self.assertGreater(int((solid < 0.0).sum()), 10)


if __name__ == "__main__":
    unittest.main()
