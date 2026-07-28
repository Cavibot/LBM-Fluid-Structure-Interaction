# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P2 tests: home reconstructed-link ME, φ-volume plugin, narrowband, CUDA graph flag."""

from __future__ import annotations

import math
import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling, LbmFeedbackMode
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    make_home_vof_model,
    seed_pool,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.phi_volume_buoyancy_warp import (
    fibonacci_shell_offsets,
)
from wanphys.examples.lbm._home_vof_phi_volume_buoyancy import (
    PhiVolumeBuoyancyConfig,
    PhiVolumeBuoyancyPlugin,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder


class TestHomeReconstructedLinkMe(unittest.TestCase):
    def test_momentum_exchange_runs_on_home_fp32(self) -> None:
        model = make_home_vof_model(
            fluid_grid_res=(16, 16, 16),
            fluid_grid_cell_size=0.05,
            gravity_z=-1.0e-4,
        )
        fluid = LbmDomain(model)
        fluid.create_state()
        home = fluid.solver._home_fp32
        assert home is not None
        home.seed_host_state(fluid.state, seed_pool((16, 16, 16), fill_z=8))

        builder = RigidModelBuilder(gravity=0.0)
        bid = builder.add_body(position=(0.4, 0.4, 0.25), label="s")
        builder.add_shape_sphere(bid, radius=0.12)
        rigid = RigidDomain(builder.finalize(device=model._device))
        rigid.create_state()

        coupling = GridLbmRigidCoupling(fluid, rigid)
        coupling.add_body_sphere(bid, radius=0.12)
        coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
        coupling.set_feedback_mode(LbmFeedbackMode.MOMENTUM_EXCHANGE)
        coupling.set_rigid_dynamics_enabled(False)

        for _ in range(3):
            coupling.step(0.01)

        bf = np.asarray(rigid.state.body_f.numpy(), dtype=np.float64)
        self.assertEqual(bf.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(bf)))


class TestPhiVolumeBuoyancyPlugin(unittest.TestCase):
    def test_fibonacci_offsets_and_apply(self) -> None:
        offs = fibonacci_shell_offsets(12, radii=(1.05,))
        self.assertEqual(len(offs), 12)

        model = make_home_vof_model(
            fluid_grid_res=(20, 20, 20),
            fluid_grid_cell_size=0.05,
        )
        fluid = LbmDomain(model)
        fluid.create_state()
        home = fluid.solver._home_fp32
        assert home is not None
        home.seed_host_state(fluid.state, seed_pool((20, 20, 20), fill_z=10))
        fluid.step(0.01)
        g = home._gpu
        assert g is not None

        r = 0.15
        vol = (4.0 / 3.0) * math.pi * r**3
        plugin = PhiVolumeBuoyancyPlugin(
            device=str(model._device),
            body_ids=(0,),
            radius=r,
            volume=vol,
            rho_liquid=1.0,
            gravity_abs=9.81,
            dh=float(model.dh),
            nx=20,
            ny=20,
            nz=20,
            config=PhiVolumeBuoyancyConfig(n_dirs=16),
        )

        builder = RigidModelBuilder(gravity=0.0)
        bid = builder.add_body(position=(0.5, 0.5, 0.35), label="s")
        builder.add_shape_sphere(bid, radius=r)
        rigid = RigidDomain(builder.finalize(device=model._device))
        rigid.create_state()

        sub = plugin.apply(
            phi=g.phi,
            cell=g.cell_type,
            solid=g.solid_phi,
            body_q=rigid.state.body_q,
            body_f_apply=rigid.state.apply_body_forces,
            sync_submerged=True,
        )
        self.assertIn(0, sub)
        self.assertGreaterEqual(sub[0], 0.0)
        self.assertLessEqual(sub[0], 1.0)


class TestSolidNarrowbandAndCudaGraphFlag(unittest.TestCase):
    def test_set_solid_narrowband(self) -> None:
        model = make_home_vof_model(
            fluid_grid_res=(12, 12, 12),
            fluid_grid_cell_size=0.05,
        )
        fluid = LbmDomain(model)
        fluid.create_state()
        builder = RigidModelBuilder(gravity=0.0)
        bid = builder.add_body(position=(0.3, 0.3, 0.3), label="s")
        builder.add_shape_sphere(bid, radius=0.1)
        rigid = RigidDomain(builder.finalize(device=model._device))
        rigid.create_state()
        coupling = GridLbmRigidCoupling(fluid, rigid)
        coupling.add_body_sphere(bid, radius=0.1)
        coupling.set_solid_narrowband(3.0)
        self.assertEqual(coupling._solid_narrowband_cells, 3.0)
        coupling.set_rigid_dynamics_enabled(False)
        coupling.step(0.01)
        phi = fluid.state.solid_phi.numpy()
        self.assertTrue(np.any(phi < 0.0))

    def test_cuda_graph_flag_smoke(self) -> None:
        model = make_home_vof_model(
            fluid_grid_res=(12, 12, 12),
            fluid_grid_cell_size=0.05,
            vof_home_cuda_graph=True,
            vof_seal_fg=True,
            vof_gamma=0.0,
        )
        self.assertTrue(model.vof_home_cuda_graph)
        fluid = LbmDomain(model)
        fluid.create_state()
        home = fluid.solver._home_fp32
        assert home is not None
        home.seed_host_state(fluid.state, seed_pool((12, 12, 12), fill_z=5))
        for _ in range(4):
            fluid.step(0.01)
        g = home._gpu
        assert g is not None
        mass = float(np.sum(g.mass.numpy()))
        self.assertGreater(mass, 0.0)


if __name__ == "__main__":
    unittest.main()
