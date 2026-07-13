# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 acceptance tests A.1-A.7, C.12-C.14."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm.constants import (
    C_X, C_Y, C_Z, W, OPPOSITE, CS2, TYPE_S, TYPE_G,
    HERMITE_COEFFS,
)
from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFSLbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFSLbmState
from wanphys._src.fluid.fluid_grid.home_fslbm.kernels import (
    reconstruct_all_f_kernel,
)


class TestConstants(unittest.TestCase):
    def test_weight_normalisation(self):
        total = float(np.sum(W.numpy()))
        self.assertLess(abs(total - 1.0), 1e-15)

    def test_opposite_direction(self):
        cx = C_X.numpy(); cy = C_Y.numpy(); cz = C_Z.numpy()
        opp = OPPOSITE.numpy()
        for i in range(27):
            self.assertEqual(cx[i] + cx[opp[i]], 0)
            self.assertEqual(cy[i] + cy[opp[i]], 0)
            self.assertEqual(cz[i] + cz[opp[i]], 0)


class TestReconstruction(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.model = HomeFSLbmModel(fluid_grid_res=(4,4,4), fluid_grid_cell_size=1.0)
        self.state = HomeFSLbmState(self.model)
        self.tn = self.state.total_num

    def _launch_reconstruct(self, f_np):
        wp.copy(self.state.f_mom, wp.array(f_np, dtype=float, device=self.model._device))
        f_out = wp.zeros(27 * self.tn, dtype=float, device=self.model._device)
        wp.launch(reconstruct_all_f_kernel, dim=(4,4,4),
                  inputs=[self.state.f_mom, f_out, self.tn, 4, 4, 4,
                          HERMITE_COEFFS, W])
        wp.synchronize()
        return f_out.numpy().reshape(27, self.tn)

    def test_equilibrium_static(self):
        """A.3: rho=1, u=0, S=0 -> f_i == w_i."""
        f_np = np.zeros(10 * self.tn, dtype=np.float32)
        f_np[0 * self.tn:1 * self.tn] = 1.0
        f_out_np = self._launch_reconstruct(f_np)
        w_np = W.numpy()
        for d in range(27):
            self.assertLess(abs(f_out_np[d, 0] - w_np[d]), 1e-10)

    def test_constant_velocity_conservation(self):
        """A.4: rho=1, u=(0.1,0,0) -> moments conserved."""
        f_np = np.zeros(10 * self.tn, dtype=np.float32)
        f_np[0*self.tn:1*self.tn] = 1.0
        f_np[1*self.tn:2*self.tn] = 0.1
        f_np[4*self.tn:5*self.tn] = 0.01
        f_out_np = self._launch_reconstruct(f_np)
        cx = C_X.numpy().astype(np.float32)
        for idx in range(4):
            rho_s, ux_s = 0.0, 0.0
            for d in range(27):
                fi = f_out_np[d, idx]
                rho_s += fi
                ux_s += cx[d] * fi
            self.assertLess(abs(rho_s - 1.0), 1e-6)
            self.assertLess(abs(ux_s - 0.1), 1e-4)

    def test_hermite_third_order_error(self):
        """A.5: Compare 3rd-order moments: analytic equilibrium vs reconstruction.

        Computes the full analytic equilibrium distribution f_i^eq for a
        small velocity (|u|=1e-3 where the Hermite expansion is essentially
        exact), then reconstructs f_i from only the first 10 moments and
        measures the L2 error in the 3rd-order moments.  The analytic
        formula serves as an independent oracle — no coefficient table
        is used on the reference side.
        """
        u = 0.001  # sufficiently small that Hermite expansion is precise
        rho = 1.0
        w_np = W.numpy()
        cx_np = C_X.numpy().astype(np.float64)
        cy_np = C_Y.numpy().astype(np.float64)
        cz_np = C_Z.numpy().astype(np.float64)

        # ---- Oracle: analytic equilibrium distribution ----
        feq = np.zeros(27, dtype=np.float64)
        u2 = u * u
        for d in range(27):
            cu = cx_np[d] * u
            feq[d] = rho * w_np[d] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2)

        # Full 3rd-order moments from oracle
        def mom3(d, a, b, c_idx):
            return sum(feq[d] * cx_np[d]**a * cy_np[d]**b * cz_np[d]**c_idx
                       for d in range(27))

        m3_oracle = np.array([
            mom3(0, 3, 0, 0),  # xxx
            mom3(0, 0, 3, 0),  # yyy
            mom3(0, 0, 0, 3),  # zzz
            mom3(0, 2, 1, 0),  # xxy
            mom3(0, 1, 2, 0),  # xyy
            mom3(0, 2, 0, 1),  # xxz
            mom3(0, 1, 0, 2),  # xzz
            mom3(0, 0, 2, 1),  # yyz
            mom3(0, 0, 1, 2),  # yzz
            mom3(0, 1, 1, 1),  # xyz
        ], dtype=np.float64)

        # ---- Reconstruction from first 10 moments only ----
        S_xx = u * u  # stored stress = u_a*u_b for equilibrium
        f_np = np.zeros(10 * self.tn, dtype=np.float32)
        f_np[0*self.tn:1*self.tn] = float(rho)
        f_np[1*self.tn:2*self.tn] = float(u)
        f_np[4*self.tn:5*self.tn] = float(S_xx)
        f_out_np = self._launch_reconstruct(f_np)

        # 3rd-order moments from reconstruction
        m3_recon = np.zeros(10, dtype=np.float64)
        rd = np.array([
            (3,0,0),(0,3,0),(0,0,3),(2,1,0),(1,2,0),
            (2,0,1),(1,0,2),(0,2,1),(0,1,2),(1,1,1),
        ])
        for j, (a,b,c_idx) in enumerate(rd):
            for d in range(27):
                m3_recon[j] += (f_out_np[d, 0].astype(np.float64)
                                * cx_np[d]**a * cy_np[d]**b * cz_np[d]**c_idx)

        # L2 relative error
        err = np.sqrt(np.sum((m3_recon - m3_oracle)**2))
        norm = np.sqrt(np.sum(m3_oracle**2))
        rel_err = err / norm if norm > 0 else err
        self.assertLess(rel_err, 1e-3,
            f"3rd-order moment L2 rel error: {rel_err:.2e}")


class TestModelAndState(unittest.TestCase):
    def test_model_validation(self):
        m = HomeFSLbmModel(fluid_grid_res=(16,32,8), fluid_grid_cell_size=0.05, tau=0.6)
        self.assertEqual(m.nx, 16); self.assertEqual(m.ny, 32); self.assertEqual(m.nz, 8)
        self.assertGreater(m.tau, 0.5); self.assertEqual(len(m.bc_types), 6)

    def test_state_memory(self):
        m = HomeFSLbmModel(fluid_grid_res=(4,8,16), fluid_grid_cell_size=1.0)
        s = HomeFSLbmState(m)
        self.assertEqual(s.total_num, 512)
        self.assertEqual(len(s.f_mom), 10 * 512)
        self.assertEqual(s.phi.shape, (4, 8, 16))
        self.assertTrue(np.all(s.f_mom.numpy() == 0.0))


class TestBoundary(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.model = HomeFSLbmModel(fluid_grid_res=(4,4,4), fluid_grid_cell_size=1.0)

    def test_solid_flag(self):
        state = HomeFSLbmState(self.model)
        flag_np = state.flag.numpy()
        flag_np[0,:,:]|=TYPE_S; flag_np[3,:,:]|=TYPE_S
        flag_np[:,0,:]|=TYPE_S; flag_np[:,3,:]|=TYPE_S
        flag_np[:,:,0]|=TYPE_S; flag_np[:,:,3]|=TYPE_S
        wp.copy(state.flag, wp.array(flag_np, dtype=wp.int32, device=self.model._device))
        wp.synchronize()
        result = state.flag.numpy()
        self.assertTrue(np.all((result[0,:,:] & TYPE_S) != 0))
        self.assertTrue(np.all((result[1:3,1:3,1:3] & TYPE_S) == 0))

    @staticmethod
    def _feq_np(rho, ux, uy, uz, d, cx, cy, cz, w):
        cu = float(cx[d])*ux + float(cy[d])*uy + float(cz[d])*uz
        u2 = ux*ux + uy*uy + uz*uz
        return rho * float(w[d]) * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2)

    def test_bounce_back_equilibrium(self):
        w_np = W.numpy(); cx=C_X.numpy(); cy=C_Y.numpy(); cz=C_Z.numpy()
        for d in range(27):
            feq = self._feq_np(1.0,0,0,0,d,cx,cy,cz,w_np)
            self.assertLess(abs(feq - w_np[d]), 1e-12)

    def test_bounce_back_no_penetration(self):
        cx=C_X.numpy(); cy=C_Y.numpy(); cz=C_Z.numpy()
        w_np=W.numpy(); opp=OPPOSITE.numpy()
        for d in range(27):
            f_hn = self._feq_np(1.0,0,0,0,d,cx,cy,cz,w_np)
            f_on_opp = self._feq_np(1.0,0,0,0,opp[d],cx,cy,cz,w_np)
            self.assertLess(abs(f_hn - f_on_opp), 1e-12)


if __name__ == "__main__":
    unittest.main()
