# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for FSLBM / VOF sharp free-surface on D3Q19 LBM."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel


class TestLbmVofSharp(unittest.TestCase):
    def test_vof_dambreak_smoke(self) -> None:
        n = 24
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            tau=0.6,
            G=0.0,
            phase_mode="vof_sharp",
            vof_rho_gas=1.0,
            vof_epsilon=1.0e-4,
            lambda_trt=0.03,
            initial_density=1.0,
            gravity_z=-0.0003,
        )
        domain = LbmDomain(model)
        domain.create_state()
        state = domain.state
        domain.solver._vof_sharp.seed_dam_break_column(
            state,
            dam_x=n // 4,
            fill_z=n // 2,
            rho_liquid=1.0,
        )
        out = domain._state_out
        for name in ("f", "density", "phi", "cell_type", "solid_phi", "solid_body_id"):
            wp.copy(getattr(out, name), getattr(state, name))

        phi0 = float(state.phi.numpy().sum())
        self.assertGreater(phi0, 0.0)

        for _ in range(40):
            domain.step(1.0)
        wp.synchronize_device(model._device)

        state = domain.state
        phi = state.phi.numpy()
        ctype = state.cell_type.numpy()
        rho = state.density.numpy()
        ux = state.velocity_x.numpy()
        uy = state.velocity_y.numpy()
        uz = state.velocity_z.numpy()

        self.assertTrue(np.isfinite(phi).all())
        self.assertTrue(np.isfinite(rho).all())
        self.assertTrue(np.isfinite(ux).all())
        self.assertTrue(np.isfinite(uy).all())
        self.assertTrue(np.isfinite(uz).all())

        phi_sum = float(phi.sum())
        self.assertGreater(phi_sum, 0.0)
        # Volume should stay close to the seeded column (projection + Körner θ).
        self.assertLess(abs(phi_sum - phi0) / phi0, 0.05)

        n_liquid = int((ctype == 2).sum())
        n_interface = int((ctype == 1).sum())
        self.assertGreater(n_liquid, 0)
        self.assertGreater(n_interface, 0)

        wet = phi > 0.5
        self.assertTrue(wet.any())
        self.assertTrue(math.isfinite(float(rho[wet].mean())))

    def test_phase_mode_validation(self) -> None:
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(8, 8, 8),
                fluid_grid_cell_size=0.1,
                tau=0.6,
                phase_mode="vof_sharp",
                G=-5.0,
            )
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(8, 8, 8),
                fluid_grid_cell_size=0.1,
                tau=0.6,
                phase_mode="shan_chen",
                G=0.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
