# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for FSLBM / VOF sharp free-surface on D3Q19 LBM."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.phases import vof_plic


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

    def test_plic_curvature_sphere(self) -> None:
        """PLIC κ on a spherical drop should be near 1/R (mean curvature)."""
        n = 48
        R = 10.0
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            tau=0.6,
            G=0.0,
            phase_mode="vof_sharp",
            vof_rho_gas=1.0,
            vof_gamma=1.0e-3,
            vof_epsilon=1.0e-4,
        )
        domain = LbmDomain(model)
        domain.create_state()
        state = domain.state
        phi = np.zeros((n, n, n), dtype=np.float32)
        ctype = np.zeros((n, n, n), dtype=np.int32)
        cx = cy = cz = (n - 1) * 0.5
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    r = math.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                    if r < R - 0.5:
                        phi[i, j, k] = 1.0
                        ctype[i, j, k] = 2
                    elif r < R + 0.5:
                        # Smooth fill near the sphere surface.
                        phi[i, j, k] = float(np.clip(0.5 - (r - R), 0.0, 1.0))
                        ctype[i, j, k] = 1
                    else:
                        phi[i, j, k] = 0.0
                        ctype[i, j, k] = 0
        state.phi.assign(phi)
        state.cell_type.assign(ctype)

        vof = domain.solver._vof_sharp
        px, py, pz = model._periodic_ints
        wp.launch(
            vof_plic.vof_compute_kappa_kernel,
            dim=(n, n, n),
            inputs=[
                state.phi,
                state.cell_type,
                state.solid_phi,
                vof._kappa,
                int(px),
                int(py),
                int(pz),
                n,
                n,
                n,
            ],
        )
        wp.synchronize_device(model._device)
        kappa = vof._kappa.numpy()
        mask = ctype == 1
        k_vals = kappa[mask]
        self.assertGreater(k_vals.size, 20)
        k_mean = float(np.mean(np.abs(k_vals)))
        # Mean curvature of sphere ≈ 1/R; allow coarse-grid error.
        self.assertGreater(k_mean, 0.04)
        self.assertLess(k_mean, 0.25)

    def test_vof_with_surface_tension_smoke(self) -> None:
        n = 24
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            tau=0.6,
            G=0.0,
            phase_mode="vof_sharp",
            vof_rho_gas=1.0,
            vof_epsilon=1.0e-4,
            vof_gamma=5.0e-4,
            lambda_trt=0.03,
            initial_density=1.0,
            gravity_z=-0.0003,
        )
        domain = LbmDomain(model)
        domain.create_state()
        state = domain.state
        domain.solver._vof_sharp.seed_dam_break_column(
            state, dam_x=n // 4, fill_z=n // 2, rho_liquid=1.0
        )
        out = domain._state_out
        for name in ("f", "density", "phi", "cell_type", "solid_phi", "solid_body_id"):
            wp.copy(getattr(out, name), getattr(state, name))

        for _ in range(30):
            domain.step(1.0)
        wp.synchronize_device(model._device)
        phi = domain.state.phi.numpy()
        self.assertTrue(np.isfinite(phi).all())
        self.assertGreater(float(phi.sum()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
