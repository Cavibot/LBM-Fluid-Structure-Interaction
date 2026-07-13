# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for FSLBM / VOF sharp free-surface on distribution LBM."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.phases import vof_plic


class TestLbmVofSharp(unittest.TestCase):
    def _run_dambreak_smoke(self, lattice: str) -> None:
        n = 24
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            lattice=lattice,
            tau=0.6,
            G=0.0,
            phase_mode="vof_sharp",
            vof_rho_gas=1.0,
            vof_epsilon=1.0e-4,
            lambda_trt=0.03,
            initial_density=1.0,
            gravity_z=-0.0003,
        )
        self.assertEqual(model.num_dirs, 27 if lattice == "D3Q27" else 19)
        domain = LbmDomain(model)
        domain.create_state()
        state = domain.state
        self.assertEqual(int(state.f.shape[0]), model.num_dirs * n * n * n)
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
        self.assertLess(abs(phi_sum - phi0) / phi0, 0.05)

        n_liquid = int((ctype == 2).sum())
        n_interface = int((ctype == 1).sum())
        self.assertGreater(n_liquid, 0)
        self.assertGreater(n_interface, 0)

        wet = phi > 0.5
        self.assertTrue(wet.any())
        self.assertTrue(math.isfinite(float(rho[wet].mean())))

    def test_vof_dambreak_smoke(self) -> None:
        self._run_dambreak_smoke("D3Q19")

    def test_vof_dambreak_smoke_d3q27(self) -> None:
        self._run_dambreak_smoke("D3Q27")

    def test_vof_home_fs_filter_dambreak_smoke(self) -> None:
        """H3: HOME-FREE filtered ``\\bar f`` FS BC (default on)."""
        n = 24
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            lattice="D3Q27",
            tau=0.6,
            G=0.0,
            phase_mode="vof_sharp",
            vof_rho_gas=1.0,
            vof_epsilon=1.0e-4,
            vof_home_fs_filter=True,
            lambda_trt=0.03,
            initial_density=1.0,
            gravity_z=-0.0003,
        )
        self.assertTrue(model.vof_home_fs_filter)
        domain = LbmDomain(model)
        domain.create_state()
        state = domain.state
        domain.solver._vof_sharp.seed_dam_break_column(
            state, dam_x=n // 4, fill_z=n // 2, rho_liquid=1.0,
        )
        out = domain._state_out
        for name in ("f", "density", "phi", "cell_type", "solid_phi", "solid_body_id"):
            wp.copy(getattr(out, name), getattr(state, name))
        phi0 = float(state.phi.numpy().sum())
        for _ in range(40):
            domain.step(1.0)
        wp.synchronize_device(model._device)
        state = domain.state
        phi = state.phi.numpy()
        self.assertTrue(np.isfinite(phi).all())
        self.assertTrue(np.isfinite(state.density.numpy()).all())
        self.assertLess(abs(float(phi.sum()) - phi0) / phi0, 0.05)
        self.assertGreater(int((state.cell_type.numpy() == 1).sum()), 0)

    def test_fs_bar_f_differs_from_raw_when_neq(self) -> None:
        """With non-eq S, Hermite ``\\bar f_ī`` ≠ classic opposite feq."""
        from wanphys._src.fluid.fluid_grid.lbm.core.hermite import (
            reconstruct_f_i_numpy,
        )

        rho_g, rho_c = 1.0, 1.0
        ux, uy, uz = 0.05, 0.0, 0.0
        # Non-equilibrium stress
        sxx, syy, szz, sxy, sxz, syz = 0.02, -0.01, 0.0, 0.0, 0.0, 0.0
        w = 1.0 / 18.0
        cx, cy, cz = 1, 0, 0
        # Classic: feq_i + feq_ī - f_ī  with f_ī = feq_ī(ρ_c,u) → feq_i(ρ_g)
        # HOME: feq_i + feq_ī - bar_f_ī
        cu = cx * ux + cy * uy + cz * uz
        u2 = ux * ux + uy * uy + uz * uz
        feq_i = rho_g * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        feq_opp = rho_g * w * (1.0 - 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        bar_opp = reconstruct_f_i_numpy(
            rho_c, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz,
            float(-cx), float(-cy), float(-cz), w,
        )
        classic_f_opp = rho_c * w * (1.0 - 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        f_home = feq_i + feq_opp - bar_opp
        f_classic = feq_i + feq_opp - classic_f_opp
        self.assertGreater(abs(f_home - f_classic), 1e-6)

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
        self.assertGreater(k_mean, 0.04)
        self.assertLess(k_mean, 0.25)

    def test_vof_with_surface_tension_smoke(self) -> None:
        n = 24
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            lattice="D3Q27",
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

    def test_single_phase_d3q27_smoke(self) -> None:
        n = 16
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.05,
            lattice="D3Q27",
            tau=0.6,
            G=0.0,
            phase_mode="none",
            gravity_z=-0.0001,
            lambda_trt=0.03,
            initial_density=1.0,
        )
        domain = LbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state, rho0=1.0)
        out = domain._state_out
        wp.copy(out.f, domain.state.f)
        for _ in range(20):
            domain.step(1.0)
        wp.synchronize_device(model._device)
        rho = domain.state.density.numpy()
        self.assertTrue(np.isfinite(rho).all())
        self.assertAlmostEqual(float(rho.mean()), 1.0, delta=0.05)

    def test_zou_he_d3q27_inlet_smoke(self) -> None:
        """D3Q27 Zou-He inlet + outflow stays finite with mild drive."""
        n = 24
        u_in = 0.02
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.05,
            lattice="D3Q27",
            tau=0.8,
            G=0.0,
            phase_mode="none",
            lambda_trt=0.0,
            initial_density=1.0,
            bc_types=(1, 2, 0, 0, 0, 0),
            bc_velocity=(
                (u_in, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        domain = LbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state, rho0=1.0)
        wp.copy(domain._state_out.f, domain.state.f)
        for _ in range(50):
            domain.step(1.0)
        wp.synchronize_device(model._device)
        state = domain.state
        ux = state.velocity_x.numpy()
        rho = state.density.numpy()
        self.assertTrue(np.isfinite(ux).all())
        self.assertTrue(np.isfinite(rho).all())
        # Mid-plane should feel some positive streamwise flow.
        mid = ux[n // 2, 1 : n - 1, 1 : n - 1]
        self.assertGreater(float(mid.mean()), 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
