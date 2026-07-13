# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HOME-LBM H1: moment collide + reconstruct-stream."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    NEXT_INCREMENT,
    collide_moments_numpy,
    make_uniform_equilibrium,
    step_periodic_numpy,
    step_periodic_warp,
)
from wanphys._src.fluid.fluid_grid.lbm.core.hermite import equilibrium_s_from_u


class TestHomeCollideH1(unittest.TestCase):
    def test_equilibrium_is_fixed_point(self) -> None:
        """At S*=u⊗u and F=0, collision leaves (ρ,u,S) unchanged."""
        rho, ux, uy, uz = 1.1, 0.04, -0.02, 0.03
        sxx, syy, szz, sxy, sxz, syz = equilibrium_s_from_u(ux, uy, uz)
        m = collide_moments_numpy(
            rho, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz, tau=0.7,
        )
        self.assertAlmostEqual(m.rho, rho, places=12)
        self.assertAlmostEqual(m.ux, ux, places=12)
        self.assertAlmostEqual(m.uy, uy, places=12)
        self.assertAlmostEqual(m.uz, uz, places=12)
        self.assertAlmostEqual(m.sxx, sxx, places=10)
        self.assertAlmostEqual(m.syy, syy, places=10)
        self.assertAlmostEqual(m.szz, szz, places=10)
        self.assertAlmostEqual(m.sxy, sxy, places=10)
        self.assertAlmostEqual(m.sxz, sxz, places=10)
        self.assertAlmostEqual(m.syz, syz, places=10)

    def test_force_updates_velocity(self) -> None:
        """Guo half-force: u ← u* + F/(2ρ)."""
        rho = 1.0
        fx = 0.001
        m = collide_moments_numpy(
            rho, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            tau=0.6, fx=fx,
        )
        self.assertAlmostEqual(m.ux, 0.5 * fx / rho, places=12)
        self.assertAlmostEqual(m.uy, 0.0, places=12)


class TestHomeStepH1(unittest.TestCase):
    def test_next_increment_points_h7(self) -> None:
        self.assertIn("H7", NEXT_INCREMENT)

    def test_quiescent_periodic_stable(self) -> None:
        """Uniform rest fluid stays at equilibrium under periodic HOME steps."""
        field = make_uniform_equilibrium((4, 4, 4), rho0=1.0)
        out = field
        for _ in range(5):
            out = step_periodic_numpy(out, lattice="D3Q27", tau=0.6)
        self.assertTrue(np.allclose(out.rho, 1.0, atol=1e-10))
        self.assertTrue(np.allclose(out.ux, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(out.uy, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(out.uz, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(out.sxx, 0.0, atol=1e-10))

    def test_mass_conserved_with_velocity_pulse(self) -> None:
        """Periodic box: total ρ conserved after a localized u pulse."""
        field = make_uniform_equilibrium((6, 6, 6), rho0=1.0)
        field.ux[2, 2, 2] = 0.05
        field.sxx[2, 2, 2] = 0.05 * 0.05
        mass0 = float(np.sum(field.rho))
        out = field
        for _ in range(10):
            out = step_periodic_numpy(out, lattice="D3Q27", tau=0.8)
        mass1 = float(np.sum(out.rho))
        self.assertAlmostEqual(mass1, mass0, places=8)
        self.assertFalse(np.any(np.isnan(out.rho)))
        self.assertFalse(np.any(np.isnan(out.ux)))

    def test_uniform_flow_advection_periodic(self) -> None:
        """Uniform (ρ,u) remains uniform (periodic + equilibrium S)."""
        field = make_uniform_equilibrium(
            (5, 5, 5), rho0=1.0, ux0=0.02, uy0=-0.01, uz0=0.0,
        )
        out = step_periodic_numpy(field, lattice="D3Q27", tau=0.55)
        self.assertTrue(np.allclose(out.rho, 1.0, atol=1e-9))
        self.assertTrue(np.allclose(out.ux, 0.02, atol=1e-9))
        self.assertTrue(np.allclose(out.uy, -0.01, atol=1e-9))

    def test_warp_matches_numpy_quiescent(self) -> None:
        """Warp kernel agrees with numpy on a small rest fluid (fp32 tol)."""
        field = make_uniform_equilibrium((3, 3, 3), rho0=1.0)
        out_np = step_periodic_numpy(field, lattice="D3Q27", tau=0.6)
        try:
            out_wp = step_periodic_warp(field, lattice="D3Q27", tau=0.6)
        except Exception as exc:  # noqa: BLE001 — CUDA may be unavailable
            self.skipTest(f"warp device unavailable: {exc}")
            return
        self.assertTrue(np.allclose(out_wp.rho, out_np.rho, atol=1e-5))
        self.assertTrue(np.allclose(out_wp.ux, out_np.ux, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
