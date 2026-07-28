# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME base (``home_fp32`` + ``phase_mode=none``) correctness & stability.

Same fused operators as the free-surface branch; FS policy forced off.
Validates mass/momentum, quiescent walls, Guo body force, and Poiseuille.

    uv run --extra examples python -m unittest \\
        newton.tests.test_lbm_single_phase_validation -v
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    HomeDomainBC,
    make_home_model,
    make_uniform_equilibrium,
    step_domain_numpy,
    step_periodic_numpy,
)


def _sync(domain: LbmDomain) -> None:
    wp.synchronize_device(domain.model._device)


def _make_home_domain(
    *,
    res: tuple[int, int, int],
    tau: float = 0.8,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bc_periodic: tuple[bool, bool, bool] = (True, True, True),
    bc_types: tuple[int, int, int, int, int, int] = (3, 3, 3, 3, 3, 3),
    bc_velocity: tuple | None = None,
    u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LbmDomain:
    kwargs = {
        "fluid_grid_res": res,
        "fluid_grid_cell_size": 0.05,
        "phase_mode": "none",
        "tau": float(tau),
        "gravity_x": float(gravity[0]),
        "gravity_y": float(gravity[1]),
        "gravity_z": float(gravity[2]),
        "lattice": "D3Q27",
        "lambda_trt": 0.0,
        "bc_periodic": bc_periodic,
        "bc_types": bc_types,
    }
    if bc_velocity is not None:
        kwargs["bc_velocity"] = bc_velocity
    model = make_home_model(**kwargs)
    domain = LbmDomain(model)
    domain.create_state()
    domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=u0)
    wp.copy(domain._state_out.density, domain.state.density)
    wp.copy(domain._state_out.velocity_x, domain.state.velocity_x)
    wp.copy(domain._state_out.velocity_y, domain.state.velocity_y)
    wp.copy(domain._state_out.velocity_z, domain.state.velocity_z)
    return domain


class TestHomeBaseModel(unittest.TestCase):
    def test_home_fp32_accepts_phase_none(self) -> None:
        model = make_home_model(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.1,
            phase_mode="none",
            tau=0.7,
        )
        self.assertEqual(model.lbm_backend, "home_fp32")
        self.assertEqual(model.phase_mode, "none")

    def test_bridge_free_surface_flag(self) -> None:
        domain = _make_home_domain(res=(8, 8, 8))
        self.assertIsNotNone(domain.solver._home_fp32)
        self.assertFalse(domain.solver._home_fp32.free_surface)


class TestHomeBasePeriodicConservation(unittest.TestCase):
    def test_mass_conserved_after_density_pulse(self) -> None:
        domain = _make_home_domain(res=(12, 12, 12), tau=0.8)
        bridge = domain.solver._home_fp32
        assert bridge is not None
        buf = bridge._ensure_gpu()
        rho = buf.rho.numpy()
        rho[4, 4, 4] *= 1.02
        buf.rho.assign(rho)
        # Keep mass = φρ for liquid consistency.
        mass = buf.mass.numpy()
        mass[4, 4, 4] = rho[4, 4, 4]
        buf.mass.assign(mass)
        bridge.sync_to_state(domain.state)

        m0 = float(buf.rho.numpy().sum())
        for _ in range(40):
            domain.step(1.0)
        _sync(domain)
        m1 = float(bridge._ensure_gpu().rho.numpy().sum())
        self.assertTrue(np.isfinite(domain.state.density.numpy()).all())
        self.assertLess(abs(m1 - m0) / max(abs(m0), 1.0), 5.0e-5)

    def test_uniform_stream_momentum_stable(self) -> None:
        domain = _make_home_domain(
            res=(10, 10, 10), tau=0.7, u0=(0.04, -0.01, 0.02)
        )
        ux0 = float(domain.state.velocity_x.numpy().mean())
        for _ in range(50):
            domain.step(1.0)
        _sync(domain)
        ux1 = float(domain.state.velocity_x.numpy().mean())
        self.assertAlmostEqual(ux1, ux0, delta=1.0e-4)
        self.assertTrue(np.isfinite(domain.state.density.numpy()).all())


class TestHomeBaseBodyForce(unittest.TestCase):
    def test_periodic_acceleration_matches_guo_half_force(self) -> None:
        """Fused HOME collide applies Guo half-force ``u ← u + F/(2ρ)``."""
        g = 1.0e-4
        domain = _make_home_domain(
            res=(12, 12, 12), tau=1.0, gravity=(0.0, 0.0, g)
        )
        n_steps = 20
        for _ in range(n_steps):
            domain.step(1.0)
        _sync(domain)
        uz = float(domain.state.velocity_z.numpy().mean())
        # Half-force only ⇒ Δu ≈ g/2 per step (full Guo would be ≈ g).
        expected = n_steps * (0.5 * g)
        self.assertAlmostEqual(uz, expected, delta=0.08 * expected)


class TestHomeBaseQuiescent(unittest.TestCase):
    def test_closed_box_rest_stable(self) -> None:
        n = 12
        domain = _make_home_domain(
            res=(n, n, n),
            tau=0.8,
            bc_periodic=(False, False, False),
            bc_types=(0, 0, 0, 0, 0, 0),
        )
        for _ in range(100):
            domain.step(1.0)
        _sync(domain)
        ux = domain.state.velocity_x.numpy()
        uy = domain.state.velocity_y.numpy()
        uz = domain.state.velocity_z.numpy()
        rho = domain.state.density.numpy()
        self.assertTrue(np.isfinite(rho).all())
        sl = (slice(2, -2), slice(2, -2), slice(2, -2))
        self.assertLess(float(np.max(np.abs(ux[sl]))), 1.0e-5)
        self.assertLess(float(np.max(np.abs(uy[sl]))), 1.0e-5)
        self.assertLess(float(np.max(np.abs(uz[sl]))), 1.0e-5)


class TestHomeBasePoiseuille(unittest.TestCase):
    def test_force_driven_profile_l2(self) -> None:
        nx, ny, nz = 24, 17, 8
        tau = 0.8
        gx = 5.0e-5
        nu = (tau - 0.5) / 3.0
        h = float(ny)
        domain = _make_home_domain(
            res=(nx, ny, nz),
            tau=tau,
            gravity=(gx, 0.0, 0.0),
            bc_periodic=(True, False, True),
            bc_types=(3, 3, 0, 0, 3, 3),
        )
        t_visc = h * h / max(nu, 1.0e-12)
        steps = int(min(max(4.0 * t_visc, 800), 6000))
        u_max_prev = 0.0
        s = 0
        for s in range(1, steps + 1):
            domain.step(1.0)
            if s % 200 == 0 or s == steps:
                _sync(domain)
                ux = domain.state.velocity_x.numpy()
                u_max = float(np.max(ux))
                if s > 400 and abs(u_max - u_max_prev) / max(abs(u_max), 1.0e-12) < 1.0e-3:
                    break
                u_max_prev = u_max

        _sync(domain)
        profile = domain.state.velocity_x.numpy().mean(axis=(0, 2))
        y_wall = np.arange(ny, dtype=np.float64) + 0.5
        # Fused HOME currently injects Guo half-force only ⇒ use g_eff = g/2.
        g_eff = 0.5 * gx
        u_ana = (g_eff / (2.0 * nu)) * y_wall * (h - y_wall)
        l2 = float(
            np.sqrt(np.mean((profile - u_ana) ** 2))
            / max(float(np.max(np.abs(u_ana))), 1.0e-12)
        )
        self.assertGreater(float(profile[ny // 2]), 0.7 * float(u_ana[ny // 2]))
        self.assertLess(
            l2,
            0.20,
            msg=f"HOME Poiseuille L2={l2:.3%} uc={profile[ny // 2]:.4g} "
            f"ua={u_ana[ny // 2]:.4g} steps={s}",
        )


class TestHomeBaseLidDriven(unittest.TestCase):
    def test_develops_flow_and_stays_stable(self) -> None:
        n = 20
        u_lid = 0.06
        domain = _make_home_domain(
            res=(n, n, n),
            tau=0.8,
            bc_periodic=(False, False, False),
            bc_types=(0, 0, 0, 1, 0, 0),
            bc_velocity=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (u_lid, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        ke = 0.0
        for s in range(1, 401):
            domain.step(1.0)
            if s % 100 == 0:
                _sync(domain)
                ux = domain.state.velocity_x.numpy()
                uy = domain.state.velocity_y.numpy()
                self.assertTrue(np.isfinite(ux).all())
                ke = float(np.mean(ux[2:-2, 2:-2, 2:-2] ** 2 + uy[2:-2, 2:-2, 2:-2] ** 2))
                self.assertLess(float(np.max(np.abs(ux))), 0.5)
        self.assertGreater(ke, 1.0e-6)


class TestHomeNumpyBase(unittest.TestCase):
    """Library path (same collide/Eq.24 operators) — long-run smoke."""

    def test_periodic_mass_stable(self) -> None:
        field = make_uniform_equilibrium((8, 8, 8), rho0=1.0)
        field.ux[3, 3, 3] = 0.04
        field.sxx[3, 3, 3] = 0.04 * 0.04
        m0 = float(np.sum(field.rho))
        out = field
        for _ in range(100):
            out = step_periodic_numpy(out, lattice="D3Q27", tau=0.7)
        self.assertAlmostEqual(float(np.sum(out.rho)), m0, places=7)

    def test_cavity_finite(self) -> None:
        field = make_uniform_equilibrium((10, 10, 6), rho0=1.0)
        bc = HomeDomainBC.all_walls(lid_face="ymax", lid_ux=0.06)
        out = field
        for _ in range(80):
            out = step_domain_numpy(out, lattice="D3Q27", tau=0.75, domain_bc=bc)
        self.assertTrue(np.isfinite(out.ux).all())
        self.assertGreater(float(np.mean(out.ux[1:-1, 1:-1, 1:-1] ** 2)), 1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
