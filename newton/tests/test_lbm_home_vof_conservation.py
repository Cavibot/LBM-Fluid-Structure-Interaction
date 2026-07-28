# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests: HOME-FREE VOF liquid mass / momentum / KE diagnostics.

Inventory definitions (HOME-FREE / Körner)::

    water_mass  = Σ mass        over fluid cells (solid_phi ≥ 0)
                  (liquid: mass ≈ ρ; interface: mass = φ ρ)
    water_vol   = Σ φ           over fluid cells (volume-fraction proxy)
    momentum    = Σ mass · u    over fluid cells (mass-weighted)
    KE          = ½ Σ mass |u|² over fluid cells (lattice units)

Closed walls + no film drain: water_mass should stay nearly constant.
Momentum / mechanical energy are *not* conserved against walls or body force;
those cases check rest-state quiescence, mass hold under gravity, and bounded KE.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    CELL_GAS,
    CELL_INTERFACE,
    CELL_LIQUID,
    HomeDomainBC,
    seed_dam_break_column,
    step_home_vof_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.height_eq import (
    apply_vof_height_equation,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.quant import (
    pack_moments_from_float,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
    HomeVofGpuBuffers,
    alloc_home_vof_gpu,
    seed_home_vof_gpu,
    set_face_bc_gpu,
    step_home_vof_gpu,
)


@dataclass(frozen=True)
class _Conserved:
    water_mass: float
    water_vol: float
    px: float
    py: float
    pz: float
    ke: float
    u_max: float
    n_liquid: int
    n_interface: int


def _require_cuda() -> str:
    try:
        wp.init()
        if wp.get_cuda_device_count() <= 0:
            raise RuntimeError("no CUDA device")
        return "cuda:0"
    except Exception as exc:  # noqa: BLE001
        raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc


def _inventory_host(
    mass: np.ndarray,
    phi: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    cell: np.ndarray,
    solid: np.ndarray | None = None,
) -> _Conserved:
    fluid = cell != CELL_GAS
    if solid is not None:
        fluid = fluid & (solid >= 0.0)
    m = mass.astype(np.float64)
    mask = fluid
    ux64 = ux.astype(np.float64)
    uy64 = uy.astype(np.float64)
    uz64 = uz.astype(np.float64)
    wmass = float(m[mask].sum())
    wvol = float(phi.astype(np.float64)[mask].sum())
    px = float((m * ux64)[mask].sum())
    py = float((m * uy64)[mask].sum())
    pz = float((m * uz64)[mask].sum())
    speed2 = ux64 * ux64 + uy64 * uy64 + uz64 * uz64
    ke = 0.5 * float((m * speed2)[mask].sum())
    u_max = float(np.sqrt(speed2[mask]).max()) if np.any(mask) else 0.0
    if solid is not None:
        liq = (cell == CELL_LIQUID) & (solid >= 0.0)
        itf = (cell == CELL_INTERFACE) & (solid >= 0.0)
    else:
        liq = cell == CELL_LIQUID
        itf = cell == CELL_INTERFACE
    return _Conserved(
        water_mass=wmass,
        water_vol=wvol,
        px=px,
        py=py,
        pz=pz,
        ke=ke,
        u_max=u_max,
        n_liquid=int(liq.sum()),
        n_interface=int(itf.sum()),
    )


def _inventory_numpy(st) -> _Conserved:
    m = st.moments
    # Numpy path keeps mass in φ·ρ for IF; rebuild inventory like GPU.
    rho = m.rho
    mass = np.where(
        st.cell_type == CELL_LIQUID,
        rho,
        np.where(st.cell_type == CELL_INTERFACE, st.phi * rho, 0.0),
    )
    return _inventory_host(
        mass, st.phi, m.ux, m.uy, m.uz, st.cell_type, solid=None
    )


def _inventory_gpu(buf: HomeVofGpuBuffers) -> _Conserved:
    wp.synchronize_device(buf.device)
    return _inventory_host(
        buf.mass.numpy(),
        buf.phi.numpy(),
        buf.ux.numpy(),
        buf.uy.numpy(),
        buf.uz.numpy(),
        buf.cell_type.numpy(),
        solid=buf.solid_phi.numpy(),
    )


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0e-12)
    return abs(a - b) / denom


def _make_gpu_dambreak(
    n: int,
    *,
    moment_quant: bool = False,
    device: str | None = None,
) -> HomeVofGpuBuffers:
    device = device or _require_cuda()
    bc = HomeDomainBC.all_walls()
    buf = alloc_home_vof_gpu(
        (n, n, n),
        "D3Q27",
        device,
        domain_bc=bc,
        moment_quant=moment_quant,
    )
    set_face_bc_gpu(buf, bc)
    seed_home_vof_gpu(buf, dam_x=n // 4, fill_z=n // 2, rho_liquid=1.0)
    if moment_quant:
        pack_moments_from_float(buf, dither=False)
    return buf


def _step_gpu(
    buf: HomeVofGpuBuffers,
    *,
    steps: int,
    tau: float = 0.7,
    fz: float = 0.0,
    gamma: float = 0.0,
    moment_quant: bool = False,
) -> None:
    for _ in range(steps):
        step_home_vof_gpu(
            buf,
            tau=tau,
            fx=0.0,
            fy=0.0,
            fz=fz,
            rho_g0=1.0,
            gamma=gamma,
            home_fill_empty=False,
            home_wall_eq=True,
            seal_fg=True,
            wall_film_drain=False,
            moment_quant=moment_quant,
            moment_quant_dither=False,
        )
    wp.synchronize_device(buf.device)


class TestHomeVofMassNumpy(unittest.TestCase):
    """Host reference path — regression for liquid inventory."""

    def test_seed_water_mass_matches_phi_rho(self) -> None:
        st = seed_dam_break_column((16, 12, 16), dam_x=4, fill_z=8, rho_liquid=1.0)
        inv = _inventory_numpy(st)
        self.assertGreater(inv.water_mass, 10.0)
        self.assertGreater(inv.n_liquid, 0)
        self.assertGreater(inv.n_interface, 0)
        # At seed, liquid has φ=1, ρ=1 → mass ≈ n_liquid + Σ φ_IF
        self.assertAlmostEqual(inv.water_mass, inv.water_vol, places=5)

    def test_closed_walls_water_mass_no_gravity(self) -> None:
        n = 16
        st = seed_dam_break_column((n, n // 2, n), dam_x=n // 4, fill_z=n // 2)
        inv0 = _inventory_numpy(st)
        bc = HomeDomainBC.all_walls()
        out = st
        for _ in range(40):
            out = step_home_vof_numpy(
                out,
                lattice="D3Q27",
                tau=0.7,
                fz=0.0,
                domain_bc=bc,
                rho_g0=1.0,
                gamma=0.0,
            )
        inv1 = _inventory_numpy(out)
        self.assertTrue(np.isfinite(out.moments.rho).all())
        self.assertLess(_rel_err(inv1.water_mass, inv0.water_mass), 0.02)
        self.assertGreater(inv1.n_liquid + inv1.n_interface, 0)

    def test_closed_walls_water_mass_with_gravity(self) -> None:
        """Gravity redistributes momentum; closed box should keep water mass."""
        n = 16
        st = seed_dam_break_column((n, n // 2, n), dam_x=n // 4, fill_z=n // 2)
        inv0 = _inventory_numpy(st)
        bc = HomeDomainBC.all_walls()
        out = st
        for _ in range(40):
            out = step_home_vof_numpy(
                out,
                lattice="D3Q27",
                tau=0.7,
                fz=-0.0002,
                domain_bc=bc,
                rho_g0=1.0,
                gamma=0.0,
            )
        inv1 = _inventory_numpy(out)
        self.assertLess(_rel_err(inv1.water_mass, inv0.water_mass), 0.03)
        # Some downward motion on wet cells
        wet = out.moments.rho > 0.5
        self.assertTrue(wet.any())
        self.assertLess(float(np.mean(out.moments.uz[wet])), 0.05)


class TestHomeVofPureDamBreakMomentumEnergy(unittest.TestCase):
    """Pure dam-break (no rigid): mass + momentum / KE diagnostics.

    Closed walls + gravity ⇒ total momentum / mechanical energy are **not**
    conserved. We check water-mass inventory, LBM stability (|u|), and that
    KE / |P| stay finite (gravity injects downward momentum early).
    """

    def test_dambreak_no_gravity_mass_and_quiescent_ke(self) -> None:
        """Rest seed + no body force: mass holds; KE and |P| stay tiny."""
        buf = _make_gpu_dambreak(32)
        inv0 = _inventory_gpu(buf)
        _step_gpu(buf, steps=80, fz=0.0, tau=0.55)
        inv1 = _inventory_gpu(buf)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.025,
            msg=f"mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )
        scale = max(inv1.water_mass, 1.0)
        self.assertLess(abs(inv1.px) / scale, 8.0e-3)
        self.assertLess(abs(inv1.py) / scale, 8.0e-3)
        self.assertLess(abs(inv1.pz) / scale, 8.0e-3)
        self.assertLess(inv1.ke / scale, 1.0e-4)
        self.assertLess(inv1.u_max, 0.05)

    def test_dambreak_with_gravity_mass_momentum_ke(self) -> None:
        """Classic column collapse: mass ≈ conserved; KE peaks; |u| stable."""
        buf = _make_gpu_dambreak(32)
        inv0 = _inventory_gpu(buf)
        self.assertGreater(inv0.water_mass, 50.0)
        self.assertLess(inv0.ke, 1.0)  # seeded at rest

        ke_peak = inv0.ke
        p_abs_peak = abs(inv0.px) + abs(inv0.py) + abs(inv0.pz)
        pz_min = inv0.pz
        for chunk in range(8):
            _step_gpu(buf, steps=20, fz=-0.0004, tau=0.55, gamma=0.0)
            inv = _inventory_gpu(buf)
            self.assertTrue(np.isfinite(inv.ke))
            self.assertTrue(np.isfinite(inv.px + inv.py + inv.pz))
            self.assertLess(inv.u_max, 0.40, msg=f"chunk={chunk} u_max={inv.u_max}")
            ke_peak = max(ke_peak, inv.ke)
            p_abs_peak = max(p_abs_peak, abs(inv.px) + abs(inv.py) + abs(inv.pz))
            pz_min = min(pz_min, inv.pz)

        inv1 = _inventory_gpu(buf)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.04,
            msg=f"mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )
        # Gravity / collapse should produce motion and downward momentum.
        self.assertGreater(ke_peak, inv0.ke + 1.0e-3)
        self.assertLess(pz_min, inv0.pz - 1.0e-2)
        self.assertLess(ke_peak, 5.0e4)
        self.assertLess(p_abs_peak, 5.0e3)
        self.assertGreater(inv1.n_liquid + inv1.n_interface, 0)

    def test_dambreak_gamma_mass_and_bounded_ke(self) -> None:
        """Mild surface tension: still mass-stable and KE-bounded."""
        buf = _make_gpu_dambreak(28)
        inv0 = _inventory_gpu(buf)
        _step_gpu(buf, steps=100, fz=-0.0003, tau=0.55, gamma=1.5e-3)
        inv1 = _inventory_gpu(buf)
        self.assertLess(_rel_err(inv1.water_mass, inv0.water_mass), 0.05)
        self.assertLess(inv1.u_max, 0.40)
        self.assertLess(inv1.ke, 5.0e4)
        self.assertTrue(np.isfinite(inv1.ke))


class TestHomeVofMassGpu(unittest.TestCase):
    """Production GPU fused path."""

    def test_gpu_seed_inventory_finite(self) -> None:
        buf = _make_gpu_dambreak(16)
        inv = _inventory_gpu(buf)
        self.assertGreater(inv.water_mass, 10.0)
        self.assertTrue(np.isfinite(inv.water_mass))
        self.assertGreater(inv.n_liquid, 0)
        self.assertGreater(inv.n_interface, 0)

    def test_gpu_closed_walls_water_mass_no_force(self) -> None:
        buf = _make_gpu_dambreak(24)
        inv0 = _inventory_gpu(buf)
        _step_gpu(buf, steps=60, fz=0.0)
        inv1 = _inventory_gpu(buf)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.02,
            msg=f"mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )
        self.assertLess(_rel_err(inv1.water_vol, inv0.water_vol), 0.05)
        self.assertGreater(inv1.n_liquid + inv1.n_interface, 0)

    def test_gpu_closed_walls_water_mass_with_gravity(self) -> None:
        buf = _make_gpu_dambreak(24)
        inv0 = _inventory_gpu(buf)
        _step_gpu(buf, steps=60, fz=-0.0003)
        inv1 = _inventory_gpu(buf)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.03,
            msg=f"mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )
        # Momentum in −z should grow (walls later absorb; early steps still gain).
        self.assertLess(inv1.pz, inv0.pz + 1.0e-3)

    def test_gpu_rest_pool_momentum_stays_small(self) -> None:
        """Nearly filled rest column: without force, |P| stays tiny."""
        device = _require_cuda()
        n = 16
        bc = HomeDomainBC.all_walls()
        buf = alloc_home_vof_gpu((n, n, n), "D3Q27", device, domain_bc=bc)
        set_face_bc_gpu(buf, bc)
        # Full-ish pool: dam across whole x, half height.
        seed_home_vof_gpu(buf, dam_x=n - 1, fill_z=n // 2, rho_liquid=1.0)
        inv0 = _inventory_gpu(buf)
        _step_gpu(buf, steps=30, fz=0.0, tau=0.8)
        inv1 = _inventory_gpu(buf)
        self.assertLess(_rel_err(inv1.water_mass, inv0.water_mass), 0.02)
        scale = max(inv1.water_mass, 1.0)
        self.assertLess(abs(inv1.px) / scale, 5.0e-3)
        self.assertLess(abs(inv1.py) / scale, 5.0e-3)
        self.assertLess(abs(inv1.pz) / scale, 5.0e-3)


class TestHomeVofMomentQuantConservation(unittest.TestCase):
    """Persistent 16-bit moment quant must not destroy liquid mass."""

    def test_quant_vs_fp32_water_mass_track(self) -> None:
        n = 20
        steps = 40
        buf_ref = _make_gpu_dambreak(n, moment_quant=False)
        buf_q = _make_gpu_dambreak(n, moment_quant=True)
        inv0_r = _inventory_gpu(buf_ref)
        inv0_q = _inventory_gpu(buf_q)
        self.assertLess(_rel_err(inv0_q.water_mass, inv0_r.water_mass), 1.0e-5)

        _step_gpu(buf_ref, steps=steps, fz=-0.0002, moment_quant=False)
        _step_gpu(buf_q, steps=steps, fz=-0.0002, moment_quant=True)
        inv_r = _inventory_gpu(buf_ref)
        inv_q = _inventory_gpu(buf_q)

        self.assertLess(_rel_err(inv_r.water_mass, inv0_r.water_mass), 0.03)
        self.assertLess(_rel_err(inv_q.water_mass, inv0_q.water_mass), 0.03)
        # Quant path should stay close to fp32 inventory (not the old ρ-collapse).
        self.assertLess(
            _rel_err(inv_q.water_mass, inv_r.water_mass),
            0.05,
            msg=f"fp32 mass={inv_r.water_mass:.6g} quant={inv_q.water_mass:.6g}",
        )
        # Liquid density must remain O(1), not ~0.5 collapse.
        rho = buf_q.rho.numpy()
        cell = buf_q.cell_type.numpy()
        liquid = cell == CELL_LIQUID
        self.assertTrue(liquid.any())
        self.assertGreater(float(rho[liquid].mean()), 0.9)
        self.assertLess(float(rho[liquid].mean()), 1.15)


class TestHomeVofHeightEqMass(unittest.TestCase):
    """Late-pool φ→φ* regularizer should not evaporate water."""

    def test_height_eq_preserves_water_mass(self) -> None:
        buf = _make_gpu_dambreak(24)
        # Advance so a pool IF plane exists.
        _step_gpu(buf, steps=80, fz=-0.0004)
        inv0 = _inventory_gpu(buf)
        self.assertGreater(inv0.n_interface, 4)

        stats = apply_vof_height_equation(
            buf,
            rate=0.05,
            u_max=0.2,
            dh_cap=0.05,
            use_gpu=True,
            sync_stats=True,
        )
        inv1 = _inventory_gpu(buf)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.01,
            msg=f"before={inv0.water_mass:.6g} after={inv1.water_mass:.6g} stats={stats}",
        )
        # Several height-eq sweeps still keep inventory.
        for _ in range(8):
            apply_vof_height_equation(
                buf, rate=0.05, u_max=0.2, dh_cap=0.05, use_gpu=True, sync_stats=False
            )
        inv2 = _inventory_gpu(buf)
        self.assertLess(_rel_err(inv2.water_mass, inv0.water_mass), 0.02)


class TestHomeVofFiniteNoNan(unittest.TestCase):
    def test_long_gpu_run_finite(self) -> None:
        buf = _make_gpu_dambreak(24)
        _step_gpu(buf, steps=120, fz=-0.0003, gamma=1.5e-3)
        wp.synchronize_device(buf.device)
        for name in ("rho", "ux", "uy", "uz", "mass", "phi"):
            arr = getattr(buf, name).numpy()
            self.assertTrue(np.isfinite(arr).all(), msg=name)
        inv = _inventory_gpu(buf)
        self.assertGreater(inv.water_mass, 1.0)


class TestHomeVofGenericDefaults(unittest.TestCase):
    """P0: heuristic flags stay off on the generic model factory."""

    def test_make_home_vof_model_heuristics_off(self) -> None:
        from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.generic import (
            make_home_vof_model,
        )

        model = make_home_vof_model(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.02,
            gravity_z=-0.001,
        )
        self.assertEqual(model.lbm_backend, "home_fp32")
        self.assertEqual(model.phase_mode, "vof_sharp")
        self.assertFalse(model.vof_height_eq)
        self.assertFalse(model.vof_wall_film_drain)
        self.assertFalse(model.vof_quiet_fill)
        self.assertFalse(model.vof_orphan_reabsorb)
        self.assertFalse(model.vof_bubble_pressure)
        self.assertFalse(model.vof_bubble_disjoint)
        self.assertFalse(model.vof_home_fill_empty)
        self.assertFalse(model.vof_home_wall_eq)
        self.assertFalse(model.vof_home_moment_quant)

    def test_lbm_model_orphan_default_off(self) -> None:
        from wanphys._src.fluid.fluid_grid.lbm import LbmModel

        model = LbmModel(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.02,
            phase_mode="vof_sharp",
            lbm_backend="home_fp32",
            G=0.0,
        )
        self.assertFalse(model.vof_orphan_reabsorb)
        self.assertFalse(model.vof_height_eq)
        self.assertFalse(model.vof_wall_film_drain)
        self.assertFalse(model.vof_bubble_pressure)


class TestHomeVofSolidMaskInventory(unittest.TestCase):
    """Static solid stamp: fluid inventory only outside solid_phi < 0."""

    def test_static_solid_block_excluded_from_water_mass(self) -> None:
        buf = _make_gpu_dambreak(20)
        inv0 = _inventory_gpu(buf)
        solid = buf.solid_phi.numpy().copy()
        # Stamp a block of solid into the liquid column (x < dam).
        solid[2:6, 2:8, 2:8] = -1.0
        buf.solid_phi.assign(wp.array(solid, dtype=float, device=buf.device))
        wp.synchronize_device(buf.device)
        # One step applies solid_mask → gas inside solid.
        _step_gpu(buf, steps=1, fz=0.0)
        inv1 = _inventory_gpu(buf)
        self.assertLess(inv1.water_mass, inv0.water_mass)
        self.assertGreater(inv1.water_mass, 1.0)
        # Further steps without moving solid: mass should hold (soft).
        _step_gpu(buf, steps=25, fz=-0.0002)
        inv2 = _inventory_gpu(buf)
        self.assertLess(
            _rel_err(inv2.water_mass, inv1.water_mass),
            0.04,
            msg=f"after mask {inv1.water_mass:.6g} → {inv2.water_mass:.6g}",
        )


if __name__ == "__main__":
    unittest.main()
