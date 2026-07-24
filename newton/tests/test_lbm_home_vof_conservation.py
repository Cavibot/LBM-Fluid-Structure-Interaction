# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests: HOME-FREE VOF liquid mass / momentum conservation.

Inventory definitions (HOME-FREE / Körner)::

    water_mass  = Σ mass        over fluid cells (solid_phi ≥ 0)
                  (liquid: mass ≈ ρ; interface: mass = φ ρ)
    water_vol   = Σ φ           over fluid cells (volume-fraction proxy)
    momentum    = Σ mass · u    over fluid cells (mass-weighted)

Closed walls + no film drain: water_mass should stay nearly constant.
Momentum is *not* conserved against walls or body force; those cases only
check rest-state or force-free periodic bounds.
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
    wmass = float(m[mask].sum())
    wvol = float(phi.astype(np.float64)[mask].sum())
    px = float((m * ux.astype(np.float64))[mask].sum())
    py = float((m * uy.astype(np.float64))[mask].sum())
    pz = float((m * uz.astype(np.float64))[mask].sum())
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


if __name__ == "__main__":
    unittest.main()
