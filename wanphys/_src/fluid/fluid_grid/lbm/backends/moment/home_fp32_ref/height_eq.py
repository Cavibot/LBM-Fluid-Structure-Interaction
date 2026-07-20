# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Free-surface interface-φ leveling for HOME-FREE VOF (solver opt-in).

Only touches cells that are already ``CELL_INTERFACE``. No bulk rewrite, and by
default **no** L↔I↔G type changes — so this cannot invent mid-air liquid.

Operator (per call)::

    φ ← clip( φ + α (φ* − φ), ε, 1−ε )
    then rescale IF masses so Σmass on the touched set is unchanged.

``φ*`` is the mass-weighted mean fill on the **dominant free-surface plane**
(mode of interface ``k`` among pool-surface IF cells: liquid below, gas/empty
above). Columns whose IF sits on another ``k`` are left alone (sub-cell leveling
only). Cross-plane leveling needs guarded L↔I type changes and is **off** —
those paths previously invented mid-air liquid.

Enable with ``LbmModel.vof_height_eq = True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
        HomeVofGpuBuffers,
    )

CELL_GAS = 0
CELL_INTERFACE = 1
CELL_LIQUID = 2


def continuous_surface_height(
    phi: np.ndarray,
    cell: np.ndarray,
    *,
    phi_dust: float = 0.05,
) -> np.ndarray:
    """Per-column continuous height ``h = k+φ`` (IF) or ``k+1`` (liquid top)."""
    nz = phi.shape[2]
    usable = (cell == CELL_LIQUID) | ((cell == CELL_INTERFACE) & (phi >= phi_dust))
    k_idx = np.arange(nz, dtype=np.int32)[None, None, :]
    k_top = np.where(usable, k_idx, np.int32(-1)).max(axis=2)
    h = np.full(k_top.shape, np.nan, dtype=np.float64)
    ii, jj = np.nonzero(k_top >= 0)
    if ii.size == 0:
        return h
    kk = k_top[ii, jj]
    is_if = cell[ii, jj, kk] == CELL_INTERFACE
    fill = np.ones(kk.shape[0], dtype=np.float64)
    fill[is_if] = np.clip(phi[ii, jj, kk][is_if], 0.0, 1.0)
    h[ii, jj] = kk.astype(np.float64) + np.where(is_if, fill, 1.0)
    return h


def _pool_surface_interfaces(
    cell: np.ndarray,
    phi: np.ndarray,
    solid: np.ndarray,
    *,
    phi_dust: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ii, jj, kk)`` of free-surface IF cells (liquid below).

    Requires the cell below to be liquid (or domain bottom). Skips climb IF
    sitting above a gas gap — those are not the pool interface.
    """
    nx, ny, nz = cell.shape
    ii_list: list[int] = []
    jj_list: list[int] = []
    kk_list: list[int] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if solid[i, j, k] < 0.0:
                    continue
                if cell[i, j, k] != CELL_INTERFACE:
                    continue
                if float(phi[i, j, k]) < phi_dust:
                    continue
                # Must have liquid immediately below (pool surface), or sit on floor.
                if k == 0:
                    below_ok = True
                elif solid[i, j, k - 1] < 0.0:
                    below_ok = True
                else:
                    below_ok = cell[i, j, k - 1] == CELL_LIQUID
                if not below_ok:
                    continue
                # Prefer gas / empty above (true free surface). Allow domain top.
                if k + 1 < nz and solid[i, j, k + 1] >= 0.0:
                    if cell[i, j, k + 1] == CELL_LIQUID:
                        continue
                ii_list.append(i)
                jj_list.append(j)
                kk_list.append(k)
                break  # one FS IF per column
    return (
        np.asarray(ii_list, dtype=np.int32),
        np.asarray(jj_list, dtype=np.int32),
        np.asarray(kk_list, dtype=np.int32),
    )


def _try_plane_shift(
    cell: np.ndarray,
    rho: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    solid: np.ndarray,
    i: int,
    j: int,
    k: int,
    *,
    toward_fill: bool,
) -> bool:
    """Disabled stub — plane shift can invent water; keep IF-only."""
    del cell, rho, mass, phi, solid, i, j, k, toward_fill
    return False


def apply_vof_height_equation(
    buf: HomeVofGpuBuffers,
    *,
    rate: float = 0.05,
    u_max: float = 0.05,
    phi_dust: float = 0.05,
    dh_cap: float = 0.08,
    n_sweeps: int = 1,
) -> dict[str, float]:
    """Relax φ on existing free-surface interface cells only.

    ``dh_cap`` here caps ``|Δφ|`` per call (not column height).
    ``n_sweeps > 1`` enables cautious single-cell plane shift after φ clamp.
    ``u_max``: skip if mean |u| on those IF tips is too large.
    """
    cell = buf.cell_type.numpy().astype(np.int32, copy=True)
    phi = buf.phi.numpy().astype(np.float32, copy=True)
    mass = buf.mass.numpy().astype(np.float32, copy=True)
    rho = buf.rho.numpy().astype(np.float32, copy=True)
    solid = buf.solid_phi.numpy()
    ux = buf.ux.numpy()
    uy = buf.uy.numpy()
    uz = buf.uz.numpy()
    fluid = solid >= 0.0

    ii, jj, kk = _pool_surface_interfaces(
        cell, phi, solid, phi_dust=float(phi_dust)
    )
    n_if = int(ii.size)
    if n_if < 4:
        return {"n_wet": 0.0, "n_if": float(n_if), "dmass": 0.0, "phi_star": 0.0}

    spd = np.sqrt(ux[ii, jj, kk] ** 2 + uy[ii, jj, kk] ** 2 + uz[ii, jj, kk] ** 2)
    u_mean = float(spd.mean())
    if u_mean > float(u_max):
        return {
            "n_wet": float(n_if),
            "n_if": float(n_if),
            "dmass": 0.0,
            "phi_star": 0.0,
            "u_mean": u_mean,
            "skipped": 1.0,
        }

    m_before = float(mass[fluid].sum())

    # Dominant interface plane (mode k).
    mode_k = int(np.bincount(kk.astype(np.int64)).argmax())
    on_plane = kk == mode_k
    n_plane = int(on_plane.sum())
    if n_plane < 4:
        return {
            "n_wet": float(n_if),
            "n_if": float(n_if),
            "n_plane": float(n_plane),
            "mode_k": float(mode_k),
            "dmass": 0.0,
            "phi_star": 0.0,
            "skipped": 1.0,
        }

    sel_i = ii[on_plane]
    sel_j = jj[on_plane]
    sel_k = kk[on_plane]

    phi_s = phi[sel_i, sel_j, sel_k].astype(np.float64)
    mass_s = mass[sel_i, sel_j, sel_k].astype(np.float64)
    rho_s = np.maximum(rho[sel_i, sel_j, sel_k].astype(np.float64), 1.0e-3)
    m_plane0 = float(mass_s.sum())

    # Mass-weighted mean φ on this plane.
    phi_star = float(np.sum(phi_s * rho_s) / max(float(np.sum(rho_s)), 1.0e-6))

    alpha = float(np.clip(rate, 0.0, 1.0))
    dphi_cap = float(max(dh_cap, 1.0e-4))
    dphi = alpha * (phi_star - phi_s)
    np.clip(dphi, -dphi_cap, dphi_cap, out=dphi)

    phi_new = np.clip(phi_s + dphi, 0.02, 0.98)
    # Convert to mass with local ρ, then renormalize Σmass on the plane.
    mass_new = phi_new * rho_s
    sum_m = float(mass_new.sum())
    if sum_m > 1.0e-6 and abs(sum_m - m_plane0) > 1.0e-8:
        mass_new *= m_plane0 / sum_m
        phi_new = np.clip(mass_new / rho_s, 0.02, 0.98)
        mass_new = phi_new * rho_s
        # Second renorm after clip.
        sum_m2 = float(mass_new.sum())
        if sum_m2 > 1.0e-6:
            mass_new *= m_plane0 / sum_m2
            phi_new = mass_new / rho_s

    phi[sel_i, sel_j, sel_k] = phi_new.astype(np.float32)
    mass[sel_i, sel_j, sel_k] = mass_new.astype(np.float32)

    # Plane shift disabled (n_sweeps ignored) — IF φ only, no type changes.
    del n_sweeps

    m_final = float(mass[fluid].sum())
    phi_star_out = float(phi[sel_i, sel_j, sel_k].mean()) if sel_i.size else phi_star

    import warp as wp

    device = buf.device
    buf.cell_type.assign(wp.array(cell, dtype=wp.int32, device=device))
    buf.phi.assign(wp.array(phi, dtype=float, device=device))
    buf.mass.assign(wp.array(mass, dtype=float, device=device))

    return {
        "n_wet": float(n_if),
        "n_if": float(n_if),
        "n_plane": float(n_plane),
        "n_touch": float(n_plane),
        "mode_k": float(mode_k),
        "phi_star": phi_star_out,
        "H_star": float(mode_k) + phi_star_out,
        "phi_std": float(phi[sel_i, sel_j, sel_k].std()) if sel_i.size else 0.0,
        "u_mean": u_mean,
        "alpha": alpha,
        "n_shift": 0.0,
        "mass_before": m_before,
        "mass_after": m_final,
        "mass_delta": m_final - m_before,
        "dmass": m_final - m_before,
        "skipped": 0.0,
    }


def _extract_height(cell, phi, solid, ux, uy, uz, *, phi_dust):
    del solid, ux, uy, uz
    h = continuous_surface_height(phi, cell, phi_dust=phi_dust)
    wet = np.isfinite(h)
    return np.where(wet, h, -1.0), wet, np.zeros_like(h)
