# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Free-surface interface-φ leveling for HOME-FREE VOF (solver opt-in).

Target look: one dominant pool IF plane with nearly uniform ``φ`` (no large
high/low patches).

Per call::

    1) Drop airborne wet cells (gas immediately below) onto the pool IF in
       the same column — splash “balls”, mass-conserving.
    2) Heal one-cell **low stairs** (pool IF at ``mode_k−1``; gas or thin IF
       above): borrow from on-plane IF, promote — wall/corner prioritized by
       horizontal wall-contact count (domain face or solid), never a fixed index.
    3) Boost thin/dust IF on the mode plane up to the safe φ band (same borrow).
    4) Plane ``φ → φ*`` with stronger α on wall-adjacent cells that sit below φ*.
    5) Light tip ``|u|`` damping (weak).

Does not invent mid-air liquid (promote only with liquid below + gas above).

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

# Stay away from fill/empty so the operator does not trigger type flips that
# show up as random pits / a circular repair wave after each flatten.
_PHI_LO = 0.18
_PHI_HI = 0.82


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
    """Return ``(ii, jj, kk)`` of free-surface IF cells (liquid below)."""
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
                if k == 0:
                    below_ok = True
                elif solid[i, j, k - 1] < 0.0:
                    below_ok = True
                else:
                    below_ok = cell[i, j, k - 1] == CELL_LIQUID
                if not below_ok:
                    continue
                if k + 1 < nz and solid[i, j, k + 1] >= 0.0:
                    if cell[i, j, k + 1] == CELL_LIQUID:
                        continue
                ii_list.append(i)
                jj_list.append(j)
                kk_list.append(k)
                break
    return (
        np.asarray(ii_list, dtype=np.int32),
        np.asarray(jj_list, dtype=np.int32),
        np.asarray(kk_list, dtype=np.int32),
    )


def _drop_airborne_onto_pool(
    cell: np.ndarray,
    rho: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    solid: np.ndarray,
    *,
    mode_k: int,
) -> int:
    """Move splash droplets (wet + gas below) onto pool IF in-column."""
    nx, ny, nz = cell.shape
    n_clear = 0
    for i in range(nx):
        for j in range(ny):
            dep_k = -1
            for k in range(min(nz - 1, mode_k + 2), -1, -1):
                if solid[i, j, k] < 0.0:
                    continue
                if cell[i, j, k] != CELL_INTERFACE:
                    continue
                if k == 0 or solid[i, j, k - 1] < 0.0 or cell[i, j, k - 1] == CELL_LIQUID:
                    dep_k = k
                    break
            if dep_k < 0:
                for k in range(min(nz - 1, mode_k + 2), -1, -1):
                    if solid[i, j, k] < 0.0:
                        continue
                    if cell[i, j, k] == CELL_LIQUID:
                        dep_k = k
                        break
            if dep_k < 0:
                continue

            for k in range(nz - 1, dep_k, -1):
                if solid[i, j, k] < 0.0:
                    continue
                ct = int(cell[i, j, k])
                if ct == CELL_GAS:
                    continue
                below = k - 1
                if below < 0 or solid[i, j, below] < 0.0:
                    unsupported = True
                else:
                    unsupported = int(cell[i, j, below]) == CELL_GAS
                if not unsupported:
                    continue
                add = float(mass[i, j, k])
                cell[i, j, k] = CELL_GAS
                mass[i, j, k] = 0.0
                phi[i, j, k] = 0.0
                rho[i, j, k] = 0.0
                n_clear += 1
                if add <= 0.0:
                    continue
                r_d = float(max(rho[i, j, dep_k], 1.0e-3))
                if cell[i, j, dep_k] == CELL_LIQUID:
                    # Absorb into liquid inventory (mass already counted as full cell).
                    # Spill remainder into a new IF above if gas, else keep on liquid.
                    above = dep_k + 1
                    nz = cell.shape[2]
                    if (
                        above < nz
                        and solid[i, j, above] >= 0.0
                        and int(cell[i, j, above]) == CELL_GAS
                    ):
                        cell[i, j, above] = CELL_INTERFACE
                        rho[i, j, above] = r_d
                        mass[i, j, above] = add
                        phi[i, j, above] = float(add / r_d)
                        dep_k = above
                    else:
                        mass[i, j, dep_k] = float(mass[i, j, dep_k]) + add
                elif cell[i, j, dep_k] == CELL_INTERFACE:
                    # Never clip mass away — overfull IF is fine; LBM fill promotes.
                    mass[i, j, dep_k] = float(mass[i, j, dep_k]) + add
                    phi[i, j, dep_k] = float(mass[i, j, dep_k]) / r_d
                else:
                    cell[i, j, dep_k] = CELL_INTERFACE
                    rho[i, j, dep_k] = max(r_d, 1.0)
                    mass[i, j, dep_k] = add
                    phi[i, j, dep_k] = float(add / max(r_d, 1.0e-3))
    return n_clear


def _xy_wall_contacts(
    i: int,
    j: int,
    k: int,
    solid: np.ndarray,
) -> int:
    """Count horizontal wall contacts (domain face or solid neighbor)."""
    nx, ny, _nz = solid.shape
    n = 0
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
            n += 1
        elif float(solid[ni, nj, k]) < 0.0:
            n += 1
    return n


def _boost_thin_pool_if(
    cell: np.ndarray,
    rho: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    solid: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    kk: np.ndarray,
    *,
    mode_k: int,
    max_boost: int = 128,
) -> int:
    """Raise under-filled pool IF (esp. wall/corner) up to ``_PHI_LO``.

    Dust IF (``φ < phi_dust``) is invisible to height maps and reads as a
    one-cell pit. Borrow from on-plane neighbors — generic, no fixed corner.
    """
    nx, ny, _nz = cell.shape
    key = np.full((nx, ny), -1, dtype=np.int32)
    for t in range(ii.size):
        key[int(ii[t]), int(jj[t])] = t

    plane_idx = [t for t in range(ii.size) if int(kk[t]) == mode_k]
    # Also scan mode-plane cells that pool finder skipped due to dust:
    # any IF at mode_k with liquid below.
    candidates: list[tuple[int, int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            if solid[i, j, mode_k] < 0.0:
                continue
            if int(cell[i, j, mode_k]) != CELL_INTERFACE:
                continue
            if mode_k > 0 and solid[i, j, mode_k - 1] >= 0.0:
                if int(cell[i, j, mode_k - 1]) != CELL_LIQUID:
                    continue
            p = float(phi[i, j, mode_k])
            if p >= _PHI_LO - 1.0e-6:
                continue
            wc = _xy_wall_contacts(i, j, mode_k, solid)
            candidates.append((-wc, i, j, mode_k))
    candidates.sort()
    n_boost = 0
    for _prio, i, j, k in candidates:
        if n_boost >= max_boost:
            break
        r = float(max(rho[i, j, k], 1.0e-3))
        m_loc = float(mass[i, j, k])
        target = _PHI_LO * r
        need = max(0.0, target - m_loc)
        if need <= 1.0e-10:
            continue
        donors: list[tuple[int, int, int]] = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                continue
            if int(cell[ni, nj, k]) != CELL_INTERFACE:
                continue
            if float(phi[ni, nj, k]) <= _PHI_LO + 0.02:
                continue
            donors.append((ni, nj, k))
        if not donors:
            for pt in plane_idx:
                ni, nj = int(ii[pt]), int(jj[pt])
                if ni == i and nj == j:
                    continue
                if int(kk[pt]) != mode_k:
                    continue
                if float(phi[ni, nj, mode_k]) <= _PHI_LO + 0.02:
                    continue
                donors.append((ni, nj, mode_k))
                if len(donors) >= 16:
                    break
        if not donors:
            continue
        avail_total = 0.0
        for di_, dj_, dk_ in donors:
            rd = float(max(rho[di_, dj_, dk_], 1.0e-3))
            avail_total += max(0.0, float(mass[di_, dj_, dk_]) - _PHI_LO * rd)
        if avail_total < need - 1.0e-8:
            continue
        taken = 0.0
        for di_, dj_, dk_ in donors:
            if taken >= need - 1.0e-10:
                break
            rd = float(max(rho[di_, dj_, dk_], 1.0e-3))
            md = float(mass[di_, dj_, dk_])
            avail = max(0.0, md - _PHI_LO * rd)
            give = min(avail, need - taken)
            if give <= 0.0:
                continue
            mass[di_, dj_, dk_] = md - give
            phi[di_, dj_, dk_] = float(mass[di_, dj_, dk_]) / rd
            taken += give
        mass[i, j, k] = m_loc + taken
        phi[i, j, k] = float(mass[i, j, k]) / r
        n_boost += 1
    return n_boost


def _heal_low_stairs(
    cell: np.ndarray,
    rho: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    solid: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    kk: np.ndarray,
    *,
    mode_k: int,
    max_heals: int = 64,
) -> int:
    """Promote pool IF at ``mode_k-1`` when above is gas **or** thin IF.

    Borrows fill(+seed) from on-plane IF. Prioritizes wall/corner contacts.
    """
    nx, ny, nz = cell.shape
    if mode_k <= 0 or mode_k >= nz:
        return 0

    key = np.full((nx, ny), -1, dtype=np.int32)
    for t in range(ii.size):
        key[int(ii[t]), int(jj[t])] = t

    plane_idx = [t for t in range(ii.size) if int(kk[t]) == mode_k]
    if not plane_idx:
        # Still allow heal using mode_k IF found by scan.
        pass

    low_idx = [t for t in range(ii.size) if int(kk[t]) == mode_k - 1]

    def _prio(t: int) -> tuple[int, float]:
        i, j, k = int(ii[t]), int(jj[t]), int(kk[t])
        return (-_xy_wall_contacts(i, j, k, solid), float(phi[i, j, k]))

    low_idx.sort(key=_prio)
    n_heal = 0
    seed_phi = 0.5 * (_PHI_LO + _PHI_HI)

    for t in low_idx:
        if n_heal >= max_heals:
            break
        i, j, k = int(ii[t]), int(jj[t]), int(kk[t])
        if solid[i, j, k] < 0.0 or int(cell[i, j, k]) != CELL_INTERFACE:
            continue
        above = k + 1
        if above >= nz or solid[i, j, above] < 0.0:
            continue
        above_ct = int(cell[i, j, above])
        above_phi = float(phi[i, j, above]) if above_ct == CELL_INTERFACE else 0.0
        # Gas: classic promote. Thin IF above: merge lower into liquid, keep upper IF.
        if above_ct == CELL_GAS:
            merge_thin = False
        elif above_ct == CELL_INTERFACE and above_phi < _PHI_HI:
            merge_thin = True
        else:
            continue
        if k > 0 and solid[i, j, k - 1] >= 0.0 and int(cell[i, j, k - 1]) != CELL_LIQUID:
            continue

        r = float(max(rho[i, j, k], 1.0e-3))
        m_loc = float(mass[i, j, k])
        need_fill = max(0.0, r - m_loc)
        need_seed = 0.0 if merge_thin else seed_phi * r
        need = need_fill + need_seed

        donors: list[tuple[int, int, int]] = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                continue
            nt = int(key[ni, nj])
            if nt >= 0 and int(kk[nt]) == mode_k:
                donors.append((ni, nj, mode_k))
            elif int(cell[ni, nj, mode_k]) == CELL_INTERFACE:
                donors.append((ni, nj, mode_k))
        if not donors:
            for pt in plane_idx:
                donors.append((int(ii[pt]), int(jj[pt]), mode_k))
                if len(donors) >= 16:
                    break
        if need > 1.0e-8 and not donors:
            continue

        if need > 1.0e-8:
            avail_total = 0.0
            for di_, dj_, dk_ in donors:
                rd = float(max(rho[di_, dj_, dk_], 1.0e-3))
                avail_total += max(0.0, float(mass[di_, dj_, dk_]) - _PHI_LO * rd)
            if avail_total < need - 1.0e-8:
                continue
            taken = 0.0
            for di_, dj_, dk_ in donors:
                if taken >= need - 1.0e-10:
                    break
                rd = float(max(rho[di_, dj_, dk_], 1.0e-3))
                md = float(mass[di_, dj_, dk_])
                avail = max(0.0, md - _PHI_LO * rd)
                give = min(avail, need - taken)
                if give <= 0.0:
                    continue
                mass[di_, dj_, dk_] = md - give
                phi[di_, dj_, dk_] = float(mass[di_, dj_, dk_]) / rd
                taken += give
            m_loc = m_loc + taken

        leftover = m_loc - r
        cell[i, j, k] = CELL_LIQUID
        mass[i, j, k] = r
        phi[i, j, k] = 1.0
        rho[i, j, k] = r

        if merge_thin:
            # Fold excess into existing upper IF.
            mass[i, j, above] = float(mass[i, j, above]) + max(0.0, leftover)
            ra = float(max(rho[i, j, above], r, 1.0e-3))
            rho[i, j, above] = ra
            phi[i, j, above] = float(mass[i, j, above]) / ra
        else:
            cell[i, j, above] = CELL_INTERFACE
            rho[i, j, above] = r
            mass[i, j, above] = leftover
            phi[i, j, above] = float(leftover / r)
        n_heal += 1

    return n_heal


def apply_vof_height_equation(
    buf: HomeVofGpuBuffers,
    *,
    rate: float = 0.05,
    u_max: float = 0.05,
    phi_dust: float = 0.05,
    dh_cap: float = 0.05,
    n_sweeps: int = 1,
    u_damp: float = 0.04,
    clean_airborne: bool = True,
) -> dict[str, float]:
    """Plane-wide ``φ → φ*`` on mode-k pool IF + airborne drop.

    ``dh_cap`` caps ``|Δφ|`` per cell per call.
    ``u_damp`` is intentionally weak (strong damping froze the left/right tilt).
    """
    del n_sweeps
    cell = buf.cell_type.numpy().astype(np.int32, copy=True)
    phi = buf.phi.numpy().astype(np.float32, copy=True)
    mass = buf.mass.numpy().astype(np.float32, copy=True)
    rho = buf.rho.numpy().astype(np.float32, copy=True)
    solid = buf.solid_phi.numpy()
    ux = buf.ux.numpy().astype(np.float32, copy=True)
    uy = buf.uy.numpy().astype(np.float32, copy=True)
    uz = buf.uz.numpy().astype(np.float32, copy=True)
    fluid = solid >= 0.0
    m_before = float(mass[fluid].sum())

    ii, jj, kk = _pool_surface_interfaces(
        cell, phi, solid, phi_dust=float(phi_dust)
    )
    n_if = int(ii.size)
    if n_if < 4:
        return {"n_wet": 0.0, "n_if": float(n_if), "dmass": 0.0, "phi_star": 0.0}

    mode_k = int(np.bincount(kk.astype(np.int64)).argmax())
    n_drop = 0
    if clean_airborne:
        n_drop = _drop_airborne_onto_pool(
            cell, rho, mass, phi, solid, mode_k=mode_k
        )
        if n_drop > 0:
            ii, jj, kk = _pool_surface_interfaces(
                cell, phi, solid, phi_dust=float(phi_dust)
            )
            n_if = int(ii.size)
            if n_if >= 4:
                mode_k = int(np.bincount(kk.astype(np.int64)).argmax())

    n_heal = 0
    n_boost = 0
    if n_if >= 4:
        n_heal = _heal_low_stairs(
            cell,
            rho,
            mass,
            phi,
            solid,
            ii,
            jj,
            kk,
            mode_k=mode_k,
            max_heals=64,
        )
        if n_heal > 0:
            ii, jj, kk = _pool_surface_interfaces(
                cell, phi, solid, phi_dust=float(phi_dust)
            )
            n_if = int(ii.size)
            if n_if >= 4:
                mode_k = int(np.bincount(kk.astype(np.int64)).argmax())
        if n_if >= 4:
            n_boost = _boost_thin_pool_if(
                cell,
                rho,
                mass,
                phi,
                solid,
                ii,
                jj,
                kk,
                mode_k=mode_k,
                max_boost=128,
            )
            if n_boost > 0:
                ii, jj, kk = _pool_surface_interfaces(
                    cell, phi, solid, phi_dust=float(phi_dust)
                )
                n_if = int(ii.size)
                if n_if >= 4:
                    mode_k = int(np.bincount(kk.astype(np.int64)).argmax())

    if n_if < 4:
        import warp as wp

        device = buf.device
        buf.cell_type.assign(wp.array(cell, dtype=wp.int32, device=device))
        buf.phi.assign(wp.array(phi, dtype=float, device=device))
        buf.mass.assign(wp.array(mass, dtype=float, device=device))
        buf.rho.assign(wp.array(rho, dtype=float, device=device))
        m_final = float(mass[fluid].sum())
        return {
            "n_wet": float(n_if),
            "n_if": float(n_if),
            "n_drop": float(n_drop),
            "n_heal": float(n_heal),
            "n_boost": float(n_boost),
            "mass_delta": m_final - m_before,
            "dmass": m_final - m_before,
            "phi_star": 0.0,
        }

    spd = np.sqrt(ux[ii, jj, kk] ** 2 + uy[ii, jj, kk] ** 2 + uz[ii, jj, kk] ** 2)
    u_mean = float(spd.mean())

    damp = float(np.clip(u_damp, 0.0, 0.5))
    if damp > 0.0:
        scale = np.float32(1.0 - damp)
        ux[ii, jj, kk] *= scale
        uy[ii, jj, kk] *= scale
        uz[ii, jj, kk] *= scale

    on_plane = kk == mode_k
    n_plane = int(on_plane.sum())
    skipped = 0.0
    phi_star = 0.0
    phi_std = 0.0
    n_touch = 0

    if u_mean > float(u_max) or n_plane < 4:
        skipped = 1.0
    else:
        sel_i = ii[on_plane]
        sel_j = jj[on_plane]
        sel_k = kk[on_plane]

        phi_s = phi[sel_i, sel_j, sel_k].astype(np.float64)
        mass_s = mass[sel_i, sel_j, sel_k].astype(np.float64)
        rho_s = np.maximum(rho[sel_i, sel_j, sel_k].astype(np.float64), 1.0e-3)
        m_plane0 = float(mass_s.sum())

        # Target inside the safe band so we never drive the plane into fill/empty.
        phi_star_raw = float(np.sum(phi_s * rho_s) / max(float(np.sum(rho_s)), 1.0e-6))
        phi_star = float(np.clip(phi_star_raw, _PHI_LO + 0.02, _PHI_HI - 0.02))

        alpha = float(np.clip(rate, 0.0, 1.0))
        dphi_cap = float(max(dh_cap, 1.0e-4))
        # Wall/corner contacts: stronger fill when below φ* (generic, not a fixed corner).
        dphi = np.empty(sel_i.size, dtype=np.float64)
        for t in range(sel_i.size):
            i, j, k = int(sel_i[t]), int(sel_j[t]), int(sel_k[t])
            n_wall = _xy_wall_contacts(i, j, k, solid)
            a = alpha
            if n_wall > 0 and float(phi_s[t]) < phi_star:
                a = min(1.0, alpha * (1.0 + 0.75 * float(n_wall)))
            dphi[t] = a * (phi_star - float(phi_s[t]))
        np.clip(dphi, -dphi_cap, dphi_cap, out=dphi)
        # Allow wall cells a slightly larger step when filling a depression.
        for t in range(sel_i.size):
            i, j, k = int(sel_i[t]), int(sel_j[t]), int(sel_k[t])
            if _xy_wall_contacts(i, j, k, solid) > 0 and dphi[t] > 0.0:
                dphi[t] = min(dphi[t], 1.5 * dphi_cap)
        n_touch = int(np.count_nonzero(np.abs(dphi) > 1.0e-12))

        phi_new = np.clip(phi_s + dphi, _PHI_LO, _PHI_HI)
        mass_new = phi_new * rho_s
        sum_m = float(mass_new.sum())
        if sum_m > 1.0e-6 and abs(sum_m - m_plane0) > 1.0e-8:
            mass_new *= m_plane0 / sum_m
        # After renorm, allow φ slightly outside the band so mass is not clipped away.
        phi_new = mass_new / rho_s
        # Soft pull of outliers back into band without destroying Σmass:
        over = phi_new > _PHI_HI
        under = phi_new < _PHI_LO
        if np.any(over) or np.any(under):
            phi_tgt = phi_new.copy()
            phi_tgt[over] = _PHI_HI
            phi_tgt[under] = _PHI_LO
            mass_tgt = phi_tgt * rho_s
            sum_t = float(mass_tgt.sum())
            if sum_t > 1.0e-6:
                mass_new = mass_tgt * (m_plane0 / sum_t)
                phi_new = mass_new / rho_s

        phi[sel_i, sel_j, sel_k] = phi_new.astype(np.float32)
        mass[sel_i, sel_j, sel_k] = mass_new.astype(np.float32)
        phi_star = float(np.clip(phi_new.mean(), _PHI_LO, _PHI_HI))
        phi_std = float(phi_new.std())

    m_final = float(mass[fluid].sum())

    import warp as wp

    device = buf.device
    buf.cell_type.assign(wp.array(cell, dtype=wp.int32, device=device))
    buf.phi.assign(wp.array(phi, dtype=float, device=device))
    buf.mass.assign(wp.array(mass, dtype=float, device=device))
    buf.rho.assign(wp.array(rho, dtype=float, device=device))
    buf.ux.assign(wp.array(ux, dtype=float, device=device))
    buf.uy.assign(wp.array(uy, dtype=float, device=device))
    buf.uz.assign(wp.array(uz, dtype=float, device=device))

    return {
        "n_wet": float(n_if),
        "n_if": float(n_if),
        "n_plane": float(n_plane),
        "n_touch": float(n_touch),
        "n_drop": float(n_drop),
        "n_heal": float(n_heal),
        "n_boost": float(n_boost),
        "mode_k": float(mode_k),
        "phi_star": float(phi_star),
        "H_star": float(mode_k) + float(phi_star),
        "phi_std": float(phi_std),
        "u_mean": u_mean,
        "alpha": float(np.clip(rate, 0.0, 1.0)),
        "n_shift": 0.0,
        "mass_before": m_before,
        "mass_after": m_final,
        "mass_delta": m_final - m_before,
        "dmass": m_final - m_before,
        "skipped": skipped,
    }
