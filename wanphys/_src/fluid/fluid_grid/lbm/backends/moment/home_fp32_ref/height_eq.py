# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Free-surface interface-φ leveling for HOME-FREE VOF (solver opt-in).

Target look: one dominant pool IF plane with nearly uniform ``φ`` (no large
high/low patches).

Per call::

    1) Drop airborne wet cells (gas immediately below) onto the pool IF in
       the same column — splash “balls”, mass-conserving.
    2) Heal one-cell **low stairs** (pool IF at ``mode_k−1``; gas or thin IF
       above): borrow from on-plane IF, promote — interior first; skip a
       2-cell domain-face halo (wall-first heal left a wrinkle strip).
    3) Boost thin/dust IF on the mode plane up to the safe φ band (same borrow;
       also skips the domain-face halo).
    4) Plane ``φ → φ*`` with uniform α (no wall boost) × soft body weight.
    5) Light tip ``|u|`` damping (weak).

Does not invent mid-air liquid (promote only with liquid below + gas above).
Rigid-adjacent pool IF uses a soft weight (Chebyshev distance → α): full
``φ→φ*`` far from bodies, none on the meniscus, linear blend in between so
a hard skip ring does not wrinkle the surface. Heal/boost only run where
weight≈1.

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
    """Count horizontal *domain-face* contacts (not rigid ``solid_phi``)."""
    del k
    nx, ny = int(solid.shape[0]), int(solid.shape[1])
    n = 0
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
            n += 1
    return n


def _xy_dist_to_domain_face(i: int, j: int, nx: int, ny: int) -> int:
    """Min cells to a domain face (0 = on the face row/col)."""
    return int(min(i, j, nx - 1 - i, ny - 1 - j))


def _has_body_solid_xy(
    i: int,
    j: int,
    k: int,
    solid: np.ndarray,
) -> bool:
    """True if a horizontal neighbor is rigid (``solid_phi < 0``)."""
    nx, ny, _nz = solid.shape
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
            continue
        if float(solid[ni, nj, k]) < 0.0:
            return True
    return False


def _xy_chebyshev_to_solid(
    i: int,
    j: int,
    k: int,
    solid: np.ndarray,
    *,
    max_r: int = 4,
) -> int:
    """Chebyshev distance in xy to nearest rigid cell (search ``k±1`` band).

    Returns ``max_r + 1`` if none found within ``max_r``.
    """
    nx, ny, nz = solid.shape
    if float(solid[i, j, k]) < 0.0:
        return 0
    k0 = max(0, k - 1)
    k1 = min(nz - 1, k + 1)
    for r in range(0, max_r + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                    continue
                for kk in range(k0, k1 + 1):
                    if float(solid[ni, nj, kk]) < 0.0:
                        return r
    return max_r + 1


def _body_soft_weight(dist: int, *, r0: int = 1, r1: int = 4) -> float:
    """0 near rigid (≤r0), 1 far (≥r1), linear in between — avoids a hard ring."""
    if dist <= r0:
        return 0.0
    if dist >= r1:
        return 1.0
    return float(dist - r0) / float(r1 - r0)


def _pool_body_weights(
    ii: np.ndarray,
    jj: np.ndarray,
    kk: np.ndarray,
    solid: np.ndarray,
    *,
    max_r: int = 4,
) -> np.ndarray:
    """Soft weights in ``[0,1]`` for each pool-IF index."""
    w = np.ones(ii.size, dtype=np.float64)
    for t in range(ii.size):
        d = _xy_chebyshev_to_solid(
            int(ii[t]), int(jj[t]), int(kk[t]), solid, max_r=max_r
        )
        w[t] = _body_soft_weight(d, r0=1, r1=max_r)
    return w


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
    """Raise under-filled pool IF up to ``_PHI_LO``.

    Dust IF (``φ < phi_dust``) is invisible to height maps and reads as a
    one-cell pit. Borrow from on-plane neighbors. Skips a 2-cell domain-face
    halo — wall-prioritized boost used to leave a wrinkle one cell in from
    the wall.
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
            if _xy_dist_to_domain_face(i, j, nx, ny) <= 2:
                continue
            if solid[i, j, mode_k] < 0.0:
                continue
            if _has_body_solid_xy(i, j, mode_k, solid):
                continue
            if int(cell[i, j, mode_k]) != CELL_INTERFACE:
                continue
            if mode_k > 0 and solid[i, j, mode_k - 1] >= 0.0:
                if int(cell[i, j, mode_k - 1]) != CELL_LIQUID:
                    continue
            p = float(phi[i, j, mode_k])
            if p >= _PHI_LO - 1.0e-6:
                continue
            # Prefer thinner cells first (no wall boost).
            candidates.append((int(round(p * 1000.0)), i, j, mode_k))
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

    Borrows fill(+seed) from on-plane IF. Skips a 2-cell domain-face halo —
    wall-first heal used to leave a parallel wrinkle just inside the wall.
    """
    nx, ny, nz = cell.shape
    if mode_k <= 0 or mode_k >= nz:
        return 0

    key = np.full((nx, ny), -1, dtype=np.int32)
    for t in range(ii.size):
        key[int(ii[t]), int(jj[t])] = t

    plane_idx = [t for t in range(ii.size) if int(kk[t]) == mode_k]
    if not plane_idx:
        pass

    low_idx = [t for t in range(ii.size) if int(kk[t]) == mode_k - 1]

    def _prio(t: int) -> float:
        # Interior first (large face-distance), then thinner φ.
        i, j, k = int(ii[t]), int(jj[t]), int(kk[t])
        return (-_xy_dist_to_domain_face(i, j, nx, ny), float(phi[i, j, k]))

    low_idx.sort(key=_prio)
    n_heal = 0
    seed_phi = 0.5 * (_PHI_LO + _PHI_HI)

    for t in low_idx:
        if n_heal >= max_heals:
            break
        i, j, k = int(ii[t]), int(jj[t]), int(kk[t])
        if _xy_dist_to_domain_face(i, j, nx, ny) <= 2:
            continue
        if solid[i, j, k] < 0.0 or int(cell[i, j, k]) != CELL_INTERFACE:
            continue
        if _has_body_solid_xy(i, j, k, solid):
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
    use_gpu: bool = True,
    sync_stats: bool = False,
) -> dict[str, float]:
    """Plane-wide ``φ → φ*`` on mode-k pool IF.

    Default ``use_gpu=True`` runs in-place Warp kernels (no full-field D2H).
    Host path kept for debugging; it is far slower (Python grid scans).
    """
    del n_sweeps, clean_airborne
    if use_gpu:
        from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.height_eq_warp import (
            apply_vof_height_equation_gpu,
        )

        # Force tip u_damp=0 on GPU: host default damping seeds circular waves.
        del u_damp
        return apply_vof_height_equation_gpu(
            buf,
            rate=float(rate),
            u_max=float(u_max),
            phi_dust=float(phi_dust),
            dh_cap=float(dh_cap),
            u_damp=0.0,
            sync_stats=bool(sync_stats),
        )
    return _apply_vof_height_equation_host(
        buf,
        rate=float(rate),
        u_max=float(u_max),
        phi_dust=float(phi_dust),
        dh_cap=float(dh_cap),
        u_damp=float(u_damp),
    )


def _apply_vof_height_equation_host(
    buf: HomeVofGpuBuffers,
    *,
    rate: float = 0.05,
    u_max: float = 0.05,
    phi_dust: float = 0.05,
    dh_cap: float = 0.05,
    u_damp: float = 0.04,
) -> dict[str, float]:
    """Legacy host implementation (slow — full D2H + Python loops)."""
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
        return {
            "n_wet": 0.0,
            "n_if": float(n_if),
            "n_body_skip": 0.0,
            "dmass": 0.0,
            "phi_star": 0.0,
            "gpu": 0.0,
        }

    body_w = _pool_body_weights(ii, jj, kk, solid, max_r=4)
    mode_mask = body_w >= 0.5
    if int(mode_mask.sum()) >= 4:
        mode_k = int(np.bincount(kk[mode_mask].astype(np.int64)).argmax())
    else:
        mode_k = int(np.bincount(kk.astype(np.int64)).argmax())
    n_body_skip = int((body_w < 1.0e-12).sum())
    n_drop = 0
    n_heal = 0
    n_boost = 0

    # Host path: φ→φ* only (heal/drop deferred — too expensive + wrinkle-prone).
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

    if body_w.shape[0] != ii.size:
        body_w = _pool_body_weights(ii, jj, kk, solid, max_r=4)
        n_body_skip = int((body_w < 1.0e-12).sum())

    if u_mean > float(u_max) or n_plane < 4:
        skipped = 1.0
    else:
        sel_i = ii[on_plane]
        sel_j = jj[on_plane]
        sel_k = kk[on_plane]
        sel_w = body_w[on_plane]
        phi_s = phi[sel_i, sel_j, sel_k].astype(np.float64)
        mass_s = mass[sel_i, sel_j, sel_k].astype(np.float64)
        rho_s = np.maximum(rho[sel_i, sel_j, sel_k].astype(np.float64), 1.0e-3)
        m_plane0 = float(mass_s.sum())
        core = sel_w >= 0.5
        if int(core.sum()) >= 4:
            phi_star_raw = float(
                np.sum(phi_s[core] * rho_s[core] * sel_w[core])
                / max(float(np.sum(rho_s[core] * sel_w[core])), 1.0e-6)
            )
        else:
            phi_star_raw = float(
                np.sum(phi_s * rho_s) / max(float(np.sum(rho_s)), 1.0e-6)
            )
        phi_star = float(np.clip(phi_star_raw, _PHI_LO + 0.02, _PHI_HI - 0.02))
        alpha = float(np.clip(rate, 0.0, 1.0))
        dphi_cap = float(max(dh_cap, 1.0e-4))
        dphi = alpha * sel_w * (phi_star - phi_s)
        np.clip(dphi, -dphi_cap, dphi_cap, out=dphi)
        n_touch = int(np.count_nonzero(np.abs(dphi) > 1.0e-12))
        phi_new = np.clip(phi_s + dphi, _PHI_LO, _PHI_HI)
        mass_new = phi_new * rho_s
        sum_m = float(mass_new.sum())
        if sum_m > 1.0e-6 and abs(sum_m - m_plane0) > 1.0e-8:
            mass_new *= m_plane0 / sum_m
        phi_new = mass_new / rho_s
        phi[sel_i, sel_j, sel_k] = phi_new.astype(np.float32)
        mass[sel_i, sel_j, sel_k] = mass_new.astype(np.float32)
        core_phi = phi_new[sel_w >= 0.5] if int((sel_w >= 0.5).sum()) else phi_new
        phi_star = float(np.clip(float(core_phi.mean()), _PHI_LO, _PHI_HI))
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
        "n_body_skip": float(n_body_skip),
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
        "gpu": 0.0,
    }

