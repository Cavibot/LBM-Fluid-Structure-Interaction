# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for FSLBM-style sharp free-surface VOF (HOME-FREE Eq. 9–12).

H3: free-surface pressure BC can replace raw ``f_ī`` with Hermite-filtered
``\\bar f_ī`` (HOME-FREE §4.1: Eq. 11 using Eq. 16 reconstruction).
"""

from __future__ import annotations

import warp as wp

from ..core.hermite import home_reconstruct_f_i
from ..kernels import _f_eq, _moving_wall_correction, _trt_collide

# cell_type flags
CELL_GAS: int = 0
CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2


@wp.func
def _wrap_index(i: int, n: int, periodic: int) -> int:
    if i < 0:
        if periodic != 0:
            return n - 1
        return -1
    if i >= n:
        if periodic != 0:
            return 0
        return -1
    return i


@wp.func
def _neighbor_type(
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    cx: int,
    cy: int,
    cz: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """Return neighbour cell type, or -1 for solid / out-of-domain wall."""
    si = _wrap_index(i - cx, nx, px)
    sj = _wrap_index(j - cy, ny, py)
    sk = _wrap_index(k - cz, nz, pz)
    if si < 0 or sj < 0 or sk < 0:
        return -1
    if solid_phi[si, sj, sk] < 0.0:
        return -1
    return int(cell_type[si, sj, sk])


@wp.kernel
def compute_moments_vof_kernel(
    f: wp.array(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    num_dirs: int,
    stride: int,
    nx: int,
    ny: int,
    nz: int,
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
) -> None:
    """Moments for liquid/interface cells; gas cells are zeroed."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_GAS:
        rho[i, j, k] = 0.0
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0
        return

    r = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)

    for d in range(num_dirs):
        val = f[d * stride + idx]
        r = r + val
        mx = mx + float(cx_arr[d]) * val
        my = my + float(cy_arr[d]) * val
        mz = mz + float(cz_arr[d]) * val

    inv_r = float(0.0)
    if r > 1.0e-12:
        inv_r = 1.0 / r
    # Reject non-finite / explosive moments early.
    if r != r or r > 1.0e6 or r < 0.0:
        rho[i, j, k] = 1.0
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0
        return

    vx = mx * inv_r
    vy = my * inv_r
    vz = mz * inv_r
    # Mach clamp (lattice): prevents FS BC / TRT blow-up.
    u_max = 0.1
    u2 = vx * vx + vy * vy + vz * vz
    if u2 > u_max * u_max:
        s = u_max / wp.sqrt(u2)
        vx = vx * s
        vy = vy * s
        vz = vz * s

    # Density clamp for collision equilibrium.
    r_min = 0.2
    r_max = 1.5
    if r < r_min:
        r = r_min
    if r > r_max:
        r = r_max

    rho[i, j, k] = r
    ux[i, j, k] = vx
    uy[i, j, k] = vy
    uz[i, j, k] = vz


@wp.func
def _local_trt_post(
    f_in: wp.array(dtype=float),
    stride: int,
    idx: int,
    d: int,
    opp: int,
    rho_c: float,
    ux_c: float,
    uy_c: float,
    uz_c: float,
    w: float,
    cx: int,
    cy: int,
    cz: int,
    omega_plus: float,
    omega_minus: float,
) -> float:
    f_src = f_in[d * stride + idx]
    f_opp = f_in[opp * stride + idx]
    return _trt_collide(
        f_src,
        f_opp,
        rho_c,
        ux_c,
        uy_c,
        uz_c,
        w,
        cx,
        cy,
        cz,
        omega_plus,
        omega_minus,
    )


@wp.func
def _clamp_fs_velocity(ux: float, uy: float, uz: float):
    u_max = 0.1
    u2 = ux * ux + uy * uy + uz * uz
    if u2 > u_max * u_max:
        s = u_max / wp.sqrt(u2)
        return ux * s, uy * s, uz * s
    return ux, uy, uz


@wp.func
def _home_s_from_f(
    f: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    num_dirs: int,
    stride: int,
    idx: int,
    rho: float,
):
    """HOME Eq. 7 second-moment tensor S from populations at one cell."""
    cs2 = 1.0 / 3.0
    axx = float(0.0)
    ayy = float(0.0)
    azz = float(0.0)
    axy = float(0.0)
    axz = float(0.0)
    ayz = float(0.0)
    for d in range(num_dirs):
        fi = f[d * stride + idx]
        cxi = float(cx_arr[d])
        cyi = float(cy_arr[d])
        czi = float(cz_arr[d])
        axx = axx + (cxi * cxi - cs2) * fi
        ayy = ayy + (cyi * cyi - cs2) * fi
        azz = azz + (czi * czi - cs2) * fi
        axy = axy + cxi * cyi * fi
        axz = axz + cxi * czi * fi
        ayz = ayz + cyi * czi * fi
    inv = 1.0 / rho
    return axx * inv, ayy * inv, azz * inv, axy * inv, axz * inv, ayz * inv


@wp.func
def _fs_reconstruct(
    f_opp_post: float,
    rho_g: float,
    ux: float,
    uy: float,
    uz: float,
    w: float,
    cx: int,
    cy: int,
    cz: int,
) -> float:
    """Classic free-surface pressure BC (Eq. 11) with raw ``f_ī``."""
    vx, vy, vz = _clamp_fs_velocity(ux, uy, uz)
    feq = _f_eq(w, rho_g, vx, vy, vz, cx, cy, cz)
    feq_opp = _f_eq(w, rho_g, vx, vy, vz, -cx, -cy, -cz)
    val = feq + feq_opp - f_opp_post
    if val < 0.0:
        val = 0.0
    return val


@wp.func
def _fs_reconstruct_home(
    rho_g: float,
    rho_c: float,
    ux: float,
    uy: float,
    uz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    w: float,
    cx: int,
    cy: int,
    cz: int,
) -> float:
    """HOME-FREE Eq. 11 using filtered ``\\bar f_ī`` (Hermite Eq. 16).

    ``f_i* = f_i^eq(ρ_g,u) + f_ī^eq(ρ_g,u) − \\bar f_ī(ρ,u,S)``
    where ``\\bar f_ī`` is third-order reconstruct at the interface cell.
    """
    vx, vy, vz = _clamp_fs_velocity(ux, uy, uz)
    feq = _f_eq(w, rho_g, vx, vy, vz, cx, cy, cz)
    feq_opp = _f_eq(w, rho_g, vx, vy, vz, -cx, -cy, -cz)
    bar_opp = home_reconstruct_f_i(
        rho_c,
        vx,
        vy,
        vz,
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        -cx,
        -cy,
        -cz,
        w,
    )
    val = feq + feq_opp - bar_opp
    if val < 0.0:
        val = 0.0
    return val


@wp.kernel
def vof_collide_stream_kernel(
    f_in: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    f_out: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    opp_arr: wp.array(dtype=wp.int32),
    omega_plus: float,
    omega_minus: float,
    rho_g0: float,
    gamma: float,
    kappa: wp.array3d(dtype=float),
    use_home_fs_filter: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    num_dirs: int,
) -> None:
    """Fused TRT collide + pull-stream with free-surface reconstruction.

    Gas density for Eq. (11)/(12): ``ρ_g = ρ_g0 - 6 γ κ`` (c_s² = 1/3).
    When ``use_home_fs_filter != 0``, unknown DFs use HOME-FREE ``\\bar f_ī``.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    ctype = int(cell_type[i, j, k])

    if solid_phi[i, j, k] < 0.0 or ctype == CELL_GAS:
        for d in range(num_dirs):
            f_out[d * stride + idx] = 0.0
        return

    # Eq. (12): ρ_g = (p_g - 2 γ κ) / c_s² = ρ_g0 - 6 γ κ.
    rho_g = rho_g0 - 6.0 * gamma * kappa[i, j, k]
    if rho_g < 0.2:
        rho_g = 0.2
    if rho_g > 1.8:
        rho_g = 1.8

    rho_c = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]
    if rho_c < 0.2:
        rho_c = 0.2
    if rho_c > 1.5:
        rho_c = 1.5

    sxx = float(0.0)
    syy = float(0.0)
    szz = float(0.0)
    sxy = float(0.0)
    sxz = float(0.0)
    syz = float(0.0)
    if use_home_fs_filter != 0:
        sxx, syy, szz, sxy, sxz, syz = _home_s_from_f(
            f_in, cx_arr, cy_arr, cz_arr, num_dirs, stride, idx, rho_c,
        )

    # Rest population: collide in place (no stream).
    f_out[0 * stride + idx] = _local_trt_post(
        f_in,
        stride,
        idx,
        0,
        0,
        rho_c,
        vx,
        vy,
        vz,
        w_arr[0],
        0,
        0,
        0,
        omega_plus,
        omega_minus,
    )

    for d in range(1, num_dirs):
        cx = int(cx_arr[d])
        cy = int(cy_arr[d])
        cz = int(cz_arr[d])
        w = w_arr[d]
        opp = int(opp_arr[d])

        si = _wrap_index(i - cx, nx, px)
        sj = _wrap_index(j - cy, ny, py)
        sk = _wrap_index(k - cz, nz, pz)

        is_wall = False
        if si < 0 or sj < 0 or sk < 0:
            is_wall = True
        elif solid_phi[si, sj, sk] < 0.0:
            is_wall = True

        if is_wall:
            f_opp_post = _local_trt_post(
                f_in,
                stride,
                idx,
                opp,
                d,
                rho_c,
                vx,
                vy,
                vz,
                w,
                -cx,
                -cy,
                -cz,
                omega_plus,
                omega_minus,
            )
            f_out[d * stride + idx] = f_opp_post + _moving_wall_correction(
                vel_solid_u,
                vel_solid_v,
                vel_solid_w,
                i,
                j,
                k,
                cx,
                cy,
                cz,
                w,
                rho_c,
                nx,
                ny,
                nz,
            )
        else:
            ntype = int(cell_type[si, sj, sk])
            if ntype == CELL_GAS:
                if use_home_fs_filter != 0:
                    f_out[d * stride + idx] = _fs_reconstruct_home(
                        rho_g,
                        rho_c,
                        vx,
                        vy,
                        vz,
                        sxx,
                        syy,
                        szz,
                        sxy,
                        sxz,
                        syz,
                        w,
                        cx,
                        cy,
                        cz,
                    )
                else:
                    f_opp_post = _local_trt_post(
                        f_in,
                        stride,
                        idx,
                        opp,
                        d,
                        rho_c,
                        vx,
                        vy,
                        vz,
                        w,
                        -cx,
                        -cy,
                        -cz,
                        omega_plus,
                        omega_minus,
                    )
                    f_out[d * stride + idx] = _fs_reconstruct(
                        f_opp_post, rho_g, vx, vy, vz, w, cx, cy, cz
                    )
            else:
                src_idx = si * ny * nz + sj * nz + sk
                f_out[d * stride + idx] = _local_trt_post(
                    f_in,
                    stride,
                    src_idx,
                    d,
                    opp,
                    rho[si, sj, sk],
                    ux[si, sj, sk],
                    uy[si, sj, sk],
                    uz[si, sj, sk],
                    w,
                    cx,
                    cy,
                    cz,
                    omega_plus,
                    omega_minus,
                )


@wp.kernel
def vof_update_phi_kernel(
    f_in: wp.array(dtype=float),
    phi_in: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    phi_out: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    opp_arr: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    num_dirs: int,
) -> None:
    """Körner mass exchange for interface cells (Eq. 9–10)."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    ctype = int(cell_type[i, j, k])
    phi_c = phi_in[i, j, k]

    if solid_phi[i, j, k] < 0.0:
        phi_out[i, j, k] = 0.0
        return

    if ctype != CELL_INTERFACE:
        phi_out[i, j, k] = phi_c
        return

    rho_c = rho[i, j, k]
    # Guard against tiny ρ (new / near-empty interface): dm/ρ would explode.
    if rho_c < 0.1:
        phi_out[i, j, k] = phi_c
        return

    dm = float(0.0)
    for d in range(1, num_dirs):
        cx = int(cx_arr[d])
        cy = int(cy_arr[d])
        cz = int(cz_arr[d])
        opp = int(opp_arr[d])

        # Neighbour in +c_d (paper: x + c_i)
        ni = _wrap_index(i + cx, nx, px)
        nj = _wrap_index(j + cy, ny, py)
        nk = _wrap_index(k + cz, nz, pz)
        if ni < 0 or nj < 0 or nk < 0:
            continue
        if solid_phi[ni, nj, nk] < 0.0:
            continue

        ntype = int(cell_type[ni, nj, nk])
        if ntype == CELL_GAS:
            continue

        # Eq. (10): θ = ½(φ + φ_n); liquid neighbours use φ_n = 1.
        phi_n = float(1.0)
        if ntype == CELL_INTERFACE:
            phi_n = phi_in[ni, nj, nk]
        theta = 0.5 * (phi_c + phi_n)

        n_idx = ni * ny * nz + nj * nz + nk
        f_opp_nb = f_in[opp * stride + n_idx]
        f_self = f_in[d * stride + idx]
        dm = dm + theta * (f_opp_nb - f_self)

    phi_new = phi_c + dm / rho_c
    phi_out[i, j, k] = phi_new


@wp.kernel
def vof_reclassify_flags_kernel(
    phi: wp.array3d(dtype=float),
    cell_type_in: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    cell_type_out: wp.array3d(dtype=wp.int32),
    fill_flag: wp.array3d(dtype=wp.int32),
    empty_flag: wp.array3d(dtype=wp.int32),
    excess_phi: wp.array3d(dtype=float),
    epsilon: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Mark filled / emptied interface cells; stash truncated φ for redistribute."""
    i, j, k = wp.tid()
    excess_phi[i, j, k] = 0.0
    if solid_phi[i, j, k] < 0.0:
        cell_type_out[i, j, k] = CELL_GAS
        phi[i, j, k] = 0.0
        fill_flag[i, j, k] = 0
        empty_flag[i, j, k] = 0
        return

    ctype = int(cell_type_in[i, j, k])
    phi_c = phi[i, j, k]
    fill_flag[i, j, k] = 0
    empty_flag[i, j, k] = 0
    cell_type_out[i, j, k] = ctype

    if ctype == CELL_INTERFACE:
        # Thürey thresholds; freshly nucleated cells must be seeded φ > ε
        # (see convert_neighbors) so they are not deleted on the next step.
        if phi_c >= 1.0 - epsilon:
            excess_phi[i, j, k] = phi_c - 1.0
            phi[i, j, k] = 1.0
            cell_type_out[i, j, k] = CELL_LIQUID
            fill_flag[i, j, k] = 1
        elif phi_c <= epsilon:
            excess_phi[i, j, k] = phi_c
            phi[i, j, k] = 0.0
            cell_type_out[i, j, k] = CELL_GAS
            empty_flag[i, j, k] = 1
        else:
            if phi_c < 0.0:
                excess_phi[i, j, k] = phi_c
                phi[i, j, k] = 0.0
            elif phi_c > 1.0:
                excess_phi[i, j, k] = phi_c - 1.0
                phi[i, j, k] = 1.0
    elif ctype == CELL_LIQUID:
        phi[i, j, k] = 1.0
    elif ctype == CELL_GAS:
        phi[i, j, k] = 0.0


@wp.kernel
def vof_redistribute_excess_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    excess_phi: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    num_dirs: int,
) -> None:
    """Spread truncated / orphan φ onto neighbouring interface cells."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    if int(cell_type[i, j, k]) != CELL_INTERFACE:
        return

    # Gather excess from neighbours that just filled/emptied / orphans.
    add = float(0.0)
    for d in range(1, num_dirs):
        ni = _wrap_index(i + int(cx_arr[d]), nx, px)
        nj = _wrap_index(j + int(cy_arr[d]), ny, py)
        nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
        if ni < 0 or nj < 0 or nk < 0:
            continue
        if solid_phi[ni, nj, nk] < 0.0:
            continue
        ex = excess_phi[ni, nj, nk]
        if ex == 0.0:
            continue
        # Count how many interface/liquid receivers that donor has.
        n_recv = float(0.0)
        for d2 in range(1, num_dirs):
            ri = _wrap_index(ni + int(cx_arr[d2]), nx, px)
            rj = _wrap_index(nj + int(cy_arr[d2]), ny, py)
            rk = _wrap_index(nk + int(cz_arr[d2]), nz, pz)
            if ri < 0 or rj < 0 or rk < 0:
                continue
            if solid_phi[ri, rj, rk] < 0.0:
                continue
            rt = int(cell_type[ri, rj, rk])
            if rt == CELL_INTERFACE:
                n_recv = n_recv + 1.0
        if n_recv > 0.0:
            add = add + ex / n_recv

    if add != 0.0:
        phi[i, j, k] = phi[i, j, k] + add


@wp.kernel
def vof_convert_neighbors_kernel(
    cell_type: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    fill_flag: wp.array3d(dtype=wp.int32),
    empty_flag: wp.array3d(dtype=wp.int32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    num_dirs: int,
) -> None:
    """Keep a closed interface layer after fill/empty conversions."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return

    ctype = int(cell_type[i, j, k])

    # Gas neighbour of a newly filled cell → interface
    if ctype == CELL_GAS:
        for d in range(1, num_dirs):
            ni = _wrap_index(i + int(cx_arr[d]), nx, px)
            nj = _wrap_index(j + int(cy_arr[d]), ny, py)
            nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
            if ni < 0 or nj < 0 or nk < 0:
                continue
            if solid_phi[ni, nj, nk] < 0.0:
                continue
            if fill_flag[ni, nj, nk] != 0:
                cell_type[i, j, k] = CELL_INTERFACE
                # Seed above empty threshold so the cell survives one step.
                phi[i, j, k] = 2.0e-3
                return

    # Liquid neighbour of a newly emptied cell → interface
    if ctype == CELL_LIQUID:
        for d in range(1, num_dirs):
            ni = _wrap_index(i + int(cx_arr[d]), nx, px)
            nj = _wrap_index(j + int(cy_arr[d]), ny, py)
            nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
            if ni < 0 or nj < 0 or nk < 0:
                continue
            if solid_phi[ni, nj, nk] < 0.0:
                continue
            if empty_flag[ni, nj, nk] != 0:
                cell_type[i, j, k] = CELL_INTERFACE
                phi[i, j, k] = 1.0
                return


@wp.kernel
def vof_fix_topology_kernel(
    cell_type: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    excess_phi: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    num_dirs: int,
) -> None:
    """Remove orphan interface films; stash φ into excess for redistribute.

    - Interface with no liquid neighbour → gas
    - Interface with no gas neighbour → liquid
    """
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    if int(cell_type[i, j, k]) != CELL_INTERFACE:
        return

    has_liquid = int(0)
    has_gas = int(0)
    has_interface = int(0)
    for d in range(1, num_dirs):
        ni = _wrap_index(i + int(cx_arr[d]), nx, px)
        nj = _wrap_index(j + int(cy_arr[d]), ny, py)
        nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
        if ni < 0 or nj < 0 or nk < 0:
            continue
        if solid_phi[ni, nj, nk] < 0.0:
            continue
        nt = int(cell_type[ni, nj, nk])
        if nt == CELL_LIQUID:
            has_liquid = 1
        elif nt == CELL_GAS:
            has_gas = 1
        elif nt == CELL_INTERFACE:
            has_interface = 1

    if has_liquid == 0 and has_interface == 0:
        # Truly isolated interface droplet/film fragment.
        excess_phi[i, j, k] = excess_phi[i, j, k] + phi[i, j, k]
        cell_type[i, j, k] = CELL_GAS
        phi[i, j, k] = 0.0
    elif has_gas == 0 and has_liquid == 1:
        excess_phi[i, j, k] = excess_phi[i, j, k] + (phi[i, j, k] - 1.0)
        cell_type[i, j, k] = CELL_LIQUID
        phi[i, j, k] = 1.0


@wp.kernel
def vof_init_new_interface_kernel(
    f: wp.array(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    cell_type_prev: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    rho_default: float,
    num_dirs: int,
) -> None:
    """Init gas→interface from neighbour macros (eq only; φ already seeded)."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    if solid_phi[i, j, k] < 0.0:
        return
    if int(cell_type[i, j, k]) != CELL_INTERFACE:
        return
    if int(cell_type_prev[i, j, k]) != CELL_GAS:
        return

    rho_sum = float(0.0)
    ux_sum = float(0.0)
    uy_sum = float(0.0)
    uz_sum = float(0.0)
    count = float(0.0)
    for d in range(1, num_dirs):
        ni = _wrap_index(i + int(cx_arr[d]), nx, px)
        nj = _wrap_index(j + int(cy_arr[d]), ny, py)
        nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
        if ni < 0 or nj < 0 or nk < 0:
            continue
        if solid_phi[ni, nj, nk] < 0.0:
            continue
        nt = int(cell_type[ni, nj, nk])
        if nt == CELL_LIQUID or nt == CELL_INTERFACE:
            rho_n = rho[ni, nj, nk]
            if rho_n > 0.1:
                rho_sum = rho_sum + rho_n
                ux_sum = ux_sum + ux[ni, nj, nk]
                uy_sum = uy_sum + uy[ni, nj, nk]
                uz_sum = uz_sum + uz[ni, nj, nk]
                count = count + 1.0

    rho_c = rho_default
    vx = float(0.0)
    vy = float(0.0)
    vz = float(0.0)
    if count > 0.0:
        inv = 1.0 / count
        rho_c = rho_sum * inv
        vx = ux_sum * inv
        vy = uy_sum * inv
        vz = uz_sum * inv

    # Standard FSLBM: equilibrium at neighbour-averaged macros.
    # φ is already seeded (~2e-3); do not invent extra VOF volume here.
    for d in range(num_dirs):
        f[d * stride + idx] = _f_eq(
            w_arr[d],
            rho_c,
            vx,
            vy,
            vz,
            int(cx_arr[d]),
            int(cy_arr[d]),
            int(cz_arr[d]),
        )


@wp.kernel
def vof_clamp_phi_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Final φ clamp consistent with cell type (after redistribute)."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        phi[i, j, k] = 0.0
        return
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_GAS:
        phi[i, j, k] = 0.0
    elif ctype == CELL_LIQUID:
        phi[i, j, k] = 1.0
    else:
        p = phi[i, j, k]
        if p < 0.0:
            phi[i, j, k] = 0.0
        if p > 1.0:
            phi[i, j, k] = 1.0


@wp.kernel
def vof_accumulate_volume_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    vol_liquid: wp.array(dtype=float),
    vol_interface: wp.array(dtype=float),
    n_interface: wp.array(dtype=wp.int32),
) -> None:
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_LIQUID:
        wp.atomic_add(vol_liquid, 0, 1.0)
    elif ctype == CELL_INTERFACE:
        wp.atomic_add(vol_interface, 0, phi[i, j, k])
        wp.atomic_add(n_interface, 0, 1)


@wp.kernel
def vof_project_interface_volume_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    scale: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Scale interface φ so total liquid volume matches the seed volume."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    if int(cell_type[i, j, k]) != CELL_INTERFACE:
        return
    p = phi[i, j, k] * scale
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    phi[i, j, k] = p


@wp.kernel
def vof_sanitize_distributions_kernel(
    f: wp.array(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    w_arr: wp.array(dtype=float),
    rho0: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    num_dirs: int,
) -> None:
    """Rescale or reset DFs when density / populations leave the safe band."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_GAS:
        return

    r = float(0.0)
    bad = int(0)
    for d in range(num_dirs):
        val = f[d * stride + idx]
        if val != val or val < -1.0e-6 or val > 1.0e3:
            bad = 1
        r = r + val

    if bad == 1 or r != r or r < 0.05 or r > 3.0:
        target = rho0
        if ctype == CELL_INTERFACE:
            p = phi[i, j, k]
            if p < 0.05:
                p = 0.05
            target = rho0 * p
        for d in range(num_dirs):
            f[d * stride + idx] = w_arr[d] * target
        return

    target = rho0
    if r > 1.25 * target or r < 0.75 * target:
        scale = target / r
        if scale > 1.1:
            scale = 1.1
        if scale < 0.9:
            scale = 0.9
        for d in range(num_dirs):
            f[d * stride + idx] = f[d * stride + idx] * scale


@wp.kernel
def vof_mild_topology_kernel(
    cell_type: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    excess_phi: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    num_dirs: int,
) -> None:
    """Light topology cleanup without wiping connected free-surface sheets.

    - Buried interface (no gas neighbour) → liquid
    - Isolated thin film (no liquid, no other interface, φ small) → gas
    """
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    if int(cell_type[i, j, k]) != CELL_INTERFACE:
        return

    has_liquid = int(0)
    has_gas = int(0)
    has_interface = int(0)
    for d in range(1, num_dirs):
        ni = _wrap_index(i + int(cx_arr[d]), nx, px)
        nj = _wrap_index(j + int(cy_arr[d]), ny, py)
        nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
        if ni < 0 or nj < 0 or nk < 0:
            continue
        if solid_phi[ni, nj, nk] < 0.0:
            continue
        nt = int(cell_type[ni, nj, nk])
        if nt == CELL_LIQUID:
            has_liquid = 1
        elif nt == CELL_GAS:
            has_gas = 1
        elif nt == CELL_INTERFACE:
            has_interface = 1

    if has_gas == 0 and has_liquid == 1:
        excess_phi[i, j, k] = excess_phi[i, j, k] + (phi[i, j, k] - 1.0)
        cell_type[i, j, k] = CELL_LIQUID
        phi[i, j, k] = 1.0
    elif has_liquid == 0 and has_interface == 0 and phi[i, j, k] < 0.05:
        excess_phi[i, j, k] = excess_phi[i, j, k] + phi[i, j, k]
        cell_type[i, j, k] = CELL_GAS
        phi[i, j, k] = 0.0


@wp.kernel
def vof_build_visual_field_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    visual: wp.array3d(dtype=float),
) -> None:
    """SSFR field: liquid=1, interface=φ, gas=0 (avoids hollow transparent shells)."""
    i, j, k = wp.tid()
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_LIQUID:
        visual[i, j, k] = 1.0
    elif ctype == CELL_INTERFACE:
        visual[i, j, k] = phi[i, j, k]
    else:
        visual[i, j, k] = 0.0


@wp.kernel
def vof_zero_gas_distributions_kernel(
    f: wp.array(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    num_dirs: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    if int(cell_type[i, j, k]) == CELL_GAS:
        for d in range(num_dirs):
            f[d * stride + idx] = 0.0


@wp.kernel
def apply_gravity_velocity_shift_vof_kernel(
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    gx: float,
    gy: float,
    gz: float,
    tau_shift: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Shan-Chen-style gravity shift on liquid/interface cells only.

    ``u_eq = u + τ · g``  (g is acceleration in lattice units).
    """
    i, j, k = wp.tid()
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_GAS:
        return
    ux[i, j, k] = ux[i, j, k] + tau_shift * gx
    uy[i, j, k] = uy[i, j, k] + tau_shift * gy
    uz[i, j, k] = uz[i, j, k] + tau_shift * gz
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]
    u_max = 0.1
    u2 = vx * vx + vy * vy + vz * vz
    if u2 > u_max * u_max:
        s = u_max / wp.sqrt(u2)
        ux[i, j, k] = vx * s
        uy[i, j, k] = vy * s
        uz[i, j, k] = vz * s


@wp.kernel
def apply_guo_force_vof_kernel(
    f: wp.array(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    fx: float,
    fy: float,
    fz: float,
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    num_dirs: int,
) -> None:
    """Guo body force on liquid/interface cells only.

    ``Deltaf_d = (1 - omega/2) · w_d · 3 · (c_d · F)`` (same as single-phase Guo).
    """
    i, j, k = wp.tid()
    ctype = int(cell_type[i, j, k])
    if ctype == CELL_GAS:
        return
    idx = i * ny * nz + j * nz + k
    factor = (1.0 - omega * 0.5) * 3.0

    for d in range(num_dirs):
        cx = float(cx_arr[d])
        cy = float(cy_arr[d])
        cz = float(cz_arr[d])
        w = w_arr[d]
        f[d * stride + idx] = f[d * stride + idx] + factor * w * (cx * fx + cy * fy + cz * fz)


@wp.kernel
def vof_seed_column_kernel(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    dam_x: int,
    fill_z: int,
    rho_liquid: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
    num_dirs: int,
) -> None:
    """Initialise a rectangular liquid column for dam-break VOF."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    is_liquid = 0
    if i < dam_x and k < fill_z:
        is_liquid = 1

    if is_liquid == 1:
        phi[i, j, k] = 1.0
        cell_type[i, j, k] = CELL_LIQUID
        rho = rho_liquid
    else:
        phi[i, j, k] = 0.0
        cell_type[i, j, k] = CELL_GAS
        rho = 0.0

    density[i, j, k] = rho
    for d in range(num_dirs):
        f[d * stride + idx] = w_arr[d] * rho


@wp.kernel
def vof_mark_initial_interface_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    num_dirs: int,
) -> None:
    """Convert liquid cells that touch gas into interface (φ = 1)."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    if int(cell_type[i, j, k]) != CELL_LIQUID:
        return

    for d in range(1, num_dirs):
        ni = _wrap_index(i + int(cx_arr[d]), nx, px)
        nj = _wrap_index(j + int(cy_arr[d]), ny, py)
        nk = _wrap_index(k + int(cz_arr[d]), nz, pz)
        if ni < 0 or nj < 0 or nk < 0:
            continue
        if solid_phi[ni, nj, nk] < 0.0:
            continue
        if int(cell_type[ni, nj, nk]) == CELL_GAS:
            cell_type[i, j, k] = CELL_INTERFACE
            phi[i, j, k] = 1.0
            return
