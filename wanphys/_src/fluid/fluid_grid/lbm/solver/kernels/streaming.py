# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""FullF/HOME population providers and transport link laws.

All production paths first form logical ``f*[Q]``.  Direct stream-to-moment
kernels remain as reference utilities, not as a boundary-bypassing solver path.
"""

from __future__ import annotations

import warp as wp

from .encoding import (
    direction_x,
    direction_y,
    direction_z,
    opposite_direction,
    reconstruct_home_population,
)


@wp.func
def _wrapped(value: int, size: int) -> int:
    result = value
    if result < 0:
        result += size
    elif result >= size:
        result -= size
    return result


@wp.func
def _fullf_incoming(
    q: int,
    i: int,
    j: int,
    k: int,
    f_post: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> float:
    si, sj, sk = i - direction_x(q), j - direction_y(q), k - direction_z(q)
    outside = False
    if si < 0 or si >= nx:
        if px != 0:
            si = _wrapped(si, nx)
        else:
            outside = True
    if sj < 0 or sj >= ny:
        if py != 0:
            sj = _wrapped(sj, ny)
        else:
            outside = True
    if sk < 0 or sk >= nz:
        if pz != 0:
            sk = _wrapped(sk, nz)
        else:
            outside = True
    local_idx = i * ny * nz + j * nz + k
    value = f_post[opposite_direction(q) * stride + local_idx]
    if not outside:
        if solid_phi[si, sj, sk] >= 0.0:
            source_idx = si * ny * nz + sj * nz + sk
            value = f_post[q * stride + source_idx]
    return value


@wp.func
def _home_at(
    q: int,
    i: int,
    j: int,
    k: int,
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float),
    jy: wp.array3d(dtype=float),
    jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
) -> float:
    return reconstruct_home_population(
        q, rho[i, j, k], jx[i, j, k], jy[i, j, k], jz[i, j, k],
        sxx[i, j, k], syy[i, j, k], szz[i, j, k],
        sxy[i, j, k], sxz[i, j, k], syz[i, j, k],
    )


@wp.func
def _home_incoming(
    q: int,
    i: int,
    j: int,
    k: int,
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float),
    jy: wp.array3d(dtype=float),
    jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    si, sj, sk = i - direction_x(q), j - direction_y(q), k - direction_z(q)
    outside = False
    if si < 0 or si >= nx:
        if px != 0:
            si = _wrapped(si, nx)
        else:
            outside = True
    if sj < 0 or sj >= ny:
        if py != 0:
            sj = _wrapped(sj, ny)
        else:
            outside = True
    if sk < 0 or sk >= nz:
        if pz != 0:
            sk = _wrapped(sk, nz)
        else:
            outside = True
    value = _home_at(
        opposite_direction(q), i, j, k,
        rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz,
    )
    if not outside:
        if solid_phi[si, sj, sk] >= 0.0:
            value = _home_at(
                q, si, sj, sk,
                rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz,
            )
    return value


@wp.kernel
def stream_fullf_to_populations_kernel(
    f_post: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    f_star: wp.array(dtype=float),
    px: int, py: int, pz: int,
    nx: int, ny: int, nz: int, stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    for q in range(19):
        f_star[q * stride + idx] = _fullf_incoming(
            q, i, j, k, f_post, solid_phi, px, py, pz, nx, ny, nz, stride
        )


@wp.kernel
def stream_home_to_populations_kernel(
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float), jy: wp.array3d(dtype=float), jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float), syy: wp.array3d(dtype=float), szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float), sxz: wp.array3d(dtype=float), syz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    f_star: wp.array(dtype=float),
    px: int, py: int, pz: int,
    nx: int, ny: int, nz: int, stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    for q in range(19):
        f_star[q * stride + idx] = _home_incoming(
            q, i, j, k, rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz,
            solid_phi, px, py, pz, nx, ny, nz,
        )


@wp.func
def _cut_link_fraction(phi_fluid: float, phi_solid: float) -> float:
    denominator = phi_fluid - phi_solid
    value = 0.5
    if denominator > 1.0e-12:
        value = wp.clamp(phi_fluid / denominator, 1.0e-6, 1.0)
    return value


@wp.func
def _is_open_corner(
    i: int,
    j: int,
    k: int,
    bc_types: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
) -> bool:
    face_count = 0
    touches_open = False
    if i == 0:
        face_count += 1
        touches_open = touches_open or bc_types[0] == 1 or bc_types[0] == 2 or bc_types[0] == 4
    if i == nx - 1:
        face_count += 1
        touches_open = touches_open or bc_types[1] == 1 or bc_types[1] == 2 or bc_types[1] == 4
    if j == 0:
        face_count += 1
        touches_open = touches_open or bc_types[2] == 1 or bc_types[2] == 2 or bc_types[2] == 4
    if j == ny - 1:
        face_count += 1
        touches_open = touches_open or bc_types[3] == 1 or bc_types[3] == 2 or bc_types[3] == 4
    if k == 0:
        face_count += 1
        touches_open = touches_open or bc_types[4] == 1 or bc_types[4] == 2 or bc_types[4] == 4
    if k == nz - 1:
        face_count += 1
        touches_open = touches_open or bc_types[5] == 1 or bc_types[5] == 2 or bc_types[5] == 4
    return face_count > 1 and touches_open


@wp.kernel
def apply_fullf_cut_link_transport_kernel(
    f_post: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    f_star: wp.array(dtype=float),
    bc_types: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Replace halfway bounce-back by Bouzidi cut-link interpolation."""

    i, j, k = wp.tid()
    if _is_open_corner(i, j, k, bc_types, nx, ny, nz):
        return
    idx = i * ny * nz + j * nz + k
    phi_fluid = solid_phi[i, j, k]
    if phi_fluid < 0.0:
        return
    for q in range(1, 19):
        cx, cy, cz = direction_x(q), direction_y(q), direction_z(q)
        si, sj, sk = i - cx, j - cy, k - cz
        if si < 0 or si >= nx or sj < 0 or sj >= ny or sk < 0 or sk >= nz:
            continue
        phi_solid = solid_phi[si, sj, sk]
        if phi_solid >= 0.0:
            continue

        fraction = _cut_link_fraction(phi_fluid, phi_solid)
        opposite = opposite_direction(q)
        reflected = f_post[opposite * stride + idx]
        if fraction < 0.5:
            ai, aj, ak = i + cx, j + cy, k + cz
            if ai < 0:
                if px != 0:
                    ai = _wrapped(ai, nx)
            elif ai >= nx:
                if px != 0:
                    ai = _wrapped(ai, nx)
            if aj < 0:
                if py != 0:
                    aj = _wrapped(aj, ny)
            elif aj >= ny:
                if py != 0:
                    aj = _wrapped(aj, ny)
            if ak < 0:
                if pz != 0:
                    ak = _wrapped(ak, nz)
            elif ak >= nz:
                if pz != 0:
                    ak = _wrapped(ak, nz)
            if (
                ai >= 0 and ai < nx
                and aj >= 0 and aj < ny
                and ak >= 0 and ak < nz
                and solid_phi[ai, aj, ak] >= 0.0
            ):
                away_idx = ai * ny * nz + aj * nz + ak
                away_reflected = f_post[opposite * stride + away_idx]
                reflected = (
                    2.0 * fraction * reflected
                    + (1.0 - 2.0 * fraction) * away_reflected
                )
        else:
            reflected = (
                reflected / (2.0 * fraction)
                + (2.0 * fraction - 1.0)
                * f_post[q * stride + idx]
                / (2.0 * fraction)
            )
        f_star[q * stride + idx] = reflected


@wp.kernel
def apply_home_cut_link_transport_kernel(
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float),
    jy: wp.array3d(dtype=float),
    jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    f_star: wp.array(dtype=float),
    bc_types: wp.array(dtype=wp.int32),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """HOME-provider variant of cut-link interpolation."""

    i, j, k = wp.tid()
    if _is_open_corner(i, j, k, bc_types, nx, ny, nz):
        return
    idx = i * ny * nz + j * nz + k
    phi_fluid = solid_phi[i, j, k]
    if phi_fluid < 0.0:
        return
    for q in range(1, 19):
        cx, cy, cz = direction_x(q), direction_y(q), direction_z(q)
        si, sj, sk = i - cx, j - cy, k - cz
        if si < 0 or si >= nx or sj < 0 or sj >= ny or sk < 0 or sk >= nz:
            continue
        phi_solid = solid_phi[si, sj, sk]
        if phi_solid >= 0.0:
            continue

        fraction = _cut_link_fraction(phi_fluid, phi_solid)
        opposite = opposite_direction(q)
        reflected = _home_at(
            opposite, i, j, k, rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz
        )
        if fraction < 0.5:
            ai, aj, ak = i + cx, j + cy, k + cz
            if ai < 0:
                if px != 0:
                    ai = _wrapped(ai, nx)
            elif ai >= nx:
                if px != 0:
                    ai = _wrapped(ai, nx)
            if aj < 0:
                if py != 0:
                    aj = _wrapped(aj, ny)
            elif aj >= ny:
                if py != 0:
                    aj = _wrapped(aj, ny)
            if ak < 0:
                if pz != 0:
                    ak = _wrapped(ak, nz)
            elif ak >= nz:
                if pz != 0:
                    ak = _wrapped(ak, nz)
            if (
                ai >= 0 and ai < nx
                and aj >= 0 and aj < ny
                and ak >= 0 and ak < nz
                and solid_phi[ai, aj, ak] >= 0.0
            ):
                away_reflected = _home_at(
                    opposite,
                    ai,
                    aj,
                    ak,
                    rho,
                    jx,
                    jy,
                    jz,
                    sxx,
                    syy,
                    szz,
                    sxy,
                    sxz,
                    syz,
                )
                reflected = (
                    2.0 * fraction * reflected
                    + (1.0 - 2.0 * fraction) * away_reflected
                )
        else:
            local_forward = _home_at(
                q, i, j, k, rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz
            )
            reflected = (
                reflected / (2.0 * fraction)
                + (2.0 * fraction - 1.0) * local_forward / (2.0 * fraction)
            )
        f_star[q * stride + idx] = reflected


@wp.func
def _store_moments(
    i: int, j: int, k: int,
    r: float, mx: float, my: float, mz: float,
    pxx: float, pyy: float, pzz: float, pxy: float, pxz: float, pyz: float,
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float), jy: wp.array3d(dtype=float), jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float), syy: wp.array3d(dtype=float), szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float), sxz: wp.array3d(dtype=float), syz: wp.array3d(dtype=float),
) -> None:
    rho[i, j, k] = r
    jx[i, j, k] = mx
    jy[i, j, k] = my
    jz[i, j, k] = mz
    sxx[i, j, k] = pxx - r / 3.0
    syy[i, j, k] = pyy - r / 3.0
    szz[i, j, k] = pzz - r / 3.0
    sxy[i, j, k] = pxy
    sxz[i, j, k] = pxz
    syz[i, j, k] = pyz


@wp.kernel
def stream_fullf_to_moments_kernel(
    f_post: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float), jy: wp.array3d(dtype=float), jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float), syy: wp.array3d(dtype=float), szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float), sxz: wp.array3d(dtype=float), syz: wp.array3d(dtype=float),
    px: int, py: int, pz: int,
    nx: int, ny: int, nz: int, stride: int,
) -> None:
    i, j, k = wp.tid()
    r, mx, my, mz = float(0.0), float(0.0), float(0.0), float(0.0)
    pxx, pyy, pzz = float(0.0), float(0.0), float(0.0)
    pxy, pxz, pyz = float(0.0), float(0.0), float(0.0)
    for q in range(19):
        fq = _fullf_incoming(q, i, j, k, f_post, solid_phi, px, py, pz, nx, ny, nz, stride)
        cx, cy, cz = float(direction_x(q)), float(direction_y(q)), float(direction_z(q))
        r += fq
        mx += cx * fq
        my += cy * fq
        mz += cz * fq
        pxx += cx * cx * fq
        pyy += cy * cy * fq
        pzz += cz * cz * fq
        pxy += cx * cy * fq
        pxz += cx * cz * fq
        pyz += cy * cz * fq
    _store_moments(i, j, k, r, mx, my, mz, pxx, pyy, pzz, pxy, pxz, pyz, rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz)


@wp.kernel
def stream_home_to_moments_kernel(
    in_rho: wp.array3d(dtype=float),
    in_jx: wp.array3d(dtype=float), in_jy: wp.array3d(dtype=float), in_jz: wp.array3d(dtype=float),
    in_sxx: wp.array3d(dtype=float), in_syy: wp.array3d(dtype=float), in_szz: wp.array3d(dtype=float),
    in_sxy: wp.array3d(dtype=float), in_sxz: wp.array3d(dtype=float), in_syz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float), jy: wp.array3d(dtype=float), jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float), syy: wp.array3d(dtype=float), szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float), sxz: wp.array3d(dtype=float), syz: wp.array3d(dtype=float),
    px: int, py: int, pz: int,
    nx: int, ny: int, nz: int,
) -> None:
    i, j, k = wp.tid()
    r, mx, my, mz = float(0.0), float(0.0), float(0.0), float(0.0)
    pxx, pyy, pzz = float(0.0), float(0.0), float(0.0)
    pxy, pxz, pyz = float(0.0), float(0.0), float(0.0)
    for q in range(19):
        fq = _home_incoming(
            q, i, j, k, in_rho, in_jx, in_jy, in_jz,
            in_sxx, in_syy, in_szz, in_sxy, in_sxz, in_syz,
            solid_phi, px, py, pz, nx, ny, nz,
        )
        cx, cy, cz = float(direction_x(q)), float(direction_y(q)), float(direction_z(q))
        r += fq
        mx += cx * fq
        my += cy * fq
        mz += cz * fq
        pxx += cx * cx * fq
        pyy += cy * cy * fq
        pzz += cz * cz * fq
        pxy += cx * cy * fq
        pxz += cx * cz * fq
        pyz += cy * cz * fq
    _store_moments(i, j, k, r, mx, my, mz, pxx, pyy, pzz, pxy, pxz, pyz, rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz)


@wp.kernel
def moment_velocity_kernel(
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float), jy: wp.array3d(dtype=float), jz: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float), uz: wp.array3d(dtype=float),
) -> None:
    i, j, k = wp.tid()
    inv_rho = 1.0 / wp.max(rho[i, j, k], 1.0e-12)
    ux[i, j, k] = jx[i, j, k] * inv_rho
    uy[i, j, k] = jy[i, j, k] * inv_rho
    uz[i, j, k] = jz[i, j, k] * inv_rho
