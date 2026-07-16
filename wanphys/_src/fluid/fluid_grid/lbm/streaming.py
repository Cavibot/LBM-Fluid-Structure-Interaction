# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Pull-streaming specializations for FullF/HOME and EPC/EMC."""

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
