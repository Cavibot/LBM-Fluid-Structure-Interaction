# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 FullF/HOME population reconstruction and state encoding."""

from __future__ import annotations

import warp as wp


@wp.func
def direction_x(q: int) -> int:
    value = 0
    if q == 1 or q == 7 or q == 9 or q == 11 or q == 13:
        value = 1
    elif q == 2 or q == 8 or q == 10 or q == 12 or q == 14:
        value = -1
    return value


@wp.func
def direction_y(q: int) -> int:
    value = 0
    if q == 3 or q == 7 or q == 8 or q == 15 or q == 17:
        value = 1
    elif q == 4 or q == 9 or q == 10 or q == 16 or q == 18:
        value = -1
    return value


@wp.func
def direction_z(q: int) -> int:
    value = 0
    if q == 5 or q == 11 or q == 12 or q == 15 or q == 16:
        value = 1
    elif q == 6 or q == 13 or q == 14 or q == 17 or q == 18:
        value = -1
    return value


@wp.func
def direction_weight(q: int) -> float:
    value = 1.0 / 36.0
    if q == 0:
        value = 1.0 / 3.0
    elif q <= 6:
        value = 1.0 / 18.0
    return value


@wp.func
def opposite_direction(q: int) -> int:
    value = 0
    if q == 1:
        value = 2
    elif q == 2:
        value = 1
    elif q == 3:
        value = 4
    elif q == 4:
        value = 3
    elif q == 5:
        value = 6
    elif q == 6:
        value = 5
    elif q == 7:
        value = 10
    elif q == 8:
        value = 9
    elif q == 9:
        value = 8
    elif q == 10:
        value = 7
    elif q == 11:
        value = 14
    elif q == 12:
        value = 13
    elif q == 13:
        value = 12
    elif q == 14:
        value = 11
    elif q == 15:
        value = 18
    elif q == 16:
        value = 17
    elif q == 17:
        value = 16
    elif q == 18:
        value = 15
    return value


@wp.func
def equilibrium_population(
    q: int, rho: float, ux: float, uy: float, uz: float
) -> float:
    cx, cy, cz = direction_x(q), direction_y(q), direction_z(q)
    cu = float(cx) * ux + float(cy) * uy + float(cz) * uz
    u2 = ux * ux + uy * uy + uz * uz
    return direction_weight(q) * rho * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
    )


@wp.func
def reconstruct_home_population(
    q: int,
    rho: float,
    jx: float,
    jy: float,
    jz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
) -> float:
    """Third-order Hermite reconstruction from density-weighted moments.

    The third-order closure matches ``mlCalDistributionFourthOrderD3Q27AtIndex``
    in Home-FSLBM, restricted to the D3Q19 velocity set.  Pure cubic Hermite
    terms vanish on this lattice and the xyz term is never sampled.
    """
    inv_rho = 1.0 / wp.max(rho, 1.0e-12)
    ux, uy, uz = jx * inv_rho, jy * inv_rho, jz * inv_rho

    axxy = -2.0 * rho * uy * ux * ux + 2.0 * sxy * ux + sxx * uy
    axyy = -2.0 * rho * ux * uy * uy + 2.0 * sxy * uy + syy * ux
    axxz = -2.0 * rho * uz * ux * ux + 2.0 * sxz * ux + sxx * uz
    axzz = -2.0 * rho * ux * uz * uz + 2.0 * sxz * uz + szz * ux
    ayyz = -2.0 * rho * uz * uy * uy + 2.0 * syz * uy + syy * uz
    ayzz = -2.0 * rho * uy * uz * uz + 2.0 * syz * uz + szz * uy

    cx = float(direction_x(q))
    cy = float(direction_y(q))
    cz = float(direction_z(q))
    cs2 = 1.0 / 3.0
    hxx, hyy, hzz = cx * cx - cs2, cy * cy - cs2, cz * cz - cs2

    first = 3.0 * (cx * jx + cy * jy + cz * jz)
    second = 4.5 * (
        hxx * sxx + hyy * syy + hzz * szz
        + 2.0 * cx * cy * sxy + 2.0 * cx * cz * sxz
        + 2.0 * cy * cz * syz
    )
    third = 13.5 * (
        hxx * cy * axxy + hyy * cx * axyy
        + hxx * cz * axxz + hzz * cx * axzz
        + hyy * cz * ayyz + hzz * cy * ayzz
    )
    return direction_weight(q) * (rho + first + second + third)


@wp.kernel
def populations_to_home_kernel(
    f: wp.array(dtype=float),
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
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    r, mx, my, mz = float(0.0), float(0.0), float(0.0), float(0.0)
    pxx, pyy, pzz = float(0.0), float(0.0), float(0.0)
    pxy, pxz, pyz = float(0.0), float(0.0), float(0.0)
    for q in range(19):
        fq = f[q * stride + idx]
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
def home_to_populations_kernel(
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
    f: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    for q in range(19):
        f[q * stride + idx] = reconstruct_home_population(
            q, rho[i, j, k], jx[i, j, k], jy[i, j, k], jz[i, j, k],
            sxx[i, j, k], syy[i, j, k], szz[i, j, k],
            sxy[i, j, k], sxz[i, j, k], syz[i, j, k],
        )


@wp.kernel
def initialize_home_kernel(
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
    rho0: float,
    ux: float,
    uy: float,
    uz: float,
) -> None:
    i, j, k = wp.tid()
    rho[i, j, k] = rho0
    jx[i, j, k] = rho0 * ux
    jy[i, j, k] = rho0 * uy
    jz[i, j, k] = rho0 * uz
    sxx[i, j, k] = rho0 * ux * ux
    syy[i, j, k] = rho0 * uy * uy
    szz[i, j, k] = rho0 * uz * uz
    sxy[i, j, k] = rho0 * ux * uy
    sxz[i, j, k] = rho0 * ux * uz
    syz[i, j, k] = rho0 * uy * uz
