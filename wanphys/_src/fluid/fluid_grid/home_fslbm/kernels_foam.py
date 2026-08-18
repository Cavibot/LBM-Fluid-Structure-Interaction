# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM foam kernels: disjoining pressure + open-tank atmosphere.

Truth source (reference CUDA):
- ``mrLbmSolverGpu3D.cu:144-272`` — ResetDisjoinForce / calculate_disjoint
- ``mrLbmSolverGpu3D.cu:1259-1298`` — atmosphere_rho / atmosphere_volme
- ``mrUtilFuncGpu3D.h:357-369`` — calculate_normal
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .kernels_fluid import calculate_phi, plic_cube


@wp.func
def _cell_idx(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return i * ny * nz + j * nz + k


@wp.func
def calculate_normal_from_phij(
    phij: wp.array(dtype=float),
    opposite: wp.array(dtype=wp.int32),
) -> wp.vec3:
    """27-pt normal after OPPOSITE remap (ref calculate_normal).

    ``phij[di]`` must be filled as phi at neighbour ``(i - cx[di], ...)``.
    """
    # p[di] = phij[opposite[di]] without a local wp.zeros buffer
    p0 = phij[opposite[0]]
    p1 = phij[opposite[1]]
    p2 = phij[opposite[2]]
    p3 = phij[opposite[3]]
    p4 = phij[opposite[4]]
    p5 = phij[opposite[5]]
    p6 = phij[opposite[6]]
    p7 = phij[opposite[7]]
    p8 = phij[opposite[8]]
    p9 = phij[opposite[9]]
    p10 = phij[opposite[10]]
    p11 = phij[opposite[11]]
    p12 = phij[opposite[12]]
    p13 = phij[opposite[13]]
    p14 = phij[opposite[14]]
    p15 = phij[opposite[15]]
    p16 = phij[opposite[16]]
    p17 = phij[opposite[17]]
    p18 = phij[opposite[18]]
    p19 = phij[opposite[19]]
    p20 = phij[opposite[20]]
    p21 = phij[opposite[21]]
    p22 = phij[opposite[22]]
    p23 = phij[opposite[23]]
    p24 = phij[opposite[24]]
    p25 = phij[opposite[25]]
    p26 = phij[opposite[26]]
    # silence unused (p0 only needed for completeness of remap convention)
    _ = p0

    bx = (
        4.0 * (p2 - p1)
        + 2.0 * (p8 - p7 + p10 - p9 + p14 - p13 + p16 - p15)
        + p20 - p19 + p22 - p21 + p24 - p23 + p25 - p26
    )
    by = (
        4.0 * (p4 - p3)
        + 2.0 * (p8 - p7 + p12 - p11 + p13 - p14 + p18 - p17)
        + p20 - p19 + p22 - p21 + p23 - p24 + p26 - p25
    )
    bz = (
        4.0 * (p6 - p5)
        + 2.0 * (p10 - p9 + p12 - p11 + p15 - p16 + p17 - p18)
        + p20 - p19 + p21 - p22 + p24 - p23 + p26 - p25
    )
    len_sq = bx * bx + by * by + bz * bz
    if len_sq < 1.0e-20:
        return wp.vec3(0.0, 0.0, 0.0)
    inv = 1.0 / wp.sqrt(len_sq)
    return wp.vec3(bx * inv, by * inv, bz * inv)


@wp.kernel
def calculate_disjoint_kernel(
    flag: wp.array3d(dtype=wp.uint8),
    phi: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    f_mom: wp.array(dtype=float),
    tag_matrix: wp.array3d(dtype=wp.int32),
    disjoin_force: wp.array3d(dtype=float),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    opposite: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Ray-cast disjoining pressure on TYPE_I cells.

    Ref: ``mrLbmSolverGpu3D.cu:162-272``. Distance formula uses normal.x
    denominator (match reference; do not "fix" anisotropy).
    """
    i, j, k = wp.tid()
    flagsn = int(flag[i, j, k])
    flagsn_su = flagsn & C.TYPE_SU_MASK
    if flagsn_su != C.TYPE_I:
        return

    idx = _cell_idx(i, j, k, ny, nz)
    massn = mass[i, j, k]

    phij = wp.zeros(27, dtype=float)
    for di in range(1, 27):
        ni = i - int(cx[di])
        nj = j - int(cy[di])
        nk = k - int(cz[di])
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            phij[di] = 0.0
            continue
        massn += massex[ni, nj, nk]
        pval = phi[ni, nj, nk]
        nflag = int(flag[ni, nj, nk])
        if (nflag & C.TYPE_BO_MASK) == C.TYPE_S:
            for fj in range(1, 7):
                si = ni - int(cx[fj])
                sj = nj - int(cy[fj])
                sk = nk - int(cz[fj])
                if si >= 0 and si < nx and sj >= 0 and sj < ny and sk >= 0 and sk < nz:
                    if (int(flag[si, sj, sk]) & C.TYPE_BO_MASK) != C.TYPE_S:
                        pval = phi[si, sj, sk]
                        break
        phij[di] = pval

    rhon = f_mom[0 * stride + idx]
    phij[0] = calculate_phi(rhon, massn, flagsn)

    tag_curind = tag_matrix[i, j, k] - 1

    # Inline calculate_normal (ref) with OPPOSITE remap — avoid local array as func arg
    p1 = phij[opposite[1]]
    p2 = phij[opposite[2]]
    p3 = phij[opposite[3]]
    p4 = phij[opposite[4]]
    p5 = phij[opposite[5]]
    p6 = phij[opposite[6]]
    p7 = phij[opposite[7]]
    p8 = phij[opposite[8]]
    p9 = phij[opposite[9]]
    p10 = phij[opposite[10]]
    p11 = phij[opposite[11]]
    p12 = phij[opposite[12]]
    p13 = phij[opposite[13]]
    p14 = phij[opposite[14]]
    p15 = phij[opposite[15]]
    p16 = phij[opposite[16]]
    p17 = phij[opposite[17]]
    p18 = phij[opposite[18]]
    p19 = phij[opposite[19]]
    p20 = phij[opposite[20]]
    p21 = phij[opposite[21]]
    p22 = phij[opposite[22]]
    p23 = phij[opposite[23]]
    p24 = phij[opposite[24]]
    p25 = phij[opposite[25]]
    p26 = phij[opposite[26]]
    bx = (
        4.0 * (p2 - p1)
        + 2.0 * (p8 - p7 + p10 - p9 + p14 - p13 + p16 - p15)
        + p20 - p19 + p22 - p21 + p24 - p23 + p25 - p26
    )
    by = (
        4.0 * (p4 - p3)
        + 2.0 * (p8 - p7 + p12 - p11 + p13 - p14 + p18 - p17)
        + p20 - p19 + p22 - p21 + p23 - p24 + p26 - p25
    )
    bz = (
        4.0 * (p6 - p5)
        + 2.0 * (p10 - p9 + p12 - p11 + p15 - p16 + p17 - p18)
        + p20 - p19 + p21 - p22 + p24 - p23 + p26 - p25
    )
    len_sq = bx * bx + by * by + bz * bz
    if len_sq < 1.0e-20:
        return
    inv = 1.0 / wp.sqrt(len_sq)
    normal = wp.vec3(bx * inv, by * inv, bz * inv)
    disjoint = float(0.0)

    for jk in range(1, 20):
        rx = int(wp.round(float(i) - float(jk) * 0.2 * normal.x))
        ry = int(wp.round(float(j) - float(jk) * 0.2 * normal.y))
        rz = int(wp.round(float(k) - float(jk) * 0.2 * normal.z))
        if rx >= 0 and rx < nx and ry >= 0 and ry < ny and rz >= 0 and rz < nz:
            rtag = tag_matrix[rx, ry, rz]
            if rtag > 0 and int(flag[rx, ry, rz]) == C.TYPE_I:
                tag_neighbor = rtag - 1
                if tag_curind != tag_neighbor:
                    center_offset = plic_cube(phij[0], normal)
                    alpha = phi[rx, ry, rz]
                    dis = wp.abs(float(jk) * 0.2 * normal.x) - (1.0 - alpha)
                    d = wp.abs(dis / (normal.x + 1.0e-8)) - center_offset
                    cand = 1.0 - d / 4.0
                    if disjoint < cand:
                        disjoint = cand

    if disjoint > 0.0:
        # Single writer per cell (this thread); atomic not required
        disjoin_force[i, j, k] = disjoin_force[i, j, k] + disjoint


@wp.kernel
def reset_disjoin_force_kernel(
    disjoin_force: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    """Clear disjoin_force and massex (ref ResetDisjoinForce)."""
    i, j, k = wp.tid()
    disjoin_force[i, j, k] = 0.0
    massex[i, j, k] = 0.0


@wp.kernel
def atmosphere_rho_update_kernel(
    tag_matrix: wp.array3d(dtype=wp.int32),
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
):
    """Open-tank atmosphere: large bubbles near outlet get rho=1.

    Ref: ``mrLbmSolverGpu3D.cu:1259-1282``.
    Condition: ``x == nx-2`` or ``z > nz-10``, and volume > 1e6.
    """
    i, j, k = wp.tid()
    if i == nx - 2 or k > nz - 10:
        tag = tag_matrix[i, j, k]
        if tag > 0:
            if bubble_volume[tag - 1] > wp.float64(1000000.0):
                # Match atomicExch semantics: overwrite density to 1.0
                bubble_rho[tag - 1] = wp.float64(1.0)


@wp.kernel
def atmosphere_volme_update_kernel(
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    bubble_count_gpu: wp.array(dtype=wp.int32),
):
    """Correct init_volume for atmosphere bubbles (rho == 1).

    Ref: ``mrLbmSolverGpu3D.cu:1286-1298`` (spelling volme preserved).
    """
    tid = wp.tid()
    if tid != 0:
        return
    bc = bubble_count_gpu[0]
    for b in range(bc):
        if bubble_rho[b] == wp.float64(1.0):
            bubble_init_volume[b] = bubble_rho[b] * bubble_volume[b]
