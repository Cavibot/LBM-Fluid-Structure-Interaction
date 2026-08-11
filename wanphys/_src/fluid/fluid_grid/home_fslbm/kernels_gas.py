# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM dissolved-gas kernels (D3Q7 CMR-MRT).

Truth source (reference CUDA):
- ``mrUtilFuncGpu3D.h:322-338`` — ``calculate_g_eq``
- ``mrUtilFuncGpu3D.h:474-518`` — CMR moment transforms
- ``mrLbmSolverGpu3D.cu:1644-1878`` — reconstruction / stream-collide / volume update
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C


@wp.struct
class GasPop7:
    """Seven D3Q7 values (distributions or central moments)."""

    v0: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float


@wp.func
def _cell_idx(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return i * ny * nz + j * nz + k


@wp.func
def _w_gas(di: int) -> float:
    if di == 0:
        return 0.25
    return 0.125


@wp.func
def _gas_s(mi: int) -> float:
    if mi == 0:
        return 1.0
    if mi <= 3:
        return 1.0 / 0.9
    return 1.5


@wp.func
def _pop_get(p: GasPop7, di: int) -> float:
    if di == 0:
        return p.v0
    if di == 1:
        return p.v1
    if di == 2:
        return p.v2
    if di == 3:
        return p.v3
    if di == 4:
        return p.v4
    if di == 5:
        return p.v5
    return p.v6


@wp.func
def _pop_set(p: GasPop7, di: int, val: float) -> GasPop7:
    if di == 0:
        p.v0 = val
    elif di == 1:
        p.v1 = val
    elif di == 2:
        p.v2 = val
    elif di == 3:
        p.v3 = val
    elif di == 4:
        p.v4 = val
    elif di == 5:
        p.v5 = val
    else:
        p.v6 = val
    return p


@wp.func
def g_eq_d3q7(rho: float, ux: float, uy: float, uz: float, di: int) -> float:
    """D3Q7 gas equilibrium for one direction (ref calculate_g_eq)."""
    ux4 = ux * 4.0
    uy4 = uy * 4.0
    uz4 = uz * 4.0
    w = _w_gas(di)
    if di == 0:
        return w * rho
    if di == 1:
        return w * (1.0 + ux4) * rho
    if di == 2:
        return w * (1.0 - ux4) * rho
    if di == 3:
        return w * (1.0 + uy4) * rho
    if di == 4:
        return w * (1.0 - uy4) * rho
    if di == 5:
        return w * (1.0 + uz4) * rho
    return w * (1.0 - uz4) * rho


@wp.func
def convert_to_central_moment_d3q7(ux: float, uy: float, uz: float, node: GasPop7) -> GasPop7:
    """Distributions -> central moments (ref mlConvertCmrMoment_d3q7)."""
    n0 = node.v0
    n1 = node.v1
    n2 = node.v2
    n3 = node.v3
    n4 = node.v4
    n5 = node.v5
    n6 = node.v6

    o0 = float(0.0)
    o1 = float(0.0)
    o2 = float(0.0)
    o3 = float(0.0)
    o4 = float(0.0)
    o5 = float(0.0)
    o6 = float(0.0)

    # Explicit 7 directions matching CX/CY/CZ[0..6]
    for k in range(7):
        if k == 0:
            ftemp = n0
            CX = 0.0 - ux
            CY = 0.0 - uy
            CZ = 0.0 - uz
        elif k == 1:
            ftemp = n1
            CX = 1.0 - ux
            CY = 0.0 - uy
            CZ = 0.0 - uz
        elif k == 2:
            ftemp = n2
            CX = -1.0 - ux
            CY = 0.0 - uy
            CZ = 0.0 - uz
        elif k == 3:
            ftemp = n3
            CX = 0.0 - ux
            CY = 1.0 - uy
            CZ = 0.0 - uz
        elif k == 4:
            ftemp = n4
            CX = 0.0 - ux
            CY = -1.0 - uy
            CZ = 0.0 - uz
        elif k == 5:
            ftemp = n5
            CX = 0.0 - ux
            CY = 0.0 - uy
            CZ = 1.0 - uz
        else:
            ftemp = n6
            CX = 0.0 - ux
            CY = 0.0 - uy
            CZ = -1.0 - uz
        o0 += ftemp
        o1 += ftemp * CX
        o2 += ftemp * CY
        o3 += ftemp * CZ
        o4 += ftemp * (CX * CX - CY * CY)
        o5 += ftemp * (CX * CX - CZ * CZ)
        o6 += ftemp * (CX * CX + CY * CY + CZ * CZ)

    out = GasPop7()
    out.v0 = o0
    out.v1 = o1
    out.v2 = o2
    out.v3 = o3
    out.v4 = o4
    out.v5 = o5
    out.v6 = o6
    return out


@wp.func
def convert_from_central_moment_d3q7(U: float, V: float, W: float, node: GasPop7) -> GasPop7:
    """Central moments -> distributions (ref mlConvertCmrF_d3q7)."""
    k0 = node.v0
    k1 = node.v1
    k2 = node.v2
    k3 = node.v3
    k4 = node.v4
    k5 = node.v5
    k6 = node.v6
    out = GasPop7()
    out.v0 = (
        -k0 * U * U - 2.0 * k1 * U - k0 * V * V - 2.0 * k2 * V
        - k0 * W * W - 2.0 * k3 * W + k0 - k6
    )
    out.v1 = (
        k1 / 2.0 + k4 / 6.0 + k5 / 6.0 + k6 / 6.0
        + (U * k0) / 2.0 + U * k1 + (U * U * k0) / 2.0
    )
    out.v2 = (
        k4 / 6.0 - k1 / 2.0 + k5 / 6.0 + k6 / 6.0
        - (U * k0) / 2.0 + U * k1 + (U * U * k0) / 2.0
    )
    out.v3 = (
        k2 / 2.0 - k4 / 3.0 + k5 / 6.0 + k6 / 6.0
        + (V * k0) / 2.0 + V * k2 + (V * V * k0) / 2.0
    )
    out.v4 = (
        k5 / 6.0 - k4 / 3.0 - k2 / 2.0 + k6 / 6.0
        - (V * k0) / 2.0 + V * k2 + (V * V * k0) / 2.0
    )
    out.v5 = (
        k3 / 2.0 + k4 / 6.0 - k5 / 3.0 + k6 / 6.0
        + (W * k0) / 2.0 + W * k3 + (W * W * k0) / 2.0
    )
    out.v6 = (
        k4 / 6.0 - k3 / 2.0 - k5 / 3.0 + k6 / 6.0
        - (W * k0) / 2.0 + W * k3 + (W * W * k0) / 2.0
    )
    return out


# ---------------------------------------------------------------------------
# Test wrappers (global 7-vector arrays)
# ---------------------------------------------------------------------------


@wp.kernel
def _kernel_g_eq(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    out: wp.array(dtype=float),
):
    for di in range(7):
        out[di] = g_eq_d3q7(rho, ux, uy, uz, di)


@wp.kernel
def _kernel_cmr_to_moment(
    ux: float,
    uy: float,
    uz: float,
    node: wp.array(dtype=float),
):
    p = GasPop7()
    p.v0 = node[0]
    p.v1 = node[1]
    p.v2 = node[2]
    p.v3 = node[3]
    p.v4 = node[4]
    p.v5 = node[5]
    p.v6 = node[6]
    p = convert_to_central_moment_d3q7(ux, uy, uz, p)
    node[0] = p.v0
    node[1] = p.v1
    node[2] = p.v2
    node[3] = p.v3
    node[4] = p.v4
    node[5] = p.v5
    node[6] = p.v6


@wp.kernel
def _kernel_cmr_from_moment(
    ux: float,
    uy: float,
    uz: float,
    node: wp.array(dtype=float),
):
    p = GasPop7()
    p.v0 = node[0]
    p.v1 = node[1]
    p.v2 = node[2]
    p.v3 = node[3]
    p.v4 = node[4]
    p.v5 = node[5]
    p.v6 = node[6]
    p = convert_from_central_moment_d3q7(ux, uy, uz, p)
    node[0] = p.v0
    node[1] = p.v1
    node[2] = p.v2
    node[3] = p.v3
    node[4] = p.v4
    node[5] = p.v5
    node[6] = p.v6


@wp.kernel
def _kernel_cmr_roundtrip(
    ux: float,
    uy: float,
    uz: float,
    node: wp.array(dtype=float),
):
    p = GasPop7()
    p.v0 = node[0]
    p.v1 = node[1]
    p.v2 = node[2]
    p.v3 = node[3]
    p.v4 = node[4]
    p.v5 = node[5]
    p.v6 = node[6]
    p = convert_to_central_moment_d3q7(ux, uy, uz, p)
    p = convert_from_central_moment_d3q7(ux, uy, uz, p)
    node[0] = p.v0
    node[1] = p.v1
    node[2] = p.v2
    node[3] = p.v3
    node[4] = p.v4
    node[5] = p.v5
    node[6] = p.v6


# ---------------------------------------------------------------------------
# Production kernels
# ---------------------------------------------------------------------------


@wp.kernel
def g_reconstruction_kernel(
    g_mom: wp.array(dtype=float),
    f_mom: wp.array(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    tag_matrix: wp.array3d(dtype=wp.int32),
    bubble_rho: wp.array(dtype=wp.float64),
    delta_g: wp.array3d(dtype=float),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    opposite: wp.array(dtype=wp.int32),
    henry_constant: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Henry-law interface exchange + gas-side fill (ref cu:1644-1747)."""
    i, j, k = wp.tid()
    flagsn = int(flag[i, j, k])
    flagsn_bo = flagsn & C.TYPE_BO_MASK
    flagsn_su = flagsn & C.TYPE_SU_MASK
    if flagsn_bo == C.TYPE_S or flagsn_su == C.TYPE_G:
        return

    idx = _cell_idx(i, j, k, ny, nz)

    rhon_g = float(0.0)
    for di in range(7):
        rhon_g += g_mom[di * stride + idx]

    ghn = GasPop7()
    gon = GasPop7()
    for di in range(7):
        gon = _pop_set(gon, di, g_mom[di * stride + idx])
        ni = i - int(cx[di])
        nj = j - int(cy[di])
        nk = k - int(cz[di])
        val = float(0.0)
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            val = g_eq_d3q7(rhon_g, 0.0, 0.0, 0.0, di)
        else:
            nflag = int(flag[ni, nj, nk])
            if (nflag & C.TYPE_BO_MASK) == C.TYPE_S:
                val = g_eq_d3q7(rhon_g, 0.0, 0.0, 0.0, di)
            else:
                nidx = _cell_idx(ni, nj, nk, ny, nz)
                val = g_mom[di * stride + nidx]
        ghn = _pop_set(ghn, di, val)
    gon = _pop_set(gon, 0, _pop_get(ghn, 0))

    if flagsn_su != C.TYPE_I:
        return

    uxn = f_mom[1 * stride + idx]
    uyn = f_mom[2 * stride + idx]
    uzn = f_mom[3 * stride + idx]

    rho_k = 1.0
    tag = tag_matrix[i, j, k]
    if tag > 0:
        rho_k = float(bubble_rho[tag - 1])
    in_rho = henry_constant / 4.0 * rho_k

    g_in = float(0.0)
    for di in range(1, 7):
        ni = i - int(cx[di])
        nj = j - int(cy[di])
        nk = k - int(cz[di])
        nsu = 0
        if ni >= 0 and ni < nx and nj >= 0 and nj < ny and nk >= 0 and nk < nz:
            nsu = int(flag[ni, nj, nk]) & C.TYPE_SU_MASK
        opp = opposite[di]
        if nsu == C.TYPE_F:
            g_in += _pop_get(ghn, di) - _pop_get(gon, opp)
    delta_g[i, j, k] = delta_g[i, j, k] + g_in

    for di in range(1, 7):
        ni = i - int(cx[di])
        nj = j - int(cy[di])
        nk = k - int(cz[di])
        nsu = 0
        if ni >= 0 and ni < nx and nj >= 0 and nj < ny and nk >= 0 and nk < nz:
            nsu = int(flag[ni, nj, nk]) & C.TYPE_SU_MASK
        if nsu == C.TYPE_G:
            opp = opposite[di]
            g_mom[di * stride + idx] = (
                g_eq_d3q7(in_rho, uxn, uyn, uzn, opp)
                - _pop_get(gon, opp)
                + g_eq_d3q7(in_rho, uxn, uyn, uzn, di)
            )


@wp.kernel
def g_stream_collide_kernel(
    g_mom: wp.array(dtype=float),
    g_mom_post: wp.array(dtype=float),
    f_mom: wp.array(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    src: wp.array3d(dtype=float),
    c_value: wp.array3d(dtype=float),
    islet: wp.array3d(dtype=wp.int32),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Pull-stream + CMR-MRT collide (ref cu:1750-1849)."""
    i, j, k = wp.tid()
    flagsn = int(flag[i, j, k])
    flagsn_bo = flagsn & C.TYPE_BO_MASK
    flagsn_su = flagsn & C.TYPE_SU_MASK
    if flagsn_bo == C.TYPE_S or flagsn_su == C.TYPE_G:
        return
    if islet[i, j, k] == 1:
        return

    idx = _cell_idx(i, j, k, ny, nz)

    rhon_g = float(0.0)
    for di in range(7):
        rhon_g += g_mom[di * stride + idx]

    pop_g = GasPop7()
    for di in range(7):
        ni = i - int(cx[di])
        nj = j - int(cy[di])
        nk = k - int(cz[di])
        val = float(0.0)
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            val = g_eq_d3q7(rhon_g, 0.0, 0.0, 0.0, di)
        else:
            nflag = int(flag[ni, nj, nk])
            if (nflag & C.TYPE_BO_MASK) == C.TYPE_S:
                val = g_eq_d3q7(rhon_g, 0.0, 0.0, 0.0, di)
            else:
                nidx = _cell_idx(ni, nj, nk, ny, nz)
                val = g_mom[di * stride + nidx]
        pop_g = _pop_set(pop_g, di, val)

    uxn = f_mom[1 * stride + idx]
    uyn = f_mom[2 * stride + idx]
    uzn = f_mom[3 * stride + idx]

    rhon_gt = (
        pop_g.v0 + pop_g.v1 + pop_g.v2 + pop_g.v3 + pop_g.v4 + pop_g.v5 + pop_g.v6
    )
    c_value[i, j, k] = rhon_gt

    g_eq = GasPop7()
    for di in range(7):
        g_eq = _pop_set(g_eq, di, g_eq_d3q7(rhon_gt, uxn, uyn, uzn, di))

    pop_g = convert_to_central_moment_d3q7(uxn, uyn, uzn, pop_g)
    g_eq = convert_to_central_moment_d3q7(uxn, uyn, uzn, g_eq)

    src_cell = src[i, j, k]
    src_Q = GasPop7()
    for di in range(7):
        src_Q = _pop_set(src_Q, di, src_cell * _w_gas(di))
    src_Q = convert_to_central_moment_d3q7(uxn, uyn, uzn, src_Q)

    pop_out = GasPop7()
    for mi in range(7):
        s = _gas_s(mi)
        src_m = _pop_get(src_Q, mi) * (1.0 - s * 0.5)
        val = (1.0 - s) * _pop_get(pop_g, mi) + s * _pop_get(g_eq, mi) + src_m
        pop_out = _pop_set(pop_out, mi, val)

    pop_out = convert_from_central_moment_d3q7(uxn, uyn, uzn, pop_out)
    for di in range(7):
        g_mom_post[di * stride + idx] = _pop_get(pop_out, di)


@wp.kernel
def bubble_volume_g_update_kernel(
    delta_g: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    tag_matrix: wp.array3d(dtype=wp.int32),
    bubble_init_volume: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
):
    """Gas flux -> bubble init_volume; clear delta_g (ref cu:1853-1878)."""
    i, j, k = wp.tid()
    dg = delta_g[i, j, k]
    if dg != 0.0:
        tag = tag_matrix[i, j, k]
        if int(flag[i, j, k]) == C.TYPE_I and tag > 0:
            contrib = wp.float64(0.25 * float(dg) * float(phi[i, j, k]))
            wp.atomic_add(bubble_init_volume, tag - 1, contrib)
        delta_g[i, j, k] = 0.0
