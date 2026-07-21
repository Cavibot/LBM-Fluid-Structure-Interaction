# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM free-surface marker-propagation kernels.

Implements the three-stage surface pipeline that runs AFTER
``stream_collide_bvh_kernel`` each timestep:

1. ``surface_1_kernel`` — marker propagation
2. ``surface_2_kernel`` — GI cell initialisation + IG neighbour handling
3. ``surface_3_kernel`` — VOF mass redistribution and flag transitions

PLIC / curvature / normal helpers are defined in ``kernels_fluid.py``
(to avoid circular imports).
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .kernels_fluid import calculate_f_eq_d3q27, calculate_phi


# ============================================================================
# surface_1 — marker propagation
# ============================================================================
# Ref: ``mrLbmSolverGpu3D.cu:444-477``


@wp.kernel
def surface_1_kernel(
    flag: wp.array3d(dtype=wp.uint8),            # [in/out]
    cx: wp.array(dtype=wp.int32),                  # [27]
    cy: wp.array(dtype=wp.int32),                  # [27]
    cz: wp.array(dtype=wp.int32),                  # [27]
    nx: int,
    ny: int,
    nz: int,
):
    """Interface marker propagation (Phase 1 of 3).

    For every TYPE_IF cell:
      - Neighbour TYPE_IG cells are forced to TYPE_I (prevent premature
        interface-to-gas transition).
      - Neighbour TYPE_G cells are set to TYPE_GI (gas-to-interface
        transition candidate).

    Ref: ``mrLbmSolverGpu3D.cu:444-477``.

    Parameters
    ----------
    flag:
        Per-cell type bitfield (modified in-place for neighbours).
    nx, ny, nz:
        Grid dimensions.
    """
    i, j, k = wp.tid()

    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return

    flagsn = int(flag[i, j, k])
    flagsn_sus = flagsn & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)

    if flagsn_sus == C.CellFlag.TYPE_IF:
        for di in range(1, 27):
            ni = i - int(cx[di])
            nj = j - int(cy[di])
            nk = k - int(cz[di])

            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue

            nflag = int(flag[ni, nj, nk])
            nflag_su = nflag & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
            nflag_r = nflag & (0xFF ^ C.CellFlag.TYPE_SU)

            if nflag_su == C.CellFlag.TYPE_IG:
                flag[ni, nj, nk] = wp.uint8(nflag_r | C.CellFlag.TYPE_I)
            elif nflag_su == C.CellFlag.TYPE_G:
                flag[ni, nj, nk] = wp.uint8(nflag_r | C.CellFlag.TYPE_GI)


# ============================================================================
# surface_2 — GI initialisation + IG neighbour handling
# ============================================================================
# Ref: ``mrLbmSolverGpu3D.cu:479-602``


@wp.kernel
def surface_2_kernel(
    f_mom_post: wp.array(dtype=float),            # [10 * N] [in/out]
    flag: wp.array3d(dtype=wp.uint8),              # [in/out]
    c_value: wp.array3d(dtype=float),              # [in/out]
    g_mom: wp.array(dtype=float),                  # [7 * N] [out]
    islet: wp.array3d(dtype=wp.int32),            # [in]
    merge_detector: wp.array3d(dtype=wp.int32),   # [out]
    cx: wp.array(dtype=wp.int32),                  # [27]
    cy: wp.array(dtype=wp.int32),                  # [27]
    cz: wp.array(dtype=wp.int32),                  # [27]
    w3d: wp.array(dtype=float),                    # [27]
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """GI initialisation and IG neighbour handling (Phase 2 of 3).

    Two independent branches:

    *GI branch* (``mrLbmSolverGpu3D.cu:494-567``):
      Average density, velocity and dissolved-gas concentration from
      neighbouring TYPE_F / TYPE_I / TYPE_IF cells; compute equilibrium
      HOME moments and D3Q7 gas distributions.

    *IG branch* (``mrLbmSolverGpu3D.cu:569-600``):
      For each TYPE_IG cell, convert neighbour TYPE_F / TYPE_IF to
      TYPE_I and set *merge_detector* = 1, unless the neighbour is an
      islet (isolated bubble).
    """
    i, j, k = wp.tid()

    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return

    idx = i * ny * nz + j * nz + k
    flagsn = int(flag[i, j, k])
    flagsn_sus = flagsn & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)

    # ==================================================================
    # Branch A: TYPE_GI — initialise newly created interface cells
    # ==================================================================
    if flagsn_sus == C.CellFlag.TYPE_GI:
        rhot = float(0.0)
        uxt = float(0.0)
        uyt = float(0.0)
        uzt = float(0.0)
        counter = float(0.0)
        rho_gt = float(0.0)
        c_k = float(0.0)

        for di in range(1, 27):
            ni = i - int(cx[di])
            nj = j - int(cy[di])
            nk = k - int(cz[di])

            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue

            n_idx = ni * ny * nz + nj * nz + nk
            nflags_sus = int(flag[ni, nj, nk]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)

            if (nflags_sus == C.CellFlag.TYPE_F
                    or nflags_sus == C.CellFlag.TYPE_I
                    or nflags_sus == C.CellFlag.TYPE_IF):
                counter += 1.0
                rhot += f_mom_post[0 * stride + n_idx]
                uxt += f_mom_post[1 * stride + n_idx]
                uyt += f_mom_post[2 * stride + n_idx]
                uzt += f_mom_post[3 * stride + n_idx]

                if di < 7:
                    rho_gt += c_value[ni, nj, nk]
                    c_k += 1.0

        if counter > 0.0:
            rhon = rhot / counter
            uxn = uxt / counter
            uyn = uyt / counter
            uzn = uzt / counter
        else:
            rhon = 1.0
            uxn = 0.0
            uyn = 0.0
            uzn = 0.0

        if c_k > 0.0:
            rho_gt_val = rho_gt / c_k
        else:
            rho_gt_val = 0.0

        # Compute equilibrium HOME moments from averaged (rho, u)
        pixx = float(0.0)
        pixy = float(0.0)
        pixz = float(0.0)
        piyy = float(0.0)
        piyz = float(0.0)
        pizz = float(0.0)
        inv_rhon = 1.0 / rhon if rhon > 0.0 else 1.0

        for di in range(27):
            feq_i = calculate_f_eq_d3q27(rhon, uxn, uyn, uzn, di)
            fval = feq_i + w3d[di]
            cxi = float(cx[di])
            cyi = float(cy[di])
            czi = float(cz[di])
            pixx += fval * cxi * cxi
            pixy += fval * cxi * cyi
            pixz += fval * cxi * czi
            piyy += fval * cyi * cyi
            piyz += fval * cyi * czi
            pizz += fval * czi * czi

        f_mom_post[0 * stride + idx] = rhon
        f_mom_post[1 * stride + idx] = uxn
        f_mom_post[2 * stride + idx] = uyn
        f_mom_post[3 * stride + idx] = uzn
        f_mom_post[4 * stride + idx] = pixx * inv_rhon - C.CS2
        f_mom_post[5 * stride + idx] = pixy * inv_rhon
        f_mom_post[6 * stride + idx] = pixz * inv_rhon
        f_mom_post[7 * stride + idx] = piyy * inv_rhon - C.CS2
        f_mom_post[8 * stride + idx] = piyz * inv_rhon
        f_mom_post[9 * stride + idx] = pizz * inv_rhon - C.CS2

        c_value[i, j, k] = rho_gt_val

        # D3Q7 gas equilibrium
        ux4 = uxn * 4.0
        uy4 = uyn * 4.0
        uz4 = uzn * 4.0
        for di in range(7):
            # D3Q7 weights: w0=0.25, w1-6=0.125; dirs from cx/cy/cz Warp arrays
            wg = 0.25 if di == 0 else 0.125
            g_eq_i = wg * rho_gt_val * (
                1.0
                + float(cx[di]) * ux4
                + float(cy[di]) * uy4
                + float(cz[di]) * uz4
            )
            g_mom[idx + di * stride] = g_eq_i

        return

    # ==================================================================
    # Branch B: TYPE_IG — convert neighbours to interface
    # ==================================================================
    if flagsn_sus == C.CellFlag.TYPE_IG:
        for di in range(1, 27):
            ni = i - int(cx[di])
            nj = j - int(cy[di])
            nk = k - int(cz[di])

            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue

            nflag = int(flag[ni, nj, nk])
            nflags_su = nflag & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
            nflag_r = nflag & (0xFF ^ C.CellFlag.TYPE_SU)

            if nflags_su == C.CellFlag.TYPE_F or nflags_su == C.CellFlag.TYPE_IF:
                if islet[ni, nj, nk] == 0:
                    flag[ni, nj, nk] = wp.uint8(nflag_r | C.CellFlag.TYPE_I)
                    merge_detector[ni, nj, nk] = 1
                else:
                    flag[i, j, k] = wp.uint8(nflag_r | C.CellFlag.TYPE_I)


# ============================================================================
# surface_3 — VOF mass redistribution
# ============================================================================
# Ref: ``mrLbmSolverGpu3D.cu:604-701``


@wp.kernel
def surface_3_kernel(
    f_mom_post: wp.array(dtype=float),            # [10 * N] [in]
    flag: wp.array3d(dtype=wp.uint8),              # [in/out]
    mass: wp.array3d(dtype=float),                  # [in/out]
    massex: wp.array3d(dtype=float),                # [out]
    phi: wp.array3d(dtype=float),                   # [in/out]
    tag_matrix: wp.array3d(dtype=wp.int32),         # [in/out]
    previous_tag: wp.array3d(dtype=wp.int32),       # [out]
    islet: wp.array3d(dtype=wp.int32),             # [in]
    delta_phi: wp.array3d(dtype=float),             # [out]
    delta_g: wp.array3d(dtype=float),               # [in/out]
    g_mom: wp.array(dtype=float),                   # [7 * N] [in]
    split_flag_gpu: wp.array(dtype=wp.int32),      # [1] [out]
    cx: wp.array(dtype=wp.int32),                  # [27]
    cy: wp.array(dtype=wp.int32),                  # [27]
    cz: wp.array(dtype=wp.int32),                  # [27]
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """VOF mass redistribution and flag transitions (Phase 3 of 3).

    For each non-solid cell:
      1. Adjust *mass* and *phi* according to the cell flag type.
      2. Handle flag transitions: IF->F, IG->G, GI->I.
      3. Distribute excess mass equally among fluid/interface neighbours.
      4. Accumulate *delta_phi* and *delta_g*.

    Ref: ``mrLbmSolverGpu3D.cu:604-701``.
    """
    i, j, k = wp.tid()

    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return

    idx = i * ny * nz + j * nz + k
    flagsn = int(flag[i, j, k])
    flagsn_sus = flagsn & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)

    if (flagsn_sus & C.CellFlag.TYPE_S) != 0:
        return

    if islet[i, j, k] == 1:
        previous_tag[i, j, k] = tag_matrix[i, j, k]
        tag_matrix[i, j, k] = -1
        return

    rhon = f_mom_post[0 * stride + idx]
    massn = mass[i, j, k]
    massexn = float(0.0)
    phin = float(0.0)

    # ==================================================================
    # Phase A: flag-based mass/phi handling
    # ==================================================================
    if flagsn_sus == C.CellFlag.TYPE_F:
        massexn = massn - rhon
        massn = rhon
        phin = 1.0
        previous_tag[i, j, k] = tag_matrix[i, j, k]
        tag_matrix[i, j, k] = -1

    elif flagsn_sus == C.CellFlag.TYPE_I:
        if massn > rhon:
            massexn = massn - rhon
        elif massn < 0.0:
            massexn = massn
        else:
            massexn = 0.0
        massn = wp.clamp(massn, 0.0, rhon)
        phin = calculate_phi(rhon, massn, C.CellFlag.TYPE_I)

    elif flagsn_sus == C.CellFlag.TYPE_G:
        massexn = massn
        massn = 0.0
        phin = 0.0

    elif flagsn_sus == C.CellFlag.TYPE_IF:
        flag[i, j, k] = wp.uint8(
            (flagsn & (0xFF ^ C.CellFlag.TYPE_SU)) | C.CellFlag.TYPE_F
        )
        # report_split (ref: mrLbmSolverGpu3D.cu:53-61, 649-652)
        # Cell transitions interface->fluid with a bubble tag:
        # record the split signal for subsequent bubble merge/split detection.
        prev_tag = tag_matrix[i, j, k]
        previous_tag[i, j, k] = prev_tag
        tag_matrix[i, j, k] = -1
        if prev_tag > 0:
            wp.atomic_add(split_flag_gpu, 0, 1)
        massexn = massn - rhon
        massn = rhon
        phin = 1.0

    elif flagsn_sus == C.CellFlag.TYPE_IG:
        flag[i, j, k] = wp.uint8(
            (flagsn & (0xFF ^ C.CellFlag.TYPE_SU)) | C.CellFlag.TYPE_G
        )
        massexn = massn
        massn = 0.0
        phin = 0.0

    elif flagsn_sus == C.CellFlag.TYPE_GI:
        flag[i, j, k] = wp.uint8(
            (flagsn & (0xFF ^ C.CellFlag.TYPE_SU)) | C.CellFlag.TYPE_I
        )
        if massn > rhon:
            massexn = massn - rhon
        elif massn < 0.0:
            massexn = massn
        else:
            massexn = 0.0
        massn = wp.clamp(massn, 0.0, rhon)
        phin = calculate_phi(rhon, massn, C.CellFlag.TYPE_I)

    # ==================================================================
    # Phase B: count fluid/interface neighbours
    # ==================================================================
    counter = int(0)
    for di in range(1, 27):
        ni = i - int(cx[di])
        nj = j - int(cy[di])
        nk = k - int(cz[di])

        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            continue

        nflags_su = int(flag[ni, nj, nk]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        if (nflags_su == C.CellFlag.TYPE_F
                or nflags_su == C.CellFlag.TYPE_I
                or nflags_su == C.CellFlag.TYPE_IF
                or nflags_su == C.CellFlag.TYPE_GI):
            counter += 1

    # ==================================================================
    # Phase C: excess mass distribution
    # ==================================================================
    if counter == 0:
        massn += massexn
        massexn = 0.0
    else:
        massexn = massexn / float(counter)

    # ==================================================================
    # Phase D: write-back
    # ==================================================================
    mass[i, j, k] = massn
    massex[i, j, k] = massexn
    delta_phi[i, j, k] = phin - phi[i, j, k]

    new_flag_sus = int(flag[i, j, k]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
    if new_flag_sus == C.CellFlag.TYPE_I:
        rhon_g = float(0.0)
        for dg in range(7):
            rhon_g += g_mom[idx + dg * stride]
        delta_g[i, j, k] -= rhon_g * delta_phi[i, j, k]

    phi[i, j, k] = phin