# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp HOME-FREE VOF (H6+), aligned with Home-FSLBM stream_collide + surface_*.

Order (ref ``mrLbmSolverGpu3D.cu``)::

  1. fused pull-stream + θ mass exchange + Eq.11 FS + moment collide
  2. surface_1 / surface_2 topology (IF/IG/GI)
  3. surface_3 excess-mass redistribute + φ = mass/ρ

Spike control comes from excess redistribution + closed-interface flags,
not from a separate Körner-φ pass that double-rebuilds ``f``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
    HomeFaceKind,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_step import (
    HomeVofState,
    seed_dam_break_column,
)
from wanphys._src.fluid.fluid_grid.lbm.core.hermite import home_reconstruct_f_i
from wanphys._src.fluid.fluid_grid.lbm.core.lattice import LatticeSpec, get_lattice_spec
from wanphys._src.fluid.fluid_grid.lbm.core.moments import home_collide_moments

CELL_GAS: int = 0
CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2
CELL_IF: int = 3  # interface → fluid
CELL_IG: int = 4  # interface → gas
CELL_GI: int = 5  # gas → interface

FACE_PERIODIC: int = 0
FACE_WALL: int = 1
FACE_ZOU_HE: int = 2


@wp.func
def _feq_w(
    w: float, rho: float, ux: float, uy: float, uz: float,
    cx: float, cy: float, cz: float,
) -> float:
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux * ux + uy * uy + uz * uz
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


@wp.func
def _clamp_u(ux: float, uy: float, uz: float):
    u2 = ux * ux + uy * uy + uz * uz
    if u2 > 0.16:  # |u| > 0.4 like Home-FSLBM
        s = 0.4 / wp.sqrt(u2)
        return ux * s, uy * s, uz * s
    return ux, uy, uz


@wp.func
def _solid_f_eq24(
    rho_x: float, ux_x: float, uy_x: float, uz_x: float,
    sxx_x: float, syy_x: float, szz_x: float,
    sxy_x: float, sxz_x: float, syz_x: float,
    ux_p: float, uy_p: float, uz_p: float,
    cx: int, cy: int, cz: int, w: float,
) -> float:
    sxx_p = ux_p * ux_p + (sxx_x - ux_x * ux_x)
    syy_p = uy_p * uy_p + (syy_x - uy_x * uy_x)
    szz_p = uz_p * uz_p + (szz_x - uz_x * uz_x)
    sxy_p = ux_p * uy_p + (sxy_x - ux_x * uy_x)
    sxz_p = ux_p * uz_p + (sxz_x - ux_x * uz_x)
    syz_p = uy_p * uz_p + (syz_x - uy_x * uz_x)
    return home_reconstruct_f_i(
        rho_x, ux_p, uy_p, uz_p, sxx_p, syy_p, szz_p, sxy_p, sxz_p, syz_p,
        cx, cy, cz, w,
    )


@wp.kernel
def home_vof_fused_kernel(
    rho_in: wp.array3d(dtype=float),
    ux_in: wp.array3d(dtype=float),
    uy_in: wp.array3d(dtype=float),
    uz_in: wp.array3d(dtype=float),
    sxx_in: wp.array3d(dtype=float),
    syy_in: wp.array3d(dtype=float),
    szz_in: wp.array3d(dtype=float),
    sxy_in: wp.array3d(dtype=float),
    sxz_in: wp.array3d(dtype=float),
    syz_in: wp.array3d(dtype=float),
    mass_in: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    rho_out: wp.array3d(dtype=float),
    ux_out: wp.array3d(dtype=float),
    uy_out: wp.array3d(dtype=float),
    uz_out: wp.array3d(dtype=float),
    sxx_out: wp.array3d(dtype=float),
    syy_out: wp.array3d(dtype=float),
    szz_out: wp.array3d(dtype=float),
    sxy_out: wp.array3d(dtype=float),
    sxz_out: wp.array3d(dtype=float),
    syz_out: wp.array3d(dtype=float),
    mass_out: wp.array3d(dtype=float),
    cell_out: wp.array3d(dtype=wp.int32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    opp_arr: wp.array(dtype=wp.int32),
    face_kind: wp.array(dtype=wp.int32),
    face_ux: wp.array(dtype=float),
    face_uy: wp.array(dtype=float),
    face_uz: wp.array(dtype=float),
    kappa: wp.array3d(dtype=float),
    gamma: float,
    num_dirs: int,
    tau: float,
    fx: float,
    fy: float,
    fz: float,
    rho_g0: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Fused pull-stream + mass + FS BC + collide (Home-FSLBM style)."""
    i, j, k = wp.tid()
    ctype = int(cell_type[i, j, k])
    # Normalize pending flags from previous surface pass
    if ctype == CELL_IF:
        ctype = CELL_LIQUID
    elif ctype == CELL_IG:
        ctype = CELL_GAS
    elif ctype == CELL_GI:
        ctype = CELL_INTERFACE

    if ctype == CELL_GAS:
        rho_out[i, j, k] = 0.0
        ux_out[i, j, k] = 0.0
        uy_out[i, j, k] = 0.0
        uz_out[i, j, k] = 0.0
        sxx_out[i, j, k] = 0.0
        syy_out[i, j, k] = 0.0
        szz_out[i, j, k] = 0.0
        sxy_out[i, j, k] = 0.0
        sxz_out[i, j, k] = 0.0
        syz_out[i, j, k] = 0.0
        mass_out[i, j, k] = 0.0
        cell_out[i, j, k] = CELL_GAS
        return

    rho_c = rho_in[i, j, k]
    if rho_c < 0.05:
        rho_c = 0.05
    vx = ux_in[i, j, k]
    vy = uy_in[i, j, k]
    vz = uz_in[i, j, k]
    sxx = sxx_in[i, j, k]
    syy = syy_in[i, j, k]
    szz = szz_in[i, j, k]
    sxy = sxy_in[i, j, k]
    sxz = sxz_in[i, j, k]
    syz = syz_in[i, j, k]
    vx, vy, vz = _clamp_u(vx, vy, vz)

    # Eq. 12: ρ_g = ρ_atm - 6 γ κ  (Laplace pressure → surface self-smoothing)
    rho_g = rho_g0 - 6.0 * gamma * kappa[i, j, k]
    if rho_g < 0.2:
        rho_g = 0.2
    if rho_g > 1.8:
        rho_g = 1.8

    fk0 = int(face_kind[0])
    fk1 = int(face_kind[1])
    fk2 = int(face_kind[2])
    fk3 = int(face_kind[3])
    fk4 = int(face_kind[4])
    fk5 = int(face_kind[5])

    # Collect excess mass from neighbors (Home-FSLBM)
    massn = mass_in[i, j, k]
    for d in range(1, num_dirs):
        cxi = int(cx_arr[d])
        cyi = int(cy_arr[d])
        czi = int(cz_arr[d])
        ni = i - cxi
        nj = j - cyi
        nk = k - czi
        if ni >= 0 and ni < nx and nj >= 0 and nj < ny and nk >= 0 and nk < nz:
            massn = massn + massex[ni, nj, nk]

    phi0 = float(1.0)
    if ctype == CELL_INTERFACE:
        if rho_c > 1.0e-6:
            phi0 = massn / rho_c
        if phi0 < 0.0:
            phi0 = 0.0
        if phi0 > 1.0:
            phi0 = 1.0

    cs2 = 1.0 / 3.0
    r = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)
    axx = float(0.0)
    ayy = float(0.0)
    azz = float(0.0)
    axy = float(0.0)
    axz = float(0.0)
    ayz = float(0.0)

    has_gas = int(0)
    has_fluid = int(0)

    # Rest population from local cell
    f0 = home_reconstruct_f_i(
        rho_c, vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, 0, 0, 0, w_arr[0],
    )
    r = r + f0

    for d in range(1, num_dirs):
        cxi = int(cx_arr[d])
        cyi = int(cy_arr[d])
        czi = int(cz_arr[d])
        w = w_arr[d]
        od = int(opp_arr[d])
        ocx = int(cx_arr[od])
        ocy = int(cy_arr[od])
        ocz = int(cz_arr[od])

        # Local pre-stream populations: fon[i] and fon[ī]
        fon_i = home_reconstruct_f_i(
            rho_c, vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, cxi, cyi, czi, w,
        )
        fon_opp = home_reconstruct_f_i(
            rho_c, vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, ocx, ocy, ocz, w_arr[od],
        )

        ni = i - cxi
        nj = j - cyi
        nk = k - czi
        is_wall = int(0)
        uxp = float(0.0)
        uyp = float(0.0)
        uzp = float(0.0)

        if ni < 0:
            if fk0 == FACE_PERIODIC:
                ni = nx - 1
            else:
                is_wall = 1
                uxp = face_ux[0]
                uyp = face_uy[0]
                uzp = face_uz[0]
        elif ni >= nx:
            if fk1 == FACE_PERIODIC:
                ni = 0
            else:
                is_wall = 1
                uxp = face_ux[1]
                uyp = face_uy[1]
                uzp = face_uz[1]

        if is_wall == 0:
            if nj < 0:
                if fk2 == FACE_PERIODIC:
                    nj = ny - 1
                else:
                    is_wall = 1
                    uxp = face_ux[2]
                    uyp = face_uy[2]
                    uzp = face_uz[2]
            elif nj >= ny:
                if fk3 == FACE_PERIODIC:
                    nj = 0
                else:
                    is_wall = 1
                    uxp = face_ux[3]
                    uyp = face_uy[3]
                    uzp = face_uz[3]

        if is_wall == 0:
            if nk < 0:
                if fk4 == FACE_PERIODIC:
                    nk = nz - 1
                else:
                    is_wall = 1
                    uxp = face_ux[4]
                    uyp = face_uy[4]
                    uzp = face_uz[4]
            elif nk >= nz:
                if fk5 == FACE_PERIODIC:
                    nk = 0
                else:
                    is_wall = 1
                    uxp = face_ux[5]
                    uyp = face_uy[5]
                    uzp = face_uz[5]

        fhn = float(0.0)
        ntype = CELL_GAS
        if is_wall != 0:
            # No-slip walls. (Vertical free-slip + wall Körner / rim-drain heuristics
            # were tried for 1-cell leveling but caused mass loss; see docs.)
            fhn = _solid_f_eq24(
                rho_c, vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                uxp, uyp, uzp, cxi, cyi, czi, w,
            )
        else:
            ntype = int(cell_type[ni, nj, nk])
            if ntype == CELL_IF:
                ntype = CELL_LIQUID
            elif ntype == CELL_IG:
                ntype = CELL_GAS
            elif ntype == CELL_GI:
                ntype = CELL_INTERFACE

            if ntype == CELL_GAS:
                has_gas = 1
                # Eq. 11 free-surface BC (interface and emergency liquid–gas).
                feg_i = _feq_w(w, rho_g, vx, vy, vz, float(cxi), float(cyi), float(czi))
                feg_o = _feq_w(
                    w_arr[od], rho_g, vx, vy, vz,
                    float(ocx), float(ocy), float(ocz),
                )
                fhn = feg_o - fon_opp + feg_i
                if fhn < 0.0:
                    fhn = 0.0
                # No mass exchange with gas.
            else:
                if ntype == CELL_LIQUID:
                    has_fluid = 1
                fhn = home_reconstruct_f_i(
                    rho_in[ni, nj, nk],
                    ux_in[ni, nj, nk],
                    uy_in[ni, nj, nk],
                    uz_in[ni, nj, nk],
                    sxx_in[ni, nj, nk],
                    syy_in[ni, nj, nk],
                    szz_in[ni, nj, nk],
                    sxy_in[ni, nj, nk],
                    sxz_in[ni, nj, nk],
                    syz_in[ni, nj, nk],
                    cxi, cyi, czi, w,
                )
                # Mass with F/I. Liquid also updates a shadow mass then
                # surface3 snaps mass→ρ and redistributes the residual via massex
                # (keeps Σmass≈Σρ_L+Σm_I from drifting one-sided).
                if ctype == CELL_LIQUID:
                    if ntype == CELL_LIQUID or ntype == CELL_INTERFACE:
                        massn = massn + (fhn - fon_opp)
                elif ctype == CELL_INTERFACE:
                    if ntype == CELL_LIQUID:
                        massn = massn + (fhn - fon_opp)
                    elif ntype == CELL_INTERFACE:
                        phi_n = phi[ni, nj, nk]
                        theta = 0.5 * (phi0 + phi_n)
                        massn = massn + theta * (fhn - fon_opp)

        fcx = float(cxi)
        fcy = float(cyi)
        fcz = float(czi)
        r = r + fhn
        mx = mx + fcx * fhn
        my = my + fcy * fhn
        mz = mz + fcz * fhn
        axx = axx + (fcx * fcx - cs2) * fhn
        ayy = ayy + (fcy * fcy - cs2) * fhn
        azz = azz + (fcz * fcz - cs2) * fhn
        axy = axy + fcx * fcy * fhn
        axz = axz + fcx * fcz * fhn
        ayz = ayz + fcy * fcz * fhn

    if r <= 1.0e-8:
        rho_out[i, j, k] = 0.0
        mass_out[i, j, k] = 0.0
        cell_out[i, j, k] = CELL_GAS
        return

    inv = 1.0 / r
    rho, ux, uy, uz, sxx_o, syy_o, szz_o, sxy_o, sxz_o, syz_o = home_collide_moments(
        r, mx * inv, my * inv, mz * inv,
        axx * inv, ayy * inv, azz * inv,
        axy * inv, axz * inv, ayz * inv,
        tau, fx, fy, fz,
    )
    ux, uy, uz = _clamp_u(ux, uy, uz)

    rho_out[i, j, k] = rho
    ux_out[i, j, k] = ux
    uy_out[i, j, k] = uy
    uz_out[i, j, k] = uz
    sxx_out[i, j, k] = sxx_o
    syy_out[i, j, k] = syy_o
    szz_out[i, j, k] = szz_o
    sxy_out[i, j, k] = sxy_o
    sxz_out[i, j, k] = sxz_o
    syz_out[i, j, k] = syz_o
    mass_out[i, j, k] = massn

    # Fill/empty by mass only. Do NOT use has_fluid==0 / has_gas==0:
    # thin wave crests are I–G only and would be erased (layer wipe).
    out_type = ctype
    if ctype == CELL_INTERFACE:
        if massn > rho + 1.0e-4:
            out_type = CELL_IF
        elif massn < -1.0e-4:
            out_type = CELL_IG
    elif ctype == CELL_LIQUID:
        out_type = CELL_LIQUID
    cell_out[i, j, k] = out_type


@wp.kernel
def home_vof_surface1_kernel(
    cell: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """IF cells: cancel neighbor IG; promote gas neighbors to GI (closed layer)."""
    i, j, k = wp.tid()
    if int(cell[i, j, k]) != CELL_IF:
        return
    # Full 26-neighborhood: face-only left F–G diagonal holes at corners
    # (persistent boundary dips that capillary cannot heal).
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue
                nt = int(cell[ni, nj, nk])
                if nt == CELL_IG:
                    cell[ni, nj, nk] = CELL_INTERFACE
                elif nt == CELL_GAS:
                    cell[ni, nj, nk] = CELL_GI


@wp.kernel
def home_vof_surface2_kernel(
    cell: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """IG: demote neighboring F/IF to I. GI: init moments from neighbors."""
    i, j, k = wp.tid()
    ctype = int(cell[i, j, k])

    if ctype == CELL_IG:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                        continue
                    nt = int(cell[ni, nj, nk])
                    if nt == CELL_LIQUID or nt == CELL_IF:
                        cell[ni, nj, nk] = CELL_INTERFACE
    elif ctype == CELL_GI:
        rhot = float(0.0)
        uxt = float(0.0)
        uyt = float(0.0)
        uzt = float(0.0)
        cnt = float(0.0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                        continue
                    nt = int(cell[ni, nj, nk])
                    if (
                        nt == CELL_LIQUID
                        or nt == CELL_INTERFACE
                        or nt == CELL_IF
                    ):
                        cnt = cnt + 1.0
                        rhot = rhot + rho[ni, nj, nk]
                        uxt = uxt + ux[ni, nj, nk]
                        uyt = uyt + uy[ni, nj, nk]
                        uzt = uzt + uz[ni, nj, nk]
        if cnt > 0.0:
            inv = 1.0 / cnt
            rr = rhot * inv
            uux = uxt * inv
            uuy = uyt * inv
            uuz = uzt * inv
            rho[i, j, k] = rr
            ux[i, j, k] = uux
            uy[i, j, k] = uuy
            uz[i, j, k] = uuz
            sxx[i, j, k] = uux * uux
            syy[i, j, k] = uuy * uuy
            szz[i, j, k] = uuz * uuz
            sxy[i, j, k] = uux * uuy
            sxz[i, j, k] = uux * uuz
            syz[i, j, k] = uuy * uuz
            mass[i, j, k] = 0.0  # filled by mass exchange; do not invent φρ
        else:
            rho[i, j, k] = 1.0
            mass[i, j, k] = 0.0


@wp.kernel
def home_vof_surface3_kernel(
    cell_in: wp.array3d(dtype=wp.int32),
    cell_out: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Commit IF/IG/GI; excess-mass redistribute; φ = mass/ρ."""
    i, j, k = wp.tid()
    ctype = int(cell_in[i, j, k])
    rhon = rho[i, j, k]
    massn = mass[i, j, k]
    massexn = float(0.0)
    phin = float(0.0)
    out_t = ctype

    if ctype == CELL_IF or ctype == CELL_LIQUID:
        out_t = CELL_LIQUID
        massexn = massn - rhon
        massn = rhon
        phin = 1.0
    elif ctype == CELL_IG:
        # Prefer dumping mass to wet neighbors; never evaporate mass into void.
        out_t = CELL_GAS
        massexn = massn
        massn = 0.0
        phin = 0.0
        rho[i, j, k] = 0.0
    elif ctype == CELL_GI or ctype == CELL_INTERFACE:
        out_t = CELL_INTERFACE
        if massn > rhon:
            massexn = massn - rhon
            massn = rhon
        elif massn < 0.0:
            massexn = massn
            massn = 0.0
        if rhon > 1.0e-8:
            phin = massn / rhon
        else:
            phin = 0.5
            if rhon < 0.05:
                rho[i, j, k] = 1.0
                rhon = 1.0
        if phin < 0.0:
            phin = 0.0
        if phin > 1.0:
            phin = 1.0
    else:
        out_t = CELL_GAS
        massexn = massn
        massn = 0.0
        phin = 0.0

    counter = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue
                nt = int(cell_in[ni, nj, nk])
                if (
                    nt == CELL_LIQUID
                    or nt == CELL_INTERFACE
                    or nt == CELL_IF
                    or nt == CELL_GI
                ):
                    counter = counter + 1

    if counter > 0:
        massex[i, j, k] = massexn / float(counter)
    else:
        # No interface neighbor to receive excess: keep mass on this cell.
        massn = massn + massexn
        massex[i, j, k] = 0.0
        if out_t == CELL_GAS and massn > 1.0e-6:
            out_t = CELL_INTERFACE
            if rho[i, j, k] < 0.05:
                rho[i, j, k] = 1.0
            rhon = rho[i, j, k]
        if out_t == CELL_LIQUID and wp.abs(massexn) > 1.0e-8:
            # Bulk fill with nowhere to dump excess → stay interface.
            out_t = CELL_INTERFACE
        if out_t == CELL_INTERFACE:
            if rhon > 1.0e-8:
                phin = massn / rhon
            else:
                phin = 0.5
            if phin < 0.0:
                phin = 0.0
            if phin > 1.0:
                phin = 1.0

    mass[i, j, k] = massn
    phi[i, j, k] = phin
    cell_out[i, j, k] = out_t


@wp.kernel
def home_vof_seal_fg_kernel(
    cell: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Liquid touching gas → interface (close F–G holes that erase waves)."""
    i, j, k = wp.tid()
    if int(cell[i, j, k]) != CELL_LIQUID:
        return
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                # Face neighbors only. Full 26-neigh seal turned near-surface
                # liquid into a thick I shell and froze A→B releveling (vx≈0).
                if wp.abs(di) + wp.abs(dj) + wp.abs(dk) != 1:
                    continue
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue
                if int(cell[ni, nj, nk]) == CELL_GAS:
                    cell[i, j, k] = CELL_INTERFACE
                    rr = rho[i, j, k]
                    if rr < 0.05:
                        rr = 1.0
                        rho[i, j, k] = rr
                    if mass[i, j, k] < 0.0:
                        mass[i, j, k] = 0.0
                    if mass[i, j, k] > rr:
                        mass[i, j, k] = rr
                    if rr > 1.0e-8:
                        phi[i, j, k] = mass[i, j, k] / rr
                    else:
                        phi[i, j, k] = 1.0
                    return


@wp.kernel
def home_vof_wall_wetting_kappa_kernel(
    kappa: wp.array3d(dtype=float),
    cell: wp.array3d(dtype=wp.int32),
    wall_wetting: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Optional Young κ bias on vertical-wall *interface* cells only.

    Strong / inset hydrophobic bias pulls the pool off the walls into a
    frustum (四棱台). Keep this mild and face-only when enabled; default off.
    ``wall_wetting`` ≈ cos θ_Y (negative → hydrophobic).
    """
    i, j, k = wp.tid()
    if wall_wetting == 0.0:
        return
    # Interface only — biasing bulk liquid next to walls accelerates wall peel.
    if int(cell[i, j, k]) != CELL_INTERFACE:
        return
    on_vwall = int(0)
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
        on_vwall = 1
    if on_vwall == 0:
        return
    kappa[i, j, k] = kappa[i, j, k] + wall_wetting


@wp.kernel
def home_vof_wall_film_drain_kernel(
    cell: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    phi_max: float,
    u_max: float,
    edge_only: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Subgrid rim-film drain on vertical walls (mass → ``massex``, like surface3).

    Targets climb coats with a dry inward (diagonal) face, near-stagnant
    (|u| < u_max), φ < phi_max. When ``edge_only!=0``, only vertical domain
    edges/corners (two walls) — faces stay intact, corners get cleaned.

    Inventory rule matches fused gather; no receivers → keep cell.
    """
    i, j, k = wp.tid()
    if int(cell[i, j, k]) != CELL_INTERFACE:
        return

    n_vwall = int(0)
    inward_i = i
    inward_j = j
    if i == 0:
        n_vwall = n_vwall + 1
        inward_i = 1
    elif i == nx - 1:
        n_vwall = n_vwall + 1
        inward_i = nx - 2
    if j == 0:
        n_vwall = n_vwall + 1
        inward_j = 1
    elif j == ny - 1:
        n_vwall = n_vwall + 1
        inward_j = ny - 2
    # Near-corner band (edge + 1 cell inset): tall spikes often sit off the exact edge.
    near_i = int(0)
    near_j = int(0)
    if i <= 1 or i >= nx - 2:
        near_i = 1
        if i == 1:
            inward_i = 2
        elif i == nx - 2:
            inward_i = nx - 3
    if j <= 1 or j >= ny - 2:
        near_j = 1
        if j == 1:
            inward_j = 2
        elif j == ny - 2:
            inward_j = ny - 3
    if edge_only != 0:
        if near_i == 0 or near_j == 0:
            return
    elif n_vwall == 0:
        return
    if k <= 0:
        return

    ux_c = ux[i, j, k]
    uy_c = uy[i, j, k]
    uz_c = uz[i, j, k]
    speed2 = ux_c * ux_c + uy_c * uy_c + uz_c * uz_c
    if speed2 > u_max * u_max:
        return

    phin = phi[i, j, k]
    if phin <= 1.0e-8:
        cell[i, j, k] = CELL_GAS
        mass[i, j, k] = 0.0
        phi[i, j, k] = 0.0
        rho[i, j, k] = 0.0
        return

    # Interior direction (away from the nearest vertical walls).
    si = int(0)
    sj = int(0)
    if i <= 1:
        si = 1
    elif i >= nx - 2:
        si = -1
    if j <= 1:
        sj = 1
    elif j >= ny - 2:
        sj = -1
    if edge_only != 0 and (si == 0 or sj == 0):
        return

    # Stranded climb: no pool liquid / thick I toward domain interior at this k.
    # (Diagonal-void failed when the whole corner column is an I film stack.)
    pool = int(0)
    for t in range(1, 3):
        ni = i + t * si
        nj = j + t * sj
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
            continue
        nt = int(cell[ni, nj, k])
        if nt == CELL_LIQUID:
            pool = 1
        elif nt == CELL_INTERFACE and phi[ni, nj, k] > 0.25:
            pool = 1
    if pool != 0:
        return

    # Need a wet path below so mass can fall into the pool inventory.
    supported = int(0)
    if k > 0:
        bt = int(cell[i, j, k - 1])
        if bt == CELL_LIQUID or bt == CELL_INTERFACE:
            supported = 1
        nb = int(cell[i + si, j + sj, k - 1]) if (
            i + si >= 0 and i + si < nx and j + sj >= 0 and j + sj < ny
        ) else CELL_GAS
        if nb == CELL_LIQUID or nb == CELL_INTERFACE:
            supported = 1
    if supported == 0:
        return
    if phin > phi_max:
        return

    # Must count ALL wet neighbors that will gather massex next fused step.
    counter = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = i + di
                nj = j + dj
                nk = k + dk
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue
                nt = int(cell[ni, nj, nk])
                if nt == CELL_LIQUID or nt == CELL_INTERFACE:
                    counter = counter + 1
    if counter == 0:
        return

    # Gentle fractional drain: avoid full I→G topology thrash that leaks Σmass.
    massn = mass[i, j, k]
    if massn <= 1.0e-8:
        cell[i, j, k] = CELL_GAS
        mass[i, j, k] = 0.0
        phi[i, j, k] = 0.0
        rho[i, j, k] = 0.0
        return
    drain = 0.12 * massn
    massex[i, j, k] = massex[i, j, k] + drain / float(counter)
    massn = massn - drain
    mass[i, j, k] = massn
    rhon = rho[i, j, k]
    if rhon < 0.05:
        rhon = 1.0
        rho[i, j, k] = rhon
    phin = massn / rhon
    if phin < 0.0:
        phin = 0.0
    if phin > 1.0:
        phin = 1.0
    phi[i, j, k] = phin
    if massn <= 1.0e-6:
        cell[i, j, k] = CELL_GAS
        mass[i, j, k] = 0.0
        phi[i, j, k] = 0.0
        rho[i, j, k] = 0.0


@wp.kernel
def home_vof_salvage_mass_on_gas_kernel(
    cell: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
) -> None:
    """Re-open INTERFACE if gas still holds mass (missed recv / race)."""
    i, j, k = wp.tid()
    if int(cell[i, j, k]) != CELL_GAS:
        return
    massn = mass[i, j, k]
    if massn <= 1.0e-6:
        mass[i, j, k] = 0.0
        phi[i, j, k] = 0.0
        return
    cell[i, j, k] = CELL_INTERFACE
    rhon = rho[i, j, k]
    if rhon < 0.05:
        rhon = 1.0
        rho[i, j, k] = rhon
    phin = massn / rhon
    if phin < 0.0:
        phin = 0.0
    if phin > 1.0:
        phin = 1.0
    phi[i, j, k] = phin


# ---------------------------------------------------------------------------
# Device buffers + driver
# ---------------------------------------------------------------------------


@dataclass
class HomeVofGpuBuffers:
    rho: wp.array
    ux: wp.array
    uy: wp.array
    uz: wp.array
    sxx: wp.array
    syy: wp.array
    szz: wp.array
    sxy: wp.array
    sxz: wp.array
    syz: wp.array
    rho_b: wp.array
    ux_b: wp.array
    uy_b: wp.array
    uz_b: wp.array
    sxx_b: wp.array
    syy_b: wp.array
    szz_b: wp.array
    sxy_b: wp.array
    sxz_b: wp.array
    syz_b: wp.array
    mass: wp.array
    mass_b: wp.array
    massex: wp.array
    phi: wp.array
    cell_type: wp.array
    cell_tmp: wp.array
    cell_tmp2: wp.array
    cx: wp.array
    cy: wp.array
    cz: wp.array
    w: wp.array
    opp: wp.array
    face_kind: wp.array
    face_ux: wp.array
    face_uy: wp.array
    face_uz: wp.array
    kappa: wp.array
    kappa_tmp: wp.array
    solid_phi: wp.array
    shape: tuple
    device: str
    num_dirs: int

    def swap_moment_buffers(self) -> None:
        self.rho, self.rho_b = self.rho_b, self.rho
        self.ux, self.ux_b = self.ux_b, self.ux
        self.uy, self.uy_b = self.uy_b, self.uy
        self.uz, self.uz_b = self.uz_b, self.uz
        self.sxx, self.sxx_b = self.sxx_b, self.sxx
        self.syy, self.syy_b = self.syy_b, self.syy
        self.szz, self.szz_b = self.szz_b, self.szz
        self.sxy, self.sxy_b = self.sxy_b, self.sxy
        self.sxz, self.sxz_b = self.sxz_b, self.sxz
        self.syz, self.syz_b = self.syz_b, self.syz
        self.mass, self.mass_b = self.mass_b, self.mass


def _face_arrays_from_bc(bc: HomeDomainBC, device: str):
    kinds, uxs, uys, uzs = [], [], [], []
    for face in (bc.xmin, bc.xmax, bc.ymin, bc.ymax, bc.zmin, bc.zmax):
        if face.kind == HomeFaceKind.PERIODIC:
            kinds.append(FACE_PERIODIC)
        elif face.kind == HomeFaceKind.ZOU_HE:
            kinds.append(FACE_ZOU_HE)
        else:
            kinds.append(FACE_WALL)
        uxs.append(float(face.ux))
        uys.append(float(face.uy))
        uzs.append(float(face.uz))
    return (
        wp.array(np.asarray(kinds, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.asarray(uxs, dtype=np.float32), dtype=float, device=device),
        wp.array(np.asarray(uys, dtype=np.float32), dtype=float, device=device),
        wp.array(np.asarray(uzs, dtype=np.float32), dtype=float, device=device),
    )


def alloc_home_vof_gpu(
    shape: tuple[int, int, int],
    lattice: LatticeSpec | str,
    device: str,
    domain_bc: HomeDomainBC | None = None,
) -> HomeVofGpuBuffers:
    spec = get_lattice_spec(lattice) if isinstance(lattice, str) else lattice
    bc = domain_bc if domain_bc is not None else HomeDomainBC.all_walls()
    nx, ny, nz = shape

    def zf():
        return wp.zeros((nx, ny, nz), dtype=float, device=device)

    fk, fux, fuy, fuz = _face_arrays_from_bc(bc, device)
    return HomeVofGpuBuffers(
        rho=zf(), ux=zf(), uy=zf(), uz=zf(),
        sxx=zf(), syy=zf(), szz=zf(), sxy=zf(), sxz=zf(), syz=zf(),
        rho_b=zf(), ux_b=zf(), uy_b=zf(), uz_b=zf(),
        sxx_b=zf(), syy_b=zf(), szz_b=zf(), sxy_b=zf(), sxz_b=zf(), syz_b=zf(),
        mass=zf(), mass_b=zf(), massex=zf(), phi=zf(),
        cell_type=wp.zeros((nx, ny, nz), dtype=wp.int32, device=device),
        cell_tmp=wp.zeros((nx, ny, nz), dtype=wp.int32, device=device),
        cell_tmp2=wp.zeros((nx, ny, nz), dtype=wp.int32, device=device),
        cx=wp.array(np.asarray(spec.cx, dtype=np.int32), dtype=wp.int32, device=device),
        cy=wp.array(np.asarray(spec.cy, dtype=np.int32), dtype=wp.int32, device=device),
        cz=wp.array(np.asarray(spec.cz, dtype=np.int32), dtype=wp.int32, device=device),
        w=wp.array(np.asarray(spec.weights, dtype=np.float32), dtype=float, device=device),
        opp=wp.array(np.asarray(spec.opposite, dtype=np.int32), dtype=wp.int32, device=device),
        face_kind=fk, face_ux=fux, face_uy=fuy, face_uz=fuz,
        kappa=zf(), kappa_tmp=zf(), solid_phi=zf(),
        shape=shape, device=device, num_dirs=int(spec.num_dirs),
    )


def upload_home_vof_state(buf: HomeVofGpuBuffers, state: HomeVofState) -> None:
    m = state.moments
    device = buf.device
    buf.rho.assign(wp.array(m.rho.astype(np.float32), dtype=float, device=device))
    buf.ux.assign(wp.array(m.ux.astype(np.float32), dtype=float, device=device))
    buf.uy.assign(wp.array(m.uy.astype(np.float32), dtype=float, device=device))
    buf.uz.assign(wp.array(m.uz.astype(np.float32), dtype=float, device=device))
    buf.sxx.assign(wp.array(m.sxx.astype(np.float32), dtype=float, device=device))
    buf.syy.assign(wp.array(m.syy.astype(np.float32), dtype=float, device=device))
    buf.szz.assign(wp.array(m.szz.astype(np.float32), dtype=float, device=device))
    buf.sxy.assign(wp.array(m.sxy.astype(np.float32), dtype=float, device=device))
    buf.sxz.assign(wp.array(m.sxz.astype(np.float32), dtype=float, device=device))
    buf.syz.assign(wp.array(m.syz.astype(np.float32), dtype=float, device=device))
    phi = state.phi.astype(np.float32)
    rho = m.rho.astype(np.float32)
    mass = (phi * rho).astype(np.float32)
    buf.phi.assign(wp.array(phi, dtype=float, device=device))
    buf.mass.assign(wp.array(mass, dtype=float, device=device))
    buf.massex.zero_()
    buf.cell_type.assign(
        wp.array(state.cell_type.astype(np.int32), dtype=wp.int32, device=device)
    )


def seed_home_vof_gpu(
    buf: HomeVofGpuBuffers,
    dam_x: int,
    fill_z: int,
    rho_liquid: float = 1.0,
) -> None:
    host = seed_dam_break_column(buf.shape, dam_x, fill_z, rho_liquid)
    upload_home_vof_state(buf, host)


def set_face_bc_gpu(buf: HomeVofGpuBuffers, domain_bc: HomeDomainBC) -> None:
    fk, fux, fuy, fuz = _face_arrays_from_bc(domain_bc, buf.device)
    buf.face_kind.assign(fk)
    buf.face_ux.assign(fux)
    buf.face_uy.assign(fuy)
    buf.face_uz.assign(fuz)


def step_home_vof_gpu(
    buf: HomeVofGpuBuffers,
    tau: float,
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
    rho_g0: float = 1.0,
    gamma: float = 0.0,
    eps_phi: float = 1.0e-4,
    rho_liquid: float = 1.0,
    kappa_smooth: int = 2,
    wall_wetting: float = 0.0,
    wall_film_drain: bool = False,
    wall_film_phi_max: float = 0.35,
    wall_film_u_max: float = 0.015,
    wall_film_edge_only: bool = True,
) -> None:
    """One HOME-FREE VOF step (fused + surface_1/2/3 + optional wetting/film)."""
    del eps_phi, rho_liquid
    from wanphys._src.fluid.fluid_grid.lbm.phases import vof_plic

    nx, ny, nz = buf.shape
    dim = (nx, ny, nz)
    g = float(gamma)

    if g != 0.0:
        wp.launch(
            vof_plic.vof_compute_kappa_kernel,
            dim=dim,
            inputs=[
                buf.phi, buf.cell_type, buf.solid_phi, buf.kappa,
                0, 0, 0, nx, ny, nz,
            ],
            device=buf.device,
        )
        for _ in range(max(int(kappa_smooth), 0)):
            wp.launch(
                vof_plic.vof_smooth_kappa_kernel,
                dim=dim,
                inputs=[
                    buf.kappa, buf.cell_type, buf.solid_phi, buf.kappa_tmp,
                    0, 0, 0, nx, ny, nz,
                ],
                device=buf.device,
            )
            buf.kappa, buf.kappa_tmp = buf.kappa_tmp, buf.kappa
        if float(wall_wetting) != 0.0:
            wp.launch(
                home_vof_wall_wetting_kappa_kernel,
                dim=dim,
                inputs=[
                    buf.kappa, buf.cell_type, float(wall_wetting), nx, ny, nz,
                ],
                device=buf.device,
            )
    else:
        buf.kappa.zero_()

    wp.launch(
        home_vof_fused_kernel,
        dim=dim,
        inputs=[
            buf.rho, buf.ux, buf.uy, buf.uz,
            buf.sxx, buf.syy, buf.szz, buf.sxy, buf.sxz, buf.syz,
            buf.mass, buf.massex, buf.phi, buf.cell_type,
            buf.rho_b, buf.ux_b, buf.uy_b, buf.uz_b,
            buf.sxx_b, buf.syy_b, buf.szz_b, buf.sxy_b, buf.sxz_b, buf.syz_b,
            buf.mass_b, buf.cell_tmp,
            buf.cx, buf.cy, buf.cz, buf.w, buf.opp,
            buf.face_kind, buf.face_ux, buf.face_uy, buf.face_uz,
            buf.kappa, g,
            buf.num_dirs, float(tau), float(fx), float(fy), float(fz),
            float(rho_g0), nx, ny, nz,
        ],
        device=buf.device,
    )
    buf.swap_moment_buffers()
    # Flags in cell_tmp; moments/mass in primary. Mutate flags in-place like Home-FSLBM.
    # Warp: wp.copy(dest, src)
    wp.copy(buf.cell_tmp2, buf.cell_tmp)
    wp.launch(
        home_vof_surface1_kernel,
        dim=dim,
        inputs=[buf.cell_tmp2, nx, ny, nz],
        device=buf.device,
    )
    wp.launch(
        home_vof_surface2_kernel,
        dim=dim,
        inputs=[
            buf.cell_tmp2,
            buf.rho, buf.ux, buf.uy, buf.uz,
            buf.sxx, buf.syy, buf.szz, buf.sxy, buf.sxz, buf.syz,
            buf.mass, nx, ny, nz,
        ],
        device=buf.device,
    )
    buf.massex.zero_()
    wp.launch(
        home_vof_surface3_kernel,
        dim=dim,
        inputs=[
            buf.cell_tmp2, buf.cell_type,
            buf.rho, buf.mass, buf.massex, buf.phi,
            nx, ny, nz,
        ],
        device=buf.device,
    )
    wp.launch(
        home_vof_seal_fg_kernel,
        dim=dim,
        inputs=[buf.cell_type, buf.rho, buf.mass, buf.phi, nx, ny, nz],
        device=buf.device,
    )
    # Optional rim-film drain via massex (surface3 inventory path) + salvage.
    if wall_film_drain:
        wp.launch(
            home_vof_wall_film_drain_kernel,
            dim=dim,
            inputs=[
                buf.cell_type, buf.rho, buf.mass, buf.massex, buf.phi,
                buf.ux, buf.uy, buf.uz,
                float(wall_film_phi_max), float(wall_film_u_max),
                1 if wall_film_edge_only else 0,
                nx, ny, nz,
            ],
            device=buf.device,
        )
        wp.launch(
            home_vof_salvage_mass_on_gas_kernel,
            dim=dim,
            inputs=[buf.cell_type, buf.rho, buf.mass, buf.phi],
            device=buf.device,
        )