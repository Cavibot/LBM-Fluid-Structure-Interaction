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
def home_vof_solid_mac_to_cell_kernel(
    solid_phi: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    solid_ux: wp.array3d(dtype=float),
    solid_uy: wp.array3d(dtype=float),
    solid_uz: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Average MAC solid face velocities onto solid cell centres."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] >= 0.0:
        solid_ux[i, j, k] = 0.0
        solid_uy[i, j, k] = 0.0
        solid_uz[i, j, k] = 0.0
        return
    solid_ux[i, j, k] = 0.5 * (vel_solid_u[i, j, k] + vel_solid_u[i + 1, j, k])
    solid_uy[i, j, k] = 0.5 * (vel_solid_v[i, j, k] + vel_solid_v[i, j + 1, k])
    solid_uz[i, j, k] = 0.5 * (vel_solid_w[i, j, k] + vel_solid_w[i, j, k + 1])


@wp.kernel
def home_vof_apply_solid_mask_kernel(
    solid_phi: wp.array3d(dtype=float),
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
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
) -> None:
    """Force rasterized solid cells to empty gas (Home TYPE_S skip)."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] >= 0.0:
        return
    rho[i, j, k] = 0.0
    ux[i, j, k] = 0.0
    uy[i, j, k] = 0.0
    uz[i, j, k] = 0.0
    sxx[i, j, k] = 0.0
    syy[i, j, k] = 0.0
    szz[i, j, k] = 0.0
    sxy[i, j, k] = 0.0
    sxz[i, j, k] = 0.0
    syz[i, j, k] = 0.0
    mass[i, j, k] = 0.0
    phi[i, j, k] = 0.0
    cell_type[i, j, k] = CELL_GAS


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
    solid_phi: wp.array3d(dtype=float),
    solid_ux: wp.array3d(dtype=float),
    solid_uy: wp.array3d(dtype=float),
    solid_uz: wp.array3d(dtype=float),
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
    gas_rho: wp.array3d(dtype=float),
    disjoin_force: wp.array3d(dtype=float),
    bubble_tag: wp.array3d(dtype=wp.int32),
    bubble_volume: wp.array(dtype=float),
    gamma: float,
    enable_disjoint: int,
    disjoint_factor: float,
    enable_small_sigma: int,
    small_bubble_vol: float,
    small_six_sigma: float,
    enable_eddy: int,
    eddy_atm_vol: float,
    max_bubbles: int,
    num_dirs: int,
    tau: float,
    fx: float,
    fy: float,
    fz: float,
    home_fill_empty: int,
    home_wall_eq: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Fused pull-stream + mass + FS BC + collide (Home-FSLBM style)."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
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

    # Eq. 12 + §4.2 + Home disjoint / small-bubble σ:
    # ρ_g = ρ_bubble − σ₆ κ − f_dis · disjoint.
    six_sigma = 6.0 * gamma
    djoin = float(0.0)
    if enable_disjoint != 0:
        djoin = disjoin_force[i, j, k]
    tag_c = int(bubble_tag[i, j, k])
    if (
        enable_small_sigma != 0
        and djoin <= 0.0
        and six_sigma > 1.0e-3
        and tag_c > 0
        and tag_c <= max_bubbles
    ):
        vol_b = bubble_volume[tag_c - 1]
        if vol_b > 0.0 and vol_b < small_bubble_vol:
            six_sigma = small_six_sigma
    rho_g = gas_rho[i, j, k] - six_sigma * kappa[i, j, k] - disjoint_factor * djoin
    if rho_g < 0.2:
        rho_g = 0.2
    if rho_g > 1.8:
        rho_g = 1.8

    # Home-FSLBM: gas equilibrium uses Guo half-force velocity u + F/2.
    vg_x = vx + 0.5 * fx
    vg_y = vy + 0.5 * fy
    vg_z = vz + 0.5 * fz
    vg_x, vg_y, vg_z = _clamp_u(vg_x, vg_y, vg_z)

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

        # Moving / static rigid solid (rasterized solid_phi < 0).
        if is_wall == 0 and solid_phi[ni, nj, nk] < 0.0:
            is_wall = 1
            uxp = solid_ux[ni, nj, nk]
            uyp = solid_uy[ni, nj, nk]
            uzp = solid_uz[ni, nj, nk]

        fhn = float(0.0)
        ntype = CELL_GAS
        if is_wall != 0:
            # Home-FSLBM solid pull is f^eq(ρ, u_wall=0). Optional HOME Eq.24
            # keeps non-eq stress (default for older wanphys runs).
            if home_wall_eq != 0:
                fhn = _feq_w(w, rho_c, uxp, uyp, uzp, float(cxi), float(cyi), float(czi))
            else:
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
                feg_i = _feq_w(
                    w, rho_g, vg_x, vg_y, vg_z, float(cxi), float(cyi), float(czi)
                )
                feg_o = _feq_w(
                    w_arr[od], rho_g, vg_x, vg_y, vg_z,
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
                # Mass with F/I. Interface uses θ·(fhn−fon_opp) like Home.
                # Liquid uses the same opposite-link form: bare Home liquid
                # ``fhn[i]-fon[i]`` drifts Σmass against interface in this port.
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
    ux_s = mx * inv
    uy_s = my * inv
    uz_s = mz * inv
    # Home near-bubble eddy viscosity: raise τ near non-atmosphere bubbles.
    # Only check INTERFACE cells (Home mainly needs FS damping; full-grid 6³ is too heavy).
    tau_use = tau
    if enable_eddy != 0 and ctype == CELL_INTERFACE:
        found = int(0)
        for t in range(216):
            odi = int(t // 36) - 3
            rem = int(t % 36)
            odj = int(rem // 6) - 3
            odk = int(rem % 6) - 3
            ni = i + odi
            nj = j + odj
            nk = k + odk
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue
            nt = int(bubble_tag[ni, nj, nk])
            if nt <= 0 or nt > max_bubbles:
                continue
            if bubble_volume[nt - 1] < eddy_atm_vol:
                found = 1
        if found != 0:
            xx = axx * inv - cs2
            yy = ayy * inv - cs2
            zz = azz * inv - cs2
            xy = axy * inv
            xz = axz * inv
            yz = ayz * inv
            vis = 4.0 * wp.sqrt(
                xx * xx + 2.0 * xy * xy + 2.0 * xz * xz
                + yy * yy + 2.0 * yz * yz + zz * zz
            )
            tau_use = (vis + 1.0e-4) * 3.0 + 0.5

    rho, ux, uy, uz, sxx_o, syy_o, szz_o, sxy_o, sxz_o, syz_o = home_collide_moments(
        r, ux_s, uy_s, uz_s,
        axx * inv, ayy * inv, azz * inv,
        axy * inv, axz * inv, ayz * inv,
        tau_use, fx, fy, fz,
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

    # Fill/empty. Softened Home-FSLBM / TechRep-05-4 rules:
    #   fill:  mass>ρ  OR  (no gas neighbor AND nearly full)
    #   empty: mass<0  OR  (no fluid neighbor AND nearly empty)
    # Bare TYPE_NO_G / TYPE_NO_F (Home GPU literal) evaporates thin I crests
    # and thrash-fills wall menisci → O(10^4) Σmass loss + oscillatory "shaving".
    out_type = ctype
    if ctype == CELL_INTERFACE:
        if home_fill_empty != 0:
            if massn > rho or (has_gas == 0 and massn > 0.99 * rho):
                out_type = CELL_IF
            elif massn < 0.0 or (has_fluid == 0 and massn < 0.1 * rho):
                out_type = CELL_IG
        else:
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
    delta_phi: wp.array3d(dtype=float),
    track_delta_phi: int,
    split_flag: wp.array(dtype=wp.int32),
    report_split: int,
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
    phi_old = phi[i, j, k]

    # Home: IF→F reports split (topology may disconnect a bubble).
    if ctype == CELL_IF and report_split != 0:
        wp.atomic_max(split_flag, 0, 1)

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
    # Home: delta_phi drives bubble.volume -= Δφ (gas volume = Σ(1-φ)).
    if track_delta_phi != 0:
        delta_phi[i, j, k] = phin - phi_old


@wp.kernel
def home_vof_seal_fg_kernel(
    cell: wp.array3d(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    delta_phi: wp.array3d(dtype=float),
    track_delta_phi: int,
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
                    phi_old = phi[i, j, k]
                    if rr > 1.0e-8:
                        phi[i, j, k] = mass[i, j, k] / rr
                    else:
                        phi[i, j, k] = 1.0
                    if track_delta_phi != 0:
                        delta_phi[i, j, k] = delta_phi[i, j, k] + (
                            phi[i, j, k] - phi_old
                        )
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


def level_surface_high_to_low(
    buf: "HomeVofGpuBuffers",
    *,
    rate: float = 0.35,
    dz_min: int = 2,
    wet_phi: float = 0.5,
    climb_margin: int | None = None,
) -> float:
    """Global high->low free-surface leveling (host, once per call).

    Uses a **main-pool band** around the median free-surface height so that
    sparse wall-climb spikes (``corn`` / full ``z_p2p``) do not become the only
    donors forever. Climb columns above ``median + climb_margin`` are peeled
    onto pool receivers; within the pool, only ``z_hi_pool - z_lo_pool`` matters.

    If pool high−low < ``dz_min`` and there are no climbers, returns 0.
    Returns inventory delta of ``Σmass`` (≈0 if conservative).
    """
    ctype = buf.cell_type.numpy().copy()
    phi = buf.phi.numpy().copy()
    mass = buf.mass.numpy().copy()
    rho = buf.rho.numpy().copy()
    ux = buf.ux.numpy().copy()
    uy = buf.uy.numpy().copy()
    uz = buf.uz.numpy().copy()
    sxx = buf.sxx.numpy().copy()
    syy = buf.syy.numpy().copy()
    szz = buf.szz.numpy().copy()
    sxy = buf.sxy.numpy().copy()
    sxz = buf.sxz.numpy().copy()
    syz = buf.syz.numpy().copy()
    nx, ny, nz = buf.shape
    m0 = float(np.nansum(mass))

    z_top = np.full((nx, ny), -1, dtype=np.int32)
    for k in range(nz):
        wet = (ctype[:, :, k] > 0) & (phi[:, :, k] > wet_phi)
        z_top[wet] = k
    wet_cols = z_top >= 0
    if not np.any(wet_cols):
        return 0.0

    med = float(np.median(z_top[wet_cols]))
    if climb_margin is None:
        climb_margin = max(3, int(nz // 16))
    climb_margin = int(max(2, climb_margin))

    climber_mask = wet_cols & (z_top > int(np.floor(med)) + climb_margin)
    pool_mask = wet_cols & (~climber_mask)
    if not np.any(pool_mask):
        pool_mask = wet_cols

    def _ensure_rho(i: int, j: int, k: int) -> float:
        r = float(rho[i, j, k])
        if r < 0.05:
            rho[i, j, k] = 1.0
            return 1.0
        return r

    def _rest_moments(i: int, j: int, k: int) -> None:
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0
        sxx[i, j, k] = 0.0
        syy[i, j, k] = 0.0
        szz[i, j, k] = 0.0
        sxy[i, j, k] = 0.0
        sxz[i, j, k] = 0.0
        syz[i, j, k] = 0.0

    def _deposit(i: int, j: int, k: int, amt: float) -> None:
        if amt <= 1.0e-12 or k < 0 or k >= nz:
            return
        r = _ensure_rho(i, j, k)
        ct = int(ctype[i, j, k])
        if ct == CELL_GAS:
            ctype[i, j, k] = CELL_INTERFACE
            _rest_moments(i, j, k)
            mass[i, j, k] = amt
            phi[i, j, k] = min(1.0, amt / r)
            if mass[i, j, k] > r:
                overflow = float(mass[i, j, k] - r)
                mass[i, j, k] = r
                phi[i, j, k] = 1.0
                ctype[i, j, k] = CELL_LIQUID
                _deposit(i, j, k + 1, overflow)
            return
        if ct == CELL_LIQUID:
            _deposit(i, j, k + 1, amt)
            return
        new_m = float(mass[i, j, k]) + amt
        if new_m <= r:
            mass[i, j, k] = new_m
            phi[i, j, k] = new_m / r
            return
        overflow = new_m - r
        mass[i, j, k] = r
        phi[i, j, k] = 1.0
        ctype[i, j, k] = CELL_LIQUID
        _deposit(i, j, k + 1, overflow)

    def _peel(donors: np.ndarray, receivers: np.ndarray, z_src: int) -> float:
        if donors.size == 0 or receivers.size == 0 or z_src < 0 or z_src >= nz:
            return 0.0
        pot = 0.0
        for d in range(donors.shape[0]):
            i = int(donors[d, 0])
            j = int(donors[d, 1])
            m_d = float(mass[i, j, z_src])
            if m_d <= 1.0e-8:
                continue
            take = min(float(rate) * m_d, 0.5 * m_d)
            if take <= 1.0e-8:
                continue
            mass[i, j, z_src] = m_d - take
            pot += take
            r_d = _ensure_rho(i, j, z_src)
            if mass[i, j, z_src] <= 1.0e-8:
                mass[i, j, z_src] = 0.0
                phi[i, j, z_src] = 0.0
                ctype[i, j, z_src] = CELL_GAS
                rho[i, j, z_src] = 0.0
            else:
                if int(ctype[i, j, z_src]) == CELL_LIQUID:
                    ctype[i, j, z_src] = CELL_INTERFACE
                phi[i, j, z_src] = float(mass[i, j, z_src]) / r_d
                if phi[i, j, z_src] > 1.0:
                    phi[i, j, z_src] = 1.0
        if pot <= 1.0e-8:
            return 0.0
        # Deposit onto receiver free-surface tops (recomputed per column).
        share = pot / float(receivers.shape[0])
        for ridx in range(receivers.shape[0]):
            i = int(receivers[ridx, 0])
            j = int(receivers[ridx, 1])
            zl = int(z_top[i, j])
            if zl < 0:
                zl = int(np.floor(med))
            _deposit(i, j, max(0, zl), share)
        return pot

    moved = 0.0

    # 1) Peel wall-climb outliers onto main-pool columns near/under median.
    if np.any(climber_mask):
        z_climb_hi = int(np.max(z_top[climber_mask]))
        donors = np.argwhere(climber_mask & (z_top == z_climb_hi))
        # Prefer pool columns at or below median as sinks (never another spike).
        recv_mask = pool_mask & (z_top <= int(np.floor(med)) + 1)
        if not np.any(recv_mask):
            recv_mask = pool_mask
        receivers = np.argwhere(recv_mask)
        moved += _peel(donors, receivers, z_climb_hi)
        # Refresh tops after climb peel (cheap: only recompute from arrays).
        z_top[:, :] = -1
        for k in range(nz):
            wet = (ctype[:, :, k] > 0) & (phi[:, :, k] > wet_phi)
            z_top[wet] = k
        wet_cols = z_top >= 0
        if np.any(wet_cols):
            med = float(np.median(z_top[wet_cols]))
        climber_mask = wet_cols & (z_top > int(np.floor(med)) + climb_margin)
        pool_mask = wet_cols & (~climber_mask)
        if not np.any(pool_mask):
            pool_mask = wet_cols

    # 2) Level within the main pool only (ignore residual climb for z_hi/z_lo).
    if np.any(pool_mask):
        z_lo = int(np.min(z_top[pool_mask]))
        z_hi = int(np.max(z_top[pool_mask]))
        if z_hi - z_lo >= int(dz_min):
            donors = np.argwhere(pool_mask & (z_top == z_hi))
            receivers = np.argwhere(pool_mask & (z_top == z_lo))
            moved += _peel(donors, receivers, z_hi)

    if moved <= 1.0e-8:
        return 0.0

    device = buf.device
    buf.mass.assign(wp.array(mass.astype(np.float32), dtype=float, device=device))
    buf.phi.assign(wp.array(phi.astype(np.float32), dtype=float, device=device))
    buf.rho.assign(wp.array(rho.astype(np.float32), dtype=float, device=device))
    buf.ux.assign(wp.array(ux.astype(np.float32), dtype=float, device=device))
    buf.uy.assign(wp.array(uy.astype(np.float32), dtype=float, device=device))
    buf.uz.assign(wp.array(uz.astype(np.float32), dtype=float, device=device))
    buf.sxx.assign(wp.array(sxx.astype(np.float32), dtype=float, device=device))
    buf.syy.assign(wp.array(syy.astype(np.float32), dtype=float, device=device))
    buf.szz.assign(wp.array(szz.astype(np.float32), dtype=float, device=device))
    buf.sxy.assign(wp.array(sxy.astype(np.float32), dtype=float, device=device))
    buf.sxz.assign(wp.array(sxz.astype(np.float32), dtype=float, device=device))
    buf.syz.assign(wp.array(syz.astype(np.float32), dtype=float, device=device))
    buf.cell_type.assign(
        wp.array(ctype.astype(np.int32), dtype=wp.int32, device=device)
    )
    return float(np.nansum(mass)) - m0


def flatten_top_interface_layer(
    buf: "HomeVofGpuBuffers",
    *,
    climb_margin: int | None = None,
    min_frac: float = 0.02,
    max_frac: float = 0.98,
) -> dict[str, float]:
    """Rebuild main-pool columns as liquid stack + one interface with shared φ.

    Uses **continuous** column volume ``h = Σ (mass/ρ)`` (sub-cell height). Pool
    columns (excluding wall-climb outliers) are assigned the same target height
    ``H* = V_pool / N_pool`` and rewritten as::

        k < ⌊H*⌋     → LIQUID (φ=1)
        k = ⌊H*⌋     → INTERFACE (φ = {H*})
        k > ⌊H*⌋     → GAS

    This is the late-pool "亚格子顶层铺平" pass: not pure LBM, but conservative
    in Σmass over the rewritten columns (inventory redistributed, not invented).

    Returns diagnostics: ``dmass``, ``H_star``, ``cont_rms``, ``cont_rob``, ``n_pool``.
    """
    ctype = buf.cell_type.numpy().copy()
    phi = buf.phi.numpy().copy()
    mass = buf.mass.numpy().copy()
    rho = buf.rho.numpy().copy()
    ux = buf.ux.numpy().copy()
    uy = buf.uy.numpy().copy()
    uz = buf.uz.numpy().copy()
    sxx = buf.sxx.numpy().copy()
    syy = buf.syy.numpy().copy()
    szz = buf.szz.numpy().copy()
    sxy = buf.sxy.numpy().copy()
    sxz = buf.sxz.numpy().copy()
    syz = buf.syz.numpy().copy()
    solid = buf.solid_phi.numpy()
    nx, ny, nz = buf.shape
    m0 = float(np.nansum(mass))

    # Continuous column fluid volume (lattice cells of liquid).
    h = np.zeros((nx, ny), dtype=np.float64)
    for k in range(nz):
        fluid = (ctype[:, :, k] > 0) & (solid[:, :, k] >= 0.0)
        r = np.maximum(rho[:, :, k], 0.05)
        h = h + np.where(fluid, mass[:, :, k] / r, 0.0)

    wet_cols = h > float(min_frac)
    if not np.any(wet_cols):
        return {
            "dmass": 0.0,
            "H_star": 0.0,
            "cont_rms": 0.0,
            "cont_rob": 0.0,
            "n_pool": 0.0,
        }

    med = float(np.median(h[wet_cols]))
    if climb_margin is None:
        climb_margin = max(3, int(nz // 16))
    climb_margin = int(max(2, climb_margin))
    climber = wet_cols & (h > med + float(climb_margin))
    pool = wet_cols & (~climber)
    if not np.any(pool):
        pool = wet_cols

    v_pool = float(np.sum(h[pool]))
    n_pool = int(np.count_nonzero(pool))
    # Shared continuous height; bias slightly so the top stays INTERFACE
    # (φ∈(min_frac, max_frac)) with minimal Σmass change.
    H_exact = v_pool / float(max(n_pool, 1))
    k_full = int(np.floor(H_exact + 1.0e-12))
    frac = float(H_exact - float(k_full))
    if frac < float(min_frac):
        # Prefer H = k_full + min_frac (tiny invent) over demoting a full cell.
        if k_full >= 1 and (1.0 - float(max_frac)) < (float(min_frac) - frac):
            k_full -= 1
            frac = float(max_frac)
        else:
            frac = float(min_frac)
    elif frac > float(max_frac):
        if (1.0 - frac) <= (frac - float(max_frac)) and k_full + 1 < nz:
            k_full += 1
            frac = float(min_frac)
        else:
            frac = float(max_frac)
    H_star = float(k_full) + float(frac)

    def _clear_fluid(i: int, j: int) -> None:
        for k in range(nz):
            if solid[i, j, k] < 0.0:
                continue
            mass[i, j, k] = 0.0
            phi[i, j, k] = 0.0
            rho[i, j, k] = 0.0
            ctype[i, j, k] = CELL_GAS
            ux[i, j, k] = 0.0
            uy[i, j, k] = 0.0
            uz[i, j, k] = 0.0
            sxx[i, j, k] = 0.0
            syy[i, j, k] = 0.0
            szz[i, j, k] = 0.0
            sxy[i, j, k] = 0.0
            sxz[i, j, k] = 0.0
            syz[i, j, k] = 0.0

    def _set_liquid(i: int, j: int, k: int) -> None:
        rho[i, j, k] = 1.0
        mass[i, j, k] = 1.0
        phi[i, j, k] = 1.0
        ctype[i, j, k] = CELL_LIQUID
        ux[i, j, k] = uy[i, j, k] = uz[i, j, k] = 0.0
        sxx[i, j, k] = syy[i, j, k] = szz[i, j, k] = 0.0
        sxy[i, j, k] = sxz[i, j, k] = syz[i, j, k] = 0.0

    def _set_interface(i: int, j: int, k: int, fphi: float) -> None:
        rho[i, j, k] = 1.0
        mass[i, j, k] = float(fphi)
        phi[i, j, k] = float(fphi)
        ctype[i, j, k] = CELL_INTERFACE
        ux[i, j, k] = uy[i, j, k] = uz[i, j, k] = 0.0
        sxx[i, j, k] = syy[i, j, k] = szz[i, j, k] = 0.0
        sxy[i, j, k] = sxz[i, j, k] = syz[i, j, k] = 0.0

    pool_ij = np.argwhere(pool)
    for i, j in pool_ij:
        i = int(i)
        j = int(j)
        _clear_fluid(i, j)
        filled = 0
        k = 0
        while filled < k_full and k < nz:
            if solid[i, j, k] >= 0.0:
                _set_liquid(i, j, k)
                filled += 1
            k += 1
        # Place interface in the next non-solid cell.
        # Place interface in the next non-solid cell (always; top is interface).
        while k < nz and solid[i, j, k] < 0.0:
            k += 1
        if k < nz:
            _set_interface(i, j, k, frac)

    # Diagnostics on rewritten pool.
    h2 = np.zeros((nx, ny), dtype=np.float64)
    for k in range(nz):
        fluid = (ctype[:, :, k] > 0) & (solid[:, :, k] >= 0.0)
        r = np.maximum(rho[:, :, k], 0.05)
        h2 = h2 + np.where(fluid, mass[:, :, k] / r, 0.0)
    hp = h2[pool]
    p05, p95 = np.percentile(hp, [5.0, 95.0]) if hp.size else (0.0, 0.0)

    device = buf.device
    buf.mass.assign(wp.array(mass.astype(np.float32), dtype=float, device=device))
    buf.phi.assign(wp.array(phi.astype(np.float32), dtype=float, device=device))
    buf.rho.assign(wp.array(rho.astype(np.float32), dtype=float, device=device))
    buf.ux.assign(wp.array(ux.astype(np.float32), dtype=float, device=device))
    buf.uy.assign(wp.array(uy.astype(np.float32), dtype=float, device=device))
    buf.uz.assign(wp.array(uz.astype(np.float32), dtype=float, device=device))
    buf.sxx.assign(wp.array(sxx.astype(np.float32), dtype=float, device=device))
    buf.syy.assign(wp.array(syy.astype(np.float32), dtype=float, device=device))
    buf.szz.assign(wp.array(szz.astype(np.float32), dtype=float, device=device))
    buf.sxy.assign(wp.array(sxy.astype(np.float32), dtype=float, device=device))
    buf.sxz.assign(wp.array(sxz.astype(np.float32), dtype=float, device=device))
    buf.syz.assign(wp.array(syz.astype(np.float32), dtype=float, device=device))
    buf.cell_type.assign(
        wp.array(ctype.astype(np.int32), dtype=wp.int32, device=device)
    )
    return {
        "dmass": float(np.nansum(mass) - m0),
        "H_star": float(H_star),
        "cont_rms": float(np.std(hp)) if hp.size else 0.0,
        "cont_rob": float(p95 - p05),
        "n_pool": float(n_pool),
    }


def topup_surface_with_budget(
    buf: "HomeVofGpuBuffers",
    *,
    budget: float,
    wet_phi: float = 0.5,
    target_z: int | None = None,
) -> float:
    """Invent <= ``budget`` mass to raise low columns (spend recovered -Delta).

    After conservative high->low leveling stalls at an O(1)-cell step, grow
    columns with ``z_top < target`` upward using at most ``budget`` invented
    inventory. Returns invented mass.
    """
    if budget <= 1.0e-6:
        return 0.0

    ctype = buf.cell_type.numpy().copy()
    phi = buf.phi.numpy().copy()
    mass = buf.mass.numpy().copy()
    rho = buf.rho.numpy().copy()
    ux = buf.ux.numpy().copy()
    uy = buf.uy.numpy().copy()
    uz = buf.uz.numpy().copy()
    sxx = buf.sxx.numpy().copy()
    syy = buf.syy.numpy().copy()
    szz = buf.szz.numpy().copy()
    sxy = buf.sxy.numpy().copy()
    sxz = buf.sxz.numpy().copy()
    syz = buf.syz.numpy().copy()
    nx, ny, nz = buf.shape

    z_top = np.full((nx, ny), -1, dtype=np.int32)
    for k in range(nz):
        wet = (ctype[:, :, k] > 0) & (phi[:, :, k] > wet_phi)
        z_top[wet] = k
    wet_cols = z_top >= 0
    if not np.any(wet_cols):
        return 0.0

    if target_z is None:
        med = float(np.median(z_top[wet_cols]))
        inliers = wet_cols & (z_top <= int(np.floor(med)) + 1)
        if np.any(inliers):
            target_z = int(np.round(float(np.median(z_top[inliers]))))
        else:
            target_z = int(np.round(med))
        h_a = float(z_top[-2, :].mean()) if nx >= 2 else med
        h_b = float(z_top[1, :].mean()) if nx >= 2 else med
        target_z = max(target_z, int(np.round(max(h_a, h_b))))

    target_z = int(target_z)
    if target_z < 1:
        return 0.0

    lows = np.argwhere((z_top >= 0) & (z_top < target_z))
    if lows.size == 0:
        return 0.0

    deficits = target_z - z_top[lows[:, 0], lows[:, 1]]
    order = np.argsort(-deficits)

    invented = 0.0
    remain = float(budget)

    def _rest_moments(i: int, j: int, k: int) -> None:
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0
        sxx[i, j, k] = 0.0
        syy[i, j, k] = 0.0
        szz[i, j, k] = 0.0
        sxy[i, j, k] = 0.0
        sxz[i, j, k] = 0.0
        syz[i, j, k] = 0.0

    def _fill_cell(i: int, j: int, k: int, amt: float, as_liquid: bool) -> float:
        if amt <= 1.0e-12 or k < 0 or k >= nz:
            return 0.0
        rho[i, j, k] = 1.0
        _rest_moments(i, j, k)
        if as_liquid or amt >= 0.99:
            ctype[i, j, k] = CELL_LIQUID
            phi[i, j, k] = 1.0
            mass[i, j, k] = 1.0
            return 1.0
        ctype[i, j, k] = CELL_INTERFACE
        phi[i, j, k] = min(1.0, max(0.05, amt))
        mass[i, j, k] = float(phi[i, j, k])
        return float(mass[i, j, k])

    for idx in order:
        if remain <= 1.0e-6:
            break
        i = int(lows[idx, 0])
        j = int(lows[idx, 1])
        zl = int(z_top[i, j])
        if zl >= 0 and int(ctype[i, j, zl]) == CELL_INTERFACE:
            r = float(rho[i, j, zl])
            if r < 0.05:
                r = 1.0
                rho[i, j, zl] = 1.0
            room = max(0.0, r - float(mass[i, j, zl]))
            take = min(room, remain)
            if take > 1.0e-8:
                mass[i, j, zl] = float(mass[i, j, zl]) + take
                phi[i, j, zl] = min(1.0, float(mass[i, j, zl]) / r)
                remain -= take
                invented += take
                if phi[i, j, zl] >= 0.99:
                    ctype[i, j, zl] = CELL_LIQUID
                    phi[i, j, zl] = 1.0
                    mass[i, j, zl] = r
        for k in range(zl + 1, target_z):
            if remain <= 1.0e-6:
                break
            used = _fill_cell(i, j, k, min(1.0, remain), as_liquid=True)
            remain -= used
            invented += used
        if remain > 1.0e-6 and zl < target_z:
            k = target_z
            if k < nz and int(ctype[i, j, k]) == CELL_GAS:
                used = _fill_cell(i, j, k, min(0.55, remain), as_liquid=False)
                remain -= used
                invented += used
            elif k < nz and int(ctype[i, j, k]) == CELL_INTERFACE:
                r = 1.0
                rho[i, j, k] = r
                room = max(0.0, r - float(mass[i, j, k]))
                take = min(room, remain, 0.55)
                if take > 1.0e-8:
                    mass[i, j, k] = float(mass[i, j, k]) + take
                    phi[i, j, k] = min(1.0, float(mass[i, j, k]) / r)
                    remain -= take
                    invented += take

    if invented <= 1.0e-8:
        return 0.0

    device = buf.device
    buf.mass.assign(wp.array(mass.astype(np.float32), dtype=float, device=device))
    buf.phi.assign(wp.array(phi.astype(np.float32), dtype=float, device=device))
    buf.rho.assign(wp.array(rho.astype(np.float32), dtype=float, device=device))
    buf.ux.assign(wp.array(ux.astype(np.float32), dtype=float, device=device))
    buf.uy.assign(wp.array(uy.astype(np.float32), dtype=float, device=device))
    buf.uz.assign(wp.array(uz.astype(np.float32), dtype=float, device=device))
    buf.sxx.assign(wp.array(sxx.astype(np.float32), dtype=float, device=device))
    buf.syy.assign(wp.array(syy.astype(np.float32), dtype=float, device=device))
    buf.szz.assign(wp.array(szz.astype(np.float32), dtype=float, device=device))
    buf.sxy.assign(wp.array(sxy.astype(np.float32), dtype=float, device=device))
    buf.sxz.assign(wp.array(sxz.astype(np.float32), dtype=float, device=device))
    buf.syz.assign(wp.array(syz.astype(np.float32), dtype=float, device=device))
    buf.cell_type.assign(
        wp.array(ctype.astype(np.int32), dtype=wp.int32, device=device)
    )
    return float(invented)


def reabsorb_orphan_liquid(
    buf: "HomeVofGpuBuffers",
    *,
    max_cells: int = 96,
    height_margin: int = 3,
    wet_mass: float = 1.0e-6,
) -> tuple[float, int]:
    """Conservative: fold disconnected airborne liquid blobs into the main pool.

    Labels 6-connected wet components (liquid/interface with ``mass>wet_mass``).
    The largest-by-mass component is the main pool. Other components are orphans
    if they are small (``≤ max_cells``) **or** sit entirely above the main free
    surface (``min_z > pool_median_z + height_margin``). Their mass is deposited
    onto the main-pool free surface; inventory is conserved.

    Returns ``(mass_moved, n_orphan_components)``.
    """
    ctype = buf.cell_type.numpy().copy()
    phi = buf.phi.numpy().copy()
    mass = buf.mass.numpy().copy()
    rho = buf.rho.numpy().copy()
    ux = buf.ux.numpy().copy()
    uy = buf.uy.numpy().copy()
    uz = buf.uz.numpy().copy()
    sxx = buf.sxx.numpy().copy()
    syy = buf.syy.numpy().copy()
    szz = buf.szz.numpy().copy()
    sxy = buf.sxy.numpy().copy()
    sxz = buf.sxz.numpy().copy()
    syz = buf.syz.numpy().copy()
    nx, ny, nz = buf.shape

    wet = (ctype > 0) & (mass > float(wet_mass))
    if not np.any(wet):
        return 0.0, 0

    labels = -np.ones((nx, ny, nz), dtype=np.int32)
    comps: list[list[tuple[int, int, int]]] = []
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    wet_idx = np.argwhere(wet)
    for si, sj, sk in wet_idx:
        si, sj, sk = int(si), int(sj), int(sk)
        if labels[si, sj, sk] >= 0:
            continue
        cid = len(comps)
        stack = [(si, sj, sk)]
        labels[si, sj, sk] = cid
        cells: list[tuple[int, int, int]] = []
        while stack:
            i, j, k = stack.pop()
            cells.append((i, j, k))
            for di, dj, dk in neighbors:
                ni, nj, nk = i + di, j + dj, k + dk
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue
                if labels[ni, nj, nk] >= 0 or not wet[ni, nj, nk]:
                    continue
                labels[ni, nj, nk] = cid
                stack.append((ni, nj, nk))
        comps.append(cells)

    if len(comps) <= 1:
        return 0.0, 0

    masses = np.array(
        [sum(float(mass[i, j, k]) for i, j, k in cells) for cells in comps],
        dtype=np.float64,
    )
    main_id = int(np.argmax(masses))
    main_cells = comps[main_id]

    z_top = np.full((nx, ny), -1, dtype=np.int32)
    for i, j, k in main_cells:
        if k > z_top[i, j]:
            z_top[i, j] = k
    main_cols = z_top >= 0
    if not np.any(main_cols):
        return 0.0, 0
    pool_med = float(np.median(z_top[main_cols]))
    elevation_cut = int(np.floor(pool_med)) + int(max(1, height_margin))

    def _clear_cell(i: int, j: int, k: int) -> float:
        taken = float(mass[i, j, k])
        mass[i, j, k] = 0.0
        phi[i, j, k] = 0.0
        rho[i, j, k] = 0.0
        ctype[i, j, k] = CELL_GAS
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0
        sxx[i, j, k] = 0.0
        syy[i, j, k] = 0.0
        szz[i, j, k] = 0.0
        sxy[i, j, k] = 0.0
        sxz[i, j, k] = 0.0
        syz[i, j, k] = 0.0
        return taken

    def _ensure_rho(i: int, j: int, k: int) -> float:
        r = float(rho[i, j, k])
        if r < 0.05:
            rho[i, j, k] = 1.0
            return 1.0
        return r

    def _deposit(i: int, j: int, k: int, amt: float) -> None:
        if amt <= 1.0e-12 or k < 0 or k >= nz:
            return
        r = _ensure_rho(i, j, k)
        ct = int(ctype[i, j, k])
        if ct == CELL_GAS:
            ctype[i, j, k] = CELL_INTERFACE
            ux[i, j, k] = uy[i, j, k] = uz[i, j, k] = 0.0
            sxx[i, j, k] = syy[i, j, k] = szz[i, j, k] = 0.0
            sxy[i, j, k] = sxz[i, j, k] = syz[i, j, k] = 0.0
            mass[i, j, k] = amt
            phi[i, j, k] = min(1.0, amt / r)
            if mass[i, j, k] > r:
                overflow = float(mass[i, j, k] - r)
                mass[i, j, k] = r
                phi[i, j, k] = 1.0
                ctype[i, j, k] = CELL_LIQUID
                _deposit(i, j, k + 1, overflow)
            return
        if ct == CELL_LIQUID:
            _deposit(i, j, k + 1, amt)
            return
        new_m = float(mass[i, j, k]) + amt
        if new_m <= r:
            mass[i, j, k] = new_m
            phi[i, j, k] = new_m / r
            return
        overflow = new_m - r
        mass[i, j, k] = r
        phi[i, j, k] = 1.0
        ctype[i, j, k] = CELL_LIQUID
        _deposit(i, j, k + 1, overflow)

    moved = 0.0
    n_orphans = 0
    receivers = np.argwhere(main_cols)
    max_cells_i = int(max(1, max_cells))

    for cid, cells in enumerate(comps):
        if cid == main_id:
            continue
        n_cells = len(cells)
        min_z = min(k for _, _, k in cells)
        elevated = min_z > elevation_cut
        small = n_cells <= max_cells_i
        if not (small or elevated):
            continue
        pot = 0.0
        for i, j, k in cells:
            pot += _clear_cell(i, j, k)
        if pot <= 1.0e-8:
            continue
        n_orphans += 1
        moved += pot
        share = pot / float(receivers.shape[0])
        for ridx in range(receivers.shape[0]):
            i = int(receivers[ridx, 0])
            j = int(receivers[ridx, 1])
            zl = int(z_top[i, j])
            _deposit(i, j, max(0, zl), share)

    if moved <= 1.0e-8:
        return 0.0, 0

    device = buf.device
    buf.mass.assign(wp.array(mass.astype(np.float32), dtype=float, device=device))
    buf.phi.assign(wp.array(phi.astype(np.float32), dtype=float, device=device))
    buf.rho.assign(wp.array(rho.astype(np.float32), dtype=float, device=device))
    buf.ux.assign(wp.array(ux.astype(np.float32), dtype=float, device=device))
    buf.uy.assign(wp.array(uy.astype(np.float32), dtype=float, device=device))
    buf.uz.assign(wp.array(uz.astype(np.float32), dtype=float, device=device))
    buf.sxx.assign(wp.array(sxx.astype(np.float32), dtype=float, device=device))
    buf.syy.assign(wp.array(syy.astype(np.float32), dtype=float, device=device))
    buf.szz.assign(wp.array(szz.astype(np.float32), dtype=float, device=device))
    buf.sxy.assign(wp.array(sxy.astype(np.float32), dtype=float, device=device))
    buf.sxz.assign(wp.array(sxz.astype(np.float32), dtype=float, device=device))
    buf.syz.assign(wp.array(syz.astype(np.float32), dtype=float, device=device))
    buf.cell_type.assign(
        wp.array(ctype.astype(np.int32), dtype=wp.int32, device=device)
    )
    return float(moved), int(n_orphans)


@wp.kernel
def bubble_calculate_disjoint_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    bubble_tag: wp.array3d(dtype=wp.int32),
    disjoin_force: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Home ``calculate_disjoint``: ray along −∇φ, different-tag interface → force."""
    i, j, k = wp.tid()
    disjoin_force[i, j, k] = 0.0
    if int(cell[i, j, k]) != CELL_INTERFACE:
        return
    tag = int(bubble_tag[i, j, k])
    if tag <= 0:
        return

    # Approximate interface normal from φ (points toward liquid).
    gx = float(0.0)
    gy = float(0.0)
    gz = float(0.0)
    if i + 1 < nx and i - 1 >= 0:
        gx = phi[i - 1, j, k] - phi[i + 1, j, k]
    elif i + 1 < nx:
        gx = phi[i, j, k] - phi[i + 1, j, k]
    elif i - 1 >= 0:
        gx = phi[i - 1, j, k] - phi[i, j, k]
    if j + 1 < ny and j - 1 >= 0:
        gy = phi[i, j - 1, k] - phi[i, j + 1, k]
    elif j + 1 < ny:
        gy = phi[i, j, k] - phi[i, j + 1, k]
    elif j - 1 >= 0:
        gy = phi[i, j - 1, k] - phi[i, j, k]
    if k + 1 < nz and k - 1 >= 0:
        gz = phi[i, j, k - 1] - phi[i, j, k + 1]
    elif k + 1 < nz:
        gz = phi[i, j, k] - phi[i, j, k + 1]
    elif k - 1 >= 0:
        gz = phi[i, j, k - 1] - phi[i, j, k]
    glen = wp.sqrt(gx * gx + gy * gy + gz * gz)
    if glen < 1.0e-8:
        return
    inv_g = 1.0 / glen
    nx_h = gx * inv_g
    ny_h = gy * inv_g
    nz_h = gz * inv_g

    best = float(0.0)
    for jk in range(1, 20):
        t = float(jk) * 0.2
        # Home: step opposite the normal (toward approaching foreign interface).
        x12 = int(wp.round(float(i) - t * nx_h))
        y12 = int(wp.round(float(j) - t * ny_h))
        z12 = int(wp.round(float(k) - t * nz_h))
        if x12 < 0 or x12 >= nx or y12 < 0 or y12 >= ny or z12 < 0 or z12 >= nz:
            continue
        nt = int(bubble_tag[x12, y12, z12])
        if nt <= 0 or nt == tag:
            continue
        if int(cell[x12, y12, z12]) != CELL_INTERFACE:
            continue
        # Simplified gap measure (Home uses PLIC center offset).
        alpha = phi[x12, y12, z12]
        d = t - (1.0 - alpha)
        if d < 0.0:
            d = 0.0
        cand = 1.0 - d / 4.0
        if cand > best:
            best = cand
    if best > 0.0:
        disjoin_force[i, j, k] = best


# ---------------------------------------------------------------------------
# §4.2 bubble pressure — Home-aligned (incremental + CCL on merge/split)
# ---------------------------------------------------------------------------

MAX_BUBBLES: int = 4096


@wp.kernel
def bubble_clear_liquid_tags_kernel(
    cell: wp.array3d(dtype=wp.int32),
    bubble_tag: wp.array3d(dtype=wp.int32),
    bubble_tag_prev: wp.array3d(dtype=wp.int32),
) -> None:
    """Liquid cells drop tags (Home previous_tag / tag_matrix clear)."""
    i, j, k = wp.tid()
    tag = int(bubble_tag[i, j, k])
    bubble_tag_prev[i, j, k] = tag
    if int(cell[i, j, k]) == CELL_LIQUID:
        bubble_tag[i, j, k] = 0


@wp.kernel
def bubble_propagate_tags_kernel(
    cell: wp.array3d(dtype=wp.int32),
    bubble_tag: wp.array3d(dtype=wp.int32),
    merge_flag: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Inherit neighbor tag on new G/I; conflicting tags → merge (Home get_tag)."""
    i, j, k = wp.tid()
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        return
    if int(bubble_tag[i, j, k]) > 0:
        return
    t0 = int(0)
    conflict = int(0)
    for s in range(6):
        di = int(0)
        dj = int(0)
        dk = int(0)
        if s == 0:
            di = 1
        elif s == 1:
            di = -1
        elif s == 2:
            dj = 1
        elif s == 3:
            dj = -1
        elif s == 4:
            dk = 1
        else:
            dk = -1
        ni = i + di
        nj = j + dj
        nk = k + dk
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            continue
        nt = int(bubble_tag[ni, nj, nk])
        if nt <= 0:
            continue
        if t0 == 0:
            t0 = nt
        elif nt != t0:
            conflict = 1
    if conflict != 0:
        wp.atomic_max(merge_flag, 0, 1)
    elif t0 > 0:
        bubble_tag[i, j, k] = t0
    # else: leave tag=0 → gas_rho uses rho_g0 until merge/split CCL (Home).


@wp.kernel
def bubble_volume_zero_kernel(
    volume: wp.array(dtype=float),
    touch_top: wp.array(dtype=wp.int32),
    n: int,
) -> None:
    i = wp.tid()
    if i < n:
        volume[i] = 0.0
        touch_top[i] = 0


@wp.kernel
def bubble_touch_top_zero_kernel(
    touch_top: wp.array(dtype=wp.int32),
    n: int,
) -> None:
    i = wp.tid()
    if i < n:
        touch_top[i] = 0


@wp.kernel
def bubble_touch_top_slab_kernel(
    cell: wp.array3d(dtype=wp.int32),
    bubble_tag: wp.array3d(dtype=wp.int32),
    touch_top: wp.array(dtype=wp.int32),
    max_bubbles: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Refresh atmosphere touch flags from the top slab only."""
    i, j = wp.tid()
    if i >= nx or j >= ny or nz < 1:
        return
    k = nz - 1
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        return
    tag = int(bubble_tag[i, j, k])
    if tag > 0 and tag <= max_bubbles:
        wp.atomic_max(touch_top, tag - 1, 1)


@wp.kernel
def bubble_volume_from_delta_phi_kernel(
    delta_phi: wp.array3d(dtype=float),
    bubble_tag: wp.array3d(dtype=wp.int32),
    bubble_tag_prev: wp.array3d(dtype=wp.int32),
    volume: wp.array(dtype=float),
    max_bubbles: int,
) -> None:
    """Home bubble_volume_update: volume[tag] -= Δφ."""
    i, j, k = wp.tid()
    dphi = delta_phi[i, j, k]
    if dphi == 0.0:
        return
    tag = int(bubble_tag[i, j, k])
    if tag <= 0:
        tag = int(bubble_tag_prev[i, j, k])
        bubble_tag_prev[i, j, k] = 0
    if tag > 0 and tag <= max_bubbles:
        wp.atomic_add(volume, tag - 1, -dphi)
    delta_phi[i, j, k] = 0.0


@wp.kernel
def bubble_volume_accumulate_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    bubble_tag: wp.array3d(dtype=wp.int32),
    volume: wp.array(dtype=float),
    touch_top: wp.array(dtype=wp.int32),
    max_bubbles: int,
    nz: int,
) -> None:
    """Full re-sum V = Σ(1-φ) per tag (CCL path / fallback)."""
    i, j, k = wp.tid()
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        return
    tag = int(bubble_tag[i, j, k])
    if tag <= 0 or tag > max_bubbles:
        return
    gas = 1.0 - phi[i, j, k]
    if gas < 0.0:
        gas = 0.0
    wp.atomic_add(volume, tag - 1, gas)
    if k >= nz - 1:
        wp.atomic_max(touch_top, tag - 1, 1)


@wp.kernel
def bubble_scatter_gas_rho_kernel(
    cell: wp.array3d(dtype=wp.int32),
    bubble_tag: wp.array3d(dtype=wp.int32),
    bubble_rho: wp.array(dtype=float),
    gas_rho: wp.array3d(dtype=float),
    rho_g0: float,
    max_bubbles: int,
) -> None:
    i, j, k = wp.tid()
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        gas_rho[i, j, k] = rho_g0
        return
    tag = int(bubble_tag[i, j, k])
    if tag <= 0 or tag > max_bubbles:
        gas_rho[i, j, k] = rho_g0
        return
    gas_rho[i, j, k] = bubble_rho[tag - 1]


# ---- GPU CCL (6-connected union-find; Home tDCCL role) --------------------

@wp.func
def _ccl_find_readonly(parent: wp.array(dtype=wp.int32), i: int) -> int:
    r = i
    for _ in range(64):
        p = parent[r]
        if p == r or p < 0:
            return r
        r = p
    return r


@wp.kernel
def bubble_ccl_init_kernel(
    cell: wp.array3d(dtype=wp.int32),
    parent: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    i, j, k = wp.tid()
    idx = (k * ny + j) * nx + i
    ct = int(cell[i, j, k])
    if ct == CELL_GAS or ct == CELL_INTERFACE:
        parent[idx] = idx
    else:
        parent[idx] = -1


@wp.kernel
def bubble_ccl_hook_kernel(
    cell: wp.array3d(dtype=wp.int32),
    parent: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Union adjacent G/I cells (atomicMin on roots)."""
    i, j, k = wp.tid()
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        return
    idx = (k * ny + j) * nx + i
    ra = _ccl_find_readonly(parent, idx)
    for s in range(6):
        di = int(0)
        dj = int(0)
        dk = int(0)
        if s == 0:
            di = 1
        elif s == 1:
            di = -1
        elif s == 2:
            dj = 1
        elif s == 3:
            dj = -1
        elif s == 4:
            dk = 1
        else:
            dk = -1
        ni = i + di
        nj = j + dj
        nk = k + dk
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            continue
        nct = int(cell[ni, nj, nk])
        if nct != CELL_GAS and nct != CELL_INTERFACE:
            continue
        nidx = (nk * ny + nj) * nx + ni
        rb = _ccl_find_readonly(parent, nidx)
        if ra == rb:
            continue
        lo = ra
        hi = rb
        if lo > hi:
            lo = rb
            hi = ra
        old = wp.atomic_min(parent, hi, lo)
        if old != lo:
            wp.atomic_max(changed, 0, 1)


@wp.kernel
def bubble_ccl_compress_kernel(
    parent: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
    n: int,
) -> None:
    """Path-halving compression."""
    i = wp.tid()
    if i >= n:
        return
    p = parent[i]
    if p < 0:
        return
    gp = parent[p]
    if gp != p and gp >= 0:
        parent[i] = gp
        wp.atomic_max(changed, 0, 1)


@wp.kernel
def bubble_ccl_flatten_kernel(
    parent: wp.array(dtype=wp.int32),
    n: int,
) -> None:
    """Full flatten to root (after UF converged)."""
    i = wp.tid()
    if i >= n:
        return
    if parent[i] < 0:
        return
    r = _ccl_find_readonly(parent, i)
    parent[i] = r


@wp.kernel
def bubble_ccl_enumerate_roots_kernel(
    parent: wp.array(dtype=wp.int32),
    root_dense: wp.array(dtype=wp.int32),
    counter: wp.array(dtype=wp.int32),
    n: int,
    max_bubbles: int,
) -> None:
    i = wp.tid()
    if i >= n:
        return
    root_dense[i] = 0
    if parent[i] != i:
        return
    nid = wp.atomic_add(counter, 0, 1)
    if nid < max_bubbles:
        root_dense[i] = nid + 1


@wp.kernel
def bubble_ccl_write_tags_kernel(
    cell: wp.array3d(dtype=wp.int32),
    parent: wp.array(dtype=wp.int32),
    root_dense: wp.array(dtype=wp.int32),
    bubble_tag: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
) -> None:
    i, j, k = wp.tid()
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        bubble_tag[i, j, k] = 0
        return
    idx = (k * ny + j) * nx + i
    r = parent[idx]
    if r < 0:
        bubble_tag[i, j, k] = 0
        return
    bubble_tag[i, j, k] = root_dense[r]


@wp.kernel
def bubble_reduce_label_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    bubble_tag: wp.array3d(dtype=wp.int32),
    bubble_tag_prev: wp.array3d(dtype=wp.int32),
    old_rho: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    init_volume: wp.array(dtype=float),
    touch_top: wp.array(dtype=wp.int32),
    max_bubbles: int,
    old_count: int,
    nz: int,
) -> None:
    """Home reduce_label_rho: V and V₀ from new labels + old ρ."""
    i, j, k = wp.tid()
    ct = int(cell[i, j, k])
    if ct != CELL_GAS and ct != CELL_INTERFACE:
        return
    tag = int(bubble_tag[i, j, k])
    if tag <= 0 or tag > max_bubbles:
        return
    gas = 1.0 - phi[i, j, k]
    if gas < 0.0:
        gas = 0.0
    wp.atomic_add(volume, tag - 1, gas)
    if k >= nz - 1:
        wp.atomic_max(touch_top, tag - 1, 1)
    pt = int(bubble_tag_prev[i, j, k])
    w = float(1.0)
    if pt > 0 and pt <= old_count:
        w = old_rho[pt - 1]
    wp.atomic_add(init_volume, tag - 1, gas * w)


def _ensure_ccl_buffers(buf: "HomeVofGpuBuffers") -> None:
    nx, ny, nz = buf.shape
    n = nx * ny * nz
    device = buf.device
    parent = getattr(buf, "ccl_parent", None)
    if parent is not None and int(parent.shape[0]) == n:
        return
    buf.ccl_parent = wp.zeros(n, dtype=wp.int32, device=device)
    buf.ccl_root_dense = wp.zeros(n, dtype=wp.int32, device=device)
    buf.ccl_changed = wp.zeros(1, dtype=wp.int32, device=device)
    buf.ccl_count = wp.zeros(1, dtype=wp.int32, device=device)


def bubble_ccl_gpu(
    buf: "HomeVofGpuBuffers",
    *,
    max_iters: int = 256,
    check_every: int = 8,
) -> int:
    """6-connected GPU CCL on G∪I → ``bubble_tag`` (1..N). Returns N."""
    _ensure_ccl_buffers(buf)
    nx, ny, nz = buf.shape
    n = nx * ny * nz
    dim = (nx, ny, nz)
    device = buf.device
    parent = buf.ccl_parent
    root_dense = buf.ccl_root_dense
    changed = buf.ccl_changed
    counter = buf.ccl_count

    wp.launch(
        bubble_ccl_init_kernel,
        dim=dim,
        inputs=[buf.cell_type, parent, nx, ny, nz],
        device=device,
    )
    every = max(1, int(check_every))
    for it in range(max_iters):
        if it % every == 0:
            changed.zero_()
        wp.launch(
            bubble_ccl_hook_kernel,
            dim=dim,
            inputs=[buf.cell_type, parent, changed, nx, ny, nz],
            device=device,
        )
        wp.launch(
            bubble_ccl_compress_kernel,
            dim=n,
            inputs=[parent, changed, n],
            device=device,
        )
        if (it % every) == (every - 1):
            wp.synchronize_device(device)
            if int(changed.numpy()[0]) == 0:
                break
    wp.launch(
        bubble_ccl_flatten_kernel,
        dim=n,
        inputs=[parent, n],
        device=device,
    )
    for _ in range(8):
        changed.zero_()
        wp.launch(
            bubble_ccl_compress_kernel,
            dim=n,
            inputs=[parent, changed, n],
            device=device,
        )
        wp.launch(
            bubble_ccl_flatten_kernel,
            dim=n,
            inputs=[parent, n],
            device=device,
        )
        wp.synchronize_device(device)
        if int(changed.numpy()[0]) == 0:
            break

    counter.zero_()
    wp.launch(
        bubble_ccl_enumerate_roots_kernel,
        dim=n,
        inputs=[parent, root_dense, counter, n, MAX_BUBBLES],
        device=device,
    )
    wp.synchronize_device(device)
    n_bub = int(counter.numpy()[0])
    if n_bub > MAX_BUBBLES:
        n_bub = MAX_BUBBLES
    wp.launch(
        bubble_ccl_write_tags_kernel,
        dim=dim,
        inputs=[buf.cell_type, parent, root_dense, buf.bubble_tag, nx, ny, nz],
        device=device,
    )
    return n_bub


def update_bubble_pressure_ccl_gpu(
    buf: "HomeVofGpuBuffers",
    *,
    rho_g0: float = 1.0,
    atm_volume: float = 1.0e6,
) -> dict[str, float | int]:
    """Full relabel on GPU + Home-style V₀ inherit (replace host CCL)."""
    nx, ny, nz = buf.shape
    dim = (nx, ny, nz)
    device = buf.device
    max_b = MAX_BUBBLES
    old_count = int(getattr(buf, "_bubble_count", 0))

    # Snapshot pre-CCL tags + ρ table (Home reduce_label_rho).
    wp.copy(buf.bubble_tag_prev, buf.bubble_tag)
    old_rho = buf.bubble_rho_gpu

    n_bub = bubble_ccl_gpu(buf)

    wp.launch(
        bubble_volume_zero_kernel,
        dim=max_b,
        inputs=[buf.bubble_volume_gpu, buf.bubble_touch_top, max_b],
        device=device,
    )
    buf.bubble_init_volume_gpu.zero_()
    wp.launch(
        bubble_reduce_label_kernel,
        dim=dim,
        inputs=[
            buf.cell_type, buf.phi, buf.bubble_tag, buf.bubble_tag_prev,
            old_rho,
            buf.bubble_volume_gpu, buf.bubble_init_volume_gpu, buf.bubble_touch_top,
            max_b, old_count, nz,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    if n_bub <= 0:
        buf._bubble_count = 0
        buf._bubble_volume = np.zeros(0, dtype=np.float64)
        buf._bubble_init_volume = np.zeros(0, dtype=np.float64)
        buf._bubble_rho = np.zeros(0, dtype=np.float64)
        buf.gas_rho.fill_(float(rho_g0))
        return {
            "n_bubbles": 0,
            "n_trapped": 0,
            "rho_max_bubble": float(rho_g0),
            "did_ccl": 1,
        }

    volumes = buf.bubble_volume_gpu.numpy()[:n_bub].astype(np.float64)
    init_volumes = buf.bubble_init_volume_gpu.numpy()[:n_bub].astype(np.float64)
    touch = buf.bubble_touch_top.numpy()[:n_bub]
    rhos = np.full(n_bub, float(rho_g0), dtype=np.float64)
    atm_cut = min(float(atm_volume), 0.25 * float(nx * ny * nz))

    for bi in range(n_bub):
        vol = float(volumes[bi])
        if vol <= 1.0e-12:
            rhos[bi] = float(rho_g0)
            init_volumes[bi] = vol
            continue
        is_atm = bool(touch[bi]) or vol > atm_cut
        if is_atm:
            rhos[bi] = float(rho_g0)
            init_volumes[bi] = vol
        else:
            v0 = float(init_volumes[bi])
            if v0 <= 1.0e-12:
                v0 = vol
                init_volumes[bi] = v0
            rhos[bi] = float(np.clip(v0 / vol, 0.2, 1.8))

    rho_pad = np.full(max_b, float(rho_g0), dtype=np.float32)
    rho_pad[:n_bub] = rhos.astype(np.float32)
    init_pad = np.zeros(max_b, dtype=np.float32)
    init_pad[:n_bub] = init_volumes.astype(np.float32)
    vol_pad = np.zeros(max_b, dtype=np.float32)
    vol_pad[:n_bub] = volumes.astype(np.float32)
    buf.bubble_rho_gpu.assign(wp.array(rho_pad, dtype=float, device=device))
    buf.bubble_init_volume_gpu.assign(wp.array(init_pad, dtype=float, device=device))
    buf.bubble_volume_gpu.assign(wp.array(vol_pad, dtype=float, device=device))
    buf._bubble_count = n_bub
    buf._bubble_volume = volumes
    buf._bubble_init_volume = init_volumes
    buf._bubble_rho = rhos

    wp.launch(
        bubble_scatter_gas_rho_kernel,
        dim=dim,
        inputs=[
            buf.cell_type, buf.bubble_tag, buf.bubble_rho_gpu, buf.gas_rho,
            float(rho_g0), max_b,
        ],
        device=device,
    )

    rho_max_b = float(rhos.max()) if n_bub > 0 else float(rho_g0)
    n_trapped = int(
        np.sum((touch == 0) & (volumes <= atm_cut) & (volumes > 1.0e-6))
    )
    return {
        "n_bubbles": n_bub,
        "n_trapped": n_trapped,
        "rho_max_bubble": rho_max_b,
        "did_ccl": 1,
    }


def _ccl_label_gas_interface(
    ctype: np.ndarray,
    max_bubbles: int,
) -> tuple[np.ndarray, int]:
    """6-connected labels on G∪I. Prefer scipy; else host BFS (fallback)."""
    gas_like = (ctype == CELL_GAS) | (ctype == CELL_INTERFACE)
    try:
        from scipy import ndimage

        struct = ndimage.generate_binary_structure(3, 1)
        labels, n_bub = ndimage.label(gas_like.astype(np.uint8), structure=struct)
        labels = labels.astype(np.int32)
        if n_bub > max_bubbles:
            labels[labels > max_bubbles] = 0
            n_bub = max_bubbles
        return labels, int(n_bub)
    except Exception:
        pass

    nx, ny, nz = ctype.shape
    labels = np.zeros((nx, ny, nz), dtype=np.int32)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    n_bub = 0
    gas_idx = np.argwhere(gas_like)
    for si, sj, sk in gas_idx:
        si, sj, sk = int(si), int(sj), int(sk)
        if labels[si, sj, sk] != 0:
            continue
        if n_bub >= max_bubbles:
            break
        n_bub += 1
        stack = [(si, sj, sk)]
        labels[si, sj, sk] = n_bub
        while stack:
            i, j, k = stack.pop()
            for di, dj, dk in neighbors:
                ni, nj, nk = i + di, j + dj, k + dk
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue
                if labels[ni, nj, nk] != 0 or not gas_like[ni, nj, nk]:
                    continue
                labels[ni, nj, nk] = n_bub
                stack.append((ni, nj, nk))
    return labels, int(n_bub)


def update_bubble_pressure_host(
    buf: "HomeVofGpuBuffers",
    *,
    rho_g0: float = 1.0,
    atm_volume: float = 1.0e6,
    max_bubbles: int = MAX_BUBBLES,
) -> dict[str, float | int]:
    """Full CCL relabel (Home handle_merge_split). Prefer rare use only."""
    ctype = buf.cell_type.numpy()
    phi = buf.phi.numpy()
    prev = buf.bubble_tag_prev.numpy()
    nx, ny, nz = buf.shape

    labels, n_bub = _ccl_label_gas_interface(ctype, max_bubbles)
    volumes = np.zeros(n_bub, dtype=np.float64)
    init_volumes = np.zeros(n_bub, dtype=np.float64)
    rhos = np.zeros(n_bub, dtype=np.float64)
    touches_top = np.zeros(n_bub, dtype=bool)

    old_init = getattr(buf, "_bubble_init_volume", None)
    old_count = int(getattr(buf, "_bubble_count", 0))
    atm_cut = min(float(atm_volume), 0.25 * float(nx * ny * nz))

    if n_bub > 0:
        gas_frac = np.maximum(0.0, 1.0 - phi.astype(np.float64))
        for bi in range(n_bub):
            mask = labels == (bi + 1)
            vol = float(gas_frac[mask].sum())
            volumes[bi] = vol
            touches_top[bi] = bool(np.any(mask[:, :, nz - 1])) if nz > 0 else False
            v0 = vol
            if old_init is not None and old_count > 0:
                pts = prev[mask].astype(np.int64).ravel()
                gfs = gas_frac[mask].ravel()
                valid = pts > 0
                if np.any(valid):
                    pts_v = np.clip(pts[valid], 1, old_count)
                    weights = np.bincount(
                        pts_v, weights=gfs[valid], minlength=old_count + 1
                    )
                    best = int(np.argmax(weights[1:])) + 1
                    v0 = float(old_init[best - 1])
                    if v0 <= 1.0e-12:
                        v0 = vol
            init_volumes[bi] = v0

        for bi in range(n_bub):
            vol = float(volumes[bi])
            if vol <= 1.0e-12:
                rhos[bi] = float(rho_g0)
                init_volumes[bi] = vol
                continue
            is_atm = bool(touches_top[bi]) or vol > atm_cut
            if is_atm:
                rhos[bi] = float(rho_g0)
                init_volumes[bi] = vol
            else:
                rhos[bi] = float(np.clip(float(init_volumes[bi]) / vol, 0.2, 1.8))

    gas_rho = np.full((nx, ny, nz), float(rho_g0), dtype=np.float32)
    if n_bub > 0:
        for bi in range(n_bub):
            gas_rho[labels == (bi + 1)] = float(rhos[bi])

    device = buf.device
    buf.bubble_tag.assign(wp.array(labels, dtype=wp.int32, device=device))
    buf.bubble_tag_prev.assign(wp.array(labels.copy(), dtype=wp.int32, device=device))
    buf.gas_rho.assign(wp.array(gas_rho, dtype=float, device=device))

    # Pad GPU tables to MAX_BUBBLES for incremental path.
    vol_pad = np.zeros(max_bubbles, dtype=np.float32)
    init_pad = np.zeros(max_bubbles, dtype=np.float32)
    rho_pad = np.full(max_bubbles, float(rho_g0), dtype=np.float32)
    if n_bub > 0:
        vol_pad[:n_bub] = volumes.astype(np.float32)
        init_pad[:n_bub] = init_volumes.astype(np.float32)
        rho_pad[:n_bub] = rhos.astype(np.float32)
    buf.bubble_volume_gpu.assign(wp.array(vol_pad, dtype=float, device=device))
    buf.bubble_init_volume_gpu.assign(wp.array(init_pad, dtype=float, device=device))
    buf.bubble_rho_gpu.assign(wp.array(rho_pad, dtype=float, device=device))

    buf._bubble_count = n_bub
    buf._bubble_volume = volumes
    buf._bubble_init_volume = init_volumes
    buf._bubble_rho = rhos if n_bub > 0 else np.zeros(0, dtype=np.float64)

    rho_max_b = float(rhos.max()) if n_bub > 0 else float(rho_g0)
    n_trapped = int(
        np.sum((~touches_top) & (volumes <= atm_cut) & (volumes > 1.0e-6))
    ) if n_bub > 0 else 0
    return {
        "n_bubbles": n_bub,
        "n_trapped": n_trapped,
        "rho_max_bubble": rho_max_b,
        "did_ccl": 1,
    }


def update_bubble_pressure(
    buf: "HomeVofGpuBuffers",
    *,
    rho_g0: float = 1.0,
    atm_volume: float = 1.0e6,
    force_ccl: bool = False,
    step_i: int = 0,
    split_ccl_every: int = 1,
) -> dict[str, float | int]:
    """Home ``update_bubble``: tag propagate + V/ρ; GPU CCL on merge/split/init.

    Everyday path: Δφ atomics into bubble.volume (Home). Topology changes run
    Warp union-find CCL then full label reduce.
    """
    del step_i, split_ccl_every
    nx, ny, nz = buf.shape
    dim = (nx, ny, nz)
    device = buf.device
    max_b = MAX_BUBBLES

    split_flag = int(buf.bubble_split_flag.numpy()[0])
    n_old = int(getattr(buf, "_bubble_count", 0))

    # Clear liquid tags, then inherit / detect merge (Home get_tag + recheck).
    wp.launch(
        bubble_clear_liquid_tags_kernel,
        dim=dim,
        inputs=[buf.cell_type, buf.bubble_tag, buf.bubble_tag_prev],
        device=device,
    )
    buf.bubble_merge_flag.zero_()
    wp.launch(
        bubble_propagate_tags_kernel,
        dim=dim,
        inputs=[
            buf.cell_type, buf.bubble_tag,
            buf.bubble_merge_flag,
            nx, ny, nz,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    merge_flag = int(buf.bubble_merge_flag.numpy()[0])
    do_ccl = bool(force_ccl) or n_old <= 0 or merge_flag > 0 or split_flag > 0

    # Home order: apply Δφ volume update first, then CCL on topology change.
    if n_old > 0:
        wp.launch(
            bubble_volume_from_delta_phi_kernel,
            dim=dim,
            inputs=[
                buf.delta_phi, buf.bubble_tag, buf.bubble_tag_prev,
                buf.bubble_volume_gpu, max_b,
            ],
            device=device,
        )
    else:
        buf.delta_phi.zero_()

    if do_ccl:
        stats = update_bubble_pressure_ccl_gpu(
            buf, rho_g0=rho_g0, atm_volume=atm_volume,
        )
        buf.delta_phi.zero_()
        buf.bubble_merge_flag.zero_()
        buf.bubble_split_flag.zero_()
        return stats

    # Incremental: top-touch refresh + ρ = V₀/V (volumes already += -Δφ).
    n_bub = n_old
    wp.launch(
        bubble_touch_top_zero_kernel,
        dim=max_b,
        inputs=[buf.bubble_touch_top, max_b],
        device=device,
    )
    wp.launch(
        bubble_touch_top_slab_kernel,
        dim=(nx, ny),
        inputs=[
            buf.cell_type, buf.bubble_tag, buf.bubble_touch_top,
            max_b, nx, ny, nz,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    volumes = buf.bubble_volume_gpu.numpy()[:n_bub].astype(np.float64)
    touch = buf.bubble_touch_top.numpy()[:n_bub]
    init_volumes = np.array(buf._bubble_init_volume, dtype=np.float64, copy=True)
    if init_volumes.shape[0] < n_bub:
        init_volumes = np.resize(init_volumes, n_bub)
    rhos = np.full(n_bub, float(rho_g0), dtype=np.float64)
    atm_cut = min(float(atm_volume), 0.25 * float(nx * ny * nz))

    for bi in range(n_bub):
        vol = float(volumes[bi])
        if vol <= 1.0e-12:
            rhos[bi] = float(rho_g0)
            init_volumes[bi] = vol
            continue
        is_atm = bool(touch[bi]) or vol > atm_cut
        if is_atm:
            rhos[bi] = float(rho_g0)
            init_volumes[bi] = vol
        else:
            v0 = float(init_volumes[bi])
            if v0 <= 1.0e-12:
                v0 = vol
                init_volumes[bi] = v0
            rhos[bi] = float(np.clip(v0 / vol, 0.2, 1.8))

    rho_pad = np.full(max_b, float(rho_g0), dtype=np.float32)
    rho_pad[:n_bub] = rhos.astype(np.float32)
    init_pad = np.zeros(max_b, dtype=np.float32)
    init_pad[:n_bub] = init_volumes.astype(np.float32)
    vol_pad = np.zeros(max_b, dtype=np.float32)
    vol_pad[:n_bub] = volumes.astype(np.float32)
    buf.bubble_rho_gpu.assign(wp.array(rho_pad, dtype=float, device=device))
    buf.bubble_init_volume_gpu.assign(wp.array(init_pad, dtype=float, device=device))
    buf.bubble_volume_gpu.assign(wp.array(vol_pad, dtype=float, device=device))
    buf._bubble_volume = volumes
    buf._bubble_init_volume = init_volumes
    buf._bubble_rho = rhos

    wp.launch(
        bubble_scatter_gas_rho_kernel,
        dim=dim,
        inputs=[
            buf.cell_type, buf.bubble_tag, buf.bubble_rho_gpu, buf.gas_rho,
            float(rho_g0), max_b,
        ],
        device=device,
    )
    buf.bubble_merge_flag.zero_()
    buf.bubble_split_flag.zero_()

    rho_max_b = float(rhos.max()) if n_bub > 0 else float(rho_g0)
    n_trapped = int(
        np.sum((touch == 0) & (volumes <= atm_cut) & (volumes > 1.0e-6))
    )
    return {
        "n_bubbles": n_bub,
        "n_trapped": n_trapped,
        "rho_max_bubble": rho_max_b,
        "did_ccl": 0,
    }


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
    solid_ux: wp.array
    solid_uy: wp.array
    solid_uz: wp.array
    bubble_tag: wp.array
    bubble_tag_prev: wp.array
    gas_rho: wp.array
    delta_phi: wp.array
    disjoin_force: wp.array
    bubble_volume_gpu: wp.array
    bubble_init_volume_gpu: wp.array
    bubble_rho_gpu: wp.array
    bubble_touch_top: wp.array
    bubble_merge_flag: wp.array
    bubble_split_flag: wp.array
    bubble_need_ccl: wp.array
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

    def swap_cell_tmp_buffers(self) -> None:
        """Ping-pong cell flag scratch (avoids full-field ``wp.copy`` each step)."""
        self.cell_tmp, self.cell_tmp2 = self.cell_tmp2, self.cell_tmp


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

    def zi():
        return wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)

    fk, fux, fuy, fuz = _face_arrays_from_bc(bc, device)
    gas_rho = zf()
    gas_rho.fill_(1.0)
    mb = MAX_BUBBLES
    buf = HomeVofGpuBuffers(
        rho=zf(), ux=zf(), uy=zf(), uz=zf(),
        sxx=zf(), syy=zf(), szz=zf(), sxy=zf(), sxz=zf(), syz=zf(),
        rho_b=zf(), ux_b=zf(), uy_b=zf(), uz_b=zf(),
        sxx_b=zf(), syy_b=zf(), szz_b=zf(), sxy_b=zf(), sxz_b=zf(), syz_b=zf(),
        mass=zf(), mass_b=zf(), massex=zf(), phi=zf(),
        cell_type=zi(), cell_tmp=zi(), cell_tmp2=zi(),
        cx=wp.array(np.asarray(spec.cx, dtype=np.int32), dtype=wp.int32, device=device),
        cy=wp.array(np.asarray(spec.cy, dtype=np.int32), dtype=wp.int32, device=device),
        cz=wp.array(np.asarray(spec.cz, dtype=np.int32), dtype=wp.int32, device=device),
        w=wp.array(np.asarray(spec.weights, dtype=np.float32), dtype=float, device=device),
        opp=wp.array(np.asarray(spec.opposite, dtype=np.int32), dtype=wp.int32, device=device),
        face_kind=fk, face_ux=fux, face_uy=fuy, face_uz=fuz,
        kappa=zf(), kappa_tmp=zf(), solid_phi=zf(),
        solid_ux=zf(), solid_uy=zf(), solid_uz=zf(),
        bubble_tag=zi(), bubble_tag_prev=zi(), gas_rho=gas_rho,
        delta_phi=zf(), disjoin_force=zf(),
        bubble_volume_gpu=wp.zeros(mb, dtype=float, device=device),
        bubble_init_volume_gpu=wp.zeros(mb, dtype=float, device=device),
        bubble_rho_gpu=wp.zeros(mb, dtype=float, device=device),
        bubble_touch_top=wp.zeros(mb, dtype=wp.int32, device=device),
        bubble_merge_flag=wp.zeros(1, dtype=wp.int32, device=device),
        bubble_split_flag=wp.zeros(1, dtype=wp.int32, device=device),
        bubble_need_ccl=wp.zeros(1, dtype=wp.int32, device=device),
        shape=shape, device=device, num_dirs=int(spec.num_dirs),
    )
    # Solid SDF starts as "far fluid" so kappa / fused treat cells as fluid
    # until coupling stamps rigid bodies (solid_phi < 0).
    buf.solid_phi.fill_(1000.0)
    buf.bubble_rho_gpu.fill_(1.0)
    buf._bubble_count = 0
    buf._bubble_volume = np.zeros(0, dtype=np.float64)
    buf._bubble_init_volume = np.zeros(0, dtype=np.float64)
    buf._bubble_rho = np.zeros(0, dtype=np.float64)
    buf._bubble_step = 0
    buf._last_bubble_stats = {
        "n_bubbles": 0,
        "n_trapped": 0,
        "rho_max_bubble": 1.0,
        "did_ccl": 0,
    }
    return buf


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


def sync_solids_from_lbm_state(buf: HomeVofGpuBuffers, state: object) -> None:
    """Copy ``LbmState`` solid SDF / MAC wall velocity into HOME-FREE buffers."""
    solid_phi = getattr(state, "solid_phi", None)
    if solid_phi is None:
        return
    wp.copy(buf.solid_phi, solid_phi)
    vel_u = getattr(state, "vel_solid_u", None)
    vel_v = getattr(state, "vel_solid_v", None)
    vel_w = getattr(state, "vel_solid_w", None)
    if vel_u is None or vel_v is None or vel_w is None:
        buf.solid_ux.zero_()
        buf.solid_uy.zero_()
        buf.solid_uz.zero_()
        return
    nx, ny, nz = buf.shape
    wp.launch(
        home_vof_solid_mac_to_cell_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.solid_phi,
            vel_u,
            vel_v,
            vel_w,
            buf.solid_ux,
            buf.solid_uy,
            buf.solid_uz,
            nx,
            ny,
            nz,
        ],
        device=buf.device,
    )


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
    home_fill_empty: bool = False,
    home_wall_eq: bool = False,
    seal_fg: bool = True,
    bubble_pressure: bool = False,
    bubble_atm_volume: float = 1.0e6,
    bubble_update_every: int = 1,
    bubble_disjoint: bool = False,
    bubble_disjoint_factor: float = 0.032,
    bubble_small_sigma: bool = False,
    bubble_small_vol: float = 64.0,
    bubble_small_six_sigma: float = 2.0e-4,
    bubble_eddy: bool = False,
    bubble_eddy_atm_vol: float = 5.0e6,
) -> None:
    """One HOME-FREE VOF step (fused + surface_1/2/3 + optional film / bubbles)."""
    del eps_phi, rho_liquid
    from wanphys._src.fluid.fluid_grid.lbm.phases import vof_plic

    nx, ny, nz = buf.shape
    dim = (nx, ny, nz)
    g = float(gamma)

    if not bubble_pressure:
        buf.gas_rho.fill_(float(rho_g0))

    use_disjoint = bool(bubble_disjoint) and bool(bubble_pressure)
    use_small = bool(bubble_small_sigma) and bool(bubble_pressure)
    use_eddy = bool(bubble_eddy) and bool(bubble_pressure)

    if use_disjoint:
        wp.launch(
            bubble_calculate_disjoint_kernel,
            dim=dim,
            inputs=[
                buf.cell_type, buf.phi, buf.bubble_tag, buf.disjoin_force,
                nx, ny, nz,
            ],
            device=buf.device,
        )
    else:
        buf.disjoin_force.zero_()

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
            buf.solid_phi, buf.solid_ux, buf.solid_uy, buf.solid_uz,
            buf.rho_b, buf.ux_b, buf.uy_b, buf.uz_b,
            buf.sxx_b, buf.syy_b, buf.szz_b, buf.sxy_b, buf.sxz_b, buf.syz_b,
            buf.mass_b, buf.cell_tmp,
            buf.cx, buf.cy, buf.cz, buf.w, buf.opp,
            buf.face_kind, buf.face_ux, buf.face_uy, buf.face_uz,
            buf.kappa, buf.gas_rho,
            buf.disjoin_force, buf.bubble_tag, buf.bubble_volume_gpu,
            g,
            1 if use_disjoint else 0,
            float(bubble_disjoint_factor),
            1 if use_small else 0,
            float(bubble_small_vol),
            float(bubble_small_six_sigma),
            1 if use_eddy else 0,
            float(bubble_eddy_atm_vol),
            MAX_BUBBLES,
            buf.num_dirs, float(tau), float(fx), float(fy), float(fz),
            1 if home_fill_empty else 0,
            1 if home_wall_eq else 0,
            nx, ny, nz,
        ],
        device=buf.device,
    )
    buf.swap_moment_buffers()
    # Fused wrote flags into cell_tmp; swap so surface* mutate that buffer as cell_tmp2
    # (pointer ping-pong — no N³ int32 copy).
    buf.swap_cell_tmp_buffers()
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
    # Split flag cleared each step; surface3 may set it (Home report_split).
    if bubble_pressure:
        buf.bubble_split_flag.zero_()
    wp.launch(
        home_vof_surface3_kernel,
        dim=dim,
        inputs=[
            buf.cell_tmp2, buf.cell_type,
            buf.rho, buf.mass, buf.massex, buf.phi,
            buf.delta_phi,
            1 if bubble_pressure else 0,
            buf.bubble_split_flag,
            1 if bubble_pressure else 0,
            nx, ny, nz,
        ],
        device=buf.device,
    )
    if seal_fg:
        wp.launch(
            home_vof_seal_fg_kernel,
            dim=dim,
            inputs=[
                buf.cell_type, buf.rho, buf.mass, buf.phi,
                buf.delta_phi, 1 if bubble_pressure else 0,
                nx, ny, nz,
            ],
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

    # Keep rasterized solids empty after surface topology (Home TYPE_S).
    wp.launch(
        home_vof_apply_solid_mask_kernel,
        dim=dim,
        inputs=[
            buf.solid_phi,
            buf.rho, buf.ux, buf.uy, buf.uz,
            buf.sxx, buf.syy, buf.szz, buf.sxy, buf.sxz, buf.syz,
            buf.mass, buf.phi, buf.cell_type,
        ],
        device=buf.device,
    )

    if bubble_pressure:
        every = max(1, int(bubble_update_every))
        step_i = int(getattr(buf, "_bubble_step", 0))
        if step_i % every == 0:
            stats = update_bubble_pressure(
                buf,
                rho_g0=float(rho_g0),
                atm_volume=float(bubble_atm_volume),
                step_i=step_i,
            )
            buf._last_bubble_stats = stats
        buf._bubble_step = step_i + 1
