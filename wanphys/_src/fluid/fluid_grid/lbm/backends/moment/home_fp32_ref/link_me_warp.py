# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in linkwise momentum exchange for HOME-FREE (no live ``f``).

Reconstructs wall / fluid link populations from moments the same way the fused
kernel does, then applies a Ladd-style link force:

    Δp ∝ −(f_wall + f_opp) · c

This is an original WanPhys port of the classical ME idea for moment SoT —
not a copy of any GPL reference implementation.
"""

from __future__ import annotations

import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.core.hermite import home_reconstruct_f_i


@wp.func
def _feq_w(
    w: float, rho: float, ux: float, uy: float, uz: float,
    cx: float, cy: float, cz: float,
) -> float:
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux * ux + uy * uy + uz * uz
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


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


@wp.func
def _accumulate_link_me(
    body_id: int,
    link_mid: wp.vec3,
    c_dir: wp.vec3,
    f_wall: float,
    f_opp: float,
    vol: float,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
) -> None:
    if body_id < 0:
        return
    df = -(f_wall + f_opp) * vol
    delta_p = c_dir * df
    com_world = wp.transform_point(body_q[body_id], body_com[body_id])
    delta_tau = wp.cross(link_mid - com_world, delta_p)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(delta_p, delta_tau))


@wp.kernel
def accumulate_home_reconstructed_link_me_kernel(
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
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    solid_ux: wp.array3d(dtype=float),
    solid_uy: wp.array3d(dtype=float),
    solid_uz: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    opp_arr: wp.array(dtype=wp.int32),
    dh: float,
    force_scale: float,
    home_wall_eq: int,
    num_dirs: int,
    nx: int,
    ny: int,
    nz: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
) -> None:
    """Reconstructed-link ME (HOME moments) — core ``home_fp32`` fluid→rigid path.

    Uses the same wall / opposite reconstruction as fused moving walls (Eq.24 or
    eq-wall). ``force_scale`` should typically come from
    ``recommended_me_force_scale(dh, dt)``; volume factor is ``dh³ * force_scale``.
    """
    i, j, k = wp.tid()

    if solid_phi[i, j, k] < 0.0:
        return

    rho_c = rho[i, j, k]
    if rho_c <= 1.0e-12:
        return

    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]
    sxx_c = sxx[i, j, k]
    syy_c = syy[i, j, k]
    szz_c = szz[i, j, k]
    sxy_c = sxy[i, j, k]
    sxz_c = sxz[i, j, k]
    syz_c = syz[i, j, k]

    vol = dh * dh * dh * force_scale
    cell_center = wp.vec3(
        (float(i) + 0.5) * dh,
        (float(j) + 0.5) * dh,
        (float(k) + 0.5) * dh,
    )

    for d in range(1, num_dirs):
        cxi = int(cx_arr[d])
        cyi = int(cy_arr[d])
        czi = int(cz_arr[d])
        w = w_arr[d]
        od = int(opp_arr[d])
        ocx = int(cx_arr[od])
        ocy = int(cy_arr[od])
        ocz = int(cz_arr[od])

        ni = i - cxi
        nj = j - cyi
        nk = k - czi
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            continue
        if solid_phi[ni, nj, nk] >= 0.0:
            continue

        body_id = int(solid_body_id[ni, nj, nk])
        if body_id < 0:
            continue

        uxp = solid_ux[ni, nj, nk]
        uyp = solid_uy[ni, nj, nk]
        uzp = solid_uz[ni, nj, nk]

        fon_opp = home_reconstruct_f_i(
            rho_c, vx, vy, vz, sxx_c, syy_c, szz_c, sxy_c, sxz_c, syz_c,
            ocx, ocy, ocz, w_arr[od],
        )
        if home_wall_eq != 0:
            f_wall = _feq_w(
                w, rho_c, uxp, uyp, uzp, float(cxi), float(cyi), float(czi),
            )
        else:
            f_wall = _solid_f_eq24(
                rho_c, vx, vy, vz, sxx_c, syy_c, szz_c, sxy_c, sxz_c, syz_c,
                uxp, uyp, uzp, cxi, cyi, czi, w,
            )

        c_dir = wp.vec3(float(cxi), float(cyi), float(czi))
        link_mid = cell_center - c_dir * (0.5 * dh)
        _accumulate_link_me(
            body_id, link_mid, c_dir, f_wall, fon_opp, vol,
            body_q, body_com, body_f,
        )


def launch_home_reconstructed_link_me(
    *,
    buf,
    solid_body_id: wp.array,
    body_q: wp.array,
    body_com: wp.array,
    body_f: wp.array,
    dh: float,
    force_scale: float,
    home_wall_eq: bool,
) -> None:
    """Host wrapper: accumulate reconstructed-link ME into ``body_f``."""
    nx, ny, nz = buf.shape
    wp.launch(
        accumulate_home_reconstructed_link_me_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.rho, buf.ux, buf.uy, buf.uz,
            buf.sxx, buf.syy, buf.szz, buf.sxy, buf.sxz, buf.syz,
            buf.solid_phi, solid_body_id,
            buf.solid_ux, buf.solid_uy, buf.solid_uz,
            buf.cx, buf.cy, buf.cz, buf.w, buf.opp,
            float(dh), float(force_scale),
            1 if home_wall_eq else 0,
            int(buf.num_dirs),
            nx, ny, nz,
            body_q, body_com, body_f,
        ],
        device=buf.device,
    )
