# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Path B: hydrostatic F_α,H in ME + (optional) modified-pressure fluid.

Under Guo + uniform ρ≈1, Eq.32's ``f*+f−2w`` cancels and ME has almost no
Archimedes lift. Modified-pressure LBM (Liu; Guo et al. JCP 2025):

    collision: Guo g → 0
    FS BC:     ρ_G(z) = ρ0 − ρ0 g_z (z − z_ref) / c_s²
    ME:        Eq.32 + F_α,H with ρ_G,w from free-surface depth

This module adds the ME hydrostatic correction onto ``body_f`` after ordinary
ME. Fluid-side zero-Guo / ρ_G(z) is handled in ``vof_warp`` when
``vof_mod_pressure_me`` is on. Default Guo FSLBM is unchanged when the flag
is off.

Not a copy of any GPL source.
"""

from __future__ import annotations

import warp as wp

CELL_GAS: int = 0
CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2


@wp.func
def _feq_w(
    w: float, rho: float, ux: float, uy: float, uz: float,
    cx: float, cy: float, cz: float,
) -> float:
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux * ux + uy * uy + uz * uz
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


@wp.kernel
def hydro_me_mark_fs_k_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=float),
    fs_k: wp.array2d(dtype=wp.int32),
    phi_wet: float,
    nz: int,
) -> None:
    i, j = wp.tid()
    fs_k[i, j] = -1
    for t in range(nz):
        k = nz - 1 - t
        if solid[i, j, k] < 0.0:
            continue
        ct = int(cell[i, j, k])
        if ct == CELL_LIQUID:
            fs_k[i, j] = k
            return
        if ct == CELL_INTERFACE and phi[i, j, k] > phi_wet:
            fs_k[i, j] = k
            return


@wp.kernel
def accumulate_hydro_me_correction_kernel(
    phi: wp.array3d(dtype=float),
    cell: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    solid_ux: wp.array3d(dtype=float),
    solid_uy: wp.array3d(dtype=float),
    solid_uz: wp.array3d(dtype=float),
    fs_k: wp.array2d(dtype=wp.int32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    dh: float,
    force_conversion: float,
    g_latt_z: float,
    rho0: float,
    me_phi_min: float,
    vertical_only: int,
    num_dirs: int,
    nx: int,
    ny: int,
    nz: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
) -> None:
    """Add F_α,H onto ``body_f`` (same units as Eq.32 ME)."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return

    ctype = int(cell[i, j, k])
    me_phi = float(0.0)
    if ctype == CELL_LIQUID:
        me_phi = 1.0
    elif ctype == CELL_INTERFACE:
        me_phi = phi[i, j, k]
        if me_phi < me_phi_min:
            return
    else:
        return
    if me_phi <= 0.0:
        return

    k_fs = int(fs_k[i, j])
    if k_fs < 0:
        return

    cs2 = 1.0 / 3.0
    z_fs = float(k_fs) + 0.5
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

        # Wall point ≈ link midpoint (lattice z for ρ_G,w).
        z_w = float(k) + 0.5 - 0.5 * float(czi)
        # Paper-style: ρ_G,w = ρ0 − ρ0 g_z (z_w − z_fs) / c_s²
        rho_h = rho0 - rho0 * g_latt_z * (z_w - z_fs) / cs2
        if rho_h < 0.05:
            rho_h = 0.05
        if rho_h > 5.0:
            rho_h = 5.0

        uxp = solid_ux[ni, nj, nk]
        uyp = solid_uy[ni, nj, nk]
        uzp = solid_uz[ni, nj, nk]
        f0 = _feq_w(w, rho0, uxp, uyp, uzp, float(cxi), float(cyi), float(czi))
        fh = _feq_w(w, rho_h, uxp, uyp, uzp, float(cxi), float(cyi), float(czi))
        # s_mod − s_std with std reference feq(ρ0):  −2 fh + 2 f0
        s_corr = 2.0 * (f0 - fh)
        fx = me_phi * s_corr * float(cxi)
        fy = me_phi * s_corr * float(cyi)
        fz = me_phi * s_corr * float(czi)
        # Keep Archimedes-like lift; drop noisy horizontal F_H (hurts FSI bore).
        if vertical_only != 0:
            fx = 0.0
            fy = 0.0
        dF = wp.vec3(-fx, -fy, -fz) * force_conversion
        f2 = dF[0] * dF[0] + dF[1] * dF[1] + dF[2] * dF[2]
        if not wp.isfinite(f2):
            continue
        com_world = wp.transform_point(body_q[body_id], body_com[body_id])
        c_dir = wp.vec3(float(cxi), float(cyi), float(czi))
        link_mid = cell_center - c_dir * (0.5 * dh)
        dtau = wp.cross(link_mid - com_world, dF)
        if not wp.isfinite(dtau[0] + dtau[1] + dtau[2]):
            continue
        wp.atomic_add(body_f, body_id, wp.spatial_vector(dF, dtau))


def ensure_hydro_me_scratch(
    *,
    device: wp.context.Device | str,
    nx: int,
    ny: int,
    scratch: dict | None,
) -> dict:
    if (
        scratch is not None
        and scratch.get("nx") == nx
        and scratch.get("ny") == ny
    ):
        return scratch
    return {
        "nx": int(nx),
        "ny": int(ny),
        "fs_k": wp.zeros((nx, ny), dtype=wp.int32, device=device),
    }


def apply_hydro_me_correction_gpu(
    *,
    buf,
    solid_body_id: wp.array,
    body_q: wp.array,
    body_com: wp.array,
    body_f: wp.array,
    dh: float,
    force_scale: float,
    g_latt_z: float,
    rho0: float = 1.0,
    me_phi_min: float = 0.0,
    phi_wet: float = 0.05,
    vertical_only: bool = True,
    scratch: dict | None = None,
) -> dict:
    """Mark column FS and add hydrostatic ME correction into ``body_f``."""
    nx, ny, nz = buf.shape
    scratch = ensure_hydro_me_scratch(
        device=buf.device, nx=nx, ny=ny, scratch=scratch
    )
    wp.launch(
        hydro_me_mark_fs_k_kernel,
        dim=(nx, ny),
        inputs=[
            buf.cell_type,
            buf.phi,
            buf.solid_phi,
            scratch["fs_k"],
            float(phi_wet),
            int(nz),
        ],
        device=buf.device,
    )
    wp.launch(
        accumulate_hydro_me_correction_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.phi,
            buf.cell_type,
            buf.solid_phi,
            solid_body_id,
            buf.solid_ux,
            buf.solid_uy,
            buf.solid_uz,
            scratch["fs_k"],
            buf.cx,
            buf.cy,
            buf.cz,
            buf.w,
            float(dh),
            float(force_scale),
            float(g_latt_z),
            float(rho0),
            float(me_phi_min),
            1 if vertical_only else 0,
            int(buf.num_dirs),
            int(nx),
            int(ny),
            int(nz),
            body_q,
            body_com,
            body_f,
        ],
        device=buf.device,
    )
    return scratch
