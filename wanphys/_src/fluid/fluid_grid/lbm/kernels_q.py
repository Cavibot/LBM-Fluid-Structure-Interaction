# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Q-parameterized distribution LBM kernels (D3Q19 / D3Q27).

Looped over ``num_dirs`` with device lattice arrays.  Used for D3Q27
single-phase; D3Q19 may keep the unrolled kernels in :mod:`kernels`.
"""

from __future__ import annotations

import warp as wp

from .kernels import _f_eq, _moving_wall_correction, _trt_collide


@wp.func
def _wrap_index(i: int, n: int, periodic: int) -> int:
    if i < 0:
        if periodic != 0:
            return n - 1
        return -1
    if i >= n:
        if periodic != 0:
            return 0
        return -1
    return i


@wp.kernel
def compute_moments_q_kernel(
    f: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    num_dirs: int,
    stride: int,
    nx: int,
    ny: int,
    nz: int,
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
) -> None:
    """Compute density and velocity from Q distributions."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    r = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)
    for d in range(num_dirs):
        val = f[d * stride + idx]
        r = r + val
        mx = mx + float(cx_arr[d]) * val
        my = my + float(cy_arr[d]) * val
        mz = mz + float(cz_arr[d]) * val
    r = wp.clamp(r, 0.005, 10.0)
    rho[i, j, k] = r
    inv = 1.0 / r
    ux[i, j, k] = mx * inv
    uy[i, j, k] = my * inv
    uz[i, j, k] = mz * inv


@wp.kernel
def collide_stream_bounceback_q_kernel(
    f_in: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    f_out: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    opp_arr: wp.array(dtype=wp.int32),
    omega_plus: float,
    omega_minus: float,
    num_dirs: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Fused TRT collide + pull-stream + halfway bounce-back (any Q)."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    rho_w = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]

    # Rest population (no stream).
    f0 = f_in[0 * stride + idx]
    w0 = w_arr[0]
    f_out[0 * stride + idx] = f0 - omega_plus * (
        f0 - _f_eq(w0, rho_w, vx, vy, vz, 0, 0, 0)
    )

    for d in range(1, num_dirs):
        cx = int(cx_arr[d])
        cy = int(cy_arr[d])
        cz = int(cz_arr[d])
        w = w_arr[d]
        opp = int(opp_arr[d])

        si = _wrap_index(i - cx, nx, px)
        sj = _wrap_index(j - cy, ny, py)
        sk = _wrap_index(k - cz, nz, pz)

        is_wall = False
        if si < 0 or sj < 0 or sk < 0:
            is_wall = True
        elif solid_phi[si, sj, sk] < 0.0:
            is_wall = True

        if is_wall:
            f_opp = f_in[opp * stride + idx]
            f_out[d * stride + idx] = (
                f_opp
                - omega_minus
                * (f_opp - _f_eq(w, rho_w, vx, vy, vz, -cx, -cy, -cz))
                + _moving_wall_correction(
                    vel_solid_u,
                    vel_solid_v,
                    vel_solid_w,
                    i,
                    j,
                    k,
                    cx,
                    cy,
                    cz,
                    w,
                    rho_w,
                    nx,
                    ny,
                    nz,
                )
            )
        else:
            src_idx = si * ny * nz + sj * nz + sk
            f_src = f_in[d * stride + src_idx]
            f_opp_src = f_in[opp * stride + src_idx]
            f_out[d * stride + idx] = _trt_collide(
                f_src,
                f_opp_src,
                rho[si, sj, sk],
                ux[si, sj, sk],
                uy[si, sj, sk],
                uz[si, sj, sk],
                w,
                cx,
                cy,
                cz,
                omega_plus,
                omega_minus,
            )


@wp.kernel
def apply_guo_force_q_kernel(
    f: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    fx: float,
    fy: float,
    fz: float,
    omega: float,
    num_dirs: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Guo body force: ``Δf = (1 - ω/2) · 3 · w · (c · F)``."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    factor = (1.0 - omega * 0.5) * 3.0
    for d in range(num_dirs):
        cx = float(cx_arr[d])
        cy = float(cy_arr[d])
        cz = float(cz_arr[d])
        cf = cx * fx + cy * fy + cz * fz
        if cf != 0.0:
            f[d * stride + idx] = f[d * stride + idx] + factor * w_arr[d] * cf


@wp.kernel
def initialize_equilibrium_q_kernel(
    f: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    rho0: float,
    u0x: float,
    u0y: float,
    u0z: float,
    num_dirs: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Set *f* to Maxwell–Boltzmann equilibrium for uniform (ρ₀, u₀)."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    for d in range(num_dirs):
        f[d * stride + idx] = _f_eq(
            w_arr[d],
            rho0,
            u0x,
            u0y,
            u0z,
            int(cx_arr[d]),
            int(cy_arr[d]),
            int(cz_arr[d]),
        )


@wp.func
def _zou_he_velocity_face(
    f: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    opp_arr: wp.array(dtype=wp.int32),
    idx: int,
    stride: int,
    num_dirs: int,
    nx_in: int,
    ny_in: int,
    nz_in: int,
    vx: float,
    vy: float,
    vz: float,
) -> None:
    """NEBB Zou–He velocity BC for one face (inward lattice normal).

    Density:
        ρ = (Σ_{c·n=0} f + 2 Σ_{c·n<0} f) / (1 - u·n)
    Unknown (c·n>0):
        f_i = f_i^eq + f_ī - f_ī^eq
    Works for D3Q19 and D3Q27 (including body diagonals).
    """
    un = float(nx_in) * vx + float(ny_in) * vy + float(nz_in) * vz
    denom = 1.0 - un
    if denom <= 1.0e-8:
        return

    known = float(0.0)
    for d in range(num_dirs):
        cx = int(cx_arr[d])
        cy = int(cy_arr[d])
        cz = int(cz_arr[d])
        cn = cx * nx_in + cy * ny_in + cz * nz_in
        val = f[d * stride + idx]
        if cn == 0:
            known = known + val
        elif cn < 0:
            known = known + 2.0 * val

    rho_w = known / denom
    if rho_w < 0.05 or rho_w != rho_w:
        return

    for d in range(num_dirs):
        cx = int(cx_arr[d])
        cy = int(cy_arr[d])
        cz = int(cz_arr[d])
        cn = cx * nx_in + cy * ny_in + cz * nz_in
        if cn <= 0:
            continue
        opp = int(opp_arr[d])
        w = w_arr[d]
        feq = _f_eq(w, rho_w, vx, vy, vz, cx, cy, cz)
        feq_opp = _f_eq(w, rho_w, vx, vy, vz, -cx, -cy, -cz)
        f[d * stride + idx] = feq + (f[opp * stride + idx] - feq_opp)


@wp.kernel
def apply_boundary_conditions_q_kernel(
    f: wp.array(dtype=float),
    bc_types: wp.array(dtype=wp.int32),
    bc_vel_x: wp.array(dtype=float),
    bc_vel_y: wp.array(dtype=float),
    bc_vel_z: wp.array(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    opp_arr: wp.array(dtype=wp.int32),
    num_dirs: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Zou–He velocity inlet or convective outflow (any Q).

    Bounce-back (bc_type==0) and periodic (3) are no-ops here.
    """
    i, j, k = wp.tid()

    on_xmin = i == 0
    on_xmax = i == nx - 1
    on_ymin = j == 0
    on_ymax = j == ny - 1
    on_zmin = k == 0
    on_zmax = k == nz - 1

    if not (on_xmin or on_xmax or on_ymin or on_ymax or on_zmin or on_zmax):
        return

    face_count = int(0)
    if on_xmin:
        face_count = face_count + 1
    if on_xmax:
        face_count = face_count + 1
    if on_ymin:
        face_count = face_count + 1
    if on_ymax:
        face_count = face_count + 1
    if on_zmin:
        face_count = face_count + 1
    if on_zmax:
        face_count = face_count + 1
    if face_count != 1:
        return

    idx = i * ny * nz + j * nz + k

    if on_xmin:
        bc = int(bc_types[0])
        if bc == 1:
            _zou_he_velocity_face(
                f,
                cx_arr,
                cy_arr,
                cz_arr,
                w_arr,
                opp_arr,
                idx,
                stride,
                num_dirs,
                1,
                0,
                0,
                bc_vel_x[0],
                bc_vel_y[0],
                bc_vel_z[0],
            )
        elif bc == 2:
            src_idx = 1 * ny * nz + j * nz + k
            for d in range(num_dirs):
                f[d * stride + idx] = f[d * stride + src_idx]

    if on_xmax:
        bc = int(bc_types[1])
        if bc == 1:
            _zou_he_velocity_face(
                f,
                cx_arr,
                cy_arr,
                cz_arr,
                w_arr,
                opp_arr,
                idx,
                stride,
                num_dirs,
                -1,
                0,
                0,
                bc_vel_x[1],
                bc_vel_y[1],
                bc_vel_z[1],
            )
        elif bc == 2:
            src_idx = (nx - 2) * ny * nz + j * nz + k
            for d in range(num_dirs):
                f[d * stride + idx] = f[d * stride + src_idx]

    if on_ymin:
        bc = int(bc_types[2])
        if bc == 1:
            _zou_he_velocity_face(
                f,
                cx_arr,
                cy_arr,
                cz_arr,
                w_arr,
                opp_arr,
                idx,
                stride,
                num_dirs,
                0,
                1,
                0,
                bc_vel_x[2],
                bc_vel_y[2],
                bc_vel_z[2],
            )
        elif bc == 2:
            src_idx = i * ny * nz + 1 * nz + k
            for d in range(num_dirs):
                f[d * stride + idx] = f[d * stride + src_idx]

    if on_ymax:
        bc = int(bc_types[3])
        if bc == 1:
            _zou_he_velocity_face(
                f,
                cx_arr,
                cy_arr,
                cz_arr,
                w_arr,
                opp_arr,
                idx,
                stride,
                num_dirs,
                0,
                -1,
                0,
                bc_vel_x[3],
                bc_vel_y[3],
                bc_vel_z[3],
            )
        elif bc == 2:
            src_idx = i * ny * nz + (ny - 2) * nz + k
            for d in range(num_dirs):
                f[d * stride + idx] = f[d * stride + src_idx]

    if on_zmin:
        bc = int(bc_types[4])
        if bc == 1:
            _zou_he_velocity_face(
                f,
                cx_arr,
                cy_arr,
                cz_arr,
                w_arr,
                opp_arr,
                idx,
                stride,
                num_dirs,
                0,
                0,
                1,
                bc_vel_x[4],
                bc_vel_y[4],
                bc_vel_z[4],
            )
        elif bc == 2:
            src_idx = i * ny * nz + j * nz + 1
            for d in range(num_dirs):
                f[d * stride + idx] = f[d * stride + src_idx]

    if on_zmax:
        bc = int(bc_types[5])
        if bc == 1:
            _zou_he_velocity_face(
                f,
                cx_arr,
                cy_arr,
                cz_arr,
                w_arr,
                opp_arr,
                idx,
                stride,
                num_dirs,
                0,
                0,
                -1,
                bc_vel_x[5],
                bc_vel_y[5],
                bc_vel_z[5],
            )
        elif bc == 2:
            src_idx = i * ny * nz + j * nz + (nz - 2)
            for d in range(num_dirs):
                f[d * stride + idx] = f[d * stride + src_idx]
