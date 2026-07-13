# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM GPU kernels — distribution reconstruction via coefficient table.

Reference:
    [REF] Home-FSLBM inc/3D/gpu/mrUtilFuncGpu3D.h:153-266
    [P4] Wang et al. 2025, Eq.(16-17)
"""

from __future__ import annotations

import warp as wp

from .constants import CS2


@wp.func
def reconstruct_fi_at_index(
    rho: float,
    ux: float, uy: float, uz: float,
    S_xx: float, S_xy: float, S_xz: float,
    S_yy: float, S_yz: float, S_zz: float,
    i: int,
    coeffs: wp.array(dtype=wp.float32),
    w: wp.array(dtype=wp.float32),
) -> float:
    """Reconstruct f_i from stored moments via coefficient table.

    Uses the (27,17) coefficient table extracted from the reference
    code's 27-case switch.  The table is stored flat (27*17) in
    row-major order: row i starts at ``i * 17``.
    """
    # ---- Moment components (matching [REF] pre-scaled names) ----
    A0 = rho
    Ax = ux * A0;  Ay = uy * A0;  Az = uz * A0
    Axx = rho * S_xx;  Ayy = rho * S_yy;  Azz = rho * S_zz
    Axy = rho * S_xy;  Axz = rho * S_xz;  Ayz = rho * S_yz

    Ax_t3  = Ax * 3.0;   Ay_t3  = Ay * 3.0;   Az_t3  = Az * 3.0
    Axx_t3 = Axx * 3.0;  Ayy_t3 = Ayy * 3.0;  Azz_t3 = Azz * 3.0
    Axy_t9 = Axy * 9.0;  Axz_t9 = Axz * 9.0;  Ayz_t9 = Ayz * 9.0

    # Third-order [P4] Eq.(17)
    Axxy = -2.0*rho*uy*ux*ux + 2.0*Axy*ux + Axx*uy
    Axyy = -2.0*rho*ux*uy*uy + 2.0*Axy*uy + Ayy*ux
    Axxz = -2.0*rho*uz*ux*ux + 2.0*Axz*ux + Axx*uz
    Axzz = -2.0*rho*ux*uz*uz + 2.0*Axz*uz + Azz*ux
    Ayyz = -2.0*rho*uz*uy*uy + 2.0*Ayz*uy + Ayy*uz
    Ayzz = -2.0*rho*uy*uz*uz + 2.0*Ayz*uz + Azz*uy
    Axyz = Axz*uy + Ayz*ux + Axy*uz - 2.0*rho*ux*uy*uz

    Axxy_t9 = Axxy * 9.0;  Axyy_t9 = Axyy * 9.0
    Axxz_t9 = Axxz * 9.0;  Axzz_t9 = Axzz * 9.0
    Ayyz_t9 = Ayyz * 9.0;  Ayzz_t9 = Ayzz * 9.0
    Axyz_t27 = Axyz * 27.0

    # ---- Dot product with row i of coefficient table ----
    base = i * 17
    val = (
        A0        * coeffs[base + 0]
        + Ax_t3   * coeffs[base + 1]
        + Ay_t3   * coeffs[base + 2]
        + Az_t3   * coeffs[base + 3]
        + Axx_t3  * coeffs[base + 4]
        + Axy_t9  * coeffs[base + 5]
        + Axz_t9  * coeffs[base + 6]
        + Ayy_t3  * coeffs[base + 7]
        + Ayz_t9  * coeffs[base + 8]
        + Azz_t3  * coeffs[base + 9]
        + Axxy_t9 * coeffs[base + 10]
        + Axyy_t9 * coeffs[base + 11]
        + Axxz_t9 * coeffs[base + 12]
        + Axzz_t9 * coeffs[base + 13]
        + Ayyz_t9 * coeffs[base + 14]
        + Ayzz_t9 * coeffs[base + 15]
        + Axyz_t27 * coeffs[base + 16]
    )

    return val * float(w[i])


# ============================================================================
# Initialisation kernel
# ============================================================================
@wp.kernel
def initialize_equilibrium_kernel(
    f_mom: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    flag: wp.array3d(dtype=wp.int32),
    total_num: int, nx: int, ny: int, nz: int,
    rho0: float, u0_x: float, u0_y: float, u0_z: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    cur_ind = k * ny * nx + j * nx + i
    f = flag[i, j, k]
    if (f & 0b00001000) != 0:
        return
    is_fluid = (f & 0b00000111) == 0b00000001
    f_mom[cur_ind + 0*total_num] = rho0
    f_mom[cur_ind + 1*total_num] = u0_x
    f_mom[cur_ind + 2*total_num] = u0_y
    f_mom[cur_ind + 3*total_num] = u0_z
    f_mom[cur_ind + 4*total_num] = u0_x * u0_x
    f_mom[cur_ind + 5*total_num] = u0_x * u0_y
    f_mom[cur_ind + 6*total_num] = u0_x * u0_z
    f_mom[cur_ind + 7*total_num] = u0_y * u0_y
    f_mom[cur_ind + 8*total_num] = u0_y * u0_z
    f_mom[cur_ind + 9*total_num] = u0_z * u0_z
    if is_fluid:
        mass[i, j, k] = rho0
        phi[i, j, k] = 1.0


# ============================================================================
# Verification kernel
# ============================================================================
@wp.kernel
def reconstruct_all_f_kernel(
    f_mom: wp.array(dtype=float),
    f_out: wp.array(dtype=float),
    total_num: int, nx: int, ny: int, nz: int,
    coeffs: wp.array(dtype=wp.float32),
    w: wp.array(dtype=wp.float32),
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    cur_ind = k * ny * nx + j * nx + i

    rho_v = f_mom[cur_ind + 0*total_num]
    ux_v  = f_mom[cur_ind + 1*total_num]
    uy_v  = f_mom[cur_ind + 2*total_num]
    uz_v  = f_mom[cur_ind + 3*total_num]
    Sxx   = f_mom[cur_ind + 4*total_num]
    Sxy   = f_mom[cur_ind + 5*total_num]
    Sxz   = f_mom[cur_ind + 6*total_num]
    Syy   = f_mom[cur_ind + 7*total_num]
    Syz   = f_mom[cur_ind + 8*total_num]
    Szz   = f_mom[cur_ind + 9*total_num]

    for d in range(27):
        f_out[d*total_num + cur_ind] = reconstruct_fi_at_index(
            rho_v, ux_v, uy_v, uz_v,
            Sxx, Sxy, Sxz, Syy, Syz, Szz, d, coeffs, w,
        )
