# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM GPU kernels — distribution reconstruction, stream-collide, MRT.

Reference:
    [REF] Home-FSLBM inc/3D/gpu/mrUtilFuncGpu3D.h:153-266 (reconstruction)
    [REF] Home-FSLBM inc/3D/gpu/mrUtilFuncGpu3D.h:424-471 (MRT collision)
    [REF] Home-FSLBM inc/3D/gpu/mrLbmSolverGpu3D.cu:703-1057 (stream_collide_bvh)
    [P4] Wang et al. 2025, Eq.(16-17) — 3rd-order Hermite expansion.
    [P4] Wang et al. 2025, Eq.(18-21) — central-moment MRT collision.
"""

from __future__ import annotations

import warp as wp

from .constants import CS2, INV_CS2, TYPE_S, TYPE_F, MAX_VELOCITY


# ============================================================================
# Distribution reconstruction  [REF] mrUtilFuncGpu3D.h:153-266
# ============================================================================
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
# Equilibrium distribution  [REF] mrUtilFuncGpu3D::calculate_f_eq
# ============================================================================
@wp.func
def equilibrium_fi(
    rho: float, ux: float, uy: float, uz: float,
    i: int,
    c_x: wp.array(dtype=wp.int32),
    c_y: wp.array(dtype=wp.int32),
    c_z: wp.array(dtype=wp.int32),
    w: wp.array(dtype=wp.float32),
) -> float:
    """Compute equilibrium distribution f_i^eq for direction *i*.

    [P4] Eq.(16) with S=0 (pure equilibrium, second-order expansion).
    """
    cu = float(c_x[i]) * ux + float(c_y[i]) * uy + float(c_z[i]) * uz
    u2 = ux * ux + uy * uy + uz * uz
    wi = float(w[i])
    return rho * wi * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


# ============================================================================
# Initialisation kernel
# ============================================================================
@wp.kernel
def initialize_equilibrium_kernel(
    f_mom: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    flag: wp.array(dtype=wp.int32),
    total_num: int, nx: int, ny: int, nz: int,
    rho0: float, u0_x: float, u0_y: float, u0_z: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    cur_ind = k * ny * nx + j * nx + i
    f = flag[cur_ind]
    if (f & TYPE_S) != 0:
        return
    is_fluid = (f & TYPE_F) == TYPE_F
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
# Flag domain faces  [REF] mrSolver3DGpu boundary setup
# ============================================================================
@wp.kernel
def flag_domain_boundary_kernel(
    flag: wp.array(dtype=wp.int32),
    nx: int, ny: int, nz: int,
    bc_types: wp.array(dtype=wp.int32),
):
    """Mark domain boundary faces as TYPE_S when bc is bounce_back.

    bc_types[0..5] = (-x, +x, -y, +y, -z, +z): 0=bounce_back, 1=periodic.
    flag is a 1D flat array of size nx*ny*nz.
    """
    i, j, k = wp.tid()
    idx = k * ny * nx + j * nx + i
    # -x face
    if i == 0 and bc_types[0] == 0:
        flag[idx] = flag[idx] | TYPE_S
    # +x face
    if i == nx - 1 and bc_types[1] == 0:
        flag[idx] = flag[idx] | TYPE_S
    # -y face
    if j == 0 and bc_types[2] == 0:
        flag[idx] = flag[idx] | TYPE_S
    # +y face
    if j == ny - 1 and bc_types[3] == 0:
        flag[idx] = flag[idx] | TYPE_S
    # -z face
    if k == 0 and bc_types[4] == 0:
        flag[idx] = flag[idx] | TYPE_S
    # +z face
    if k == nz - 1 and bc_types[5] == 0:
        flag[idx] = flag[idx] | TYPE_S


# ============================================================================
# Stream + Collide fused kernel  [REF] stream_collide_bvh (simplified, stage 2)
# ============================================================================
@wp.kernel
def stream_collide_kernel(
    f_mom: wp.array(dtype=float),
    f_mom_post: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    flag: wp.array(dtype=wp.int32),
    force_x: wp.array3d(dtype=float),
    force_y: wp.array3d(dtype=float),
    force_z: wp.array3d(dtype=float),
    total_num: int, nx: int, ny: int, nz: int,
    tau: float,
    gx: float, gy: float, gz: float,
    max_vel: float,
    bc_types: wp.array(dtype=wp.int32),
    coeffs: wp.array(dtype=wp.float32),
    lattice_w: wp.array(dtype=wp.float32),
    c_x: wp.array(dtype=wp.int32),
    c_y: wp.array(dtype=wp.int32),
    c_z: wp.array(dtype=wp.int32),
    opp: wp.array(dtype=wp.int32),
):
    """Fused stream + central-moment MRT collide kernel.

    Stage 2: single-phase LBGK/MRT without free-surface VOF tracking.
    Non-TYPE_S cells are treated as bulk fluid.

    Algorithm (one cell per thread):
        1. Early exit if TYPE_S.
        2. For each direction d, pull f_i from neighbour (stream):
           - Bounce-back if neighbour is TYPE_S or out-of-domain solid face.
           - Otherwise reconstruct from neighbour's 10 stored moments.
        3. Accumulate post-stream moments (rho*, u*, pixx*..pizz*).
        4. Apply body force F = rho* * g.
        5. Velocity clamp: |u*| <= max_vel.
        6. Central-moment MRT collision [REF] mlGetPIAfterCollision.
        7. Convert TRUE post-collision moments to stored stress format.
        8. Write fMomPost.

    Reference:
        [REF] mrLbmSolverGpu3D.cu:703-1057 (stream_collide_bvh)
        [REF] mrUtilFuncGpu3D.h:424-471 (mlGetPIAfterCollision)
    """
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    cur_ind = k * ny * nx + j * nx + i
    flagsn = flag[cur_ind]

    # Early exit for solid cells
    if (flagsn & TYPE_S) != 0:
        return

    # ---- Load current cell moments (stored as stress S_ab) ----
    rho_cur = f_mom[cur_ind + 0 * total_num]
    ux_cur  = f_mom[cur_ind + 1 * total_num]
    uy_cur  = f_mom[cur_ind + 2 * total_num]
    uz_cur  = f_mom[cur_ind + 3 * total_num]
    Sxx_cur = f_mom[cur_ind + 4 * total_num]
    Sxy_cur = f_mom[cur_ind + 5 * total_num]
    Sxz_cur = f_mom[cur_ind + 6 * total_num]
    Syy_cur = f_mom[cur_ind + 7 * total_num]
    Syz_cur = f_mom[cur_ind + 8 * total_num]
    Szz_cur = f_mom[cur_ind + 9 * total_num]

    # ---- Post-stream moments (accumulate over 27 directions) ----
    # Warp requires dynamic variables declared with float() for use in
    # dynamic for-loops.
    rho_star = float(0.0)
    ux_star = float(0.0);   uy_star = float(0.0);   uz_star = float(0.0)
    pixx_star = float(0.0); pixy_star = float(0.0); pixz_star = float(0.0)
    piyy_star = float(0.0); piyz_star = float(0.0); pizz_star = float(0.0)

    # Mass tracking [REF] stream_collide_bvh:806-936
    # Stage 2 (single-phase): compute mass flux fhn - fon for TYPE_F cells.
    massn = float(0.0)
    if (flagsn & TYPE_F) != 0:
        massn = mass[i, j, k]  # start with current cell mass

    for d in range(27):
        nx_i = i - c_x[d]
        ny_i = j - c_y[d]
        nz_i = k - c_z[d]

        use_bounce = False

        # Handle domain-boundary faces — independent ifs, NOT elif!
        # Corner directions have multiple out-of-bounds dims.
        if nx_i < 0:
            if bc_types[0] == 0:
                use_bounce = True
            else:
                nx_i = nx_i + nx
        if nx_i >= nx:
            if bc_types[1] == 0:
                use_bounce = True
            else:
                nx_i = nx_i - nx
        if ny_i < 0:
            if bc_types[2] == 0:
                use_bounce = True
            else:
                ny_i = ny_i + ny
        if ny_i >= ny:
            if bc_types[3] == 0:
                use_bounce = True
            else:
                ny_i = ny_i - ny
        if nz_i < 0:
            if bc_types[4] == 0:
                use_bounce = True
            else:
                nz_i = nz_i + nz
        if nz_i >= nz:
            if bc_types[5] == 0:
                use_bounce = True
            else:
                nz_i = nz_i - nz

        pop_d = float(0.0)

        if use_bounce:
            # Half-way bounce-back for stationary wall:
            # f_hn[d] = f_on[OPPOSITE[d]]
            # Reconstruct the outgoing distribution along the OPPOSITE
            # direction from the CURRENT cell's moments.
            # [REF] standard LBM: f_in,i = f_out,opp(i)
            d_opp = opp[d]
            pop_d = reconstruct_fi_at_index(
                rho_cur, ux_cur, uy_cur, uz_cur,
                Sxx_cur, Sxy_cur, Sxz_cur, Syy_cur, Syz_cur, Szz_cur,
                d_opp, coeffs, lattice_w,
            )
        else:
            # Check if neighbour is a solid cell
            nb_ind = nz_i * ny * nx + ny_i * nx + nx_i
            nb_flags = flag[nb_ind]
            if (nb_flags & TYPE_S) != 0:
                # Half-way bounce-back — same as above
                d_opp = opp[d]
                pop_d = reconstruct_fi_at_index(
                    rho_cur, ux_cur, uy_cur, uz_cur,
                    Sxx_cur, Sxy_cur, Sxz_cur, Syy_cur, Syz_cur, Szz_cur,
                    d_opp, coeffs, lattice_w,
                )
            else:
                # Read neighbour's 10 stored moments
                nb_rho = f_mom[nb_ind + 0 * total_num]
                nb_ux  = f_mom[nb_ind + 1 * total_num]
                nb_uy  = f_mom[nb_ind + 2 * total_num]
                nb_uz  = f_mom[nb_ind + 3 * total_num]
                nb_Sxx = f_mom[nb_ind + 4 * total_num]
                nb_Sxy = f_mom[nb_ind + 5 * total_num]
                nb_Sxz = f_mom[nb_ind + 6 * total_num]
                nb_Syy = f_mom[nb_ind + 7 * total_num]
                nb_Syz = f_mom[nb_ind + 8 * total_num]
                nb_Szz = f_mom[nb_ind + 9 * total_num]
                # Reconstruct full distribution from neighbour moments
                # [REF] stream_collide_bvh:751-776
                pop_d = reconstruct_fi_at_index(
                    nb_rho, nb_ux, nb_uy, nb_uz,
                    nb_Sxx, nb_Sxy, nb_Sxz, nb_Syy, nb_Syz, nb_Szz,
                    d, coeffs, lattice_w,
                )

        # ---- Outgoing distribution (f_on) for mass flux ----
        # [REF] stream_collide_bvh:779-803 — computed for every direction
        fon_d = reconstruct_fi_at_index(
            rho_cur, ux_cur, uy_cur, uz_cur,
            Sxx_cur, Sxy_cur, Sxz_cur, Syy_cur, Syz_cur, Szz_cur,
            d, coeffs, lattice_w,
        )
        if d > 0 and (flagsn & TYPE_F) != 0:
            massn += pop_d - fon_d

        # Accumulate post-stream moments from pop_d
        # [REF] stream_collide_bvh:939-972
        rho_star += pop_d
        ux_star  += pop_d * float(c_x[d])
        uy_star  += pop_d * float(c_y[d])
        uz_star  += pop_d * float(c_z[d])
        pixx_star += pop_d * float(c_x[d] * c_x[d])
        pixy_star += pop_d * float(c_x[d] * c_y[d])
        pixz_star += pop_d * float(c_x[d] * c_z[d])
        piyy_star += pop_d * float(c_y[d] * c_y[d])
        piyz_star += pop_d * float(c_y[d] * c_z[d])
        pizz_star += pop_d * float(c_z[d] * c_z[d])

    # ---- Normalise velocity ----
    inv_rho = 1.0 / rho_star
    ux_star = ux_star * inv_rho
    uy_star = uy_star * inv_rho
    uz_star = uz_star * inv_rho

    # ---- Apply body force ----
    # F = rho * g  (force per unit volume in lattice units)
    Fx = rho_star * gx
    Fy = rho_star * gy
    Fz = rho_star * gz

    # ---- Central-moment MRT collision ----
    # [REF] mrUtilFuncGpu3D.h:424-471 (mlGetPIAfterCollision)
    # The collision operates on TRUE second-order moments (not stress).
    # The collision uses pre-shift velocity (ux_star etc.); the Guo
    # forcing half-step F/(2*rho) is applied AFTER collision in the
    # output section, matching [REF] stream_collide_bvh.
    # omega = 1 / tau
    omega = 1.0 / tau

    R = rho_star
    U = ux_star
    V = uy_star
    W = uz_star

    # =========================================================================
    # Central-moment MRT collision [P4] Eq.(18-21); [P1] HOME-LBM
    #
    # 1. Convert TRUE second moments → central moments
    #    κ_ab = π_ab / ρ  −  u_a * u_b
    # 2. Relax central moments toward equilibrium: κ_eq = CS2*δ_ab
    #    κ_ab_post = (1−ω)*κ_ab + ω*CS2*δ_ab
    # 3. Convert back to TRUE moments + Guo force terms
    #    Force coefficient: (2τ−1)/(2τ) = (1 − ω/2) / ω … no, the plan
    #    document gives coeff_f = (2τ−1)/(2τ) * 1/ρ.
    # =========================================================================
    inv_rho = 1.0 / R
    om1 = 1.0 - omega

    # ---- Step 1: TRUE → central moments ----
    pixx_cm = pixx_star * inv_rho - U * U
    pixy_cm = pixy_star * inv_rho - U * V
    pixz_cm = pixz_star * inv_rho - U * W
    piyy_cm = piyy_star * inv_rho - V * V
    piyz_cm = piyz_star * inv_rho - V * W
    pizz_cm = pizz_star * inv_rho - W * W

    # ---- Step 2: MRT relaxation in central-moment space ----
    pixx_cm_post = om1 * pixx_cm + omega * CS2
    piyy_cm_post = om1 * piyy_cm + omega * CS2
    pizz_cm_post = om1 * pizz_cm + omega * CS2
    pixy_cm_post = om1 * pixy_cm
    pixz_cm_post = om1 * pixz_cm
    piyz_cm_post = om1 * piyz_cm

    # ---- Step 3: central → TRUE moments + Guo force [P4] Eq.(18-21) ----
    # coeff_f = (2τ−1) / (2τ) = (1 − ω/2) / ω … correcting for plan doc
    # The plan document, section 3.2: coeff_f = (2.0*tau - 1.0)/(2.0*tau)/rho
    # = (1 − ω/2) / ρ  (since τ = 1/ω, 2τ−1 = 2/ω−1, /2τ = (2−ω)/(2) = 1−ω/2)
    coeff_f = (1.0 - 0.5 * omega) * inv_rho

    pixx_post = R * (pixx_cm_post + U * U) + coeff_f * 2.0 * Fx * U
    piyy_post = R * (piyy_cm_post + V * V) + coeff_f * 2.0 * Fy * V
    pizz_post = R * (pizz_cm_post + W * W) + coeff_f * 2.0 * Fz * W
    pixy_post = R * (pixy_cm_post + U * V) + coeff_f * (Fx * V + Fy * U)
    pixz_post = R * (pixz_cm_post + U * W) + coeff_f * (Fx * W + Fz * U)
    piyz_post = R * (piyz_cm_post + V * W) + coeff_f * (Fy * W + Fz * V)

    # ---- Convert TRUE post-collision moments to stored stress ----
    # Stored stress: S_ab = pi_ab / rho - CS2 * delta_ab
    # [REF] stream_collide_bvh:1046-1055
    #
    # Guo forcing: velocity is shifted by du = F/(2*rho) AFTER collision.
    # The stored stress from collision corresponds to pre-shift velocity.
    # We add du_a*du_b so the stress matches the shifted velocity's equilibrium.
    inv_R = 1.0 / R
    du_x = Fx * inv_R * 0.5
    du_y = Fy * inv_R * 0.5
    du_z = Fz * inv_R * 0.5

    u_out_x = U + du_x
    u_out_y = V + du_y
    u_out_z = W + du_z
    vel_sq = u_out_x * u_out_x + u_out_y * u_out_y + u_out_z * u_out_z
    max_v2 = max_vel * max_vel
    if vel_sq > max_v2:
        scale = max_vel / wp.sqrt(vel_sq)
        u_out_x = u_out_x * scale
        u_out_y = u_out_y * scale
        u_out_z = u_out_z * scale
        du_x = u_out_x - U
        du_y = u_out_y - V
        du_z = u_out_z - W

    f_mom_post[cur_ind + 0 * total_num] = R
    f_mom_post[cur_ind + 1 * total_num] = u_out_x
    f_mom_post[cur_ind + 2 * total_num] = u_out_y
    f_mom_post[cur_ind + 3 * total_num] = u_out_z
    f_mom_post[cur_ind + 4 * total_num] = pixx_post * inv_R - CS2 + du_x * du_x
    f_mom_post[cur_ind + 5 * total_num] = pixy_post * inv_R + du_x * du_y
    f_mom_post[cur_ind + 6 * total_num] = pixz_post * inv_R + du_x * du_z
    f_mom_post[cur_ind + 7 * total_num] = piyy_post * inv_R - CS2 + du_y * du_y
    f_mom_post[cur_ind + 8 * total_num] = piyz_post * inv_R + du_y * du_z
    f_mom_post[cur_ind + 9 * total_num] = pizz_post * inv_R - CS2 + du_z * du_z

    # ---- Scalar fields ----
    # Mass: use mass flux from fhn-fon for TYPE_F [REF], rho_star for others
    if (flagsn & TYPE_F) != 0:
        mass[i, j, k] = massn
    else:
        mass[i, j, k] = R
    phi[i, j, k] = 1.0
    force_x[i, j, k] = Fx
    force_y[i, j, k] = Fy
    force_z[i, j, k] = Fz


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
