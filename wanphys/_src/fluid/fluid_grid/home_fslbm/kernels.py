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

from .constants import (
    CS2, INV_CS2,
    TYPE_F, TYPE_I, TYPE_G, TYPE_S,
    TYPE_IF, TYPE_IG, TYPE_GI,
    TYPE_SU,
    MAX_VELOCITY,
)


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
    massex: wp.array3d(dtype=float),
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

    # Early exit for solid and gas cells
    if (flagsn & (TYPE_S | TYPE_G)) != 0:
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
    # Stage 3 (free-surface): collect massex from neighbours,
    # then compute phi-weighted mass flux for TYPE_I cells.
    massn = float(0.0)
    if (flagsn & (TYPE_F | TYPE_I)) != 0:
        massn = mass[i, j, k]  # start with current cell mass

    # ---- Collect excess mass redistributed by surface_3 ----
    # [REF] stream_collide_bvh:808-819
    if (flagsn & (TYPE_F | TYPE_I)) != 0:
        for dd in range(1, 27):
            nx_m = i - int(c_x[dd])
            ny_m = j - int(c_y[dd])
            nz_m = k - int(c_z[dd])
            # Skip out-of-domain neighbours (periodic / bounce-back handled later)
            if nx_m < 0 or nx_m >= nx or ny_m < 0 or ny_m >= ny or nz_m < 0 or nz_m >= nz:
                continue
            massn += massex[nx_m, ny_m, nz_m]

    # Determine if this is an interface cell (phi-weighted mass flux)
    is_interface = bool((flagsn & TYPE_I) == TYPE_I)

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
        # ---- Mass flux (phi-weighted for interface cells) ----
        # [REF] stream_collide_bvh:821-936
        if d > 0:
            if (flagsn & TYPE_F) != 0:
                massn += pop_d - fon_d
            elif is_interface:
                # Interface mass flux weighted by phi at face
                # phi_on_face: neighbour's phi for even dir, current's for odd
                # (approximation of interface orientation, [REF] L946-948)
                if d % 2 == 0:
                    # Even direction: read neighbour phi
                    if not use_bounce:
                        phi_face = phi[nx_i, ny_i, nz_i]
                    else:
                        phi_face = float(0.5)  # bounce-back face: average
                else:
                    phi_face = phi[i, j, k]
                # Clamp phi_face to [EPS, 1] for stability
                if phi_face < 1.0e-6:
                    phi_face = float(1.0e-6)
                if phi_face > 1.0:
                    phi_face = float(1.0)
                massn += phi_face * (pop_d - fon_d)

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
    # Stage 3: TYPE_I also tracks mass; phi updated by surface_3, not here.
    if (flagsn & TYPE_F) != 0:
        mass[i, j, k] = massn
        phi[i, j, k] = 1.0
    elif (flagsn & TYPE_I) != 0:
        mass[i, j, k] = massn
        # phi kept as-is — will be updated by surface_3
    else:
        mass[i, j, k] = R
        phi[i, j, k] = 1.0
    force_x[i, j, k] = Fx
    force_y[i, j, k] = Fy
    force_z[i, j, k] = Fz


# ============================================================================
# Free-surface helper: VOF fill fraction  [REF] mrUtilFuncGpu3D.h:351-353
# ============================================================================
@wp.func
def _calculate_phi(rho: float, mass: float, surf_flag: int) -> float:
    """Inline VOF fill fraction for use inside surface_3_kernel.

    Args:
        rho: Cell density.
        mass: Actual fluid mass in cell.
        surf_flag: Surface type (TYPE_F / TYPE_I / TYPE_G).

    Returns:
        Phi value in [0, 1].
    """
    if (surf_flag & TYPE_F) != 0:
        return 1.0
    if (surf_flag & TYPE_I) != 0:
        if rho > 0.0:
            inv = mass / rho
            if inv > 1.0:
                return 1.0
            if inv < 0.0:
                return 0.0
            return inv
        return 0.5
    return 0.0


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


# ============================================================================
# Free-surface step 1: mark transitions  [REF] mrLbmSolverGpu3D.cu:444-477
# ============================================================================
@wp.kernel
def surface_1_kernel(
    flag: wp.array(dtype=wp.int32),
    total_num: int, nx: int, ny: int, nz: int,
    c_x: wp.array(dtype=wp.int32),
    c_y: wp.array(dtype=wp.int32),
    c_z: wp.array(dtype=wp.int32),
):
    """Mark gas→interface transitions and prevent interface→gas degradation.

    [REF] surface_1 in mrLbmSolverGpu3D.cu:444-477.

    For each TYPE_IF (interface→fluid) cell:
        - If neighbour is TYPE_IG → change to TYPE_I (prevent disappearance).
        - If neighbour is TYPE_G  → change to TYPE_GI (gas becomes interface).
    """
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    cur_ind = k * ny * nx + j * nx + i
    flagsn = flag[cur_ind]
    flagsn_sus = flagsn & (TYPE_SU | TYPE_S)

    if flagsn_sus == TYPE_IF:
        for d in range(1, 27):
            x1 = i - int(c_x[d])
            y1 = j - int(c_y[d])
            z1 = k - int(c_z[d])
            if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny or z1 < 0 or z1 >= nz:
                continue
            nb_idx = z1 * ny * nx + y1 * nx + x1
            nb_flags = flag[nb_idx]
            nb_su = nb_flags & (TYPE_SU | TYPE_S)
            nb_non_su = nb_flags & ~TYPE_SU

            if nb_su == TYPE_IG:
                # Prevent interface neighbour from becoming gas
                flag[nb_idx] = nb_non_su | TYPE_I
            elif nb_su == TYPE_G:
                # Gas neighbour must become interface
                flag[nb_idx] = nb_non_su | TYPE_GI


# ============================================================================
# Free-surface step 2: init new interface cells  [REF] mrLbmSolverGpu3D.cu:479-601
# ============================================================================
@wp.kernel
def surface_2_kernel(
    f_mom_post: wp.array(dtype=float),
    flag: wp.array(dtype=wp.int32),
    total_num: int, nx: int, ny: int, nz: int,
    c_x: wp.array(dtype=wp.int32),
    c_y: wp.array(dtype=wp.int32),
    c_z: wp.array(dtype=wp.int32),
    w: wp.array(dtype=wp.float32),
):
    """Initialise new interface (TYPE_GI) cells and handle TYPE_IG transitions.

    [REF] surface_2 in mrLbmSolverGpu3D.cu:479-601.

    TYPE_GI cells:
        - Average rho, u from all fluid/interface neighbours.
        - Compute equilibrium distributions feq from average (rho, u).
        - Compute second moments from feq and write to fMomPost.

    TYPE_IG cells:
        - For neighbours that are TYPE_F or TYPE_IF: mark as TYPE_I
          (prevents fluid neighbours from being/becoming fluid).
    """
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    cur_ind = k * ny * nx + j * nx + i
    flagsn = flag[cur_ind]
    flagsn_sus = flagsn & (TYPE_SU | TYPE_S)

    if flagsn_sus == TYPE_GI:
        # ---- Average rho, u from fluid/interface neighbours ----
        rhot = float(0.0)
        uxt = float(0.0)
        uyt = float(0.0)
        uzt = float(0.0)
        counter = float(0.0)

        for d in range(1, 27):
            x1 = i - int(c_x[d])
            y1 = j - int(c_y[d])
            z1 = k - int(c_z[d])
            if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny or z1 < 0 or z1 >= nz:
                continue
            nb_idx = z1 * ny * nx + y1 * nx + x1
            nb_sus = flag[nb_idx] & (TYPE_SU | TYPE_S)
            if nb_sus == TYPE_F or nb_sus == TYPE_I or nb_sus == TYPE_IF:
                counter += 1.0
                rhot += f_mom_post[nb_idx + 0 * total_num]
                uxt += f_mom_post[nb_idx + 1 * total_num]
                uyt += f_mom_post[nb_idx + 2 * total_num]
                uzt += f_mom_post[nb_idx + 3 * total_num]

        rhon = rhot / counter if counter > 0.0 else 1.0
        uxn = uxt / counter if counter > 0.0 else 0.0
        uyn = uyt / counter if counter > 0.0 else 0.0
        uzn = uzt / counter if counter > 0.0 else 0.0

        # ---- Compute equilibrium second moments ----
        # [REF] surface_2:529-548 — feq → second moments
        pixx_eq = float(0.0); pixy_eq = float(0.0); pixz_eq = float(0.0)
        piyy_eq = float(0.0); piyz_eq = float(0.0); pizz_eq = float(0.0)

        for d_i in range(27):
            cu = float(c_x[d_i]) * uxn + float(c_y[d_i]) * uyn + float(c_z[d_i]) * uzn
            u2 = uxn * uxn + uyn * uyn + uzn * uzn
            feq_d = rhon * float(w[d_i]) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
            feq_d += float(w[d_i])  # add back weight (bias trick)
            pixx_eq += feq_d * float(c_x[d_i] * c_x[d_i])
            pixy_eq += feq_d * float(c_x[d_i] * c_y[d_i])
            pixz_eq += feq_d * float(c_x[d_i] * c_z[d_i])
            piyy_eq += feq_d * float(c_y[d_i] * c_y[d_i])
            piyz_eq += feq_d * float(c_y[d_i] * c_z[d_i])
            pizz_eq += feq_d * float(c_z[d_i] * c_z[d_i])

        inv_rhon = 1.0 / rhon
        # Convert to stored stress format: S_ab = pi_ab / rho - CS2 * delta_ab
        cs2 = float(1.0 / 3.0)
        Sxx = pixx_eq * inv_rhon - cs2
        Sxy = pixy_eq * inv_rhon
        Sxz = pixz_eq * inv_rhon
        Syy = piyy_eq * inv_rhon - cs2
        Syz = piyz_eq * inv_rhon
        Szz = pizz_eq * inv_rhon - cs2

        f_mom_post[cur_ind + 0 * total_num] = rhon
        f_mom_post[cur_ind + 1 * total_num] = uxn
        f_mom_post[cur_ind + 2 * total_num] = uyn
        f_mom_post[cur_ind + 3 * total_num] = uzn
        f_mom_post[cur_ind + 4 * total_num] = Sxx
        f_mom_post[cur_ind + 5 * total_num] = Sxy
        f_mom_post[cur_ind + 6 * total_num] = Sxz
        f_mom_post[cur_ind + 7 * total_num] = Syy
        f_mom_post[cur_ind + 8 * total_num] = Syz
        f_mom_post[cur_ind + 9 * total_num] = Szz

    elif flagsn_sus == TYPE_IG:
        # ---- Prevent fluid neighbours from being/becoming fluid ----
        # [REF] surface_2:569-598
        for d in range(1, 27):
            x1 = i - int(c_x[d])
            y1 = j - int(c_y[d])
            z1 = k - int(c_z[d])
            if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny or z1 < 0 or z1 >= nz:
                continue
            nb_idx = z1 * ny * nx + y1 * nx + x1
            nb_flags = flag[nb_idx]
            nb_sus = nb_flags & (TYPE_SU | TYPE_S)
            nb_non_su = nb_flags & ~TYPE_SU

            if nb_sus == TYPE_F or nb_sus == TYPE_IF:
                # Convert fluid/fluid-to-be neighbour to interface
                flag[nb_idx] = nb_non_su | TYPE_I


# ============================================================================
# Free-surface step 3: mass exchange + reflag  [REF] mrLbmSolverGpu3D.cu:604-701
# ============================================================================
@wp.kernel
def surface_3_kernel(
    f_mom_post: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    flag: wp.array(dtype=wp.int32),
    total_num: int, nx: int, ny: int, nz: int,
    c_x: wp.array(dtype=wp.int32),
    c_y: wp.array(dtype=wp.int32),
    c_z: wp.array(dtype=wp.int32),
):
    """Mass redistribution, phi update, and flag conversion.

    [REF] surface_3 in mrLbmSolverGpu3D.cu:604-701.

    For each non-solid cell:
        - TYPE_F:   massex = mass - rho, mass = rho, phi = 1
        - TYPE_I:   massex = excess(mass, rho), mass = clamp(mass, 0, rho),
                    phi = calculate_phi(rho, mass, TYPE_I)
        - TYPE_G:   massex = mass, mass = 0, phi = 0
        - TYPE_IF:  flag → TYPE_F, massex = mass - rho, mass = rho, phi = 1
        - TYPE_IG:  flag → TYPE_G, massex = mass, mass = 0, phi = 0
        - TYPE_GI:  flag → TYPE_I, massex = excess(mass, rho),
                    mass = clamp(mass, 0, rho), phi = calculate_phi(rho, mass, TYPE_I)

        Distribute massex equally to fluid/interface neighbours.
        Detect transitions: mass > rho & no GAS neighbours → TYPE_IF;
                           mass < 0 & no Fluid neighbours → TYPE_IG.
    """
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    cur_ind = k * ny * nx + j * nx + i
    flagsn = flag[cur_ind]
    flagsn_sus = flagsn & (TYPE_SU | TYPE_S)

    if (flagsn_sus & TYPE_S) != 0:
        return

    rhon = f_mom_post[cur_ind + 0 * total_num]
    massn = mass[i, j, k]
    massexn = float(0.0)
    phin = float(0.0)

    # ---- Per-type mass / phi / flag processing ----
    if flagsn_sus == TYPE_F:
        massexn = massn - rhon
        massn = rhon
        phin = 1.0
    elif flagsn_sus == TYPE_I:
        # Interface: excess if mass > rho or mass < 0
        if massn > rhon:
            massexn = massn - rhon
        elif massn < 0.0:
            massexn = massn
        else:
            massexn = 0.0
        # Clamp mass to [0, rho]
        if massn > rhon:
            massn = rhon
        if massn < 0.0:
            massn = 0.0
        phin = _calculate_phi(rhon, massn, TYPE_I)
    elif flagsn_sus == TYPE_G:
        massexn = massn
        massn = 0.0
        phin = 0.0
    elif flagsn_sus == TYPE_IF:
        # Interface → Fluid
        non_su = flagsn & ~TYPE_SU
        flag[cur_ind] = non_su | TYPE_F
        massexn = massn - rhon
        massn = rhon
        phin = 1.0
    elif flagsn_sus == TYPE_IG:
        # Interface → Gas
        non_su = flagsn & ~TYPE_SU
        flag[cur_ind] = non_su | TYPE_G
        massexn = massn
        massn = 0.0
        phin = 0.0
    elif flagsn_sus == TYPE_GI:
        # Gas → Interface
        non_su = flagsn & ~TYPE_SU
        flag[cur_ind] = non_su | TYPE_I
        if massn > rhon:
            massexn = massn - rhon
        elif massn < 0.0:
            massexn = massn
        else:
            massexn = 0.0
        if massn > rhon:
            massn = rhon
        if massn < 0.0:
            massn = 0.0
        phin = _calculate_phi(rhon, massn, TYPE_I)

    # ---- Distribute excess mass to fluid/interface neighbours ----
    # [REF] surface_3:669-684
    counter = int(0)
    for d in range(1, 27):
        x1 = i - int(c_x[d])
        y1 = j - int(c_y[d])
        z1 = k - int(c_z[d])
        if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny or z1 < 0 or z1 >= nz:
            continue
        nb_idx = z1 * ny * nx + y1 * nx + x1
        nb_su = flag[nb_idx] & (TYPE_SU | TYPE_S)
        if nb_su == TYPE_F or nb_su == TYPE_I or nb_su == TYPE_IF or nb_su == TYPE_GI:
            counter = counter + 1

    if counter > 0:
        massn += 0.0  # massex goes to neighbours
        massexn = massexn / float(counter)
    else:
        # Can't distribute — add back to local mass (mass conservation)
        massn += massexn
        massexn = 0.0

    # ---- Write back ----
    mass[i, j, k] = massn
    massex[i, j, k] = massexn
    phi[i, j, k] = phin

    # ---- Detect and mark transitions ----
    # [REF] surface_3 logic after mass redistribution
    # Transition conditions after reflag check the updated flag:
    new_flags = flag[cur_ind]
    new_sus = new_flags & (TYPE_SU | TYPE_S)

    if new_sus == TYPE_F or new_sus == TYPE_I:
        # Check for mass > rho (fill) and mass < 0 (empty)
        if massn > rhon:
            # Cell overfull — convert to fluid if no gas neighbours
            _has_gas = int(0)
            for d in range(1, 27):
                x1 = i - int(c_x[d])
                y1 = j - int(c_y[d])
                z1 = k - int(c_z[d])
                if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny or z1 < 0 or z1 >= nz:
                    continue
                nb_idx = z1 * ny * nx + y1 * nx + x1
                nb_su_v = flag[nb_idx] & (TYPE_SU | TYPE_S)
                if nb_su_v == TYPE_G:
                    _has_gas = int(1)
            if _has_gas == 0:
                non_su = flag[cur_ind] & ~TYPE_SU
                flag[cur_ind] = non_su | TYPE_IF
        if massn < 0.0:
            # Cell empty — convert to gas if no fluid neighbours
            _has_fluid = int(0)
            for d in range(1, 27):
                x1 = i - int(c_x[d])
                y1 = j - int(c_y[d])
                z1 = k - int(c_z[d])
                if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny or z1 < 0 or z1 >= nz:
                    continue
                nb_idx = z1 * ny * nx + y1 * nx + x1
                nb_su_v = flag[nb_idx] & (TYPE_SU | TYPE_S)
                if nb_su_v == TYPE_F or nb_su_v == TYPE_IF:
                    _has_fluid = int(1)
            if _has_fluid == 0:
                non_su = flag[cur_ind] & ~TYPE_SU
                flag[cur_ind] = non_su | TYPE_IG


# ============================================================================
# Separation force skeleton (deferred to stage 6)
# ============================================================================
@wp.kernel
def calculate_disjoint_kernel(
    disjoin_force: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
):
    """Skeleton: zero out separation force (stage 6 deferred).

    [REF] disjoin_force used in stream_collide_bvh for bubble separation.
    Initialised to zero for stages 3-5.
    """
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz:
        disjoin_force[i, j, k] = 0.0
