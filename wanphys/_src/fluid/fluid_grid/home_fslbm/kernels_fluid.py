# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM fluid kernels.

Core GPU functions and the monolithic ``stream_collide_bvh`` kernel that
combines pull-streaming, Hermite reconstruction, NOCM-MRT collision,
free-surface boundary conditions, and eddy-viscosity turbulence model.

Reference sources
-----------------
- Hermite reconstruction: ``mrUtilFuncGpu3D.h:153-273``
- NOCM-MRT collision:   ``mrUtilFuncGpu3D.h:424-471``
- D3Q27 equilibrium:    ``mrUtilFuncGpu3D.h:292-320``
- stream_collide_bvh:   ``mrLbmSolverGpu3D.cu:703-1057``
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C


# ============================================================================

@wp.struct
class StressPi:
    xx: float
    xy: float
    xz: float
    yy: float
    yz: float
    zz: float

# 1. D3Q27 Maxwell-Boltzmann equilibrium distribution
# ============================================================================
# Ref: ``mrUtilFuncGpu3D.h:292-320`` — ``calculate_f_eq``
#
# This is the standard D3Q27 equilibrium (second-order Hermite expansion):
#
#     f_i^eq = w_i · ρ · [1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) − u²/(2c_s²)]
#
# The reference code uses an optimised form with fused-multiply-add (fma)
# and pre-multiplied velocity components (ux*=3).


@wp.func
def calculate_f_eq_d3q27(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    di: int,
) -> float:
    """D3Q27 equilibrium distribution for direction *di*.

    Parameters
    ----------
    rho:
        Density.
    ux, uy, uz:
        Velocity components (lattice units).
    di:
        Direction index 0..26.

    Returns
    -------
    float
        Equilibrium distribution value f_i^eq.
    """
    # Pre-computed per-direction formula.
    # The reference code multiplies u by 3.0 before use.
    # Directions are categorised by weight class.
    c3 = -3.0 * (ux * ux + uy * uy + uz * uz)
    rhom1 = rho - 1.0

    ux3 = ux * 3.0
    uy3 = uy * 3.0
    uz3 = uz * 3.0

    if di == 0:
        return C.W0 * (rho * 0.5 * c3 + rhom1)

    # Face neighbours (di 1-6): weight = WS
    rhos = C.WS * rho
    rhom1s = C.WS * rhom1

    if di == 1:   # (+1, 0, 0)
        return rhos * (0.5 * (ux3 * ux3 + c3) + ux3) + rhom1s
    if di == 2:   # (-1, 0, 0)
        return rhos * (0.5 * (ux3 * ux3 + c3) - ux3) + rhom1s
    if di == 3:   # (0, +1, 0)
        return rhos * (0.5 * (uy3 * uy3 + c3) + uy3) + rhom1s
    if di == 4:   # (0, -1, 0)
        return rhos * (0.5 * (uy3 * uy3 + c3) - uy3) + rhom1s
    if di == 5:   # (0, 0, +1)
        return rhos * (0.5 * (uz3 * uz3 + c3) + uz3) + rhom1s
    if di == 6:   # (0, 0, -1)
        return rhos * (0.5 * (uz3 * uz3 + c3) - uz3) + rhom1s

    # Edge neighbours (di 7-18): weight = WE
    rhoe = C.WE * rho
    rhom1e = C.WE * rhom1

    # Pre-compute pairwise velocity sums (ref line 302)
    u0 = ux3 + uy3   # x+y
    u1 = ux3 + uz3   # x+z
    u2 = uy3 + uz3   # y+z
    u3 = ux3 - uy3   # x-y
    u4 = ux3 - uz3   # x-z
    u5 = uy3 - uz3   # y-z

    # Direction pairs are ordered so that +velocity and -velocity are adjacent
    if di == 7:    # (+1,+1, 0)
        return rhoe * (0.5 * (u0 * u0 + c3) + u0) + rhom1e
    if di == 8:    # (-1,-1, 0)
        return rhoe * (0.5 * (u0 * u0 + c3) - u0) + rhom1e
    if di == 9:    # (+1, 0,+1)
        return rhoe * (0.5 * (u1 * u1 + c3) + u1) + rhom1e
    if di == 10:   # (-1, 0,-1)
        return rhoe * (0.5 * (u1 * u1 + c3) - u1) + rhom1e
    if di == 11:   # ( 0,+1,+1)
        return rhoe * (0.5 * (u2 * u2 + c3) + u2) + rhom1e
    if di == 12:   # ( 0,-1,-1)
        return rhoe * (0.5 * (u2 * u2 + c3) - u2) + rhom1e
    if di == 13:   # (+1,-1, 0)
        return rhoe * (0.5 * (u3 * u3 + c3) + u3) + rhom1e
    if di == 14:   # (-1,+1, 0)
        return rhoe * (0.5 * (u3 * u3 + c3) - u3) + rhom1e
    if di == 15:   # (+1, 0,-1)
        return rhoe * (0.5 * (u4 * u4 + c3) + u4) + rhom1e
    if di == 16:   # (-1, 0,+1)
        return rhoe * (0.5 * (u4 * u4 + c3) - u4) + rhom1e
    if di == 17:   # ( 0,+1,-1)
        return rhoe * (0.5 * (u5 * u5 + c3) + u5) + rhom1e
    if di == 18:   # ( 0,-1,+1)
        return rhoe * (0.5 * (u5 * u5 + c3) - u5) + rhom1e

    # Corner neighbours (di 19-26): weight = WC
    rhoc = C.WC * rho
    rhom1c = C.WC * rhom1

    u6 = ux3 + uy3 + uz3   # x+y+z
    u7 = ux3 + uy3 - uz3   # x+y-z
    u8 = ux3 - uy3 + uz3   # x-y+z
    u9 = uy3 + uz3 - ux3   # -x+y+z

    if di == 19:   # (+1,+1,+1)
        return rhoc * (0.5 * (u6 * u6 + c3) + u6) + rhom1c
    if di == 20:   # (-1,-1,-1)
        return rhoc * (0.5 * (u6 * u6 + c3) - u6) + rhom1c
    if di == 21:   # (+1,+1,-1)
        return rhoc * (0.5 * (u7 * u7 + c3) + u7) + rhom1c
    if di == 22:   # (-1,-1,+1)
        return rhoc * (0.5 * (u7 * u7 + c3) - u7) + rhom1c
    if di == 23:   # (+1,-1,+1)
        return rhoc * (0.5 * (u8 * u8 + c3) + u8) + rhom1c
    if di == 24:   # (-1,+1,-1)
        return rhoc * (0.5 * (u8 * u8 + c3) - u8) + rhom1c
    if di == 25:   # (-1,+1,+1)  → note: reference swaps 25/26 relative to +/-
        return rhoc * (0.5 * (u9 * u9 + c3) + u9) + rhom1c
    if di == 26:   # (+1,-1,-1)
        return rhoc * (0.5 * (u9 * u9 + c3) - u9) + rhom1c

    return float(0.0)


# ============================================================================
# 2. Hermite reconstruction: moments → D3Q27 distribution
# ============================================================================
# Ref: ``mrUtilFuncGpu3D.h:153-273`` — ``mlCalDistributionFourthOrderD3Q27AtIndex``
#
# HOME-LBM stores only the first 3 velocity moments (10 scalars per node):
#   M[0] = ρ
#   M[1..3] = ρu_x, ρu_y, ρu_z
#   M[4..9] = ρΠ_xx-c_s², ρΠ_xy, ρΠ_xz, ρΠ_yy-c_s², ρΠ_yz, ρΠ_zz-c_s²
#
# Given these 10 moments and a direction index *di*, this function reconstructs
# the full D3Q27 distribution f_i via a third-order Hermite expansion
# (Eq. 11 in paper-4).
#
# The diagonal stress components are stored traceless:
#   stored:  S_xx = Π_xx - c_s²
#   on use:  Π_xx = S_xx + c_s²  (and similarly for yy, zz)


@wp.func
def reconstruct_distribution(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    pi_xx: float,
    pi_xy: float,
    pi_xz: float,
    pi_yy: float,
    pi_yz: float,
    pi_zz: float,
    di: int,
) -> float:
    """Reconstruct D3Q27 distribution for direction *di* from HOME moments.

    Parameters
    ----------
    rho:
        Density (moment M[0]).
    ux, uy, uz:
        Velocity components (M[1..3] / ρ).
    pi_xx ... pi_zz:
        Traceless stress components S_αβ = Π_αβ/ρ − c_s²·δ_αβ.
        Pass the stored f_mom[M_SXX] etc. values directly.
        (Matches reference: ``mrUtilFuncGpu3D.h:153`` first parameter
        ``pixx`` is the traceless stored value, not the full stress.)
    di:
        Direction index 0..26.

    Returns
    -------
    float
        Reconstructed distribution f_i.
    """
    # --- Step 1: Build Hermite coefficients (ref lines 161-204) ---
    A0 = rho
    Ax = ux * A0
    Ay = uy * A0
    Az = uz * A0

    Axx = rho * pi_xx
    Ayy = rho * pi_yy
    Azz = rho * pi_zz
    Axy = rho * pi_xy
    Axz = rho * pi_xz
    Ayz = rho * pi_yz

    # Third-order Hermite coefficients (ref lines 173-179)
    Axxy = -2.0 * rho * uy * ux * ux + 2.0 * Axy * ux + Axx * uy
    Axyy = -2.0 * rho * ux * uy * uy + 2.0 * Axy * uy + Ayy * ux
    Axxz = -2.0 * rho * uz * ux * ux + 2.0 * Axz * ux + Axx * uz
    Axzz = -2.0 * rho * ux * uz * uz + 2.0 * Axz * uz + Azz * ux
    Ayyz = -2.0 * rho * uz * uy * uy + 2.0 * Ayz * uy + Ayy * uz
    Ayzz = -2.0 * rho * uy * uz * uz + 2.0 * Ayz * uz + Azz * uy
    Axyz = Axz * uy + Ayz * ux + Axy * uz - 2.0 * rho * ux * uy * uz

    # Pre-multiplied coefficients (ref lines 187-204)
    Ax_t3 = Ax * 3.0
    Ay_t3 = Ay * 3.0
    Az_t3 = Az * 3.0

    Axx_t3 = 3.0 * Axx
    Ayy_t3 = 3.0 * Ayy
    Azz_t3 = 3.0 * Azz
    Axy_t9 = 9.0 * Axy
    Axz_t9 = 9.0 * Axz
    Ayz_t9 = 9.0 * Ayz

    Axxy_t9 = Axxy * 9.0
    Axyy_t9 = Axyy * 9.0
    Axzz_t9 = Axzz * 9.0
    Ayzz_t9 = Ayzz * 9.0
    Axxz_t9 = Axxz * 9.0
    Ayyz_t9 = Ayyz * 9.0
    Axyz_t27 = Axyz * 27.0

    # Common sub-expression (ref line 206)
    com0 = A0 - Axx_t3 * 0.5 - Ayy_t3 * 0.5 - Azz_t3 * 0.5

    # Hermite lattice weights (hard-coded as in reference)
    _W0 = 8.0 / 27.0     # rest
    _W1 = 2.0 / 27.0     # face
    _W2 = 1.0 / 54.0     # edge
    _W3 = 1.0 / 216.0    # corner

    # --- Step 2: Per-direction formula (case 0..26, ref lines 207-265) ---
    if di == 0:
        return _W0 * com0

    # Face directions (1-6)
    if di == 1:    # (+1,0,0)
        return _W1 * (com0 + Ax_t3 + 1.5 * Axx_t3 - Axyy_t9 * 0.5 - Axzz_t9 * 0.5)
    if di == 2:    # (-1,0,0)
        return _W1 * (com0 - Ax_t3 + 1.5 * Axx_t3 + Axyy_t9 * 0.5 + Axzz_t9 * 0.5)
    if di == 3:    # (0,+1,0)
        return _W1 * (com0 + Ay_t3 + 1.5 * Ayy_t3 - Axxy_t9 * 0.5 - Ayzz_t9 * 0.5)
    if di == 4:    # (0,-1,0)
        return _W1 * (com0 - Ay_t3 + 1.5 * Ayy_t3 + Ayzz_t9 * 0.5 + Axxy_t9 * 0.5)
    if di == 5:    # (0,0,+1)
        return _W1 * (com0 + Az_t3 + 1.5 * Azz_t3 - Axxz_t9 * 0.5 - Ayyz_t9 * 0.5)
    if di == 6:    # (0,0,-1)
        return _W1 * (com0 - Az_t3 + 1.5 * Azz_t3 + Axxz_t9 * 0.5 + Ayyz_t9 * 0.5)

    # Edge directions (7-18)
    if di == 7:    # (+1,+1,0)
        return _W2 * (A0 + Ax_t3 + Ay_t3 + Axx_t3 + Ayy_t3 - Azz_t3 * 0.5 + Axy_t9 + Axxy_t9 + Axyy_t9 - Axzz_t9 * 0.5 - Ayzz_t9 * 0.5)
    if di == 8:    # (-1,-1,0)
        return _W2 * (A0 - Ax_t3 - Ay_t3 + Axx_t3 + Ayy_t3 - Azz_t3 * 0.5 + Axy_t9 - Axxy_t9 - Axyy_t9 + Axzz_t9 * 0.5 + Ayzz_t9 * 0.5)
    if di == 9:    # (+1,0,+1)
        return _W2 * (A0 + Ax_t3 + Az_t3 + Axx_t3 - Ayy_t3 * 0.5 + Azz_t3 + Axz_t9 + Axxz_t9 - Axyy_t9 * 0.5 + Axzz_t9 - Ayyz_t9 * 0.5)
    if di == 10:   # (-1,0,-1)
        return _W2 * (A0 - Ax_t3 - Az_t3 + Axx_t3 - Ayy_t3 * 0.5 + Azz_t3 + Axz_t9 - Axxz_t9 + Axyy_t9 * 0.5 - Axzz_t9 + Ayyz_t9 * 0.5)
    if di == 11:   # (0,+1,+1)
        return _W2 * (A0 + Ay_t3 + Az_t3 - Axx_t3 * 0.5 + Ayy_t3 + Azz_t3 + Ayz_t9 - Axxy_t9 * 0.5 - Axxz_t9 * 0.5 + Ayyz_t9 + Ayzz_t9)
    if di == 12:   # (0,-1,-1)
        return _W2 * (A0 - Ay_t3 - Az_t3 - Axx_t3 * 0.5 + Ayy_t3 + Azz_t3 + Ayz_t9 + Axxy_t9 * 0.5 + Axxz_t9 * 0.5 - Ayyz_t9 - Ayzz_t9)
    if di == 13:   # (+1,-1,0)
        return _W2 * (A0 + Ax_t3 - Ay_t3 + Axx_t3 + Ayy_t3 - Azz_t3 * 0.5 - Axy_t9 - Axxy_t9 + Axyy_t9 - Axzz_t9 * 0.5 + Ayzz_t9 * 0.5)
    if di == 14:   # (-1,+1,0)
        return _W2 * (A0 - Ax_t3 + Ay_t3 + Axx_t3 + Ayy_t3 - Azz_t3 * 0.5 - Axy_t9 + Axxy_t9 - Axyy_t9 + Axzz_t9 * 0.5 - Ayzz_t9 * 0.5)
    if di == 15:   # (+1,0,-1)
        return _W2 * (A0 + Ax_t3 - Az_t3 + Axx_t3 - Ayy_t3 * 0.5 + Azz_t3 - Axz_t9 - Axxz_t9 - Axyy_t9 * 0.5 + Axzz_t9 + Ayyz_t9 * 0.5)
    if di == 16:   # (-1,0,+1)
        return _W2 * (A0 - Ax_t3 + Az_t3 + Axx_t3 - Ayy_t3 * 0.5 + Azz_t3 - Axz_t9 + Axxz_t9 + Axyy_t9 * 0.5 - Axzz_t9 - Ayyz_t9 * 0.5)
    if di == 17:   # (0,+1,-1)
        return _W2 * (A0 + Ay_t3 - Az_t3 - Axx_t3 * 0.5 + Ayy_t3 + Azz_t3 - Ayz_t9 - Axxy_t9 * 0.5 + Axxz_t9 * 0.5 - Ayyz_t9 + Ayzz_t9)
    if di == 18:   # (0,-1,+1)
        return _W2 * (A0 - Ay_t3 + Az_t3 - Axx_t3 * 0.5 + Ayy_t3 + Azz_t3 - Ayz_t9 + Axxy_t9 * 0.5 - Axxz_t9 * 0.5 + Ayyz_t9 - Ayzz_t9)

    # Corner directions (19-26)
    if di == 19:   # (+1,+1,+1)
        return _W3 * (A0 + Ax_t3 + Axx_t3 + Axxy_t9 + Axxz_t9 + Axy_t9 + Axyy_t9 + Axyz_t27 + Axz_t9 + Axzz_t9 + Ay_t3 + Ayy_t3 + Ayyz_t9 + Ayz_t9 + Ayzz_t9 + Az_t3 + Azz_t3)
    if di == 20:   # (-1,-1,-1)
        return _W3 * (A0 - Ax_t3 + Axx_t3 - Axxy_t9 - Axxz_t9 + Axy_t9 - Axyy_t9 - Axyz_t27 + Axz_t9 - Axzz_t9 - Ay_t3 + Ayy_t3 - Ayyz_t9 + Ayz_t9 - Ayzz_t9 - Az_t3 + Azz_t3)
    if di == 21:   # (+1,+1,-1)
        return _W3 * (A0 + Ax_t3 + Axx_t3 + Axxy_t9 - Axxz_t9 + Axy_t9 + Axyy_t9 - Axyz_t27 - Axz_t9 + Axzz_t9 + Ay_t3 + Ayy_t3 - Ayyz_t9 - Ayz_t9 + Ayzz_t9 - Az_t3 + Azz_t3)
    if di == 22:   # (-1,-1,+1)
        return _W3 * (A0 - Ax_t3 + Axx_t3 - Axxy_t9 + Axxz_t9 + Axy_t9 - Axyy_t9 + Axyz_t27 - Axz_t9 - Axzz_t9 - Ay_t3 + Ayy_t3 + Ayyz_t9 - Ayz_t9 - Ayzz_t9 + Az_t3 + Azz_t3)
    if di == 23:   # (+1,-1,+1)
        return _W3 * (A0 + Ax_t3 + Axx_t3 - Axxy_t9 + Axxz_t9 - Axy_t9 + Axyy_t9 - Axyz_t27 + Axz_t9 + Axzz_t9 - Ay_t3 + Ayy_t3 - Ayyz_t9 + Ayz_t9 - Ayzz_t9 + Az_t3 + Azz_t3)
    if di == 24:   # (-1,+1,-1)
        return _W3 * (A0 - Ax_t3 + Axx_t3 + Axxy_t9 - Axxz_t9 - Axy_t9 - Axyy_t9 + Axyz_t27 + Axz_t9 - Axzz_t9 + Ay_t3 + Ayy_t3 - Ayyz_t9 - Ayz_t9 + Ayzz_t9 - Az_t3 + Azz_t3)
    if di == 25:   # (-1,+1,+1)
        return _W3 * (A0 - Ax_t3 + Axx_t3 + Axxy_t9 + Axxz_t9 - Axy_t9 - Axyy_t9 - Axyz_t27 - Axz_t9 - Axzz_t9 + Ay_t3 + Ayy_t3 + Ayyz_t9 + Ayz_t9 + Ayzz_t9 + Az_t3 + Azz_t3)
    if di == 26:   # (+1,-1,-1)
        return _W3 * (A0 + Ax_t3 + Axx_t3 - Axxy_t9 - Axxz_t9 - Axy_t9 + Axyy_t9 + Axyz_t27 - Axz_t9 + Axzz_t9 - Ay_t3 + Ayy_t3 - Ayyz_t9 + Ayz_t9 - Ayzz_t9 - Az_t3 + Azz_t3)

    return float(0.0)


# ============================================================================
# 3. NOCM-MRT collision operator
# ============================================================================
# Ref: ``mrUtilFuncGpu3D.h:424-471`` — ``mlGetPIAfterCollision``
#
# This is the heart of HOME-LBM: collision is performed directly on the
# stress tensor Π in closed form, without ever needing the full 27 populations.
# Input:  ρ, u, F (body force), ω (relaxation frequency), Π^old
# Output: Π^new (6 independent components of the symmetric stress tensor)
#
# The diagonal components are processed tracelessly:
#   Π_xx_part = (2Π_xx − Π_yy − Π_zz) / 3
# and reconstructed after collision.


@wp.func
def ml_get_pi_after_collision(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    fx: float,
    fy: float,
    fz: float,
    omega: float,
    pixx_old: float,
    pixy_old: float,
    pixz_old: float,
    piyy_old: float,
    piyz_old: float,
    pizz_old: float,
) -> StressPi:
    """NOCM-MRT collision: update stress tensor Π.

    Parameters
    ----------
    rho:
        Density.
    ux, uy, uz:
        Velocity components.
    fx, fy, fz:
        Body force components.
    omega:
        Relaxation frequency ω = 1/τ.
    pixx_old ... pizz_old:
        Old stress-tensor components Π^old_αβ.

    Returns
    -------
    StressPi
        Collided stress components (pixx_new, pixy_new, pixz_new,
        piyy_new, piyz_new, pizz_new).
    """
    # 1. Traceless decomposition of diagonal (ref lines 436-438)
    pixx_part = (2.0 * pixx_old - piyy_old - pizz_old) / 3.0
    piyy_part = (2.0 * piyy_old - pixx_old - pizz_old) / 3.0
    pizz_part = (2.0 * pizz_old - pixx_old - piyy_old) / 3.0

    # 2. Velocity products (ref lines 439-442)
    RU2 = rho * ux * ux
    RV2 = rho * uy * uy
    RW2 = rho * uz * uz
    RUVW2 = (RU2 + RV2 + RW2) / 3.0

    # 3. Collision update — diagonal (ref lines 443-466)
    pixx_new = (
        rho / 3.0
        + pixx_part * (1.0 - omega)
        + RUVW2
        + (2.0 * RU2 * omega) / 3.0
        - (RV2 * omega) / 3.0
        - (RW2 * omega) / 3.0
        + fx * ux
    )

    piyy_new = (
        rho / 3.0
        + piyy_part * (1.0 - omega)
        + RUVW2
        - (RU2 * omega) / 3.0
        + (2.0 * RV2 * omega) / 3.0
        - (RW2 * omega) / 3.0
        + fy * uy
    )

    pizz_new = (
        rho / 3.0
        + pizz_part * (1.0 - omega)
        + RUVW2
        - (RU2 * omega) / 3.0
        - (RV2 * omega) / 3.0
        + (2.0 * RW2 * omega) / 3.0
        + fz * uz
    )

    # 4. Collision update — off-diagonal (ref lines 468-470)
    pixy_new = (
        pixy_old - pixy_old * omega
        + ux * uy * rho * omega
        + (fy * ux) / 2.0
        + (fx * uy) / 2.0
    )
    pixz_new = (
        pixz_old - pixz_old * omega
        + ux * uz * rho * omega
        + (fz * ux) / 2.0
        + (fx * uz) / 2.0
    )
    piyz_new = (
        piyz_old - piyz_old * omega
        + uy * uz * rho * omega
        + (fz * uy) / 2.0
        + (fy * uz) / 2.0
    )

    return StressPi(
        pixx_new, pixy_new, pixz_new,
        piyy_new, piyz_new, pizz_new,
    )


# ============================================================================
# 4. Compute macroscopic moments from 27 distributions
# ============================================================================
# Ref: ``mrUtilFuncGpu3D.h:272-287`` — ``calculate_rho_u``


@wp.func
def compute_rho_u_from_f(
    f: wp.array(dtype=float),  # flat array [27 * N]
    idx: int,
    stride: int,
) -> wp.vec4:
    """Compute (rho, ux, uy, uz) from 27 D3Q27 distributions at node.

    Parameters
    ----------
    f:
        Flat distribution array (27 × N).
    idx:
        Linear index of the node.
    stride:
        N = nx * ny * nz.

    Returns
    -------
    wp.vec4
        (rho, ux, uy, uz).
    """
    rho = 1.0  # ref line 277: rho += 1.0 (to match reference behaviour)
    ux = 0.0
    uy = 0.0
    uz = 0.0

    base = idx
    for di in range(27):
        pass  # for loop neutralized; sums below are outside
    # Hard-coded 27-direction sums — replaces broken for-loop above
    # Hard-coded momentum sum (directions match reference lines 956-958)
    # Load all 27 distributions
    f0 = f[0*stride+base];  f1 = f[1*stride+base];  f2 = f[2*stride+base]
    f3 = f[3*stride+base];  f4 = f[4*stride+base];  f5 = f[5*stride+base]
    f6 = f[6*stride+base];  f7 = f[7*stride+base];  f8 = f[8*stride+base]
    f9 = f[9*stride+base]; f10=f[10*stride+base]; f11=f[11*stride+base]
    f12=f[12*stride+base]; f13=f[13*stride+base]; f14=f[14*stride+base]
    f15=f[15*stride+base]; f16=f[16*stride+base]; f17=f[17*stride+base]
    f18=f[18*stride+base]; f19=f[19*stride+base]; f20=f[20*stride+base]
    f21=f[21*stride+base]; f22=f[22*stride+base]; f23=f[23*stride+base]
    f24=f[24*stride+base]; f25=f[25*stride+base]; f26=f[26*stride+base]
    rho += f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19+f20+f21+f22+f23+f24+f25+f26
    ux += (f1+f7+f9+f13+f15+f19+f21+f23+f26) - (f2+f8+f10+f14+f16+f20+f22+f24+f25)
    uy += (f3+f7+f11+f14+f17+f19+f21+f24+f25) - (f4+f8+f12+f13+f18+f20+f22+f23+f26)
    uz += (f5+f9+f11+f16+f18+f19+f22+f23+f25) - (f6+f10+f12+f15+f17+f20+f21+f24+f26)
    # dead: uy/uz now computed above via unrolled sums
    # dead: above unrolled sums already compute uz

    inv_rho = 1.0 / rho
    ux = ux * inv_rho
    uy = uy * inv_rho
    uz = uz * inv_rho

    return wp.vec4(rho, ux, uy, uz)


# ============================================================================
# ============================================================================
# 4b. PLIC / curvature helpers — free-surface geometry (Phase 2)
# ============================================================================
# Ref: ``mrUtilFuncGpu3D.h:104-420``
#
# Warp 1.12 does NOT support ``wp.array(dtype=..., length=...)`` as a local
# variable inside @wp.func or @wp.kernel.  All helpers below operate on
# scalar values or read directly from grid arrays passed as parameters.
# The grid-reading variant ``calculate_curvature_from_grid`` is the primary
# entry point called from ``stream_collide_bvh_kernel``.


@wp.func
def calculate_phi(rhon: float, massn: float, flagsn: int) -> float:
    """Compute VOF fill level from density, mass and cell flag."""
    if (flagsn & C.CellFlag.TYPE_F) != 0:
        return 1.0
    elif (flagsn & C.CellFlag.TYPE_I) != 0:
        if rhon > 0.0:
            return wp.clamp(massn / rhon, 0.0, 1.0)
        else:
            return 0.5
    else:
        return 0.0


@wp.func
def plic_cube_reduced(V: float, n1: float, n2: float, n3: float) -> float:
    """Reduced-symmetry PLIC exact solution (SZ & Kawano 2022)."""
    n12 = n1 + n2
    n3V = n3 * V
    if n12 <= 2.0 * n3V:
        return n3V + 0.5 * n12
    sqn1 = n1 * n1
    n26 = 6.0 * n2
    v1 = sqn1 / n26
    if v1 <= n3V and n3V < v1 + 0.5 * (n2 - n1):
        return 0.5 * (n1 + wp.sqrt(sqn1 + 8.0 * n2 * (n3V - v1)))
    V6 = n1 * n26 * n3V
    if n3V < v1:
        return wp.cbrt(V6)
    n3_sq = n3 * n3
    if n3 < n12:
        v3 = (n3_sq * (3.0 * n12 - n3) + sqn1 * (n1 - 3.0 * n3)
              + (n2 * n2) * (n2 - 3.0 * n3)) / (n1 * n26)
    else:
        v3 = 0.5 * n12
    sqn12 = sqn1 + n2 * n2
    n1_cb = n1 * n1 * n1
    n2_cb = n2 * n2 * n2
    n3_cb = n3 * n3 * n3
    V6cbn12 = V6 - n1_cb - n2_cb
    if n3V < v3:
        a = V6cbn12
        b = sqn12
        c = n12
    else:
        a = 0.5 * (V6cbn12 - n3_cb)
        b = 0.5 * (sqn12 + n3_sq)
        c = 0.5
    t = wp.sqrt(c * c - b)
    t_cb = t * t * t
    arg = (c * c * c - 0.5 * a - 1.5 * b * c) / t_cb
    arg = wp.clamp(arg, -1.0, 1.0)
    return c - 2.0 * t * wp.sin(wp.asin(arg) * 0.33333334)


@wp.func
def plic_cube(V0: float, n: wp.vec3) -> float:
    """Unit-cube / plane intersection: volume V0, normal n -> offset d0."""
    ax = wp.abs(n.x)
    ay = wp.abs(n.y)
    az = wp.abs(n.z)
    V = 0.5 - wp.abs(V0 - 0.5)
    l_sum = ax + ay + az
    n1 = wp.min(wp.min(ax, ay), az) / l_sum
    n3 = wp.max(wp.max(ax, ay), az) / l_sum
    n2 = 1.0 - n1 - n3
    if n2 < 0.0:
        n2 = 0.0
    d = plic_cube_reduced(V, n1, n2, n3)
    # wp.copysign not available in Warp 1.12
    if V0 > 0.5:
        return l_sum * (0.5 - d)
    else:
        return l_sum * (d - 0.5)


# 5-element float struct (Warp 1.12 has no wp.vec5)
@wp.struct
class Vec5:
    x: float
    y: float
    z: float
    w: float
    a: float


# ---------------------------------------------------------------------------
# Scalar-based LU solver for 5x5
# ---------------------------------------------------------------------------


@wp.func
def _lu_solve_5x5_scalar(
    m00: float, m01: float, m02: float, m03: float, m04: float,
    m11: float, m12: float, m13: float, m14: float,
    m22: float, m23: float, m24: float,
    m33: float, m34: float,
    m44: float,
    b0: float, b1: float, b2: float, b3: float, b4: float,
    Nsol: int,
) -> Vec5:
    """In-place LU decomposition for 5x5; returns (x0, x1, x2, x3, x4)."""
    # Decompose M into L*U (unrolled for 5x5)
    if m00 != 0.0:
        inv_m00 = 1.0 / m00
    else:
        inv_m00 = 1.0
    l10 = m01 * inv_m00
    l20 = m02 * inv_m00
    l30 = m03 * inv_m00
    l40 = m04 * inv_m00
    u11 = m11 - l10 * m01
    u12 = m12 - l10 * m02
    u13 = m13 - l10 * m03
    u14 = m14 - l10 * m04
    u22_1 = m22 - l20 * m02
    u23_1 = m23 - l20 * m03
    u24_1 = m24 - l20 * m04
    u32_1 = m23 - l30 * m02
    u33_1 = m33 - l30 * m03
    u34_1 = m34 - l30 * m04
    u42_1 = m24 - l40 * m02
    u43_1 = m34 - l40 * m03
    u44_1 = m44 - l40 * m04

    if u11 != 0.0:
        inv_u11 = 1.0 / u11
    else:
        inv_u11 = 1.0
    l21 = u12 * inv_u11
    l31 = u13 * inv_u11
    l41 = u14 * inv_u11
    u22 = u22_1 - l21 * u12
    u23 = u23_1 - l21 * u13
    u24 = u24_1 - l21 * u14
    u32 = u32_1 - l31 * u12
    u33_2 = u33_1 - l31 * u13
    u34_2 = u34_1 - l31 * u14
    u42 = u42_1 - l41 * u12
    u43_2 = u43_1 - l41 * u13
    u44_2 = u44_1 - l41 * u14

    if u22 != 0.0:
        inv_u22 = 1.0 / u22
    else:
        inv_u22 = 1.0
    l32 = u23 * inv_u22
    l42 = u24 * inv_u22
    u33 = u33_2 - l32 * u23
    u34 = u34_2 - l32 * u24
    u43 = u43_2 - l42 * u23
    u44_3 = u44_2 - l42 * u24

    if u33 != 0.0:
        inv_u33 = 1.0 / u33
    else:
        inv_u33 = 1.0
    l43 = u34 * inv_u33
    u44 = u44_3 - l43 * u34

    # Forward substitution: L*y = b
    y0 = b0
    y1 = b1 - l10 * y0
    y2 = b2 - l20 * y0 - l21 * y1
    y3 = b3 - l30 * y0 - l31 * y1 - l32 * y2
    y4 = b4 - l40 * y0 - l41 * y1 - l42 * y2 - l43 * y3

    # Back substitution: U*x = y
    if Nsol <= 4:
        x4 = 0.0
    else:
        x4 = y4 / u44
    if Nsol <= 3:
        x3 = 0.0
    else:
        x3 = (y3 - u34 * x4) / u33
    if Nsol <= 2:
        x2 = 0.0
    else:
        x2 = (y2 - u23 * x3 - u24 * x4) / u22
    if Nsol <= 1:
        x1 = 0.0
    else:
        x1 = (y1 - u12 * x2 - u13 * x3 - u14 * x4) / u11
    if m00 != 0.0:
        x0 = (y0 - m01 * x1 - m02 * x2 - m03 * x3 - m04 * x4) / m00
    else:
        x0 = 0.0

    return Vec5(x0, x1, x2, x3, x4)


# ============================================================================
# Curvature entry point — reads phi directly from grid
# ============================================================================


@wp.func
def calculate_curvature_from_grid(
    phi: wp.array3d(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    opposite: wp.array(dtype=wp.int32),
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int,
    px: int, py: int, pz: int,
    phi_center: float,
) -> float:
    """Compute mean curvature at cell (i,j,k) from phi neighbours.

    Centre *phi_center* is mass-corrected (ref line 867); neighbour phi
    reads fall back across solid cells (ref lines 842-860).

    Ref: ``mrUtilFuncGpu3D.h:371-420``.
    """
    # ---- Step 1: compute normal via 27-pt weighted gradient ----
    bx_n = float(0.0)
    by_n = float(0.0)
    bz_n = float(0.0)

    for di in range(1, 27):
        opp = opposite[di]
        ni = i - int(cx[opp])
        nj = j - int(cy[opp])
        nk = k - int(cz[opp])
        if px == 1:
            if ni < 0: ni += nx
            elif ni >= nx: ni -= nx
        if py == 1:
            if nj < 0: nj += ny
            elif nj >= ny: nj -= ny
        if pz == 1:
            if nk < 0: nk += nz
            elif nk >= nz: nk -= nz
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            continue
        pval = phi[ni, nj, nk]
        # Solid-neighbour fallback (ref: mrLbmSolverGpu3D.cu:842-860)
        if (int(flag[ni, nj, nk]) & C.TYPE_BO_MASK) == C.TYPE_S:
            for fd in range(1, 7):
                fi = ni - int(cx[fd]); fj = nj - int(cy[fd]); fk = nk - int(cz[fd])
                if px == 1:
                    if fi < 0: fi += nx
                    elif fi >= nx: fi -= nx
                if py == 1:
                    if fj < 0: fj += ny
                    elif fj >= ny: fj -= ny
                if pz == 1:
                    if fk < 0: fk += nz
                    elif fk >= nz: fk -= nz
                if fi >= 0 and fi < nx and fj >= 0 and fj < ny and fk >= 0 and fk < nz:
                    if (int(flag[fi, fj, fk]) & C.TYPE_BO_MASK) != C.TYPE_S:
                        pval = phi[fi, fj, fk]
                        break

        if di <= 6:      w = 4.0
        elif di <= 18:    w = 2.0
        else:             w = 1.0

        cxi = float(cx[di])
        cyi = float(cy[di])
        czi = float(cz[di])
        bx_n += w * cxi * pval
        by_n += w * cyi * pval
        bz_n += w * czi * pval

    # Match reference convention: normal = -grad(phi) (points fluid→gas)
    bz = wp.vec3(-bx_n, -by_n, -bz_n)
    len_sq = bx_n * bx_n + by_n * by_n + bz_n * bz_n
    if len_sq < 1.0e-20:
        return 0.0
    bz = wp.normalize(bz)

    # ---- Step 2: local frame ----
    rn = wp.vec3(0.56270900, 0.32704452, 0.75921047)
    by_raw = wp.cross(bz, rn)
    by_len_sq = by_raw.x * by_raw.x + by_raw.y * by_raw.y + by_raw.z * by_raw.z
    if by_len_sq < 1.0e-20:
        return 0.0
    by = wp.normalize(by_raw)
    bx = wp.cross(by, bz)

    # ---- Step 3: accumulate M and b from interface neighbours ----
    centre_offset = plic_cube(phi_center, bz)

    m00 = float(0.0); m01 = float(0.0); m02 = float(0.0); m03 = float(0.0); m04 = float(0.0)
    m11 = float(0.0); m12 = float(0.0); m13 = float(0.0); m14 = float(0.0)
    m22 = float(0.0); m23 = float(0.0); m24 = float(0.0)
    m33 = float(0.0); m34 = float(0.0)
    m44 = float(0.0)
    b0 = float(0.0); b1 = float(0.0); b2 = float(0.0); b3 = float(0.0); b4 = float(0.0)
    num = int(0)

    for di in range(1, 27):
        opp = opposite[di]
        ni = i - int(cx[opp])
        nj = j - int(cy[opp])
        nk = k - int(cz[opp])
        if px == 1:
            if ni < 0: ni += nx
            elif ni >= nx: ni -= nx
        if py == 1:
            if nj < 0: nj += ny
            elif nj >= ny: nj -= ny
        if pz == 1:
            if nk < 0: nk += nz
            elif nk >= nz: nk -= nz
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            continue
        phi_i = phi[ni, nj, nk]
        # Solid-neighbour fallback (ref: mrLbmSolverGpu3D.cu:842-860)
        if (int(flag[ni, nj, nk]) & C.TYPE_BO_MASK) == C.TYPE_S:
            for fd in range(1, 7):
                fi = ni - int(cx[fd]); fj = nj - int(cy[fd]); fk = nk - int(cz[fd])
                if px == 1:
                    if fi < 0: fi += nx
                    elif fi >= nx: fi -= nx
                if py == 1:
                    if fj < 0: fj += ny
                    elif fj >= ny: fj -= ny
                if pz == 1:
                    if fk < 0: fk += nz
                    elif fk >= nz: fk -= nz
                if fi >= 0 and fi < nx and fj >= 0 and fj < ny and fk >= 0 and fk < nz:
                    if (int(flag[fi, fj, fk]) & C.TYPE_BO_MASK) != C.TYPE_S:
                        phi_i = phi[fi, fj, fk]
                        break

        if phi_i > 0.0 and phi_i < 1.0:
            cxi = float(cx[di])
            cyi = float(cy[di])
            czi = float(cz[di])
            offset = plic_cube(phi_i, bz) - centre_offset

            px_v = cxi * bx.x + cyi * bx.y + czi * bx.z
            py_v = cxi * by.x + cyi * by.y + czi * by.z
            pz_v = (cxi * bz.x + cyi * bz.y + czi * bz.z) + offset

            x2 = px_v * px_v
            y2 = py_v * py_v
            x3 = x2 * px_v
            y3 = y2 * py_v

            m00 += x2 * x2
            m01 += x2 * y2
            m02 += x3 * py_v
            m03 += x3
            m04 += x2 * py_v
            m11 += y2 * y2
            m12 += px_v * y3
            m13 += px_v * y2
            m14 += y3
            m22 += x2 * y2
            m23 += x2 * py_v
            m24 += px_v * y2
            m33 += x2
            m34 += px_v * py_v
            m44 += y2

            b0 += x2 * pz_v
            b1 += y2 * pz_v
            b2 += px_v * py_v * pz_v
            b3 += px_v * pz_v
            b4 += py_v * pz_v

            num += 1

    if num < 5:
        return 0.0

    # ---- Step 4: solve 5x5 ----
    use_n = 5 if num >= 5 else num
    sol = _lu_solve_5x5_scalar(
        m00, m01, m02, m03, m04,
        m11, m12, m13, m14,
        m22, m23, m24,
        m33, m34, m44,
        b0, b1, b2, b3, b4,
        use_n,
    )

    A = sol.x; B = sol.y; C2 = sol.z; H = sol.w; I = sol.a

    # ---- Step 5: mean curvature ----
    d_inner = H * H + I * I + 1.0
    if d_inner < 1.0e-10:
        return 0.0
    denom = d_inner * wp.sqrt(d_inner)
    K_val = (A * (I * I + 1.0) + B * (H * H + 1.0) - C2 * H * I) / denom
    return wp.clamp(K_val, -1.0, 1.0)

# 5. stream_collide_bvh_kernel — monolithic fluid kernel
# ============================================================================
# Ref: ``mrLbmSolverGpu3D.cu:703-1057``
#
# This is the SINGLE kernel (audit item B4) that executes all fluid logic
# for the mrSolver3DGpu phase.  It is NOT split into separate
# stream_collide and free_surface kernels — the reference code
# conditionally branches inside ONE kernel based on cell flag.
#
# Pipeline phases (in order of execution per cell):
#
#   Phase A: Pull-streaming + Hermite reconstruction (lines 729-804)
#     - Loop over 27 neighbours, reconstruct fi from their HOME moments
#     - Solid neighbours → bounce-back with moving-wall correction
#     - Also reconstruct outgoing distributions from current cell (fon)
#
#   Phase B: Compute macroscopic moments from streamed populations
#            (lines 805-817)
#
#   Phase C: Free-surface interface handling (lines 862-910, TYPE_I only)
#     - PLIC curvature, bubble pressure, surface tension
#     - Guo forcing for gas pressure boundary condition
#
#   Phase D: Mass exchange (lines 926-929, TYPE_I only)
#     - Compute Δmass = f_opposite_i(neighbour_gas) − f_i(interface)
#     - Accumulate mass, massex, delta_phi, delta_g
#
#   Phase E: Eddy-viscosity turbulence model (lines 1001-1028)
#     - Neighbourhood search (±3 → 6×6×6)
#     - Strain-rate tensor from velocity gradients
#     - Smagorinsky closure: ν_e = factor · ‖S‖_F
#     - Modify local omega: ω_eff = 1 / (3·(ν + ν_e) + 0.5)
#
#   Phase F: NOCM-MRT collision (lines 1046-1055)
#     - Call ml_get_pi_after_collision
#     - Store post-collision moments to f_mom_post


@wp.kernel
def stream_collide_bvh_kernel(
    # ---- State arrays (read) ----
    f_mom: wp.array(dtype=float),              # [10 × N] current HOME moments
    flag: wp.array3d(dtype=wp.uint8),          # per-cell type bitfield
    phi: wp.array3d(dtype=float),              # volume fraction
    tag_matrix: wp.array3d(dtype=wp.int32),    # bubble ID tag
    disjoin_force: wp.array3d(dtype=float),    # disjoining pressure
    islet: wp.array3d(dtype=wp.int32),         # isolated-bubble flag

    # ---- Bubble properties (read) ----
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),

    # ---- Solid coupling (read) ----
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),

    # ---- Solver-owned accumulators (read+write) ----
    mass: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    delta_g: wp.array3d(dtype=float),
    delta_phi: wp.array3d(dtype=float),
    force_x: wp.array3d(dtype=float),
    force_y: wp.array3d(dtype=float),
    force_z: wp.array3d(dtype=float),
    c_value: wp.array3d(dtype=float),
    src: wp.array3d(dtype=float),

    # ---- State arrays (write) ----
    f_mom_post: wp.array(dtype=float),          # [10 × N] post-collision moments

    # ---- Parameters ----
    nx: int,
    ny: int,
    nz: int,
    stride: int,                                # nx * ny * nz
    omega: float,                               # NOCM-MRT relaxation frequency
    surface_tension: float,                     # def_6_sigma
    henry_constant: float,                      # K_h
    disjoin_factor: float,
    turbulence_factor: float,
    turbulence_radius: int,
    atmosphere_open: int,                       # bool → int
    px: int,
    py: int,
    pz: int,
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    w3d: wp.array(dtype=float),
    opposite: wp.array(dtype=wp.int32),
):
    """Monolithic fluid kernel: pull-streaming → collision → free-surface.

    Each thread processes one lattice node (i, j, k).
    """
    i, j, k = wp.tid()

    cur_idx = i * ny * nz + j * nz + k
    flagsn = int(flag[i, j, k])

    # ---- Per-direction neighbour index table (CX/CY/CZ) ----
    # (Hard-coded in the kernel for compiler unrolling — replicated from C.CX/CY/CZ.)

    # =====================================================================
    # Phase A: Pull-streaming + Hermite reconstruction
    # =====================================================================
    # Reference: ``mrLbmSolverGpu3D.cu:729-804``

    # Skip cells that do not participate in fluid dynamics
    flagsn_bo = flagsn & C.TYPE_BO_MASK
    flagsn_su = flagsn & C.TYPE_SU_MASK

    if flagsn_bo == C.TYPE_S:
        return  # solid cell — no fluid computation
    if flagsn_su == C.TYPE_G:
        return  # pure gas cell — no fluid computation

    # Islet (isolated bubble) handling (ref lines 723-726)
    # Islet=1 means this cell is an isolated bubble that should be treated as fluid.
    if islet[i, j, k] == 1:
        # Set flag to TYPE_F and skip — this cell acts as fluid
        flag[i, j, k] = wp.uint8(C.TYPE_F)
        return

    # ---- Load current-cell moments ----
    cr = f_mom[C.M_RHO * stride + cur_idx]
    cu = f_mom[C.M_UX * stride + cur_idx]
    cv = f_mom[C.M_UY * stride + cur_idx]
    cw = f_mom[C.M_UZ * stride + cur_idx]
    cSxx = f_mom[C.M_SXX * stride + cur_idx]
    cSxy = f_mom[C.M_SXY * stride + cur_idx]
    cSxz = f_mom[C.M_SXZ * stride + cur_idx]
    cSyy = f_mom[C.M_SYY * stride + cur_idx]
    cSyz = f_mom[C.M_SYZ * stride + cur_idx]
    cSzz = f_mom[C.M_SZZ * stride + cur_idx]

    # ---- Pull-streaming: gather from 27 neighbours (lines 741-777) ----
    # For each direction di, we pull the distribution f_i from the neighbour
    # at (i - CX[di], j - CY[di], k - CZ[di]).
    # The distribution is reconstructed from the neighbour's HOME moments.
    f_streamed = wp.zeros(27, dtype=float)
    fon = wp.zeros(27, dtype=float)

    # Compiler-unrolled loop over 27 directions
    for di in range(27):
        cx_di = cx[di]
        cy_di = cy[di]
        cz_di = cz[di]

        ni = i - cx_di
        nj = j - cy_di
        nk = k - cz_di

        # Periodic wrapping
        if px == 1:
            if ni < 0: ni += nx
            elif ni >= nx: ni -= nx
        if py == 1:
            if nj < 0: nj += ny
            elif nj >= ny: nj -= ny
        if pz == 1:
            if nk < 0: nk += nz
            elif nk >= nz: nk -= nz

        # Out-of-bounds → zero-velocity equilibrium bounce-back (matches TYPE_S branch)
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            f_streamed[di] = calculate_f_eq_d3q27(cr, 0.0, 0.0, 0.0, di)
        else:
            nflag_int = int(flag[ni, nj, nk])
            nflag_bo = nflag_int & C.TYPE_BO_MASK

            if nflag_bo == C.TYPE_S:
                # ---- Solid neighbour: bounce-back (ref lines 743-748) ----
                # fhn[i] = feq[i] — equilibrium in direction i at wall density
                f_streamed[di] = calculate_f_eq_d3q27(cr, 0.0, 0.0, 0.0, di)
            else:
                # ---- Fluid/interface/gas neighbour: normal stream (lines 764-777) ----
                nidx = ni * ny * nz + nj * nz + nk
                nr = f_mom[C.M_RHO * stride + nidx]
                nu = f_mom[C.M_UX * stride + nidx]
                nv = f_mom[C.M_UY * stride + nidx]
                nw = f_mom[C.M_UZ * stride + nidx]
                nSxx = f_mom[C.M_SXX * stride + nidx]
                nSxy = f_mom[C.M_SXY * stride + nidx]
                nSxz = f_mom[C.M_SXZ * stride + nidx]
                nSyy = f_mom[C.M_SYY * stride + nidx]
                nSyz = f_mom[C.M_SYZ * stride + nidx]
                nSzz = f_mom[C.M_SZZ * stride + nidx]
                f_streamed[di] = reconstruct_distribution(
                    nr, nu, nv, nw,
                    nSxx, nSxy, nSxz, nSyy, nSyz, nSzz,
                    di,
                )
                # ---- Convert from physical to HOME-stored format (ref line 776) ----
                f_streamed[di] -= w3d[di]

    # ---- Also reconstruct outgoing distributions (fon, lines 790-803) ----
    for di in range(27):
        fon[di] = reconstruct_distribution(
            cr, cu, cv, cw,
            cSxx, cSxy, cSxz, cSyy, cSyz, cSzz,
            di,
        )
        # ---- Convert from physical to HOME-stored format (ref line 803) ----
        fon[di] -= w3d[di]

    # ---- Mass accumulation from neighbours (ref lines 806-826) ----
    massn = mass[i, j, k]
    for di in range(1, 27):
        ni2 = i - cx[di]
        nj2 = j - cy[di]
        nk2 = k - cz[di]
        if px == 1:
            if ni2 < 0: ni2 += nx
            elif ni2 >= nx: ni2 -= nx
        if py == 1:
            if nj2 < 0: nj2 += ny
            elif nj2 >= ny: nj2 -= ny
        if pz == 1:
            if nk2 < 0: nk2 += nz
            elif nk2 >= nz: nk2 -= nz
        if ni2 >= 0 and ni2 < nx and nj2 >= 0 and nj2 < ny and nk2 >= 0 and nk2 < nz:
            massn += massex[ni2, nj2, nk2]

    if flagsn_su == C.TYPE_F:
        for di in range(1, 27):
            massn += f_streamed[di] - fon[di]

    mass[i, j, k] = massn

    # =====================================================================
    # Phase B: Restore full populations and compute post-streaming moments
    # =====================================================================
    # Reference: ``mrLbmSolverGpu3D.cu:940-971``
    #
    # Recover the full D3Q27 distributions from the streamed arrays
    # (fhn / fon had w3d_gpu subtracted during reconstruction).
    # Then compute density rho, velocity u (with half-force correction
    # and clamping), and stress components directly from pop.

    # ---- Restore pop = f_streamed + w3d (ref lines 940-942) ----
    pop = wp.zeros(27, dtype=float)
    for di in range(27):
        pop[di] = f_streamed[di] + w3d[di]

    # ---- Load body force at this cell ----
    fx = force_x[i, j, k]
    fy = force_y[i, j, k]
    fz = force_z[i, j, k]

    # ---- Compute density from pop (ref line 948) ----
    rho_new = float(0.0)
    for di in range(27):
        rho_new += pop[di]
    inv_rho = 1.0 / rho_new
    FX_scaled = fx * rho_new
    FY_scaled = fy * rho_new
    FZ_scaled = fz * rho_new

    # ---- Compute velocity with half-force correction (ref lines 956-958) ----
    ux_new = float(0.0)
    uy_new = float(0.0)
    uz_new = float(0.0)
    for di in range(27):
        pi = pop[di]
        ux_new += pi * float(cx[di])
        uy_new += pi * float(cy[di])
        uz_new += pi * float(cz[di])
    ux_new = (ux_new + 0.5 * FX_scaled) * inv_rho
    uy_new = (uy_new + 0.5 * FY_scaled) * inv_rho
    uz_new = (uz_new + 0.5 * FZ_scaled) * inv_rho

    # ---- Clamp velocity magnitude to 0.4 (ref lines 960-964) ----
    vel_sq = ux_new * ux_new + uy_new * uy_new + uz_new * uz_new
    if vel_sq > 0.16:
        scale_v = 0.4 / wp.sqrt(vel_sq)
        ux_new = ux_new * scale_v
        uy_new = uy_new * scale_v
        uz_new = uz_new * scale_v

    # ---- Compute stress components from pop (ref lines 966-971) ----
    # pixx_t45 = Σ f_i * c_ix²  (sum over all directions; c_ix² = 1 for directions
    # that have ±1 in x, 0 otherwise)
    pixx_pop = float(0.0)
    pixy_pop = float(0.0)
    pixz_pop = float(0.0)
    piyy_pop = float(0.0)
    piyz_pop = float(0.0)
    pizz_pop = float(0.0)
    for di in range(27):
        pi = pop[di]
        cx_i = float(cx[di])
        cy_i = float(cy[di])
        cz_i = float(cz[di])
        pixx_pop += pi * cx_i * cx_i
        pixy_pop += pi * cx_i * cy_i
        pixz_pop += pi * cx_i * cz_i
        piyy_pop += pi * cy_i * cy_i
        piyz_pop += pi * cy_i * cz_i
        pizz_pop += pi * cz_i * cz_i

    # =====================================================================
    # Phase C: Free-surface interface handling (TYPE_I only)
    # =====================================================================
    # Reference: ``mrLbmSolverGpu3D.cu:862-910``
    if flagsn_su == C.TYPE_I:
        # ---- Recalculate centre phi from current mass (ref line 867) ----
        # "don't load phi[n] from memory, instead recalculate it with mass
        #  corrected by excess mass"
        phi_self = calculate_phi(rho_new, massn, C.CellFlag.TYPE_I)

        # ---- Gas pressure ρ_k from bubble ----
        tag = tag_matrix[i, j, k]
        rho_k = 1.0  # default gas density
        if tag > 0:
            # bubble_rho is double; cast to float for arithmetic
            rho_k = float(bubble_rho[tag - 1])

        # ---- Surface tension modulation (ref lines 880-888) ----
        sigma_k = surface_tension
        if tag > 0:
            # For large bubbles (air layer): use init_volume (ref line 880)
            init_bv = float(bubble_init_volume[tag - 1])
            if init_bv > 5000000.0:
                sigma_k = 1.0e-6
            # For small bubbles: additional preconditions (ref lines 885-888)
            bv = float(bubble_volume[tag - 1])
            if disjoin_force[i, j, k] <= 0.0:
                if sigma_k > 1.0e-3:
                    if bv < 64.0:
                        sigma_k = 2.0e-4

        # ---- PLIC curvature ----
        # Centre phi is mass-corrected (ref line 867); neighbours read from grid.
        curv = calculate_curvature_from_grid(
            phi, flag, cx, cy, cz, opposite, i, j, k, nx, ny, nz, px, py, pz,
            phi_self)

        # Compute Laplace pressure, guarding against zero surface tension
        if sigma_k == 0.0:
            rho_laplace = 0.0
        else:
            rho_laplace = sigma_k * curv
        disjoint_term = disjoin_factor * disjoin_force[i, j, k]
        gas_pressure = rho_k - rho_laplace - disjoint_term

        # ---- Correct velocity for Guo forcing at interface (ref lines 900-908) ----
        # Half-force correction applied before the equilibrium reconstruction
        uxn_corrected = ux_new + 0.5 * fx * inv_rho
        uyn_corrected = uy_new + 0.5 * fy * inv_rho
        uzn_corrected = uz_new + 0.5 * fz * inv_rho

        # Clamp velocity magnitude to 0.4 (ref line 908)
        vel_mag_sq = (
            uxn_corrected * uxn_corrected
            + uyn_corrected * uyn_corrected
            + uzn_corrected * uzn_corrected
        )
        if vel_mag_sq > 0.16:  # 0.4^2
            scale = 0.4 / wp.sqrt(vel_mag_sq)
            uxn_corrected = uxn_corrected * scale
            uyn_corrected = uyn_corrected * scale
            uzn_corrected = uzn_corrected * scale

        # ---- Gas boundary condition: free-surface bounce-back (ref lines 930-934) ----
        # For each direction di (1..26), if the neighbour at direction di is gas (TYPE_G),
        # replace f_streamed[di] with the reference formula:
        #   fhn[i] = feg[opp(i)] - fon[opp(i)] + feg[i]
        # where opp(i) = opposite[i], feg[*] = gas equilibrium.
        for di in range(1, 27):
            opp = opposite[di]
            ni = i - cx[di]
            nj = j - cy[di]
            nk = k - cz[di]

            # Periodic wrap
            if px == 1:
                if ni < 0: ni += nx
                elif ni >= nx: ni -= nx
            if py == 1:
                if nj < 0: nj += ny
                elif nj >= ny: nj -= ny
            if pz == 1:
                if nk < 0: nk += nz
                elif nk >= nz: nk -= nz

            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue

            nflag_int2 = int(flag[ni, nj, nk])
            nflag_su2 = nflag_int2 & C.TYPE_SU_MASK
            if nflag_su2 == C.TYPE_G:
                # ---- Reference free-surface gas BC (mrLbmSolverGpu3D.cu:930-934) ----
                # feg[*] is HOME-stored; fon is HOME-stored (after -= w3d fix).
                # The formula ensures the incoming distribution from the gas side
                # reflects the gas pressure while correctly subtracting the
                # outgoing flux toward the gas.
                feg_di = calculate_f_eq_d3q27(
                    gas_pressure, uxn_corrected, uyn_corrected, uzn_corrected, di,
                )
                feg_opp = calculate_f_eq_d3q27(
                    gas_pressure, uxn_corrected, uyn_corrected, uzn_corrected, opp,
                )
                f_streamed[di] = feg_opp - fon[opp] + feg_di

        # ---- Mass exchange: compute Δmass (Phase D, ref lines 926-929) ----
        # For TYPE_I cells: accumulate mass flux across interface.
        # Δmass = Σ [0.5*(phi_self + phi_neighbour) * (fhn[di] - fon[opp])]
        # for all fluid/interface neighbours.
        # phi_self was already recalculated at the top of Phase C (ref line 867).
        mass_exchange = float(0.0)
        for di in range(1, 27):
            mni = i - cx[di]
            mnj = j - cy[di]
            mnk = k - cz[di]

            # Periodic wrap
            if px == 1:
                if mni < 0: mni += nx
                elif mni >= nx: mni -= nx
            if py == 1:
                if mnj < 0: mnj += ny
                elif mnj >= ny: mnj -= ny
            if pz == 1:
                if mnk < 0: mnk += nz
                elif mnk >= nz: mnk -= nz

            if mni < 0 or mni >= nx or mnj < 0 or mnj >= ny or mnk < 0 or mnk >= nz:
                continue

            mnflag_su = int(flag[mni, mnj, mnk]) & C.TYPE_SU_MASK
            if mnflag_su == C.TYPE_F or mnflag_su == C.TYPE_I:
                nphi = phi[mni, mnj, mnk]
                opp_di = int(opposite[di])
                dflux = f_streamed[di] - fon[opp_di]
                if mnflag_su == C.TYPE_F:
                    mass_exchange += dflux
                else:  # TYPE_I
                    mass_exchange += 0.5 * (nphi + phi_self) * dflux

        # Accumulate massex from neighbours (ref lines 808-818)
        massn_accum = massn + mass_exchange
        for di in range(1, 27):
            eni = i - cx[di]
            enj = j - cy[di]
            enk = k - cz[di]
            if px == 1:
                if eni < 0: eni += nx
                elif eni >= nx: eni -= nx
            if py == 1:
                if enj < 0: enj += ny
                elif enj >= ny: enj -= ny
            if pz == 1:
                if enk < 0: enk += nz
                elif enk >= nz: enk -= nz
            if eni >= 0 and eni < nx and enj >= 0 and enj < ny and enk >= 0 and enk < nz:
                massn_accum += massex[eni, enj, enk]

        mass[i, j, k] = massn_accum
        phi[i, j, k] = calculate_phi(rho_new, massn_accum, C.CellFlag.TYPE_I)

    elif flagsn_su == C.TYPE_F:
        # ---- Mass exchange for TYPE_F (ref lines 821-825) ----
        mass_exchange_f = float(0.0)
        for di in range(1, 27):
            mass_exchange_f += f_streamed[di] - fon[di]
        massn = massn + mass_exchange_f
        mass[i, j, k] = massn
        phi[i, j, k] = 1.0

    # =====================================================================
    # TYPE_NO_F / TYPE_NO_G flag transitions (ref lines 974-998)
    # =====================================================================
    # After mass exchange, TYPE_I cells are checked for neighbour deficiency.
    # If a TYPE_I cell has no fluid neighbours or mass exceeds density, it is
    # flagged TYPE_IF (interface->fluid).  If it has no gas neighbours or
    # mass is negative, it is flagged TYPE_IG (interface->gas).
    if flagsn_su == C.TYPE_I:
        type_no_f = int(1)  # int bool: 1 = no TYPE_F neighbour found yet
        type_no_g = int(1)
        for di in range(1, 27):
            nni = i - cx[di]
            nnj = j - cy[di]
            nnk = k - cz[di]
            if px == 1:
                if nni < 0: nni += nx
                elif nni >= nx: nni -= nx
            if py == 1:
                if nnj < 0: nnj += ny
                elif nnj >= ny: nnj -= ny
            if pz == 1:
                if nnk < 0: nnk += nz
                elif nnk >= nz: nnk -= nz
            if nni >= 0 and nni < nx and nnj >= 0 and nnj < ny and nnk >= 0 and nnk < nz:
                nsu = int(flag[nni, nnj, nnk]) & C.TYPE_SU_MASK
                if nsu == C.TYPE_F:
                    type_no_f = 0
                if nsu == C.TYPE_G:
                    type_no_g = 0

        cur_mass = mass[i, j, k]
        if cur_mass > rho_new:
            flag[i, j, k] = wp.uint8((flagsn & (0xFF ^ C.TYPE_SU_MASK)) | C.CellFlag.TYPE_IF)
        else:
            if type_no_g == 1:
                flag[i, j, k] = wp.uint8((flagsn & (0xFF ^ C.TYPE_SU_MASK)) | C.CellFlag.TYPE_IF)
            else:
                if cur_mass < 0.0:
                    flag[i, j, k] = wp.uint8((flagsn & (0xFF ^ C.TYPE_SU_MASK)) | C.CellFlag.TYPE_IG)
                else:
                    if type_no_f == 1:
                        flag[i, j, k] = wp.uint8((flagsn & (0xFF ^ C.TYPE_SU_MASK)) | C.CellFlag.TYPE_IG)

    # =====================================================================
    # Phase E: Eddy-viscosity turbulence model
    # =====================================================================
    # Reference: ``mrLbmSolverGpu3D.cu:1001-1028``
    #
    # Search a 6x6x6 neighbourhood (radius +/-3) for gas bubbles.
    # When a nearby bubble with volume < 5e6 is found, compute the
    # local strain-rate magnitude from the post-streaming stress and
    # modify the relaxation frequency via a Smagorinsky-type closure:
    #
    #   s_ab = Pi_ab / rho - c_s^2 * delta_ab
    #   |S|  = sqrt(s_xx^2 + 2 s_xy^2 + 2 s_xz^2 + s_yy^2 + 2 s_yz^2 + s_zz^2)
    #   nu_e = turbulence_factor * |S|
    #   omega_eff = 1 / ((nu_e + nu_0) * 3 + 0.5)    where nu_0 = 1e-4
    #
    # Without nearby bubbles the molecular omega is used.

    omega_eff = omega
    omega_base = omega

    if turbulence_factor > 0.0 and turbulence_radius > 0:
        # ---- Strain-rate from pop-computed stress (ref lines 1015-1020) ----
        sxx = pixx_pop * inv_rho - C.CS2
        syy = piyy_pop * inv_rho - C.CS2
        szz = pizz_pop * inv_rho - C.CS2
        sxy = pixy_pop * inv_rho
        sxz = pixz_pop * inv_rho
        syz = piyz_pop * inv_rho

        # ---- Neighbourhood search (ref lines 1001-1027) ----
        r = int(turbulence_radius)
        _found = int(0)
        for dij in range(-r, r):
            for djk in range(-r, r):
                for dkh in range(-r, r):
                    if _found == 1:
                        break
                    ni = i + djk
                    nj = j + dij
                    nk = k + dkh
                    if ni >= 0 and ni < nx and nj >= 0 and nj < ny and nk >= 0 and nk < nz:
                        ntag = tag_matrix[ni, nj, nk]
                        if ntag > 0:
                            bv = float(bubble_volume[ntag - 1])
                            if bv < 5000000.0:
                                fact2 = turbulence_factor
                                vis = fact2 * wp.sqrt(
                                    sxx * sxx + 2.0 * sxy * sxy + 2.0 * sxz * sxz
                                    + syy * syy + 2.0 * syz * syz + szz * szz
                                )
                                omega_eff = 1.0 / ((vis + 1.0e-4) * 3.0 + 0.5)
                                _found = 1

    # =====================================================================
    # Phase F: NOCM-MRT collision
    # =====================================================================
    # Reference: ``mrLbmSolverGpu3D.cu:1046-1055``
    #
    # Collide the stress tensor using ml_get_pi_after_collision.
    # IMPORTANT: The reference code uses the post-streaming stress
    # components computed from pop (lines 966-971), NOT the fMom
    # old stresses.  The collision acts on stress that already
    # incorporates the streaming information.

    pixx_old = pixx_pop
    pixy_old = pixy_pop
    pixz_old = pixz_pop
    piyy_old = piyy_pop
    piyz_old = piyz_pop
    pizz_old = pizz_pop

    pi_new = ml_get_pi_after_collision(
        rho_new, ux_new, uy_new, uz_new,
        FX_scaled, FY_scaled, FZ_scaled,
        omega_eff,
        pixx_old, pixy_old, pixz_old,
        piyy_old, piyz_old, pizz_old,
    )

    # ---- Store post-collision moments (ref lines 1046-1055) ----
    #   fMomPost[0] = rho
    #   fMomPost[1] = ux + Fx/(2ρ)
    #   fMomPost[2] = uy + Fy/(2ρ)
    #   fMomPost[3] = uz + Fz/(2ρ)
    #   fMomPost[4] = Π_xx_new / ρ - c_s²   (stored traceless)
    #   fMomPost[5] = Π_xy_new / ρ
    #   fMomPost[6] = Π_xz_new / ρ
    #   fMomPost[7] = Π_yy_new / ρ - c_s²
    #   fMomPost[8] = Π_yz_new / ρ
    #   fMomPost[9] = Π_zz_new / ρ - c_s²

    inv_rho_new = 1.0 / rho_new

    f_mom_post[C.M_RHO * stride + cur_idx] = rho_new
    f_mom_post[C.M_UX * stride + cur_idx] = ux_new + fx * 0.5
    f_mom_post[C.M_UY * stride + cur_idx] = uy_new + fy * 0.5
    f_mom_post[C.M_UZ * stride + cur_idx] = uz_new + fz * 0.5
    f_mom_post[C.M_SXX * stride + cur_idx] = pi_new.xx * inv_rho_new - C.CS2
    f_mom_post[C.M_SXY * stride + cur_idx] = pi_new.xy * inv_rho_new
    f_mom_post[C.M_SXZ * stride + cur_idx] = pi_new.xz * inv_rho_new
    f_mom_post[C.M_SYY * stride + cur_idx] = pi_new.yy * inv_rho_new - C.CS2
    f_mom_post[C.M_SYZ * stride + cur_idx] = pi_new.yz * inv_rho_new
    f_mom_post[C.M_SZZ * stride + cur_idx] = pi_new.zz * inv_rho_new - C.CS2


# ============================================================================
# 5.5 Test wrapper kernels — expose @wp.func helpers for unit testing


@wp.kernel
def _kernel_f_eq(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    output[tid] = calculate_f_eq_d3q27(rho, ux, uy, uz, tid)


@wp.kernel
def _kernel_reconstruct(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    pi_xx: float,
    pi_xy: float,
    pi_xz: float,
    pi_yy: float,
    pi_yz: float,
    pi_zz: float,
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    output[tid] = reconstruct_distribution(
        rho, ux, uy, uz,
        pi_xx, pi_xy, pi_xz, pi_yy, pi_yz, pi_zz,
        tid,
    )


@wp.kernel
def _kernel_collision(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    fx: float,
    fy: float,
    fz: float,
    omega: float,
    pixx_old: float,
    pixy_old: float,
    pixz_old: float,
    piyy_old: float,
    piyz_old: float,
    pizz_old: float,
    pi_new: wp.array(dtype=float),
):
    result = ml_get_pi_after_collision(
        rho, ux, uy, uz,
        fx, fy, fz, omega,
        pixx_old, pixy_old, pixz_old,
        piyy_old, piyz_old, pizz_old,
    )
    pi_new[0] = result.xx
    pi_new[1] = result.xy
    pi_new[2] = result.xz
    pi_new[3] = result.yy
    pi_new[4] = result.yz
    pi_new[5] = result.zz


@wp.kernel
def _kernel_compute_rho_u(
    f: wp.array(dtype=float),
    stride: int,
    out_rho: wp.array(dtype=float),
    out_ux: wp.array(dtype=float),
    out_uy: wp.array(dtype=float),
    out_uz: wp.array(dtype=float),
):
    tid = wp.tid()
    r = compute_rho_u_from_f(f, tid, stride)
    out_rho[tid] = r[0]
    out_ux[tid] = r[1]
    out_uy[tid] = r[2]
    out_uz[tid] = r[3]


# 6. Post-step swap kernel: f_mom_post → f_mom, g_mom_post → g_mom
# ============================================================================


@wp.kernel
def swap_moments_kernel(
    f_mom: wp.array(dtype=float),
    f_mom_post: wp.array(dtype=float),
    g_mom: wp.array(dtype=float),
    g_mom_post: wp.array(dtype=float),
    stride: int,
    gas_stride: int,
):
    """Swap post-collision moments into current arrays.

    This implements the ``mrSolver3D_step2Kernel`` logical equivalent:
    at the end of each phase, fMom = fMomPost and gMom = gMomPost.
    """
    tid = wp.tid()
    if tid < stride:
        for m in range(C.NUM_MOMENTS):
            f_mom[m * stride + tid] = f_mom_post[m * stride + tid]
        for m in range(C.NUM_DIRS_GAS):
            g_mom[m * gas_stride + tid] = g_mom_post[m * gas_stride + tid]


# ============================================================================
# 7. Gravity injection kernel — add gravity to body-force arrays
# ============================================================================


@wp.kernel
def add_gravity_kernel(
    force_x: wp.array3d(dtype=float),
    force_y: wp.array3d(dtype=float),
    force_z: wp.array3d(dtype=float),
    gx: float,
    gy: float,
    gz: float,
):
    """Add constant gravity to per-cell body-force arrays."""
    i, j, k = wp.tid()
    force_x[i, j, k] = force_x[i, j, k] + gx
    force_y[i, j, k] = force_y[i, j, k] + gy
    force_z[i, j, k] = force_z[i, j, k] + gz


# ============================================================================
# 8. Surface test wrappers — thin kernels for @wp.func unit tests
# ============================================================================


@wp.kernel
def _kernel_calculate_phi(
    rhon: wp.array(dtype=float),
    massn: wp.array(dtype=float),
    flagsn: wp.array(dtype=wp.int32),
    result: wp.array(dtype=float),
):
    tid = wp.tid()
    result[tid] = calculate_phi(rhon[tid], massn[tid], int(flagsn[tid]))


@wp.kernel
def _kernel_plic_cube(
    V0: wp.array(dtype=float),
    nx_v: wp.array(dtype=float),
    ny_v: wp.array(dtype=float),
    nz_v: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    tid = wp.tid()
    n = wp.vec3(nx_v[tid], ny_v[tid], nz_v[tid])
    result[tid] = plic_cube(V0[tid], n)