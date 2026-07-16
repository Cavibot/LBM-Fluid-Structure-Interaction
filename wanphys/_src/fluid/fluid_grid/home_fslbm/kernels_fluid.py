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

#wp.set_module_options({"enable_backward": False})

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
    if di == 25:   # (-1,-1,-1) — wait, reference has 25→(-1,-1,-1) and 26→(+1,-1,-1)
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

        # Out-of-bounds → reconstruct using current cell's own moments
        # (equivalent to equilibrium bounce-back at domain wall)
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            opp = opposite[di]
            f_streamed[di] = reconstruct_distribution(
                cr, cu, cv, cw,
                cSxx, cSxy, cSxz, cSyy, cSyz, cSzz,
                opp,
            )
        else:
            nflag_int = int(flag[ni, nj, nk])
            nflag_bo = nflag_int & C.TYPE_BO_MASK

            if nflag_bo == C.TYPE_S:
                # ---- Solid neighbour: bounce-back (lines 743-748) ----
                # Bounce-back replaces the incoming distribution with the
                # equilibrium from the current node (stationary wall) or
                # with a moving-wall correction.
                # For now: simple bounce-back (equilibrium at current node).
                # Moving-wall correction will be added in a later refinement.
                opp = opposite[di]
                f_streamed[di] = reconstruct_distribution(
                    cr, cu, cv, cw,
                    cSxx, cSxy, cSxz, cSyy, cSyz, cSzz,
                    opp,
                )
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

    # ---- Also reconstruct outgoing distributions (fon, lines 790-803) ----
    for di in range(27):
        fon[di] = reconstruct_distribution(
            cr, cu, cv, cw,
            cSxx, cSxy, cSxz, cSyy, cSyz, cSzz,
            di,
        )

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
    #
    # TODO Phase 2: Implement PLIC curvature, bubble pressure lookup,
    # surface tension modulation, Laplace pressure, gas pressure BC,
    # and Guo forcing correction for interface cells.
    #
    # Skeleton: skip if not interface.
    # Full implementation will replace the skip with the logic below.
    if flagsn_su == C.TYPE_I:
        # ---- Placeholder: gas pressure ρ_k from bubble ----
        tag = tag_matrix[i, j, k]
        rho_k = 1.0  # default gas density
        if tag > 0:
            # bubble_rho is double; cast to float for arithmetic
            rho_k = float(bubble_rho[tag - 1])

        # ---- Surface tension modulation (ref lines 880-888) ----
        sigma_k = surface_tension
        if tag > 0:
            bv = float(bubble_volume[tag - 1])
            if bv > 5.0e6:
                sigma_k = 1.0e-6
            elif bv < 64.0:
                sigma_k = 2.0e-4

        # ---- PLIC curvature → placeholder (Phase 2) ----
        curv = 0.0  # TODO Phase 2: calculate_curvature from phi neighbours

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

        # ---- Gas boundary condition: replace incoming from gas side ----
        # For each direction di, if the neighbour in OPPOSITE[di] is gas (TYPE_G),
        # replace f_streamed[di] with equilibrium at gas_pressure + corrected velocity.
        for di in range(27):
            opp = opposite[di]
            ni = i - cx[opp]
            nj = j - cy[opp]
            nk = k - cz[opp]

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
                # Replace with equilibrium at gas pressure (Guo forcing form)
                f_streamed[di] = calculate_f_eq_d3q27(
                    gas_pressure, uxn_corrected, uyn_corrected, uzn_corrected, di,
                )

        # ---- Mass exchange: compute Δmass (Phase D skeleton) ----
        # TODO Phase 2: Full mass exchange logic (lines 926-929, 780-999)
        # acc_mass: float = 0.0
        # acc_mass2: float = 0.0
        # for di in range(27):
        #     ...  # Detailed mass flux calculation
        # atomic_add mass, massex, delta_g, delta_phi

        # ---- Placeholder: store current mass/phi for now ----
        phi[i, j, k] = 0.5  # TODO Phase 2: proper PLIC phi

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
        fx, fy, fz,
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

    half_dt_inv_rho = 0.5 / rho_new
    inv_rho_new = 1.0 / rho_new

    f_mom_post[C.M_RHO * stride + cur_idx] = rho_new
    f_mom_post[C.M_UX * stride + cur_idx] = ux_new + fx * half_dt_inv_rho
    f_mom_post[C.M_UY * stride + cur_idx] = uy_new + fy * half_dt_inv_rho
    f_mom_post[C.M_UZ * stride + cur_idx] = uz_new + fz * half_dt_inv_rho
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