# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Boundary-condition helpers for HOME-FSLBM.

Reference:
    [REF] Home-FSLBM inc/3D/gpu/mrLbmSolverGpu3D.cu — bounce-back at
          stream_collide_bvh:743-748.
    [REF] normalizing_clamp (velocity clamping).
"""

from __future__ import annotations

import warp as wp


@wp.func
def normalizing_clamp(u: wp.vec3, max_val: float) -> wp.vec3:
    """Clamp velocity vector magnitude to ``max_val``.

    [REF]: normalizing_clamp in stream_collide_bvh (mrLbmSolverGpu3D.cu).
    """
    mag_sq = u.x * u.x + u.y * u.y + u.z * u.z
    if mag_sq > max_val * max_val:
        scale = max_val * wp.rsqrt(mag_sq)
        return wp.vec3(u.x * scale, u.y * scale, u.z * scale)
    return u


# ============================================================================
# Free-surface VOF helpers  [REF] mrUtilFuncGpu3D.h:351-353
# ============================================================================
@wp.func
def calculate_phi(rho: float, mass: float, surf_flag: int) -> float:
    """Compute VOF fill fraction phi from density and mass.

    [REF] mrUtilFuncGpu3D::calculate_phi:
        - TYPE_F -> 1.0
        - TYPE_I -> clamp(mass/rho, 0, 1) if rho > 0 else 0.5
        - otherwise -> 0.0

    Args:
        rho: Cell density.
        mass: Actual fluid mass in cell.
        surf_flag: Surface type (TYPE_F / TYPE_I / TYPE_G extracted via & TYPE_SU).

    Returns:
        Phi value in [0, 1].
    """
    from .constants import TYPE_F, TYPE_I
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
# Curvature / surface tension skeleton  (sigma=0 for stage 3)
# ============================================================================
@wp.func
def calculate_curvature(phi_n: float) -> float:
    """Compute mean curvature at an interface cell (skeleton for stage 3).

    Full implementation deferred to stage 6a (surface tension).
    Currently returns 0.0 (zero surface tension).

    [REF] mrUtilFuncGpu3D::calculate_curvature — D3Q27 stencil + PLIC fit.

    Args:
        phi_n: Phi value at current cell (unused in skeleton).

    Returns:
        0.0 (zero curvature / no Laplace pressure).
    """
    return 0.0


@wp.func
def equilibrium_fi(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    i: int,
    c_x: wp.array(dtype=wp.int32),
    c_y: wp.array(dtype=wp.int32),
    c_z: wp.array(dtype=wp.int32),
    w: wp.array(dtype=wp.float32),
) -> float:
    """Compute equilibrium distribution f_i^eq for direction *i*.

    [P4] Eq.(16) with S=0 (pure equilibrium, second-order expansion).

    Args:
        rho: Density.
        ux, uy, uz: Velocity components.
        i: Lattice direction index (0..26).
        c_x, c_y, c_z: D3Q27 velocity component arrays.
        w: D3Q27 weight array.

    Returns:
        Equilibrium distribution value f_i^eq.
    """
    cu = float(c_x[i]) * ux + float(c_y[i]) * uy + float(c_z[i]) * uz
    u2 = ux * ux + uy * uy + uz * uz
    w_i = float(w[i])
    return rho * w_i * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
