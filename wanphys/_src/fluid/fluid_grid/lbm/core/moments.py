# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME moment-space collision (HOME-FREE Eq. 18–21 / HOME-LBM Eq. 21–23).

After reconstruct-stream, temporary moments ``(ρ*, u*, S*)`` are collided to
post-collision moments stored for the next step::

    ρ  ← ρ*
    u  ← u* + F / (2 ρ*)
    S_αβ (α≠β) ← (1 − 1/τ) S*_αβ + (1/τ) u*_α u*_β
                 + (2τ−1)/(2τ ρ*) (F_α u*_β + F_β u*_α)
    S_αα ← NOCM diagonal form (HOME Eq. for S_xx / S_yy / S_zz)

Note: HOME-FREE Eq. 21 OCR writes ``1/(ρ* F_α u*_α)``; we follow the HOME-LBM
closed form ``F_α u*_α / ρ*``.

Increment **H1**: collision primitives. Reconstruct-stream lives in
``backends/moment/home_fp32_ref``.
"""

from __future__ import annotations

from wanphys._src.fluid.fluid_grid.lbm.core.hermite import HomeMoments

import warp as wp


def collide_moments_numpy(
    rho_s: float,
    ux_s: float,
    uy_s: float,
    uz_s: float,
    sxx_s: float,
    syy_s: float,
    szz_s: float,
    sxy_s: float,
    sxz_s: float,
    syz_s: float,
    tau: float,
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
) -> HomeMoments:
    """HOME-FREE / HOME-LBM moment collision (numpy).

    Parameters
    ----------
    *_s
        Temporary (post-stream) moments ``ρ*, u*, S*``.
    tau
        Shear relaxation time τ (> 0.5).
    fx, fy, fz
        Body force density F (lattice units), Guo-style half-force on ``u``.
    """
    if rho_s <= 0.0:
        return HomeMoments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    inv_tau = 1.0 / tau
    one_m = 1.0 - inv_tau
    inv_rho = 1.0 / rho_s
    force_fac = (2.0 * tau - 1.0) / (2.0 * tau) * inv_rho

    rho = rho_s
    ux = ux_s + 0.5 * fx * inv_rho
    uy = uy_s + 0.5 * fy * inv_rho
    uz = uz_s + 0.5 * fz * inv_rho

    # Off-diagonal (HOME Eq. 23 / HOME-FREE Eq. 20)
    sxy = (
        one_m * sxy_s
        + inv_tau * ux_s * uy_s
        + force_fac * (fx * uy_s + fy * ux_s)
    )
    sxz = (
        one_m * sxz_s
        + inv_tau * ux_s * uz_s
        + force_fac * (fx * uz_s + fz * ux_s)
    )
    syz = (
        one_m * syz_s
        + inv_tau * uy_s * uz_s
        + force_fac * (fy * uz_s + fz * uy_s)
    )

    # Diagonal NOCM (HOME closed form; FREE Eq. 21)
    u2 = ux_s * ux_s + uy_s * uy_s + uz_s * uz_s
    pref = (tau - 1.0) / (3.0 * tau)
    inv_3tau = 1.0 / (3.0 * tau)
    third = 1.0 / 3.0
    force_diag = (tau - 1.0) / (3.0 * tau) * inv_rho

    sxx = (
        pref * (2.0 * sxx_s - syy_s - szz_s)
        + third * u2
        + inv_3tau * (2.0 * ux_s * ux_s - uy_s * uy_s - uz_s * uz_s)
        + inv_rho * fx * ux_s
        + force_diag * (2.0 * fx * ux_s - fy * uy_s - fz * uz_s)
    )
    syy = (
        pref * (2.0 * syy_s - sxx_s - szz_s)
        + third * u2
        + inv_3tau * (2.0 * uy_s * uy_s - ux_s * ux_s - uz_s * uz_s)
        + inv_rho * fy * uy_s
        + force_diag * (2.0 * fy * uy_s - fx * ux_s - fz * uz_s)
    )
    szz = (
        pref * (2.0 * szz_s - sxx_s - syy_s)
        + third * u2
        + inv_3tau * (2.0 * uz_s * uz_s - ux_s * ux_s - uy_s * uy_s)
        + inv_rho * fz * uz_s
        + force_diag * (2.0 * fz * uz_s - fx * ux_s - fy * uy_s)
    )

    return HomeMoments(rho, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz)


@wp.func
def home_collide_moments(
    rho_s: float,
    ux_s: float,
    uy_s: float,
    uz_s: float,
    sxx_s: float,
    syy_s: float,
    szz_s: float,
    sxy_s: float,
    sxz_s: float,
    syz_s: float,
    tau: float,
    fx: float,
    fy: float,
    fz: float,
):
    """Warp: HOME moment collision → (ρ,u,S) as 10 floats."""
    inv_tau = 1.0 / tau
    one_m = 1.0 - inv_tau
    inv_rho = 1.0 / rho_s
    force_fac = (2.0 * tau - 1.0) / (2.0 * tau) * inv_rho

    rho = rho_s
    ux = ux_s + 0.5 * fx * inv_rho
    uy = uy_s + 0.5 * fy * inv_rho
    uz = uz_s + 0.5 * fz * inv_rho

    sxy = (
        one_m * sxy_s
        + inv_tau * ux_s * uy_s
        + force_fac * (fx * uy_s + fy * ux_s)
    )
    sxz = (
        one_m * sxz_s
        + inv_tau * ux_s * uz_s
        + force_fac * (fx * uz_s + fz * ux_s)
    )
    syz = (
        one_m * syz_s
        + inv_tau * uy_s * uz_s
        + force_fac * (fy * uz_s + fz * uy_s)
    )

    u2 = ux_s * ux_s + uy_s * uy_s + uz_s * uz_s
    pref = (tau - 1.0) / (3.0 * tau)
    inv_3tau = 1.0 / (3.0 * tau)
    third = 1.0 / 3.0
    force_diag = (tau - 1.0) / (3.0 * tau) * inv_rho

    sxx = (
        pref * (2.0 * sxx_s - syy_s - szz_s)
        + third * u2
        + inv_3tau * (2.0 * ux_s * ux_s - uy_s * uy_s - uz_s * uz_s)
        + inv_rho * fx * ux_s
        + force_diag * (2.0 * fx * ux_s - fy * uy_s - fz * uz_s)
    )
    syy = (
        pref * (2.0 * syy_s - sxx_s - szz_s)
        + third * u2
        + inv_3tau * (2.0 * uy_s * uy_s - ux_s * ux_s - uz_s * uz_s)
        + inv_rho * fy * uy_s
        + force_diag * (2.0 * fy * uy_s - fx * ux_s - fz * uz_s)
    )
    szz = (
        pref * (2.0 * szz_s - sxx_s - syy_s)
        + third * u2
        + inv_3tau * (2.0 * uz_s * uz_s - ux_s * ux_s - uy_s * uy_s)
        + inv_rho * fz * uz_s
        + force_diag * (2.0 * fz * uz_s - fx * ux_s - fy * uy_s)
    )
    return rho, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz
