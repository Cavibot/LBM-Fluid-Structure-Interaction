# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-LBM Hermite primitives (Li et al. 2023 / HOME-FREE Eq. 16–17).

Moment definition (HOME Eq. 7)::

    ρ = Σ f_i
    ρ u = Σ c_i f_i
    ρ S_αβ = Σ (c_iα c_iβ − c_s² δ_αβ) f_i

Third-order reconstruction (HOME Eq. 17 / HOME-FREE Eq. 16)::

    f_i = ρ w_i [ 1 + (c·u)/c_s² + H^[2](c):S / (2 c_s⁴)
                  + Σ H^[3]_αβγ(c) T_αβγ / (2 c_s⁶) ]

with T_αβγ = S_αβ u_γ + S_αγ u_β + S_βγ u_α − 2 u_α u_β u_γ
(HOME-FREE Eq. 17). Note: some OCR of HOME Eq. 17 write ``2 S_yz u_z``
in the yyz channel; we use ``2 S_yz u_y`` from the tensor definition.

H0: reconstruction + moment extraction.
H1: collision / stream in ``core/moments.py`` and ``home_fp32_ref/step.py``.
H2 / G4: VOF coupling later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

CS2: float = 1.0 / 3.0
INV_CS2: float = 3.0
INV_2CS4: float = 4.5  # 1 / (2 cs⁴) = 1/(2/9) = 4.5
INV_2CS6: float = 13.5  # 1 / (2 cs⁶) = 1/(2/27) = 13.5


@dataclass(frozen=True)
class HomeMoments:
    """Per-cell HOME moments (ρ, u, symmetric S)."""

    rho: float
    ux: float
    uy: float
    uz: float
    sxx: float
    syy: float
    szz: float
    sxy: float
    sxz: float
    syz: float


def moments_from_f_numpy(
    f: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
) -> HomeMoments:
    """Extract HOME moments from a length-Q population vector (numpy)."""
    f = np.asarray(f, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    cz = np.asarray(cz, dtype=np.float64)
    rho = float(np.sum(f))
    if rho <= 0.0:
        return HomeMoments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ux = float(np.dot(cx, f) / rho)
    uy = float(np.dot(cy, f) / rho)
    uz = float(np.dot(cz, f) / rho)
    sxx = float(np.dot(cx * cx - CS2, f) / rho)
    syy = float(np.dot(cy * cy - CS2, f) / rho)
    szz = float(np.dot(cz * cz - CS2, f) / rho)
    sxy = float(np.dot(cx * cy, f) / rho)
    sxz = float(np.dot(cx * cz, f) / rho)
    syz = float(np.dot(cy * cz, f) / rho)
    return HomeMoments(rho, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz)


def reconstruct_f_numpy(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """HOME Eq. 17 third-order Hermite reconstruction (numpy, all directions)."""
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    cz = np.asarray(cz, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    q = w.shape[0]
    out = np.zeros(q, dtype=np.float64)
    for i in range(q):
        out[i] = reconstruct_f_i_numpy(
            rho, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz,
            float(cx[i]), float(cy[i]), float(cz[i]), float(w[i]),
        )
    return out


def reconstruct_f_i_numpy(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    cx: float,
    cy: float,
    cz: float,
    w: float,
) -> float:
    """Single-direction HOME Eq. 17 reconstruction."""
    cu = cx * ux + cy * uy + cz * uz
    # H^[2](c):S  with H_αβ = c_α c_β − cs² δ_αβ
    h2s = (
        (cx * cx - CS2) * sxx
        + (cy * cy - CS2) * syy
        + (cz * cz - CS2) * szz
        + 2.0 * cx * cy * sxy
        + 2.0 * cx * cz * sxz
        + 2.0 * cy * cz * syz
    )
    # Hermite H^[3] components used in HOME Eq. 17
    h_xxy = cx * cx * cy - CS2 * cy
    h_xyy = cx * cy * cy - CS2 * cx
    h_xxz = cx * cx * cz - CS2 * cz
    h_xzz = cx * cz * cz - CS2 * cx
    h_yzz = cy * cz * cz - CS2 * cy
    h_yyz = cy * cy * cz - CS2 * cz
    h_xyz = cx * cy * cz
    # Contracted third-order coefficients (already include T_αβγ)
    t3 = (
        h_xxy * (sxx * uy + 2.0 * sxy * ux - 2.0 * ux * ux * uy)
        + h_xyy * (syy * ux + 2.0 * sxy * uy - 2.0 * ux * uy * uy)
        + h_xxz * (sxx * uz + 2.0 * sxz * ux - 2.0 * ux * ux * uz)
        + h_xzz * (szz * ux + 2.0 * sxz * uz - 2.0 * ux * uz * uz)
        + h_yzz * (szz * uy + 2.0 * syz * uz - 2.0 * uy * uz * uz)
        + h_yyz * (syy * uz + 2.0 * syz * uy - 2.0 * uy * uy * uz)
        + h_xyz * (sxz * uy + syz * ux + sxy * uz - 2.0 * ux * uy * uz)
    )
    bracket = 1.0 + INV_CS2 * cu + INV_2CS4 * h2s + INV_2CS6 * t3
    return rho * w * bracket


def equilibrium_s_from_u(ux: float, uy: float, uz: float) -> tuple[float, ...]:
    """Equilibrium S = u ⊗ u (HOME collision target)."""
    return (ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz)


@wp.func
def home_reconstruct_f_i(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    cx: int,
    cy: int,
    cz: int,
    w: float,
) -> float:
    """Warp: HOME Eq. 17 third-order reconstruction for one lattice direction."""
    fcx = float(cx)
    fcy = float(cy)
    fcz = float(cz)
    cs2 = 1.0 / 3.0
    cu = fcx * ux + fcy * uy + fcz * uz
    h2s = (
        (fcx * fcx - cs2) * sxx
        + (fcy * fcy - cs2) * syy
        + (fcz * fcz - cs2) * szz
        + 2.0 * fcx * fcy * sxy
        + 2.0 * fcx * fcz * sxz
        + 2.0 * fcy * fcz * syz
    )
    h_xxy = fcx * fcx * fcy - cs2 * fcy
    h_xyy = fcx * fcy * fcy - cs2 * fcx
    h_xxz = fcx * fcx * fcz - cs2 * fcz
    h_xzz = fcx * fcz * fcz - cs2 * fcx
    h_yzz = fcy * fcz * fcz - cs2 * fcy
    h_yyz = fcy * fcy * fcz - cs2 * fcz
    h_xyz = fcx * fcy * fcz
    t3 = (
        h_xxy * (sxx * uy + 2.0 * sxy * ux - 2.0 * ux * ux * uy)
        + h_xyy * (syy * ux + 2.0 * sxy * uy - 2.0 * ux * uy * uy)
        + h_xxz * (sxx * uz + 2.0 * sxz * ux - 2.0 * ux * ux * uz)
        + h_xzz * (szz * ux + 2.0 * sxz * uz - 2.0 * ux * uz * uz)
        + h_yzz * (szz * uy + 2.0 * syz * uz - 2.0 * uy * uz * uz)
        + h_yyz * (syy * uz + 2.0 * syz * uy - 2.0 * uy * uy * uz)
        + h_xyz * (sxz * uy + syz * ux + sxy * uz - 2.0 * ux * uy * uz)
    )
    bracket = 1.0 + 3.0 * cu + 4.5 * h2s + 13.5 * t3
    return rho * w * bracket
