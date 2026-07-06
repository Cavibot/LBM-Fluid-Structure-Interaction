# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Hyperelastic constitutive (material) laws for FEM force evaluation.

Each law computes the first Piola-Kirchhoff (PK1) stress **P** from the
deformation gradient **F**.  Both 3-D (tetrahedral) and 2-D (membrane /
triangle) variants are provided.

Design note
-----------
The models here supply the *deviatoric / shear* contribution controlled by
the shear modulus  μ.  Volumetric (or area-preservation) response is handled
separately inside the element kernels via λ-scaled terms, following the
standard isochoric-volumetric split used in computational mechanics.

Available laws
--------------
* **NEOHOOKEAN_REST_STABLE** – rest-stability regularised Neo-Hookean
  (Smith et al. 2018 inspired).  Robust default for large deformations.
* **NEOHOOKEAN_ISOCHORIC** – classical deviatoric Neo-Hookean (J^{-2/3}
  scaling).  Suitable when combined with a separate bulk-modulus penalty.
* **NEOHOOKEAN_SMOOTH** – smooth √(|I_C − 3|) variant; softer near rest.
* **NEOHOOKEAN_SPHERICAL** – spherical-constraint variant.
* **NEOHOOKEAN_QUADRATIC** – quadratic strain-measure variant.
* **HOOKEAN_LINEAR** – linearised Hookean (small-strain approximation).

References
----------
Smith, B. et al. "Stable Neo-Hookean Flesh Simulation." ACM TOG, 2018.
Bonet, J. & Wood, R. *Nonlinear Continuum Mechanics for FE Analysis*.
"""

from __future__ import annotations

from enum import IntEnum

import warp as wp


# ---------------------------------------------------------------------------
# Material law catalogue
# ---------------------------------------------------------------------------


class MaterialLaw(IntEnum):
    """Catalogue of hyperelastic material laws."""

    NEOHOOKEAN_REST_STABLE = 0
    """Rest-stability regularised Neo-Hookean.

    P = μ (1 − 1/(I_C + 1)) F  +  k_d dF/dt

    Numerically robust even at near-zero deformation.
    """

    NEOHOOKEAN_ISOCHORIC = 1
    """Isochoric (deviatoric) Neo-Hookean – standard industrial split form.

    P = μ J^{-2/3} ( F − (Ī₁/3) F^{-T} )  +  k_d dF/dt
    """

    NEOHOOKEAN_SMOOTH = 2
    """Smooth √ variant based on |I_C − 3|.

    P = μ (I_C−3) / √((I_C−3)²+ε²)  F  +  k_d dF/dt
    """

    NEOHOOKEAN_SPHERICAL = 3
    """Spherical-constraint Neo-Hookean.

    C = √I_C − √3 ;   P = μ C (F / √I_C)  +  k_d dF/dt
    """

    NEOHOOKEAN_QUADRATIC = 4
    """Quadratic strain-measure (C_D) variant.

    C = I_C − 3 ;   P = 2μ C F  +  k_d dF/dt
    """

    HOOKEAN_LINEAR = 5
    """Linearised Hookean (small-strain regime).

    ε = (F + Fᵀ)/2 − I ;   P ≈ 2μ ε  +  k_d dF/dt
    """


# =====================================================================
# 3-D stress functions (tetrahedra)
# =====================================================================


@wp.func
def _first_invariant_3d(F: wp.mat33) -> float:
    """I_C = tr(FᵀF) = ‖F‖²_F."""
    c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    return wp.dot(c0, c0) + wp.dot(c1, c1) + wp.dot(c2, c2)


@wp.func
def _pk1_rest_stable_3d(
    F: wp.mat33, dFdt: wp.mat33, mu: float, damp: float,
) -> wp.mat33:
    Ic = _first_invariant_3d(F)
    return F * (mu * (1.0 - 1.0 / (Ic + 1.0))) + dFdt * damp


@wp.func
def _pk1_isochoric_3d(
    F: wp.mat33, dFdt: wp.mat33, mu: float, damp: float,
) -> wp.mat33:
    J = wp.determinant(F)
    Jc = wp.max(J, 1.0e-12)
    Finv_T = wp.transpose(wp.inverse(F))
    I1 = _first_invariant_3d(F)
    Jm23 = wp.pow(Jc, -2.0 / 3.0)
    return (F - Finv_T * (I1 / 3.0)) * (mu * Jm23) + dFdt * damp


@wp.func
def _pk1_smooth_3d(
    F: wp.mat33, dFdt: wp.mat33, mu: float, damp: float,
) -> wp.mat33:
    Ic = _first_invariant_3d(F)
    x = Ic - 3.0
    eps = 1.0e-6
    scale = x / wp.sqrt(x * x + eps * eps)
    return F * (mu * scale) + dFdt * damp


@wp.func
def _pk1_spherical_3d(
    F: wp.mat33, dFdt: wp.mat33, mu: float, damp: float,
) -> wp.mat33:
    Ic = _first_invariant_3d(F)
    r = wp.sqrt(Ic)
    if r > 1.0e-12:
        C = r - wp.sqrt(3.0)
        P = F * (mu * C / r) + dFdt * damp
    else:
        P = dFdt * damp
    return P


@wp.func
def _pk1_quadratic_3d(
    F: wp.mat33, dFdt: wp.mat33, mu: float, damp: float,
) -> wp.mat33:
    Ic = _first_invariant_3d(F)
    C = Ic - 3.0
    return F * (2.0 * mu * C) + dFdt * damp


@wp.func
def _pk1_hookean_3d(
    F: wp.mat33, dFdt: wp.mat33, mu: float, damp: float,
) -> wp.mat33:
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return (F + wp.transpose(F) - I * 2.0) * mu + dFdt * damp


# -- unified dispatcher ---------------------------------------------------

@wp.func
def pk1_stress_3d(
    law: int,
    F: wp.mat33,
    dFdt: wp.mat33,
    mu: float,
    damp: float,
) -> wp.mat33:
    """Dispatch PK1 stress computation for a 3-D element."""
    P = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if law == 0:
        P = _pk1_rest_stable_3d(F, dFdt, mu, damp)
    elif law == 1:
        P = _pk1_isochoric_3d(F, dFdt, mu, damp)
    elif law == 2:
        P = _pk1_smooth_3d(F, dFdt, mu, damp)
    elif law == 3:
        P = _pk1_spherical_3d(F, dFdt, mu, damp)
    elif law == 4:
        P = _pk1_quadratic_3d(F, dFdt, mu, damp)
    elif law == 5:
        P = _pk1_hookean_3d(F, dFdt, mu, damp)
    else:
        P = _pk1_rest_stable_3d(F, dFdt, mu, damp)
    return P


# =====================================================================
# 2-D stress functions (membrane / triangle elements)
# =====================================================================


@wp.func
def _first_invariant_2d(g1: wp.vec3, g2: wp.vec3) -> float:
    """I_C = ‖g1‖² + ‖g2‖² for the 2-D column representation."""
    return wp.dot(g1, g1) + wp.dot(g2, g2)


@wp.func
def _pk1_rest_stable_2d(
    g1: wp.vec3, g2: wp.vec3,
    dg1: wp.vec3, dg2: wp.vec3,
    mu: float, damp: float,
) -> tuple[wp.vec3, wp.vec3]:
    Ic = _first_invariant_2d(g1, g2)
    s = mu * (1.0 - 1.0 / (Ic + 1.0))
    return g1 * s + dg1 * damp, g2 * s + dg2 * damp


@wp.func
def _pk1_isochoric_2d(
    g1: wp.vec3, g2: wp.vec3,
    dg1: wp.vec3, dg2: wp.vec3,
    mu: float, damp: float,
) -> tuple[wp.vec3, wp.vec3]:
    a = wp.dot(g1, g1)
    b = wp.dot(g1, g2)
    c = wp.dot(g2, g2)

    det_G = a * c - b * b
    det_Gc = wp.max(det_G, 1.0e-12)
    Ja = wp.sqrt(det_Gc)
    Ja_inv = 1.0 / Ja

    I1bar = (a + c) * Ja_inv

    dJa_dg1 = (g1 * c - g2 * b) * Ja_inv
    dJa_dg2 = (g2 * a - g1 * b) * Ja_inv

    P1 = (g1 * Ja_inv - dJa_dg1 * (0.5 * I1bar * Ja_inv)) * mu + dg1 * damp
    P2 = (g2 * Ja_inv - dJa_dg2 * (0.5 * I1bar * Ja_inv)) * mu + dg2 * damp
    return P1, P2


# -- unified 2-D dispatcher -----------------------------------------------

@wp.func
def pk1_stress_2d(
    law: int,
    g1: wp.vec3, g2: wp.vec3,
    dg1: wp.vec3, dg2: wp.vec3,
    mu: float, damp: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Dispatch PK1 stress computation for a 2-D (membrane) element."""
    if law == 0:
        return _pk1_rest_stable_2d(g1, g2, dg1, dg2, mu, damp)
    elif law == 1:
        return _pk1_isochoric_2d(g1, g2, dg1, dg2, mu, damp)
    else:
        return _pk1_rest_stable_2d(g1, g2, dg1, dg2, mu, damp)
