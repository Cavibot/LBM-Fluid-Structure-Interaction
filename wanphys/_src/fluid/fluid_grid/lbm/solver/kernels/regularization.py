"""Regularized TRT kernels."""

from __future__ import annotations

import warp as wp

from .common import equilibrium_f

@wp.kernel
def reg_trt_kernel(
    f: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    omega_reg: float,     # blend: 1.0 = full reg, 0.0 = no-op (keep original f)
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Regularized non-equilibrium filter (pre-collision step).

    Projects the full non-equilibrium part of *f* onto the second-order
    Hermite basis (stress tensor) and reconstructs a regularized
    distribution.  The 2nd-order Hermite projection is purely even,
    which eliminates both even ghost modes (non-hydrodynamic stress) and
    odd ghost modes (spurious currents) simultaneously.

    The regularized distribution is blended with the original *f* using
    *omega_reg*:  ``f_out = omega_reg * f_reg + (1-omega_reg) * f_in``.

    No relaxation rates are applied here — relaxation is done by the
    subsequent TRT/BGK collide-stream kernel.

    .. note::

       Full regularization eliminates the odd non-equilibrium, so the
       TRT collision on the regularized fraction reduces to BGK-like
       relaxation with ``omega_plus``.  The TRT magic parameter still
       applies to the unregularized fraction when ``omega_reg < 1``.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    # ---- skip regularization near non-periodic domain-boundary cells (2 layers)
    # Bounce-back at walls relies on odd (momentum-carrying) non-equilibrium
    # in the distributions that stream to the wall cell from the neighbour.
    # Regularizing those neighbour cells damps the odd modes before they
    # reach the wall, weakening no-slip and causing systematic mass drift
    # when body forces (gravity / SC) are present.
    #
    # Periodic faces have no bounce-back wall to protect, so cells near
    # periodic boundaries can safely participate in regularization.
    if (
        (not px and i <= 1) or (not px and i >= nx - 2)
        or (not py and j <= 1) or (not py and j >= ny - 2)
        or (not pz and k <= 1) or (not pz and k >= nz - 2)
    ):
        return

    r = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]

    # ---- accumulate non-equilibrium stress tensor Pi_ab ---------------------
    # Pi_ab = Σ_d (f_d - f_d^eq) * c_da * c_db
    pixx = 0.0
    piyy = 0.0
    pizz = 0.0
    pixy = 0.0
    piyz = 0.0
    pixz = 0.0

    feq = 0.0
    fne = 0.0

    # d=0: rest (0,0,0) — contributes nothing to Pi
    # (c_a = 0 for all a, so all products are 0)

    # d=1: +x (1,0,0) w=1/18
    feq = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 1, 0, 0)
    fne = f[1 * stride + idx] - feq
    pixx += fne * 1.0  # cx^2 = 1

    # d=2: -x (-1,0,0) w=1/18
    feq = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, -1, 0, 0)
    fne = f[2 * stride + idx] - feq
    pixx += fne * 1.0  # cx^2 = 1

    # d=3: +y (0,1,0) w=1/18
    feq = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, 1, 0)
    fne = f[3 * stride + idx] - feq
    piyy += fne * 1.0  # cy^2 = 1

    # d=4: -y (0,-1,0) w=1/18
    feq = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, -1, 0)
    fne = f[4 * stride + idx] - feq
    piyy += fne * 1.0  # cy^2 = 1

    # d=5: +z (0,0,1) w=1/18
    feq = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, 0, 1)
    fne = f[5 * stride + idx] - feq
    pizz += fne * 1.0  # cz^2 = 1

    # d=6: -z (0,0,-1) w=1/18
    feq = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, 0, -1)
    fne = f[6 * stride + idx] - feq
    pizz += fne * 1.0  # cz^2 = 1

    # Edge directions (w=1/36, each contributes to multiple Pi components)
    # d=7: +x+y (1,1,0)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, 1, 0)
    fne = f[7 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * 1.0

    # d=8: -x+y (-1,1,0)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, 1, 0)
    fne = f[8 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * (-1.0)

    # d=9: +x-y (1,-1,0)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, -1, 0)
    fne = f[9 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * (-1.0)

    # d=10: -x-y (-1,-1,0)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, -1, 0)
    fne = f[10 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * 1.0

    # d=11: +x+z (1,0,1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, 0, 1)
    fne = f[11 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * 1.0

    # d=12: -x+z (-1,0,1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, 0, 1)
    fne = f[12 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * (-1.0)

    # d=13: +x-z (1,0,-1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, 0, -1)
    fne = f[13 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * (-1.0)

    # d=14: -x-z (-1,0,-1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, 0, -1)
    fne = f[14 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * 1.0

    # d=15: +y+z (0,1,1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, 1, 1)
    fne = f[15 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * 1.0

    # d=16: -y+z (0,-1,1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, -1, 1)
    fne = f[16 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * (-1.0)

    # d=17: +y-z (0,1,-1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, 1, -1)
    fne = f[17 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * (-1.0)

    # d=18: -y-z (0,-1,-1)
    feq = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, -1, -1)
    fne = f[18 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * 1.0

    # ---- reconstruct regularized even part from stress tensor ---------------
    # f_d^(neq,reg) = w_d * (9/2) * Q_dab * Pi_ab  (Hermite projection)
    fac_face = (1.0 / 18.0) * 4.5
    fac_edge = (1.0 / 36.0) * 4.5

    # Odd-preserving even-part blend.
    #
    # The 2nd-order Hermite projection is purely even (nr_d == nr_opp for
    # every direction pair).  Blending the full distribution:
    #     f_new = omega_reg * (feq + nr) + (1-omega_reg) * f
    # damps the odd (momentum-carrying) non-equilibrium by omega_reg.
    # At wall-adjacent cells this weakens bounce-back no-slip, causing
    # systematic mass drift in multiphase simulations with body forces.
    #
    # Instead we blend ONLY the even part while fully preserving the odd
    # part — both the equilibrium momentum and the shear non-equilibrium:
    #     inc = omega_reg * (nr + 0.5*(feq_d + feq_opp - f_d - f_opp))
    #     f_new_d = f_d + inc        (inc is the same for d and opp_d)
    #     f_new_opp = f_opp + inc
    # This preserves odd non-equilibrium (f_d - f_opp unchanged) while
    # still damping even ghost modes via the Hermite projection.
    # Boundary cells (2 layers from each face) skip regularization entirely
    # to avoid any contamination of bounce-back distributions.
    #
    # Mass conservation requires d=0 to be regularized:
    #   f_0^(neq,reg) = -1/2 * (pixx + piyy + pizz)

    # d=0: rest — no opposite direction, odd part is identically zero.
    feq0 = equilibrium_f(1.0 / 3.0, r, vx, vy, vz, 0, 0, 0)
    nr0 = -0.5 * (pixx + piyy + pizz)
    f[0 * stride + idx] = omega_reg * (feq0 + nr0) + (1.0 - omega_reg) * f[0 * stride + idx]

    # d=1 (+x), d=2 (-x): Q_xx=2/3, Q_yy=-1/3, Q_zz=-1/3
    feq1 = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 1, 0, 0)
    feq2 = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, -1, 0, 0)
    nr1 = fac_face * ((2.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz)
    _inc = omega_reg * (nr1 + 0.5 * (feq1 + feq2 - f[1 * stride + idx] - f[2 * stride + idx]))
    f[1 * stride + idx] = f[1 * stride + idx] + _inc
    f[2 * stride + idx] = f[2 * stride + idx] + _inc

    # d=3 (+y), d=4 (-y): Q_xx=-1/3, Q_yy=2/3, Q_zz=-1/3
    feq3 = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, 1, 0)
    feq4 = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, -1, 0)
    nr34 = fac_face * ((-1.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz)
    _inc = omega_reg * (nr34 + 0.5 * (feq3 + feq4 - f[3 * stride + idx] - f[4 * stride + idx]))
    f[3 * stride + idx] = f[3 * stride + idx] + _inc
    f[4 * stride + idx] = f[4 * stride + idx] + _inc

    # d=5 (+z), d=6 (-z): Q_xx=-1/3, Q_yy=-1/3, Q_zz=2/3
    feq5 = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, 0, 1)
    feq6 = equilibrium_f(1.0 / 18.0, r, vx, vy, vz, 0, 0, -1)
    nr56 = fac_face * ((-1.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (2.0 / 3.0) * pizz)
    _inc = omega_reg * (nr56 + 0.5 * (feq5 + feq6 - f[5 * stride + idx] - f[6 * stride + idx]))
    f[5 * stride + idx] = f[5 * stride + idx] + _inc
    f[6 * stride + idx] = f[6 * stride + idx] + _inc

    # d=7 (+x+y), d=10 (-x-y): Q_xx=2/3, Q_yy=2/3, Q_zz=-1/3, Q_xy=1
    feq7 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, 1, 0)
    feq10 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, -1, 0)
    nr7 = fac_edge * ((2.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz + 2.0 * pixy)
    _inc = omega_reg * (nr7 + 0.5 * (feq7 + feq10 - f[7 * stride + idx] - f[10 * stride + idx]))
    f[7 * stride + idx] = f[7 * stride + idx] + _inc
    f[10 * stride + idx] = f[10 * stride + idx] + _inc

    # d=8 (-x+y), d=9 (+x-y): Q_xx=2/3, Q_yy=2/3, Q_zz=-1/3, Q_xy=-1
    feq8 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, 1, 0)
    feq9 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, -1, 0)
    nr8 = fac_edge * ((2.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz + 2.0 * (-pixy))
    _inc = omega_reg * (nr8 + 0.5 * (feq8 + feq9 - f[8 * stride + idx] - f[9 * stride + idx]))
    f[8 * stride + idx] = f[8 * stride + idx] + _inc
    f[9 * stride + idx] = f[9 * stride + idx] + _inc

    # d=11 (+x+z), d=14 (-x-z): Q_xx=2/3, Q_yy=-1/3, Q_zz=2/3, Q_xz=1
    feq11 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, 0, 1)
    feq14 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, 0, -1)
    nr11 = fac_edge * ((2.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * pixz)
    _inc = omega_reg * (nr11 + 0.5 * (feq11 + feq14 - f[11 * stride + idx] - f[14 * stride + idx]))
    f[11 * stride + idx] = f[11 * stride + idx] + _inc
    f[14 * stride + idx] = f[14 * stride + idx] + _inc

    # d=12 (-x+z), d=13 (+x-z): Q_xx=2/3, Q_yy=-1/3, Q_zz=2/3, Q_xz=-1
    feq12 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, -1, 0, 1)
    feq13 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 1, 0, -1)
    nr12 = fac_edge * ((2.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * (-pixz))
    _inc = omega_reg * (nr12 + 0.5 * (feq12 + feq13 - f[12 * stride + idx] - f[13 * stride + idx]))
    f[12 * stride + idx] = f[12 * stride + idx] + _inc
    f[13 * stride + idx] = f[13 * stride + idx] + _inc

    # d=15 (+y+z), d=18 (-y-z): Q_xx=-1/3, Q_yy=2/3, Q_zz=2/3, Q_yz=1
    feq15 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, 1, 1)
    feq18 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, -1, -1)
    nr15 = fac_edge * ((-1.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * piyz)
    _inc = omega_reg * (nr15 + 0.5 * (feq15 + feq18 - f[15 * stride + idx] - f[18 * stride + idx]))
    f[15 * stride + idx] = f[15 * stride + idx] + _inc
    f[18 * stride + idx] = f[18 * stride + idx] + _inc

    # d=16 (-y+z), d=17 (+y-z): Q_xx=-1/3, Q_yy=2/3, Q_zz=2/3, Q_yz=-1
    feq16 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, -1, 1)
    feq17 = equilibrium_f(1.0 / 36.0, r, vx, vy, vz, 0, 1, -1)
    nr16 = fac_edge * ((-1.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * (-piyz))
    _inc = omega_reg * (nr16 + 0.5 * (feq16 + feq17 - f[16 * stride + idx] - f[17 * stride + idx]))
    f[16 * stride + idx] = f[16 * stride + idx] + _inc
    f[17 * stride + idx] = f[17 * stride + idx] + _inc
