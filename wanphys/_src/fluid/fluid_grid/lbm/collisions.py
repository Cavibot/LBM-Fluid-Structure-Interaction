# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Population- and encoded-moment collision kernels and backend metadata.

``SrtCollision`` / ``TrtCollision`` / ``HomeNocmMrtCollision`` describe which
input form and relaxation rates the solver should use.  They are not
independent collision executors: ``LbmSolver`` still launches the Warp
kernels below.
"""

from __future__ import annotations

import warp as wp

from .encoding import equilibrium_population, opposite_direction
from .model import LbmModel


@wp.kernel
def epc_collision_kernel(
    f_star: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    f_post: wp.array(dtype=float),
    omega_even: float,
    omega_odd: float,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    r, vx, vy, vz = rho[i, j, k], ux[i, j, k], uy[i, j, k], uz[i, j, k]
    for q in range(19):
        qo = opposite_direction(q)
        fq, fo = f_star[q * stride + idx], f_star[qo * stride + idx]
        eq = equilibrium_population(q, r, vx, vy, vz)
        eqo = equilibrium_population(qo, r, vx, vy, vz)
        even_delta = 0.5 * ((fq + fo) - (eq + eqo))
        odd_delta = 0.5 * ((fq - fo) - (eq - eqo))
        f_post[q * stride + idx] = fq - omega_even * even_delta - omega_odd * odd_delta


@wp.kernel
def home_nocm_collision_kernel(
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float),
    jy: wp.array3d(dtype=float),
    jz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
    out_rho: wp.array3d(dtype=float),
    out_jx: wp.array3d(dtype=float),
    out_jy: wp.array3d(dtype=float),
    out_jz: wp.array3d(dtype=float),
    out_sxx: wp.array3d(dtype=float),
    out_syy: wp.array3d(dtype=float),
    out_szz: wp.array3d(dtype=float),
    out_sxy: wp.array3d(dtype=float),
    out_sxz: wp.array3d(dtype=float),
    out_syz: wp.array3d(dtype=float),
    omega: float,
) -> None:
    """Null-force 3D HOME/NOCM closed-form second-moment update.

    This is the density-weighted form of Home-FSLBM's
    ``mlGetPIAfterCollision``.  Conserved density/momentum pass through and
    the six retained second Hermite coefficients are relaxed in closed form.
    """
    i, j, k = wp.tid()
    r = wp.max(rho[i, j, k], 1.0e-12)
    mx, my, mz = jx[i, j, k], jy[i, j, k], jz[i, j, k]
    ux, uy, uz = mx / r, my / r, mz / r
    pxx, pyy, pzz = sxx[i, j, k] + r / 3.0, syy[i, j, k] + r / 3.0, szz[i, j, k] + r / 3.0
    trace = (pxx + pyy + pzz) / 3.0
    dev_xx, dev_yy, dev_zz = pxx - trace, pyy - trace, pzz - trace
    u2trace = r * (ux * ux + uy * uy + uz * uz) / 3.0
    post_pxx = r / 3.0 + (1.0 - omega) * dev_xx + u2trace + omega * r * (2.0 * ux * ux - uy * uy - uz * uz) / 3.0
    post_pyy = r / 3.0 + (1.0 - omega) * dev_yy + u2trace + omega * r * (-ux * ux + 2.0 * uy * uy - uz * uz) / 3.0
    post_pzz = r / 3.0 + (1.0 - omega) * dev_zz + u2trace + omega * r * (-ux * ux - uy * uy + 2.0 * uz * uz) / 3.0

    out_rho[i, j, k] = r
    out_jx[i, j, k] = mx
    out_jy[i, j, k] = my
    out_jz[i, j, k] = mz
    out_sxx[i, j, k] = post_pxx - r / 3.0
    out_syy[i, j, k] = post_pyy - r / 3.0
    out_szz[i, j, k] = post_pzz - r / 3.0
    out_sxy[i, j, k] = (1.0 - omega) * sxy[i, j, k] + omega * r * ux * uy
    out_sxz[i, j, k] = (1.0 - omega) * sxz[i, j, k] + omega * r * ux * uz
    out_syz[i, j, k] = (1.0 - omega) * syz[i, j, k] + omega * r * uy * uz


class PopulationCollisionBackend:
    """Common EPC contract implemented by population collision families."""

    input_kind = "population"


class SrtCollision(PopulationCollisionBackend):
    @staticmethod
    def relaxation_rates(model: LbmModel) -> tuple[float, float]:
        omega = float(model.omega)
        return omega, omega


class TrtCollision(PopulationCollisionBackend):
    @staticmethod
    def relaxation_rates(model: LbmModel) -> tuple[float, float]:
        return float(model.omega_plus), float(model.omega_minus)


class HomeNocmMrtCollision:
    input_kind = "moments"

    @staticmethod
    def relaxation_rates(model: LbmModel) -> tuple[float, float]:
        del model
        raise TypeError("HOME-NOCM uses an encoded-moment relaxation profile")
