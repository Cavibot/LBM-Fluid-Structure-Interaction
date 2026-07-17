# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Population/HOME collision kernels and collision-space backend metadata.

FullF Raw MRT and FullF NOCM-MRT transforms live in :mod:`moments`; this module
owns SRT/TRT population injection and the retained-moment HOME NOCM closure.
"""

from __future__ import annotations

import warp as wp

from .contracts import CollisionSpace
from .encoding import equilibrium_population, opposite_direction
from .model import LbmModel
from .moments import guo_population_source


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
def epc_forced_collision_kernel(
    f_star: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    f_post: wp.array(dtype=float),
    omega_even: float,
    omega_odd: float,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """SRT/TRT collision with the force translated in even/odd space."""

    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    r, vx, vy, vz = rho[i, j, k], ux[i, j, k], uy[i, j, k], uz[i, j, k]
    force_x, force_y, force_z = fx[i, j, k], fy[i, j, k], fz[i, j, k]
    for q in range(19):
        qo = opposite_direction(q)
        fq, fo = f_star[q * stride + idx], f_star[qo * stride + idx]
        eq = equilibrium_population(q, r, vx, vy, vz)
        eqo = equilibrium_population(qo, r, vx, vy, vz)
        even_delta = 0.5 * ((fq + fo) - (eq + eqo))
        odd_delta = 0.5 * ((fq - fo) - (eq - eqo))
        source_q = guo_population_source(q, vx, vy, vz, force_x, force_y, force_z)
        source_o = guo_population_source(qo, vx, vy, vz, force_x, force_y, force_z)
        source_even = 0.5 * (source_q + source_o)
        source_odd = 0.5 * (source_q - source_o)
        f_post[q * stride + idx] = (
            fq
            - omega_even * even_delta
            - omega_odd * odd_delta
            + (1.0 - 0.5 * omega_even) * source_even
            + (1.0 - 0.5 * omega_odd) * source_odd
        )


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


@wp.kernel
def home_nocm_mrt_collision_kernel(
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
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
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
    """Canonical HOME NOCM-MRT collision with central-moment forcing."""

    i, j, k = wp.tid()
    r = wp.max(rho[i, j, k], 1.0e-12)
    mx, my, mz = jx[i, j, k], jy[i, j, k], jz[i, j, k]
    vx, vy, vz = ux[i, j, k], uy[i, j, k], uz[i, j, k]
    force_x, force_y, force_z = fx[i, j, k], fy[i, j, k], fz[i, j, k]

    pxx, pyy, pzz = sxx[i, j, k] + r / 3.0, syy[i, j, k] + r / 3.0, szz[i, j, k] + r / 3.0
    pxy, pxz, pyz = sxy[i, j, k], sxz[i, j, k], syz[i, j, k]

    kx, ky, kz = mx - r * vx, my - r * vy, mz - r * vz
    kxx = pxx - 2.0 * vx * mx + r * vx * vx
    kyy = pyy - 2.0 * vy * my + r * vy * vy
    kzz = pzz - 2.0 * vz * mz + r * vz * vz
    kxy = pxy - vx * my - vy * mx + r * vx * vy
    kxz = pxz - vx * mz - vz * mx + r * vx * vz
    kyz = pyz - vy * mz - vz * my + r * vy * vz

    post_kx, post_ky, post_kz = kx + force_x, ky + force_y, kz + force_z
    post_kxx = kxx - omega * (kxx - r / 3.0)
    post_kyy = kyy - omega * (kyy - r / 3.0)
    post_kzz = kzz - omega * (kzz - r / 3.0)
    post_kxy = (1.0 - omega) * kxy
    post_kxz = (1.0 - omega) * kxz
    post_kyz = (1.0 - omega) * kyz

    post_jx = post_kx + r * vx
    post_jy = post_ky + r * vy
    post_jz = post_kz + r * vz
    post_pxx = post_kxx + 2.0 * vx * post_kx + r * vx * vx
    post_pyy = post_kyy + 2.0 * vy * post_ky + r * vy * vy
    post_pzz = post_kzz + 2.0 * vz * post_kz + r * vz * vz
    post_pxy = post_kxy + vx * post_ky + vy * post_kx + r * vx * vy
    post_pxz = post_kxz + vx * post_kz + vz * post_kx + r * vx * vz
    post_pyz = post_kyz + vy * post_kz + vz * post_ky + r * vy * vz

    out_rho[i, j, k] = r
    out_jx[i, j, k] = post_jx
    out_jy[i, j, k] = post_jy
    out_jz[i, j, k] = post_jz
    out_sxx[i, j, k] = post_pxx - r / 3.0
    out_syy[i, j, k] = post_pyy - r / 3.0
    out_szz[i, j, k] = post_pzz - r / 3.0
    out_sxy[i, j, k] = post_pxy
    out_sxz[i, j, k] = post_pxz
    out_syz[i, j, k] = post_pyz


class PopulationCollisionBackend:
    """Common EPC contract implemented by population collision families."""

    input_kind = "population"
    collision_space = CollisionSpace.POPULATION


class SrtCollision(PopulationCollisionBackend):
    @staticmethod
    def relaxation_rates(model: LbmModel) -> tuple[float, float]:
        omega = float(model.omega)
        return omega, omega


class TrtCollision(PopulationCollisionBackend):
    collision_space = CollisionSpace.EVEN_ODD_POPULATION

    @staticmethod
    def relaxation_rates(model: LbmModel) -> tuple[float, float]:
        return float(model.omega_plus), float(model.omega_minus)


class HomeNocmMrtCollision:
    input_kind = "moments"
    collision_space = CollisionSpace.HOME_ENCODED_MOMENT

    @staticmethod
    def relaxation_rates(model: LbmModel) -> tuple[float, float]:
        del model
        raise TypeError("HOME-NOCM uses an encoded-moment relaxation profile")


class RawMrtCollision:
    input_kind = "raw_moments"
    collision_space = CollisionSpace.RAW_MOMENT


class FullNocmMrtCollision:
    input_kind = "nocm_moments"
    collision_space = CollisionSpace.NOCM_MOMENT
