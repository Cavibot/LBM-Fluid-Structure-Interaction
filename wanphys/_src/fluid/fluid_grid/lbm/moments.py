# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 raw- and non-orthogonal-central-moment transforms.

The implementation intentionally uses flat device arrays for the 19 moments.
That keeps the collision basis explicit and avoids hiding FullF MRT behind the
ten-field HOME closure.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import warp as wp

from .constants import CX, CY, CZ
from .encoding import (
    direction_weight,
    direction_x,
    direction_y,
    direction_z,
    equilibrium_population,
)


# A complete, invertible monomial basis on D3Q19.  The first ten entries are
# density, momentum and the symmetric second-order tensor retained by HOME.
MOMENT_EXPONENTS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (0, 2, 0), (0, 0, 2),
    (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (2, 1, 0), (2, 0, 1), (1, 2, 0),
    (0, 2, 1), (1, 0, 2), (0, 1, 2),
    (2, 2, 0), (2, 0, 2), (0, 2, 2),
)


@lru_cache(maxsize=1)
def host_moment_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return the raw-moment transform and its inverse as float32 arrays."""

    transform = np.empty((19, 19), dtype=np.float64)
    for a, (px, py, pz) in enumerate(MOMENT_EXPONENTS):
        for q, (cx, cy, cz) in enumerate(zip(CX, CY, CZ)):
            transform[a, q] = (cx ** px) * (cy ** py) * (cz ** pz)
    inverse = np.linalg.inv(transform)
    return transform.astype(np.float32), inverse.astype(np.float32)


def host_relaxation_rates(
    omega_shear: float,
    omega_third: float,
    omega_fourth: float,
) -> np.ndarray:
    """Build the diagonal MRT profile in the declared monomial basis."""

    rates = np.zeros(19, dtype=np.float32)
    rates[4:10] = float(omega_shear)
    rates[10:16] = float(omega_third)
    rates[16:19] = float(omega_fourth)
    return rates


@wp.func
def _moment_px(a: int) -> int:
    value = 0
    if a == 1 or a == 7 or a == 8 or a == 12 or a == 14:
        value = 1
    elif a == 4 or a == 10 or a == 11 or a == 16 or a == 17:
        value = 2
    return value


@wp.func
def _moment_py(a: int) -> int:
    value = 0
    if a == 2 or a == 7 or a == 9 or a == 10 or a == 15:
        value = 1
    elif a == 5 or a == 12 or a == 13 or a == 16 or a == 18:
        value = 2
    return value


@wp.func
def _moment_pz(a: int) -> int:
    value = 0
    if a == 3 or a == 8 or a == 9 or a == 11 or a == 13:
        value = 1
    elif a == 6 or a == 14 or a == 15 or a == 17 or a == 18:
        value = 2
    return value


@wp.func
def _pow_0_2(value: float, exponent: int) -> float:
    result = 1.0
    if exponent == 1:
        result = value
    elif exponent == 2:
        result = value * value
    return result


@wp.func
def _int_pow_0_2(value: int, exponent: int) -> float:
    result = 1.0
    if exponent == 1:
        result = float(value)
    elif exponent == 2:
        result = float(value * value)
    return result


@wp.func
def _binomial_0_2(n: int, k: int) -> float:
    value = 0.0
    if k >= 0 and k <= n:
        value = 1.0
        if n == 2 and k == 1:
            value = 2.0
    return value


@wp.func
def _monomial(a: int, q: int) -> float:
    return (
        _int_pow_0_2(direction_x(q), _moment_px(a))
        * _int_pow_0_2(direction_y(q), _moment_py(a))
        * _int_pow_0_2(direction_z(q), _moment_pz(a))
    )


@wp.func
def _shift_coefficient(
    target: int,
    source: int,
    sx: float,
    sy: float,
    sz: float,
) -> float:
    tx, ty, tz = _moment_px(target), _moment_py(target), _moment_pz(target)
    px, py, pz = _moment_px(source), _moment_py(source), _moment_pz(source)
    coefficient = 0.0
    if px <= tx and py <= ty and pz <= tz:
        coefficient = (
            _binomial_0_2(tx, px)
            * _binomial_0_2(ty, py)
            * _binomial_0_2(tz, pz)
            * _pow_0_2(sx, tx - px)
            * _pow_0_2(sy, ty - py)
            * _pow_0_2(sz, tz - pz)
        )
    return coefficient


@wp.func
def guo_population_source(
    q: int,
    ux: float,
    uy: float,
    uz: float,
    fx: float,
    fy: float,
    fz: float,
) -> float:
    cx, cy, cz = float(direction_x(q)), float(direction_y(q)), float(direction_z(q))
    cu = cx * ux + cy * uy + cz * uz
    c_force = cx * fx + cy * fy + cz * fz
    return direction_weight(q) * (
        3.0 * ((cx - ux) * fx + (cy - uy) * fy + (cz - uz) * fz)
        + 9.0 * cu * c_force
    )


@wp.func
def _central_equilibrium(a: int, rho: float) -> float:
    value = 0.0
    if a == 0:
        value = rho
    elif a == 4 or a == 5 or a == 6:
        value = rho / 3.0
    elif a == 16 or a == 17 or a == 18:
        value = rho / 9.0
    return value


@wp.kernel
def populations_to_raw_moments_kernel(
    f: wp.array(dtype=float),
    raw: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    for a in range(19):
        value = float(0.0)
        for q in range(19):
            value += _monomial(a, q) * f[q * stride + idx]
        raw[a * stride + idx] = value


@wp.kernel
def raw_moments_to_populations_kernel(
    raw: wp.array(dtype=float),
    inverse_transform: wp.array(dtype=float),
    f: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    for q in range(19):
        value = float(0.0)
        for a in range(19):
            value += inverse_transform[q * 19 + a] * raw[a * stride + idx]
        f[q * stride + idx] = value


@wp.kernel
def guo_source_to_raw_moments_kernel(
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    raw_source: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Translate the population Guo source into the fixed raw basis once."""

    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    vx, vy, vz = ux[i, j, k], uy[i, j, k], uz[i, j, k]
    force_x, force_y, force_z = fx[i, j, k], fy[i, j, k], fz[i, j, k]
    for a in range(19):
        value = float(0.0)
        for q in range(19):
            value += _monomial(a, q) * guo_population_source(
                q, vx, vy, vz, force_x, force_y, force_z
            )
        raw_source[a * stride + idx] = value


@wp.kernel
def raw_mrt_collision_kernel(
    raw: wp.array(dtype=float),
    raw_source: wp.array(dtype=float),
    post_raw: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    rates: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    r = rho[i, j, k]
    vx, vy, vz = ux[i, j, k], uy[i, j, k], uz[i, j, k]
    force_x, force_y, force_z = fx[i, j, k], fy[i, j, k], fz[i, j, k]
    for a in range(19):
        equilibrium = float(0.0)
        for q in range(19):
            basis = _monomial(a, q)
            equilibrium += basis * equilibrium_population(q, r, vx, vy, vz)
        source = raw_source[a * stride + idx]
        current = raw[a * stride + idx]
        rate = rates[a]
        value = current - rate * (current - equilibrium) + (1.0 - 0.5 * rate) * source
        if a == 0:
            value = current
        elif a == 1:
            value = current + force_x
        elif a == 2:
            value = current + force_y
        elif a == 3:
            value = current + force_z
        post_raw[a * stride + idx] = value


@wp.kernel
def nocm_collision_kernel(
    raw: wp.array(dtype=float),
    raw_source: wp.array(dtype=float),
    post_central: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    rates: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    r = rho[i, j, k]
    vx, vy, vz = ux[i, j, k], uy[i, j, k], uz[i, j, k]
    force_x, force_y, force_z = fx[i, j, k], fy[i, j, k], fz[i, j, k]

    for a in range(19):
        central = float(0.0)
        central_source = float(0.0)
        for b in range(19):
            coefficient = _shift_coefficient(a, b, -vx, -vy, -vz)
            if coefficient != 0.0:
                central += coefficient * raw[b * stride + idx]
                central_source += coefficient * raw_source[b * stride + idx]

        rate = rates[a]
        equilibrium = _central_equilibrium(a, r)
        value = central - rate * (central - equilibrium) + (1.0 - 0.5 * rate) * central_source
        if a == 0:
            value = central
        elif a == 1:
            value = central + force_x
        elif a == 2:
            value = central + force_y
        elif a == 3:
            value = central + force_z
        post_central[a * stride + idx] = value


@wp.kernel
def central_to_raw_moments_kernel(
    central: wp.array(dtype=float),
    raw: wp.array(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    vx, vy, vz = ux[i, j, k], uy[i, j, k], uz[i, j, k]
    for a in range(19):
        value = float(0.0)
        for b in range(19):
            value += _shift_coefficient(a, b, vx, vy, vz) * central[b * stride + idx]
        raw[a * stride + idx] = value
