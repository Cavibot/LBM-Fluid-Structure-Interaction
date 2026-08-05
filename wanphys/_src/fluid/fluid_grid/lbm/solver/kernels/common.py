"""Shared device functions used by multiple LBM stages."""

import warp as wp


@wp.func
def direction_x(q: int) -> int:
    value = 0
    if q == 1 or q == 7 or q == 9 or q == 11 or q == 13:
        value = 1
    elif q == 2 or q == 8 or q == 10 or q == 12 or q == 14:
        value = -1
    return value


@wp.func
def direction_y(q: int) -> int:
    value = 0
    if q == 3 or q == 7 or q == 8 or q == 15 or q == 17:
        value = 1
    elif q == 4 or q == 9 or q == 10 or q == 16 or q == 18:
        value = -1
    return value


@wp.func
def direction_z(q: int) -> int:
    value = 0
    if q == 5 or q == 11 or q == 12 or q == 15 or q == 16:
        value = 1
    elif q == 6 or q == 13 or q == 14 or q == 17 or q == 18:
        value = -1
    return value


@wp.func
def direction_weight(q: int) -> float:
    value = 1.0 / 36.0
    if q == 0:
        value = 1.0 / 3.0
    elif q <= 6:
        value = 1.0 / 18.0
    return value


@wp.func
def opposite_direction(q: int) -> int:
    value = 0
    if q == 1:
        value = 2
    elif q == 2:
        value = 1
    elif q == 3:
        value = 4
    elif q == 4:
        value = 3
    elif q == 5:
        value = 6
    elif q == 6:
        value = 5
    elif q == 7:
        value = 10
    elif q == 8:
        value = 9
    elif q == 9:
        value = 8
    elif q == 10:
        value = 7
    elif q == 11:
        value = 14
    elif q == 12:
        value = 13
    elif q == 13:
        value = 12
    elif q == 14:
        value = 11
    elif q == 15:
        value = 18
    elif q == 16:
        value = 17
    elif q == 17:
        value = 16
    elif q == 18:
        value = 15
    return value


@wp.func
def equilibrium_population(
    q: int, rho: float, ux: float, uy: float, uz: float
) -> float:
    cx, cy, cz = direction_x(q), direction_y(q), direction_z(q)
    cu = float(cx) * ux + float(cy) * uy + float(cz) * uz
    u2 = ux * ux + uy * uy + uz * uz
    return direction_weight(q) * rho * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
    )


@wp.func
def equilibrium_f(
    weight: float,
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    cx: int,
    cy: int,
    cz: int,
) -> float:
    """D3Q19 equilibrium helper shared by boundary-stage kernels."""

    cu = float(cx) * ux + float(cy) * uy + float(cz) * uz
    u_sq = ux * ux + uy * uy + uz * uz
    return weight * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)

__all__ = [
    "direction_weight",
    "direction_x",
    "direction_y",
    "direction_z",
    "equilibrium_population",
    "opposite_direction",
    "equilibrium_f",
]
