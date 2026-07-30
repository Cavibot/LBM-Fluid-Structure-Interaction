# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P3 FullF free-surface population completion and GAS-state isolation."""

from __future__ import annotations

import warp as wp

from ..encoding import (
    direction_x,
    direction_y,
    direction_z,
    equilibrium_population,
    opposite_direction,
)


@wp.func
def _wrap_once(value: int, size: int) -> int:
    mapped = value
    if mapped < 0:
        mapped += size
    elif mapped >= size:
        mapped -= size
    return mapped


@wp.kernel
def complete_gas_to_interface_fullf_kernel(
    f_post_n: wp.array(dtype=float),
    velocity_x_n: wp.array3d(dtype=float),
    velocity_y_n: wp.array3d(dtype=float),
    velocity_z_n: wp.array3d(dtype=float),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    curvature_n: wp.array3d(dtype=float),
    f_star: wp.array(dtype=float),
    atmosphere_density: float,
    atmosphere_pressure: float,
    surface_tension: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Overwrite gas-to-interface pull links with paper Eq. (11)."""

    i, j, k = wp.tid()
    if cell_type_n[i, j, k] != wp.uint8(1):
        return

    local_index = i * ny * nz + j * nz + k
    ux = velocity_x_n[i, j, k]
    uy = velocity_y_n[i, j, k]
    uz = velocity_z_n[i, j, k]
    rho_g = atmosphere_density
    if surface_tension != 0.0:
        rho_g = 3.0 * (
            atmosphere_pressure
            - 2.0 * surface_tension * curvature_n[i, j, k]
        )
    for q in range(1, 19):
        si = i - direction_x(q)
        sj = j - direction_y(q)
        sk = k - direction_z(q)
        outside = bool(False)
        if si < 0 or si >= nx:
            if periodic_x != 0:
                si = _wrap_once(si, nx)
            else:
                outside = True
        if sj < 0 or sj >= ny:
            if periodic_y != 0:
                sj = _wrap_once(sj, ny)
            else:
                outside = True
        if sk < 0 or sk >= nz:
            if periodic_z != 0:
                sk = _wrap_once(sk, nz)
            else:
                outside = True

        if not outside and cell_type_n[si, sj, sk] == wp.uint8(0):
            opposite = opposite_direction(q)
            f_star[q * stride + local_index] = (
                equilibrium_population(q, rho_g, ux, uy, uz)
                + equilibrium_population(opposite, rho_g, ux, uy, uz)
                - f_post_n[opposite * stride + local_index]
            )


@wp.kernel
def restore_gas_fullf_state_kernel(
    f_post_n: wp.array(dtype=float),
    density_n: wp.array3d(dtype=float),
    velocity_x_n: wp.array3d(dtype=float),
    velocity_y_n: wp.array3d(dtype=float),
    velocity_z_n: wp.array3d(dtype=float),
    force_x_n: wp.array3d(dtype=float),
    force_y_n: wp.array3d(dtype=float),
    force_z_n: wp.array3d(dtype=float),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    f_post_out: wp.array(dtype=float),
    density_out: wp.array3d(dtype=float),
    velocity_x_out: wp.array3d(dtype=float),
    velocity_y_out: wp.array3d(dtype=float),
    velocity_z_out: wp.array3d(dtype=float),
    force_x_out: wp.array3d(dtype=float),
    force_y_out: wp.array3d(dtype=float),
    force_z_out: wp.array3d(dtype=float),
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Preserve semantically invalid GAS kinetic/macroscopic storage."""

    i, j, k = wp.tid()
    if cell_type_n[i, j, k] != wp.uint8(0):
        return
    local_index = i * ny * nz + j * nz + k
    for q in range(19):
        f_post_out[q * stride + local_index] = f_post_n[
            q * stride + local_index
        ]
    density_out[i, j, k] = density_n[i, j, k]
    velocity_x_out[i, j, k] = velocity_x_n[i, j, k]
    velocity_y_out[i, j, k] = velocity_y_n[i, j, k]
    velocity_z_out[i, j, k] = velocity_z_n[i, j, k]
    force_x_out[i, j, k] = force_x_n[i, j, k]
    force_y_out[i, j, k] = force_y_n[i, j, k]
    force_z_out[i, j, k] = force_z_n[i, j, k]


__all__ = [
    "complete_gas_to_interface_fullf_kernel",
    "restore_gas_fullf_state_kernel",
]
