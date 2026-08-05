# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P5 FullF new-interface donor averaging and equilibrium initialization."""

from __future__ import annotations

import warp as wp

from ....solver.kernels.common import (
    direction_x,
    direction_y,
    direction_z,
    equilibrium_population,
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
def prepare_new_interface_donors_kernel(
    density_out: wp.array3d(dtype=float),
    velocity_x_out: wp.array3d(dtype=float),
    velocity_y_out: wp.array3d(dtype=float),
    velocity_z_out: wp.array3d(dtype=float),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    final_type: wp.array3d(dtype=wp.uint8),
    new_interface: wp.array3d(dtype=wp.uint8),
    donor_count: wp.array3d(dtype=wp.uint8),
    rho_init: wp.array3d(dtype=float),
    ux_init: wp.array3d(dtype=float),
    uy_init: wp.array3d(dtype=float),
    uz_init: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Average provisional post-collision fields from old/final active donors."""

    i, j, k = wp.tid()
    count = int(0)
    rho_sum = float(0.0)
    ux_sum = float(0.0)
    uy_sum = float(0.0)
    uz_sum = float(0.0)

    if new_interface[i, j, k] != wp.uint8(0):
        for q in range(1, 19):
            ni = i - direction_x(q)
            nj = j - direction_y(q)
            nk = k - direction_z(q)
            outside = bool(False)
            if ni < 0 or ni >= nx:
                if periodic_x != 0:
                    ni = _wrap_once(ni, nx)
                else:
                    outside = True
            if nj < 0 or nj >= ny:
                if periodic_y != 0:
                    nj = _wrap_once(nj, ny)
                else:
                    outside = True
            if nk < 0 or nk >= nz:
                if periodic_z != 0:
                    nk = _wrap_once(nk, nz)
                else:
                    outside = True
            if (
                not outside
                and cell_type_n[ni, nj, nk] != wp.uint8(0)
                and final_type[ni, nj, nk] != wp.uint8(0)
            ):
                count += 1
                rho_sum += density_out[ni, nj, nk]
                ux_sum += velocity_x_out[ni, nj, nk]
                uy_sum += velocity_y_out[ni, nj, nk]
                uz_sum += velocity_z_out[ni, nj, nk]

    inverse_count = float(0.0)
    if count > 0:
        inverse_count = 1.0 / float(count)
    donor_count[i, j, k] = wp.uint8(count)
    rho_init[i, j, k] = rho_sum * inverse_count
    ux_init[i, j, k] = ux_sum * inverse_count
    uy_init[i, j, k] = uy_sum * inverse_count
    uz_init[i, j, k] = uz_sum * inverse_count


@wp.kernel
def initialize_new_interface_fullf_kernel(
    new_interface: wp.array3d(dtype=wp.uint8),
    mass_final: wp.array3d(dtype=float),
    phi_final: wp.array3d(dtype=float),
    rho_init: wp.array3d(dtype=float),
    ux_init: wp.array3d(dtype=float),
    uy_init: wp.array3d(dtype=float),
    uz_init: wp.array3d(dtype=float),
    f_post_out: wp.array(dtype=float),
    density_out: wp.array3d(dtype=float),
    velocity_x_out: wp.array3d(dtype=float),
    velocity_y_out: wp.array3d(dtype=float),
    velocity_z_out: wp.array3d(dtype=float),
    force_x_out: wp.array3d(dtype=float),
    force_y_out: wp.array3d(dtype=float),
    force_z_out: wp.array3d(dtype=float),
    gravity_x: float,
    gravity_y: float,
    gravity_z: float,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Write equilibrium FullF and consistent macro/VOF fields for new cells."""

    i, j, k = wp.tid()
    if new_interface[i, j, k] == wp.uint8(0):
        return

    rho = rho_init[i, j, k]
    ux = ux_init[i, j, k]
    uy = uy_init[i, j, k]
    uz = uz_init[i, j, k]
    local_index = i * ny * nz + j * nz + k
    for q in range(19):
        f_post_out[q * stride + local_index] = equilibrium_population(
            q,
            rho,
            ux,
            uy,
            uz,
        )

    density_out[i, j, k] = rho
    velocity_x_out[i, j, k] = ux
    velocity_y_out[i, j, k] = uy
    velocity_z_out[i, j, k] = uz
    force_x_out[i, j, k] = rho * gravity_x
    force_y_out[i, j, k] = rho * gravity_y
    force_z_out[i, j, k] = rho * gravity_z
    phi_final[i, j, k] = mass_final[i, j, k] / rho


@wp.kernel
def initialize_new_interface_home_kernel(
    new_interface: wp.array3d(dtype=wp.uint8),
    mass_final: wp.array3d(dtype=float),
    phi_final: wp.array3d(dtype=float),
    rho_init: wp.array3d(dtype=float),
    ux_init: wp.array3d(dtype=float),
    uy_init: wp.array3d(dtype=float),
    uz_init: wp.array3d(dtype=float),
    rho_out: wp.array3d(dtype=float),
    jx_out: wp.array3d(dtype=float),
    jy_out: wp.array3d(dtype=float),
    jz_out: wp.array3d(dtype=float),
    sxx_out: wp.array3d(dtype=float),
    syy_out: wp.array3d(dtype=float),
    szz_out: wp.array3d(dtype=float),
    sxy_out: wp.array3d(dtype=float),
    sxz_out: wp.array3d(dtype=float),
    syz_out: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
    velocity_x_out: wp.array3d(dtype=float),
    velocity_y_out: wp.array3d(dtype=float),
    velocity_z_out: wp.array3d(dtype=float),
    force_x_out: wp.array3d(dtype=float),
    force_y_out: wp.array3d(dtype=float),
    force_z_out: wp.array3d(dtype=float),
    gravity_x: float,
    gravity_y: float,
    gravity_z: float,
) -> None:
    """Write equilibrium HOME moments and consistent macro/VOF fields."""

    i, j, k = wp.tid()
    if new_interface[i, j, k] == wp.uint8(0):
        return
    rho = rho_init[i, j, k]
    ux = ux_init[i, j, k]
    uy = uy_init[i, j, k]
    uz = uz_init[i, j, k]
    rho_out[i, j, k] = rho
    jx_out[i, j, k] = rho * ux
    jy_out[i, j, k] = rho * uy
    jz_out[i, j, k] = rho * uz
    sxx_out[i, j, k] = rho * ux * ux
    syy_out[i, j, k] = rho * uy * uy
    szz_out[i, j, k] = rho * uz * uz
    sxy_out[i, j, k] = rho * ux * uy
    sxz_out[i, j, k] = rho * ux * uz
    syz_out[i, j, k] = rho * uy * uz
    density_out[i, j, k] = rho
    velocity_x_out[i, j, k] = ux
    velocity_y_out[i, j, k] = uy
    velocity_z_out[i, j, k] = uz
    force_x_out[i, j, k] = rho * gravity_x
    force_y_out[i, j, k] = rho * gravity_y
    force_z_out[i, j, k] = rho * gravity_z
    phi_final[i, j, k] = mass_final[i, j, k] / rho


__all__ = [
    "initialize_new_interface_fullf_kernel",
    "initialize_new_interface_home_kernel",
    "prepare_new_interface_donors_kernel",
]
