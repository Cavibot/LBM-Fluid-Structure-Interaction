# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 fixed-topology mass advection for authoritative VOF."""

from __future__ import annotations

import warp as wp

from ..encoding import (
    direction_x,
    direction_y,
    direction_z,
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
def advect_vof_mass_fullf_fixed_topology_kernel(
    f_post: wp.array(dtype=float),
    density_n: wp.array3d(dtype=float),
    mass_n: wp.array3d(dtype=float),
    phi_n: wp.array3d(dtype=float),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    mass_tmp: wp.array3d(dtype=float),
    phi_tmp: wp.array3d(dtype=float),
    mass_delta: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Apply the frozen FSLBM neighbour-type mass exchange scheme.

    The center cell receives direction ``q`` from ``x-c_q`` and sends the
    opposite population toward that source cell.  Interface-gas and all
    out-of-domain links carry zero VOF mass.  Consequently this kernel never
    reads persistent populations from GAS cells.
    """

    i, j, k = wp.tid()
    center_type = cell_type_n[i, j, k]
    center_mass = mass_n[i, j, k]
    delta = float(0.0)
    center_index = i * ny * nz + j * nz + k

    if center_type != wp.uint8(0):
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

            if not outside:
                source_type = cell_type_n[si, sj, sk]
                weight = float(0.0)
                if center_type == wp.uint8(2):
                    if source_type != wp.uint8(0):
                        weight = 1.0
                elif source_type == wp.uint8(2):
                    weight = 1.0
                elif source_type == wp.uint8(1):
                    weight = 0.5 * (
                        phi_n[i, j, k] + phi_n[si, sj, sk]
                    )

                if weight != 0.0:
                    source_index = si * ny * nz + sj * nz + sk
                    incoming = f_post[q * stride + source_index]
                    outgoing = f_post[
                        opposite_direction(q) * stride + center_index
                    ]
                    delta += weight * (incoming - outgoing)

    updated_mass = center_mass + delta
    mass_delta[i, j, k] = delta
    mass_tmp[i, j, k] = updated_mass
    phi_tmp[i, j, k] = updated_mass / density_n[i, j, k]


__all__ = ["advect_vof_mass_fullf_fixed_topology_kernel"]
