# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multi-pass HOME-FREE topology kernels."""

from __future__ import annotations

import warp as wp


GAS = wp.constant(0)
INTERFACE = wp.constant(1)
LIQUID = wp.constant(2)
SOLID = wp.constant(3)
KEEP = wp.constant(0)
INTERFACE_TO_LIQUID = wp.constant(1)
INTERFACE_TO_GAS = wp.constant(2)
GAS_TO_INTERFACE = wp.constant(3)
LIQUID_TO_INTERFACE = wp.constant(4)


@wp.func
def _neighbor_coordinate(value: int, delta: int, size: int, periodic: int) -> int:
    result = value + delta
    if result < 0 or result >= size:
        if periodic != 0:
            result = (result + size) % size
        else:
            return -1
    return result


@wp.kernel
def mark_topology_candidates_kernel(
    moments: wp.array(dtype=float),
    advected_mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    fill_epsilon: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    candidates[i, j, k] = KEEP
    if flags[i, j, k] != INTERFACE:
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    transported = advected_mass[i, j, k]
    if not wp.isfinite(rho) or rho <= 0.0 or not wp.isfinite(transported):
        wp.atomic_add(invalid_cell_count, 0, 1)
        return

    has_liquid = int(0)
    has_gas = int(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                ni = _neighbor_coordinate(i, dx, nx, periodic_x)
                nj = _neighbor_coordinate(j, dy, ny, periodic_y)
                nk = _neighbor_coordinate(k, dz, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                neighbor_flag = flags[ni, nj, nk]
                if neighbor_flag == LIQUID:
                    has_liquid = 1
                elif neighbor_flag == GAS:
                    has_gas = 1
    if transported >= (1.0 + fill_epsilon) * rho or has_gas == 0:
        candidates[i, j, k] = INTERFACE_TO_LIQUID
    elif transported <= -fill_epsilon * rho or has_liquid == 0:
        candidates[i, j, k] = INTERFACE_TO_GAS


@wp.kernel
def protect_liquid_growth_kernel(
    flags: wp.array3d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.int32),
    protected_transitions: wp.array3d(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    intent = candidates[i, j, k]
    next_to_liquid_growth = int(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                ni = _neighbor_coordinate(i, dx, nx, periodic_x)
                nj = _neighbor_coordinate(j, dy, ny, periodic_y)
                nk = _neighbor_coordinate(k, dz, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if candidates[ni, nj, nk] == INTERFACE_TO_LIQUID:
                        next_to_liquid_growth = 1
    if flags[i, j, k] == GAS and next_to_liquid_growth != 0:
        intent = GAS_TO_INTERFACE
    elif intent == INTERFACE_TO_GAS and next_to_liquid_growth != 0:
        intent = KEEP
    protected_transitions[i, j, k] = intent


@wp.kernel
def protect_gas_growth_kernel(
    flags: wp.array3d(dtype=wp.int32),
    protected_transitions: wp.array3d(dtype=wp.int32),
    resolved: wp.array3d(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    intent = protected_transitions[i, j, k]
    next_to_gas_growth = int(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                ni = _neighbor_coordinate(i, dx, nx, periodic_x)
                nj = _neighbor_coordinate(j, dy, ny, periodic_y)
                nk = _neighbor_coordinate(k, dz, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if protected_transitions[ni, nj, nk] == INTERFACE_TO_GAS:
                        next_to_gas_growth = 1
    if next_to_gas_growth != 0:
        if flags[i, j, k] == LIQUID:
            intent = LIQUID_TO_INTERFACE
        elif intent == INTERFACE_TO_LIQUID:
            intent = KEEP
    resolved[i, j, k] = intent


@wp.kernel
def apply_topology_transitions_kernel(
    flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    resolved_flags: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    transition = transitions[i, j, k]
    if transition == INTERFACE_TO_LIQUID:
        flag = LIQUID
    elif transition == INTERFACE_TO_GAS:
        flag = GAS
    elif transition == GAS_TO_INTERFACE or transition == LIQUID_TO_INTERFACE:
        flag = INTERFACE
    resolved_flags[i, j, k] = flag


@wp.kernel
def validate_new_interface_donors_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    new_interface_count: wp.array(dtype=wp.int32),
    invalid_new_interface_count: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if transitions[i, j, k] != GAS_TO_INTERFACE:
        return
    wp.atomic_add(new_interface_count, 0, 1)
    donor_count = int(0)
    valid = bool(True)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                ni = _neighbor_coordinate(i, dx, nx, periodic_x)
                nj = _neighbor_coordinate(j, dy, ny, periodic_y)
                nk = _neighbor_coordinate(k, dz, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                donor_flag = flags[ni, nj, nk]
                if donor_flag == LIQUID or donor_flag == INTERFACE:
                    donor = ni * ny * nz + nj * nz + nk
                    rho = moments[donor]
                    valid = valid and wp.isfinite(rho) and rho > 0.0
                    for component in range(1, 10):
                        valid = valid and wp.isfinite(moments[component * nx * ny * nz + donor])
                    donor_count += 1
    if donor_count == 0 or not valid:
        wp.atomic_add(invalid_new_interface_count, 0, 1)


@wp.kernel
def initialize_new_interface_moments_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if transitions[i, j, k] != GAS_TO_INTERFACE:
        return
    rho_sum = float(0.0)
    ux_sum = float(0.0)
    uy_sum = float(0.0)
    uz_sum = float(0.0)
    donor_count = int(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                ni = _neighbor_coordinate(i, dx, nx, periodic_x)
                nj = _neighbor_coordinate(j, dy, ny, periodic_y)
                nk = _neighbor_coordinate(k, dz, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                donor_flag = flags[ni, nj, nk]
                if donor_flag == LIQUID or donor_flag == INTERFACE:
                    donor = ni * ny * nz + nj * nz + nk
                    donor_rho = moments[donor]
                    rho_sum += donor_rho
                    ux_sum += moments[stride + donor] / donor_rho
                    uy_sum += moments[2 * stride + donor] / donor_rho
                    uz_sum += moments[3 * stride + donor] / donor_rho
                    donor_count += 1
    inverse_count = 1.0 / float(donor_count)
    rho = rho_sum * inverse_count
    ux = ux_sum * inverse_count
    uy = uy_sum * inverse_count
    uz = uz_sum * inverse_count
    cell = i * ny * nz + j * nz + k
    moments[cell] = rho
    moments[stride + cell] = rho * ux
    moments[2 * stride + cell] = rho * uy
    moments[3 * stride + cell] = rho * uz
    moments[4 * stride + cell] = rho * ux * ux
    moments[5 * stride + cell] = rho * uy * uy
    moments[6 * stride + cell] = rho * uz * uz
    moments[7 * stride + cell] = rho * ux * uy
    moments[8 * stride + cell] = rho * ux * uz
    moments[9 * stride + cell] = rho * uy * uz
