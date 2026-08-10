# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic staged Warp kernels for HOME-Free topology changes."""

import warp as wp


@wp.kernel
def count_active_neighbors_kernel(
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    counts: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    count = int(0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if flags[ni, nj, nk] != 0:
            count += 1
    counts[i, j, k] = count


@wp.kernel
def receive_excess_kernel(
    advected_mass: wp.array3d(dtype=float),
    queued_excess: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    source_recipient_count: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    working_mass: wp.array3d(dtype=float),
    stranded_excess: wp.array3d(dtype=float),
    initial_mass_cells: wp.array(dtype=wp.float64),
    invalid_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    mass = advected_mass[i, j, k]
    own_excess = queued_excess[i, j, k]
    own_count = source_recipient_count[i, j, k]
    stranded = float(0.0)
    if own_count == 0:
        stranded = own_excess
    stranded_excess[i, j, k] = stranded
    if source_flags[i, j, k] != 0:
        for q in range(1, 27):
            c = directions[q]
            ni = (i + int(c[0]) + nx) % nx
            nj = (j + int(c[1]) + ny) % ny
            nk = (k + int(c[2]) + nz) % nz
            count = source_recipient_count[ni, nj, nk]
            if count > 0:
                mass += queued_excess[ni, nj, nk] / float(count)
    if not wp.isfinite(mass) or not wp.isfinite(own_excess):
        wp.atomic_add(invalid_count, 0, 1)
        initial_mass_cells[cell] = wp.float64(0.0)
    else:
        initial_mass_cells[cell] = wp.float64(advected_mass[i, j, k]) + wp.float64(own_excess)
    working_mass[i, j, k] = mass


@wp.kernel
def mark_candidates_kernel(
    moments: wp.array(dtype=float),
    working_mass: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    candidates: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    transition_tolerance: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = source_flags[i, j, k]
    candidates[i, j, k] = 0
    if flag != 1:
        return
    rho = moments[cell]
    mass = working_mass[i, j, k]
    has_liquid = int(0)
    has_gas = int(0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        neighbor_flag = source_flags[ni, nj, nk]
        has_liquid = wp.max(has_liquid, int(neighbor_flag == 2))
        has_gas = wp.max(has_gas, int(neighbor_flag == 0))
    if not wp.isfinite(rho) or rho <= 0.0 or not wp.isfinite(mass):
        wp.atomic_add(invalid_count, 0, 1)
    elif mass > rho + transition_tolerance or has_gas == 0:
        candidates[i, j, k] = 1
    elif mass < -transition_tolerance or has_liquid == 0:
        candidates[i, j, k] = 2


@wp.kernel
def resolve_growth_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    transitions: wp.array3d(dtype=wp.int32),
    cancelled_to_gas_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    candidate = candidates[i, j, k]
    neighbor_growth = int(0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        neighbor_growth = wp.max(neighbor_growth, int(candidates[ni, nj, nk] == 1))
    if candidate == 2 and neighbor_growth == 1:
        transitions[i, j, k] = 0
        wp.atomic_add(cancelled_to_gas_count, 0, 1)
    elif source_flags[i, j, k] == 0 and neighbor_growth == 1:
        transitions[i, j, k] = 3
    else:
        transitions[i, j, k] = candidate


@wp.kernel
def apply_transitions_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    final_flags: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    transition = transitions[i, j, k]
    flag = source_flags[i, j, k]
    if transition == 1:
        flag = 2
    elif transition == 2:
        flag = 0
    elif transition == 3:
        flag = 1
    if flag == 2:
        for q in range(1, 27):
            c = directions[q]
            ni = (i + int(c[0]) + nx) % nx
            nj = (j + int(c[1]) + ny) % ny
            nk = (k + int(c[2]) + nz) % nz
            if transitions[ni, nj, nk] == 2:
                flag = 1
    final_flags[i, j, k] = flag


@wp.kernel
def initialize_fresh_moments_kernel(
    source_moments: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    final_flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    destination_moments: wp.array(dtype=float),
    missing_donor_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if transitions[i, j, k] != 3:
        return
    count = int(0)
    rho_sum = float(0.0)
    ux_sum = float(0.0)
    uy_sum = float(0.0)
    uz_sum = float(0.0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if source_flags[ni, nj, nk] != 0 and final_flags[ni, nj, nk] != 0:
            neighbor = ni * ny * nz + nj * nz + nk
            rho = source_moments[neighbor]
            if wp.isfinite(rho) and rho > 0.0:
                rho_sum += rho
                ux_sum += source_moments[stride + neighbor] / rho
                uy_sum += source_moments[2 * stride + neighbor] / rho
                uz_sum += source_moments[3 * stride + neighbor] / rho
                count += 1
    if count == 0:
        wp.atomic_add(missing_donor_count, 0, 1)
        return
    cell = i * ny * nz + j * nz + k
    rho = rho_sum / float(count)
    ux = ux_sum / float(count)
    uy = uy_sum / float(count)
    uz = uz_sum / float(count)
    destination_moments[cell] = rho
    destination_moments[stride + cell] = rho * ux
    destination_moments[2 * stride + cell] = rho * uy
    destination_moments[3 * stride + cell] = rho * uz
    destination_moments[4 * stride + cell] = rho * ux * ux
    destination_moments[5 * stride + cell] = rho * uy * uy
    destination_moments[6 * stride + cell] = rho * uz * uz
    destination_moments[7 * stride + cell] = rho * ux * uy
    destination_moments[8 * stride + cell] = rho * ux * uz
    destination_moments[9 * stride + cell] = rho * uy * uz


@wp.kernel
def commit_topology_kernel(
    moments: wp.array(dtype=float),
    working_mass: wp.array3d(dtype=float),
    stranded_excess: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    final_flags: wp.array3d(dtype=wp.int32),
    committed_mass: wp.array3d(dtype=float),
    committed_fill: wp.array3d(dtype=float),
    committed_excess: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    transition_counts: wp.array(dtype=wp.int32),
    final_mass_cells: wp.array(dtype=wp.float64),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    mass = working_mass[i, j, k]
    queued = stranded_excess[i, j, k]
    source_flag = source_flags[i, j, k]
    transition = transitions[i, j, k]
    flag = final_flags[i, j, k]
    output_mass = float(0.0)
    output_fill = float(0.0)
    if flag == 2:
        output_mass = rho
        output_fill = 1.0
        queued += mass - rho
    elif flag == 1:
        output_mass = wp.clamp(mass, 0.0, rho)
        output_fill = output_mass / rho
        queued += mass - output_mass
    else:
        queued += mass
    valid = flag >= 0 and flag <= 2 and wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(output_mass) and wp.isfinite(output_fill) and wp.isfinite(queued)
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        final_mass_cells[cell] = wp.float64(0.0)
    else:
        final_mass_cells[cell] = wp.float64(output_mass) + wp.float64(queued)
    committed_mass[i, j, k] = output_mass
    committed_fill[i, j, k] = output_fill
    committed_excess[i, j, k] = queued
    if transition == 1 and flag == 2:
        wp.atomic_add(transition_counts, 0, 1)
    elif transition == 2:
        wp.atomic_add(transition_counts, 1, 1)
    elif transition == 3:
        wp.atomic_add(transition_counts, 2, 1)
    if source_flag == 2 and flag == 1:
        wp.atomic_add(transition_counts, 3, 1)


@wp.kernel
def validate_separation_kernel(
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    direct_link_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if flags[i, j, k] != 2:
        return
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if flags[ni, nj, nk] == 0:
            wp.atomic_add(direct_link_count, 0, 1)
