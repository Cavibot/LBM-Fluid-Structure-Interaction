# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Device kernels for strict geometric FSL topology commit."""

import warp as wp


GAS = 0
INTERFACE = 1
LIQUID = 2


@wp.func
def _neighbor(value: int, size: int, periodic: int) -> int:
    if value >= 0 and value < size:
        return value
    if periodic != 0:
        return (value + size) % size
    return -1


@wp.kernel
def prepare_endpoint_closure_kernel(
    fill: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    tail_volume_cells: wp.array(dtype=wp.float64),
    tail_mass_cells: wp.array(dtype=wp.float64),
    tail_momentum_x_cells: wp.array(dtype=wp.float64),
    tail_momentum_y_cells: wp.array(dtype=wp.float64),
    tail_momentum_z_cells: wp.array(dtype=wp.float64),
    recipient_capacity_cells: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    endpoint_tolerance: float,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    value = fill[i, j, k]
    liquid_mass = mass[i, j, k]
    liquid_momentum = momentum[i, j, k]
    tail = (
        solid[i, j, k] == 0
        and value > 0.0
        and value <= endpoint_tolerance
    )
    recipient = (
        solid[i, j, k] == 0
        and value > endpoint_tolerance
        and value < 1.0 - endpoint_tolerance
    )
    tail_volume_cells[cell] = wp.float64(value) if tail else wp.float64(0.0)
    tail_mass_cells[cell] = wp.float64(liquid_mass) if tail else wp.float64(0.0)
    tail_momentum_x_cells[cell] = (
        wp.float64(liquid_momentum[0]) if tail else wp.float64(0.0)
    )
    tail_momentum_y_cells[cell] = (
        wp.float64(liquid_momentum[1]) if tail else wp.float64(0.0)
    )
    tail_momentum_z_cells[cell] = (
        wp.float64(liquid_momentum[2]) if tail else wp.float64(0.0)
    )
    recipient_capacity_cells[cell] = (
        wp.float64(1.0 - value) if recipient else wp.float64(0.0)
    )
    if tail:
        wp.atomic_add(counts, 12, 1)


@wp.kernel
def validate_endpoint_closure_totals_kernel(
    tail_volume: wp.array(dtype=wp.float64),
    tail_mass: wp.array(dtype=wp.float64),
    tail_momentum_x: wp.array(dtype=wp.float64),
    tail_momentum_y: wp.array(dtype=wp.float64),
    tail_momentum_z: wp.array(dtype=wp.float64),
    recipient_capacity: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
):
    volume = tail_volume[0]
    capacity = recipient_capacity[0]
    valid = wp.isfinite(volume) and volume >= wp.float64(0.0)
    valid = valid and wp.isfinite(tail_mass[0]) and tail_mass[0] >= wp.float64(0.0)
    valid = valid and wp.isfinite(tail_momentum_x[0])
    valid = valid and wp.isfinite(tail_momentum_y[0])
    valid = valid and wp.isfinite(tail_momentum_z[0])
    valid = valid and wp.isfinite(capacity) and capacity >= wp.float64(0.0)
    if volume > wp.float64(0.0):
        valid = valid and capacity >= volume
    if not valid:
        wp.atomic_add(counts, 13, 1)


@wp.kernel
def apply_endpoint_closure_kernel(
    fill: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    tail_volume: wp.array(dtype=wp.float64),
    tail_mass: wp.array(dtype=wp.float64),
    tail_momentum_x: wp.array(dtype=wp.float64),
    tail_momentum_y: wp.array(dtype=wp.float64),
    tail_momentum_z: wp.array(dtype=wp.float64),
    recipient_capacity: wp.array(dtype=wp.float64),
    closed_fill: wp.array3d(dtype=float),
    closed_mass: wp.array3d(dtype=float),
    closed_momentum: wp.array3d(dtype=wp.vec3),
    endpoint_tolerance: float,
):
    i, j, k = wp.tid()
    value = fill[i, j, k]
    tail = (
        solid[i, j, k] == 0
        and value > 0.0
        and value <= endpoint_tolerance
    )
    recipient = (
        solid[i, j, k] == 0
        and value > endpoint_tolerance
        and value < 1.0 - endpoint_tolerance
    )
    output_fill = 0.0 if tail else value
    output_mass = 0.0 if tail else mass[i, j, k]
    output_momentum = wp.vec3(0.0) if tail else momentum[i, j, k]
    total_capacity = recipient_capacity[0]
    if recipient and tail_volume[0] > wp.float64(0.0):
        weight = wp.float64(1.0 - value) / total_capacity
        output_fill += float(tail_volume[0] * weight)
        output_mass += float(tail_mass[0] * weight)
        output_momentum += wp.vec3(
            float(tail_momentum_x[0] * weight),
            float(tail_momentum_y[0] * weight),
            float(tail_momentum_z[0] * weight),
        )
    closed_fill[i, j, k] = output_fill
    closed_mass[i, j, k] = output_mass
    closed_momentum[i, j, k] = output_momentum


@wp.kernel
def classify_topology_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    fill: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    target_flags: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    endpoint_tolerance: float,
):
    i, j, k = wp.tid()
    source = source_flags[i, j, k]
    value = fill[i, j, k]
    liquid_mass = mass[i, j, k]
    liquid_momentum = momentum[i, j, k]
    target = INTERFACE
    if value == 0.0:
        target = GAS
    elif value >= 1.0 - endpoint_tolerance:
        target = LIQUID
    if solid[i, j, k] != 0:
        target = GAS
        if value != 0.0 or liquid_mass != 0.0 or wp.length(liquid_momentum) != 0.0:
            wp.atomic_add(counts, 5, 1)
    if source < GAS or source > LIQUID or not wp.isfinite(value) or value < 0.0 or value > 1.0:
        wp.atomic_add(counts, 5, 1)
    if not wp.isfinite(liquid_mass) or liquid_mass < 0.0:
        wp.atomic_add(counts, 5, 1)
    if not wp.isfinite(liquid_momentum[0]) or not wp.isfinite(liquid_momentum[1]) or not wp.isfinite(liquid_momentum[2]):
        wp.atomic_add(counts, 5, 1)
    if source == GAS and target == LIQUID:
        wp.atomic_add(counts, 6, 1)
    if source == LIQUID and target == GAS:
        wp.atomic_add(counts, 7, 1)
    if target == LIQUID and value < 1.0:
        wp.atomic_add(counts, 10, 1)
    target_flags[i, j, k] = target


@wp.kernel
def repair_separation_kernel(
    classified_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    repaired_flags: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    target = classified_flags[i, j, k]
    if solid[i, j, k] != 0 or target != LIQUID:
        repaired_flags[i, j, k] = target
        return
    adjacent_gas = bool(False)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if (
                        solid[ni, nj, nk] == 0
                        and classified_flags[ni, nj, nk] == GAS
                    ):
                        adjacent_gas = True
    if adjacent_gas:
        repaired_flags[i, j, k] = INTERFACE
        wp.atomic_add(counts, 11, 1)
    else:
        repaired_flags[i, j, k] = target


@wp.kernel
def count_topology_transitions_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    target_flags: wp.array3d(dtype=wp.int32),
    fill: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    endpoint_tolerance: float,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        return
    source = source_flags[i, j, k]
    target = target_flags[i, j, k]
    value = fill[i, j, k]
    if source == GAS and target == INTERFACE:
        wp.atomic_add(counts, 0, 1)
    elif source == LIQUID and target == INTERFACE:
        wp.atomic_add(counts, 1, 1)
    elif source == INTERFACE and target == GAS:
        wp.atomic_add(counts, 2, 1)
    elif source == INTERFACE and target == LIQUID:
        wp.atomic_add(counts, 3, 1)
    if (
        target == INTERFACE
        and (value <= endpoint_tolerance or value >= 1.0 - endpoint_tolerance)
    ):
        wp.atomic_add(counts, 4, 1)


@wp.kernel
def validate_separation_kernel(
    target_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0 or target_flags[i, j, k] != LIQUID:
        return
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if solid[ni, nj, nk] == 0 and target_flags[ni, nj, nk] == GAS:
                        wp.atomic_add(counts, 8, 1)


@wp.kernel
def count_donors_kernel(
    moments: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    target_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    donor_count: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    target = target_flags[i, j, k]
    cell = i * ny * nz + j * nz + k
    if source_flags[i, j, k] != GAS and target != GAS and (not wp.isfinite(moments[cell]) or moments[cell] <= 0.0):
        wp.atomic_add(counts, 5, 1)
    if source_flags[i, j, k] != GAS or target != INTERFACE:
        donor_count[i, j, k] = 0
        return
    donors = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if solid[ni, nj, nk] == 0 and source_flags[ni, nj, nk] != GAS and target_flags[ni, nj, nk] != GAS:
                        donor = ni * ny * nz + nj * nz + nk
                        if wp.isfinite(moments[donor]) and moments[donor] > 0.0:
                            donors += 1
    donor_count[i, j, k] = donors
    if donors == 0:
        wp.atomic_add(counts, 9, 1)


@wp.kernel
def initialize_fresh_interface_kernel(
    source_moments: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    target_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    donor_count: wp.array3d(dtype=wp.int32),
    target_moments: wp.array(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if source_flags[i, j, k] != GAS or target_flags[i, j, k] != INTERFACE:
        return
    donors = donor_count[i, j, k]
    if donors == 0:
        return
    rho_sum = float(0.0)
    velocity_sum = wp.vec3(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if solid[ni, nj, nk] == 0 and source_flags[ni, nj, nk] != GAS and target_flags[ni, nj, nk] != GAS:
                        donor = ni * ny * nz + nj * nz + nk
                        rho = source_moments[donor]
                        if wp.isfinite(rho) and rho > 0.0:
                            rho_sum += rho
                            velocity_sum += wp.vec3(
                                source_moments[stride + donor] / rho,
                                source_moments[2 * stride + donor] / rho,
                                source_moments[3 * stride + donor] / rho,
                            )
    inverse = 1.0 / float(donors)
    rho = rho_sum * inverse
    velocity = velocity_sum * inverse
    cell = i * ny * nz + j * nz + k
    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]
    target_moments[cell] = rho
    target_moments[stride + cell] = rho * ux
    target_moments[2 * stride + cell] = rho * uy
    target_moments[3 * stride + cell] = rho * uz
    target_moments[4 * stride + cell] = rho * ux * ux
    target_moments[5 * stride + cell] = rho * uy * uy
    target_moments[6 * stride + cell] = rho * uz * uz
    target_moments[7 * stride + cell] = rho * ux * uy
    target_moments[8 * stride + cell] = rho * ux * uz
    target_moments[9 * stride + cell] = rho * uy * uz
