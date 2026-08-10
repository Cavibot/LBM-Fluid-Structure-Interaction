# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for materializing HOME-Free excess into geometric VOF."""

from __future__ import annotations

import warp as wp


GAS = wp.constant(0)
INTERFACE = wp.constant(1)
LIQUID = wp.constant(2)
SOLID = wp.constant(3)


@wp.func
def _neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.func
def _active(flag: int) -> bool:
    return flag == INTERFACE or flag == LIQUID


@wp.func
def _queue_capacity(
    queued_mass: float,
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    i: int,
    j: int,
    k: int,
    ny: int,
    nz: int,
) -> float:
    flag = flags[i, j, k]
    if not _active(flag):
        return 0.0
    committed = mass[i, j, k]
    cell_fill = fill[i, j, k]
    if queued_mass > 0.0:
        if cell_fill <= 0.0 or committed <= 0.0:
            return 0.0
        concentration = committed / cell_fill
        return wp.max(concentration * (1.0 - cell_fill), 0.0)
    if queued_mass < 0.0:
        # Removing liquid from a full LIQUID cell would manufacture a new
        # interface away from the resolved free-surface band.  HOME-Free
        # excess is therefore absorbed by existing interface capacity only.
        if flag != INTERFACE:
            return 0.0
        return wp.max(committed, 0.0)
    return 0.0


@wp.kernel
def prepare_geometric_excess_sources_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    excess_mass: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    represented_queue_mass: wp.array3d(dtype=float),
    represented_queue_momentum: wp.array(dtype=float),
    queue_capacity_sum: wp.array3d(dtype=float),
    queue_recipient_count: wp.array3d(dtype=wp.int32),
    queue_source_count: wp.array(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    source_ledger: wp.array(dtype=wp.float64),
    represented_queue_ledger: wp.array(dtype=wp.float64),
    bound_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = flags[i, j, k]
    rho = moments[cell]
    cell_mass = mass[i, j, k]
    cell_fill = fill[i, j, k]
    queued_mass = excess_mass[i, j, k]
    queued_momentum = wp.vec3(
        excess_momentum[cell],
        excess_momentum[stride + cell],
        excess_momentum[2 * stride + cell],
    )
    represented_queue_mass[i, j, k] = 0.0
    queue_capacity_sum[i, j, k] = 0.0
    queue_recipient_count[i, j, k] = 0
    for component in range(3):
        represented_queue_momentum[component * stride + cell] = 0.0

    valid = flag >= GAS and flag <= SOLID
    valid = valid and wp.isfinite(cell_mass) and wp.isfinite(cell_fill)
    valid = valid and wp.isfinite(queued_mass)
    valid = valid and wp.isfinite(queued_momentum[0])
    valid = valid and wp.isfinite(queued_momentum[1])
    valid = valid and wp.isfinite(queued_momentum[2])
    if _active(flag):
        valid = valid and wp.isfinite(rho) and rho > 0.0
        valid = valid and cell_mass > 0.0 and cell_fill > 0.0
        wp.atomic_add(source_ledger, 0, wp.float64(cell_mass))
        wp.atomic_add(
            source_ledger, 1, wp.float64(cell_mass * moments[stride + cell] / rho)
        )
        wp.atomic_add(
            source_ledger,
            2,
            wp.float64(cell_mass * moments[2 * stride + cell] / rho),
        )
        wp.atomic_add(
            source_ledger,
            3,
            wp.float64(cell_mass * moments[3 * stride + cell] / rho),
        )
    elif queued_mass != 0.0:
        valid = False

    has_queued_momentum = (
        queued_momentum[0] != 0.0
        or queued_momentum[1] != 0.0
        or queued_momentum[2] != 0.0
    )
    has_queue = queued_mass != 0.0 or has_queued_momentum
    if queued_mass == 0.0 and has_queued_momentum:
        valid = False
    if has_queue:
        wp.atomic_add(queue_source_count, 0, 1)
        recipients = int(0)
        capacity_sum = float(0.0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    ni = _neighbor(i + di, nx, periodic_x)
                    nj = _neighbor(j + dj, ny, periodic_y)
                    nk = _neighbor(k + dk, nz, periodic_z)
                    if ni >= 0 and nj >= 0 and nk >= 0:
                        if (di != 0 or dj != 0 or dk != 0) and _active(
                            flags[ni, nj, nk]
                        ):
                            recipients += 1
                        capacity_sum += _queue_capacity(
                            queued_mass,
                            mass,
                            fill,
                            flags,
                            ni,
                            nj,
                            nk,
                            ny,
                            nz,
                        )
        if recipients == 0:
            valid = False
        if capacity_sum <= 0.0 or not wp.isfinite(capacity_sum):
            valid = False
        represented_mass = float(recipients) * queued_mass
        represented_queue_mass[i, j, k] = represented_mass
        queue_capacity_sum[i, j, k] = capacity_sum
        queue_recipient_count[i, j, k] = recipients
        wp.atomic_add(source_ledger, 0, wp.float64(represented_mass))
        wp.atomic_add(
            represented_queue_ledger, 0, wp.float64(represented_mass)
        )
        for component in range(3):
            represented_momentum = float(recipients) * queued_momentum[component]
            represented_queue_momentum[component * stride + cell] = (
                represented_momentum
            )
            wp.atomic_add(
                source_ledger, component + 1, wp.float64(represented_momentum)
            )
            wp.atomic_add(
                represented_queue_ledger,
                component + 1,
                wp.float64(represented_momentum),
            )
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)


@wp.kernel
def gather_prepared_geometric_excess_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    represented_queue_mass: wp.array3d(dtype=float),
    represented_queue_momentum: wp.array(dtype=float),
    queue_capacity_sum: wp.array3d(dtype=float),
    queue_recipient_count: wp.array3d(dtype=wp.int32),
    materialized_fill: wp.array3d(dtype=float),
    materialized_mass: wp.array3d(dtype=float),
    incoming_mass: wp.array3d(dtype=float),
    incoming_momentum: wp.array(dtype=float),
    materialized_marker: wp.array3d(dtype=wp.int32),
    materialized_cell_count: wp.array(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    invalid_counts: wp.array(dtype=wp.int32),
    gathered_ledger: wp.array(dtype=wp.float64),
    redistribution_ledger: wp.array(dtype=wp.float64),
    maximum_fill_delta: wp.array(dtype=float),
    bound_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = flags[i, j, k]
    rho = moments[cell]
    cell_mass = mass[i, j, k]
    cell_fill = fill[i, j, k]
    materialized_fill[i, j, k] = cell_fill
    materialized_mass[i, j, k] = cell_mass
    incoming_mass[i, j, k] = 0.0
    materialized_marker[i, j, k] = 0
    for component in range(3):
        incoming_momentum[component * stride + cell] = 0.0
    if not _active(flag):
        return

    gathered_mass = float(0.0)
    gathered_momentum = wp.vec3(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    source_total = represented_queue_mass[ni, nj, nk]
                    if source_total != 0.0:
                        source = ni * ny * nz + nj * nz + nk
                        capacity_sum = queue_capacity_sum[ni, nj, nk]
                        recipients = queue_recipient_count[ni, nj, nk]
                        destination_capacity = _queue_capacity(
                            source_total,
                            mass,
                            fill,
                            flags,
                            i,
                            j,
                            k,
                            ny,
                            nz,
                        )
                        valid_source = capacity_sum > 0.0 and recipients > 0
                        if not valid_source:
                            wp.atomic_add(invalid_cell_count, 0, 1)
                            wp.atomic_add(invalid_counts, 0, 1)
                        else:
                            weight = destination_capacity / capacity_sum
                            gathered_mass += source_total * weight
                            for component in range(3):
                                gathered_momentum[component] += (
                                    represented_queue_momentum[
                                        component * stride + source
                                    ]
                                    * weight
                                )

    concentration = cell_mass / cell_fill
    requested_mass = cell_mass + gathered_mass
    requested_fill = requested_mass / concentration
    finite_target = wp.isfinite(requested_mass) and wp.isfinite(requested_fill)
    if not finite_target:
        wp.atomic_add(invalid_cell_count, 0, 1)
        wp.atomic_add(invalid_counts, 1, 1)
        return
    target_mass = requested_mass
    accepted_mass = gathered_mass
    accepted_momentum = gathered_momentum
    if requested_mass < 0.0 or requested_mass > concentration:
        target_mass = wp.clamp(requested_mass, 0.0, concentration)
        accepted_mass = target_mass - cell_mass
        accepted_momentum = wp.vec3(0.0)
        if gathered_mass != 0.0:
            accepted_fraction = accepted_mass / gathered_mass
            accepted_momentum = gathered_momentum * accepted_fraction
    target_fill = target_mass / concentration
    residual_mass = gathered_mass - accepted_mass
    residual_momentum = gathered_momentum - accepted_momentum
    wp.atomic_add(redistribution_ledger, 0, wp.float64(residual_mass))
    for component in range(3):
        wp.atomic_add(
            redistribution_ledger,
            component + 1,
            wp.float64(residual_momentum[component]),
        )
    wp.atomic_max(maximum_fill_delta, 0, wp.abs(target_fill - cell_fill))
    if (
        accepted_mass != 0.0
        or accepted_momentum[0] != 0.0
        or accepted_momentum[1] != 0.0
        or accepted_momentum[2] != 0.0
    ):
        materialized_marker[i, j, k] = 1
        wp.atomic_add(materialized_cell_count, 0, 1)
    materialized_fill[i, j, k] = target_fill
    materialized_mass[i, j, k] = target_mass
    incoming_mass[i, j, k] = accepted_mass
    for component in range(3):
        incoming_momentum[component * stride + cell] = accepted_momentum[component]
    wp.atomic_add(gathered_ledger, 0, wp.float64(gathered_mass))
    for component in range(3):
        wp.atomic_add(
            gathered_ledger,
            component + 1,
            wp.float64(gathered_momentum[component]),
        )


@wp.kernel
def sum_global_geometric_excess_capacity_kernel(
    source_mass: wp.array3d(dtype=float),
    source_fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    materialized_mass: wp.array3d(dtype=float),
    redistribution_ledger: wp.array(dtype=wp.float64),
    capacity_sum: wp.array(dtype=wp.float64),
):
    i, j, k = wp.tid()
    if flags[i, j, k] != INTERFACE:
        return
    residual = float(redistribution_ledger[0])
    if residual == 0.0:
        return
    concentration = source_mass[i, j, k] / source_fill[i, j, k]
    capacity = materialized_mass[i, j, k]
    if residual > 0.0:
        capacity = concentration - materialized_mass[i, j, k]
    wp.atomic_add(capacity_sum, 0, wp.float64(wp.max(capacity, 0.0)))


@wp.kernel
def apply_global_geometric_excess_kernel(
    source_mass: wp.array3d(dtype=float),
    source_fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    materialized_fill: wp.array3d(dtype=float),
    materialized_mass: wp.array3d(dtype=float),
    incoming_mass: wp.array3d(dtype=float),
    incoming_momentum: wp.array(dtype=float),
    materialized_marker: wp.array3d(dtype=wp.int32),
    materialized_cell_count: wp.array(dtype=wp.int32),
    redistribution_ledger: wp.array(dtype=wp.float64),
    capacity_sum: wp.array(dtype=wp.float64),
    invalid_cell_count: wp.array(dtype=wp.int32),
    invalid_counts: wp.array(dtype=wp.int32),
    maximum_fill_delta: wp.array(dtype=float),
    bound_tolerance: float,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if flags[i, j, k] != INTERFACE:
        return
    residual = float(redistribution_ledger[0])
    if residual == 0.0:
        return
    total_capacity = float(capacity_sum[0])
    if (
        not wp.isfinite(total_capacity)
        or total_capacity <= 0.0
        or wp.abs(residual) > total_capacity + bound_tolerance
    ):
        wp.atomic_add(invalid_cell_count, 0, 1)
        if residual < 0.0:
            wp.atomic_add(invalid_counts, 2, 1)
        else:
            wp.atomic_add(invalid_counts, 3, 1)
        return
    concentration = source_mass[i, j, k] / source_fill[i, j, k]
    capacity = materialized_mass[i, j, k]
    if residual > 0.0:
        capacity = concentration - materialized_mass[i, j, k]
    capacity = wp.max(capacity, 0.0)
    if capacity == 0.0:
        return
    weight = capacity / total_capacity
    mass_delta = residual * weight
    target_mass = materialized_mass[i, j, k] + mass_delta
    target_fill = target_mass / concentration
    if (
        not wp.isfinite(target_mass)
        or not wp.isfinite(target_fill)
        or target_fill < -bound_tolerance
        or target_fill > 1.0 + bound_tolerance
    ):
        wp.atomic_add(invalid_cell_count, 0, 1)
        wp.atomic_add(invalid_counts, 1, 1)
        return
    cell = i * ny * nz + j * nz + k
    materialized_mass[i, j, k] = target_mass
    materialized_fill[i, j, k] = wp.clamp(target_fill, 0.0, 1.0)
    wp.atomic_max(
        maximum_fill_delta,
        0,
        wp.abs(materialized_fill[i, j, k] - source_fill[i, j, k]),
    )
    incoming_mass[i, j, k] += mass_delta
    for component in range(3):
        incoming_momentum[component * stride + cell] += (
            float(redistribution_ledger[component + 1]) * weight
        )
    if materialized_marker[i, j, k] == 0:
        materialized_marker[i, j, k] = 1
        wp.atomic_add(materialized_cell_count, 0, 1)


@wp.kernel
def accumulate_committed_geometric_ledger_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    ledger: wp.array(dtype=wp.float64),
    invalid_cell_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if not _active(flag):
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    committed = mass[i, j, k]
    if not wp.isfinite(rho) or rho <= 0.0 or not wp.isfinite(committed):
        wp.atomic_add(invalid_cell_count, 0, 1)
        return
    wp.atomic_add(ledger, 0, wp.float64(committed))
    wp.atomic_add(
        ledger, 1, wp.float64(committed * moments[stride + cell] / rho)
    )
    wp.atomic_add(
        ledger, 2, wp.float64(committed * moments[2 * stride + cell] / rho)
    )
    wp.atomic_add(
        ledger, 3, wp.float64(committed * moments[3 * stride + cell] / rho)
    )
