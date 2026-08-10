# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-conservative HOME moment and VOF remap for moving rigid cells."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from ..state import HomeLbmState
from .state import HomeFreeState


_Vec10 = wp.types.vector(length=10, dtype=wp.float32)


_EXCESS_LOCALITY_RADIUS = 3


@wp.func
def _transition_neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.kernel
def _accumulate_old_state_kernel(
    moments: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=float),
    excess: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    old_moment_sum: wp.array(dtype=float),
    old_represented_mass: wp.array(dtype=wp.float64),
    old_represented_momentum: wp.array(dtype=float),
    invalid_old_state_count: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    was_solid = previous_phi[i, j, k] < 0.0
    if was_solid != (flag == 3):
        wp.atomic_add(invalid_old_state_count, 0, 1)
        return
    if flag == 3:
        return
    cell = i * ny * nz + j * nz + k
    recipients = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di != 0 or dj != 0 or dk != 0:
                    ni = _transition_neighbor(i + di, nx, periodic_x)
                    nj = _transition_neighbor(j + dj, ny, periodic_y)
                    nk = _transition_neighbor(k + dk, nz, periodic_z)
                    if ni >= 0 and nj >= 0 and nk >= 0:
                        neighbor_flag = flags[ni, nj, nk]
                        if neighbor_flag == 1 or neighbor_flag == 2:
                            recipients += 1
    committed = mass[i, j, k]
    represented = committed + excess[i, j, k] * float(recipients)
    wp.atomic_add(old_represented_mass, 0, wp.float64(represented))
    rho = moments[cell]
    if committed != 0.0:
        if not wp.isfinite(rho) or rho <= 0.0:
            wp.atomic_add(invalid_old_state_count, 0, 1)
            return
        for component in range(3):
            value = committed * moments[(component + 1) * stride + cell] / rho
            wp.atomic_add(old_represented_momentum, component, value)
    for component in range(3):
        wp.atomic_add(
            old_represented_momentum,
            component,
            float(recipients) * excess_momentum[component * stride + cell],
        )
    if flag != 1 and flag != 2:
        return
    for component in range(10):
        wp.atomic_add(old_moment_sum, component, moments[component * stride + cell])


@wp.kernel
def _classify_and_initialize_kernel(
    moments_before: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    flags_before: wp.array3d(dtype=wp.int32),
    fill_before: wp.array3d(dtype=float),
    moments_after: wp.array(dtype=float),
    proposed_flags: wp.array3d(dtype=wp.int32),
    proposed_fill: wp.array3d(dtype=float),
    fresh_count: wp.array(dtype=wp.int32),
    dead_count: wp.array(dtype=wp.int32),
    fresh_phase_counts: wp.array(dtype=wp.int32),
    unresolved_count: wp.array(dtype=wp.int32),
    invalid_donor_count: wp.array(dtype=wp.int32),
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
    was_fluid = previous_phi[i, j, k] >= 0.0
    is_fluid = current_phi[i, j, k] >= 0.0
    for component in range(10):
        moments_after[component * stride + cell] = moments_before[component * stride + cell]
    if not is_fluid:
        proposed_flags[i, j, k] = 3
        proposed_fill[i, j, k] = 0.0
        if was_fluid:
            wp.atomic_add(dead_count, 0, 1)
        return
    if was_fluid:
        proposed_flags[i, j, k] = flags_before[i, j, k]
        proposed_fill[i, j, k] = fill_before[i, j, k]
        return

    wp.atomic_add(fresh_count, 0, 1)
    donor_count = int(0)
    active_count = int(0)
    gas_count = int(0)
    liquid_count = int(0)
    interface_count = int(0)
    donor_fill = float(0.0)
    active_moments = _Vec10(0.0)
    gas_moments = _Vec10(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di != 0 or dj != 0 or dk != 0:
                    ni = _transition_neighbor(i + di, nx, periodic_x)
                    nj = _transition_neighbor(j + dj, ny, periodic_y)
                    nk = _transition_neighbor(k + dk, nz, periodic_z)
                    if ni >= 0 and nj >= 0 and nk >= 0:
                        persistent = previous_phi[ni, nj, nk] >= 0.0
                        persistent = persistent and current_phi[ni, nj, nk] >= 0.0
                        if persistent:
                            donor_count += 1
                            donor_flag = flags_before[ni, nj, nk]
                            donor_fill += fill_before[ni, nj, nk]
                            donor_cell = ni * ny * nz + nj * nz + nk
                            if donor_flag == 1 or donor_flag == 2:
                                active_count += 1
                                if donor_flag == 1:
                                    interface_count += 1
                                else:
                                    liquid_count += 1
                                for component in range(10):
                                    active_moments[component] += moments_before[
                                        component * stride + donor_cell
                                    ]
                            elif donor_flag == 0:
                                gas_count += 1
                                for component in range(10):
                                    gas_moments[component] += moments_before[
                                        component * stride + donor_cell
                                    ]
                            else:
                                wp.atomic_add(invalid_donor_count, 0, 1)
    if donor_count == 0:
        wp.atomic_add(unresolved_count, 0, 1)
        proposed_flags[i, j, k] = 0
        proposed_fill[i, j, k] = 0.0
        return

    new_flag = int(0)
    new_fill = float(0.0)
    if interface_count > 0 or (liquid_count > 0 and gas_count > 0):
        new_flag = 1
        new_fill = donor_fill / float(donor_count)
    elif liquid_count > 0:
        new_flag = 2
        new_fill = 1.0
    elif gas_count > 0:
        new_flag = 0
        new_fill = 0.0
    else:
        wp.atomic_add(unresolved_count, 0, 1)
        return
    proposed_flags[i, j, k] = new_flag
    proposed_fill[i, j, k] = new_fill
    wp.atomic_add(fresh_phase_counts, new_flag, 1)
    if new_flag == 1 or new_flag == 2:
        if active_count == 0:
            wp.atomic_add(unresolved_count, 0, 1)
            return
        for component in range(10):
            moments_after[component * stride + cell] = (
                active_moments[component] / float(active_count)
            )
    else:
        if gas_count == 0:
            wp.atomic_add(unresolved_count, 0, 1)
            return
        for component in range(10):
            moments_after[component * stride + cell] = (
                gas_moments[component] / float(gas_count)
            )


@wp.kernel
def _accumulate_new_active_kernel(
    moments: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    new_moment_sum: wp.array(dtype=float),
    correction_recipient_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if flag != 1 and flag != 2:
        return
    cell = i * ny * nz + j * nz + k
    for component in range(10):
        wp.atomic_add(new_moment_sum, component, moments[component * stride + cell])
    if previous_phi[i, j, k] >= 0.0 and current_phi[i, j, k] >= 0.0:
        wp.atomic_add(correction_recipient_count, 0, 1)


@wp.kernel
def _compute_moment_delta_kernel(
    old_sum: wp.array(dtype=float),
    new_sum: wp.array(dtype=float),
    delta: wp.array(dtype=float),
):
    component = wp.tid()
    delta[component] = old_sum[component] - new_sum[component]


@wp.kernel
def _apply_moment_correction_kernel(
    moments: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    delta: wp.array(dtype=float),
    recipient_count: wp.array(dtype=wp.int32),
    invalid_moment_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    persistent_active = previous_phi[i, j, k] >= 0.0
    persistent_active = persistent_active and current_phi[i, j, k] >= 0.0
    persistent_active = persistent_active and (flag == 1 or flag == 2)
    if not persistent_active:
        return
    count = recipient_count[0]
    if count <= 0:
        wp.atomic_add(invalid_moment_count, 0, 1)
        return
    cell = i * ny * nz + j * nz + k
    valid = bool(True)
    for component in range(10):
        value = moments[component * stride + cell] + delta[component] / float(count)
        moments[component * stride + cell] = value
        valid = valid and wp.isfinite(value)
    valid = valid and moments[cell] > 0.0
    if not valid:
        wp.atomic_add(invalid_moment_count, 0, 1)


@wp.kernel
def _commit_vof_kernel(
    moments: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    flags_before: wp.array3d(dtype=wp.int32),
    mass_before: wp.array3d(dtype=float),
    proposed_flags: wp.array3d(dtype=wp.int32),
    proposed_fill: wp.array3d(dtype=float),
    committed_mass: wp.array3d(dtype=float),
    committed_fill: wp.array3d(dtype=float),
    committed_excess: wp.array3d(dtype=float),
    committed_excess_momentum: wp.array(dtype=float),
    committed_mass_sum: wp.array(dtype=wp.float64),
    interface_density_sum: wp.array(dtype=wp.float64),
    invalid_commit_count: wp.array(dtype=wp.int32),
    allow_independent_mass: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = proposed_flags[i, j, k]
    rho = moments[cell]
    mass = float(0.0)
    fill = float(0.0)
    valid = bool(True)
    if flag == 2:
        valid = wp.isfinite(rho) and rho > 0.0
        persistent = previous_phi[i, j, k] >= 0.0
        persistent = persistent and current_phi[i, j, k] >= 0.0
        persistent = persistent and (
            flags_before[i, j, k] == 1 or flags_before[i, j, k] == 2
        )
        if allow_independent_mass != 0 and persistent:
            mass = mass_before[i, j, k]
        else:
            mass = rho
        fill = 1.0
    elif flag == 1:
        valid = wp.isfinite(rho) and rho > 0.0
        persistent_interface = previous_phi[i, j, k] >= 0.0
        persistent_interface = persistent_interface and current_phi[i, j, k] >= 0.0
        persistent_interface = persistent_interface and flags_before[i, j, k] == 1
        if persistent_interface:
            mass = mass_before[i, j, k]
            if allow_independent_mass != 0:
                fill = proposed_fill[i, j, k]
        else:
            mass = rho * proposed_fill[i, j, k]
        if allow_independent_mass == 0:
            mass = wp.clamp(mass, 0.0, rho)
            fill = mass / rho
        elif not persistent_interface:
            fill = proposed_fill[i, j, k]
        wp.atomic_add(interface_density_sum, 0, wp.float64(rho))
    elif flag != 0 and flag != 3:
        valid = False
    committed_mass[i, j, k] = mass
    committed_fill[i, j, k] = fill
    committed_excess[i, j, k] = 0.0
    for component in range(3):
        committed_excess_momentum[component * stride + cell] = 0.0
    if valid:
        wp.atomic_add(committed_mass_sum, 0, wp.float64(mass))
    else:
        wp.atomic_add(invalid_commit_count, 0, 1)


@wp.kernel
def _seed_transition_neighborhood_kernel(
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    neighborhood: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    was_fluid = previous_phi[i, j, k] >= 0.0
    is_fluid = current_phi[i, j, k] >= 0.0
    neighborhood[i, j, k] = int(was_fluid != is_fluid)


@wp.kernel
def _seed_transition_columns_kernel(
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    columns: wp.array2d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    was_fluid = previous_phi[i, j, k] >= 0.0
    is_fluid = current_phi[i, j, k] >= 0.0
    if was_fluid != is_fluid:
        wp.atomic_max(columns, i, j, 1)


@wp.kernel
def _dilate_transition_neighborhood_kernel(
    source: wp.array3d(dtype=wp.int32),
    destination: wp.array3d(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    marked = source[i, j, k] != 0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if not marked:
                    ni = _transition_neighbor(i + di, nx, periodic_x)
                    nj = _transition_neighbor(j + dj, ny, periodic_y)
                    nk = _transition_neighbor(k + dk, nz, periodic_z)
                    if ni >= 0 and nj >= 0 and nk >= 0:
                        marked = source[ni, nj, nk] != 0
    destination[i, j, k] = int(marked)


@wp.kernel
def _dilate_transition_columns_kernel(
    source: wp.array2d(dtype=wp.int32),
    destination: wp.array2d(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    marked = source[i, j] != 0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if not marked:
                ni = _transition_neighbor(i + di, nx, periodic_x)
                nj = _transition_neighbor(j + dj, ny, periodic_y)
                if ni >= 0 and nj >= 0:
                    marked = source[ni, nj] != 0
    destination[i, j] = int(marked)


@wp.kernel
def _merge_transition_support_kernel(
    neighborhood: wp.array3d(dtype=wp.int32),
    columns: wp.array2d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    if columns[i, j] != 0:
        neighborhood[i, j, k] = 1


@wp.kernel
def _count_new_recipients_kernel(
    flags: wp.array3d(dtype=wp.int32),
    recipient_counts: wp.array3d(dtype=wp.int32),
    total_recipient_count: wp.array(dtype=wp.int32),
    interface_count: wp.array(dtype=wp.int32),
    zero_recipient_interface_count: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    count = int(0)
    if flags[i, j, k] == 1:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di != 0 or dj != 0 or dk != 0:
                        ni = _transition_neighbor(i + di, nx, periodic_x)
                        nj = _transition_neighbor(j + dj, ny, periodic_y)
                        nk = _transition_neighbor(k + dk, nz, periodic_z)
                        if ni >= 0 and nj >= 0 and nk >= 0:
                            neighbor_flag = flags[ni, nj, nk]
                            if neighbor_flag == 1 or neighbor_flag == 2:
                                count += 1
        wp.atomic_add(interface_count, 0, 1)
        wp.atomic_add(total_recipient_count, 0, count)
        if count == 0:
            wp.atomic_add(zero_recipient_interface_count, 0, 1)
    recipient_counts[i, j, k] = count


@wp.kernel
def _select_local_queue_sources_kernel(
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    transition_support: wp.array3d(dtype=wp.int32),
    recipient_counts: wp.array3d(dtype=wp.int32),
    selected: wp.array3d(dtype=wp.int32),
    selected_recipient_count: wp.array(dtype=wp.int32),
    selected_source_count: wp.array(dtype=wp.int32),
    queued_mass: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    selected[i, j, k] = 0
    if flags[i, j, k] != 1 or transition_support[i, j, k] == 0:
        return
    capacity = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = _transition_neighbor(i + di, nx, periodic_x)
                nj = _transition_neighbor(j + dj, ny, periodic_y)
                nk = _transition_neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0 and flags[ni, nj, nk] == 1:
                    cell_mass = mass[ni, nj, nk]
                    cell_fill = fill[ni, nj, nk]
                    if queued_mass > 0.0 and cell_fill > 0.0:
                        concentration = cell_mass / cell_fill
                        capacity += wp.max(
                            concentration * (1.0 - cell_fill), 0.0
                        )
                    elif queued_mass < 0.0:
                        capacity += wp.max(cell_mass, 0.0)
    if capacity > 0.0 and wp.isfinite(capacity):
        selected[i, j, k] = 1
        wp.atomic_add(selected_source_count, 0, 1)
        wp.atomic_add(
            selected_recipient_count, 0, recipient_counts[i, j, k]
        )


@wp.kernel
def _assign_local_excess_share_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    selected_sources: wp.array3d(dtype=wp.int32),
    excess: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    share: float,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if flags[i, j, k] == 1 and selected_sources[i, j, k] != 0:
        excess[i, j, k] = share
        cell = i * ny * nz + j * nz + k
        rho = moments[cell]
        for component in range(3):
            excess_momentum[component * stride + cell] = (
                share * moments[(component + 1) * stride + cell] / rho
            )


@wp.kernel
def _sum_represented_mass_kernel(
    mass: wp.array3d(dtype=float),
    excess: wp.array3d(dtype=float),
    recipient_counts: wp.array3d(dtype=wp.int32),
    represented_mass: wp.array(dtype=wp.float64),
):
    i, j, k = wp.tid()
    value = mass[i, j, k] + excess[i, j, k] * float(
        recipient_counts[i, j, k]
    )
    wp.atomic_add(represented_mass, 0, wp.float64(value))


@wp.kernel
def _sum_represented_momentum_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    excess: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    recipient_counts: wp.array3d(dtype=wp.int32),
    represented_momentum: wp.array(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    committed = mass[i, j, k]
    recipients = recipient_counts[i, j, k]
    if committed == 0.0 and recipients == 0:
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    if committed != 0.0 and (not wp.isfinite(rho) or rho <= 0.0):
        wp.atomic_add(invalid_count, 0, 1)
        return
    for component in range(3):
        value = float(recipients) * excess_momentum[component * stride + cell]
        if committed != 0.0:
            value += committed * moments[(component + 1) * stride + cell] / rho
        if wp.isfinite(value):
            wp.atomic_add(represented_momentum, component, value)
        else:
            wp.atomic_add(invalid_count, 0, 1)


@wp.kernel
def _compute_velocity_shift_kernel(
    old_momentum: wp.array(dtype=float),
    current_momentum: wp.array(dtype=float),
    represented_mass: wp.array(dtype=wp.float64),
    velocity_shift: wp.array(dtype=float),
):
    component = wp.tid()
    velocity_shift[component] = (
        old_momentum[component] - current_momentum[component]
    ) / float(represented_mass[0])


@wp.kernel
def _apply_galilean_shift_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    excess_mass: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    velocity_shift: wp.array(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if flag != 1 and flag != 2:
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    jx = moments[stride + cell]
    jy = moments[2 * stride + cell]
    jz = moments[3 * stride + cell]
    dx = velocity_shift[0]
    dy = velocity_shift[1]
    dz = velocity_shift[2]
    moments[4 * stride + cell] += 2.0 * dx * jx + rho * dx * dx
    moments[5 * stride + cell] += 2.0 * dy * jy + rho * dy * dy
    moments[6 * stride + cell] += 2.0 * dz * jz + rho * dz * dz
    moments[7 * stride + cell] += dx * jy + dy * jx + rho * dx * dy
    moments[8 * stride + cell] += dx * jz + dz * jx + rho * dx * dz
    moments[9 * stride + cell] += dy * jz + dz * jy + rho * dy * dz
    moments[stride + cell] = jx + rho * dx
    moments[2 * stride + cell] = jy + rho * dy
    moments[3 * stride + cell] = jz + rho * dz
    queued_mass = excess_mass[i, j, k]
    excess_momentum[cell] += queued_mass * dx
    excess_momentum[stride + cell] += queued_mass * dy
    excess_momentum[2 * stride + cell] += queued_mass * dz
    valid = wp.isfinite(rho) and rho > 0.0
    for component in range(1, 10):
        valid = valid and wp.isfinite(moments[component * stride + cell])
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)


@wp.kernel
def _validate_committed_state_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    excess: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    tolerance: float,
    allow_independent_mass: int,
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
    queued_mass = excess[i, j, k]
    legal_flag = flag >= 0 and flag <= 3
    active = flag == 1 or flag == 2
    empty = flag == 0 or flag == 3
    valid = legal_flag
    valid = valid and wp.isfinite(rho) and wp.isfinite(cell_mass)
    valid = valid and wp.isfinite(cell_fill) and wp.isfinite(queued_mass)
    valid = valid and cell_fill >= 0.0 and cell_fill <= 1.0
    if active:
        valid = valid and rho > 0.0
        if allow_independent_mass == 0:
            valid = valid and wp.abs(cell_mass - rho * cell_fill) <= tolerance
        else:
            valid = valid and cell_mass > 0.0
    if flag == 2:
        valid = valid and wp.abs(cell_fill - 1.0) <= tolerance
    if empty:
        valid = valid and wp.abs(cell_fill) <= tolerance
        valid = valid and wp.abs(cell_mass) <= tolerance
    if flag == 3:
        valid = valid and wp.abs(queued_mass) <= tolerance
    for component in range(3):
        queued_momentum = excess_momentum[component * stride + cell]
        valid = valid and wp.isfinite(queued_momentum)
        if wp.abs(queued_mass) <= tolerance or flag == 3:
            valid = valid and wp.abs(queued_momentum) <= tolerance
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)

    if flag != 2:
        return
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _transition_neighbor(i + di, nx, periodic_x)
                nj = _transition_neighbor(j + dj, ny, periodic_y)
                nk = _transition_neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if flags[ni, nj, nk] == 0:
                        wp.atomic_add(direct_liquid_gas_link_count, 0, 1)


@dataclass(frozen=True)
class HomeFreeRigidTransitionDiagnostics:
    fresh_cell_count: int
    dead_cell_count: int
    fresh_liquid_cells: int
    fresh_interface_cells: int
    fresh_gas_cells: int
    old_represented_mass: float
    final_represented_mass: float
    relative_mass_error: float
    queued_mass: float
    queued_mass_ratio: float
    old_represented_momentum: tuple[float, float, float]
    final_represented_momentum: tuple[float, float, float]
    relative_momentum_error: float
    queue_source_count: int = 0
    queue_locality_radius: int = 0


class HomeFreeRigidTransitionRemapper:
    """Conserve active HOME moments and represented liquid mass across SDF motion."""

    def __init__(
        self,
        model: HomeLbmModel,
        fluid_state: HomeLbmState,
        free_state: HomeFreeState,
        *,
        mass_tolerance: float = 2.0e-5,
        momentum_tolerance: float = 2.0e-5,
        max_queued_mass_ratio: float = 0.5,
        allow_independent_mass: bool = False,
    ) -> None:
        if not math.isfinite(mass_tolerance) or mass_tolerance <= 0.0:
            raise ValueError("mass_tolerance must be finite and positive")
        if not math.isfinite(max_queued_mass_ratio) or not 0.0 < max_queued_mass_ratio <= 1.0:
            raise ValueError("max_queued_mass_ratio must be in (0, 1]")
        if not math.isfinite(momentum_tolerance) or momentum_tolerance <= 0.0:
            raise ValueError("momentum_tolerance must be finite and positive")
        self.model = model
        self.res = fluid_state.res
        self.stride = fluid_state.cell_count
        self.device = fluid_state.device
        if free_state.res != self.res or free_state.device != self.device:
            raise ValueError("HOME and HOME-Free states must share resolution and device")
        self.mass_tolerance = float(mass_tolerance)
        self.momentum_tolerance = float(momentum_tolerance)
        self.max_queued_mass_ratio = float(max_queued_mass_ratio)
        self.allow_independent_mass = bool(allow_independent_mass)
        self.previous_phi = wp.empty_like(fluid_state.solid_phi)
        self._moments_before = wp.empty_like(fluid_state.moments)
        self._flags_before = wp.empty_like(free_state.flags)
        self._fill_before = wp.empty_like(free_state.fill_level)
        self._mass_before = wp.empty_like(free_state.mass)
        self._excess_before = wp.empty_like(free_state.excess_mass)
        self._excess_momentum_before = wp.empty_like(
            free_state.excess_momentum
        )
        self._moments_after = wp.empty_like(fluid_state.moments)
        self.proposed_flags = wp.empty_like(free_state.flags)
        self.proposed_fill = wp.empty_like(free_state.fill_level)
        self.committed_mass = wp.empty_like(free_state.mass)
        self.committed_fill = wp.empty_like(free_state.fill_level)
        self.committed_excess = wp.empty_like(free_state.excess_mass)
        self.committed_excess_momentum = wp.empty_like(
            free_state.excess_momentum
        )
        self.active_neighbor_count = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._transition_neighborhood = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._transition_neighborhood_scratch = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._transition_columns = wp.zeros(
            self.res[:2], dtype=wp.int32, device=self.device
        )
        self._transition_columns_scratch = wp.zeros(
            self.res[:2], dtype=wp.int32, device=self.device
        )
        self._queue_source_mask = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._old_moment_sum = wp.zeros(10, dtype=float, device=self.device)
        self._new_moment_sum = wp.zeros(10, dtype=float, device=self.device)
        self._moment_delta = wp.zeros(10, dtype=float, device=self.device)
        self._old_represented_mass = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._old_represented_momentum = wp.zeros(3, dtype=float, device=self.device)
        self._pre_shift_represented_momentum = wp.zeros(3, dtype=float, device=self.device)
        self._final_represented_momentum = wp.zeros(3, dtype=float, device=self.device)
        self._velocity_shift = wp.zeros(3, dtype=float, device=self.device)
        self._committed_mass_sum = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._final_represented_mass = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._interface_density_sum = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._invalid_old_state_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._fresh_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._dead_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._fresh_phase_counts = wp.zeros(4, dtype=wp.int32, device=self.device)
        self._unresolved_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_donor_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._correction_recipient_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_moment_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_commit_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_momentum_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_committed_state_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._direct_liquid_gas_link_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._total_recipient_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._interface_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._local_recipient_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._local_interface_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._zero_recipient_interface_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self.capture_geometry(fluid_state, free_state)

    def capture_geometry(
        self, fluid_state: HomeLbmState, free_state: HomeFreeState
    ) -> None:
        self._validate_states(fluid_state, free_state)
        free_state.validate(
            fluid_state,
            allow_interface_endpoints=True,
            require_mass_fill_consistency=not self.allow_independent_mass,
        )
        wp.copy(self.previous_phi, fluid_state.solid_phi)

    def remap(
        self, fluid_state: HomeLbmState, free_state: HomeFreeState
    ) -> HomeFreeRigidTransitionDiagnostics:
        self._validate_states(fluid_state, free_state)
        wp.copy(self._moments_before, fluid_state.moments)
        wp.copy(self._flags_before, free_state.flags)
        wp.copy(self._fill_before, free_state.fill_level)
        wp.copy(self._mass_before, free_state.mass)
        wp.copy(self._excess_before, free_state.excess_mass)
        wp.copy(self._excess_momentum_before, free_state.excess_momentum)
        self._old_moment_sum.zero_()
        self._new_moment_sum.zero_()
        self._moment_delta.zero_()
        self._old_represented_mass.zero_()
        self._old_represented_momentum.zero_()
        self._pre_shift_represented_momentum.zero_()
        self._final_represented_momentum.zero_()
        self._velocity_shift.zero_()
        self._committed_mass_sum.zero_()
        self._final_represented_mass.zero_()
        self._interface_density_sum.zero_()
        self._invalid_old_state_count.zero_()
        self._fresh_count.zero_()
        self._dead_count.zero_()
        self._fresh_phase_counts.zero_()
        self._unresolved_count.zero_()
        self._invalid_donor_count.zero_()
        self._correction_recipient_count.zero_()
        self._invalid_moment_count.zero_()
        self._invalid_commit_count.zero_()
        self._invalid_momentum_count.zero_()
        self._total_recipient_count.zero_()
        self._interface_count.zero_()
        self._local_recipient_count.zero_()
        self._local_interface_count.zero_()
        self._transition_columns.zero_()
        self._zero_recipient_interface_count.zero_()
        periodic = tuple(int(value) for value in self.model.periodic)
        common = [*periodic, *self.res, self.stride]
        wp.launch(
            _accumulate_old_state_kernel,
            dim=self.res,
            inputs=[
                self._moments_before,
                self.previous_phi,
                self._flags_before,
                self._mass_before,
                self._excess_before,
                self._excess_momentum_before,
                self._old_moment_sum,
                self._old_represented_mass,
                self._old_represented_momentum,
                self._invalid_old_state_count,
                *common,
            ],
            device=self.device,
        )
        wp.launch(
            _classify_and_initialize_kernel,
            dim=self.res,
            inputs=[
                self._moments_before,
                self.previous_phi,
                fluid_state.solid_phi,
                self._flags_before,
                self._fill_before,
                self._moments_after,
                self.proposed_flags,
                self.proposed_fill,
                self._fresh_count,
                self._dead_count,
                self._fresh_phase_counts,
                self._unresolved_count,
                self._invalid_donor_count,
                *common,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid_old = int(self._invalid_old_state_count.numpy()[0])
        unresolved = int(self._unresolved_count.numpy()[0])
        invalid_donor = int(self._invalid_donor_count.numpy()[0])
        fresh_count = int(self._fresh_count.numpy()[0])
        dead_count = int(self._dead_count.numpy()[0])
        if invalid_old:
            raise RuntimeError(
                f"{invalid_old} previous solid masks and HOME-Free flags disagree"
            )
        if invalid_donor:
            raise RuntimeError(
                f"{invalid_donor} persistent donors have invalid phase flags"
            )
        if unresolved:
            raise RuntimeError(f"{unresolved} fresh cells have no persistent phase donor")
        old_mass = float(self._old_represented_mass.numpy()[0])
        old_momentum = self._old_represented_momentum.numpy().astype(np.float64)
        if fresh_count == 0 and dead_count == 0:
            wp.copy(self.previous_phi, fluid_state.solid_phi)
            return HomeFreeRigidTransitionDiagnostics(
                fresh_cell_count=0,
                dead_cell_count=0,
                fresh_liquid_cells=0,
                fresh_interface_cells=0,
                fresh_gas_cells=0,
                old_represented_mass=old_mass,
                final_represented_mass=old_mass,
                relative_mass_error=0.0,
                queued_mass=0.0,
                queued_mass_ratio=0.0,
                old_represented_momentum=tuple(float(value) for value in old_momentum),
                final_represented_momentum=tuple(float(value) for value in old_momentum),
                relative_momentum_error=0.0,
            )

        wp.launch(
            _accumulate_new_active_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.previous_phi,
                fluid_state.solid_phi,
                self.proposed_flags,
                self._new_moment_sum,
                self._correction_recipient_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            _compute_moment_delta_kernel,
            dim=10,
            inputs=[self._old_moment_sum, self._new_moment_sum, self._moment_delta],
            device=self.device,
        )
        wp.launch(
            _apply_moment_correction_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.previous_phi,
                fluid_state.solid_phi,
                self.proposed_flags,
                self._moment_delta,
                self._correction_recipient_count,
                self._invalid_moment_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(self._invalid_moment_count.numpy()[0]):
            raise FloatingPointError("moving-solid moment correction produced invalid density/moments")

        wp.launch(
            _commit_vof_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.previous_phi,
                fluid_state.solid_phi,
                self._flags_before,
                self._mass_before,
                self.proposed_flags,
                self.proposed_fill,
                self.committed_mass,
                self.committed_fill,
                self.committed_excess,
                self.committed_excess_momentum,
                self._committed_mass_sum,
                self._interface_density_sum,
                self._invalid_commit_count,
                int(self.allow_independent_mass),
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            _seed_transition_neighborhood_kernel,
            dim=self.res,
            inputs=[self.previous_phi, fluid_state.solid_phi, self._transition_neighborhood],
            device=self.device,
        )
        wp.launch(
            _seed_transition_columns_kernel,
            dim=self.res,
            inputs=[self.previous_phi, fluid_state.solid_phi, self._transition_columns],
            device=self.device,
        )
        for _ in range(_EXCESS_LOCALITY_RADIUS):
            wp.launch(
                _dilate_transition_neighborhood_kernel,
                dim=self.res,
                inputs=[
                    self._transition_neighborhood,
                    self._transition_neighborhood_scratch,
                    *periodic,
                    *self.res,
                ],
                device=self.device,
            )
            self._transition_neighborhood, self._transition_neighborhood_scratch = (
                self._transition_neighborhood_scratch,
                self._transition_neighborhood,
            )
            wp.launch(
                _dilate_transition_columns_kernel,
                dim=self.res[:2],
                inputs=[
                    self._transition_columns,
                    self._transition_columns_scratch,
                    periodic[0],
                    periodic[1],
                    self.res[0],
                    self.res[1],
                ],
                device=self.device,
            )
            self._transition_columns, self._transition_columns_scratch = (
                self._transition_columns_scratch,
                self._transition_columns,
            )
        wp.launch(
            _merge_transition_support_kernel,
            dim=self.res,
            inputs=[self._transition_neighborhood, self._transition_columns],
            device=self.device,
        )
        wp.launch(
            _count_new_recipients_kernel,
            dim=self.res,
            inputs=[
                self.proposed_flags,
                self.active_neighbor_count,
                self._total_recipient_count,
                self._interface_count,
                self._zero_recipient_interface_count,
                *periodic,
                *self.res,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(self._invalid_commit_count.numpy()[0]):
            raise FloatingPointError("moving-solid VOF commit produced invalid fields")
        if int(self._zero_recipient_interface_count.numpy()[0]):
            raise RuntimeError("moving-solid remap created an interface with no active recipient")

        committed_mass = float(self._committed_mass_sum.numpy()[0])
        queued_mass = old_mass - committed_mass
        interface_density = float(self._interface_density_sum.numpy()[0])
        recipient_total = int(self._total_recipient_count.numpy()[0])
        interface_total = int(self._interface_count.numpy()[0])
        local_recipient_total = 0
        local_interface_total = 0
        support_radius = _EXCESS_LOCALITY_RADIUS
        scale = max(abs(old_mass), 1.0)
        if interface_total == 0 or recipient_total == 0:
            if abs(queued_mass) > self.mass_tolerance * scale:
                raise RuntimeError(
                    "moving-solid remap has liquid-mass residual but no interface recipients"
                )
            share = 0.0
            queued_ratio = 0.0
        else:
            queued_ratio = abs(queued_mass) / interface_density
            if queued_ratio > self.max_queued_mass_ratio:
                raise RuntimeError(
                    f"moving-solid queued mass ratio {queued_ratio:.6e} exceeds "
                    f"limit {self.max_queued_mass_ratio:.6e}"
                )
            share = 0.0
            if queued_mass != 0.0:
                maximum_support_radius = max(self.res[0], self.res[1])
                while True:
                    self._local_recipient_count.zero_()
                    self._local_interface_count.zero_()
                    wp.launch(
                        _select_local_queue_sources_kernel,
                        dim=self.res,
                        inputs=[
                            self.committed_mass,
                            self.committed_fill,
                            self.proposed_flags,
                            self._transition_neighborhood,
                            self.active_neighbor_count,
                            self._queue_source_mask,
                            self._local_recipient_count,
                            self._local_interface_count,
                            queued_mass,
                            *periodic,
                            *self.res,
                        ],
                        device=self.device,
                    )
                    wp.synchronize_device(self.device)
                    local_recipient_total = int(
                        self._local_recipient_count.numpy()[0]
                    )
                    local_interface_total = int(
                        self._local_interface_count.numpy()[0]
                    )
                    if local_interface_total > 0 and local_recipient_total > 0:
                        break
                    if support_radius >= maximum_support_radius:
                        break
                    wp.launch(
                        _dilate_transition_columns_kernel,
                        dim=self.res[:2],
                        inputs=[
                            self._transition_columns,
                            self._transition_columns_scratch,
                            periodic[0],
                            periodic[1],
                            self.res[0],
                            self.res[1],
                        ],
                        device=self.device,
                    )
                    self._transition_columns, self._transition_columns_scratch = (
                        self._transition_columns_scratch,
                        self._transition_columns,
                    )
                    wp.launch(
                        _merge_transition_support_kernel,
                        dim=self.res,
                        inputs=[
                            self._transition_neighborhood,
                            self._transition_columns,
                        ],
                        device=self.device,
                    )
                    support_radius += 1
                if local_interface_total == 0 or local_recipient_total == 0:
                    raise RuntimeError(
                        "moving-solid remap has liquid-mass residual but no local "
                        f"interface capacity within radius {support_radius}"
                    )
                share = queued_mass / float(local_recipient_total)
                wp.launch(
                    _assign_local_excess_share_kernel,
                    dim=self.res,
                    inputs=[
                        self._moments_after,
                        self.proposed_flags,
                        self._queue_source_mask,
                        self.committed_excess,
                        self.committed_excess_momentum,
                        share,
                        self.res[1],
                        self.res[2],
                        self.stride,
                    ],
                    device=self.device,
                )

        wp.launch(
            _sum_represented_mass_kernel,
            dim=self.res,
            inputs=[
                self.committed_mass,
                self.committed_excess,
                self.active_neighbor_count,
                self._final_represented_mass,
            ],
            device=self.device,
        )
        wp.launch(
            _sum_represented_momentum_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.committed_mass,
                self.committed_excess,
                self.committed_excess_momentum,
                self.active_neighbor_count,
                self._pre_shift_represented_momentum,
                self._invalid_momentum_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        final_mass = float(self._final_represented_mass.numpy()[0])
        relative_error = abs(final_mass - old_mass) / scale
        if relative_error > self.mass_tolerance:
            raise RuntimeError(
                f"moving-solid represented mass error {relative_error:.6e} exceeds "
                f"tolerance {self.mass_tolerance:.6e}"
            )
        if final_mass <= 0.0:
            raise RuntimeError("moving-solid remap has no represented liquid mass")
        wp.launch(
            _compute_velocity_shift_kernel,
            dim=3,
            inputs=[
                self._old_represented_momentum,
                self._pre_shift_represented_momentum,
                self._final_represented_mass,
                self._velocity_shift,
            ],
            device=self.device,
        )
        wp.launch(
            _apply_galilean_shift_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.proposed_flags,
                self.committed_excess,
                self.committed_excess_momentum,
                self._velocity_shift,
                self._invalid_momentum_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            _sum_represented_momentum_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.committed_mass,
                self.committed_excess,
                self.committed_excess_momentum,
                self.active_neighbor_count,
                self._final_represented_momentum,
                self._invalid_momentum_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        if int(self._invalid_momentum_count.numpy()[0]):
            raise FloatingPointError("moving-solid represented-momentum correction failed")
        final_momentum = self._final_represented_momentum.numpy().astype(np.float64)
        momentum_scale = max(float(np.linalg.norm(old_momentum)), abs(old_mass), 1.0)
        momentum_error = float(np.linalg.norm(final_momentum - old_momentum)) / momentum_scale
        if momentum_error > self.momentum_tolerance:
            raise RuntimeError(
                f"moving-solid represented momentum error {momentum_error:.6e} exceeds "
                f"tolerance {self.momentum_tolerance:.6e}"
            )

        wp.copy(fluid_state.moments, self._moments_after)
        wp.copy(free_state.flags, self.proposed_flags)
        wp.copy(free_state.fill_level, self.committed_fill)
        wp.copy(free_state.mass, self.committed_mass)
        wp.copy(free_state.excess_mass, self.committed_excess)
        wp.copy(free_state.excess_momentum, self.committed_excess_momentum)
        self._validate_committed_state(fluid_state, free_state)
        wp.copy(self.previous_phi, fluid_state.solid_phi)
        return HomeFreeRigidTransitionDiagnostics(
            fresh_cell_count=fresh_count,
            dead_cell_count=dead_count,
            fresh_liquid_cells=int(self._fresh_phase_counts.numpy()[2]),
            fresh_interface_cells=int(self._fresh_phase_counts.numpy()[1]),
            fresh_gas_cells=int(self._fresh_phase_counts.numpy()[0]),
            old_represented_mass=old_mass,
            final_represented_mass=final_mass,
            relative_mass_error=relative_error,
            queued_mass=queued_mass,
            queued_mass_ratio=queued_ratio,
            old_represented_momentum=tuple(float(value) for value in old_momentum),
            final_represented_momentum=tuple(float(value) for value in final_momentum),
            relative_momentum_error=momentum_error,
            queue_source_count=local_interface_total if share != 0.0 else 0,
            queue_locality_radius=support_radius if share != 0.0 else 0,
        )

    def _validate_committed_state(
        self, fluid_state: HomeLbmState, free_state: HomeFreeState
    ) -> None:
        """Validate the remap transaction on-device without full-grid readback."""

        self._invalid_committed_state_count.zero_()
        self._direct_liquid_gas_link_count.zero_()
        periodic = tuple(int(value) for value in self.model.periodic)
        wp.launch(
            _validate_committed_state_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                free_state.mass,
                free_state.fill_level,
                free_state.excess_mass,
                free_state.excess_momentum,
                free_state.flags,
                self._invalid_committed_state_count,
                self._direct_liquid_gas_link_count,
                2.0e-6,
                int(self.allow_independent_mass),
                *periodic,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_committed_state_count.numpy()[0])
        if invalid:
            raise ValueError(
                f"moving-solid remap committed {invalid} invalid HOME-Free cells"
            )
        direct_links = int(self._direct_liquid_gas_link_count.numpy()[0])
        if direct_links:
            raise ValueError(
                "moving-solid remap violated the sharp interface invariant with "
                f"{direct_links} direct liquid-gas links"
            )

    def _validate_states(
        self, fluid_state: HomeLbmState, free_state: HomeFreeState
    ) -> None:
        if fluid_state.res != self.res or free_state.res != self.res:
            raise ValueError("moving-solid remapper state resolution changed")
        if fluid_state.device != self.device or free_state.device != self.device:
            raise ValueError("moving-solid remapper state device changed")
