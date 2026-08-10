# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for paper FSL link-wise mass exchange."""

from __future__ import annotations

import warp as wp

from ..core.kernels import _reconstruct_population


@wp.func
def _equilibrium_population(rho: float, velocity: wp.vec3, c: wp.vec3, weight: float) -> float:
    cu = wp.dot(c, velocity)
    speed_squared = wp.dot(velocity, velocity)
    return weight * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared)


@wp.kernel
def only_missing_stream_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    gas_density: float,
    streamed_moments: wp.array(dtype=float),
    gas_link_count: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    total_gas_link_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    destination_flag = flags[i, j, k]
    gas_link_count[i, j, k] = 0
    if destination_flag == 0:
        for component in range(10):
            streamed_moments[component * stride + cell] = moments[component * stride + cell]
        return

    local_rho = moments[cell]
    local_jx = moments[stride + cell]
    local_jy = moments[2 * stride + cell]
    local_jz = moments[3 * stride + cell]
    valid = destination_flag == 1 or destination_flag == 2
    valid = valid and wp.isfinite(local_rho) and local_rho > 0.0
    valid = valid and wp.isfinite(local_jx) and wp.isfinite(local_jy) and wp.isfinite(local_jz)
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        for component in range(10):
            streamed_moments[component * stride + cell] = moments[component * stride + cell]
        return

    velocity = wp.vec3(local_jx / local_rho, local_jy / local_rho, local_jz / local_rho)
    rho = float(0.0)
    jx = float(0.0)
    jy = float(0.0)
    jz = float(0.0)
    qxx = float(0.0)
    qyy = float(0.0)
    qzz = float(0.0)
    qxy = float(0.0)
    qxz = float(0.0)
    qyz = float(0.0)
    missing = int(0)
    for q in range(27):
        c = directions[q]
        source_i = (i - int(c[0]) + nx) % nx
        source_j = (j - int(c[1]) + ny) % ny
        source_k = (k - int(c[2]) + nz) % nz
        source_flag = flags[source_i, source_j, source_k]
        value = float(0.0)
        if source_flag == 0:
            if destination_flag == 2:
                wp.atomic_add(direct_liquid_gas_link_count, 0, 1)
            opposite = opposites[q]
            outgoing = _reconstruct_population(
                moments, directions[opposite], weights[opposite], cell, stride
            )
            value = _equilibrium_population(gas_density, velocity, c, weights[q])
            value += _equilibrium_population(
                gas_density, velocity, directions[opposite], weights[opposite]
            )
            value -= outgoing
            missing += 1
        elif source_flag == 1 or source_flag == 2:
            source = source_i * ny * nz + source_j * nz + source_k
            value = _reconstruct_population(moments, c, weights[q], source, stride)
        else:
            wp.atomic_add(invalid_cell_count, 0, 1)

        cx = c[0]
        cy = c[1]
        cz = c[2]
        rho += value
        jx += value * cx
        jy += value * cy
        jz += value * cz
        qxx += value * cx * cx
        qyy += value * cy * cy
        qzz += value * cz * cz
        qxy += value * cx * cy
        qxz += value * cx * cz
        qyz += value * cy * cz

    output_valid = wp.isfinite(rho) and rho > 0.0
    output_valid = output_valid and wp.isfinite(jx) and wp.isfinite(jy) and wp.isfinite(jz)
    output_valid = output_valid and wp.isfinite(qxx) and wp.isfinite(qyy) and wp.isfinite(qzz)
    output_valid = output_valid and wp.isfinite(qxy) and wp.isfinite(qxz) and wp.isfinite(qyz)
    if not output_valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        for component in range(10):
            streamed_moments[component * stride + cell] = moments[component * stride + cell]
        return

    streamed_moments[cell] = rho
    streamed_moments[stride + cell] = jx
    streamed_moments[2 * stride + cell] = jy
    streamed_moments[3 * stride + cell] = jz
    streamed_moments[4 * stride + cell] = qxx - rho / 3.0
    streamed_moments[5 * stride + cell] = qyy - rho / 3.0
    streamed_moments[6 * stride + cell] = qzz - rho / 3.0
    streamed_moments[7 * stride + cell] = qxy
    streamed_moments[8 * stride + cell] = qxz
    streamed_moments[9 * stride + cell] = qyz
    gas_link_count[i, j, k] = missing
    wp.atomic_add(total_gas_link_count, 0, missing)
    wp.atomic_min(min_density, 0, rho)
    wp.atomic_max(max_density, 0, rho)
    wp.atomic_max(max_speed_squared, 0, (jx * jx + jy * jy + jz * jz) / (rho * rho))


@wp.kernel
def fixed_topology_commit_kernel(
    advected_mass: wp.array3d(dtype=float),
    source_mass: wp.array3d(dtype=float),
    source_fill_level: wp.array3d(dtype=float),
    source_excess_mass: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    destination_moments: wp.array(dtype=float),
    candidate_mass: wp.array3d(dtype=float),
    candidate_fill_level: wp.array3d(dtype=float),
    candidate_excess_mass: wp.array3d(dtype=float),
    candidate_flags: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    phase_crossing_cell_count: wp.array(dtype=wp.int32),
    liquid_mass_normalization: wp.array(dtype=wp.float64),
    max_abs_liquid_mass_normalization: wp.array(dtype=float),
    initial_total_mass_cells: wp.array(dtype=wp.float64),
    candidate_total_mass_cells: wp.array(dtype=wp.float64),
    max_fill_change: wp.array(dtype=float),
    fill_epsilon: float,
    mass_density_tolerance: float,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = source_flags[i, j, k]
    mass = advected_mass[i, j, k]
    old_mass = source_mass[i, j, k]
    old_fill = source_fill_level[i, j, k]
    excess = source_excess_mass[i, j, k]
    rho = destination_moments[cell]
    initial_total_mass_cells[cell] = wp.float64(old_mass) + wp.float64(excess)
    liquid_mass_normalization[cell] = wp.float64(0.0)
    candidate_flags[i, j, k] = flag
    committed_mass = mass
    committed_fill = float(0.0)
    committed_excess = excess
    if flag == 0:
        committed_mass = 0.0
        if not wp.isfinite(old_mass) or not wp.isfinite(old_fill) or not wp.isfinite(excess):
            wp.atomic_add(invalid_cell_count, 0, 1)
    else:
        valid = (flag == 1 or flag == 2) and wp.isfinite(mass) and wp.isfinite(excess)
        valid = valid and wp.isfinite(rho) and rho > 0.0
        if not valid:
            wp.atomic_add(invalid_cell_count, 0, 1)
        elif flag == 2:
            normalization = mass - rho
            committed_mass = rho
            committed_fill = 1.0
            committed_excess = excess + normalization
            scale = wp.max(1.0, wp.abs(rho))
            if normalization != 0.0:
                liquid_mass_normalization[cell] = wp.float64(normalization)
                wp.atomic_max(
                    max_abs_liquid_mass_normalization, 0, wp.abs(normalization)
                )
            if wp.abs(normalization) > mass_density_tolerance * scale:
                wp.atomic_add(phase_crossing_cell_count, 0, 1)
            if not wp.isfinite(committed_excess):
                wp.atomic_add(invalid_cell_count, 0, 1)
        else:
            committed_fill = mass / rho
            if not wp.isfinite(committed_fill):
                wp.atomic_add(invalid_cell_count, 0, 1)
            elif committed_fill <= fill_epsilon or committed_fill >= 1.0 - fill_epsilon:
                wp.atomic_add(phase_crossing_cell_count, 0, 1)

    candidate_mass[i, j, k] = committed_mass
    candidate_fill_level[i, j, k] = committed_fill
    candidate_excess_mass[i, j, k] = committed_excess
    if wp.isfinite(committed_mass) and wp.isfinite(committed_excess):
        candidate_total_mass_cells[cell] = (
            wp.float64(committed_mass) + wp.float64(committed_excess)
        )
    else:
        candidate_total_mass_cells[cell] = wp.float64(0.0)
    if wp.isfinite(committed_fill) and wp.isfinite(old_fill):
        wp.atomic_max(max_fill_change, 0, wp.abs(committed_fill - old_fill))


@wp.kernel
def link_mass_exchange_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill_level: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    advected_mass: wp.array3d(dtype=float),
    invalid_cell_count: wp.array(dtype=wp.int32),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    max_abs_mass_delta: wp.array(dtype=float),
    initial_mass_cells: wp.array(dtype=wp.float64),
    advected_mass_cells: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    destination_flag = flags[i, j, k]
    destination_mass = mass[i, j, k]
    destination_fill = fill_level[i, j, k]
    valid = destination_flag >= 0 and destination_flag <= 2
    valid = valid and wp.isfinite(destination_mass) and wp.isfinite(destination_fill)
    valid = valid and destination_fill >= 0.0 and destination_fill <= 1.0
    if destination_flag != 0:
        valid = valid and wp.isfinite(moments[cell]) and moments[cell] > 0.0
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        advected_mass[i, j, k] = destination_mass
        if wp.isfinite(destination_mass):
            initial_mass_cells[cell] = wp.float64(destination_mass)
            advected_mass_cells[cell] = wp.float64(destination_mass)
        else:
            initial_mass_cells[cell] = wp.float64(0.0)
            advected_mass_cells[cell] = wp.float64(0.0)
        return
    if destination_flag == 0:
        advected_mass[i, j, k] = destination_mass
        initial_mass_cells[cell] = wp.float64(destination_mass)
        advected_mass_cells[cell] = wp.float64(destination_mass)
        return

    updated_mass = destination_mass
    for q in range(1, 27):
        c = directions[q]
        source_i = (i - int(c[0]) + nx) % nx
        source_j = (j - int(c[1]) + ny) % ny
        source_k = (k - int(c[2]) + nz) % nz
        source_flag = flags[source_i, source_j, source_k]
        if source_flag == 0:
            if destination_flag == 2:
                wp.atomic_add(direct_liquid_gas_link_count, 0, 1)
            continue
        if source_flag < 0 or source_flag > 2:
            wp.atomic_add(invalid_cell_count, 0, 1)
            continue
        source = source_i * ny * nz + source_j * nz + source_k
        alpha = float(1.0)
        if destination_flag == 1 and source_flag == 1:
            alpha = 0.5 * (destination_fill + fill_level[source_i, source_j, source_k])
        incoming = _reconstruct_population(moments, c, weights[q], source, stride)
        opposite = opposites[q]
        outgoing = _reconstruct_population(
            moments, directions[opposite], weights[opposite], cell, stride
        )
        updated_mass += alpha * (incoming - outgoing)

    if not wp.isfinite(updated_mass):
        wp.atomic_add(invalid_cell_count, 0, 1)
        advected_mass[i, j, k] = destination_mass
        return
    advected_mass[i, j, k] = updated_mass
    initial_mass_cells[cell] = wp.float64(destination_mass)
    advected_mass_cells[cell] = wp.float64(updated_mass)
    wp.atomic_max(max_abs_mass_delta, 0, wp.abs(updated_mass - destination_mass))
