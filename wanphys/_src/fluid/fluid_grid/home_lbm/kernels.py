# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels that validate the HOME moment representation on device."""

from __future__ import annotations

import warp as wp


@wp.func
def _reconstruct_population_components(
    rho: float,
    jx: float,
    jy: float,
    jz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    c: wp.vec3,
    weight: float,
) -> float:
    if rho <= 0.0:
        return 0.0

    ux = jx / rho
    uy = jy / rho
    uz = jz / rho
    cx = c[0]
    cy = c[1]
    cz = c[2]
    cu = cx * ux + cy * uy + cz * uz
    cj = cx * jx + cy * jy + cz * jz
    trace_s = sxx + syy + szz
    csc = (
        sxx * cx * cx
        + syy * cy * cy
        + szz * cz * cz
        + 2.0 * (sxy * cx * cy + sxz * cx * cz + syz * cy * cz)
    )
    h2a2 = csc - trace_s / 3.0
    su_x = sxx * ux + sxy * uy + sxz * uz
    su_y = sxy * ux + syy * uy + syz * uz
    su_z = sxz * ux + syz * uy + szz * uz
    u2 = ux * ux + uy * uy + uz * uz
    trace_a3_x = 2.0 * su_x + trace_s * ux - 2.0 * rho * u2 * ux
    trace_a3_y = 2.0 * su_y + trace_s * uy - 2.0 * rho * u2 * uy
    trace_a3_z = 2.0 * su_z + trace_s * uz - 2.0 * rho * u2 * uz
    a3_ccc = 3.0 * csc * cu - 2.0 * rho * cu * cu * cu
    h3a3 = a3_ccc - (cx * trace_a3_x + cy * trace_a3_y + cz * trace_a3_z)
    return weight * (rho + 3.0 * cj + 4.5 * h2a2 + 4.5 * h3a3)


@wp.func
def _reconstruct_population(
    moments: wp.array(dtype=float),
    c: wp.vec3,
    weight: float,
    cell: int,
    stride: int,
) -> float:
    return _reconstruct_population_components(
        moments[cell],
        moments[stride + cell],
        moments[2 * stride + cell],
        moments[3 * stride + cell],
        moments[4 * stride + cell],
        moments[5 * stride + cell],
        moments[6 * stride + cell],
        moments[7 * stride + cell],
        moments[8 * stride + cell],
        moments[9 * stride + cell],
        c,
        weight,
    )


@wp.func
def _equilibrium_population(
    rho: float,
    velocity: wp.vec3,
    c: wp.vec3,
    weight: float,
) -> float:
    cu = wp.dot(c, velocity)
    u2 = wp.dot(velocity, velocity)
    return weight * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


@wp.func
def _equilibrium_convective_population(
    moments: wp.array(dtype=float),
    c: wp.vec3,
    weight: float,
    cell: int,
    stride: int,
) -> float:
    rho = moments[cell]
    if rho <= 0.0:
        return 0.0
    ux = moments[stride + cell] / rho
    uy = moments[2 * stride + cell] / rho
    uz = moments[3 * stride + cell] / rho
    cu = c[0] * ux + c[1] * uy + c[2] * uz
    u2 = ux * ux + uy * uy + uz * uz
    h2 = cu * cu - u2 / 3.0
    h3 = cu * cu * cu - cu * u2
    return weight * rho * (4.5 * h2 + 4.5 * h3)


@wp.func
def _find_cut_link(
    cell: int,
    direction: int,
    cut_counts: wp.array(dtype=wp.int32),
    cut_offsets: wp.array(dtype=wp.int32),
    cut_directions: wp.array(dtype=wp.int32),
) -> int:
    index = cut_offsets[cell]
    end = index + cut_counts[cell]
    while index < end:
        if cut_directions[index] == direction:
            return index
        index += 1
    return -1


@wp.func
def _accumulate_moment_diagnostics(
    rho: float,
    jx: float,
    jy: float,
    jz: float,
    mxx: float,
    myy: float,
    mzz: float,
    mxy: float,
    mxz: float,
    myz: float,
    invalid_cell_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_nonequilibrium_stress_squared: wp.array(dtype=float),
):
    valid = wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(jx) and wp.isfinite(jy) and wp.isfinite(jz)
    valid = valid and wp.isfinite(mxx) and wp.isfinite(myy) and wp.isfinite(mzz)
    valid = valid and wp.isfinite(mxy) and wp.isfinite(mxz) and wp.isfinite(myz)
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        return

    ux = jx / rho
    uy = jy / rho
    uz = jz / rho
    speed_squared = ux * ux + uy * uy + uz * uz
    nxx = mxx / rho - ux * ux
    nyy = myy / rho - uy * uy
    nzz = mzz / rho - uz * uz
    nxy = mxy / rho - ux * uy
    nxz = mxz / rho - ux * uz
    nyz = myz / rho - uy * uz
    stress_squared = (
        nxx * nxx
        + nyy * nyy
        + nzz * nzz
        + 2.0 * (nxy * nxy + nxz * nxz + nyz * nyz)
    )
    wp.atomic_min(min_density, 0, rho)
    wp.atomic_max(max_density, 0, rho)
    wp.atomic_max(max_speed_squared, 0, speed_squared)
    wp.atomic_max(max_nonequilibrium_stress_squared, 0, stress_squared)


@wp.kernel
def reconstruct_populations_kernel(
    moments: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    populations: wp.array(dtype=float),
    stride: int,
):
    direction, cell = wp.tid()
    c = directions[direction]
    populations[direction * stride + cell] = _reconstruct_population(
        moments, c, weights[direction], cell, stride
    )


@wp.kernel
def extract_moments_kernel(
    populations: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    moments: wp.array(dtype=float),
    stride: int,
):
    cell = wp.tid()
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
    for direction in range(27):
        value = populations[direction * stride + cell]
        c = directions[direction]
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

    moments[cell] = rho
    moments[stride + cell] = jx
    moments[2 * stride + cell] = jy
    moments[3 * stride + cell] = jz
    moments[4 * stride + cell] = qxx - rho / 3.0
    moments[5 * stride + cell] = qyy - rho / 3.0
    moments[6 * stride + cell] = qzz - rho / 3.0
    moments[7 * stride + cell] = qxy
    moments[8 * stride + cell] = qxz
    moments[9 * stride + cell] = qyz


@wp.kernel
def collide_force_free_moments_kernel(
    moments_in: wp.array(dtype=float),
    active: wp.array3d(dtype=wp.int32),
    moments_out: wp.array(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    shear_omega: float,
    ny: int,
    nz: int,
    stride: int,
):
    """Apply the force-free HOME collision to an explicit active-node mask."""

    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    if active[i, j, k] == 0:
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[
                component * stride + cell
            ]
        return
    rho = moments_in[cell]
    valid = wp.isfinite(rho) and rho > 0.0
    for component in range(1, 10):
        valid = valid and wp.isfinite(moments_in[component * stride + cell])
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[
                component * stride + cell
            ]
        return

    ux = moments_in[stride + cell] / rho
    uy = moments_in[2 * stride + cell] / rho
    uz = moments_in[3 * stride + cell] / rho
    sxx = moments_in[4 * stride + cell] / rho
    syy = moments_in[5 * stride + cell] / rho
    szz = moments_in[6 * stride + cell] / rho
    sxy = moments_in[7 * stride + cell] / rho
    sxz = moments_in[8 * stride + cell] / rho
    syz = moments_in[9 * stride + cell] / rho
    trace_s = sxx + syy + szz
    u2 = ux * ux + uy * uy + uz * uz
    keep = 1.0 - shear_omega
    moments_out[cell] = rho
    moments_out[stride + cell] = moments_in[stride + cell]
    moments_out[2 * stride + cell] = moments_in[2 * stride + cell]
    moments_out[3 * stride + cell] = moments_in[3 * stride + cell]
    moments_out[4 * stride + cell] = rho * (
        ux * ux + keep * (sxx - trace_s / 3.0 - ux * ux + u2 / 3.0)
    )
    moments_out[5 * stride + cell] = rho * (
        uy * uy + keep * (syy - trace_s / 3.0 - uy * uy + u2 / 3.0)
    )
    moments_out[6 * stride + cell] = rho * (
        uz * uz + keep * (szz - trace_s / 3.0 - uz * uz + u2 / 3.0)
    )
    moments_out[7 * stride + cell] = rho * (
        ux * uy + keep * (sxy - ux * uy)
    )
    moments_out[8 * stride + cell] = rho * (
        ux * uz + keep * (sxz - ux * uz)
    )
    moments_out[9 * stride + cell] = rho * (
        uy * uz + keep * (syz - uy * uz)
    )


@wp.kernel
def periodic_convective_momentum_increment_kernel(
    moments: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    density_increment: wp.array3d(dtype=float),
    increment: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    delta_rho = float(0.0)
    delta = wp.vec3(0.0)
    for direction in range(27):
        c = directions[direction]
        source_i = (i - int(c[0]) + nx) % nx
        source_j = (j - int(c[1]) + ny) % ny
        source_k = (k - int(c[2]) + nz) % nz
        source = source_i * ny * nz + source_j * nz + source_k
        value = _equilibrium_convective_population(
            moments, c, weights[direction], source, stride
        )
        delta_rho += value
        delta += value * c
    density_increment[i, j, k] = delta_rho
    increment[cell] = delta[0]
    increment[stride + cell] = delta[1]
    increment[2 * stride + cell] = delta[2]


@wp.kernel
def initialize_uniform_kernel(
    moments: wp.array(dtype=float),
    rho: float,
    velocity: wp.vec3,
    stride: int,
):
    cell = wp.tid()
    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]
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


@wp.kernel
def initialize_hydrostatic_kernel(
    moments: wp.array(dtype=float),
    acceleration: wp.vec3,
    log_density_ratio: wp.vec3,
    density_scale: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    exponent = (
        (float(i) + 0.5 - 0.5 * float(nx)) * log_density_ratio[0]
        + (float(j) + 0.5 - 0.5 * float(ny)) * log_density_ratio[1]
        + (float(k) + 0.5 - 0.5 * float(nz)) * log_density_ratio[2]
    )
    rho = density_scale * wp.exp(exponent)
    moments[cell] = rho
    moments[stride + cell] = 0.5 * rho * acceleration[0]
    moments[2 * stride + cell] = 0.5 * rho * acceleration[1]
    moments[3 * stride + cell] = 0.5 * rho * acceleration[2]
    for component in range(4, 10):
        moments[component * stride + cell] = 0.0


@wp.kernel
def stream_collide_kernel(
    moments_in: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    free_surface_flags: wp.array3d(dtype=wp.int32),
    gas_density_field: wp.array3d(dtype=float),
    force_density: wp.array3d(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    cut_counts: wp.array(dtype=wp.int32),
    cut_offsets: wp.array(dtype=wp.int32),
    cut_directions: wp.array(dtype=wp.int32),
    cut_fraction: wp.array(dtype=float),
    cut_wall_velocity: wp.array(dtype=wp.vec3),
    cut_impulse: wp.array(dtype=wp.vec3),
    missing_cut_link_count: wp.array(dtype=wp.int32),
    unsupported_interpolation_count: wp.array(dtype=wp.int32),
    wall_confined_interpolation_count: wp.array(dtype=wp.int32),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    invalid_gas_density_count: wp.array(dtype=wp.int32),
    gas_boundary_impulse: wp.array(dtype=wp.float64),
    cut_link_frame_correction: wp.array(dtype=wp.float64),
    invalid_cell_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_nonequilibrium_stress_squared: wp.array(dtype=float),
    remove_reference_pressure: int,
    pressure_reference_scale: float,
    pressure_reference_log_ratio: wp.vec3,
    has_cut_links: int,
    has_free_surface: int,
    has_gas_density_field: int,
    has_force_density: int,
    gas_density: float,
    moments_out: wp.array(dtype=float),
    shear_omega: float,
    acceleration: wp.vec3,
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
    if solid_phi[i, j, k] < 0.0:
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[component * stride + cell]
        return
    cell_flag = int(2)
    if has_free_surface != 0:
        cell_flag = free_surface_flags[i, j, k]
        if cell_flag == 0:
            for component in range(10):
                moments_out[component * stride + cell] = moments_in[component * stride + cell]
            return
    boundary_gas_density = gas_density
    if has_gas_density_field != 0:
        boundary_gas_density = gas_density_field[i, j, k]
        if cell_flag == 1 and (
            not wp.isfinite(boundary_gas_density) or boundary_gas_density <= 0.0
        ):
            wp.atomic_add(invalid_gas_density_count, 0, 1)
            for component in range(10):
                moments_out[component * stride + cell] = moments_in[component * stride + cell]
            return
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

    for direction in range(27):
        c = directions[direction]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        hit_wall = int(0)
        if source_i < 0 or source_i >= nx:
            if periodic_x != 0:
                source_i = (source_i + nx) % nx
            else:
                hit_wall = 1
        if source_j < 0 or source_j >= ny:
            if periodic_y != 0:
                source_j = (source_j + ny) % ny
            else:
                hit_wall = 1
        if source_k < 0 or source_k >= nz:
            if periodic_z != 0:
                source_k = (source_k + nz) % nz
            else:
                hit_wall = 1

        value = float(0.0)
        hit_solid = int(0)
        if hit_wall == 0 and solid_phi[source_i, source_j, source_k] < 0.0:
            hit_solid = 1

        if hit_wall != 0:
            opposite = opposites[direction]
            value = _reconstruct_population(
                moments_in, directions[opposite], weights[opposite], cell, stride
            )
        elif hit_solid != 0:
            cut_index = int(-1)
            if has_cut_links != 0:
                cut_index = _find_cut_link(
                    cell, direction, cut_counts, cut_offsets, cut_directions
                )
            if cut_index < 0:
                wp.atomic_add(missing_cut_link_count, 0, 1)
                opposite = opposites[direction]
                value = _reconstruct_population(
                    moments_in, directions[opposite], weights[opposite], cell, stride
                )
            else:
                rho_x = moments_in[cell]
                up = cut_wall_velocity[cut_index]
                opposite = opposites[direction]
                incoming_c = directions[opposite]
                incoming = _reconstruct_population(
                    moments_in, incoming_c, weights[opposite], cell, stride
                )
                outgoing = _reconstruct_population(
                    moments_in, c, weights[direction], cell, stride
                )
                q = cut_fraction[cut_index]
                wall_correction = 6.0 * weights[direction] * rho_x * wp.dot(c, up)
                include_cut_impulse = int(1)
                if q < 0.5:
                    support_i = i + int(c[0])
                    support_j = j + int(c[1])
                    support_k = k + int(c[2])
                    support_valid = int(1)
                    if support_i < 0 or support_i >= nx:
                        if periodic_x != 0:
                            support_i = (support_i + nx) % nx
                        else:
                            support_valid = 0
                    if support_j < 0 or support_j >= ny:
                        if periodic_y != 0:
                            support_j = (support_j + ny) % ny
                        else:
                            support_valid = 0
                    if support_k < 0 or support_k >= nz:
                        if periodic_z != 0:
                            support_k = (support_k + nz) % nz
                        else:
                            support_valid = 0
                    if support_valid != 0 and solid_phi[support_i, support_j, support_k] >= 0.0:
                        support = support_i * ny * nz + support_j * nz + support_k
                        support_incoming = _reconstruct_population(
                            moments_in, incoming_c, weights[opposite], support, stride
                        )
                        value = (
                            2.0 * q * incoming
                            + (1.0 - 2.0 * q) * support_incoming
                            + wall_correction
                        )
                    elif support_valid == 0:
                        wp.atomic_add(wall_confined_interpolation_count, 0, 1)
                        value = incoming + wall_correction
                        # The local reflection only closes fluid populations.
                        # Its unresolved wall-gap load belongs to lubrication.
                        include_cut_impulse = 0
                    else:
                        wp.atomic_add(unsupported_interpolation_count, 0, 1)
                        value = incoming + wall_correction
                else:
                    inverse_distance = 1.0 / (2.0 * q)
                    value = (
                        inverse_distance * (incoming + wall_correction)
                        + (1.0 - inverse_distance) * outgoing
                    )
                if include_cut_impulse != 0:
                    impulse = incoming * (incoming_c - up) - value * (c - up)
                    frame_correction = (value - incoming) * up
                    wp.atomic_add(
                        cut_link_frame_correction,
                        0,
                        wp.float64(frame_correction[0]),
                    )
                    wp.atomic_add(
                        cut_link_frame_correction,
                        1,
                        wp.float64(frame_correction[1]),
                    )
                    wp.atomic_add(
                        cut_link_frame_correction,
                        2,
                        wp.float64(frame_correction[2]),
                    )
                    if remove_reference_pressure != 0:
                        exponent = (
                            (float(i) + 0.5 - 0.5 * float(nx))
                            * pressure_reference_log_ratio[0]
                            + (float(j) + 0.5 - 0.5 * float(ny))
                            * pressure_reference_log_ratio[1]
                            + (float(k) + 0.5 - 0.5 * float(nz))
                            * pressure_reference_log_ratio[2]
                        )
                        reference_rho = pressure_reference_scale * wp.exp(exponent)
                        impulse += 2.0 * weights[direction] * reference_rho * c
                    cut_impulse[cut_index] = impulse
                else:
                    cut_impulse[cut_index] = wp.vec3(0.0)
        elif has_free_surface != 0 and free_surface_flags[source_i, source_j, source_k] == 0:
            if cell_flag == 2:
                wp.atomic_add(direct_liquid_gas_link_count, 0, 1)
            opposite = opposites[direction]
            local_rho = moments_in[cell]
            local_velocity = wp.vec3(
                moments_in[stride + cell] / local_rho,
                moments_in[2 * stride + cell] / local_rho,
                moments_in[3 * stride + cell] / local_rho,
            )
            outgoing = _reconstruct_population(
                moments_in,
                directions[opposite],
                weights[opposite],
                cell,
                stride,
            )
            value = (
                _equilibrium_population(
                    boundary_gas_density, local_velocity, c, weights[direction]
                )
                + _equilibrium_population(
                    boundary_gas_density,
                    local_velocity,
                    directions[opposite],
                    weights[opposite],
                )
                - outgoing
            )
            boundary_impulse = (value + outgoing) * c
            wp.atomic_add(
                gas_boundary_impulse,
                0,
                wp.float64(boundary_impulse[0]),
            )
            wp.atomic_add(
                gas_boundary_impulse,
                1,
                wp.float64(boundary_impulse[1]),
            )
            wp.atomic_add(
                gas_boundary_impulse,
                2,
                wp.float64(boundary_impulse[2]),
            )
        else:
            source = source_i * ny * nz + source_j * nz + source_k
            value = _reconstruct_population(moments_in, c, weights[direction], source, stride)
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

    if rho <= 0.0:
        for component in range(10):
            moments_out[component * stride + cell] = 0.0
        _accumulate_moment_diagnostics(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            invalid_cell_count,
            min_density,
            max_density,
            max_speed_squared,
            max_nonequilibrium_stress_squared,
        )
        return

    fx = rho * acceleration[0]
    fy = rho * acceleration[1]
    fz = rho * acceleration[2]
    if has_force_density != 0:
        cell_force = force_density[i, j, k]
        fx = cell_force[0]
        fy = cell_force[1]
        fz = cell_force[2]
    ux = (jx + 0.5 * fx) / rho
    uy = (jy + 0.5 * fy) / rho
    uz = (jz + 0.5 * fz) / rho
    sxx = qxx / rho - 1.0 / 3.0
    syy = qyy / rho - 1.0 / 3.0
    szz = qzz / rho - 1.0 / 3.0
    sxy = qxy / rho
    sxz = qxz / rho
    syz = qyz / rho
    trace_s = sxx + syy + szz
    u2 = ux * ux + uy * uy + uz * uz
    keep = 1.0 - shear_omega
    post_sxx = ux * ux + keep * (sxx - trace_s / 3.0 - ux * ux + u2 / 3.0)
    post_syy = uy * uy + keep * (syy - trace_s / 3.0 - uy * uy + u2 / 3.0)
    post_szz = uz * uz + keep * (szz - trace_s / 3.0 - uz * uz + u2 / 3.0)
    post_sxy = ux * uy + keep * (sxy - ux * uy)
    post_sxz = ux * uz + keep * (sxz - ux * uz)
    post_syz = uy * uz + keep * (syz - uy * uz)

    force_power = fx * ux + fy * uy + fz * uz
    post_sxx += (fx * ux + keep * (3.0 * fx * ux - force_power) / 3.0) / rho
    post_syy += (fy * uy + keep * (3.0 * fy * uy - force_power) / 3.0) / rho
    post_szz += (fz * uz + keep * (3.0 * fz * uz - force_power) / 3.0) / rho
    force_shear = (1.0 - 0.5 * shear_omega) / rho
    post_sxy += force_shear * (fx * uy + fy * ux)
    post_sxz += force_shear * (fx * uz + fz * ux)
    post_syz += force_shear * (fy * uz + fz * uy)

    moments_out[cell] = rho
    moments_out[stride + cell] = jx + fx
    moments_out[2 * stride + cell] = jy + fy
    moments_out[3 * stride + cell] = jz + fz
    moments_out[4 * stride + cell] = rho * post_sxx
    moments_out[5 * stride + cell] = rho * post_syy
    moments_out[6 * stride + cell] = rho * post_szz
    moments_out[7 * stride + cell] = rho * post_sxy
    moments_out[8 * stride + cell] = rho * post_sxz
    moments_out[9 * stride + cell] = rho * post_syz
    _accumulate_moment_diagnostics(
        rho,
        jx + fx,
        jy + fy,
        jz + fz,
        rho * post_sxx,
        rho * post_syy,
        rho * post_szz,
        rho * post_sxy,
        rho * post_sxz,
        rho * post_syz,
        invalid_cell_count,
        min_density,
        max_density,
        max_speed_squared,
        max_nonequilibrium_stress_squared,
    )


@wp.kernel
def diagnose_moments_kernel(
    moments: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    free_surface_flags: wp.array3d(dtype=wp.int32),
    gas_density_field: wp.array3d(dtype=float),
    has_free_surface: int,
    has_gas_density_field: int,
    invalid_cell_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_nonequilibrium_stress_squared: wp.array(dtype=float),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    invalid_gas_density_count: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    stride: int,
    ny: int,
    nz: int,
):
    cell = wp.tid()
    i = cell // (ny * nz)
    remainder = cell - i * ny * nz
    j = remainder // nz
    k = remainder - j * nz
    if solid_phi[i, j, k] < 0.0:
        return
    if has_free_surface != 0 and free_surface_flags[i, j, k] == 0:
        return
    if (
        has_gas_density_field != 0
        and free_surface_flags[i, j, k] == 1
        and (
            not wp.isfinite(gas_density_field[i, j, k])
            or gas_density_field[i, j, k] <= 0.0
        )
    ):
        wp.atomic_add(invalid_gas_density_count, 0, 1)
    if has_free_surface != 0 and free_surface_flags[i, j, k] == 2:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    neighbor_i = i + di
                    neighbor_j = j + dj
                    neighbor_k = k + dk
                    valid = int(1)
                    if neighbor_i < 0 or neighbor_i >= nx:
                        if periodic_x != 0:
                            neighbor_i = (neighbor_i + nx) % nx
                        else:
                            valid = 0
                    if neighbor_j < 0 or neighbor_j >= ny:
                        if periodic_y != 0:
                            neighbor_j = (neighbor_j + ny) % ny
                        else:
                            valid = 0
                    if neighbor_k < 0 or neighbor_k >= nz:
                        if periodic_z != 0:
                            neighbor_k = (neighbor_k + nz) % nz
                        else:
                            valid = 0
                    if valid != 0 and free_surface_flags[neighbor_i, neighbor_j, neighbor_k] == 0:
                        wp.atomic_add(direct_liquid_gas_link_count, 0, 1)
    _accumulate_moment_diagnostics(
        moments[cell],
        moments[stride + cell],
        moments[2 * stride + cell],
        moments[3 * stride + cell],
        moments[4 * stride + cell],
        moments[5 * stride + cell],
        moments[6 * stride + cell],
        moments[7 * stride + cell],
        moments[8 * stride + cell],
        moments[9 * stride + cell],
        invalid_cell_count,
        min_density,
        max_density,
        max_speed_squared,
        max_nonequilibrium_stress_squared,
    )


@wp.kernel
def pack_diagnostics_kernel(
    invalid_cell_count: wp.array(dtype=wp.int32),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    invalid_gas_density_count: wp.array(dtype=wp.int32),
    missing_cut_link_count: wp.array(dtype=wp.int32),
    unsupported_interpolation_count: wp.array(dtype=wp.int32),
    wall_confined_interpolation_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_nonequilibrium_stress_squared: wp.array(dtype=float),
    packed_counts: wp.array(dtype=wp.int32),
    packed_values: wp.array(dtype=float),
):
    packed_counts[0] = invalid_cell_count[0]
    packed_counts[1] = direct_liquid_gas_link_count[0]
    packed_counts[2] = invalid_gas_density_count[0]
    packed_counts[3] = missing_cut_link_count[0]
    packed_counts[4] = unsupported_interpolation_count[0]
    packed_counts[5] = wall_confined_interpolation_count[0]
    packed_values[0] = min_density[0]
    packed_values[1] = max_density[0]
    packed_values[2] = max_speed_squared[0]
    packed_values[3] = max_nonequilibrium_stress_squared[0]
