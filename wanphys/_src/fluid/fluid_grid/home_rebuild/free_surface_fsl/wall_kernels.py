# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for closed-wall HOME-Free composition."""

import warp as wp

from ..core.kernels import _collide_stress, _reconstruct_population, _record_diagnostics
from .kernels import _equilibrium_population


@wp.kernel
def wall_only_missing_stream_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    gas_density: float,
    gas_density_field: wp.array3d(dtype=float),
    use_gas_density_field: int,
    streamed_moments: wp.array(dtype=float),
    gas_link_count: wp.array3d(dtype=wp.int32),
    wall_link_count: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    direct_link_count: wp.array(dtype=wp.int32),
    total_gas_links: wp.array(dtype=wp.int32),
    total_wall_links: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    gas_link_count[i, j, k] = 0
    wall_link_count[i, j, k] = 0
    destination_flag = flags[i, j, k]
    if solid[i, j, k] != 0 or destination_flag == 0:
        for component in range(10):
            streamed_moments[component * stride + cell] = moments[component * stride + cell]
        return
    local_rho = moments[cell]
    local_jx = moments[stride + cell]
    local_jy = moments[2 * stride + cell]
    local_jz = moments[3 * stride + cell]
    valid = (destination_flag == 1 or destination_flag == 2) and local_rho > 0.0
    valid = valid and wp.isfinite(local_rho) and wp.isfinite(local_jx)
    valid = valid and wp.isfinite(local_jy) and wp.isfinite(local_jz)
    local_gas_density = gas_density
    if use_gas_density_field != 0:
        local_gas_density = gas_density_field[i, j, k]
    valid = valid and wp.isfinite(local_gas_density) and local_gas_density > 0.0
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
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
    gas_links = int(0)
    wall_links = int(0)
    for q in range(27):
        c = directions[q]
        si = (i - int(c[0]) + nx) % nx
        sj = (j - int(c[1]) + ny) % ny
        sk = (k - int(c[2]) + nz) % nz
        value = float(0.0)
        if solid[si, sj, sk] != 0:
            opposite = opposites[q]
            value = _reconstruct_population(
                moments, directions[opposite], weights[opposite], cell, stride
            )
            wall_links += 1
        else:
            source_flag = flags[si, sj, sk]
            if source_flag == 0:
                if destination_flag == 2:
                    wp.atomic_add(direct_link_count, 0, 1)
                opposite = opposites[q]
                outgoing = _reconstruct_population(
                    moments, directions[opposite], weights[opposite], cell, stride
                )
                value = _equilibrium_population(
                    local_gas_density, velocity, c, weights[q]
                )
                value += _equilibrium_population(
                    local_gas_density,
                    velocity,
                    directions[opposite],
                    weights[opposite],
                )
                value -= outgoing
                gas_links += 1
            elif source_flag == 1 or source_flag == 2:
                source = si * ny * nz + sj * nz + sk
                value = _reconstruct_population(moments, c, weights[q], source, stride)
            else:
                wp.atomic_add(invalid_count, 0, 1)
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
        wp.atomic_add(invalid_count, 0, 1)
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
    gas_link_count[i, j, k] = gas_links
    wall_link_count[i, j, k] = wall_links
    wp.atomic_add(total_gas_links, 0, gas_links)
    wp.atomic_add(total_wall_links, 0, wall_links)


@wp.kernel
def wall_mass_exchange_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    advected_mass: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    direct_link_count: wp.array(dtype=wp.int32),
    initial_mass_cells: wp.array(dtype=wp.float64),
    final_mass_cells: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    destination_mass = mass[i, j, k]
    initial_mass_cells[cell] = wp.float64(destination_mass)
    if solid[i, j, k] != 0:
        advected_mass[i, j, k] = 0.0
        final_mass_cells[cell] = wp.float64(0.0)
        if destination_mass != 0.0:
            wp.atomic_add(invalid_count, 0, 1)
        return
    destination_flag = flags[i, j, k]
    if destination_flag == 0:
        advected_mass[i, j, k] = destination_mass
        final_mass_cells[cell] = wp.float64(destination_mass)
        return
    updated_mass = destination_mass
    for q in range(1, 27):
        c = directions[q]
        si = (i - int(c[0]) + nx) % nx
        sj = (j - int(c[1]) + ny) % ny
        sk = (k - int(c[2]) + nz) % nz
        if solid[si, sj, sk] != 0:
            continue
        source_flag = flags[si, sj, sk]
        if source_flag == 0:
            if destination_flag == 2:
                wp.atomic_add(direct_link_count, 0, 1)
            continue
        source = si * ny * nz + sj * nz + sk
        alpha = float(1.0)
        if destination_flag == 1 and source_flag == 1:
            alpha = 0.5 * (fill[i, j, k] + fill[si, sj, sk])
        incoming = _reconstruct_population(moments, c, weights[q], source, stride)
        opposite = opposites[q]
        outgoing = _reconstruct_population(
            moments, directions[opposite], weights[opposite], cell, stride
        )
        updated_mass += alpha * (incoming - outgoing)
    if not wp.isfinite(updated_mass):
        wp.atomic_add(invalid_count, 0, 1)
        updated_mass = destination_mass
    advected_mass[i, j, k] = updated_mass
    final_mass_cells[cell] = wp.float64(updated_mass)


@wp.kernel
def wall_active_collide_kernel(
    moments_in: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    moments_out: wp.array(dtype=float),
    shear_omega: float,
    acceleration: wp.vec3,
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    if solid[i, j, k] != 0 or flags[i, j, k] == 0:
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[component * stride + cell]
        return
    rho = moments_in[cell]
    jx = moments_in[stride + cell]
    jy = moments_in[2 * stride + cell]
    jz = moments_in[3 * stride + cell]
    mxx = moments_in[4 * stride + cell]
    myy = moments_in[5 * stride + cell]
    mzz = moments_in[6 * stride + cell]
    mxy = moments_in[7 * stride + cell]
    mxz = moments_in[8 * stride + cell]
    myz = moments_in[9 * stride + cell]
    valid = wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(jx) and wp.isfinite(jy) and wp.isfinite(jz)
    valid = valid and wp.isfinite(mxx) and wp.isfinite(myy) and wp.isfinite(mzz)
    valid = valid and wp.isfinite(mxy) and wp.isfinite(mxz) and wp.isfinite(myz)
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        return
    stress = _collide_stress(
        rho, jx, jy, jz, mxx, myy, mzz, mxy, mxz, myz, shear_omega, acceleration
    )
    out_jx = jx + rho * acceleration[0]
    out_jy = jy + rho * acceleration[1]
    out_jz = jz + rho * acceleration[2]
    moments_out[cell] = rho
    moments_out[stride + cell] = out_jx
    moments_out[2 * stride + cell] = out_jy
    moments_out[3 * stride + cell] = out_jz
    moments_out[4 * stride + cell] = stress[0, 0]
    moments_out[5 * stride + cell] = stress[1, 1]
    moments_out[6 * stride + cell] = stress[2, 2]
    moments_out[7 * stride + cell] = stress[0, 1]
    moments_out[8 * stride + cell] = stress[0, 2]
    moments_out[9 * stride + cell] = stress[1, 2]
    _record_diagnostics(
        rho, out_jx, out_jy, out_jz,
        stress[0, 0], stress[1, 1], stress[2, 2],
        stress[0, 1], stress[0, 2], stress[1, 2],
        invalid_count, min_density, max_density, max_speed_squared, max_stress_squared,
    )


@wp.kernel
def diagnose_wall_active_state_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0 or flags[i, j, k] == 0:
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    jx = moments[stride + cell]
    jy = moments[2 * stride + cell]
    jz = moments[3 * stride + cell]
    mxx = moments[4 * stride + cell]
    myy = moments[5 * stride + cell]
    mzz = moments[6 * stride + cell]
    mxy = moments[7 * stride + cell]
    mxz = moments[8 * stride + cell]
    myz = moments[9 * stride + cell]
    valid = wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(jx) and wp.isfinite(jy) and wp.isfinite(jz)
    valid = valid and wp.isfinite(mxx) and wp.isfinite(myy) and wp.isfinite(mzz)
    valid = valid and wp.isfinite(mxy) and wp.isfinite(mxz) and wp.isfinite(myz)
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        return
    _record_diagnostics(
        rho, jx, jy, jz, mxx, myy, mzz, mxy, mxz, myz,
        invalid_count, min_density, max_density, max_speed_squared, max_stress_squared,
    )


@wp.kernel
def wall_count_active_neighbors_kernel(
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    counts: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        counts[i, j, k] = 0
        return
    count = int(0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if solid[ni, nj, nk] == 0 and flags[ni, nj, nk] != 0:
            count += 1
    counts[i, j, k] = count


@wp.kernel
def wall_receive_excess_kernel(
    advected_mass: wp.array3d(dtype=float),
    excess: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    recipient_count: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    working_mass: wp.array3d(dtype=float),
    stranded: wp.array3d(dtype=float),
    initial_cells: wp.array(dtype=wp.float64),
    invalid_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    own_excess = excess[i, j, k]
    if solid[i, j, k] != 0:
        working_mass[i, j, k] = 0.0
        stranded[i, j, k] = 0.0
        initial_cells[cell] = wp.float64(0.0)
        if advected_mass[i, j, k] != 0.0 or own_excess != 0.0:
            wp.atomic_add(invalid_count, 0, 1)
        return
    mass = advected_mass[i, j, k]
    local_stranded = float(0.0)
    if recipient_count[i, j, k] == 0:
        local_stranded = own_excess
    if flags[i, j, k] != 0:
        for q in range(1, 27):
            c = directions[q]
            ni = (i + int(c[0]) + nx) % nx
            nj = (j + int(c[1]) + ny) % ny
            nk = (k + int(c[2]) + nz) % nz
            count = recipient_count[ni, nj, nk]
            if solid[ni, nj, nk] == 0 and count > 0:
                mass += excess[ni, nj, nk] / float(count)
    if not wp.isfinite(mass) or not wp.isfinite(own_excess):
        wp.atomic_add(invalid_count, 0, 1)
        initial_cells[cell] = wp.float64(0.0)
    else:
        initial_cells[cell] = wp.float64(advected_mass[i, j, k]) + wp.float64(own_excess)
    working_mass[i, j, k] = mass
    stranded[i, j, k] = local_stranded


@wp.kernel
def wall_mark_candidates_kernel(
    moments: wp.array(dtype=float),
    working_mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    candidates: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    tolerance: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    candidates[i, j, k] = 0
    if solid[i, j, k] != 0 or flags[i, j, k] != 1:
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    mass = working_mass[i, j, k]
    has_liquid = int(0)
    has_gas = int(0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if solid[ni, nj, nk] == 0:
            neighbor = flags[ni, nj, nk]
            has_liquid = wp.max(has_liquid, int(neighbor == 2))
            has_gas = wp.max(has_gas, int(neighbor == 0))
    if not wp.isfinite(rho) or rho <= 0.0 or not wp.isfinite(mass):
        wp.atomic_add(invalid_count, 0, 1)
    elif mass > rho + tolerance or has_gas == 0:
        candidates[i, j, k] = 1
    elif mass < -tolerance or has_liquid == 0:
        candidates[i, j, k] = 2


@wp.kernel
def wall_resolve_growth_kernel(
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    candidates: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    transitions: wp.array3d(dtype=wp.int32),
    cancelled_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        transitions[i, j, k] = 0
        return
    candidate = candidates[i, j, k]
    neighbor_growth = int(0)
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if solid[ni, nj, nk] == 0:
            neighbor_growth = wp.max(neighbor_growth, int(candidates[ni, nj, nk] == 1))
    if candidate == 2 and neighbor_growth == 1:
        transitions[i, j, k] = 0
        wp.atomic_add(cancelled_count, 0, 1)
    elif flags[i, j, k] == 0 and neighbor_growth == 1:
        transitions[i, j, k] = 3
    else:
        transitions[i, j, k] = candidate


@wp.kernel
def wall_apply_transitions_kernel(
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    final_flags: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        final_flags[i, j, k] = 0
        return
    transition = transitions[i, j, k]
    flag = flags[i, j, k]
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
            if solid[ni, nj, nk] == 0 and transitions[ni, nj, nk] == 2:
                flag = 1
    final_flags[i, j, k] = flag


@wp.kernel
def wall_initialize_fresh_kernel(
    source_moments: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    final_flags: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    destination_moments: wp.array(dtype=float),
    missing_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0 or transitions[i, j, k] != 3:
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
        if solid[ni, nj, nk] == 0 and source_flags[ni, nj, nk] != 0 and final_flags[ni, nj, nk] != 0:
            neighbor = ni * ny * nz + nj * nz + nk
            rho = source_moments[neighbor]
            if wp.isfinite(rho) and rho > 0.0:
                rho_sum += rho
                ux_sum += source_moments[stride + neighbor] / rho
                uy_sum += source_moments[2 * stride + neighbor] / rho
                uz_sum += source_moments[3 * stride + neighbor] / rho
                count += 1
    if count == 0:
        wp.atomic_add(missing_count, 0, 1)
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
def wall_commit_topology_kernel(
    moments: wp.array(dtype=float),
    working_mass: wp.array3d(dtype=float),
    stranded: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    transitions: wp.array3d(dtype=wp.int32),
    final_flags: wp.array3d(dtype=wp.int32),
    output_mass: wp.array3d(dtype=float),
    output_fill: wp.array3d(dtype=float),
    output_excess: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    transition_counts: wp.array(dtype=wp.int32),
    final_cells: wp.array(dtype=wp.float64),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    if solid[i, j, k] != 0:
        output_mass[i, j, k] = 0.0
        output_fill[i, j, k] = 0.0
        output_excess[i, j, k] = 0.0
        final_cells[cell] = wp.float64(0.0)
        return
    rho = moments[cell]
    mass = working_mass[i, j, k]
    queued = stranded[i, j, k]
    source_flag = source_flags[i, j, k]
    transition = transitions[i, j, k]
    flag = final_flags[i, j, k]
    committed_mass = float(0.0)
    committed_fill = float(0.0)
    if flag == 2:
        committed_mass = rho
        committed_fill = 1.0
        queued += mass - rho
    elif flag == 1:
        committed_mass = wp.clamp(mass, 0.0, rho)
        committed_fill = committed_mass / rho
        queued += mass - committed_mass
    else:
        queued += mass
    valid = wp.isfinite(rho) and rho > 0.0 and wp.isfinite(queued)
    valid = valid and wp.isfinite(committed_mass) and wp.isfinite(committed_fill)
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        final_cells[cell] = wp.float64(0.0)
    else:
        final_cells[cell] = wp.float64(committed_mass) + wp.float64(queued)
    output_mass[i, j, k] = committed_mass
    output_fill[i, j, k] = committed_fill
    output_excess[i, j, k] = queued
    if transition == 1 and flag == 2:
        wp.atomic_add(transition_counts, 0, 1)
    elif transition == 2:
        wp.atomic_add(transition_counts, 1, 1)
    elif transition == 3:
        wp.atomic_add(transition_counts, 2, 1)
    if source_flag == 2 and flag == 1:
        wp.atomic_add(transition_counts, 3, 1)


@wp.kernel
def wall_validate_separation_kernel(
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    direct_count: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0 or flags[i, j, k] != 2:
        return
    for q in range(1, 27):
        c = directions[q]
        ni = (i + int(c[0]) + nx) % nx
        nj = (j + int(c[1]) + ny) % ny
        nk = (k + int(c[2]) + nz) % nz
        if solid[ni, nj, nk] == 0 and flags[ni, nj, nk] == 0:
            wp.atomic_add(direct_count, 0, 1)
