# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for conservative HOME-FREE liquid-mass transport."""

from __future__ import annotations

import warp as wp

from ..kernels import (
    _equilibrium_convective_population,
    _equilibrium_population,
    _reconstruct_population,
    _reconstruct_population_components,
)


GAS = wp.constant(0)
INTERFACE = wp.constant(1)
LIQUID = wp.constant(2)
SOLID = wp.constant(3)


@wp.func
def _home_equilibrium_population(
    rho: float,
    velocity: wp.vec3,
    c: wp.vec3,
    weight: float,
) -> float:
    """Evaluate the third-order HOME equilibrium used by moment transport."""

    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]
    return _reconstruct_population_components(
        rho,
        rho * ux,
        rho * uy,
        rho * uz,
        rho * ux * ux,
        rho * uy * uy,
        rho * uz * uz,
        rho * ux * uy,
        rho * ux * uz,
        rho * uy * uz,
        c,
        weight,
    )


@wp.func
def _fsl_boundary_population(
    local_moments: wp.array(dtype=float),
    interior_moments: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    local_cell: int,
    interior_cell: int,
    q: int,
    delta: float,
    rho_boundary: float,
    velocity_boundary: wp.vec3,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    shear_omega: float,
    stride: int,
) -> float:
    opposite = opposites[q]
    c = directions[q]
    opposite_c = directions[opposite]
    weight = weights[q]
    local_rho = local_moments[local_cell]
    local_velocity = wp.vec3(
        local_moments[stride + local_cell] / local_rho,
        local_moments[2 * stride + local_cell] / local_rho,
        local_moments[3 * stride + local_cell] / local_rho,
    )
    local_incoming = _reconstruct_population(
        local_moments, c, weight, local_cell, stride
    )
    local_outgoing = _reconstruct_population(
        local_moments, opposite_c, weights[opposite], local_cell, stride
    )
    interior_outgoing = _reconstruct_population(
        interior_moments, opposite_c, weights[opposite], interior_cell, stride
    )
    local_even_equilibrium = 0.5 * (
        _equilibrium_population(local_rho, local_velocity, c, weight)
        + _equilibrium_population(
            local_rho, local_velocity, opposite_c, weights[opposite]
        )
    )
    local_even_nonequilibrium = (
        0.5 * (local_incoming + local_outgoing) - local_even_equilibrium
    )
    boundary_even_equilibrium = 0.5 * (
        _equilibrium_population(rho_boundary, velocity_boundary, c, weight)
        + _equilibrium_population(
            rho_boundary, velocity_boundary, opposite_c, weights[opposite]
        )
    )
    cx = c[0]
    cy = c[1]
    cz = c[2]
    strain_contraction = (
        sxx * cx * cx
        + syy * cy * cy
        + szz * cz * cz
        + 2.0 * (sxy * cx * cy + sxz * cx * cz + syz * cy * cz)
    )
    lambda_plus = 1.0 / shear_omega - 0.5
    return (
        (0.5 - delta) * local_outgoing
        + 0.5 * local_incoming
        + (delta - 1.0) * interior_outgoing
        + shear_omega * (1.5 - delta) * local_even_nonequilibrium
        + boundary_even_equilibrium
        - 3.0 * lambda_plus * weight * strain_contraction
    )


@wp.kernel
def fsl_boundary_populations_kernel(
    local_moments: wp.array(dtype=float),
    interior_moments: wp.array(dtype=float),
    incoming_direction: wp.array(dtype=wp.int32),
    fraction: wp.array(dtype=float),
    boundary_density: wp.array(dtype=float),
    boundary_velocity: wp.array(dtype=wp.vec3),
    boundary_strain: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    population: wp.array(dtype=float),
    status: wp.array(dtype=wp.int32),
    shear_omega: float,
    stride: int,
):
    """Batch Bogner FSL closure using WanPhys pull-direction conventions."""

    link = wp.tid()
    population[link] = 0.0
    status[link] = 0
    q = incoming_direction[link]
    delta = fraction[link]
    rho_boundary = boundary_density[link]
    valid = q > 0 and q < 27
    valid = valid and wp.isfinite(delta) and delta >= 0.0 and delta <= 1.0
    valid = valid and wp.isfinite(rho_boundary) and rho_boundary > 0.0
    valid = valid and wp.isfinite(shear_omega) and shear_omega > 0.0 and shear_omega < 2.0
    for component in range(10):
        valid = valid and wp.isfinite(local_moments[component * stride + link])
        valid = valid and wp.isfinite(interior_moments[component * stride + link])
    local_rho = local_moments[link]
    interior_rho = interior_moments[link]
    valid = valid and local_rho > 0.0 and interior_rho > 0.0
    velocity_boundary = boundary_velocity[link]
    valid = (
        valid
        and wp.isfinite(velocity_boundary[0])
        and wp.isfinite(velocity_boundary[1])
        and wp.isfinite(velocity_boundary[2])
    )
    for component in range(6):
        valid = valid and wp.isfinite(boundary_strain[component * stride + link])
    if not valid:
        status[link] = -1
        return

    sxx = boundary_strain[link]
    syy = boundary_strain[stride + link]
    szz = boundary_strain[2 * stride + link]
    sxy = boundary_strain[3 * stride + link]
    sxz = boundary_strain[4 * stride + link]
    syz = boundary_strain[5 * stride + link]
    population[link] = _fsl_boundary_population(
        local_moments,
        interior_moments,
        directions,
        weights,
        opposites,
        link,
        link,
        q,
        delta,
        rho_boundary,
        velocity_boundary,
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        shear_omega,
        stride,
    )
    status[link] = 1


@wp.kernel
def extrapolate_fsl_boundary_velocity_kernel(
    moments: wp.array(dtype=float),
    active: wp.array3d(dtype=wp.int32),
    link_status: wp.array(dtype=wp.int32),
    fraction: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    boundary_velocity: wp.array(dtype=wp.vec3),
    invalid_count: wp.array(dtype=wp.int32),
    no_support_policy: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    link_index = wp.tid()
    boundary_velocity[link_index] = wp.vec3(0.0)
    q = link_index % 27
    cell = link_index // 27
    k = cell % nz
    remainder = cell // nz
    j = remainder % ny
    i = remainder // ny
    status = link_status[link_index]
    if status != 1 and status != 2 and status != -7:
        return
    rho = moments[cell]
    if not wp.isfinite(rho) or rho <= 0.0:
        wp.atomic_add(invalid_count, 0, 1)
        return
    local_velocity = wp.vec3(
        moments[stride + cell] / rho,
        moments[2 * stride + cell] / rho,
        moments[3 * stride + cell] / rho,
    )
    if status == -7:
        if no_support_policy == 0:
            wp.atomic_add(invalid_count, 0, 1)
            return
        boundary_velocity[link_index] = local_velocity
        return

    c = directions[q]
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
    if support_valid == 0 or active[support_i, support_j, support_k] == 0:
        wp.atomic_add(invalid_count, 0, 1)
        return
    support = support_i * ny * nz + support_j * nz + support_k
    support_rho = moments[support]
    delta = fraction[link_index]
    if (
        not wp.isfinite(support_rho)
        or support_rho <= 0.0
        or not wp.isfinite(delta)
        or delta < 0.0
        or delta > 1.0
    ):
        wp.atomic_add(invalid_count, 0, 1)
        return
    support_velocity = wp.vec3(
        moments[stride + support] / support_rho,
        moments[2 * stride + support] / support_rho,
        moments[3 * stride + support] / support_rho,
    )
    boundary_velocity[link_index] = (
        (1.0 + delta) * local_velocity - delta * support_velocity
    )


@wp.kernel
def extrapolate_fsl_boundary_strain_kernel(
    bulk_strain: wp.array(dtype=float),
    interface_normal: wp.array3d(dtype=wp.vec3),
    active: wp.array3d(dtype=wp.int32),
    link_status: wp.array(dtype=wp.int32),
    plane_owner: wp.array(dtype=wp.int32),
    fraction: wp.array(dtype=float),
    normal_strain: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    boundary_strain: wp.array(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    no_support_policy: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Linear strain extrapolation with Bogner normal/tangential projection."""

    link = wp.tid()
    link_stride = 27 * stride
    for component in range(6):
        boundary_strain[component * link_stride + link] = 0.0
    q = link % 27
    cell = link // 27
    k = cell % nz
    remainder = cell // nz
    j = remainder % ny
    i = remainder // ny
    status = link_status[link]
    if status != 1 and status != 2 and status != -7:
        return
    if status == -7:
        if no_support_policy == 0:
            wp.atomic_add(invalid_count, 0, 1)
        return

    c = directions[q]
    support_i = i + int(c[0])
    support_j = j + int(c[1])
    support_k = k + int(c[2])
    source_i = i - int(c[0])
    source_j = j - int(c[1])
    source_k = k - int(c[2])
    valid = int(1)
    if support_i < 0 or support_i >= nx:
        if periodic_x != 0:
            support_i = (support_i + nx) % nx
        else:
            valid = 0
    if support_j < 0 or support_j >= ny:
        if periodic_y != 0:
            support_j = (support_j + ny) % ny
        else:
            valid = 0
    if support_k < 0 or support_k >= nz:
        if periodic_z != 0:
            support_k = (support_k + nz) % nz
        else:
            valid = 0
    if source_i < 0 or source_i >= nx:
        if periodic_x != 0:
            source_i = (source_i + nx) % nx
        else:
            valid = 0
    if source_j < 0 or source_j >= ny:
        if periodic_y != 0:
            source_j = (source_j + ny) % ny
        else:
            valid = 0
    if source_k < 0 or source_k >= nz:
        if periodic_z != 0:
            source_k = (source_k + nz) % nz
        else:
            valid = 0
    if valid == 0 or active[support_i, support_j, support_k] == 0:
        wp.atomic_add(invalid_count, 0, 1)
        return
    support = support_i * ny * nz + support_j * nz + support_k
    owner_i = i
    owner_j = j
    owner_k = k
    owner = plane_owner[link]
    if owner == 2:
        owner_i = source_i
        owner_j = source_j
        owner_k = source_k
    elif owner != 1:
        wp.atomic_add(invalid_count, 0, 1)
        return
    n = interface_normal[owner_i, owner_j, owner_k]
    normal_length = wp.length(n)
    delta = fraction[link]
    target = normal_strain[link]
    if (
        not wp.isfinite(normal_length)
        or normal_length <= 0.0
        or not wp.isfinite(delta)
        or delta < 0.0
        or delta > 1.0
        or not wp.isfinite(target)
    ):
        wp.atomic_add(invalid_count, 0, 1)
        return
    n = n / normal_length
    local_xx = bulk_strain[cell]
    local_yy = bulk_strain[stride + cell]
    local_zz = bulk_strain[2 * stride + cell]
    local_xy = bulk_strain[3 * stride + cell]
    local_xz = bulk_strain[4 * stride + cell]
    local_yz = bulk_strain[5 * stride + cell]
    sxx = (1.0 + delta) * local_xx - delta * bulk_strain[support]
    syy = (1.0 + delta) * local_yy - delta * bulk_strain[stride + support]
    szz = (1.0 + delta) * local_zz - delta * bulk_strain[2 * stride + support]
    sxy = (1.0 + delta) * local_xy - delta * bulk_strain[3 * stride + support]
    sxz = (1.0 + delta) * local_xz - delta * bulk_strain[4 * stride + support]
    syz = (1.0 + delta) * local_yz - delta * bulk_strain[5 * stride + support]
    finite = (
        wp.isfinite(sxx)
        and wp.isfinite(syy)
        and wp.isfinite(szz)
        and wp.isfinite(sxy)
        and wp.isfinite(sxz)
        and wp.isfinite(syz)
    )
    if not finite:
        wp.atomic_add(invalid_count, 0, 1)
        return

    normal_vector = wp.vec3(
        sxx * n[0] + sxy * n[1] + sxz * n[2],
        sxy * n[0] + syy * n[1] + syz * n[2],
        sxz * n[0] + syz * n[1] + szz * n[2],
    )
    normal_component = wp.dot(n, normal_vector)
    tangent = normal_vector - normal_component * n
    correction = target - normal_component
    boundary_strain[link] = (
        sxx - 2.0 * n[0] * tangent[0] + correction * n[0] * n[0]
    )
    boundary_strain[link_stride + link] = (
        syy - 2.0 * n[1] * tangent[1] + correction * n[1] * n[1]
    )
    boundary_strain[2 * link_stride + link] = (
        szz - 2.0 * n[2] * tangent[2] + correction * n[2] * n[2]
    )
    boundary_strain[3 * link_stride + link] = (
        sxy
        - n[0] * tangent[1]
        - tangent[0] * n[1]
        + correction * n[0] * n[1]
    )
    boundary_strain[4 * link_stride + link] = (
        sxz
        - n[0] * tangent[2]
        - tangent[0] * n[2]
        + correction * n[0] * n[2]
    )
    boundary_strain[5 * link_stride + link] = (
        syz
        - n[1] * tangent[2]
        - tangent[1] * n[2]
        + correction * n[1] * n[2]
    )


@wp.kernel
def fsl_stream_moments_kernel(
    moments_in: wp.array(dtype=float),
    active: wp.array3d(dtype=wp.int32),
    link_status: wp.array(dtype=wp.int32),
    plane_owner: wp.array(dtype=wp.int32),
    fraction: wp.array(dtype=float),
    gas_density: wp.array3d(dtype=float),
    boundary_velocity: wp.array(dtype=wp.vec3),
    boundary_strain: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    moments_out: wp.array(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    fallback_count: wp.array(dtype=wp.int32),
    shear_omega: float,
    has_boundary_strain: int,
    no_support_policy: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Owner-aware FSL/hybrid pull stream into raw HOME moments."""

    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    if active[i, j, k] == 0:
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[
                component * stride + cell
            ]
        return
    local_rho = moments_in[cell]
    if not wp.isfinite(local_rho) or local_rho <= 0.0:
        wp.atomic_add(invalid_count, 0, 1)
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
    valid = int(1)
    link_stride = 27 * stride
    for q in range(27):
        c = directions[q]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        source_valid = int(1)
        if source_i < 0 or source_i >= nx:
            if periodic_x != 0:
                source_i = (source_i + nx) % nx
            else:
                source_valid = 0
        if source_j < 0 or source_j >= ny:
            if periodic_y != 0:
                source_j = (source_j + ny) % ny
            else:
                source_valid = 0
        if source_k < 0 or source_k >= nz:
            if periodic_z != 0:
                source_k = (source_k + nz) % nz
            else:
                source_valid = 0
        if source_valid == 0:
            valid = 0
            continue
        source = source_i * ny * nz + source_j * nz + source_k
        value = float(0.0)
        if active[source_i, source_j, source_k] != 0:
            source_rho = moments_in[source]
            if not wp.isfinite(source_rho) or source_rho <= 0.0:
                valid = 0
                continue
            value = _reconstruct_population(
                moments_in, c, weights[q], source, stride
            )
        else:
            link = cell * 27 + q
            status = link_status[link]
            owner = plane_owner[link]
            owner_i = i
            owner_j = j
            owner_k = k
            if owner == 2:
                owner_i = source_i
                owner_j = source_j
                owner_k = source_k
            elif owner != 1:
                valid = 0
                continue
            rho_boundary = gas_density[owner_i, owner_j, owner_k]
            velocity_boundary = boundary_velocity[link]
            boundary_valid = wp.isfinite(rho_boundary) and rho_boundary > 0.0
            boundary_valid = (
                boundary_valid
                and wp.isfinite(velocity_boundary[0])
                and wp.isfinite(velocity_boundary[1])
                and wp.isfinite(velocity_boundary[2])
            )
            if not boundary_valid:
                valid = 0
                continue

            if status == 1 or status == 2:
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
                if (
                    support_valid == 0
                    or active[support_i, support_j, support_k] == 0
                ):
                    valid = 0
                    continue
                support = support_i * ny * nz + support_j * nz + support_k
                delta = fraction[link]
                if not wp.isfinite(delta) or delta < 0.0 or delta > 1.0:
                    valid = 0
                    continue
                sxx = float(0.0)
                syy = float(0.0)
                szz = float(0.0)
                sxy = float(0.0)
                sxz = float(0.0)
                syz = float(0.0)
                if has_boundary_strain != 0:
                    sxx = boundary_strain[link]
                    syy = boundary_strain[link_stride + link]
                    szz = boundary_strain[2 * link_stride + link]
                    sxy = boundary_strain[3 * link_stride + link]
                    sxz = boundary_strain[4 * link_stride + link]
                    syz = boundary_strain[5 * link_stride + link]
                value = _fsl_boundary_population(
                    moments_in,
                    moments_in,
                    directions,
                    weights,
                    opposites,
                    cell,
                    support,
                    q,
                    delta,
                    rho_boundary,
                    velocity_boundary,
                    sxx,
                    syy,
                    szz,
                    sxy,
                    sxz,
                    syz,
                    shear_omega,
                    stride,
                )
            elif status == -7 and no_support_policy != 0:
                opposite = opposites[q]
                outgoing = _reconstruct_population(
                    moments_in,
                    directions[opposite],
                    weights[opposite],
                    cell,
                    stride,
                )
                boundary_even = 0.5 * (
                    _equilibrium_population(
                        rho_boundary, velocity_boundary, c, weights[q]
                    )
                    + _equilibrium_population(
                        rho_boundary,
                        velocity_boundary,
                        directions[opposite],
                        weights[opposite],
                    )
                )
                value = 2.0 * boundary_even - outgoing
                wp.atomic_add(fallback_count, 0, 1)
            else:
                valid = 0
                continue

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

    if valid == 0 or not wp.isfinite(rho) or rho <= 0.0:
        wp.atomic_add(invalid_count, 0, 1)
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[
                component * stride + cell
            ]
        return
    moments_out[cell] = rho
    moments_out[stride + cell] = jx
    moments_out[2 * stride + cell] = jy
    moments_out[3 * stride + cell] = jz
    moments_out[4 * stride + cell] = qxx - rho / 3.0
    moments_out[5 * stride + cell] = qyy - rho / 3.0
    moments_out[6 * stride + cell] = qzz - rho / 3.0
    moments_out[7 * stride + cell] = qxy
    moments_out[8 * stride + cell] = qxz
    moments_out[9 * stride + cell] = qyz


@wp.kernel
def only_missing_boundary_residual_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    gas_density: wp.array3d(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    pressure_residual: wp.array(dtype=float),
    equilibrium_transport_residual: wp.array(dtype=float),
    non_equilibrium_residual: wp.array(dtype=float),
    total_residual: wp.array(dtype=float),
    gas_link_count: wp.array3d(dtype=wp.int32),
    invalid_ownership_count: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Warp counterpart of ``only_missing_boundary_residual``."""

    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = flags[i, j, k]
    for component in range(4):
        pressure_residual[component * stride + cell] = 0.0
        equilibrium_transport_residual[component * stride + cell] = 0.0
        non_equilibrium_residual[component * stride + cell] = 0.0
        total_residual[component * stride + cell] = 0.0
    gas_link_count[i, j, k] = 0
    if flag != INTERFACE and flag != LIQUID:
        return

    rho = moments[cell]
    valid = wp.isfinite(rho) and rho > 0.0
    velocity = wp.vec3(0.0)
    if valid:
        velocity = wp.vec3(
            moments[stride + cell] / rho,
            moments[2 * stride + cell] / rho,
            moments[3 * stride + cell] / rho,
        )
        valid = (
            wp.isfinite(velocity[0])
            and wp.isfinite(velocity[1])
            and wp.isfinite(velocity[2])
        )
    boundary_rho = gas_density[i, j, k]
    if flag == INTERFACE:
        valid = valid and wp.isfinite(boundary_rho) and boundary_rho > 0.0

    pressure = wp.vec4(0.0)
    equilibrium_transport = wp.vec4(0.0)
    nonequilibrium = wp.vec4(0.0)
    total = wp.vec4(0.0)
    gas_links = int(0)
    for direction in range(27):
        c = directions[direction]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        source_valid = int(1)
        if source_i < 0 or source_i >= nx:
            if periodic_x != 0:
                source_i = (source_i + nx) % nx
            else:
                source_valid = 0
        if source_j < 0 or source_j >= ny:
            if periodic_y != 0:
                source_j = (source_j + ny) % ny
            else:
                source_valid = 0
        if source_k < 0 or source_k >= nz:
            if periodic_z != 0:
                source_k = (source_k + nz) % nz
            else:
                source_valid = 0
        if source_valid == 0:
            valid = False
            continue

        source_flag = flags[source_i, source_j, source_k]
        source = source_i * ny * nz + source_j * nz + source_k
        pressure_delta = float(0.0)
        equilibrium_delta = float(0.0)
        nonequilibrium_delta = float(0.0)
        direct_delta = float(0.0)
        if source_flag == INTERFACE or source_flag == LIQUID:
            source_rho = moments[source]
            source_valid_rho = wp.isfinite(source_rho) and source_rho > 0.0
            valid = valid and source_valid_rho
            if source_valid_rho:
                source_velocity = wp.vec3(
                    moments[stride + source] / source_rho,
                    moments[2 * stride + source] / source_rho,
                    moments[3 * stride + source] / source_rho,
                )
                local_population = _reconstruct_population(
                    moments, c, weights[direction], cell, stride
                )
                source_population = _reconstruct_population(
                    moments, c, weights[direction], source, stride
                )
                local_equilibrium = _home_equilibrium_population(
                    rho, velocity, c, weights[direction]
                )
                source_equilibrium = _home_equilibrium_population(
                    source_rho, source_velocity, c, weights[direction]
                )
                pressure_delta = weights[direction] * (source_rho - rho)
                equilibrium_delta = (
                    source_equilibrium
                    - weights[direction] * source_rho
                    - local_equilibrium
                    + weights[direction] * rho
                )
                nonequilibrium_delta = (
                    source_population
                    - source_equilibrium
                    - local_population
                    + local_equilibrium
                )
                direct_delta = source_population - local_population
        elif source_flag == GAS:
            if flag != INTERFACE:
                valid = False
                continue
            gas_links += 1
            opposite = opposites[direction]
            opposite_c = directions[opposite]
            local_population = _reconstruct_population(
                moments, c, weights[direction], cell, stride
            )
            opposite_population = _reconstruct_population(
                moments, opposite_c, weights[opposite], cell, stride
            )
            local_equilibrium = _home_equilibrium_population(
                rho, velocity, c, weights[direction]
            )
            opposite_equilibrium = _home_equilibrium_population(
                rho, velocity, opposite_c, weights[opposite]
            )
            gas_pair = (
                _equilibrium_population(
                    boundary_rho, velocity, c, weights[direction]
                )
                + _equilibrium_population(
                    boundary_rho, velocity, opposite_c, weights[opposite]
                )
            )
            local_pair = (
                _equilibrium_population(rho, velocity, c, weights[direction])
                + _equilibrium_population(
                    rho, velocity, opposite_c, weights[opposite]
                )
            )
            boundary_population = gas_pair - opposite_population
            boundary_equilibrium = gas_pair - opposite_equilibrium
            boundary_local_pressure = local_pair - opposite_equilibrium
            pressure_delta = boundary_equilibrium - boundary_local_pressure
            equilibrium_delta = boundary_local_pressure - local_equilibrium
            nonequilibrium_delta = (
                boundary_population
                - boundary_equilibrium
                - local_population
                + local_equilibrium
            )
            direct_delta = boundary_population - local_population
        else:
            valid = False
            continue

        projection = wp.vec4(1.0, c[0], c[1], c[2])
        pressure += pressure_delta * projection
        equilibrium_transport += equilibrium_delta * projection
        nonequilibrium += nonequilibrium_delta * projection
        total += direct_delta * projection

    if not valid:
        wp.atomic_add(invalid_ownership_count, 0, 1)
        return
    gas_link_count[i, j, k] = gas_links
    for component in range(4):
        pressure_residual[component * stride + cell] = pressure[component]
        equilibrium_transport_residual[component * stride + cell] = (
            equilibrium_transport[component]
        )
        non_equilibrium_residual[component * stride + cell] = nonequilibrium[component]
        total_residual[component * stride + cell] = total[component]


@wp.kernel
def advect_mass_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill_level: wp.array3d(dtype=float),
    excess_mass: wp.array3d(dtype=float),
    excess_momentum: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    advected_mass: wp.array3d(dtype=float),
    incoming_excess_mass: wp.array3d(dtype=float),
    incoming_excess_momentum: wp.array(dtype=float),
    advected_momentum: wp.array(dtype=float),
    home_internal_link_momentum: wp.array(dtype=float),
    reference_pressure_momentum: wp.array(dtype=float),
    direct_liquid_gas_link_count: wp.array(dtype=wp.int32),
    invalid_advected_momentum_count: wp.array(dtype=wp.int32),
    track_exact_momentum: int,
    pressure_reference_density: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell_flag = flags[i, j, k]
    cell = i * ny * nz + j * nz + k
    if cell_flag == GAS or cell_flag == SOLID:
        advected_mass[i, j, k] = mass[i, j, k]
        incoming_excess_mass[i, j, k] = 0.0
        for component in range(3):
            incoming_excess_momentum[component * stride + cell] = 0.0
            if track_exact_momentum != 0:
                advected_momentum[component * stride + cell] = 0.0
                home_internal_link_momentum[component * stride + cell] = 0.0
                reference_pressure_momentum[component * stride + cell] = 0.0
        return

    cell_rho = moments[cell]
    if not wp.isfinite(cell_rho) or cell_rho <= 0.0:
        wp.atomic_add(invalid_advected_momentum_count, 0, 1)
        advected_mass[i, j, k] = mass[i, j, k]
        incoming_excess_mass[i, j, k] = 0.0
        for component in range(3):
            incoming_excess_momentum[component * stride + cell] = 0.0
            if track_exact_momentum != 0:
                advected_momentum[component * stride + cell] = 0.0
                home_internal_link_momentum[component * stride + cell] = 0.0
                reference_pressure_momentum[component * stride + cell] = 0.0
        return
    cell_fill = fill_level[i, j, k]
    cell_velocity = wp.vec3(
        moments[stride + cell] / cell_rho,
        moments[2 * stride + cell] / cell_rho,
        moments[3 * stride + cell] / cell_rho,
    )
    updated_mass = mass[i, j, k]
    updated_momentum = wp.vec3(0.0)
    home_internal_momentum = wp.vec3(0.0)
    reference_momentum = wp.vec3(0.0)
    if track_exact_momentum != 0:
        updated_momentum = mass[i, j, k] * cell_velocity
        home_internal_momentum = wp.vec3(
            moments[stride + cell],
            moments[2 * stride + cell],
            moments[3 * stride + cell],
        )
    queued_mass = float(0.0)
    queued_momentum = wp.vec3(0.0)
    for direction in range(1, 27):
        c = directions[direction]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        valid = int(1)
        if source_i < 0 or source_i >= nx:
            if periodic_x != 0:
                source_i = (source_i + nx) % nx
            else:
                valid = 0
        if source_j < 0 or source_j >= ny:
            if periodic_y != 0:
                source_j = (source_j + ny) % ny
            else:
                valid = 0
        if source_k < 0 or source_k >= nz:
            if periodic_z != 0:
                source_k = (source_k + nz) % nz
            else:
                valid = 0
        if valid == 0:
            if track_exact_momentum != 0:
                reference_momentum += (
                    2.0 * weights[direction] * pressure_reference_density * c
                )
            continue

        source_flag = flags[source_i, source_j, source_k]
        source = source_i * ny * nz + source_j * nz + source_k
        source_excess = excess_mass[source_i, source_j, source_k]
        updated_mass += source_excess
        queued_mass += source_excess
        for component in range(3):
            queued_momentum[component] += excess_momentum[
                component * stride + source
            ]
        if source_flag != LIQUID and source_flag != INTERFACE:
            if track_exact_momentum != 0:
                reference_momentum += (
                    2.0 * weights[direction] * pressure_reference_density * c
                )
            if cell_flag == LIQUID and source_flag == GAS:
                wp.atomic_add(direct_liquid_gas_link_count, 0, 1)
            continue

        opposite = opposites[direction]
        incoming = _reconstruct_population(
            moments, c, weights[direction], source, stride
        )
        outgoing = _reconstruct_population(
            moments, directions[opposite], weights[opposite], cell, stride
        )
        exchange_weight = float(1.0)
        if cell_flag == INTERFACE and source_flag == INTERFACE:
            exchange_weight = 0.5 * (
                cell_fill + fill_level[source_i, source_j, source_k]
            )
        mass_delta = exchange_weight * (incoming - outgoing)
        updated_mass += mass_delta
        if track_exact_momentum != 0:
            updated_momentum += exchange_weight * (incoming + outgoing) * c
            home_internal_momentum += (incoming + outgoing) * c
            reference_momentum += (
                exchange_weight
                * 2.0
                * weights[direction]
                * pressure_reference_density
                * c
            )
    advected_mass[i, j, k] = updated_mass
    incoming_excess_mass[i, j, k] = queued_mass
    for component in range(3):
        incoming_excess_momentum[component * stride + cell] = queued_momentum[component]
        if track_exact_momentum != 0:
            advected_momentum[component * stride + cell] = updated_momentum[component]
            home_internal_link_momentum[component * stride + cell] = (
                home_internal_momentum[component]
            )
            reference_pressure_momentum[component * stride + cell] = (
                reference_momentum[component]
            )
    if not (
        wp.isfinite(queued_momentum[0])
        and wp.isfinite(queued_momentum[1])
        and wp.isfinite(queued_momentum[2])
    ):
        wp.atomic_add(invalid_advected_momentum_count, 0, 1)
    if track_exact_momentum != 0 and not (
        wp.isfinite(updated_momentum[0])
        and wp.isfinite(updated_momentum[1])
        and wp.isfinite(updated_momentum[2])
        and wp.isfinite(home_internal_momentum[0])
        and wp.isfinite(home_internal_momentum[1])
        and wp.isfinite(home_internal_momentum[2])
        and wp.isfinite(reference_momentum[0])
        and wp.isfinite(reference_momentum[1])
        and wp.isfinite(reference_momentum[2])
    ):
        wp.atomic_add(invalid_advected_momentum_count, 0, 1)


@wp.kernel
def finalize_advected_momentum_kernel(
    destination_moments: wp.array(dtype=float),
    advected_mass: wp.array3d(dtype=float),
    internal_link_advected_momentum: wp.array(dtype=float),
    incoming_excess_momentum: wp.array(dtype=float),
    home_internal_link_momentum: wp.array(dtype=float),
    reference_pressure_momentum: wp.array(dtype=float),
    capillary_momentum_correction: wp.array(dtype=float),
    apply_capillary_correction: int,
    source_flags: wp.array3d(dtype=wp.int32),
    target_momentum: wp.array(dtype=float),
    invalid_cell_count: wp.array(dtype=wp.int32),
    acceleration: wp.vec3,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = source_flags[i, j, k]
    if flag != LIQUID and flag != INTERFACE:
        for component in range(3):
            target_momentum[component * stride + cell] = 0.0
        return

    rho = destination_moments[cell]
    transported_mass = advected_mass[i, j, k]
    valid = wp.isfinite(rho) and rho > 0.0 and wp.isfinite(transported_mass)
    force_mass_correction = transported_mass - rho
    for component in range(3):
        value = (
            internal_link_advected_momentum[component * stride + cell]
            + incoming_excess_momentum[component * stride + cell]
            + destination_moments[(component + 1) * stride + cell]
            - home_internal_link_momentum[component * stride + cell]
            - reference_pressure_momentum[component * stride + cell]
            + force_mass_correction * acceleration[component]
        )
        if apply_capillary_correction != 0:
            value += capillary_momentum_correction[component * stride + cell]
        target_momentum[component * stride + cell] = value
        valid = valid and wp.isfinite(value)
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)


@wp.kernel
def apply_full_cell_momentum_correction_kernel(
    moments: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    momentum_correction: wp.array(dtype=float),
    invalid_cell_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if source_flags[i, j, k] != INTERFACE:
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    old_jx = moments[stride + cell]
    old_jy = moments[2 * stride + cell]
    old_jz = moments[3 * stride + cell]
    delta_j = wp.vec3(
        momentum_correction[cell],
        momentum_correction[stride + cell],
        momentum_correction[2 * stride + cell],
    )
    valid = (
        wp.isfinite(rho)
        and rho > 0.0
        and wp.isfinite(delta_j[0])
        and wp.isfinite(delta_j[1])
        and wp.isfinite(delta_j[2])
    )
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        return

    du = delta_j / rho
    moments[4 * stride + cell] += 2.0 * du[0] * old_jx + rho * du[0] * du[0]
    moments[5 * stride + cell] += 2.0 * du[1] * old_jy + rho * du[1] * du[1]
    moments[6 * stride + cell] += 2.0 * du[2] * old_jz + rho * du[2] * du[2]
    moments[7 * stride + cell] += (
        du[0] * old_jy + du[1] * old_jx + rho * du[0] * du[1]
    )
    moments[8 * stride + cell] += (
        du[0] * old_jz + du[2] * old_jx + rho * du[0] * du[2]
    )
    moments[9 * stride + cell] += (
        du[1] * old_jz + du[2] * old_jy + rho * du[1] * du[2]
    )
    moments[stride + cell] = old_jx + delta_j[0]
    moments[2 * stride + cell] = old_jy + delta_j[1]
    moments[3 * stride + cell] = old_jz + delta_j[2]
    for component in range(1, 10):
        valid = valid and wp.isfinite(moments[component * stride + cell])
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)


@wp.kernel
def analyze_momentum_defect_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill_level: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    opposites: wp.array(dtype=wp.int32),
    convective_density_increment: wp.array3d(dtype=float),
    convective_momentum_increment: wp.array(dtype=float),
    interface_momentum_candidate: wp.array(dtype=float),
    internal_link_advected_momentum: wp.array(dtype=float),
    home_internal_link_momentum: wp.array(dtype=float),
    invalid_cell_count: wp.array(dtype=wp.int32),
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
    convective_density_increment[i, j, k] = 0.0
    for component in range(3):
        convective_momentum_increment[component * stride + cell] = 0.0
        interface_momentum_candidate[component * stride + cell] = 0.0
        internal_link_advected_momentum[component * stride + cell] = 0.0
        home_internal_link_momentum[component * stride + cell] = 0.0
    cell_flag = flags[i, j, k]
    if cell_flag != LIQUID and cell_flag != INTERFACE:
        return
    cell_rho = moments[cell]
    if not wp.isfinite(cell_rho) or cell_rho <= 0.0:
        wp.atomic_add(invalid_cell_count, 0, 1)
        return
    cell_velocity = wp.vec3(
        moments[stride + cell] / cell_rho,
        moments[2 * stride + cell] / cell_rho,
        moments[3 * stride + cell] / cell_rho,
    )
    density_increment = float(0.0)
    momentum_increment = wp.vec3(0.0)
    candidate = wp.vec3(0.0)
    internal_momentum = mass[i, j, k] * cell_velocity
    home_internal_momentum = wp.vec3(
        moments[stride + cell],
        moments[2 * stride + cell],
        moments[3 * stride + cell],
    )
    for direction in range(27):
        c = directions[direction]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        valid = int(1)
        if source_i < 0 or source_i >= nx:
            if periodic_x != 0:
                source_i = (source_i + nx) % nx
            else:
                valid = 0
        if source_j < 0 or source_j >= ny:
            if periodic_y != 0:
                source_j = (source_j + ny) % ny
            else:
                valid = 0
        if source_k < 0 or source_k >= nz:
            if periodic_z != 0:
                source_k = (source_k + nz) % nz
            else:
                valid = 0
        if valid == 0:
            continue
        source_flag = flags[source_i, source_j, source_k]
        if source_flag != LIQUID and source_flag != INTERFACE:
            continue
        interface_link = cell_flag == INTERFACE or source_flag == INTERFACE
        source = source_i * ny * nz + source_j * nz + source_k
        source_rho = moments[source]
        if not wp.isfinite(source_rho) or source_rho <= 0.0:
            wp.atomic_add(invalid_cell_count, 0, 1)
            continue
        if interface_link:
            convective = _equilibrium_convective_population(
                moments, c, weights[direction], source, stride
            )
            density_increment += convective
            momentum_increment += convective * c
        if direction == 0:
            continue
        opposite = opposites[direction]
        incoming = _reconstruct_population(
            moments, c, weights[direction], source, stride
        )
        outgoing = _reconstruct_population(
            moments, directions[opposite], weights[opposite], cell, stride
        )
        exchange_weight = float(1.0)
        if cell_flag == INTERFACE and source_flag == INTERFACE:
            exchange_weight = 0.5 * (
                fill_level[i, j, k]
                + fill_level[source_i, source_j, source_k]
            )
        mass_delta = exchange_weight * (incoming - outgoing)
        internal_momentum += exchange_weight * (incoming + outgoing) * c
        home_internal_momentum += (incoming + outgoing) * c
        if mass_delta > 0.0 and interface_link:
            source_velocity = wp.vec3(
                moments[stride + source] / source_rho,
                moments[2 * stride + source] / source_rho,
                moments[3 * stride + source] / source_rho,
            )
            candidate += mass_delta * (source_velocity - cell_velocity)
    convective_density_increment[i, j, k] = density_increment
    for component in range(3):
        convective_momentum_increment[component * stride + cell] = (
            momentum_increment[component]
        )
        interface_momentum_candidate[component * stride + cell] = candidate[component]
        internal_link_advected_momentum[component * stride + cell] = (
            internal_momentum[component]
        )
        home_internal_link_momentum[component * stride + cell] = (
            home_internal_momentum[component]
        )


@wp.kernel
def capture_density_momentum_kernel(
    moments: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    momentum: wp.array(dtype=float),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    density[i, j, k] = moments[cell]
    momentum[cell] = moments[stride + cell]
    momentum[stride + cell] = moments[2 * stride + cell]
    momentum[2 * stride + cell] = moments[3 * stride + cell]


@wp.kernel
def apply_queue_momentum_correction_kernel(
    destination_moments: wp.array(dtype=float),
    advected_mass: wp.array3d(dtype=float),
    incoming_excess_mass: wp.array3d(dtype=float),
    incoming_excess_momentum: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = source_flags[i, j, k]
    if flag != LIQUID and flag != INTERFACE:
        return
    cell = i * ny * nz + j * nz + k
    rho = destination_moments[cell]
    transported = advected_mass[i, j, k]
    queued_mass = incoming_excess_mass[i, j, k]
    valid = wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(transported)
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        return

    jx = destination_moments[stride + cell]
    jy = destination_moments[2 * stride + cell]
    jz = destination_moments[3 * stride + cell]
    delta_x = incoming_excess_momentum[cell] - queued_mass * jx / rho
    delta_y = incoming_excess_momentum[stride + cell] - queued_mass * jy / rho
    delta_z = incoming_excess_momentum[2 * stride + cell] - queued_mass * jz / rho
    valid = valid and wp.isfinite(delta_x) and wp.isfinite(delta_y)
    valid = valid and wp.isfinite(delta_z)
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
        return
    delta_norm = wp.sqrt(
        delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
    )
    if delta_norm <= 1.0e-12:
        return
    if wp.abs(transported) <= 1.0e-7:
        wp.atomic_add(invalid_cell_count, 0, 1)
        return

    dx = delta_x / transported
    dy = delta_y / transported
    dz = delta_z / transported
    destination_moments[4 * stride + cell] += 2.0 * dx * jx + rho * dx * dx
    destination_moments[5 * stride + cell] += 2.0 * dy * jy + rho * dy * dy
    destination_moments[6 * stride + cell] += 2.0 * dz * jz + rho * dz * dz
    destination_moments[7 * stride + cell] += dx * jy + dy * jx + rho * dx * dy
    destination_moments[8 * stride + cell] += dx * jz + dz * jx + rho * dx * dz
    destination_moments[9 * stride + cell] += dy * jz + dz * jy + rho * dy * dz
    destination_moments[stride + cell] = jx + rho * dx
    destination_moments[2 * stride + cell] = jy + rho * dy
    destination_moments[3 * stride + cell] = jz + rho * dz
    for component in range(1, 10):
        valid = valid and wp.isfinite(destination_moments[component * stride + cell])
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)


@wp.kernel
def normalize_mass_kernel(
    moments: wp.array(dtype=float),
    advected_mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    normalized_mass: wp.array3d(dtype=float),
    normalized_fill: wp.array3d(dtype=float),
    excess_share: wp.array3d(dtype=float),
    excess_momentum_share: wp.array(dtype=float),
    active_neighbor_count: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
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
    transported = advected_mass[i, j, k]
    committed = float(0.0)
    fill = float(0.0)
    excess = float(0.0)
    valid = wp.isfinite(transported)

    if flag == LIQUID:
        valid = valid and wp.isfinite(rho) and rho > 0.0
        committed = rho
        fill = 1.0
        excess = transported - rho
    elif flag == INTERFACE:
        valid = valid and wp.isfinite(rho) and rho > 0.0
        committed = wp.clamp(transported, 0.0, rho)
        fill = committed / rho
        excess = transported - committed
    elif flag == GAS:
        excess = transported
    elif flag == SOLID:
        valid = valid and wp.abs(transported) <= 1.0e-7
    else:
        valid = False

    recipients = int(0)
    if flag != SOLID:
        for direction in range(1, 27):
            c = directions[direction]
            neighbor_i = i + int(c[0])
            neighbor_j = j + int(c[1])
            neighbor_k = k + int(c[2])
            neighbor_valid = int(1)
            if neighbor_i < 0 or neighbor_i >= nx:
                if periodic_x != 0:
                    neighbor_i = (neighbor_i + nx) % nx
                else:
                    neighbor_valid = 0
            if neighbor_j < 0 or neighbor_j >= ny:
                if periodic_y != 0:
                    neighbor_j = (neighbor_j + ny) % ny
                else:
                    neighbor_valid = 0
            if neighbor_k < 0 or neighbor_k >= nz:
                if periodic_z != 0:
                    neighbor_k = (neighbor_k + nz) % nz
                else:
                    neighbor_valid = 0
            if neighbor_valid != 0:
                neighbor_flag = flags[neighbor_i, neighbor_j, neighbor_k]
                if neighbor_flag == LIQUID or neighbor_flag == INTERFACE:
                    recipients += 1

    if recipients > 0:
        excess /= float(recipients)
    else:
        committed += excess
        excess = 0.0
    normalized_mass[i, j, k] = committed
    normalized_fill[i, j, k] = fill
    excess_share[i, j, k] = excess
    if excess != 0.0:
        valid = valid and wp.isfinite(rho) and rho > 0.0
        for component in range(3):
            excess_momentum_share[component * stride + cell] = (
                excess * moments[(component + 1) * stride + cell] / rho
            )
    else:
        for component in range(3):
            excess_momentum_share[component * stride + cell] = 0.0
    active_neighbor_count[i, j, k] = recipients
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)


@wp.kernel
def normalize_mass_momentum_kernel(
    moments: wp.array(dtype=float),
    advected_mass: wp.array3d(dtype=float),
    advected_momentum: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    normalized_mass: wp.array3d(dtype=float),
    normalized_fill: wp.array3d(dtype=float),
    excess_share: wp.array3d(dtype=float),
    excess_momentum_share: wp.array(dtype=float),
    active_neighbor_count: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
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
    transported = advected_mass[i, j, k]
    target = wp.vec3(
        advected_momentum[cell],
        advected_momentum[stride + cell],
        advected_momentum[2 * stride + cell],
    )
    committed = float(0.0)
    fill = float(0.0)
    excess = float(0.0)
    valid = wp.isfinite(transported)
    valid = valid and wp.isfinite(target[0]) and wp.isfinite(target[1])
    valid = valid and wp.isfinite(target[2])

    if flag == LIQUID:
        valid = valid and wp.isfinite(rho) and rho > 0.0
        committed = rho
        fill = 1.0
        excess = transported - rho
    elif flag == INTERFACE:
        valid = valid and wp.isfinite(rho) and rho > 0.0
        committed = wp.clamp(transported, 0.0, rho)
        fill = committed / rho
        excess = transported - committed
    elif flag == GAS:
        excess = transported
    elif flag == SOLID:
        valid = valid and wp.abs(transported) <= 1.0e-7
        valid = valid and wp.length(target) <= 1.0e-7
    else:
        valid = False

    recipients = int(0)
    if flag != SOLID:
        for direction in range(1, 27):
            c = directions[direction]
            neighbor_i = i + int(c[0])
            neighbor_j = j + int(c[1])
            neighbor_k = k + int(c[2])
            neighbor_valid = int(1)
            if neighbor_i < 0 or neighbor_i >= nx:
                if periodic_x != 0:
                    neighbor_i = (neighbor_i + nx) % nx
                else:
                    neighbor_valid = 0
            if neighbor_j < 0 or neighbor_j >= ny:
                if periodic_y != 0:
                    neighbor_j = (neighbor_j + ny) % ny
                else:
                    neighbor_valid = 0
            if neighbor_k < 0 or neighbor_k >= nz:
                if periodic_z != 0:
                    neighbor_k = (neighbor_k + nz) % nz
                else:
                    neighbor_valid = 0
            if neighbor_valid != 0:
                neighbor_flag = flags[neighbor_i, neighbor_j, neighbor_k]
                if neighbor_flag == LIQUID or neighbor_flag == INTERFACE:
                    recipients += 1

    if recipients > 0:
        excess /= float(recipients)
    elif wp.abs(excess) > 1.0e-7:
        valid = False
        excess = 0.0

    velocity = wp.vec3(0.0)
    has_transport = wp.abs(transported) > 1.0e-7
    if has_transport:
        velocity = target / transported
    else:
        valid = valid and wp.length(target) <= 1.0e-7
        if flag == LIQUID or flag == INTERFACE:
            velocity = wp.vec3(
                moments[stride + cell] / rho,
                moments[2 * stride + cell] / rho,
                moments[3 * stride + cell] / rho,
            )
    valid = valid and wp.isfinite(velocity[0]) and wp.isfinite(velocity[1])
    valid = valid and wp.isfinite(velocity[2])

    excess_j = excess * velocity
    normalized_mass[i, j, k] = committed
    normalized_fill[i, j, k] = fill
    excess_share[i, j, k] = excess
    active_neighbor_count[i, j, k] = recipients
    for component in range(3):
        excess_momentum_share[component * stride + cell] = excess_j[component]

    if flag == LIQUID or flag == INTERFACE:
        old_jx = moments[stride + cell]
        old_jy = moments[2 * stride + cell]
        old_jz = moments[3 * stride + cell]
        dx = velocity[0] - old_jx / rho
        dy = velocity[1] - old_jy / rho
        dz = velocity[2] - old_jz / rho
        moments[4 * stride + cell] += 2.0 * dx * old_jx + rho * dx * dx
        moments[5 * stride + cell] += 2.0 * dy * old_jy + rho * dy * dy
        moments[6 * stride + cell] += 2.0 * dz * old_jz + rho * dz * dz
        moments[7 * stride + cell] += (
            dx * old_jy + dy * old_jx + rho * dx * dy
        )
        moments[8 * stride + cell] += (
            dx * old_jz + dz * old_jx + rho * dx * dz
        )
        moments[9 * stride + cell] += (
            dy * old_jz + dz * old_jy + rho * dy * dz
        )
        moments[stride + cell] = rho * velocity[0]
        moments[2 * stride + cell] = rho * velocity[1]
        moments[3 * stride + cell] = rho * velocity[2]
        for component in range(1, 10):
            valid = valid and wp.isfinite(moments[component * stride + cell])

    represented_mass = committed + float(recipients) * excess
    represented_momentum = committed * velocity + float(recipients) * excess_j
    valid = valid and wp.abs(represented_mass - transported) <= 2.0e-6
    valid = valid and wp.length(represented_momentum - target) <= 2.0e-6
    if not valid:
        wp.atomic_add(invalid_cell_count, 0, 1)
