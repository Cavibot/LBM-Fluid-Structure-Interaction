# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for isolated Parker-Youngs PLIC reconstruction."""

from __future__ import annotations

import warp as wp


@wp.func
def _positive2(value: float) -> float:
    positive = wp.max(value, 0.0)
    return positive * positive


@wp.func
def _positive3(value: float) -> float:
    positive = wp.max(value, 0.0)
    return positive * positive * positive


@wp.func
def _reduced_components(normal: wp.vec3) -> wp.vec3:
    ax = wp.abs(normal[0])
    ay = wp.abs(normal[1])
    az = wp.abs(normal[2])
    threshold = 1.0e-4 * wp.max(ax, wp.max(ay, az))
    return wp.vec3(
        ax if ax > threshold else 0.0,
        ay if ay > threshold else 0.0,
        az if az > threshold else 0.0,
    )


@wp.func
def _plane_volume_fraction(offset: float, normal: wp.vec3) -> float:
    coefficients = _reduced_components(normal)
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    support = 0.5 * (a + b + c)
    if offset <= -support:
        return 0.0
    if offset >= support:
        return 1.0
    complement = offset > 0.0
    evaluation_offset = -offset if complement else offset
    dimension = int(0)
    if a > 0.0:
        dimension += 1
    if b > 0.0:
        dimension += 1
    if c > 0.0:
        dimension += 1
    shifted = evaluation_offset + support
    volume = float(0.0)
    if dimension == 1:
        coefficient = wp.max(a, wp.max(b, c))
        volume = wp.max(shifted, 0.0) / coefficient
    elif dimension == 2:
        first = a if a > 0.0 else b
        second = c if c > 0.0 else b
        if a == 0.0:
            first = b
            second = c
        elif b == 0.0:
            first = a
            second = c
        else:
            first = a
            second = b
        volume = (
            _positive2(shifted)
            - _positive2(shifted - first)
            - _positive2(shifted - second)
            + _positive2(shifted - first - second)
        ) / (2.0 * first * second)
    else:
        volume = (
            _positive3(shifted)
            - _positive3(shifted - a)
            - _positive3(shifted - b)
            - _positive3(shifted - c)
            + _positive3(shifted - a - b)
            + _positive3(shifted - a - c)
            + _positive3(shifted - b - c)
            - _positive3(shifted - a - b - c)
        ) / (6.0 * a * b * c)
    volume = wp.clamp(volume, 0.0, 1.0)
    return 1.0 - volume if complement else volume


@wp.func
def _plic_offset(fill: float, normal: wp.vec3) -> float:
    coefficients = _reduced_components(normal)
    bound = 0.5 * (coefficients[0] + coefficients[1] + coefficients[2])
    if fill <= 0.0:
        return -bound
    if fill >= 1.0:
        return bound
    lower = -bound
    upper = bound
    for _iteration in range(32):
        midpoint = 0.5 * (lower + upper)
        if _plane_volume_fraction(midpoint, normal) < fill:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


@wp.func
def _section_area(offset: float, normal: wp.vec3) -> float:
    coefficients = _reduced_components(normal)
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    magnitude = wp.length(normal)
    dimension = int(0)
    if a > 0.0:
        dimension += 1
    if b > 0.0:
        dimension += 1
    if c > 0.0:
        dimension += 1
    support = 0.5 * (a + b + c)
    shifted = wp.min(offset + support, support - offset)
    if dimension == 1:
        return 1.0
    if dimension == 2:
        first = a if a > 0.0 else b
        second = c if c > 0.0 else b
        if a == 0.0:
            first = b
            second = c
        elif b == 0.0:
            first = a
            second = c
        else:
            first = a
            second = b
        small = wp.min(first, second)
        large = wp.max(first, second)
        if shifted >= small:
            return magnitude / large
        return magnitude * wp.max(shifted, 0.0) / (small * large)
    derivative = (
        _positive2(shifted)
        - _positive2(shifted - a)
        - _positive2(shifted - b)
        - _positive2(shifted - c)
        + _positive2(shifted - a - b)
        + _positive2(shifted - a - c)
        + _positive2(shifted - b - c)
        - _positive2(shifted - a - b - c)
    ) / (2.0 * a * b * c)
    return wp.max(magnitude * derivative, 0.0)


@wp.func
def _coordinate(value: int, size: int, closed: int) -> int:
    if closed == 0:
        wrapped = value % size
        return wrapped + size if wrapped < 0 else wrapped
    return wp.clamp(value, 0, size - 1)


@wp.kernel
def reconstruct_geometry_kernel(
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    interface_area: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    endpoint_fallback_count: wp.array(dtype=wp.int32),
    wetting_count: wp.array(dtype=wp.int32),
    invalid_wall_normal_count: wp.array(dtype=wp.int32),
    parallel_wall_interface_count: wp.array(dtype=wp.int32),
    closed_x: int,
    closed_y: int,
    closed_z: int,
    nx: int,
    ny: int,
    nz: int,
    derive_interface: int,
    interface_epsilon: float,
    endpoint_fallback_tolerance: float,
    wetting_enabled: int,
    contact_angle_cos: float,
    contact_angle_sin: float,
):
    i, j, k = wp.tid()
    normal[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    plane_offset[i, j, k] = 0.0
    interface_area[i, j, k] = 0.0
    valid[i, j, k] = 0
    is_interface = flags[i, j, k] == 1
    if derive_interface != 0:
        is_interface = (
            fill[i, j, k] > interface_epsilon
            and fill[i, j, k] < 1.0 - interface_epsilon
        )
    if solid[i, j, k] != 0 or not is_interface:
        return
    center_fill = fill[i, j, k]
    gradient = wp.vec3(0.0, 0.0, 0.0)
    for first in range(3):
        for second in range(3):
            weight = float((1 if first != 1 else 2) * (1 if second != 1 else 2))
            x0 = _coordinate(i - 1, nx, closed_x)
            x1 = _coordinate(i + 1, nx, closed_x)
            y0 = _coordinate(j - 1, ny, closed_y)
            y1 = _coordinate(j + 1, ny, closed_y)
            z0 = _coordinate(k - 1, nz, closed_z)
            z1 = _coordinate(k + 1, nz, closed_z)
            y_for_x = _coordinate(j + first - 1, ny, closed_y)
            z_for_x = _coordinate(k + second - 1, nz, closed_z)
            x_for_y = _coordinate(i + first - 1, nx, closed_x)
            z_for_y = _coordinate(k + second - 1, nz, closed_z)
            x_for_z = _coordinate(i + first - 1, nx, closed_x)
            y_for_z = _coordinate(j + second - 1, ny, closed_y)
            sample_x0 = (
                center_fill
                if solid[x0, y_for_x, z_for_x] != 0
                else fill[x0, y_for_x, z_for_x]
            )
            sample_x1 = (
                center_fill
                if solid[x1, y_for_x, z_for_x] != 0
                else fill[x1, y_for_x, z_for_x]
            )
            sample_y0 = (
                center_fill
                if solid[x_for_y, y0, z_for_y] != 0
                else fill[x_for_y, y0, z_for_y]
            )
            sample_y1 = (
                center_fill
                if solid[x_for_y, y1, z_for_y] != 0
                else fill[x_for_y, y1, z_for_y]
            )
            sample_z0 = (
                center_fill
                if solid[x_for_z, y_for_z, z0] != 0
                else fill[x_for_z, y_for_z, z0]
            )
            sample_z1 = (
                center_fill
                if solid[x_for_z, y_for_z, z1] != 0
                else fill[x_for_z, y_for_z, z1]
            )
            gradient += weight * wp.vec3(
                sample_x1 - sample_x0,
                sample_y1 - sample_y0,
                sample_z1 - sample_z0,
            )
    magnitude = wp.length(gradient)
    if not wp.isfinite(magnitude) or magnitude <= 1.0e-8:
        endpoint_fallback = (
            derive_interface != 0
            and (
                center_fill <= endpoint_fallback_tolerance
                or center_fill >= 1.0 - endpoint_fallback_tolerance
            )
        )
        if endpoint_fallback:
            wp.atomic_add(endpoint_fallback_count, 0, 1)
            return
        wp.atomic_add(invalid_count, 0, 1)
        return
    reconstructed_normal = -gradient / magnitude
    if wetting_enabled != 0:
        wall_gradient = wp.vec3(0.0, 0.0, 0.0)
        has_solid_neighbor = int(0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = _coordinate(i + di, nx, closed_x)
                    nj = _coordinate(j + dj, ny, closed_y)
                    nk = _coordinate(k + dk, nz, closed_z)
                    if solid[ni, nj, nk] != 0:
                        weight = float(
                            (1 if di != 0 else 2)
                            * (1 if dj != 0 else 2)
                            * (1 if dk != 0 else 2)
                        )
                        wall_gradient -= weight * wp.vec3(
                            float(di), float(dj), float(dk)
                        )
                        has_solid_neighbor = 1
        if has_solid_neighbor != 0:
            wall_magnitude = wp.length(wall_gradient)
            if not wp.isfinite(wall_magnitude) or wall_magnitude <= 1.0e-8:
                wp.atomic_add(invalid_wall_normal_count, 0, 1)
                return
            wall_normal = wall_gradient / wall_magnitude
            tangent = (
                reconstructed_normal
                - wp.dot(reconstructed_normal, wall_normal) * wall_normal
            )
            tangent_magnitude = wp.length(tangent)
            if not wp.isfinite(tangent_magnitude) or tangent_magnitude <= 1.0e-8:
                wp.atomic_add(parallel_wall_interface_count, 0, 1)
            else:
                reconstructed_normal = (
                    contact_angle_cos * wall_normal
                    + contact_angle_sin * tangent / tangent_magnitude
                )
                normal_magnitude = wp.length(reconstructed_normal)
                if not wp.isfinite(normal_magnitude) or normal_magnitude <= 1.0e-8:
                    wp.atomic_add(invalid_wall_normal_count, 0, 1)
                    return
                reconstructed_normal /= normal_magnitude
                wp.atomic_add(wetting_count, 0, 1)
    offset = _plic_offset(center_fill, reconstructed_normal)
    normal[i, j, k] = reconstructed_normal
    plane_offset[i, j, k] = offset
    interface_area[i, j, k] = _section_area(offset, reconstructed_normal)
    valid[i, j, k] = 1
