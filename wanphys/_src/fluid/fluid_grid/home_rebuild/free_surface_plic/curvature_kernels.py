# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for isolated PLIC curvature reconstruction."""

import warp as wp

from .geometry_kernels import _coordinate, _plic_offset

Vec5f = wp.types.vector(5, wp.float32)
Mat55f = wp.types.matrix((5, 5), wp.float32)


@wp.kernel
def reconstruct_extruded_curvature_kernel(
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    geometry_valid: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=float),
    curvature_valid: wp.array3d(dtype=wp.int32),
    curvature_required: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    closed_x: int,
    closed_y: int,
    nx: int,
    ny: int,
):
    i, j, k = wp.tid()
    curvature[i, j, k] = 0.0
    curvature_valid[i, j, k] = 0
    curvature_required[i, j, k] = 0
    if (
        solid[i, j, k] != 0
        or flags[i, j, k] != 1
        or geometry_valid[i, j, k] == 0
    ):
        return

    required = plane_offset[i, j, k] < 0.0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            ni = _coordinate(i + di, nx, closed_x)
            nj = _coordinate(j + dj, ny, closed_y)
            if solid[ni, nj, k] == 0 and flags[ni, nj, k] == 0:
                required = True
    if not required:
        return
    curvature_required[i, j, k] = 1
    wp.atomic_add(counts, 0, 1)

    n = normal[i, j, k]
    planar_magnitude = wp.sqrt(n[0] * n[0] + n[1] * n[1])
    if not wp.isfinite(planar_magnitude) or planar_magnitude <= 1.0e-8:
        wp.atomic_add(counts, 2, 1)
        return
    planar_normal = wp.vec3(
        n[0] / planar_magnitude,
        n[1] / planar_magnitude,
        0.0,
    )
    tangent = wp.vec3(-planar_normal[1], planar_normal[0], 0.0)
    center_offset = plane_offset[i, j, k]
    gram = float(0.0)
    rhs = float(0.0)
    sample_count = int(0)

    # In the local frame the center PLIC normal already imposes zero slope.
    # Fitting z=a*x^2 therefore needs one resolved neighboring interface.
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if di == 0 and dj == 0:
                continue
            ni = _coordinate(i + di, nx, closed_x)
            nj = _coordinate(j + dj, ny, closed_y)
            if (
                solid[ni, nj, k] != 0
                or flags[ni, nj, k] != 1
                or geometry_valid[ni, nj, k] == 0
            ):
                continue
            displacement = wp.vec3(float(di), float(dj), 0.0)
            x = wp.dot(displacement, tangent)
            row = x * x
            if row <= 1.0e-8:
                continue
            neighbor_offset = _plic_offset(fill[ni, nj, k], planar_normal)
            z = (
                wp.dot(displacement, planar_normal)
                + neighbor_offset
                - center_offset
            )
            gram += row * row
            rhs += row * z
            sample_count += 1

    if sample_count == 0 or not wp.isfinite(gram) or gram <= 1.0e-8:
        wp.atomic_add(counts, 1, 1)
        return
    value = rhs / gram
    if not wp.isfinite(value):
        wp.atomic_add(counts, 2, 1)
        return
    curvature[i, j, k] = value
    curvature_valid[i, j, k] = 1


@wp.kernel
def reconstruct_bulk_curvature_3d_kernel(
    fill: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    geometry_valid: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=float),
    curvature_valid: wp.array3d(dtype=wp.int32),
    curvature_required: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    max_condition_number: float,
    closed_x: int,
    closed_y: int,
    closed_z: int,
    fit_wall_contact: int,
    nx: int,
    ny: int,
    nz: int,
):
    """Fit a quadratic interface patch away from unresolved wall contact lines."""

    i, j, k = wp.tid()
    curvature[i, j, k] = 0.0
    curvature_valid[i, j, k] = 0
    curvature_required[i, j, k] = 0
    if (
        solid[i, j, k] != 0
        or flags[i, j, k] != 1
        or geometry_valid[i, j, k] == 0
    ):
        return

    required = int(plane_offset[i, j, k] < 0.0)
    touches_wall = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _coordinate(i + di, nx, closed_x)
                nj = _coordinate(j + dj, ny, closed_y)
                nk = _coordinate(k + dk, nz, closed_z)
                if solid[ni, nj, nk] != 0:
                    touches_wall = 1
                elif flags[ni, nj, nk] == 0:
                    required = 1
    if required == 0:
        return
    curvature_required[i, j, k] = 1
    wp.atomic_add(counts, 0, 1)
    if touches_wall != 0 and fit_wall_contact == 0:
        wp.atomic_add(counts, 4, 1)
        return

    n = normal[i, j, k]
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[1]) <= wp.abs(n[0]) and wp.abs(n[1]) <= wp.abs(n[2]):
        axis = wp.vec3(0.0, 1.0, 0.0)
    elif wp.abs(n[2]) <= wp.abs(n[0]) and wp.abs(n[2]) <= wp.abs(n[1]):
        axis = wp.vec3(0.0, 0.0, 1.0)
    tangent_y = wp.normalize(wp.cross(n, axis))
    tangent_x = wp.cross(tangent_y, n)
    center_offset = plane_offset[i, j, k]
    gram = Mat55f()
    rhs = Vec5f()
    sample_count = int(0)

    wide_stencil = touches_wall
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-2, 3):
                if wide_stencil == 0 and (
                    wp.abs(di) > 1 or wp.abs(dj) > 1 or wp.abs(dk) > 1
                ):
                    continue
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _coordinate(i + di, nx, closed_x)
                nj = _coordinate(j + dj, ny, closed_y)
                nk = _coordinate(k + dk, nz, closed_z)
                if (
                    solid[ni, nj, nk] != 0
                    or flags[ni, nj, nk] != 1
                    or geometry_valid[ni, nj, nk] == 0
                ):
                    continue
                displacement = wp.vec3(float(di), float(dj), float(dk))
                x = wp.dot(displacement, tangent_x)
                y = wp.dot(displacement, tangent_y)
                neighbor_offset = _plic_offset(fill[ni, nj, nk], n)
                z = wp.dot(displacement, n) + neighbor_offset - center_offset
                row = Vec5f(x * x, y * y, x * y, x, y)
                for row_index in range(5):
                    rhs[row_index] += row[row_index] * z
                    for column_index in range(5):
                        gram[row_index, column_index] += (
                            row[row_index] * row[column_index]
                        )
                sample_count += 1
    if sample_count < 5:
        wp.atomic_add(counts, 1, 1)
        return

    largest_entry = float(0.0)
    for row_index in range(5):
        for column_index in range(5):
            largest_entry = wp.max(
                largest_entry, wp.abs(gram[row_index, column_index])
            )
    largest_pivot = float(0.0)
    smallest_pivot = float(1.0e30)
    full_rank = int(1)
    for column_index in range(5):
        pivot_row = column_index
        pivot_magnitude = wp.abs(gram[column_index, column_index])
        for candidate_row in range(column_index + 1, 5):
            candidate = wp.abs(gram[candidate_row, column_index])
            if candidate > pivot_magnitude:
                pivot_magnitude = candidate
                pivot_row = candidate_row
        if pivot_magnitude <= 1.0e-7 * largest_entry:
            full_rank = 0
        if full_rank != 0:
            if pivot_row != column_index:
                for swap_column in range(5):
                    temporary = gram[column_index, swap_column]
                    gram[column_index, swap_column] = gram[pivot_row, swap_column]
                    gram[pivot_row, swap_column] = temporary
                temporary_rhs = rhs[column_index]
                rhs[column_index] = rhs[pivot_row]
                rhs[pivot_row] = temporary_rhs
            pivot = gram[column_index, column_index]
            pivot_absolute = wp.abs(pivot)
            largest_pivot = wp.max(largest_pivot, pivot_absolute)
            smallest_pivot = wp.min(smallest_pivot, pivot_absolute)
            for eliminate_row in range(column_index + 1, 5):
                factor = gram[eliminate_row, column_index] / pivot
                gram[eliminate_row, column_index] = 0.0
                for eliminate_column in range(column_index + 1, 5):
                    gram[eliminate_row, eliminate_column] -= (
                        factor * gram[column_index, eliminate_column]
                    )
                rhs[eliminate_row] -= factor * rhs[column_index]
    if (
        full_rank == 0
        or smallest_pivot * max_condition_number * max_condition_number
        < largest_pivot
    ):
        wp.atomic_add(counts, 2, 1)
        return

    coefficients = Vec5f()
    for reverse_index in range(5):
        row_index = 4 - reverse_index
        value = rhs[row_index]
        for column_index in range(row_index + 1, 5):
            value -= gram[row_index, column_index] * coefficients[column_index]
        coefficients[row_index] = value / gram[row_index, row_index]
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    slope_x = coefficients[3]
    slope_y = coefficients[4]
    denominator = wp.pow(1.0 + slope_x * slope_x + slope_y * slope_y, 1.5)
    value = (
        a * (1.0 + slope_y * slope_y)
        + b * (1.0 + slope_x * slope_x)
        - c * slope_x * slope_y
    ) / denominator
    if not wp.isfinite(value):
        wp.atomic_add(counts, 3, 1)
        return
    curvature[i, j, k] = value
    curvature_valid[i, j, k] = 1
    if touches_wall != 0:
        wp.atomic_add(counts, 5, 1)
