# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp face-velocity extension kernels for geometric FSL/VOF coupling."""

from __future__ import annotations

import warp as wp


Vec4f = wp.types.vector(4, wp.float32)
Mat44f = wp.types.matrix((4, 4), wp.float32)
SOLID = wp.constant(3)


@wp.func
def _fsl_courant_neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + 4 * size) % size
        return -1
    return value


@wp.func
def _fsl_axis_velocity(
    moments: wp.array(dtype=float), cell: int, axis: int, stride: int
) -> float:
    return moments[(axis + 1) * stride + cell] / moments[cell]


@wp.kernel
def construct_fsl_face_courant_kernel(
    moments: wp.array(dtype=float),
    active: wp.array3d(dtype=wp.int32),
    fill_level: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    face_courant: wp.array3d(dtype=float),
    donor_count: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    axis: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    maximum_fit_condition: float,
    maximum_courant: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    face_courant[i, j, k] = 0.0
    donor_count[i, j, k] = 0
    face = i
    size = nx
    if axis == 1:
        face = j
        size = ny
    elif axis == 2:
        face = k
        size = nz
    canonical_face = face
    axis_periodic = periodic_x
    if axis == 1:
        axis_periodic = periodic_y
    elif axis == 2:
        axis_periodic = periodic_z
    if face == size:
        if axis_periodic == 0:
            return
        canonical_face = 0
    if canonical_face == 0 and axis_periodic == 0:
        return

    lower_i = i
    lower_j = j
    lower_k = k
    upper_i = i
    upper_j = j
    upper_k = k
    if axis == 0:
        lower_i = _fsl_courant_neighbor(canonical_face - 1, nx, periodic_x)
        upper_i = _fsl_courant_neighbor(canonical_face, nx, periodic_x)
    elif axis == 1:
        lower_j = _fsl_courant_neighbor(canonical_face - 1, ny, periodic_y)
        upper_j = _fsl_courant_neighbor(canonical_face, ny, periodic_y)
    else:
        lower_k = _fsl_courant_neighbor(canonical_face - 1, nz, periodic_z)
        upper_k = _fsl_courant_neighbor(canonical_face, nz, periodic_z)
    if (
        lower_i < 0
        or lower_j < 0
        or lower_k < 0
        or upper_i < 0
        or upper_j < 0
        or upper_k < 0
    ):
        wp.atomic_add(counts, 0, 1)
        return

    if (
        flags[lower_i, lower_j, lower_k] == SOLID
        or flags[upper_i, upper_j, upper_k] == SOLID
    ):
        wp.atomic_add(counts, 5, 1)
        return

    lower_active = active[lower_i, lower_j, lower_k] != 0
    upper_active = active[upper_i, upper_j, upper_k] != 0
    value = float(0.0)
    if lower_active and upper_active:
        lower_cell = lower_i * ny * nz + lower_j * nz + lower_k
        upper_cell = upper_i * ny * nz + upper_j * nz + upper_k
        lower_rho = moments[lower_cell]
        upper_rho = moments[upper_cell]
        valid = (
            wp.isfinite(lower_rho)
            and lower_rho > 0.0
            and wp.isfinite(upper_rho)
            and upper_rho > 0.0
        )
        if valid:
            lower_velocity = _fsl_axis_velocity(moments, lower_cell, axis, stride)
            upper_velocity = _fsl_axis_velocity(moments, upper_cell, axis, stride)
            valid = wp.isfinite(lower_velocity) and wp.isfinite(upper_velocity)
            value = 0.5 * (lower_velocity + upper_velocity)
        if not valid:
            wp.atomic_add(counts, 0, 1)
            return
        donor_count[i, j, k] = 2
    elif (
        fill_level[lower_i, lower_j, lower_k] == 0.0
        and fill_level[upper_i, upper_j, upper_k] == 0.0
    ):
        value = 0.0
    else:
        gram = Mat44f()
        rhs = Vec4f()
        donors = int(0)
        valid_donors = int(1)
        uniform_samples = int(1)
        first_sample = float(0.0)
        for along in range(-3, 3):
            for transverse_0 in range(-2, 3):
                for transverse_1 in range(-2, 3):
                    donor_i = i
                    donor_j = j
                    donor_k = k
                    dx = float(0.0)
                    dy = float(0.0)
                    dz = float(0.0)
                    if axis == 0:
                        donor_i = _fsl_courant_neighbor(
                            canonical_face + along, nx, periodic_x
                        )
                        donor_j = _fsl_courant_neighbor(j + transverse_0, ny, periodic_y)
                        donor_k = _fsl_courant_neighbor(k + transverse_1, nz, periodic_z)
                        dx = float(along) + 0.5
                        dy = float(transverse_0)
                        dz = float(transverse_1)
                    elif axis == 1:
                        donor_i = _fsl_courant_neighbor(i + transverse_0, nx, periodic_x)
                        donor_j = _fsl_courant_neighbor(
                            canonical_face + along, ny, periodic_y
                        )
                        donor_k = _fsl_courant_neighbor(k + transverse_1, nz, periodic_z)
                        dx = float(transverse_0)
                        dy = float(along) + 0.5
                        dz = float(transverse_1)
                    else:
                        donor_i = _fsl_courant_neighbor(i + transverse_0, nx, periodic_x)
                        donor_j = _fsl_courant_neighbor(j + transverse_1, ny, periodic_y)
                        donor_k = _fsl_courant_neighbor(
                            canonical_face + along, nz, periodic_z
                        )
                        dx = float(transverse_0)
                        dy = float(transverse_1)
                        dz = float(along) + 0.5
                    if donor_i < 0 or donor_j < 0 or donor_k < 0:
                        continue
                    if active[donor_i, donor_j, donor_k] == 0:
                        continue
                    cell = donor_i * ny * nz + donor_j * nz + donor_k
                    rho = moments[cell]
                    sample = float(0.0)
                    if wp.isfinite(rho) and rho > 0.0:
                        sample = _fsl_axis_velocity(moments, cell, axis, stride)
                    if not wp.isfinite(rho) or rho <= 0.0 or not wp.isfinite(sample):
                        valid_donors = 0
                        continue
                    if donors == 0:
                        first_sample = sample
                    elif sample != first_sample:
                        uniform_samples = 0
                    row = Vec4f(1.0, dx, dy, dz)
                    weight = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz)
                    for row_index in range(4):
                        rhs[row_index] += weight * row[row_index] * sample
                        for column_index in range(4):
                            gram[row_index, column_index] += (
                                weight * row[row_index] * row[column_index]
                            )
                    donors += 1
        donor_count[i, j, k] = donors
        if valid_donors == 0:
            wp.atomic_add(counts, 0, 1)
            return
        if donors < 4:
            wp.atomic_add(counts, 1, 1)
            return
        if uniform_samples != 0:
            if wp.abs(first_sample) > maximum_courant:
                wp.atomic_add(counts, 3, 1)
                return
            face_courant[i, j, k] = first_sample
            wp.atomic_add(counts, 4, 1)
            return

        largest_entry = float(0.0)
        for row_index in range(4):
            for column_index in range(4):
                largest_entry = wp.max(
                    largest_entry, wp.abs(gram[row_index, column_index])
                )
        largest_pivot = float(0.0)
        smallest_pivot = float(1.0e30)
        full_rank = int(1)
        for column_index in range(4):
            pivot_row = column_index
            pivot_magnitude = wp.abs(gram[column_index, column_index])
            for candidate_row in range(column_index + 1, 4):
                candidate = wp.abs(gram[candidate_row, column_index])
                if candidate > pivot_magnitude:
                    pivot_magnitude = candidate
                    pivot_row = candidate_row
            if pivot_magnitude <= 1.0e-7 * largest_entry:
                full_rank = 0
            if full_rank != 0:
                if pivot_row != column_index:
                    for swap_column in range(4):
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
                for eliminate_row in range(column_index + 1, 4):
                    factor = gram[eliminate_row, column_index] / pivot
                    gram[eliminate_row, column_index] = 0.0
                    for eliminate_column in range(column_index + 1, 4):
                        gram[eliminate_row, eliminate_column] -= (
                            factor * gram[column_index, eliminate_column]
                        )
                    rhs[eliminate_row] -= factor * rhs[column_index]
        if (
            full_rank == 0
            or smallest_pivot * maximum_fit_condition * maximum_fit_condition
            < largest_pivot
        ):
            wp.atomic_add(counts, 2, 1)
            return
        coefficients = Vec4f()
        for reverse_index in range(4):
            row_index = 3 - reverse_index
            coefficient = rhs[row_index]
            for column_index in range(row_index + 1, 4):
                coefficient -= gram[row_index, column_index] * coefficients[column_index]
            coefficients[row_index] = coefficient / gram[row_index, row_index]
        value = coefficients[0]
        wp.atomic_add(counts, 4, 1)

    # Preserve the rejected value for host-side failure diagnostics. The
    # builder raises before a rejected field can be consumed by transport.
    face_courant[i, j, k] = value
    if not wp.isfinite(value) or wp.abs(value) > maximum_courant:
        wp.atomic_add(counts, 3, 1)
        return
