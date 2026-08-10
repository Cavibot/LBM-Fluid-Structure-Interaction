# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free Hodge projection kernels for geometric VOF face Courants."""

from __future__ import annotations

import warp as wp


GAS = wp.constant(0)
INTERFACE = wp.constant(1)
LIQUID = wp.constant(2)
SOLID = wp.constant(3)


@wp.func
def _active(flag: int) -> bool:
    return flag == INTERFACE or flag == LIQUID


@wp.func
def _neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.kernel
def build_projection_system_kernel(
    face_x: wp.array3d(dtype=float),
    face_y: wp.array3d(dtype=float),
    face_z: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    rhs: wp.array3d(dtype=float),
    inverse_diagonal: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    counts: wp.array(dtype=wp.int32),
    maximum_divergence: wp.array(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if not _active(flags[i, j, k]):
        rhs[i, j, k] = 0.0
        inverse_diagonal[i, j, k] = 0.0
        divergence[i, j, k] = 0.0
        return
    lower_x = _neighbor(i - 1, nx, periodic_x)
    upper_x = _neighbor(i + 1, nx, periodic_x)
    lower_y = _neighbor(j - 1, ny, periodic_y)
    upper_y = _neighbor(j + 1, ny, periodic_y)
    lower_z = _neighbor(k - 1, nz, periodic_z)
    upper_z = _neighbor(k + 1, nz, periodic_z)
    flux_x_lower = face_x[i, j, k]
    flux_x_upper = face_x[i + 1, j, k]
    flux_y_lower = face_y[i, j, k]
    flux_y_upper = face_y[i, j + 1, k]
    flux_z_lower = face_z[i, j, k]
    flux_z_upper = face_z[i, j, k + 1]
    if lower_x >= 0 and flags[lower_x, j, k] == SOLID:
        flux_x_lower = 0.0
    if upper_x >= 0 and flags[upper_x, j, k] == SOLID:
        flux_x_upper = 0.0
    if lower_y >= 0 and flags[i, lower_y, k] == SOLID:
        flux_y_lower = 0.0
    if upper_y >= 0 and flags[i, upper_y, k] == SOLID:
        flux_y_upper = 0.0
    if lower_z >= 0 and flags[i, j, lower_z] == SOLID:
        flux_z_lower = 0.0
    if upper_z >= 0 and flags[i, j, upper_z] == SOLID:
        flux_z_upper = 0.0
    value = (
        flux_x_upper - flux_x_lower
        + flux_y_upper - flux_y_lower
        + flux_z_upper - flux_z_lower
    )
    diagonal = int(0)
    if lower_x >= 0 and flags[lower_x, j, k] != SOLID:
        diagonal += 1
    if upper_x >= 0 and flags[upper_x, j, k] != SOLID:
        diagonal += 1
    if lower_y >= 0 and flags[i, lower_y, k] != SOLID:
        diagonal += 1
    if upper_y >= 0 and flags[i, upper_y, k] != SOLID:
        diagonal += 1
    if lower_z >= 0 and flags[i, j, lower_z] != SOLID:
        diagonal += 1
    if upper_z >= 0 and flags[i, j, upper_z] != SOLID:
        diagonal += 1
    rhs[i, j, k] = -value
    inverse_diagonal[i, j, k] = 1.0 / float(diagonal)
    divergence[i, j, k] = value
    wp.atomic_add(counts, 0, 1)
    wp.atomic_max(maximum_divergence, 0, wp.abs(value))


@wp.kernel
def apply_projection_operator_kernel(
    x: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    inverse_diagonal: wp.array3d(dtype=float),
    result: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if not _active(flags[i, j, k]):
        result[i, j, k] = 0.0
        return
    value = x[i, j, k] / inverse_diagonal[i, j, k]
    ni = _neighbor(i - 1, nx, periodic_x)
    if ni >= 0 and _active(flags[ni, j, k]):
        value -= x[ni, j, k]
    ni = _neighbor(i + 1, nx, periodic_x)
    if ni >= 0 and _active(flags[ni, j, k]):
        value -= x[ni, j, k]
    nj = _neighbor(j - 1, ny, periodic_y)
    if nj >= 0 and _active(flags[i, nj, k]):
        value -= x[i, nj, k]
    nj = _neighbor(j + 1, ny, periodic_y)
    if nj >= 0 and _active(flags[i, nj, k]):
        value -= x[i, nj, k]
    nk = _neighbor(k - 1, nz, periodic_z)
    if nk >= 0 and _active(flags[i, j, nk]):
        value -= x[i, j, nk]
    nk = _neighbor(k + 1, nz, periodic_z)
    if nk >= 0 and _active(flags[i, j, nk]):
        value -= x[i, j, nk]
    result[i, j, k] = value


@wp.kernel
def combine_kernel(
    a: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    result: wp.array3d(dtype=float),
    alpha: float,
    beta: float,
):
    i, j, k = wp.tid()
    result[i, j, k] = alpha * a[i, j, k] + beta * b[i, j, k]


@wp.kernel
def apply_preconditioner_kernel(
    residual: wp.array3d(dtype=float),
    inverse_diagonal: wp.array3d(dtype=float),
    result: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    result[i, j, k] = residual[i, j, k] * inverse_diagonal[i, j, k]


@wp.kernel
def update_solution_residual_kernel(
    solution: wp.array3d(dtype=float),
    residual: wp.array3d(dtype=float),
    direction: wp.array3d(dtype=float),
    operator_direction: wp.array3d(dtype=float),
    alpha: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    value = alpha[0]
    solution[i, j, k] += value * direction[i, j, k]
    residual[i, j, k] -= value * operator_direction[i, j, k]


@wp.kernel
def update_direction_kernel(
    direction: wp.array3d(dtype=float),
    preconditioned: wp.array3d(dtype=float),
    beta: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    direction[i, j, k] = preconditioned[i, j, k] + beta[0] * direction[i, j, k]


@wp.kernel
def dot_chunk_kernel(
    a: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    partial: wp.array(dtype=wp.float64),
    element_count: int,
    ny: int,
    nz: int,
    chunk_size: int,
):
    chunk = wp.tid()
    start = chunk * chunk_size
    end = wp.min(start + chunk_size, element_count)
    value = wp.float64(0.0)
    for linear in range(start, end):
        i = linear // (ny * nz)
        remainder = linear - i * ny * nz
        j = remainder // nz
        k = remainder - j * nz
        value += wp.float64(a[i, j, k]) * wp.float64(b[i, j, k])
    partial[chunk] = value


@wp.kernel
def reduce_double_chunks_kernel(
    source: wp.array(dtype=wp.float64),
    destination: wp.array(dtype=wp.float64),
    element_count: int,
    chunk_size: int,
):
    chunk = wp.tid()
    start = chunk * chunk_size
    end = wp.min(start + chunk_size, element_count)
    value = wp.float64(0.0)
    for index in range(start, end):
        value += source[index]
    destination[chunk] = value


@wp.kernel
def ratio_kernel(
    result: wp.array(dtype=float),
    numerator: wp.array(dtype=wp.float64),
    denominator: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    epsilon: float,
    converged_numerator_tolerance: wp.array(dtype=wp.float64),
):
    value = denominator[0]
    if not wp.isfinite(value) or wp.abs(value) <= epsilon:
        result[0] = 0.0
        numerator_value = numerator[0]
        if not wp.isfinite(numerator_value) or wp.abs(
            numerator_value
        ) > converged_numerator_tolerance[0]:
            wp.atomic_add(invalid, 0, 1)
    else:
        result[0] = float(numerator[0] / value)


@wp.kernel
def copy_double_scalar_kernel(
    destination: wp.array(dtype=wp.float64),
    source: wp.array(dtype=wp.float64),
):
    destination[0] = source[0]


@wp.kernel
def project_face_courant_kernel(
    source: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    destination: wp.array3d(dtype=float),
    counts: wp.array(dtype=wp.int32),
    maximum_correction: wp.array(dtype=float),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    face = i
    size = nx
    if axis == 1:
        face = j
        size = ny
    elif axis == 2:
        face = k
        size = nz
    if periodic == 0 and (face == 0 or face == size):
        destination[i, j, k] = source[i, j, k]
        return
    lower = face - 1
    upper = face
    if periodic != 0:
        lower = (lower + size) % size
        upper = upper % size
    li = i
    lj = j
    lk = k
    ui = i
    uj = j
    uk = k
    if axis == 0:
        li = lower
        ui = upper
    elif axis == 1:
        lj = lower
        uj = upper
    else:
        lk = lower
        uk = upper
    lower_active = _active(flags[li, lj, lk])
    upper_active = _active(flags[ui, uj, uk])
    if flags[li, lj, lk] == SOLID or flags[ui, uj, uk] == SOLID:
        destination[i, j, k] = 0.0
        return
    correction = float(0.0)
    if lower_active or upper_active:
        lower_pressure = float(0.0)
        upper_pressure = float(0.0)
        if lower_active:
            lower_pressure = pressure[li, lj, lk]
        if upper_active:
            upper_pressure = pressure[ui, uj, uk]
        correction = upper_pressure - lower_pressure
        wp.atomic_add(counts, 1, 1)
        wp.atomic_max(maximum_correction, 0, wp.abs(correction))
    destination[i, j, k] = source[i, j, k] - correction


@wp.kernel
def measure_projected_divergence_kernel(
    face_x: wp.array3d(dtype=float),
    face_y: wp.array3d(dtype=float),
    face_z: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    maximum_divergence: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    if _active(flags[i, j, k]):
        divergence = (
            face_x[i + 1, j, k]
            - face_x[i, j, k]
            + face_y[i, j + 1, k]
            - face_y[i, j, k]
            + face_z[i, j, k + 1]
            - face_z[i, j, k]
        )
        wp.atomic_max(maximum_divergence, 0, wp.abs(divergence))
