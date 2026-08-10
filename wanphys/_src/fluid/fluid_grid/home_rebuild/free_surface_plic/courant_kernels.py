# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for HOME face Courants and matrix-free Hodge projection."""

from __future__ import annotations

import warp as wp


@wp.func
def _active(flag: int, solid: int) -> bool:
    return solid == 0 and (flag == 1 or flag == 2)


@wp.func
def _neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.func
def _linear(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return i * ny * nz + j * nz + k


@wp.kernel
def build_face_courant_kernel(
    moments: wp.array(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    face_courant: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    maximum_courant: wp.array(dtype=float),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    coordinate = i if axis == 0 else (j if axis == 1 else k)
    size = nx if axis == 0 else (ny if axis == 1 else nz)
    face_courant[i, j, k] = 0.0
    if periodic == 0 and (coordinate == 0 or coordinate == size):
        return
    lower_coordinate = (coordinate - 1 + size) % size
    upper_coordinate = coordinate % size
    li, lj, lk = i, j, k
    ui, uj, uk = i, j, k
    if axis == 0:
        li, ui = lower_coordinate, upper_coordinate
    elif axis == 1:
        lj, uj = lower_coordinate, upper_coordinate
    else:
        lk, uk = lower_coordinate, upper_coordinate
    if solid[li, lj, lk] != 0 or solid[ui, uj, uk] != 0:
        return
    lower_active = _active(flags[li, lj, lk], solid[li, lj, lk])
    upper_active = _active(flags[ui, uj, uk], solid[ui, uj, uk])
    if not lower_active and not upper_active:
        return
    lower_velocity = float(0.0)
    upper_velocity = float(0.0)
    if lower_active:
        index = _linear(li, lj, lk, ny, nz)
        density = moments[index]
        if not wp.isfinite(density) or density <= 0.0:
            wp.atomic_add(invalid_count, 0, 1)
            return
        lower_velocity = moments[(axis + 1) * stride + index] / density
    if upper_active:
        index = _linear(ui, uj, uk, ny, nz)
        density = moments[index]
        if not wp.isfinite(density) or density <= 0.0:
            wp.atomic_add(invalid_count, 0, 1)
            return
        upper_velocity = moments[(axis + 1) * stride + index] / density
    value = lower_velocity
    if lower_active and upper_active:
        value = 0.5 * (lower_velocity + upper_velocity)
    elif upper_active:
        value = upper_velocity
    if not wp.isfinite(value):
        wp.atomic_add(invalid_count, 0, 1)
        return
    face_courant[i, j, k] = value
    wp.atomic_max(maximum_courant, 0, wp.abs(value))


@wp.kernel
def build_projection_system_kernel(
    face_x: wp.array3d(dtype=float),
    face_y: wp.array3d(dtype=float),
    face_z: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    rhs: wp.array3d(dtype=float),
    inverse_diagonal: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    active_count: wp.array(dtype=wp.int32),
    maximum_divergence: wp.array(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if not _active(flags[i, j, k], solid[i, j, k]):
        rhs[i, j, k] = 0.0
        inverse_diagonal[i, j, k] = 0.0
        divergence[i, j, k] = 0.0
        return
    value = (
        face_x[i + 1, j, k] - face_x[i, j, k]
        + face_y[i, j + 1, k] - face_y[i, j, k]
        + face_z[i, j, k + 1] - face_z[i, j, k]
    )
    diagonal = int(0)
    for sign in range(2):
        step = -1 if sign == 0 else 1
        ni = _neighbor(i + step, nx, periodic_x)
        if ni >= 0 and ni != i and solid[ni, j, k] == 0:
            diagonal += 1
        nj = _neighbor(j + step, ny, periodic_y)
        if nj >= 0 and nj != j and solid[i, nj, k] == 0:
            diagonal += 1
        nk = _neighbor(k + step, nz, periodic_z)
        if nk >= 0 and nk != k and solid[i, j, nk] == 0:
            diagonal += 1
    if diagonal == 0:
        rhs[i, j, k] = 0.0
        inverse_diagonal[i, j, k] = 0.0
        divergence[i, j, k] = value
        return
    rhs[i, j, k] = -value
    inverse_diagonal[i, j, k] = 1.0 / float(diagonal)
    divergence[i, j, k] = value
    wp.atomic_add(active_count, 0, 1)
    wp.atomic_max(maximum_divergence, 0, wp.abs(value))


@wp.kernel
def apply_operator_kernel(
    x: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
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
    if not _active(flags[i, j, k], solid[i, j, k]):
        result[i, j, k] = 0.0
        return
    inverse = inverse_diagonal[i, j, k]
    if inverse <= 0.0:
        result[i, j, k] = 0.0
        return
    value = x[i, j, k] / inverse
    for sign in range(2):
        step = -1 if sign == 0 else 1
        ni = _neighbor(i + step, nx, periodic_x)
        if ni >= 0 and ni != i and _active(flags[ni, j, k], solid[ni, j, k]):
            value -= x[ni, j, k]
        nj = _neighbor(j + step, ny, periodic_y)
        if nj >= 0 and nj != j and _active(flags[i, nj, k], solid[i, nj, k]):
            value -= x[i, nj, k]
        nk = _neighbor(k + step, nz, periodic_z)
        if nk >= 0 and nk != k and _active(flags[i, j, nk], solid[i, j, nk]):
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
def precondition_kernel(
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
def reduce_chunks_kernel(
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
def scalar_ratio_kernel(
    result: wp.array(dtype=float),
    numerator: wp.array(dtype=wp.float64),
    denominator: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    epsilon: float,
):
    if not wp.isfinite(denominator[0]) or wp.abs(denominator[0]) <= epsilon:
        result[0] = 0.0
        if wp.abs(numerator[0]) > epsilon:
            wp.atomic_add(invalid, 0, 1)
    else:
        result[0] = float(numerator[0] / denominator[0])


@wp.kernel
def copy_scalar_kernel(
    destination: wp.array(dtype=wp.float64),
    source: wp.array(dtype=wp.float64),
):
    destination[0] = source[0]


@wp.kernel
def project_face_kernel(
    source: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    destination: wp.array3d(dtype=float),
    maximum_correction: wp.array(dtype=float),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    coordinate = i if axis == 0 else (j if axis == 1 else k)
    size = nx if axis == 0 else (ny if axis == 1 else nz)
    if periodic == 0 and (coordinate == 0 or coordinate == size):
        destination[i, j, k] = 0.0
        return
    lower_coordinate = (coordinate - 1 + size) % size
    upper_coordinate = coordinate % size
    li, lj, lk = i, j, k
    ui, uj, uk = i, j, k
    if axis == 0:
        li, ui = lower_coordinate, upper_coordinate
    elif axis == 1:
        lj, uj = lower_coordinate, upper_coordinate
    else:
        lk, uk = lower_coordinate, upper_coordinate
    if solid[li, lj, lk] != 0 or solid[ui, uj, uk] != 0:
        destination[i, j, k] = 0.0
        return
    lower_active = _active(flags[li, lj, lk], solid[li, lj, lk])
    upper_active = _active(flags[ui, uj, uk], solid[ui, uj, uk])
    if not lower_active and not upper_active:
        destination[i, j, k] = 0.0
        return
    lower_pressure = pressure[li, lj, lk] if lower_active else 0.0
    upper_pressure = pressure[ui, uj, uk] if upper_active else 0.0
    correction = upper_pressure - lower_pressure
    destination[i, j, k] = source[i, j, k] - correction
    wp.atomic_max(maximum_correction, 0, wp.abs(correction))


@wp.kernel
def measure_divergence_kernel(
    face_x: wp.array3d(dtype=float),
    face_y: wp.array3d(dtype=float),
    face_z: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    maximum_divergence: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    if _active(flags[i, j, k], solid[i, j, k]):
        value = (
            face_x[i + 1, j, k] - face_x[i, j, k]
            + face_y[i, j + 1, k] - face_y[i, j, k]
            + face_z[i, j, k + 1] - face_z[i, j, k]
        )
        wp.atomic_max(maximum_divergence, 0, wp.abs(value))
