# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Momentum ledger coupling between geometric transport and HOME moments."""

import warp as wp


GAS = 0
INTERFACE = 1
LIQUID = 2


@wp.kernel
def initialize_liquid_momentum_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    momentum: wp.array3d(dtype=wp.vec3),
    invalid: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell_mass = mass[i, j, k]
    flag = flags[i, j, k]
    if solid[i, j, k] != 0 or flag == GAS:
        momentum[i, j, k] = wp.vec3(0.0)
        return
    cell = i * ny * nz + j * nz + k
    rho = moments[cell]
    if (flag != INTERFACE and flag != LIQUID) or not wp.isfinite(cell_mass) or cell_mass < 0.0 or not wp.isfinite(rho) or rho <= 0.0:
        momentum[i, j, k] = wp.vec3(0.0)
        wp.atomic_add(invalid, 0, 1)
        return
    value = wp.vec3(
        cell_mass * moments[stride + cell] / rho,
        cell_mass * moments[2 * stride + cell] / rho,
        cell_mass * moments[3 * stride + cell] / rho,
    )
    if not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
        wp.atomic_add(invalid, 0, 1)
        value = wp.vec3(0.0)
    momentum[i, j, k] = value


@wp.kernel
def initialize_projected_liquid_momentum_kernel(
    face_x: wp.array3d(dtype=float),
    face_y: wp.array3d(dtype=float),
    face_z: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    momentum: wp.array3d(dtype=wp.vec3),
    invalid: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    cell_mass = mass[i, j, k]
    flag = flags[i, j, k]
    if solid[i, j, k] != 0 or flag == GAS:
        momentum[i, j, k] = wp.vec3(0.0)
        return
    if (
        (flag != INTERFACE and flag != LIQUID)
        or not wp.isfinite(cell_mass)
        or cell_mass < 0.0
    ):
        momentum[i, j, k] = wp.vec3(0.0)
        wp.atomic_add(invalid, 0, 1)
        return
    velocity = wp.vec3(
        0.5 * (face_x[i, j, k] + face_x[i + 1, j, k]),
        0.5 * (face_y[i, j, k] + face_y[i, j + 1, k]),
        0.5 * (face_z[i, j, k] + face_z[i, j, k + 1]),
    )
    value = cell_mass * velocity
    if (
        not wp.isfinite(value[0])
        or not wp.isfinite(value[1])
        or not wp.isfinite(value[2])
    ):
        wp.atomic_add(invalid, 0, 1)
        value = wp.vec3(0.0)
    momentum[i, j, k] = value


@wp.kernel
def apply_transported_momentum_kernel(
    source_moments: wp.array(dtype=float),
    source_mass: wp.array3d(dtype=float),
    candidate_moments: wp.array(dtype=float),
    transported_mass: wp.array3d(dtype=float),
    transported_fill: wp.array3d(dtype=float),
    transported_momentum: wp.array3d(dtype=wp.vec3),
    target_flags: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = target_flags[i, j, k]
    if solid[i, j, k] != 0 or flag == GAS:
        return
    cell = i * ny * nz + j * nz + k
    old_rho = source_moments[cell]
    new_rho = candidate_moments[cell]
    old_mass = source_mass[i, j, k]
    new_mass = transported_mass[i, j, k]
    if not wp.isfinite(old_rho) or old_rho <= 0.0 or not wp.isfinite(new_rho) or new_rho <= 0.0 or not wp.isfinite(old_mass) or old_mass < 0.0 or not wp.isfinite(new_mass) or new_mass <= 1.0e-12:
        wp.atomic_add(invalid, 0, 1)
        return
    old_velocity = wp.vec3(
        source_moments[stride + cell] / old_rho,
        source_moments[2 * stride + cell] / old_rho,
        source_moments[3 * stride + cell] / old_rho,
    )
    candidate_j = wp.vec3(
        candidate_moments[stride + cell],
        candidate_moments[2 * stride + cell],
        candidate_moments[3 * stride + cell],
    )
    candidate_velocity = candidate_j / new_rho
    liquid_momentum = transported_momentum[i, j, k] + new_mass * (
        candidate_velocity - old_velocity
    )
    liquid_velocity = liquid_momentum / new_mass
    fraction = 1.0
    if flag == INTERFACE:
        fraction = wp.clamp(transported_fill[i, j, k], 0.0, 1.0)
    target_velocity = candidate_velocity + fraction * (
        liquid_velocity - candidate_velocity
    )
    delta = target_velocity - candidate_velocity
    if not wp.isfinite(delta[0]) or not wp.isfinite(delta[1]) or not wp.isfinite(delta[2]):
        wp.atomic_add(invalid, 0, 1)
        return
    dx = delta[0]
    dy = delta[1]
    dz = delta[2]
    jx = candidate_j[0]
    jy = candidate_j[1]
    jz = candidate_j[2]
    candidate_moments[4 * stride + cell] += 2.0 * dx * jx + new_rho * dx * dx
    candidate_moments[5 * stride + cell] += 2.0 * dy * jy + new_rho * dy * dy
    candidate_moments[6 * stride + cell] += 2.0 * dz * jz + new_rho * dz * dz
    candidate_moments[7 * stride + cell] += dx * jy + dy * jx + new_rho * dx * dy
    candidate_moments[8 * stride + cell] += dx * jz + dz * jx + new_rho * dx * dz
    candidate_moments[9 * stride + cell] += dy * jz + dz * jy + new_rho * dy * dz
    candidate_moments[stride + cell] = new_rho * target_velocity[0]
    candidate_moments[2 * stride + cell] = new_rho * target_velocity[1]
    candidate_moments[3 * stride + cell] = new_rho * target_velocity[2]
