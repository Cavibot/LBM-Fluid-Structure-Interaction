# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for one conservative directional PLIC sweep."""

from __future__ import annotations

import warp as wp

from .geometry_kernels import _plane_volume_fraction


@wp.func
def _minmod(left: float, right: float) -> float:
    if left * right <= 0.0:
        return 0.0
    return wp.sign(left) * wp.min(wp.abs(left), wp.abs(right))


@wp.func
def _swept_slab_volume(
    fill: float,
    normal: wp.vec3,
    offset: float,
    axis: int,
    signed_courant: float,
) -> float:
    width = wp.abs(signed_courant)
    if width == 0.0 or fill == 0.0:
        return 0.0
    if fill == 1.0:
        return width
    magnitude = wp.length(normal)
    if not wp.isfinite(magnitude) or magnitude <= 0.0 or not wp.isfinite(offset):
        return -1.0
    unit = normal / magnitude
    face_sign = 1.0 if signed_courant > 0.0 else -1.0
    slab_center = face_sign * (0.5 - 0.5 * width)
    scale = wp.vec3(1.0, 1.0, 1.0)
    scale[axis] = width
    transformed = wp.cw_mul(unit, scale)
    transformed_offset = offset / magnitude - unit[axis] * slab_center
    return width * _plane_volume_fraction(transformed_offset, transformed)


@wp.kernel
def face_flux_kernel(
    fill: wp.array3d(dtype=float),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    geometry_valid: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    face_courant: wp.array3d(dtype=float),
    face_flux: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    endpoint_fallback_count: wp.array(dtype=wp.int32),
    maximum_correction: wp.array(dtype=float),
    axis: int,
    periodic: int,
    interface_epsilon: float,
    endpoint_fallback_tolerance: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    coordinate = i if axis == 0 else (j if axis == 1 else k)
    size = nx if axis == 0 else (ny if axis == 1 else nz)
    signed_courant = face_courant[i, j, k]
    face_flux[i, j, k] = 0.0
    if not wp.isfinite(signed_courant) or wp.abs(signed_courant) > 1.0:
        wp.atomic_add(invalid_count, 0, 1)
        return
    boundary = coordinate == 0 or coordinate == size
    if boundary and periodic == 0:
        if wp.abs(signed_courant) > 2.0e-6:
            wp.atomic_add(invalid_count, 0, 1)
        return
    left = wp.vec3i(i, j, k)
    right = wp.vec3i(i, j, k)
    left[axis] = (coordinate - 1 + size) % size
    right[axis] = coordinate % size
    if solid[left[0], left[1], left[2]] != 0 or solid[right[0], right[1], right[2]] != 0:
        if wp.abs(signed_courant) > 2.0e-6:
            wp.atomic_add(invalid_count, 0, 1)
        return
    if signed_courant == 0.0:
        return
    donor = left if signed_courant > 0.0 else right
    donor_fill = fill[donor[0], donor[1], donor[2]]
    if donor_fill <= interface_epsilon:
        swept = 0.0
    elif donor_fill >= 1.0 - interface_epsilon:
        swept = wp.abs(signed_courant)
    elif geometry_valid[donor[0], donor[1], donor[2]] == 0:
        if donor_fill <= endpoint_fallback_tolerance:
            swept = 0.0
            wp.atomic_add(endpoint_fallback_count, 0, 1)
        elif donor_fill >= 1.0 - endpoint_fallback_tolerance:
            swept = wp.abs(signed_courant)
            wp.atomic_add(endpoint_fallback_count, 0, 1)
        else:
            wp.atomic_add(invalid_count, 0, 1)
            return
    else:
        swept = _swept_slab_volume(
            donor_fill,
            normal[donor[0], donor[1], donor[2]],
            plane_offset[donor[0], donor[1], donor[2]],
            axis,
            signed_courant,
        )
    lower = wp.max(0.0, wp.abs(signed_courant) - (1.0 - donor_fill))
    upper = wp.min(wp.abs(signed_courant), donor_fill)
    if swept < 0.0:
        wp.atomic_add(invalid_count, 0, 1)
        return
    bounded = wp.clamp(swept, lower, upper)
    wp.atomic_max(maximum_correction, 0, wp.abs(bounded - swept))
    face_flux[i, j, k] = bounded if signed_courant > 0.0 else -bounded


@wp.kernel
def update_fill_kernel(
    fill: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    face_flux: wp.array3d(dtype=float),
    face_courant: wp.array3d(dtype=float),
    compression: wp.array3d(dtype=wp.int32),
    updated_fill: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    maximum_correction: wp.array(dtype=float),
    axis: int,
    use_compression: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        updated_fill[i, j, k] = 0.0
        return
    lower = wp.vec3i(i, j, k)
    upper = wp.vec3i(i, j, k)
    upper[axis] += 1
    candidate = fill[i, j, k] + face_flux[lower[0], lower[1], lower[2]] - face_flux[
        upper[0], upper[1], upper[2]
    ]
    if use_compression != 0:
        candidate += float(compression[i, j, k]) * (
            face_courant[upper[0], upper[1], upper[2]]
            - face_courant[lower[0], lower[1], lower[2]]
        )
    bounded = wp.clamp(candidate, 0.0, 1.0)
    correction = wp.abs(candidate - bounded)
    wp.atomic_max(maximum_correction, 0, correction)
    if correction > 2.0e-6 or not wp.isfinite(candidate):
        wp.atomic_add(invalid_count, 0, 1)
    updated_fill[i, j, k] = bounded


@wp.kernel
def mass_flux_kernel(
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    volume_flux: wp.array3d(dtype=float),
    mass_flux: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    flux = volume_flux[i, j, k]
    mass_flux[i, j, k] = 0.0
    if flux == 0.0:
        return
    face = i if axis == 0 else (j if axis == 1 else k)
    donor = wp.vec3i(i, j, k)
    donor[axis] = face if flux < 0.0 else face - 1
    size = nx if axis == 0 else (ny if axis == 1 else nz)
    if periodic != 0:
        donor[axis] = (donor[axis] + size) % size
    if donor[axis] < 0 or donor[axis] >= size:
        wp.atomic_add(invalid_count, 0, 1)
        return
    donor_fill = fill[donor[0], donor[1], donor[2]]
    donor_mass = mass[donor[0], donor[1], donor[2]]
    if donor_fill <= 0.0 or donor_mass <= 0.0 or not wp.isfinite(donor_mass):
        wp.atomic_add(invalid_count, 0, 1)
        return
    value = flux * donor_mass / donor_fill
    if not wp.isfinite(value):
        wp.atomic_add(invalid_count, 0, 1)
        return
    mass_flux[i, j, k] = value


@wp.kernel
def update_mass_kernel(
    mass: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    face_mass_flux: wp.array3d(dtype=float),
    updated_mass: wp.array3d(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    axis: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        updated_mass[i, j, k] = 0.0
        return
    upper = wp.vec3i(i, j, k)
    upper[axis] += 1
    value = mass[i, j, k] + face_mass_flux[i, j, k] - face_mass_flux[
        upper[0], upper[1], upper[2]
    ]
    if not wp.isfinite(value) or value < -2.0e-6:
        wp.atomic_add(invalid_count, 0, 1)
    updated_mass[i, j, k] = wp.max(value, 0.0)


@wp.kernel
def momentum_flux_kernel(
    momentum: wp.array3d(dtype=wp.vec3),
    mass: wp.array3d(dtype=float),
    face_mass_flux: wp.array3d(dtype=float),
    face_momentum_flux: wp.array3d(dtype=wp.vec3),
    invalid_count: wp.array(dtype=wp.int32),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    flux = face_mass_flux[i, j, k]
    face_momentum_flux[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    if flux == 0.0:
        return
    face = i if axis == 0 else (j if axis == 1 else k)
    donor = wp.vec3i(i, j, k)
    receiver = wp.vec3i(i, j, k)
    upstream = wp.vec3i(i, j, k)
    donor[axis] = face if flux < 0.0 else face - 1
    receiver[axis] = face - 1 if flux < 0.0 else face
    upstream[axis] = donor[axis] + 1 if flux < 0.0 else donor[axis] - 1
    size = nx if axis == 0 else (ny if axis == 1 else nz)
    if periodic != 0:
        donor[axis] = (donor[axis] + size) % size
        receiver[axis] = (receiver[axis] + size) % size
        upstream[axis] = (upstream[axis] + size) % size
    if donor[axis] < 0 or donor[axis] >= size:
        wp.atomic_add(invalid_count, 0, 1)
        return
    donor_mass = mass[donor[0], donor[1], donor[2]]
    if donor_mass <= 0.0 or not wp.isfinite(donor_mass):
        wp.atomic_add(invalid_count, 0, 1)
        return
    velocity = momentum[donor[0], donor[1], donor[2]] / donor_mass
    receiver_valid = receiver[axis] >= 0 and receiver[axis] < size
    upstream_valid = upstream[axis] >= 0 and upstream[axis] < size
    if receiver_valid and upstream_valid:
        receiver_mass = mass[receiver[0], receiver[1], receiver[2]]
        upstream_mass = mass[upstream[0], upstream[1], upstream[2]]
        if receiver_mass > 1.0e-6 and upstream_mass > 1.0e-6:
            receiver_velocity = momentum[
                receiver[0], receiver[1], receiver[2]
            ] / receiver_mass
            upstream_velocity = momentum[
                upstream[0], upstream[1], upstream[2]
            ] / upstream_mass
            velocity += 0.5 * wp.vec3(
                _minmod(
                    velocity[0] - upstream_velocity[0],
                    receiver_velocity[0] - velocity[0],
                ),
                _minmod(
                    velocity[1] - upstream_velocity[1],
                    receiver_velocity[1] - velocity[1],
                ),
                _minmod(
                    velocity[2] - upstream_velocity[2],
                    receiver_velocity[2] - velocity[2],
                ),
            )
    value = flux * velocity
    if not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
        wp.atomic_add(invalid_count, 0, 1)
        return
    face_momentum_flux[i, j, k] = value


@wp.kernel
def update_momentum_kernel(
    momentum: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    face_momentum_flux: wp.array3d(dtype=wp.vec3),
    updated_momentum: wp.array3d(dtype=wp.vec3),
    invalid_count: wp.array(dtype=wp.int32),
    axis: int,
):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        updated_momentum[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        return
    upper = wp.vec3i(i, j, k)
    upper[axis] += 1
    value = momentum[i, j, k] + face_momentum_flux[i, j, k] - face_momentum_flux[
        upper[0], upper[1], upper[2]
    ]
    if not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
        wp.atomic_add(invalid_count, 0, 1)
        value = wp.vec3(0.0, 0.0, 0.0)
    updated_momentum[i, j, k] = value
