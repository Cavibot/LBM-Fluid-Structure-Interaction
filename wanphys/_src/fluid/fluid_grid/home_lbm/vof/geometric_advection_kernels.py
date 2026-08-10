# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for conservative geometric PLIC volume transport."""

from __future__ import annotations

import warp as wp


GAS = wp.constant(0)
INTERFACE = wp.constant(1)
LIQUID = wp.constant(2)
SOLID = wp.constant(3)


@wp.kernel
def initialize_planar_mass_weighted_hydrostatic_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    fill_level: wp.array3d(dtype=float),
    acceleration: wp.vec3,
    gas_density: float,
    interface_fill: float,
    interface_axis: int,
    interface_index: int,
    gas_direction: int,
    ny: int,
    nz: int,
    stride: int,
):
    """Initialize the exact planar fixed point for force density ``M*a``."""

    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    coordinate = i
    normal_acceleration = acceleration[0] * float(gas_direction)
    if interface_axis == 1:
        coordinate = j
        normal_acceleration = acceleration[1] * float(gas_direction)
    elif interface_axis == 2:
        coordinate = k
        normal_acceleration = acceleration[2] * float(gas_direction)

    signed_layer = gas_direction * (coordinate - interface_index)
    rho = gas_density
    if signed_layer <= 0:
        interface_denominator = 1.0 + 1.5 * interface_fill * normal_acceleration
        rho = gas_density / interface_denominator
        depth = -signed_layer
        if depth > 0:
            interface_step = (
                (1.0 - 1.5 * interface_fill * normal_acceleration)
                / (1.0 + 1.5 * normal_acceleration)
            )
            full_step = (
                (1.0 - 1.5 * normal_acceleration)
                / (1.0 + 1.5 * normal_acceleration)
            )
            rho *= interface_step * wp.pow(full_step, float(depth - 1))

    fill = fill_level[i, j, k]
    moments[cell] = rho
    moments[stride + cell] = 0.5 * rho * fill * acceleration[0]
    moments[2 * stride + cell] = 0.5 * rho * fill * acceleration[1]
    moments[3 * stride + cell] = 0.5 * rho * fill * acceleration[2]
    for component in range(4, 10):
        moments[component * stride + cell] = 0.0
    mass[i, j, k] = rho * fill


@wp.kernel
def build_liquid_body_force_kernel(
    mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    acceleration: wp.vec3,
    force_density: wp.array3d(dtype=wp.vec3),
    invalid_count: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    cell_mass = mass[i, j, k]
    active = flag == INTERFACE or flag == LIQUID
    if flag < GAS or flag > SOLID or not wp.isfinite(cell_mass) or cell_mass < 0.0:
        wp.atomic_add(invalid_count, 0, 1)
        force_density[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    elif active:
        force_density[i, j, k] = cell_mass * acceleration
    else:
        force_density[i, j, k] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _positive_power2(value: float) -> float:
    positive = wp.max(value, 0.0)
    return positive * positive


@wp.func
def _positive_power3(value: float) -> float:
    positive = wp.max(value, 0.0)
    return positive * positive * positive


@wp.func
def _minmod(left: float, right: float) -> float:
    if left * right <= 0.0:
        return 0.0
    return wp.sign(left) * wp.min(wp.abs(left), wp.abs(right))


@wp.func
def _plane_volume_fraction(offset: float, normal: wp.vec3) -> float:
    """Exact centered-cube half-space volume with near-axis reduction."""

    cx = wp.abs(normal[0])
    cy = wp.abs(normal[1])
    cz = wp.abs(normal[2])
    support = 0.5 * (cx + cy + cz)
    if offset <= -support:
        return 0.0
    if offset >= support:
        return 1.0
    maximum = wp.max(cx, wp.max(cy, cz))
    threshold = 1.0e-4 * maximum
    a = float(0.0)
    b = float(0.0)
    c = float(0.0)
    dimension = int(0)
    if cx > threshold:
        a = cx
        dimension = 1
    if cy > threshold:
        if dimension == 0:
            a = cy
        else:
            b = cy
        dimension += 1
    if cz > threshold:
        if dimension == 0:
            a = cz
        elif dimension == 1:
            b = cz
        else:
            c = cz
        dimension += 1

    complement = offset > 0.0
    evaluation_offset = offset
    if complement:
        evaluation_offset = -offset
    volume = float(0.0)
    if dimension == 1:
        shifted = evaluation_offset + 0.5 * a
        volume = wp.max(shifted, 0.0) / a
    elif dimension == 2:
        shifted = evaluation_offset + 0.5 * (a + b)
        numerator = (
            _positive_power2(shifted)
            - _positive_power2(shifted - a)
            - _positive_power2(shifted - b)
            + _positive_power2(shifted - a - b)
        )
        volume = numerator / (2.0 * a * b)
    elif dimension == 3:
        shifted = evaluation_offset + 0.5 * (a + b + c)
        numerator = (
            _positive_power3(shifted)
            - _positive_power3(shifted - a)
            - _positive_power3(shifted - b)
            - _positive_power3(shifted - c)
            + _positive_power3(shifted - a - b)
            + _positive_power3(shifted - a - c)
            + _positive_power3(shifted - b - c)
            - _positive_power3(shifted - a - b - c)
        )
        volume = numerator / (6.0 * a * b * c)
    volume = wp.clamp(volume, 0.0, 1.0)
    if complement:
        volume = 1.0 - volume
    return volume


@wp.func
def _positive_power2_f64(value: wp.float64) -> wp.float64:
    positive = wp.max(value, wp.float64(0.0))
    return positive * positive


@wp.func
def _positive_power3_f64(value: wp.float64) -> wp.float64:
    positive = wp.max(value, wp.float64(0.0))
    return positive * positive * positive


@wp.func
def _plane_volume_fraction_f64(
    offset: wp.float64,
    cx: wp.float64,
    cy: wp.float64,
    cz: wp.float64,
) -> wp.float64:
    """Endpoint-stable counterpart of ``_plane_volume_fraction``."""

    support = wp.float64(0.5) * (cx + cy + cz)
    if offset <= -support:
        return wp.float64(0.0)
    if offset >= support:
        return wp.float64(1.0)
    maximum = wp.max(cx, wp.max(cy, cz))
    threshold = wp.float64(1.0e-4) * maximum
    a = wp.float64(0.0)
    b = wp.float64(0.0)
    c = wp.float64(0.0)
    dimension = int(0)
    if cx > threshold:
        a = cx
        dimension = 1
    if cy > threshold:
        if dimension == 0:
            a = cy
        else:
            b = cy
        dimension += 1
    if cz > threshold:
        if dimension == 0:
            a = cz
        elif dimension == 1:
            b = cz
        else:
            c = cz
        dimension += 1

    complement = offset > wp.float64(0.0)
    evaluation_offset = offset
    if complement:
        evaluation_offset = -offset
    volume = wp.float64(0.0)
    if dimension == 1:
        shifted = evaluation_offset + wp.float64(0.5) * a
        volume = wp.max(shifted, wp.float64(0.0)) / a
    elif dimension == 2:
        shifted = evaluation_offset + wp.float64(0.5) * (a + b)
        numerator = (
            _positive_power2_f64(shifted)
            - _positive_power2_f64(shifted - a)
            - _positive_power2_f64(shifted - b)
            + _positive_power2_f64(shifted - a - b)
        )
        volume = numerator / (wp.float64(2.0) * a * b)
    elif dimension == 3:
        shifted = evaluation_offset + wp.float64(0.5) * (a + b + c)
        numerator = (
            _positive_power3_f64(shifted)
            - _positive_power3_f64(shifted - a)
            - _positive_power3_f64(shifted - b)
            - _positive_power3_f64(shifted - c)
            + _positive_power3_f64(shifted - a - b)
            + _positive_power3_f64(shifted - a - c)
            + _positive_power3_f64(shifted - b - c)
            - _positive_power3_f64(shifted - a - b - c)
        )
        volume = numerator / (wp.float64(6.0) * a * b * c)
    volume = wp.clamp(volume, wp.float64(0.0), wp.float64(1.0))
    if complement:
        volume = wp.float64(1.0) - volume
    return volume


@wp.func
def _swept_slab_volume_unbounded(
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
    length = wp.length(normal)
    if (
        not wp.isfinite(fill)
        or fill < 0.0
        or fill > 1.0
        or not wp.isfinite(length)
        or length <= 0.0
        or not wp.isfinite(offset)
    ):
        return -1.0
    maximum = wp.max(
        wp.abs(normal[0]), wp.max(wp.abs(normal[1]), wp.abs(normal[2]))
    )
    threshold = 1.0e-4 * maximum
    reduced_normal = wp.vec3(
        normal[0] if wp.abs(normal[0]) > threshold else 0.0,
        normal[1] if wp.abs(normal[1]) > threshold else 0.0,
        normal[2] if wp.abs(normal[2]) > threshold else 0.0,
    )
    reduced_length = wp.length(reduced_normal)
    if not wp.isfinite(reduced_length) or reduced_length <= 0.0:
        return -1.0
    n = reduced_normal / reduced_length
    face_sign = float(-1.0)
    if signed_courant > 0.0:
        face_sign = 1.0
    slab_center = face_sign * (0.5 - 0.5 * width)
    scale_x = float(1.0)
    scale_y = float(1.0)
    scale_z = float(1.0)
    if axis == 0:
        scale_x = width
    elif axis == 1:
        scale_y = width
    else:
        scale_z = width
    transformed = wp.vec3(n[0] * scale_x, n[1] * scale_y, n[2] * scale_z)
    transformed_offset = offset / reduced_length - n[axis] * slab_center
    fast_volume = width * _plane_volume_fraction(
        transformed_offset, transformed
    )
    lower_bound = wp.max(0.0, width - (1.0 - fill))
    upper_bound = wp.min(width, fill)
    needs_stable_evaluation = (
        wp.min(fill, 1.0 - fill) <= 1.0e-3
        or fast_volume < lower_bound - 1.0e-6
        or fast_volume > upper_bound + 1.0e-6
    )
    if needs_stable_evaluation:
        endpoint_volume = _plane_volume_fraction_f64(
            wp.float64(transformed_offset),
            wp.float64(wp.abs(transformed[0])),
            wp.float64(wp.abs(transformed[1])),
            wp.float64(wp.abs(transformed[2])),
        )
        return width * wp.float32(endpoint_volume)
    return fast_volume


@wp.kernel
def plic_axis_face_flux_kernel(
    fill: wp.array3d(dtype=float),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    face_courant: wp.array3d(dtype=float),
    face_flux: wp.array3d(dtype=float),
    face_error: wp.array3d(dtype=wp.int32),
    raw_face_volume: wp.array3d(dtype=float),
    invalid_face_count: wp.array(dtype=wp.int32),
    bounded_face_count: wp.array(dtype=wp.int32),
    maximum_bound_correction: wp.array(dtype=float),
    bound_correction_tolerance: float,
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    face_flux[i, j, k] = 0.0
    face_error[i, j, k] = 0
    raw_face_volume[i, j, k] = 0.0
    face = i
    size = nx
    if axis == 1:
        face = j
        size = ny
    elif axis == 2:
        face = k
        size = nz
    courant = face_courant[i, j, k]
    if not wp.isfinite(courant) or wp.abs(courant) > 1.0:
        face_error[i, j, k] = 1
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    if face == 0 or face == size:
        if periodic == 0:
            if courant != 0.0:
                face_error[i, j, k] = 2
                wp.atomic_add(invalid_face_count, 0, 1)
            return
        paired_i = i
        paired_j = j
        paired_k = k
        if axis == 0:
            paired_i = size - face
        elif axis == 1:
            paired_j = size - face
        else:
            paired_k = size - face
        if courant != face_courant[paired_i, paired_j, paired_k]:
            face_error[i, j, k] = 4
            wp.atomic_add(invalid_face_count, 0, 1)
            return
    if courant == 0.0:
        return

    donor_i = i
    donor_j = j
    donor_k = k
    if axis == 0:
        donor_i = face
        if courant > 0.0:
            donor_i = face - 1
        if periodic != 0:
            donor_i = (donor_i + nx) % nx
    elif axis == 1:
        donor_j = face
        if courant > 0.0:
            donor_j = face - 1
        if periodic != 0:
            donor_j = (donor_j + ny) % ny
    else:
        donor_k = face
        if courant > 0.0:
            donor_k = face - 1
        if periodic != 0:
            donor_k = (donor_k + nz) % nz
    if (
        donor_i < 0
        or donor_i >= nx
        or donor_j < 0
        or donor_j >= ny
        or donor_k < 0
        or donor_k >= nz
    ):
        face_error[i, j, k] = 8
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    raw_volume = _swept_slab_volume_unbounded(
        fill[donor_i, donor_j, donor_k],
        normal[donor_i, donor_j, donor_k],
        plane_offset[donor_i, donor_j, donor_k],
        axis,
        courant,
    )
    raw_face_volume[i, j, k] = raw_volume
    if not wp.isfinite(raw_volume) or raw_volume < 0.0:
        face_error[i, j, k] = 16
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    width = wp.abs(courant)
    donor_fill = fill[donor_i, donor_j, donor_k]
    lower_bound = wp.max(0.0, width - (1.0 - donor_fill))
    upper_bound = wp.min(width, donor_fill)
    volume = wp.clamp(raw_volume, lower_bound, upper_bound)
    correction = wp.abs(volume - raw_volume)
    if correction > 0.0:
        wp.atomic_add(bounded_face_count, 0, 1)
        wp.atomic_max(maximum_bound_correction, 0, correction)
    if correction > bound_correction_tolerance:
        face_error[i, j, k] = 32
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    face_flux[i, j, k] = wp.sign(courant) * volume


@wp.kernel
def apply_plic_axis_flux_kernel(
    fill: wp.array3d(dtype=float),
    face_flux: wp.array3d(dtype=float),
    face_courant: wp.array3d(dtype=float),
    compression: wp.array3d(dtype=wp.int32),
    updated_fill: wp.array3d(dtype=float),
    invalid_cell_count: wp.array(dtype=wp.int32),
    axis: int,
    use_compression: int,
    bound_tolerance: float,
):
    i, j, k = wp.tid()
    upper_i = i
    upper_j = j
    upper_k = k
    if axis == 0:
        upper_i += 1
    elif axis == 1:
        upper_j += 1
    else:
        upper_k += 1
    lower_flux = face_flux[i, j, k]
    upper_flux = face_flux[upper_i, upper_j, upper_k]
    lower_courant = face_courant[i, j, k]
    upper_courant = face_courant[upper_i, upper_j, upper_k]
    cell_fill = fill[i, j, k]
    exact_full_cell = (
        use_compression != 0
        and compression[i, j, k] == 1
        and cell_fill == 1.0
        and lower_flux == lower_courant
        and upper_flux == upper_courant
    )
    value = cell_fill + lower_flux - upper_flux
    if use_compression != 0:
        value += float(compression[i, j, k]) * (upper_courant - lower_courant)
    if exact_full_cell:
        value = 1.0
    if (
        not wp.isfinite(value)
        or value < -bound_tolerance
        or value > 1.0 + bound_tolerance
    ):
        wp.atomic_add(invalid_cell_count, 0, 1)
        updated_fill[i, j, k] = value
        return
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    updated_fill[i, j, k] = value


@wp.kernel
def plic_axis_mass_flux_kernel(
    mass: wp.array3d(dtype=float),
    fill: wp.array3d(dtype=float),
    face_volume_flux: wp.array3d(dtype=float),
    face_mass_flux: wp.array3d(dtype=float),
    invalid_face_count: wp.array(dtype=wp.int32),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    """Convert a shared PLIC volume flux to a conservative liquid-mass flux."""

    i, j, k = wp.tid()
    face_mass_flux[i, j, k] = 0.0
    volume_flux = face_volume_flux[i, j, k]
    if volume_flux == 0.0:
        return
    face = i
    size = nx
    if axis == 1:
        face = j
        size = ny
    elif axis == 2:
        face = k
        size = nz
    donor_i = i
    donor_j = j
    donor_k = k
    if axis == 0:
        donor_i = face if volume_flux < 0.0 else face - 1
        if periodic != 0:
            donor_i = (donor_i + nx) % nx
    elif axis == 1:
        donor_j = face if volume_flux < 0.0 else face - 1
        if periodic != 0:
            donor_j = (donor_j + ny) % ny
    else:
        donor_k = face if volume_flux < 0.0 else face - 1
        if periodic != 0:
            donor_k = (donor_k + nz) % nz
    if (
        donor_i < 0
        or donor_i >= nx
        or donor_j < 0
        or donor_j >= ny
        or donor_k < 0
        or donor_k >= nz
    ):
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    donor_fill = fill[donor_i, donor_j, donor_k]
    donor_mass = mass[donor_i, donor_j, donor_k]
    if (
        not wp.isfinite(donor_fill)
        or not wp.isfinite(donor_mass)
        or donor_fill <= 0.0
        or donor_mass <= 0.0
    ):
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    value = volume_flux * donor_mass / donor_fill
    if not wp.isfinite(value):
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    face_mass_flux[i, j, k] = value


@wp.kernel
def apply_plic_axis_mass_flux_kernel(
    mass: wp.array3d(dtype=float),
    face_mass_flux: wp.array3d(dtype=float),
    updated_mass: wp.array3d(dtype=float),
    invalid_cell_count: wp.array(dtype=wp.int32),
    axis: int,
    negative_tolerance: float,
):
    i, j, k = wp.tid()
    upper_i = i
    upper_j = j
    upper_k = k
    if axis == 0:
        upper_i += 1
    elif axis == 1:
        upper_j += 1
    else:
        upper_k += 1
    value = mass[i, j, k] + face_mass_flux[i, j, k] - face_mass_flux[
        upper_i, upper_j, upper_k
    ]
    if not wp.isfinite(value) or value < -negative_tolerance:
        wp.atomic_add(invalid_cell_count, 0, 1)
        updated_mass[i, j, k] = value
        return
    updated_mass[i, j, k] = wp.max(value, 0.0)


@wp.kernel
def initialize_liquid_momentum_kernel(
    moments: wp.array(dtype=float),
    mass: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    momentum: wp.array3d(dtype=wp.vec3),
    invalid_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    cell_mass = mass[i, j, k]
    flag = flags[i, j, k]
    if flag == GAS or flag == SOLID:
        momentum[i, j, k] = wp.vec3(0.0)
        return
    rho = moments[cell]
    if (
        (flag != INTERFACE and flag != LIQUID)
        or not wp.isfinite(cell_mass)
        or cell_mass < 0.0
        or not wp.isfinite(rho)
        or rho <= 0.0
    ):
        momentum[i, j, k] = wp.vec3(0.0)
        wp.atomic_add(invalid_count, 0, 1)
        return
    value = wp.vec3(
        cell_mass * moments[stride + cell] / rho,
        cell_mass * moments[2 * stride + cell] / rho,
        cell_mass * moments[3 * stride + cell] / rho,
    )
    if not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
        momentum[i, j, k] = wp.vec3(0.0)
        wp.atomic_add(invalid_count, 0, 1)
        return
    momentum[i, j, k] = value


@wp.kernel
def plic_axis_momentum_flux_kernel(
    momentum: wp.array3d(dtype=wp.vec3),
    mass: wp.array3d(dtype=float),
    face_mass_flux: wp.array3d(dtype=float),
    face_momentum_flux: wp.array3d(dtype=wp.vec3),
    invalid_face_count: wp.array(dtype=wp.int32),
    axis: int,
    periodic: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    face_momentum_flux[i, j, k] = wp.vec3(0.0)
    mass_flux = face_mass_flux[i, j, k]
    if mass_flux == 0.0:
        return
    face = i
    if axis == 1:
        face = j
    elif axis == 2:
        face = k
    donor_i = i
    donor_j = j
    donor_k = k
    receiver_i = i
    receiver_j = j
    receiver_k = k
    upstream_i = i
    upstream_j = j
    upstream_k = k
    if axis == 0:
        donor_i = face if mass_flux < 0.0 else face - 1
        receiver_i = face - 1 if mass_flux < 0.0 else face
        upstream_i = donor_i + 1 if mass_flux < 0.0 else donor_i - 1
        if periodic != 0:
            donor_i = (donor_i + nx) % nx
            receiver_i = (receiver_i + nx) % nx
            upstream_i = (upstream_i + nx) % nx
    elif axis == 1:
        donor_j = face if mass_flux < 0.0 else face - 1
        receiver_j = face - 1 if mass_flux < 0.0 else face
        upstream_j = donor_j + 1 if mass_flux < 0.0 else donor_j - 1
        if periodic != 0:
            donor_j = (donor_j + ny) % ny
            receiver_j = (receiver_j + ny) % ny
            upstream_j = (upstream_j + ny) % ny
    else:
        donor_k = face if mass_flux < 0.0 else face - 1
        receiver_k = face - 1 if mass_flux < 0.0 else face
        upstream_k = donor_k + 1 if mass_flux < 0.0 else donor_k - 1
        if periodic != 0:
            donor_k = (donor_k + nz) % nz
            receiver_k = (receiver_k + nz) % nz
            upstream_k = (upstream_k + nz) % nz
    if (
        donor_i < 0
        or donor_i >= nx
        or donor_j < 0
        or donor_j >= ny
        or donor_k < 0
        or donor_k >= nz
    ):
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    donor_mass = mass[donor_i, donor_j, donor_k]
    donor_momentum = momentum[donor_i, donor_j, donor_k]
    if (
        not wp.isfinite(donor_mass)
        or donor_mass <= 0.0
        or not wp.isfinite(donor_momentum[0])
        or not wp.isfinite(donor_momentum[1])
        or not wp.isfinite(donor_momentum[2])
    ):
        wp.atomic_add(invalid_face_count, 0, 1)
        return
    face_velocity = donor_momentum / donor_mass
    receiver_valid = (
        receiver_i >= 0
        and receiver_i < nx
        and receiver_j >= 0
        and receiver_j < ny
        and receiver_k >= 0
        and receiver_k < nz
    )
    upstream_valid = (
        upstream_i >= 0
        and upstream_i < nx
        and upstream_j >= 0
        and upstream_j < ny
        and upstream_k >= 0
        and upstream_k < nz
    )
    if receiver_valid and upstream_valid:
        receiver_mass = mass[receiver_i, receiver_j, receiver_k]
        receiver_momentum = momentum[receiver_i, receiver_j, receiver_k]
        upstream_mass = mass[upstream_i, upstream_j, upstream_k]
        upstream_momentum = momentum[upstream_i, upstream_j, upstream_k]
        if (
            wp.isfinite(receiver_mass)
            and receiver_mass > 1.0e-6
            and wp.isfinite(upstream_mass)
            and upstream_mass > 1.0e-6
            and wp.isfinite(receiver_momentum[0])
            and wp.isfinite(receiver_momentum[1])
            and wp.isfinite(receiver_momentum[2])
            and wp.isfinite(upstream_momentum[0])
            and wp.isfinite(upstream_momentum[1])
            and wp.isfinite(upstream_momentum[2])
        ):
            receiver_velocity = receiver_momentum / receiver_mass
            upstream_velocity = upstream_momentum / upstream_mass
            slope = wp.vec3(
                _minmod(
                    face_velocity[0] - upstream_velocity[0],
                    receiver_velocity[0] - face_velocity[0],
                ),
                _minmod(
                    face_velocity[1] - upstream_velocity[1],
                    receiver_velocity[1] - face_velocity[1],
                ),
                _minmod(
                    face_velocity[2] - upstream_velocity[2],
                    receiver_velocity[2] - face_velocity[2],
                ),
            )
            face_velocity += 0.5 * slope
    face_momentum_flux[i, j, k] = mass_flux * face_velocity


@wp.kernel
def apply_plic_axis_momentum_flux_kernel(
    momentum: wp.array3d(dtype=wp.vec3),
    face_momentum_flux: wp.array3d(dtype=wp.vec3),
    updated_momentum: wp.array3d(dtype=wp.vec3),
    invalid_cell_count: wp.array(dtype=wp.int32),
    axis: int,
):
    i, j, k = wp.tid()
    upper_i = i
    upper_j = j
    upper_k = k
    if axis == 0:
        upper_i += 1
    elif axis == 1:
        upper_j += 1
    else:
        upper_k += 1
    value = (
        momentum[i, j, k]
        + face_momentum_flux[i, j, k]
        - face_momentum_flux[upper_i, upper_j, upper_k]
    )
    if not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
        wp.atomic_add(invalid_cell_count, 0, 1)
        updated_momentum[i, j, k] = wp.vec3(0.0)
        return
    updated_momentum[i, j, k] = value


@wp.kernel
def reconcile_transported_momentum_kernel(
    transported_momentum: wp.array3d(dtype=wp.vec3),
    transported_mass: wp.array3d(dtype=float),
    committed_mass: wp.array3d(dtype=float),
    committed_flags: wp.array3d(dtype=wp.int32),
    moments: wp.array(dtype=float),
    committed_momentum: wp.array3d(dtype=wp.vec3),
    invalid_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = committed_flags[i, j, k]
    if flag == GAS or flag == SOLID:
        committed_momentum[i, j, k] = wp.vec3(0.0)
        return
    source_mass = transported_mass[i, j, k]
    target_mass = committed_mass[i, j, k]
    value = transported_momentum[i, j, k]
    if (
        (flag != INTERFACE and flag != LIQUID)
        or not wp.isfinite(source_mass)
        or not wp.isfinite(target_mass)
        or target_mass <= 0.0
        or not wp.isfinite(value[0])
        or not wp.isfinite(value[1])
        or not wp.isfinite(value[2])
    ):
        committed_momentum[i, j, k] = wp.vec3(0.0)
        wp.atomic_add(invalid_count, 0, 1)
        return
    if source_mass > 0.0:
        committed_momentum[i, j, k] = value * (target_mass / source_mass)
        return
    rho = moments[cell]
    if not wp.isfinite(rho) or rho <= 0.0:
        committed_momentum[i, j, k] = wp.vec3(0.0)
        wp.atomic_add(invalid_count, 0, 1)
        return
    committed_momentum[i, j, k] = wp.vec3(
        target_mass * moments[stride + cell] / rho,
        target_mass * moments[2 * stride + cell] / rho,
        target_mass * moments[3 * stride + cell] / rho,
    )


@wp.kernel
def apply_geometric_momentum_transport_kernel(
    source_moments: wp.array(dtype=float),
    source_mass: wp.array3d(dtype=float),
    candidate_moments: wp.array(dtype=float),
    transported_mass: wp.array3d(dtype=float),
    transported_fill: wp.array3d(dtype=float),
    transported_momentum: wp.array3d(dtype=wp.vec3),
    transported_flags: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = transported_flags[i, j, k]
    if flag != INTERFACE and flag != LIQUID:
        return
    cell = i * ny * nz + j * nz + k
    old_rho = source_moments[cell]
    new_rho = candidate_moments[cell]
    old_mass = source_mass[i, j, k]
    new_mass = transported_mass[i, j, k]
    valid = (
        wp.isfinite(old_rho)
        and old_rho > 0.0
        and wp.isfinite(new_rho)
        and new_rho > 0.0
        and wp.isfinite(old_mass)
        and old_mass >= 0.0
        and wp.isfinite(new_mass)
        and new_mass > 1.0e-12
    )
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
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
    target_momentum = (
        transported_momentum[i, j, k]
        + new_mass * (candidate_velocity - old_velocity)
    )
    transported_velocity = target_momentum / new_mass
    liquid_fraction = float(1.0)
    if flag == INTERFACE:
        liquid_fraction = wp.clamp(transported_fill[i, j, k], 0.0, 1.0)
    # HOME collision owns the gas-side hydrodynamic closure while geometric
    # transport owns represented liquid momentum. Blend those contributions
    # continuously by liquid volume fraction; the step-wide projection below
    # conservatively returns the resulting represented-momentum difference.
    target_velocity = candidate_velocity + liquid_fraction * (
        transported_velocity - candidate_velocity
    )
    delta = target_velocity - candidate_velocity
    valid = (
        wp.isfinite(delta[0])
        and wp.isfinite(delta[1])
        and wp.isfinite(delta[2])
    )
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
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
    for component in range(1, 10):
        valid = valid and wp.isfinite(candidate_moments[component * stride + cell])
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)


@wp.kernel
def accumulate_expected_liquid_momentum_kernel(
    source_moments: wp.array(dtype=float),
    source_mass: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    home_candidate_moments: wp.array(dtype=float),
    ledger: wp.array(dtype=wp.float64),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = source_flags[i, j, k]
    if flag != INTERFACE and flag != LIQUID:
        return
    cell = i * ny * nz + j * nz + k
    rho = source_moments[cell]
    cell_mass = source_mass[i, j, k]
    if rho <= 0.0 or cell_mass < 0.0:
        return
    for component in range(3):
        source_j = source_moments[(component + 1) * stride + cell]
        candidate_j = home_candidate_moments[(component + 1) * stride + cell]
        wp.atomic_add(
            ledger,
            component,
            wp.float64(cell_mass * source_j / rho),
        )
        wp.atomic_add(
            ledger,
            component + 3,
            wp.float64(candidate_j - source_j),
        )


@wp.kernel
def accumulate_committed_liquid_momentum_kernel(
    candidate_moments: wp.array(dtype=float),
    transported_mass: wp.array3d(dtype=float),
    transported_flags: wp.array3d(dtype=wp.int32),
    ledger: wp.array(dtype=wp.float64),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = transported_flags[i, j, k]
    if flag != INTERFACE and flag != LIQUID:
        return
    cell = i * ny * nz + j * nz + k
    rho = candidate_moments[cell]
    cell_mass = transported_mass[i, j, k]
    if rho <= 0.0 or cell_mass <= 0.0:
        return
    for component in range(3):
        wp.atomic_add(
            ledger,
            component + 6,
            wp.float64(
                cell_mass
                * candidate_moments[(component + 1) * stride + cell]
                / rho
            ),
        )
    wp.atomic_add(ledger, 9, wp.float64(cell_mass))


@wp.kernel
def apply_uniform_liquid_velocity_correction_kernel(
    candidate_moments: wp.array(dtype=float),
    transported_flags: wp.array3d(dtype=wp.int32),
    velocity_correction: wp.vec3,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = transported_flags[i, j, k]
    if flag != INTERFACE and flag != LIQUID:
        return
    cell = i * ny * nz + j * nz + k
    rho = candidate_moments[cell]
    if rho <= 0.0:
        return
    dx = velocity_correction[0]
    dy = velocity_correction[1]
    dz = velocity_correction[2]
    jx = candidate_moments[stride + cell]
    jy = candidate_moments[2 * stride + cell]
    jz = candidate_moments[3 * stride + cell]
    candidate_moments[4 * stride + cell] += 2.0 * dx * jx + rho * dx * dx
    candidate_moments[5 * stride + cell] += 2.0 * dy * jy + rho * dy * dy
    candidate_moments[6 * stride + cell] += 2.0 * dz * jz + rho * dz * dz
    candidate_moments[7 * stride + cell] += dx * jy + dy * jx + rho * dx * dy
    candidate_moments[8 * stride + cell] += dx * jz + dz * jx + rho * dx * dz
    candidate_moments[9 * stride + cell] += dy * jz + dz * jy + rho * dy * dz
    candidate_moments[stride + cell] = jx + rho * dx
    candidate_moments[2 * stride + cell] = jy + rho * dy
    candidate_moments[3 * stride + cell] = jz + rho * dz


@wp.kernel
def prepare_uniform_liquid_velocity_correction_kernel(
    ledger: wp.array(dtype=wp.float64),
    velocity_correction: wp.array(dtype=wp.vec3),
    maximum_lattice_speed: float,
    invalid_count: wp.array(dtype=wp.int32),
):
    represented_mass = ledger[9]
    valid = wp.isfinite(represented_mass) and represented_mass > 0.0
    correction = wp.vec3(0.0, 0.0, 0.0)
    if valid:
        correction = wp.vec3(
            float((ledger[0] + ledger[3] - ledger[6]) / represented_mass),
            float((ledger[1] + ledger[4] - ledger[7]) / represented_mass),
            float((ledger[2] + ledger[5] - ledger[8]) / represented_mass),
        )
        valid = (
            wp.isfinite(correction[0])
            and wp.isfinite(correction[1])
            and wp.isfinite(correction[2])
            and wp.length(correction) <= maximum_lattice_speed
        )
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        correction = wp.vec3(0.0, 0.0, 0.0)
    velocity_correction[0] = correction


@wp.kernel
def apply_device_liquid_velocity_correction_kernel(
    candidate_moments: wp.array(dtype=float),
    transported_flags: wp.array3d(dtype=wp.int32),
    velocity_correction: wp.array(dtype=wp.vec3),
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = transported_flags[i, j, k]
    if flag != INTERFACE and flag != LIQUID:
        return
    cell = i * ny * nz + j * nz + k
    rho = candidate_moments[cell]
    if rho <= 0.0:
        return
    delta = velocity_correction[0]
    dx = delta[0]
    dy = delta[1]
    dz = delta[2]
    jx = candidate_moments[stride + cell]
    jy = candidate_moments[2 * stride + cell]
    jz = candidate_moments[3 * stride + cell]
    candidate_moments[4 * stride + cell] += 2.0 * dx * jx + rho * dx * dx
    candidate_moments[5 * stride + cell] += 2.0 * dy * jy + rho * dy * dy
    candidate_moments[6 * stride + cell] += 2.0 * dz * jz + rho * dz * dz
    candidate_moments[7 * stride + cell] += dx * jy + dy * jx + rho * dx * dy
    candidate_moments[8 * stride + cell] += dx * jz + dz * jx + rho * dx * dz
    candidate_moments[9 * stride + cell] += dy * jz + dz * jy + rho * dy * dz
    candidate_moments[stride + cell] = jx + rho * dx
    candidate_moments[2 * stride + cell] = jy + rho * dy
    candidate_moments[3 * stride + cell] = jz + rho * dz


@wp.kernel
def classify_weymouth_yue_compression_kernel(
    fill: wp.array3d(dtype=float),
    compression: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()
    compression[i, j, k] = int(fill[i, j, k] >= 0.5)


@wp.kernel
def classify_only_missing_active_kernel(
    flags: wp.array3d(dtype=wp.int32),
    active: wp.array3d(dtype=wp.int32),
    invalid_count: wp.array(dtype=wp.int32),
):
    """Classify the official only-missing fluid ownership (LIQUID or INTERFACE)."""

    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if flag < GAS or flag > SOLID:
        active[i, j, k] = 0
        wp.atomic_add(invalid_count, 0, 1)
        return
    active[i, j, k] = int(flag == INTERFACE or flag == LIQUID)


@wp.kernel
def commit_fixed_topology_plic_state_kernel(
    moments: wp.array(dtype=float),
    updated_fill: wp.array3d(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    destination_mass: wp.array3d(dtype=float),
    destination_fill: wp.array3d(dtype=float),
    destination_excess_mass: wp.array3d(dtype=float),
    destination_excess_momentum: wp.array(dtype=float),
    destination_flags: wp.array3d(dtype=wp.int32),
    invalid_cell_count: wp.array(dtype=wp.int32),
    endpoint_tolerance: float,
    ny: int,
    nz: int,
    stride: int,
):
    """Commit a geometric sweep when no VOF category transition is allowed."""

    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    flag = source_flags[i, j, k]
    fill = updated_fill[i, j, k]
    rho = moments[cell]
    valid = wp.isfinite(fill) and fill >= 0.0 and fill <= 1.0
    mass = float(0.0)
    if flag == LIQUID:
        valid = valid and wp.abs(fill - 1.0) <= endpoint_tolerance
        valid = valid and wp.isfinite(rho) and rho > 0.0
        fill = 1.0
        mass = rho
    elif flag == INTERFACE:
        valid = valid and wp.isfinite(rho) and rho > 0.0
        mass = rho * fill
    elif flag == GAS or flag == SOLID:
        valid = valid and wp.abs(fill) <= endpoint_tolerance
        fill = 0.0
        mass = 0.0
    else:
        valid = False
    if not valid or not wp.isfinite(mass):
        wp.atomic_add(invalid_cell_count, 0, 1)
        return
    destination_mass[i, j, k] = mass
    destination_fill[i, j, k] = fill
    destination_excess_mass[i, j, k] = 0.0
    destination_flags[i, j, k] = flag
    for component in range(3):
        destination_excess_momentum[component * stride + cell] = 0.0
