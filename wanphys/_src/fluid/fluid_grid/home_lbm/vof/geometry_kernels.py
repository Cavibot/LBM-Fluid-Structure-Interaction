# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for PLIC normal and plane-offset construction."""

from __future__ import annotations

import warp as wp

from .geometric_advection_kernels import _plane_volume_fraction

Vec5f = wp.types.vector(5, wp.float32)
Mat55f = wp.types.matrix((5, 5), wp.float32)


@wp.func
def _plic_reduced(fill: float, n1: float, n2: float, n3: float) -> float:
    n12 = n1 + n2
    n3_fill = n3 * fill
    if n12 <= 2.0 * n3_fill:
        return n3_fill + 0.5 * n12
    n1_squared = n1 * n1
    n2_times_six = 6.0 * n2
    v1 = n1_squared / n2_times_six
    if v1 <= n3_fill and n3_fill < v1 + 0.5 * (n2 - n1):
        return 0.5 * (
            n1 + wp.sqrt(n1_squared + 8.0 * n2 * (n3_fill - v1))
        )
    volume_six = n1 * n2_times_six * n3_fill
    if n3_fill < v1:
        return wp.pow(volume_six, 1.0 / 3.0)
    if n3 < n12:
        v3 = (
            n3 * n3 * (3.0 * n12 - n3)
            + n1_squared * (n1 - 3.0 * n3)
            + n2 * n2 * (n2 - 3.0 * n3)
        ) / (n1 * n2_times_six)
    else:
        v3 = 0.5 * n12
    volume_term = volume_six - n1 * n1_squared - n2 * n2 * n2
    case_three = n3_fill < v3
    if case_three:
        a = volume_term
        b = n1_squared + n2 * n2
        c = n12
    else:
        a = 0.5 * (volume_term - n3 * n3 * n3)
        b = 0.5 * (n1_squared + n2 * n2 + n3 * n3)
        c = 0.5
    t = wp.sqrt(wp.max(c * c - b, 0.0))
    argument = (c * c * c - 0.5 * a - 1.5 * b * c) / (t * t * t)
    argument = wp.clamp(argument, -1.0, 1.0)
    return c - 2.0 * t * wp.sin((1.0 / 3.0) * wp.asin(argument))


@wp.func
def _plic_offset(fill: float, normal: wp.vec3) -> float:
    ax = wp.abs(normal[0])
    ay = wp.abs(normal[1])
    az = wp.abs(normal[2])
    maximum = wp.max(wp.max(ax, ay), az)
    threshold = 1.0e-4 * maximum
    if ax <= threshold:
        ax = 0.0
    if ay <= threshold:
        ay = 0.0
    if az <= threshold:
        az = 0.0
    scale = ax + ay + az
    reduced_fill = 0.5 - wp.abs(fill - 0.5)
    if reduced_fill <= 1.0e-3:
        bound = 0.5 * scale
        if fill <= 0.0:
            return -bound
        if fill >= 1.0:
            return bound
        lower = -bound
        upper = bound
        reduced_normal = wp.vec3(ax, ay, az)
        for _iteration in range(24):
            midpoint = 0.5 * (lower + upper)
            if _plane_volume_fraction(midpoint, reduced_normal) < fill:
                lower = midpoint
            else:
                upper = midpoint
        return 0.5 * (lower + upper)
    n1 = wp.min(wp.min(ax, ay), az) / scale
    n3 = wp.max(wp.max(ax, ay), az) / scale
    n2 = wp.max(1.0 - n1 - n3, 0.0)
    distance = _plic_reduced(reduced_fill, n1, n2, n3)
    sign = float(-1.0)
    if fill >= 0.5:
        sign = 1.0
    return scale * sign * (0.5 - distance)


@wp.func
def _positive_part(value: float) -> float:
    return wp.max(value, 0.0)


@wp.func
def _plic_section_area(offset: float, normal: wp.vec3) -> float:
    """Exact unit-cube plane-section area from the clipped-volume derivative."""

    ax = wp.abs(normal[0])
    ay = wp.abs(normal[1])
    az = wp.abs(normal[2])
    threshold = 1.0e-4 * wp.max(wp.max(ax, ay), az)
    a = float(0.0)
    b = float(0.0)
    c = float(0.0)
    dimension = int(0)
    if ax > threshold:
        a = ax
        dimension = 1
    if ay > threshold:
        if dimension == 0:
            a = ay
        else:
            b = ay
        dimension += 1
    if az > threshold:
        if dimension == 0:
            a = az
        elif dimension == 1:
            b = az
        else:
            c = az
        dimension += 1

    normal_length = wp.length(normal)
    if dimension == 1:
        return normal_length / a
    if dimension == 2:
        small = wp.min(a, b)
        large = wp.max(a, b)
        total = small + large
        shifted = wp.min(
            offset + 0.5 * total,
            0.5 * total - offset,
        )
        if shifted >= small:
            return normal_length / large
        return normal_length * wp.max(shifted, 0.0) / (small * large)

    small = wp.min(wp.min(a, b), c)
    large = wp.max(wp.max(a, b), c)
    middle = a + b + c - small - large
    total = small + middle + large
    shifted = wp.min(
        offset + 0.5 * total,
        0.5 * total - offset,
    )
    if large >= small + middle and shifted >= small + middle:
        return normal_length / large
    derivative = (
        wp.pow(_positive_part(shifted), 2.0)
        - wp.pow(_positive_part(shifted - small), 2.0)
        - wp.pow(_positive_part(shifted - middle), 2.0)
        - wp.pow(_positive_part(shifted - large), 2.0)
        + wp.pow(_positive_part(shifted - small - middle), 2.0)
        + wp.pow(_positive_part(shifted - small - large), 2.0)
        + wp.pow(_positive_part(shifted - middle - large), 2.0)
        - wp.pow(_positive_part(shifted - total), 2.0)
    ) / (2.0 * small * middle * large)
    return wp.max(normal_length * derivative, 0.0)


@wp.kernel
def plic_offsets_kernel(
    fill_level: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    plane_offset: wp.array(dtype=float),
):
    index = wp.tid()
    plane_offset[index] = _plic_offset(fill_level[index], normal[index])


@wp.kernel
def plic_areas_kernel(
    plane_offset: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    interface_area: wp.array(dtype=float),
):
    index = wp.tid()
    interface_area[index] = _plic_section_area(plane_offset[index], normal[index])


@wp.kernel
def plic_pull_link_intersections_kernel(
    plane_offset: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    population_direction: wp.array(dtype=wp.vec3),
    fraction: wp.array(dtype=float),
    point: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=float),
    status: wp.array(dtype=wp.int32),
    tolerance: float,
):
    """Batch PLIC/pull-link intersections with explicit rejection status."""

    index = wp.tid()
    fraction[index] = 0.0
    point[index] = wp.vec3(0.0)
    distance[index] = 0.0
    status[index] = 0
    offset = plane_offset[index]
    n = normal[index]
    c = population_direction[index]
    normal_length = wp.length(n)
    direction_length = wp.length(c)
    if (
        not wp.isfinite(offset)
        or not wp.isfinite(normal_length)
        or not wp.isfinite(direction_length)
        or normal_length <= 0.0
        or direction_length <= 0.0
        or not wp.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        status[index] = -1
        return
    denominator = -wp.dot(n, c)
    if denominator <= tolerance * normal_length * direction_length:
        status[index] = -2
        return
    delta = offset / denominator
    if delta < -tolerance:
        status[index] = -3
        return
    if delta < 0.0:
        delta = 0.0
    intersection = -delta * c
    if (
        wp.max(
            wp.max(wp.abs(intersection[0]), wp.abs(intersection[1])),
            wp.abs(intersection[2]),
        )
        > 0.5 + tolerance
    ):
        status[index] = -4
        return
    residual = wp.abs(wp.dot(n, intersection) - offset)
    residual_scale = wp.max(wp.max(1.0, wp.abs(offset)), normal_length)
    if residual > tolerance * residual_scale:
        status[index] = -5
        return
    fraction[index] = delta
    point[index] = intersection
    distance[index] = delta * direction_length
    status[index] = 1


@wp.kernel
def classify_fsl_hydrodynamic_nodes_kernel(
    flags: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    geometry_valid: wp.array3d(dtype=wp.int32),
    active: wp.array3d(dtype=wp.int32),
    invalid_counts: wp.array(dtype=wp.int32),
    tolerance: float,
):
    """Separate VOF cell categories from FSL hydrodynamic ownership."""

    i, j, k = wp.tid()
    active[i, j, k] = 0
    flag = flags[i, j, k]
    if flag < 0 or flag > 3:
        wp.atomic_add(invalid_counts, 0, 1)
        return
    if flag == 2:
        active[i, j, k] = 1
        return
    if flag != 1:
        return

    n = normal[i, j, k]
    offset = plane_offset[i, j, k]
    normal_length = wp.length(n)
    if (
        geometry_valid[i, j, k] == 0
        or not wp.isfinite(offset)
        or not wp.isfinite(normal_length)
        or normal_length <= 0.0
    ):
        wp.atomic_add(invalid_counts, 1, 1)
        return
    center_tolerance = tolerance * wp.max(normal_length, 1.0)
    if offset >= -center_tolerance:
        active[i, j, k] = 1


@wp.kernel
def construct_fsl_link_coverage_kernel(
    flags: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    geometry_valid: wp.array3d(dtype=wp.int32),
    active: wp.array3d(dtype=wp.int32),
    population_direction: wp.array(dtype=wp.vec3),
    fraction: wp.array(dtype=float),
    extrapolation: wp.array(dtype=float),
    status: wp.array(dtype=wp.int32),
    plane_owner: wp.array(dtype=wp.int32),
    tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    """Construct owner-aware FSL boundary links from classified fluid nodes."""

    link_index = wp.tid()
    q = link_index % 27
    cell_index = link_index // 27
    k = cell_index % nz
    cell_index = cell_index // nz
    j = cell_index % ny
    i = cell_index // ny
    fraction[link_index] = 0.0
    extrapolation[link_index] = 0.0
    status[link_index] = 0
    plane_owner[link_index] = 0
    if q == 0 or active[i, j, k] == 0:
        return

    c = population_direction[q]
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
        return

    source_flag = flags[source_i, source_j, source_k]
    if active[source_i, source_j, source_k] != 0 or source_flag == 3:
        return

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
    has_support = int(0)
    if support_valid != 0 and active[support_i, support_j, support_k] != 0:
        has_support = 1

    plane_i = i
    plane_j = j
    plane_k = k
    owner_center = wp.vec3(0.0)
    owner = int(0)
    if source_flag == 1:
        plane_i = source_i
        plane_j = source_j
        plane_k = source_k
        owner_center = -c
        owner = 2
    elif source_flag == 0 and flags[i, j, k] == 1:
        owner = 1
    else:
        status[link_index] = -6
        return
    plane_owner[link_index] = owner

    n = normal[plane_i, plane_j, plane_k]
    offset = plane_offset[plane_i, plane_j, plane_k]
    normal_length = wp.length(n)
    direction_length = wp.length(c)
    if (
        geometry_valid[plane_i, plane_j, plane_k] == 0
        or not wp.isfinite(offset)
        or not wp.isfinite(normal_length)
        or normal_length <= 0.0
        or direction_length <= 0.0
    ):
        status[link_index] = -6
        return

    target_offset = offset + wp.dot(n, owner_center)
    denominator = -wp.dot(n, c)
    if denominator <= tolerance * normal_length * direction_length:
        status[link_index] = -2
        return
    delta = target_offset / denominator
    if delta < -tolerance:
        status[link_index] = -3
        return
    if delta < 0.0:
        delta = 0.0
    if delta > 1.0 + tolerance:
        status[link_index] = -4
        return

    intersection = -delta * c
    owner_local_point = intersection - owner_center
    owner_extent = wp.max(
        wp.max(wp.abs(owner_local_point[0]), wp.abs(owner_local_point[1])),
        wp.abs(owner_local_point[2]),
    )
    residual = wp.abs(wp.dot(n, owner_local_point) - offset)
    residual_scale = wp.max(wp.max(1.0, wp.abs(offset)), normal_length)
    if residual > tolerance * residual_scale:
        status[link_index] = -5
        return

    fraction[link_index] = delta
    extrapolation[link_index] = wp.max(owner_extent - 0.5, 0.0)
    if has_support == 0:
        status[link_index] = -7
    elif owner_extent > 0.5 + tolerance:
        status[link_index] = 2
    else:
        status[link_index] = 1


@wp.kernel
def construct_plic_geometry_kernel(
    fill_level: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    interface_area: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    wetting_enabled: int,
    contact_angle_cos: float,
    contact_angle_sin: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    normal[i, j, k] = wp.vec3(0.0)
    plane_offset[i, j, k] = 0.0
    interface_area[i, j, k] = 0.0
    valid[i, j, k] = 0
    if flags[i, j, k] != 1:
        return
    wp.atomic_add(counts, 3, 1)

    gradient = wp.vec3(0.0)
    wall_gradient = wp.vec3(0.0)
    fill_stencil_valid = int(1)
    has_solid_neighbor = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                transverse_i = float(1)
                transverse_j = float(1)
                transverse_k = float(1)
                if di == 0:
                    transverse_i = 2.0
                if dj == 0:
                    transverse_j = 2.0
                if dk == 0:
                    transverse_k = 2.0
                neighbor_i = i + di
                neighbor_j = j + dj
                neighbor_k = k + dk
                domain_wall = int(0)
                if neighbor_i < 0 or neighbor_i >= nx:
                    if periodic_x != 0:
                        neighbor_i = (neighbor_i + nx) % nx
                    else:
                        wall_gradient[0] -= (
                            float(di) * transverse_j * transverse_k
                        )
                        domain_wall = 1
                if neighbor_j < 0 or neighbor_j >= ny:
                    if periodic_y != 0:
                        neighbor_j = (neighbor_j + ny) % ny
                    else:
                        wall_gradient[1] -= (
                            float(dj) * transverse_i * transverse_k
                        )
                        domain_wall = 1
                if neighbor_k < 0 or neighbor_k >= nz:
                    if periodic_z != 0:
                        neighbor_k = (neighbor_k + nz) % nz
                    else:
                        wall_gradient[2] -= (
                            float(dk) * transverse_i * transverse_j
                        )
                        domain_wall = 1
                if domain_wall != 0:
                    has_solid_neighbor = 1
                else:
                    neighbor_fill = fill_level[neighbor_i, neighbor_j, neighbor_k]
                    neighbor_phi = solid_phi[neighbor_i, neighbor_j, neighbor_k]
                    if (
                        not wp.isfinite(neighbor_fill)
                        or neighbor_fill < 0.0
                        or neighbor_fill > 1.0
                        or not wp.isfinite(neighbor_phi)
                    ):
                        fill_stencil_valid = 0
                    else:
                        wall_gradient[0] += float(di) * transverse_j * transverse_k * neighbor_phi
                        wall_gradient[1] += float(dj) * transverse_i * transverse_k * neighbor_phi
                        wall_gradient[2] += float(dk) * transverse_i * transverse_j * neighbor_phi
                        if flags[neighbor_i, neighbor_j, neighbor_k] == 3:
                            has_solid_neighbor = 1
                        else:
                            gradient[0] += float(di) * transverse_j * transverse_k * neighbor_fill
                            gradient[1] += float(dj) * transverse_i * transverse_k * neighbor_fill
                            gradient[2] += float(dk) * transverse_i * transverse_j * neighbor_fill
    if fill_stencil_valid == 0:
        wp.atomic_add(counts, 0, 1)
        return

    if has_solid_neighbor != 0 and wetting_enabled == 0:
        wp.atomic_add(counts, 12, 1)
        return

    magnitude = wp.length(gradient)
    if not wp.isfinite(magnitude) or magnitude <= 1.0e-12:
        wp.atomic_add(counts, 1, 1)
        return
    outward = -gradient / magnitude
    if has_solid_neighbor != 0:
        wall_magnitude = wp.length(wall_gradient)
        if not wp.isfinite(wall_magnitude) or wall_magnitude <= 1.0e-12:
            wp.atomic_add(counts, 10, 1)
            return
        wall_normal = wall_gradient / wall_magnitude
        tangent = outward - wp.dot(outward, wall_normal) * wall_normal
        tangent_magnitude = wp.length(tangent)
        if not wp.isfinite(tangent_magnitude) or tangent_magnitude <= 1.0e-12:
            wp.atomic_add(counts, 11, 1)
            return
        outward = (
            contact_angle_cos * wall_normal
            + contact_angle_sin * tangent / tangent_magnitude
        )
        outward_magnitude = wp.length(outward)
        if not wp.isfinite(outward_magnitude) or outward_magnitude <= 1.0e-12:
            wp.atomic_add(counts, 11, 1)
            return
        outward /= outward_magnitude
        wp.atomic_add(counts, 9, 1)
    offset = _plic_offset(fill_level[i, j, k], outward)
    if not wp.isfinite(offset):
        wp.atomic_add(counts, 2, 1)
        return
    area = _plic_section_area(offset, outward)
    cell_fill = fill_level[i, j, k]
    fractional_fill = cell_fill > 0.0 and cell_fill < 1.0
    if not wp.isfinite(area) or (fractional_fill and area <= 0.0):
        wp.atomic_add(counts, 7, 1)
        return
    normal[i, j, k] = outward
    plane_offset[i, j, k] = offset
    interface_area[i, j, k] = area
    valid[i, j, k] = 1


@wp.kernel
def construct_plic_curvature_kernel(
    fill_level: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    plane_offset: wp.array3d(dtype=float),
    geometry_valid: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=float),
    curvature_valid: wp.array3d(dtype=wp.int32),
    curvature_required: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    max_condition_number: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    curvature[i, j, k] = 0.0
    curvature_valid[i, j, k] = 0
    curvature_required[i, j, k] = 0
    if flags[i, j, k] != 1 or geometry_valid[i, j, k] == 0:
        return

    n = normal[i, j, k]
    required = int(plane_offset[i, j, k] < 0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                neighbor_i = i + di
                neighbor_j = j + dj
                neighbor_k = k + dk
                if neighbor_i < 0 or neighbor_i >= nx:
                    if periodic_x != 0:
                        neighbor_i = (neighbor_i + nx) % nx
                    else:
                        continue
                if neighbor_j < 0 or neighbor_j >= ny:
                    if periodic_y != 0:
                        neighbor_j = (neighbor_j + ny) % ny
                    else:
                        continue
                if neighbor_k < 0 or neighbor_k >= nz:
                    if periodic_z != 0:
                        neighbor_k = (neighbor_k + nz) % nz
                    else:
                        continue
                if flags[neighbor_i, neighbor_j, neighbor_k] == 0:
                    required = 1
    if required == 0:
        return
    curvature_required[i, j, k] = 1
    wp.atomic_add(counts, 8, 1)
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

    # Keep the accepted 3x3x3 reconstruction in the interior. A contact cell
    # on a nonperiodic domain wall needs a one-sided 5x5x5 stencil to retain
    # five independent rows at a 90-degree contact line.
    wide_stencil = int(0)
    if periodic_x == 0 and (i == 0 or i == nx - 1):
        wide_stencil = 1
    if periodic_y == 0 and (j == 0 or j == ny - 1):
        wide_stencil = 1
    if periodic_z == 0 and (k == 0 or k == nz - 1):
        wide_stencil = 1
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-2, 3):
                if wide_stencil == 0 and (
                    wp.abs(di) > 1 or wp.abs(dj) > 1 or wp.abs(dk) > 1
                ):
                    continue
                if di == 0 and dj == 0 and dk == 0:
                    continue
                neighbor_i = i + di
                neighbor_j = j + dj
                neighbor_k = k + dk
                valid_neighbor = int(1)
                if neighbor_i < 0 or neighbor_i >= nx:
                    if periodic_x != 0:
                        neighbor_i = (neighbor_i + nx) % nx
                    else:
                        valid_neighbor = 0
                if neighbor_j < 0 or neighbor_j >= ny:
                    if periodic_y != 0:
                        neighbor_j = (neighbor_j + ny) % ny
                    else:
                        valid_neighbor = 0
                if neighbor_k < 0 or neighbor_k >= nz:
                    if periodic_z != 0:
                        neighbor_k = (neighbor_k + nz) % nz
                    else:
                        valid_neighbor = 0
                if valid_neighbor == 0:
                    continue
                if flags[neighbor_i, neighbor_j, neighbor_k] != 1:
                    continue
                displacement = wp.vec3(float(di), float(dj), float(dk))
                x = wp.dot(displacement, tangent_x)
                y = wp.dot(displacement, tangent_y)
                neighbor_offset = _plic_offset(
                    fill_level[neighbor_i, neighbor_j, neighbor_k], n
                )
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
        wp.atomic_add(counts, 4, 1)
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
        wp.atomic_add(counts, 5, 1)
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
    h = coefficients[3]
    slope_i = coefficients[4]
    denominator = wp.pow(1.0 + h * h + slope_i * slope_i, 1.5)
    value = (
        a * (1.0 + slope_i * slope_i)
        + b * (1.0 + h * h)
        - c * h * slope_i
    ) / denominator
    if not wp.isfinite(value):
        wp.atomic_add(counts, 6, 1)
        return
    curvature[i, j, k] = value
    curvature_valid[i, j, k] = 1


@wp.kernel
def compute_laplace_gas_density_kernel(
    flags: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=float),
    curvature_valid: wp.array3d(dtype=wp.int32),
    curvature_required: wp.array3d(dtype=wp.int32),
    ambient_gas_density: float,
    six_surface_tension: float,
    gas_density: wp.array3d(dtype=float),
    invalid_density_count: wp.array(dtype=wp.int32),
    min_gas_density: wp.array(dtype=float),
    max_gas_density: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    value = ambient_gas_density
    if flags[i, j, k] == 1 and curvature_required[i, j, k] != 0:
        if curvature_valid[i, j, k] == 0:
            wp.atomic_add(invalid_density_count, 0, 1)
            gas_density[i, j, k] = 0.0
            return
        value -= six_surface_tension * curvature[i, j, k]
        if not wp.isfinite(value) or value <= 0.0:
            wp.atomic_add(invalid_density_count, 0, 1)
            gas_density[i, j, k] = 0.0
            return
        wp.atomic_min(min_gas_density, 0, value)
        wp.atomic_max(max_gas_density, 0, value)
    gas_density[i, j, k] = value


@wp.kernel
def compute_plic_capillary_momentum_correction_kernel(
    flags: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    interface_area: wp.array3d(dtype=float),
    gas_density: wp.array3d(dtype=float),
    ambient_gas_density: float,
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    momentum_correction: wp.array(dtype=float),
    invalid_correction_count: wp.array(dtype=wp.int32),
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
    for component in range(3):
        momentum_correction[component * stride + cell] = 0.0
    if flags[i, j, k] != 1:
        return

    n = normal[i, j, k]
    normal_length = wp.length(n)
    area = interface_area[i, j, k]
    density_jump = gas_density[i, j, k] - ambient_gas_density
    valid_geometry = (
        wp.isfinite(normal_length)
        and normal_length > 0.0
        and wp.isfinite(area)
        and area >= 0.0
        and wp.isfinite(density_jump)
    )
    if not valid_geometry:
        wp.atomic_add(invalid_correction_count, 0, 1)
        return

    discrete = wp.vec3(0.0)
    for direction in range(1, 27):
        c = directions[direction]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        valid_source = int(1)
        if source_i < 0 or source_i >= nx:
            if periodic_x != 0:
                source_i = (source_i + nx) % nx
            else:
                valid_source = 0
        if source_j < 0 or source_j >= ny:
            if periodic_y != 0:
                source_j = (source_j + ny) % ny
            else:
                valid_source = 0
        if source_k < 0 or source_k >= nz:
            if periodic_z != 0:
                source_k = (source_k + nz) % nz
            else:
                valid_source = 0
        if valid_source != 0 and flags[source_i, source_j, source_k] == 0:
            discrete += 2.0 * weights[direction] * density_jump * c

    plic_traction = -(density_jump / 3.0) * area * n / normal_length
    correction = plic_traction - discrete
    if not (
        wp.isfinite(correction[0])
        and wp.isfinite(correction[1])
        and wp.isfinite(correction[2])
    ):
        wp.atomic_add(invalid_correction_count, 0, 1)
        return
    for component in range(3):
        momentum_correction[component * stride + cell] = correction[component]
