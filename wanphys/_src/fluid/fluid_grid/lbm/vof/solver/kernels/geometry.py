# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Authoritative P6 normal, PLIC and mean-curvature kernels."""

from __future__ import annotations

import warp as wp


@wp.func
def _map_axis(index: int, size: int, periodic: int) -> int:
    mapped = index
    if periodic != 0:
        if mapped < 0:
            mapped += size
        elif mapped >= size:
            mapped -= size
    else:
        mapped = wp.clamp(mapped, 0, size - 1)
    return mapped


@wp.func
def _sample_fill_fraction(
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    center_i: int,
    center_j: int,
    center_k: int,
    offset_i: int,
    offset_j: int,
    offset_k: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    i = _map_axis(center_i + offset_i, nx, periodic_x)
    j = _map_axis(center_j + offset_j, ny, periodic_y)
    k = _map_axis(center_k + offset_k, nz, periodic_z)
    if solid_phi[i, j, k] < 0.0:
        return phi[center_i, center_j, center_k]
    return phi[i, j, k]


@wp.kernel
def parker_youngs_normal_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.uint8),
    solid_phi: wp.array3d(dtype=float),
    normal: wp.array3d(dtype=wp.vec3),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Compute liquid-to-gas normals using a 3-D weighted Sobel stencil."""

    i, j, k = wp.tid()
    if cell_type[i, j, k] != wp.uint8(1) or solid_phi[i, j, k] < 0.0:
        normal[i, j, k] = wp.vec3(0.0)
        return

    gradient_x = float(0.0)
    gradient_y = float(0.0)
    gradient_z = float(0.0)

    for a in range(-1, 2):
        weight_a = 2.0 if a == 0 else 1.0
        for b in range(-1, 2):
            weight_b = 2.0 if b == 0 else 1.0
            weight = weight_a * weight_b
            gradient_x += weight * (
                _sample_fill_fraction(
                    phi, solid_phi, i, j, k, 1, a, b,
                    periodic_x, periodic_y, periodic_z, nx, ny, nz,
                )
                - _sample_fill_fraction(
                    phi, solid_phi, i, j, k, -1, a, b,
                    periodic_x, periodic_y, periodic_z, nx, ny, nz,
                )
            )
            gradient_y += weight * (
                _sample_fill_fraction(
                    phi, solid_phi, i, j, k, a, 1, b,
                    periodic_x, periodic_y, periodic_z, nx, ny, nz,
                )
                - _sample_fill_fraction(
                    phi, solid_phi, i, j, k, a, -1, b,
                    periodic_x, periodic_y, periodic_z, nx, ny, nz,
                )
            )
            gradient_z += weight * (
                _sample_fill_fraction(
                    phi, solid_phi, i, j, k, a, b, 1,
                    periodic_x, periodic_y, periodic_z, nx, ny, nz,
                )
                - _sample_fill_fraction(
                    phi, solid_phi, i, j, k, a, b, -1,
                    periodic_x, periodic_y, periodic_z, nx, ny, nz,
                )
            )

    liquid_to_gas = wp.vec3(-gradient_x, -gradient_y, -gradient_z)
    magnitude = wp.length(liquid_to_gas)
    if magnitude > 1.0e-12:
        normal[i, j, k] = liquid_to_gas / magnitude
    else:
        normal[i, j, k] = wp.vec3(0.0)


@wp.func
def _geometry_map_axis(index: int, size: int, periodic: int) -> int:
    mapped = index
    if periodic != 0:
        if mapped < 0:
            mapped += size
        elif mapped >= size:
            mapped -= size
    else:
        mapped = wp.clamp(mapped, 0, size - 1)
    return mapped


@wp.func
def _geometry_sample_phi(
    phi: wp.array3d(dtype=float),
    center_i: int,
    center_j: int,
    center_k: int,
    offset_i: int,
    offset_j: int,
    offset_k: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    i = _geometry_map_axis(center_i + offset_i, nx, periodic_x)
    j = _geometry_map_axis(center_j + offset_j, ny, periodic_y)
    k = _geometry_map_axis(center_k + offset_k, nz, periodic_z)
    return phi[i, j, k]


@wp.func
def parker_youngs_normal_at(
    phi: wp.array3d(dtype=float),
    center_i: int,
    center_j: int,
    center_k: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    """Return a liquid-to-gas Parker–Youngs normal at one lattice cell."""

    gradient_x = float(0.0)
    gradient_y = float(0.0)
    gradient_z = float(0.0)
    for a in range(-1, 2):
        weight_a = 2.0 if a == 0 else 1.0
        for b in range(-1, 2):
            weight_b = 2.0 if b == 0 else 1.0
            weight = weight_a * weight_b
            gradient_x += weight * (
                _geometry_sample_phi(
                    phi,
                    center_i,
                    center_j,
                    center_k,
                    1,
                    a,
                    b,
                    periodic_x,
                    periodic_y,
                    periodic_z,
                    nx,
                    ny,
                    nz,
                )
                - _geometry_sample_phi(
                    phi,
                    center_i,
                    center_j,
                    center_k,
                    -1,
                    a,
                    b,
                    periodic_x,
                    periodic_y,
                    periodic_z,
                    nx,
                    ny,
                    nz,
                )
            )
            gradient_y += weight * (
                _geometry_sample_phi(
                    phi,
                    center_i,
                    center_j,
                    center_k,
                    a,
                    1,
                    b,
                    periodic_x,
                    periodic_y,
                    periodic_z,
                    nx,
                    ny,
                    nz,
                )
                - _geometry_sample_phi(
                    phi,
                    center_i,
                    center_j,
                    center_k,
                    a,
                    -1,
                    b,
                    periodic_x,
                    periodic_y,
                    periodic_z,
                    nx,
                    ny,
                    nz,
                )
            )
            gradient_z += weight * (
                _geometry_sample_phi(
                    phi,
                    center_i,
                    center_j,
                    center_k,
                    a,
                    b,
                    1,
                    periodic_x,
                    periodic_y,
                    periodic_z,
                    nx,
                    ny,
                    nz,
                )
                - _geometry_sample_phi(
                    phi,
                    center_i,
                    center_j,
                    center_k,
                    a,
                    b,
                    -1,
                    periodic_x,
                    periodic_y,
                    periodic_z,
                    nx,
                    ny,
                    nz,
                )
            )

    liquid_to_gas = wp.vec3(-gradient_x, -gradient_y, -gradient_z)
    magnitude = wp.length(liquid_to_gas)
    result = wp.vec3(0.0)
    if magnitude > 1.0e-12:
        result = liquid_to_gas / magnitude
    return result


@wp.func
def _plic_cube_offset_reduced(
    volume: wp.float64,
    n1: wp.float64,
    n2: wp.float64,
    n3: wp.float64,
) -> wp.float64:
    """Analytical symmetry-reduced PLIC inverse in cancellation-safe precision."""

    zero = wp.float64(0.0)
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    two = wp.float64(2.0)
    three = wp.float64(3.0)
    six = wp.float64(6.0)
    eight = wp.float64(8.0)
    result = zero
    n12 = n1 + n2
    n3_volume = n3 * volume
    if n12 <= two * n3_volume:
        result = n3_volume + half * n12
    else:
        square_n1 = n1 * n1
        six_n2 = six * n2
        v1 = square_n1 / six_n2
        if (
            v1 <= n3_volume
            and n3_volume < v1 + half * (n2 - n1)
        ):
            result = half * (
                n1
                + wp.sqrt(
                    square_n1
                    + eight * n2 * (n3_volume - v1)
                )
            )
        else:
            volume6 = n1 * six_n2 * n3_volume
            if n3_volume < v1:
                result = wp.pow(volume6, one / three)
            else:
                v3 = half * n12
                if n3 < n12:
                    v3 = (
                        n3 * n3 * (three * n12 - n3)
                        + square_n1 * (n1 - three * n3)
                        + n2 * n2 * (n2 - three * n3)
                    ) / (n1 * six_n2)
                square_n12 = square_n1 + n2 * n2
                volume6_minus_cubes = (
                    volume6
                    - n1 * n1 * n1
                    - n2 * n2 * n2
                )
                case_three = n3_volume < v3
                a = volume6_minus_cubes
                b = square_n12
                c = n12
                if not case_three:
                    a = half * (
                        volume6_minus_cubes - n3 * n3 * n3
                    )
                    b = half * (square_n12 + n3 * n3)
                    c = half
                t = wp.sqrt(wp.max(c * c - b, zero))
                argument = (
                    c * c * c - half * a - (one + half) * b * c
                ) / (t * t * t)
                result = c - two * t * wp.sin(
                    wp.asin(
                        wp.clamp(
                            argument,
                            -one,
                            one,
                        )
                    )
                    / three
                )
    return result


@wp.func
def plic_cube_offset(fill: float, normal: wp.vec3) -> float:
    """Invert unit-cube volume for the centered liquid plane ``n dot r <= d``."""

    epsilon = float(1.0e-4)
    ax = wp.abs(normal[0])
    ay = wp.abs(normal[1])
    az = wp.abs(normal[2])
    if ax < epsilon:
        ax = 0.0
    if ay < epsilon:
        ay = 0.0
    if az < epsilon:
        az = 0.0
    l1 = ax + ay + az
    if l1 > 1.0e-12:
        if ax / l1 <= epsilon:
            ax = 0.0
        if ay / l1 <= epsilon:
            ay = 0.0
        if az / l1 <= epsilon:
            az = 0.0
        l1 = ax + ay + az

    result = float(0.0)
    if l1 > 1.0e-12:
        m_x = ax / l1
        m_y = ay / l1
        m_z = az / l1
        n1 = wp.min(wp.min(m_x, m_y), m_z)
        n3 = wp.max(wp.max(m_x, m_y), m_z)
        n2 = wp.max(1.0 - n1 - n3, 0.0)
        target = wp.clamp(fill, 0.0, 1.0)
        reduced_volume = 0.5 - wp.abs(target - 0.5)
        reduced_offset = _plic_cube_offset_reduced(
            wp.float64(reduced_volume),
            wp.float64(n1),
            wp.float64(n2),
            wp.float64(n3),
        )
        result = l1 * wp.float32(wp.float64(0.5) - reduced_offset)
        if target < 0.5:
            result = -result
        elif target == 0.5:
            result = 0.0
    return result


@wp.kernel
def authoritative_normal_plic_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.uint8),
    normal: wp.array3d(dtype=wp.vec3),
    plic_offset: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    i, j, k = wp.tid()
    if cell_type[i, j, k] != wp.uint8(1):
        normal[i, j, k] = wp.vec3(0.0)
        plic_offset[i, j, k] = 0.0
        return
    local_normal = parker_youngs_normal_at(
        phi,
        i,
        j,
        k,
        periodic_x,
        periodic_y,
        periodic_z,
        nx,
        ny,
        nz,
    )
    normal[i, j, k] = local_normal
    plic_offset[i, j, k] = plic_cube_offset(phi[i, j, k], local_normal)


@wp.kernel
def authoritative_curvature_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.uint8),
    curvature: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Compute mean curvature ``-0.5 div(n)`` from PLIC normal samples."""

    i, j, k = wp.tid()
    if cell_type[i, j, k] != wp.uint8(1):
        curvature[i, j, k] = 0.0
        return
    center_normal = parker_youngs_normal_at(
        phi, i, j, k, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    if wp.length(center_normal) <= 1.0e-12:
        curvature[i, j, k] = 0.0
        return

    n_x_plus = parker_youngs_normal_at(
        phi, i + 1, j, k, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    n_x_minus = parker_youngs_normal_at(
        phi, i - 1, j, k, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    n_y_plus = parker_youngs_normal_at(
        phi, i, j + 1, k, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    n_y_minus = parker_youngs_normal_at(
        phi, i, j - 1, k, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    n_z_plus = parker_youngs_normal_at(
        phi, i, j, k + 1, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    n_z_minus = parker_youngs_normal_at(
        phi, i, j, k - 1, periodic_x, periodic_y, periodic_z, nx, ny, nz
    )
    divergence = 0.5 * (
        n_x_plus[0]
        - n_x_minus[0]
        + n_y_plus[1]
        - n_y_minus[1]
        + n_z_plus[2]
        - n_z_minus[2]
    )
    curvature[i, j, k] = wp.clamp(-0.5 * divergence, -1.0, 1.0)


__all__ = [
    "authoritative_curvature_kernel",
    "authoritative_normal_plic_kernel",
    "parker_youngs_normal_at",
    "plic_cube_offset",
]
