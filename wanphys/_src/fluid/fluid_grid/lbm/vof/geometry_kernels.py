# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Authoritative P6 normal, PLIC and mean-curvature kernels."""

from __future__ import annotations

import warp as wp


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
def _positive_cube(value: float) -> float:
    result = float(0.0)
    if value > 0.0:
        result = value * value * value
    return result


@wp.func
def plic_cube_volume_from_alpha(
    alpha: float,
    m_x: float,
    m_y: float,
    m_z: float,
) -> float:
    """Volume below ``m dot x <= alpha`` for x in [0,1]^3 and sum(m)=1."""

    eps = float(1.0e-4)
    active = int(0)
    if m_x > eps:
        active += 1
    if m_y > eps:
        active += 1
    if m_z > eps:
        active += 1

    volume = float(0.0)
    if alpha <= 0.0:
        volume = 0.0
    elif alpha >= 1.0:
        volume = 1.0
    elif active == 1:
        volume = alpha
    elif active == 2:
        a = float(0.0)
        b = float(0.0)
        if m_x > eps:
            if a == 0.0:
                a = m_x
            else:
                b = m_x
        if m_y > eps:
            if a == 0.0:
                a = m_y
            else:
                b = m_y
        if m_z > eps:
            if a == 0.0:
                a = m_z
            else:
                b = m_z
        if a > b:
            temporary = a
            a = b
            b = temporary
        if alpha < a:
            volume = alpha * alpha / (2.0 * a * b)
        elif alpha <= b:
            volume = (alpha - 0.5 * a) / b
        else:
            complement = 1.0 - alpha
            volume = 1.0 - complement * complement / (2.0 * a * b)
    elif active == 3:
        numerator = (
            _positive_cube(alpha)
            - _positive_cube(alpha - m_x)
            - _positive_cube(alpha - m_y)
            - _positive_cube(alpha - m_z)
            + _positive_cube(alpha - m_x - m_y)
            + _positive_cube(alpha - m_x - m_z)
            + _positive_cube(alpha - m_y - m_z)
            - _positive_cube(alpha - 1.0)
        )
        volume = numerator / (6.0 * m_x * m_y * m_z)
    return wp.clamp(volume, 0.0, 1.0)


@wp.func
def plic_cube_offset(fill: float, normal: wp.vec3) -> float:
    """Invert unit-cube volume for the centered liquid plane ``n dot r <= d``."""

    ax = wp.abs(normal[0])
    ay = wp.abs(normal[1])
    az = wp.abs(normal[2])
    if ax < 1.0e-4:
        ax = 0.0
    if ay < 1.0e-4:
        ay = 0.0
    if az < 1.0e-4:
        az = 0.0
    l1 = ax + ay + az
    result = float(0.0)
    if l1 > 1.0e-12:
        m_x = ax / l1
        m_y = ay / l1
        m_z = az / l1
        lower = float(0.0)
        upper = float(1.0)
        target = wp.clamp(fill, 0.0, 1.0)
        for _iteration in range(30):
            middle = 0.5 * (lower + upper)
            volume = plic_cube_volume_from_alpha(middle, m_x, m_y, m_z)
            if volume < target:
                lower = middle
            else:
                upper = middle
        alpha = 0.5 * (lower + upper)
        result = (alpha - 0.5) * l1
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
    "plic_cube_volume_from_alpha",
]
