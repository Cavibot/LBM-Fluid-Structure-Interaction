# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable interface-normal estimation for VOF fill fractions."""

from __future__ import annotations

import warp as wp

from .state import DebugMockScToVofState


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


class InterfaceGeometry:
    """Host-side interface geometry operator reusable by future VOF modes."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
    ) -> None:
        self.shape = shape
        self.device = device
        self.periodic = periodic

    def compute_normal(
        self,
        debug_mock: DebugMockScToVofState,
        solid_phi: wp.array3d,
    ) -> None:
        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            parker_youngs_normal_kernel,
            dim=self.shape,
            inputs=[
                debug_mock.phi,
                debug_mock.cell_type,
                solid_phi,
                debug_mock.normal,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        debug_mock.normal_valid_epoch = debug_mock.epoch


__all__ = ["InterfaceGeometry", "parker_youngs_normal_kernel"]
