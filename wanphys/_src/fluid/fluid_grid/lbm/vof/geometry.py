# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Debug and authoritative interface geometry for VOF fill fractions."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from .geometry_kernels import (
    authoritative_curvature_kernel,
    authoritative_normal_plic_kernel,
)
from .state import DebugMockScToVofState, VofGridState


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


def _positive_power(value: float, power: int) -> float:
    return max(float(value), 0.0) ** power


def plic_cube_volume(
    normal: tuple[float, float, float] | np.ndarray,
    offset: float,
) -> float:
    """Return the liquid volume of ``n dot r <= offset`` in a centered cube."""

    vector = np.abs(np.asarray(normal, dtype=np.float64))
    vector[vector < 1.0e-4] = 0.0
    l1 = float(np.sum(vector))
    if not math.isfinite(l1) or l1 <= 1.0e-14:
        return 0.0
    weights = vector / l1
    alpha = float(offset) / l1 + 0.5
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    active = weights[weights > 1.0e-4]
    dimension = int(active.size)
    if dimension == 1:
        return alpha
    if dimension == 2:
        a, b = (float(value) for value in active)
        result = (
            alpha**2
            - _positive_power(alpha - a, 2)
            - _positive_power(alpha - b, 2)
        ) / (2.0 * a * b)
        return float(np.clip(result, 0.0, 1.0))
    mx, my, mz = (float(value) for value in weights)
    result = (
        _positive_power(alpha, 3)
        - _positive_power(alpha - mx, 3)
        - _positive_power(alpha - my, 3)
        - _positive_power(alpha - mz, 3)
        + _positive_power(alpha - mx - my, 3)
        + _positive_power(alpha - mx - mz, 3)
        + _positive_power(alpha - my - mz, 3)
        - _positive_power(alpha - 1.0, 3)
    ) / (6.0 * mx * my * mz)
    return float(np.clip(result, 0.0, 1.0))


def plic_plane_offset(
    fill: float,
    normal: tuple[float, float, float] | np.ndarray,
    *,
    iterations: int = 60,
) -> float:
    """Host oracle for the centered unit-cube PLIC offset."""

    target = float(fill)
    vector = np.asarray(normal, dtype=np.float64)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("PLIC fill must be finite and lie in [0, 1]")
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("PLIC normal must contain three finite components")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1.0e-14:
        return 0.0
    vector = vector / magnitude
    l1 = float(np.sum(np.abs(vector)))
    lower = -0.5 * l1
    upper = 0.5 * l1
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        if plic_cube_volume(vector, middle) < target:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def validate_authoritative_geometry(
    state: VofGridState,
    *,
    volume_tolerance: float = 2.0e-5,
) -> None:
    """Validate geometry finiteness, epochs and PLIC volume closure."""

    if state.epoch < 0 or state.geometry_epoch != state.epoch:
        raise ValueError(
            "authoritative VOF geometry is stale: "
            f"geometry_epoch={state.geometry_epoch}, epoch={state.epoch}"
        )
    phi = np.asarray(state.phi.numpy()).copy()
    cell_type = np.asarray(state.cell_type.numpy()).copy()
    normal = np.asarray(state.normal.numpy()).copy()
    offset = np.asarray(state.plic_offset.numpy()).copy()
    curvature = np.asarray(state.curvature.numpy()).copy()
    expected_shape = tuple(int(value) for value in state.shape)
    if normal.shape != (*expected_shape, 3):
        raise ValueError("authoritative normal shape does not match VOF grid")
    for name, values in (
        ("normal", normal),
        ("plic_offset", offset),
        ("curvature", curvature),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"authoritative {name} contains NaN or Inf")
    interface = cell_type == 1
    non_interface = ~interface
    if np.any(normal[non_interface] != 0.0):
        raise ValueError("authoritative normal must be zero outside INTERFACE")
    if np.any(offset[non_interface] != 0.0):
        raise ValueError("authoritative PLIC offset must be zero outside INTERFACE")
    if np.any(curvature[non_interface] != 0.0):
        raise ValueError("authoritative curvature must be zero outside INTERFACE")
    lengths = np.linalg.norm(normal[interface], axis=-1)
    if np.any((lengths > 1.0e-6) & (np.abs(lengths - 1.0) > 2.0e-5)):
        raise ValueError("non-degenerate authoritative normals must be unit length")
    for index in zip(*np.nonzero(interface), strict=True):
        vector = normal[index]
        if float(np.linalg.norm(vector)) <= 1.0e-6:
            if offset[index] != 0.0 or curvature[index] != 0.0:
                raise ValueError("degenerate interface geometry must be canonical zero")
            continue
        reconstructed = plic_cube_volume(vector, float(offset[index]))
        target_fill = float(np.clip(phi[index], 0.0, 1.0))
        if abs(reconstructed - target_fill) > volume_tolerance:
            raise ValueError(
                "authoritative PLIC volume does not close phi at "
                f"{index}: {reconstructed} != clip({float(phi[index])})"
            )
    if np.any(np.abs(curvature[interface]) > 1.0 + 1.0e-6):
        raise ValueError("authoritative curvature exceeds the frozen limiter")


class VofInterfaceGeometry:
    """Build epoch-consistent normal, PLIC and curvature from final VOF fill."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)

    def compute(self, state: VofGridState, *, validate: bool = True) -> None:
        if tuple(int(value) for value in state.shape) != self.shape:
            raise ValueError(
                f"P6 geometry shape {state.shape} does not match {self.shape}"
            )
        if state.epoch < 0:
            raise ValueError("P6 geometry requires an initialized VOF epoch")
        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            authoritative_normal_plic_kernel,
            dim=self.shape,
            inputs=[
                state.phi,
                state.cell_type,
                state.normal,
                state.plic_offset,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        wp.launch(
            authoritative_curvature_kernel,
            dim=self.shape,
            inputs=[
                state.phi,
                state.cell_type,
                state.curvature,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        state.geometry_epoch = state.epoch
        if validate:
            wp.synchronize_device(self.device)
            validate_authoritative_geometry(state)


__all__ = [
    "InterfaceGeometry",
    "VofInterfaceGeometry",
    "parker_youngs_normal_kernel",
    "plic_cube_volume",
    "plic_plane_offset",
    "validate_authoritative_geometry",
]
