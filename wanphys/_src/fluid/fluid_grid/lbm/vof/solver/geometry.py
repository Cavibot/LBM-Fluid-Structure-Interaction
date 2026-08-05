# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Debug and authoritative interface geometry for VOF fill fractions."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from .kernels.geometry import (
    authoritative_curvature_kernel,
    authoritative_normal_plic_kernel,
    parker_youngs_normal_kernel,
)
from ..state import DebugMockScToVofState, VofGridState


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


_PLIC_COMPONENT_EPSILON = 1.0e-4


def _prepare_plic_weights(
    normal: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return thresholded, L1-normalized PLIC weights and their scale."""

    vector = np.abs(np.asarray(normal, dtype=np.float64))
    vector[vector < _PLIC_COMPONENT_EPSILON] = 0.0
    scale = float(np.sum(vector))
    if not math.isfinite(scale) or scale <= 1.0e-14:
        return np.zeros(3, dtype=np.float64), 0.0
    vector[(vector / scale) <= _PLIC_COMPONENT_EPSILON] = 0.0
    scale = float(np.sum(vector))
    if scale <= 1.0e-14:
        return np.zeros(3, dtype=np.float64), 0.0
    return vector / scale, scale


def _scaled_positive_cube_difference(value: float, interval: float) -> float:
    """Evaluate ``((t+)^3 - ((t-h)+)^3) / h`` without cancellation."""

    if value <= 0.0:
        return 0.0
    if value < interval:
        return value**3 / interval
    return (
        3.0 * value * value
        - 3.0 * value * interval
        + interval * interval
    )


def _plic_cube_offset_reduced(
    volume: float,
    n1: float,
    n2: float,
    n3: float,
) -> float:
    """Invert the symmetry-reduced unit-cube plane volume analytically."""

    n12 = n1 + n2
    n3_volume = n3 * volume
    if n12 <= 2.0 * n3_volume:
        return n3_volume + 0.5 * n12

    square_n1 = n1 * n1
    six_n2 = 6.0 * n2
    v1 = square_n1 / six_n2
    if v1 <= n3_volume < v1 + 0.5 * (n2 - n1):
        return 0.5 * (
            n1
            + math.sqrt(
                square_n1 + 8.0 * n2 * (n3_volume - v1)
            )
        )

    volume6 = n1 * six_n2 * n3_volume
    if n3_volume < v1:
        return math.cbrt(volume6)

    v3 = 0.5 * n12
    if n3 < n12:
        v3 = (
            n3 * n3 * (3.0 * n12 - n3)
            + square_n1 * (n1 - 3.0 * n3)
            + n2 * n2 * (n2 - 3.0 * n3)
        ) / (n1 * six_n2)
    square_n12 = square_n1 + n2 * n2
    volume6_minus_cubes = volume6 - n1**3 - n2**3
    case_three = n3_volume < v3
    a = (
        volume6_minus_cubes
        if case_three
        else 0.5 * (volume6_minus_cubes - n3**3)
    )
    b = (
        square_n12
        if case_three
        else 0.5 * (square_n12 + n3 * n3)
    )
    c = n12 if case_three else 0.5
    t = math.sqrt(max(c * c - b, 0.0))
    argument = (
        c**3 - 0.5 * a - 1.5 * b * c
    ) / (t**3)
    return c - 2.0 * t * math.sin(
        math.asin(float(np.clip(argument, -1.0, 1.0))) / 3.0
    )


def plic_cube_volume(
    normal: tuple[float, float, float] | np.ndarray,
    offset: float,
) -> float:
    """Return the liquid volume of ``n dot r <= offset`` in a centered cube."""

    weights, l1 = _prepare_plic_weights(normal)
    if l1 <= 1.0e-14:
        return 0.0
    alpha = float(offset) / l1 + 0.5
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    active = weights[weights > 0.0]
    dimension = int(active.size)
    if dimension == 1:
        return alpha
    if dimension == 2:
        a, b = (float(value) for value in active)
        a, b = min(a, b), max(a, b)
        if alpha < a:
            result = alpha**2 / (2.0 * a * b)
        elif alpha <= b:
            result = (alpha - 0.5 * a) / b
        else:
            result = 1.0 - (1.0 - alpha) ** 2 / (2.0 * a * b)
        return float(np.clip(result, 0.0, 1.0))
    m1, m2, m3 = sorted(float(value) for value in active)
    result = (
        _scaled_positive_cube_difference(alpha, m1)
        - _scaled_positive_cube_difference(alpha - m2, m1)
        - _scaled_positive_cube_difference(alpha - m3, m1)
        + _scaled_positive_cube_difference(alpha - m2 - m3, m1)
    ) / (6.0 * m2 * m3)
    return float(np.clip(result, 0.0, 1.0))


def plic_plane_offset(
    fill: float,
    normal: tuple[float, float, float] | np.ndarray,
    *,
    iterations: int = 60,
) -> float:
    """Host oracle for the centered unit-cube PLIC offset.

    ``iterations`` remains for source compatibility; the current
    Scardovelli–Zaleski/Kawano reduced inverse is analytical.
    """

    target = float(fill)
    vector = np.asarray(normal, dtype=np.float64)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("PLIC fill must be finite and lie in [0, 1]")
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("PLIC normal must contain three finite components")
    if iterations <= 0:
        raise ValueError("PLIC iterations must be positive")
    weights, l1 = _prepare_plic_weights(vector)
    if l1 <= 1.0e-14:
        return 0.0
    n1, n2, n3 = sorted(float(value) for value in weights)
    reduced_volume = 0.5 - abs(target - 0.5)
    reduced_offset = _plic_cube_offset_reduced(
        reduced_volume,
        n1,
        n2,
        n3,
    )
    magnitude = l1 * (0.5 - reduced_offset)
    if target < 0.5:
        return -magnitude
    if target > 0.5:
        return magnitude
    return 0.0


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
