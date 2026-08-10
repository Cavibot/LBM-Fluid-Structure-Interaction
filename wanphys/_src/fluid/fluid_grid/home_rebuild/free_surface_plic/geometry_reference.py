# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy oracle for isolated Parker-Youngs PLIC reconstruction."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

PLIC_INTERFACE_FLAG = 1

_NORMAL_COMPONENT_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class PlicGeometryFields:
    normal: np.ndarray
    plane_offset: np.ndarray
    interface_area: np.ndarray
    valid: np.ndarray


def _validated_normal(normal: np.ndarray) -> np.ndarray:
    value = np.asarray(normal, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("normal must be one finite three-vector")
    magnitude = float(np.linalg.norm(value))
    if magnitude <= 1.0e-14:
        raise ValueError("normal must have nonzero magnitude")
    return value


def _active_components(normal: np.ndarray) -> np.ndarray:
    coefficients = np.abs(_validated_normal(normal))
    return coefficients[
        coefficients > _NORMAL_COMPONENT_TOLERANCE * float(np.max(coefficients))
    ]


def plane_volume_fraction(offset: float, normal: np.ndarray) -> float:
    """Volume in the centered unit cube satisfying ``normal dot x <= offset``."""

    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    active = _active_components(normal)
    support = 0.5 * float(np.sum(active))
    if offset <= -support:
        return 0.0
    if offset >= support:
        return 1.0
    complement = offset > 0.0
    evaluation_offset = -offset if complement else offset
    shifted = evaluation_offset + 0.5 * float(np.sum(active))
    dimension = int(active.size)
    denominator = math.factorial(dimension) * float(np.prod(active))
    volume = 0.0
    for mask in itertools.product((0, 1), repeat=dimension):
        corner = float(np.dot(active, np.asarray(mask, dtype=np.float64)))
        volume += (-1.0) ** sum(mask) * max(shifted - corner, 0.0) ** dimension
    fraction = float(np.clip(volume / denominator, 0.0, 1.0))
    return 1.0 - fraction if complement else fraction


def plic_plane_offset(
    fill_level: float, normal: np.ndarray, *, iterations: int = 64
) -> float:
    """Invert the exact centered-cube volume by monotone bisection."""

    if not math.isfinite(fill_level) or not 0.0 <= fill_level <= 1.0:
        raise ValueError("fill_level must be finite and in [0, 1]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    active = _active_components(normal)
    bound = 0.5 * float(np.sum(active))
    if fill_level == 0.0:
        return -bound
    if fill_level == 1.0:
        return bound
    lower = -bound
    upper = bound
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if plane_volume_fraction(midpoint, normal) < fill_level:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def plane_interface_area(offset: float, normal: np.ndarray) -> float:
    """Exact area of the reconstructed plane section through the unit cube."""

    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    value = _validated_normal(normal)
    active = _active_components(value)
    dimension = int(active.size)
    shifted = offset + 0.5 * float(np.sum(active))
    denominator = math.factorial(dimension - 1) * float(np.prod(active))
    derivative = 0.0
    for mask in itertools.product((0, 1), repeat=dimension):
        corner = float(np.dot(active, np.asarray(mask, dtype=np.float64)))
        distance = shifted - corner
        if distance > 0.0:
            derivative += (-1.0) ** sum(mask) * distance ** (dimension - 1)
    return max(float(np.linalg.norm(value)) * derivative / denominator, 0.0)


def youngs_interface_normal(fill_stencil: np.ndarray) -> np.ndarray:
    """Return the Parker-Youngs liquid-to-gas normal for one 3x3x3 stencil."""

    fill = np.asarray(fill_stencil, dtype=np.float64)
    if fill.shape != (3, 3, 3):
        raise ValueError("fill_stencil must have shape (3, 3, 3)")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_stencil must contain finite values in [0, 1]")
    transverse = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
    weights = np.outer(transverse, transverse)
    gradient = np.asarray(
        [
            np.sum(weights * (fill[2, :, :] - fill[0, :, :])),
            np.sum(weights * (fill[:, 2, :] - fill[:, 0, :])),
            np.sum(weights * (fill[:, :, 2] - fill[:, :, 0])),
        ]
    )
    magnitude = float(np.linalg.norm(gradient))
    if magnitude <= 1.0e-14:
        raise ValueError("interface normal is undefined for a zero-gradient stencil")
    return -gradient / magnitude


def _sample_index(
    coordinate: int, size: int, *, periodic: bool
) -> int:
    if periodic:
        return coordinate % size
    return min(max(coordinate, 0), size - 1)


def reconstruct_plic_geometry(
    fill_level: np.ndarray,
    flags: np.ndarray,
    solid_mask: np.ndarray,
    *,
    closed_axes: tuple[bool, bool, bool],
) -> PlicGeometryFields:
    """Reconstruct every non-solid interface without changing fluid state."""

    fill = np.asarray(fill_level, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    solid = np.asarray(solid_mask, dtype=bool)
    if fill.ndim != 3 or fill.shape != cell_flags.shape or fill.shape != solid.shape:
        raise ValueError("fill_level, flags, and solid_mask must share one 3-D grid")
    if len(closed_axes) != 3:
        raise ValueError("closed_axes must contain three values")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain finite and in [0, 1]")
    interface = (cell_flags == PLIC_INTERFACE_FLAG) & ~solid
    normal = np.zeros((*fill.shape, 3), dtype=np.float64)
    plane_offset = np.zeros(fill.shape, dtype=np.float64)
    interface_area = np.zeros(fill.shape, dtype=np.float64)
    valid = np.zeros(fill.shape, dtype=bool)
    for center in zip(*np.nonzero(interface), strict=True):
        stencil = np.empty((3, 3, 3), dtype=np.float64)
        for local in np.ndindex(stencil.shape):
            delta = tuple(value - 1 for value in local)
            neighbor = tuple(
                _sample_index(
                    center[axis] + delta[axis],
                    fill.shape[axis],
                    periodic=not closed_axes[axis],
                )
                for axis in range(3)
            )
            stencil[local] = fill[center] if solid[neighbor] else fill[neighbor]
        reconstructed_normal = youngs_interface_normal(stencil)
        reconstructed_offset = plic_plane_offset(fill[center], reconstructed_normal)
        normal[center] = reconstructed_normal
        plane_offset[center] = reconstructed_offset
        interface_area[center] = plane_interface_area(
            reconstructed_offset, reconstructed_normal
        )
        valid[center] = True
    return PlicGeometryFields(normal, plane_offset, interface_area, valid)
