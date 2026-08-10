# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative directional PLIC advection oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry_reference import plane_volume_fraction


@dataclass(frozen=True)
class PlicAxisAdvectionResult:
    fill_level: np.ndarray
    face_flux: np.ndarray
    volume_before: float
    volume_after: float
    maximum_courant: float


def plic_swept_slab_volume(
    fill_level: float,
    normal: np.ndarray,
    plane_offset: float,
    *,
    axis: int,
    signed_courant: float,
) -> float:
    """Return liquid volume in the donor-side slab swept through one face."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not math.isfinite(fill_level) or not 0.0 <= fill_level <= 1.0:
        raise ValueError("fill_level must be finite and in [0, 1]")
    if not math.isfinite(signed_courant) or abs(signed_courant) > 1.0:
        raise ValueError("signed_courant must be finite and in [-1, 1]")
    width = abs(signed_courant)
    if width == 0.0 or fill_level == 0.0:
        return 0.0
    if fill_level == 1.0:
        return width
    value = np.asarray(normal, dtype=np.float64)
    magnitude = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or magnitude <= 1.0e-14:
        raise ValueError("interface normal must be one finite nonzero three-vector")
    if not math.isfinite(plane_offset):
        raise ValueError("plane_offset must be finite")
    unit_normal = value / magnitude
    face_sign = 1.0 if signed_courant > 0.0 else -1.0
    slab_center = face_sign * (0.5 - 0.5 * width)
    scale = np.ones(3, dtype=np.float64)
    scale[axis] = width
    transformed_normal = unit_normal * scale
    transformed_offset = plane_offset / magnitude - unit_normal[axis] * slab_center
    volume = width * plane_volume_fraction(transformed_offset, transformed_normal)
    lower = max(0.0, width - (1.0 - fill_level))
    upper = min(width, fill_level)
    if volume < lower - 2.0e-12 or volume > upper + 2.0e-12:
        raise FloatingPointError("PLIC slab volume violates intersection bounds")
    return float(np.clip(volume, lower, upper))


def advect_plic_axis(
    fill_level: np.ndarray,
    normal: np.ndarray,
    plane_offset: np.ndarray,
    solid_mask: np.ndarray,
    face_courant: np.ndarray,
    *,
    axis: int,
    periodic: bool,
    bound_tolerance: float = 2.0e-12,
    compression: np.ndarray | None = None,
) -> PlicAxisAdvectionResult:
    """Apply one shared-face conservative PLIC sweep."""

    fill = np.asarray(fill_level, dtype=np.float64)
    normals = np.asarray(normal, dtype=np.float64)
    offsets = np.asarray(plane_offset, dtype=np.float64)
    solid = np.asarray(solid_mask, dtype=bool)
    courant = np.asarray(face_courant, dtype=np.float64)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    face_shape = list(fill.shape)
    face_shape[axis] += 1
    if (
        fill.ndim != 3
        or normals.shape != fill.shape + (3,)
        or offsets.shape != fill.shape
        or solid.shape != fill.shape
        or courant.shape != tuple(face_shape)
    ):
        raise ValueError("PLIC advection fields have incompatible shapes")
    compression_field = None
    if compression is not None:
        compression_field = np.asarray(compression, dtype=np.int32)
        if compression_field.shape != fill.shape or not np.isin(
            compression_field, (0, 1)
        ).all():
            raise ValueError("compression must be a binary cell field")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain finite and in [0, 1]")
    if not np.isfinite(courant).all() or np.max(np.abs(courant), initial=0.0) > 1.0:
        raise ValueError("face_courant must be finite and in [-1, 1]")
    flux = np.zeros(tuple(face_shape), dtype=np.float64)
    size = fill.shape[axis]
    for face in np.ndindex(tuple(face_shape)):
        face_index = face[axis]
        boundary = face_index in (0, size)
        if boundary and not periodic:
            if abs(courant[face]) > bound_tolerance:
                raise ValueError("closed boundary face must have zero Courant number")
            continue
        left = list(face)
        right = list(face)
        left[axis] = (face_index - 1) % size
        right[axis] = face_index % size
        left_cell = tuple(left)
        right_cell = tuple(right)
        if solid[left_cell] or solid[right_cell]:
            if abs(courant[face]) > bound_tolerance:
                raise ValueError("solid-adjacent face must have zero Courant number")
            continue
        signed_courant = float(courant[face])
        if signed_courant == 0.0:
            continue
        donor = left_cell if signed_courant > 0.0 else right_cell
        donor_fill = float(fill[donor])
        swept = plic_swept_slab_volume(
            donor_fill,
            normals[donor],
            float(offsets[donor]),
            axis=axis,
            signed_courant=signed_courant,
        )
        flux[face] = math.copysign(swept, signed_courant)
    if periodic:
        lower_face = [slice(None)] * 3
        upper_face = [slice(None)] * 3
        lower_face[axis] = 0
        upper_face[axis] = -1
        flux[tuple(upper_face)] = flux[tuple(lower_face)]
    updated = fill.copy()
    for cell in np.ndindex(fill.shape):
        if solid[cell]:
            updated[cell] = 0.0
            continue
        lower = list(cell)
        upper = list(cell)
        lower[axis] = cell[axis]
        upper[axis] = cell[axis] + 1
        updated[cell] += flux[tuple(lower)] - flux[tuple(upper)]
        if compression_field is not None:
            updated[cell] += compression_field[cell] * (
                courant[tuple(upper)] - courant[tuple(lower)]
            )
    violation = max(
        float(np.max(-updated, initial=0.0)),
        float(np.max(updated - 1.0, initial=0.0)),
    )
    if violation > bound_tolerance:
        raise FloatingPointError(
            f"PLIC axis sweep exceeds fill bounds by {violation}"
        )
    updated = np.clip(updated, 0.0, 1.0)
    volume_before = float(np.sum(fill[~solid], dtype=np.float64))
    volume_after = float(np.sum(updated[~solid], dtype=np.float64))
    if compression_field is None and abs(volume_after - volume_before) > 2.0e-12 * max(
        1.0, volume_before
    ):
        raise FloatingPointError("PLIC axis sweep does not conserve liquid volume")
    return PlicAxisAdvectionResult(
        updated,
        flux,
        volume_before,
        volume_after,
        float(np.max(np.abs(courant), initial=0.0)),
    )
