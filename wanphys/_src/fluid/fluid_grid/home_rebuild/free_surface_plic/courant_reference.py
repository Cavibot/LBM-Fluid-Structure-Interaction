# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference HOME face-Courant construction and active-domain projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..free_surface_fsl import FslCellFlag


@dataclass(frozen=True)
class FaceCourantFields:
    face_courant: tuple[np.ndarray, np.ndarray, np.ndarray]
    maximum_courant: float
    maximum_active_divergence: float


@dataclass(frozen=True)
class CourantProjectionResult:
    face_courant: tuple[np.ndarray, np.ndarray, np.ndarray]
    pressure: np.ndarray
    iteration_count: int
    initial_maximum_divergence: float
    projected_maximum_divergence: float
    maximum_face_correction: float


def _active_mask(flags: np.ndarray, solid: np.ndarray) -> np.ndarray:
    return (flags != int(FslCellFlag.GAS)) & ~solid


def _face_cells(
    face: tuple[int, int, int],
    shape: tuple[int, int, int],
    *,
    axis: int,
    periodic: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    coordinate = face[axis]
    size = shape[axis]
    if coordinate in (0, size) and not periodic:
        return None
    left = list(face)
    right = list(face)
    left[axis] = (coordinate - 1) % size
    right[axis] = coordinate % size
    return tuple(left), tuple(right)


def active_divergence(
    face_courant: tuple[np.ndarray, np.ndarray, np.ndarray],
    active: np.ndarray,
) -> np.ndarray:
    divergence = np.zeros(active.shape, dtype=np.float64)
    for axis, face in enumerate(face_courant):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, active.shape[axis])
        upper[axis] = slice(1, active.shape[axis] + 1)
        divergence += face[tuple(upper)] - face[tuple(lower)]
    divergence[~active] = 0.0
    return divergence


def build_face_courant_reference(
    moments: np.ndarray,
    flags: np.ndarray,
    solid_mask: np.ndarray,
    *,
    closed_axes: tuple[bool, bool, bool],
    maximum_courant: float = 0.95,
) -> FaceCourantFields:
    """Interpolate HOME velocity to shared faces without crossing solids."""

    home = np.asarray(moments, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    solid = np.asarray(solid_mask, dtype=bool)
    if (
        home.ndim != 4
        or home.shape[-1] != 10
        or home.shape[:-1] != cell_flags.shape
        or solid.shape != cell_flags.shape
        or len(closed_axes) != 3
    ):
        raise ValueError("HOME moments, flags, solid mask, and axes are incompatible")
    if maximum_courant <= 0.0 or maximum_courant > 1.0:
        raise ValueError("maximum_courant must be in (0, 1]")
    active = _active_mask(cell_flags, solid)
    if np.any(active & (home[..., 0] <= 0.0)) or not np.isfinite(home).all():
        raise ValueError("active HOME moments must be finite with positive density")
    velocity = np.zeros(home.shape[:-1] + (3,), dtype=np.float64)
    velocity[active] = home[..., 1:4][active] / home[..., 0, None][active]
    faces: list[np.ndarray] = []
    for axis in range(3):
        face_shape = list(active.shape)
        face_shape[axis] += 1
        field = np.zeros(tuple(face_shape), dtype=np.float64)
        for face in np.ndindex(field.shape):
            cells = _face_cells(
                face,
                active.shape,
                axis=axis,
                periodic=not closed_axes[axis],
            )
            if cells is None:
                continue
            left, right = cells
            if solid[left] or solid[right]:
                continue
            left_active = active[left]
            right_active = active[right]
            if left_active and right_active:
                field[face] = 0.5 * (
                    velocity[left + (axis,)] + velocity[right + (axis,)]
                )
            elif left_active:
                field[face] = velocity[left + (axis,)]
            elif right_active:
                field[face] = velocity[right + (axis,)]
        if not closed_axes[axis]:
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = 0
            upper[axis] = -1
            field[tuple(upper)] = field[tuple(lower)]
        faces.append(field)
    largest = max(float(np.max(np.abs(field), initial=0.0)) for field in faces)
    if largest > maximum_courant:
        raise FloatingPointError(
            f"HOME face Courant {largest} exceeds {maximum_courant}"
        )
    divergence = active_divergence(tuple(faces), active)
    return FaceCourantFields(
        tuple(faces),
        largest,
        float(np.max(np.abs(divergence[active]), initial=0.0)),
    )


def project_face_courant_reference(
    face_courant: tuple[np.ndarray, np.ndarray, np.ndarray],
    flags: np.ndarray,
    solid_mask: np.ndarray,
    *,
    closed_axes: tuple[bool, bool, bool],
    maximum_iterations: int = 20000,
    absolute_tolerance: float = 1.0e-11,
) -> CourantProjectionResult:
    """Project shared faces to zero divergence in active free-surface cells."""

    cell_flags = np.asarray(flags, dtype=np.int32)
    solid = np.asarray(solid_mask, dtype=bool)
    if cell_flags.ndim != 3 or solid.shape != cell_flags.shape or len(closed_axes) != 3:
        raise ValueError("projection flags, solids, and axes are incompatible")
    if maximum_iterations < 1 or absolute_tolerance <= 0.0:
        raise ValueError("projection controls must be positive")
    source = tuple(np.asarray(field, dtype=np.float64) for field in face_courant)
    for axis, field in enumerate(source):
        expected = list(cell_flags.shape)
        expected[axis] += 1
        if field.shape != tuple(expected) or not np.isfinite(field).all():
            raise ValueError("face Courant field has an invalid shape or value")
    active = _active_mask(cell_flags, solid)
    divergence = active_divergence(source, active)
    initial_maximum = float(np.max(np.abs(divergence[active]), initial=0.0))
    pressure = np.zeros(cell_flags.shape, dtype=np.float64)
    iteration_count = 0
    for iteration in range(1, maximum_iterations + 1):
        updated = np.zeros_like(pressure)
        for cell in zip(*np.nonzero(active)):
            neighbor_sum = 0.0
            degree = 0
            for axis in range(3):
                for sign in (-1, 1):
                    neighbor = list(cell)
                    neighbor[axis] += sign
                    if closed_axes[axis] and not 0 <= neighbor[axis] < active.shape[axis]:
                        continue
                    neighbor[axis] %= active.shape[axis]
                    neighbor_cell = tuple(neighbor)
                    if neighbor_cell == cell or solid[neighbor_cell]:
                        continue
                    degree += 1
                    if active[neighbor_cell]:
                        neighbor_sum += pressure[neighbor_cell]
            if degree == 0:
                raise RuntimeError("active projection cell has no pressure face")
            updated[cell] = (neighbor_sum - divergence[cell]) / degree
        pressure = updated
        iteration_count = iteration
        if iteration % 20 == 0 or iteration == maximum_iterations:
            residual = np.zeros_like(divergence)
            for cell in zip(*np.nonzero(active)):
                laplacian = 0.0
                for axis in range(3):
                    for sign in (-1, 1):
                        neighbor = list(cell)
                        neighbor[axis] += sign
                        if closed_axes[axis] and not 0 <= neighbor[axis] < active.shape[axis]:
                            continue
                        neighbor[axis] %= active.shape[axis]
                        neighbor_cell = tuple(neighbor)
                        if neighbor_cell == cell or solid[neighbor_cell]:
                            continue
                        neighbor_pressure = (
                            pressure[neighbor_cell] if active[neighbor_cell] else 0.0
                        )
                        laplacian += neighbor_pressure - pressure[cell]
                residual[cell] = divergence[cell] - laplacian
            if float(np.max(np.abs(residual[active]), initial=0.0)) <= absolute_tolerance:
                break
    projected = [field.copy() for field in source]
    maximum_correction = 0.0
    for axis, field in enumerate(projected):
        for face in np.ndindex(field.shape):
            cells = _face_cells(
                face,
                active.shape,
                axis=axis,
                periodic=not closed_axes[axis],
            )
            if cells is None:
                field[face] = 0.0
                continue
            left, right = cells
            if solid[left] or solid[right] or not (active[left] or active[right]):
                field[face] = 0.0
                continue
            left_pressure = pressure[left] if active[left] else 0.0
            right_pressure = pressure[right] if active[right] else 0.0
            correction = right_pressure - left_pressure
            field[face] -= correction
            maximum_correction = max(maximum_correction, abs(correction))
        if not closed_axes[axis]:
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = 0
            upper[axis] = -1
            field[tuple(upper)] = field[tuple(lower)]
    projected_divergence = active_divergence(tuple(projected), active)
    projected_maximum = float(
        np.max(np.abs(projected_divergence[active]), initial=0.0)
    )
    if projected_maximum > absolute_tolerance:
        raise FloatingPointError(
            f"Courant projection residual {projected_maximum} exceeds {absolute_tolerance}"
        )
    return CourantProjectionResult(
        tuple(projected),
        pressure,
        iteration_count,
        initial_maximum,
        projected_maximum,
        maximum_correction,
    )
