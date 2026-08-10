# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared face-Courant construction for geometric VOF on an FSL mask."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import fsl_courant_kernels


@dataclass(frozen=True)
class FslFaceCourantReference:
    face_courant: np.ndarray
    donor_count: np.ndarray
    extrapolated_face_count: int
    solid_face_count: int
    maximum_condition_number: float


@dataclass(frozen=True)
class FslFaceCourantDiagnostics:
    invalid_moment_count: int
    insufficient_donor_face_count: int
    ill_conditioned_face_count: int
    invalid_courant_face_count: int
    extrapolated_face_count: int
    solid_face_count: int


class HomeFreeFslFaceCourantBuilder:
    """Warp owner for shared geometric-VOF face Courant numbers."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        axis: int,
        maximum_fit_condition: float = 1.0e3,
        maximum_courant: float = 1.0,
    ) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("FSL face Courant axis must be 0, 1, or 2")
        if not math.isfinite(maximum_fit_condition) or maximum_fit_condition <= 1.0:
            raise ValueError("maximum_fit_condition must be finite and greater than one")
        if not math.isfinite(maximum_courant) or maximum_courant <= 0.0:
            raise ValueError("maximum_courant must be finite and positive")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.axis = int(axis)
        self.maximum_fit_condition = float(maximum_fit_condition)
        self.maximum_courant = float(maximum_courant)
        face_shape = list(self.res)
        face_shape[axis] += 1
        self.face_shape = tuple(face_shape)
        self.face_courant = wp.zeros(
            self.face_shape, dtype=float, device=self.device
        )
        self.donor_count = wp.zeros(
            self.face_shape, dtype=wp.int32, device=self.device
        )
        self._counts = wp.zeros(6, dtype=wp.int32, device=self.device)
        self.last_diagnostics: FslFaceCourantDiagnostics | None = None

    def build(
        self,
        fluid_state: HomeLbmState,
        active: wp.array,
        fill_level: wp.array,
        flags: wp.array,
    ) -> wp.array:
        if fluid_state.model is not self.model:
            raise ValueError("FSL face Courant builder and HOME state must share a model")
        self._validate_field(active, wp.int32, "active")
        self._validate_field(fill_level, wp.float32, "fill_level")
        self._validate_field(flags, wp.int32, "flags")
        self._counts.zero_()
        wp.launch(
            fsl_courant_kernels.construct_fsl_face_courant_kernel,
            dim=self.face_shape,
            inputs=[
                fluid_state.moments,
                active,
                fill_level,
                flags,
                self.face_courant,
                self.donor_count,
                self._counts,
                self.axis,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                self.maximum_fit_condition,
                self.maximum_courant,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        diagnostics = FslFaceCourantDiagnostics(
            invalid_moment_count=int(counts[0]),
            insufficient_donor_face_count=int(counts[1]),
            ill_conditioned_face_count=int(counts[2]),
            invalid_courant_face_count=int(counts[3]),
            extrapolated_face_count=int(counts[4]),
            solid_face_count=int(counts[5]),
        )
        self.last_diagnostics = diagnostics
        if diagnostics.invalid_moment_count:
            raise FloatingPointError(
                "FSL face Courant construction found "
                f"{diagnostics.invalid_moment_count} invalid moment samples"
            )
        if diagnostics.insufficient_donor_face_count:
            raise RuntimeError(
                "FSL face Courant construction found "
                f"{diagnostics.insufficient_donor_face_count} faces with fewer than four donors"
            )
        if diagnostics.ill_conditioned_face_count:
            raise RuntimeError(
                "FSL face Courant construction found "
                f"{diagnostics.ill_conditioned_face_count} rank-deficient or ill-conditioned fits"
            )
        if diagnostics.invalid_courant_face_count:
            values = self.face_courant.numpy()
            invalid = np.argwhere(
                ~np.isfinite(values) | (np.abs(values) > self.maximum_courant)
            )
            samples = [
                {
                    "index": tuple(int(component) for component in index),
                    "value": float(values[tuple(index)]),
                    "donors": int(self.donor_count.numpy()[tuple(index)]),
                }
                for index in invalid[:8]
            ]
            raise RuntimeError(
                "FSL face Courant construction found "
                f"{diagnostics.invalid_courant_face_count} non-finite or super-CFL "
                f"faces on axis {self.axis}: {samples}"
            )
        return self.face_courant

    def _validate_field(self, field: wp.array, dtype: object, name: str) -> None:
        if tuple(field.shape) != self.res or field.device != self.device:
            raise ValueError(f"{name} must match the FSL grid shape and device")
        if field.dtype != dtype:
            raise TypeError(f"{name} has the wrong Warp dtype")


def fsl_face_courant(
    moments: np.ndarray,
    active: np.ndarray,
    fill_level: np.ndarray,
    *,
    axis: int,
    flags: np.ndarray | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    maximum_fit_condition: float = 1.0e3,
    maximum_courant: float = 1.0,
) -> FslFaceCourantReference:
    """Construct one shared face velocity from active HOME moments.

    Active-active faces use the centered arithmetic value.  Faces adjacent to
    positive liquid volume but lacking two active cells use a weighted affine
    fit over a deterministic ``6 x 5 x 5`` active-node window.  The fit target
    is the face center, so its intercept is the required normal Courant number.
    Faces adjacent only to zero-fill cells are exactly zero.
    """

    values = np.asarray(moments, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    fill = np.asarray(fill_level, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("FSL face Courant moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if mask.shape != shape or fill.shape != shape:
        raise ValueError("FSL face Courant fields must share one grid shape")
    if flags is None:
        category = np.zeros(shape, dtype=np.int32)
    else:
        category = np.asarray(flags, dtype=np.int32)
        if category.shape != shape or np.any(category < 0) or np.any(category > 3):
            raise ValueError("FSL face Courant flags are invalid")
    if axis not in (0, 1, 2):
        raise ValueError("FSL face Courant axis must be 0, 1, or 2")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not math.isfinite(maximum_fit_condition) or maximum_fit_condition <= 1.0:
        raise ValueError("maximum_fit_condition must be finite and greater than one")
    if not math.isfinite(maximum_courant) or maximum_courant <= 0.0:
        raise ValueError("maximum_courant must be finite and positive")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("FSL face Courant fill must be finite and in [0, 1]")
    if not np.isfinite(values[mask]).all() or np.any(values[..., 0][mask] <= 0.0):
        raise ValueError("active FSL moments must be finite with positive density")

    face_shape = list(shape)
    face_shape[axis] += 1
    courant = np.zeros(face_shape, dtype=np.float64)
    donor_count = np.zeros(face_shape, dtype=np.int32)
    extrapolated = 0
    solid_faces = 0
    largest_condition = 0.0
    unique_face_shape = list(face_shape)
    if periodic[axis]:
        unique_face_shape[axis] -= 1

    def cell_index(unwrapped: list[int]) -> tuple[int, int, int] | None:
        result = unwrapped.copy()
        for component in range(3):
            if result[component] < 0 or result[component] >= shape[component]:
                if periodic[component]:
                    result[component] %= shape[component]
                else:
                    return None
        return tuple(result)

    for face_index in np.ndindex(tuple(unique_face_shape)):
        face = face_index[axis]
        if not periodic[axis] and (face == 0 or face == shape[axis]):
            continue
        lower = list(face_index)
        lower[axis] = face - 1
        upper = list(face_index)
        upper[axis] = face
        lower_index = cell_index(lower)
        upper_index = cell_index(upper)
        assert lower_index is not None and upper_index is not None
        if category[lower_index] == 3 or category[upper_index] == 3:
            solid_faces += 1
            continue
        lower_active = bool(mask[lower_index])
        upper_active = bool(mask[upper_index])
        if lower_active and upper_active:
            lower_velocity = values[lower_index][1 + axis] / values[lower_index][0]
            upper_velocity = values[upper_index][1 + axis] / values[upper_index][0]
            value = 0.5 * (lower_velocity + upper_velocity)
            donor_count[face_index] = 2
        elif fill[lower_index] == 0.0 and fill[upper_index] == 0.0:
            value = 0.0
        else:
            rows: list[tuple[float, float, float, float]] = []
            samples: list[float] = []
            weights: list[float] = []
            for along in range(-3, 3):
                for transverse_0 in range(-2, 3):
                    for transverse_1 in range(-2, 3):
                        unwrapped = list(face_index)
                        displacement = np.zeros(3, dtype=np.float64)
                        unwrapped[axis] = face + along
                        displacement[axis] = along + 0.5
                        transverse_axes = [value for value in range(3) if value != axis]
                        unwrapped[transverse_axes[0]] += transverse_0
                        unwrapped[transverse_axes[1]] += transverse_1
                        displacement[transverse_axes[0]] = transverse_0
                        displacement[transverse_axes[1]] = transverse_1
                        donor = cell_index(unwrapped)
                        if donor is None or not mask[donor]:
                            continue
                        rho = values[donor][0]
                        velocity = values[donor][1 + axis] / rho
                        row = (1.0, *displacement)
                        radius_squared = float(np.dot(displacement, displacement))
                        rows.append(row)
                        samples.append(float(velocity))
                        weights.append(1.0 / (1.0 + radius_squared))
            donor_count[face_index] = len(rows)
            if len(rows) < 4:
                raise RuntimeError(
                    f"FSL face {face_index} has only {len(rows)} active affine donors"
                )
            design = np.asarray(rows, dtype=np.float64)
            sample = np.asarray(samples, dtype=np.float64)
            root_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
            weighted_design = root_weight[:, None] * design
            singular_values = np.linalg.svd(weighted_design, compute_uv=False)
            if singular_values[-1] <= 1.0e-12 * singular_values[0]:
                raise RuntimeError(f"FSL face {face_index} affine donor fit is rank deficient")
            condition = float(singular_values[0] / singular_values[-1])
            if condition > maximum_fit_condition:
                raise RuntimeError(
                    f"FSL face {face_index} affine donor condition {condition:.6g} "
                    f"exceeds {maximum_fit_condition:.6g}"
                )
            coefficients, _, rank, _ = np.linalg.lstsq(
                weighted_design, root_weight * sample, rcond=None
            )
            if rank != 4 or not np.isfinite(coefficients).all():
                raise RuntimeError(f"FSL face {face_index} affine velocity fit failed")
            value = float(coefficients[0])
            extrapolated += 1
            largest_condition = max(largest_condition, condition)
        if not math.isfinite(value) or abs(value) > maximum_courant:
            raise RuntimeError(
                f"FSL face {face_index} produced invalid Courant number {value:.9g}"
            )
        courant[face_index] = value

    if periodic[axis]:
        lower_face = [slice(None)] * 3
        lower_face[axis] = 0
        upper_face = [slice(None)] * 3
        upper_face[axis] = shape[axis]
        courant[tuple(upper_face)] = courant[tuple(lower_face)]
        donor_count[tuple(upper_face)] = donor_count[tuple(lower_face)]
    return FslFaceCourantReference(
        face_courant=courant,
        donor_count=donor_count,
        extrapolated_face_count=extrapolated,
        solid_face_count=solid_faces,
        maximum_condition_number=largest_condition,
    )
