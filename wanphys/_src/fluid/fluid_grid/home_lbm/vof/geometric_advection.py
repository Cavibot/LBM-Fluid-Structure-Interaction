# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometric PLIC volume-flux reference for FSL-compatible VOF advection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from . import geometric_advection_kernels
from .plic import plane_volume_fraction


@dataclass(frozen=True)
class PlicAxisAdvectionResult:
    """One conservative directional PLIC sweep."""

    fill_level: np.ndarray
    face_flux: np.ndarray
    volume_before: float
    volume_after: float
    maximum_courant: float
    bounded_face_count: int
    maximum_bound_correction: float


@dataclass(frozen=True)
class PlicAxisAdvectionDiagnostics:
    invalid_face_count: int
    invalid_cell_count: int
    bounded_face_count: int
    maximum_bound_correction: float
    invalid_mass_face_count: int = 0
    invalid_mass_cell_count: int = 0
    invalid_momentum_face_count: int = 0
    invalid_momentum_cell_count: int = 0


class HomeFreeGeometricAxisAdvector:
    """Transactional Warp owner for one geometric PLIC directional sweep."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        axis: int,
        bound_tolerance: float = 2.0e-6,
        flux_bound_correction_tolerance: float = 4.0e-6,
    ) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("PLIC sweep axis must be 0, 1, or 2")
        if not math.isfinite(bound_tolerance) or bound_tolerance < 0.0:
            raise ValueError("PLIC bound tolerance must be finite and nonnegative")
        if (
            not math.isfinite(flux_bound_correction_tolerance)
            or flux_bound_correction_tolerance < 0.0
        ):
            raise ValueError(
                "PLIC flux-bound correction tolerance must be finite and nonnegative"
            )
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.device = model._device
        self.axis = int(axis)
        self.periodic = bool(model.periodic[axis])
        self.bound_tolerance = float(bound_tolerance)
        self.flux_bound_correction_tolerance = float(
            flux_bound_correction_tolerance
        )
        face_shape = list(self.res)
        face_shape[axis] += 1
        self.face_shape = tuple(face_shape)
        self.face_flux = wp.zeros(self.face_shape, dtype=float, device=self.device)
        self.face_error = wp.zeros(
            self.face_shape, dtype=wp.int32, device=self.device
        )
        self.raw_face_volume = wp.zeros(
            self.face_shape, dtype=float, device=self.device
        )
        self.face_mass_flux = wp.zeros(
            self.face_shape, dtype=float, device=self.device
        )
        self.updated_fill = wp.zeros(self.res, dtype=float, device=self.device)
        self.updated_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.face_momentum_flux = wp.zeros(
            self.face_shape, dtype=wp.vec3, device=self.device
        )
        self.updated_momentum = wp.zeros(
            self.res, dtype=wp.vec3, device=self.device
        )
        self._fill_before = wp.zeros(self.res, dtype=float, device=self.device)
        self._zero_compression = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._invalid_face_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._bounded_face_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._maximum_bound_correction = wp.zeros(
            1, dtype=float, device=self.device
        )
        self._invalid_mass_face_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_mass_cell_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_momentum_face_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_momentum_cell_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self.last_diagnostics: PlicAxisAdvectionDiagnostics | None = None
        self._diagnostic_normal: wp.array | None = None
        self._diagnostic_plane_offset: wp.array | None = None
        self._diagnostic_face_courant: wp.array | None = None

    def advect(
        self,
        fill_level: wp.array,
        normal: wp.array,
        plane_offset: wp.array,
        face_courant: wp.array,
        compression: wp.array | None = None,
        *,
        synchronize: bool = True,
    ) -> wp.array:
        self._validate_input(fill_level, self.res, wp.float32, "fill_level")
        self._validate_input(normal, self.res, wp.vec3, "normal")
        self._validate_input(plane_offset, self.res, wp.float32, "plane_offset")
        self._validate_input(
            face_courant, self.face_shape, wp.float32, "face_courant"
        )
        use_compression = compression is not None
        compression_field = self._zero_compression
        if compression is not None:
            self._validate_input(
                compression, self.res, wp.int32, "compression"
            )
            compression_field = compression
        wp.copy(self._fill_before, fill_level)
        self._invalid_face_count.zero_()
        self.face_error.zero_()
        self.raw_face_volume.zero_()
        self._invalid_cell_count.zero_()
        self._bounded_face_count.zero_()
        self._maximum_bound_correction.zero_()
        self._invalid_mass_face_count.zero_()
        self._invalid_mass_cell_count.zero_()
        self._invalid_momentum_face_count.zero_()
        self._invalid_momentum_cell_count.zero_()
        wp.launch(
            geometric_advection_kernels.plic_axis_face_flux_kernel,
            dim=self.face_shape,
            inputs=[
                self._fill_before,
                normal,
                plane_offset,
                face_courant,
                self.face_flux,
                self.face_error,
                self.raw_face_volume,
                self._invalid_face_count,
                self._bounded_face_count,
                self._maximum_bound_correction,
                self.flux_bound_correction_tolerance,
                self.axis,
                int(self.periodic),
                *self.res,
            ],
            device=self.device,
        )
        self._diagnostic_normal = normal
        self._diagnostic_plane_offset = plane_offset
        self._diagnostic_face_courant = face_courant
        wp.launch(
            geometric_advection_kernels.apply_plic_axis_flux_kernel,
            dim=self.res,
            inputs=[
                self._fill_before,
                self.face_flux,
                face_courant,
                compression_field,
                self.updated_fill,
                self._invalid_cell_count,
                self.axis,
                int(use_compression),
                self.bound_tolerance,
            ],
            device=self.device,
        )
        if synchronize:
            self.finalize_diagnostics()
        return self.updated_fill

    def advect_mass(
        self,
        mass: wp.array,
        fill_level: wp.array,
        *,
        synchronize: bool = True,
    ) -> wp.array:
        """Advect liquid mass with the already constructed PLIC volume flux."""

        self._validate_input(mass, self.res, wp.float32, "mass")
        self._validate_input(fill_level, self.res, wp.float32, "fill_level")
        self._invalid_mass_face_count.zero_()
        self._invalid_mass_cell_count.zero_()
        wp.launch(
            geometric_advection_kernels.plic_axis_mass_flux_kernel,
            dim=self.face_shape,
            inputs=[
                mass,
                fill_level,
                self.face_flux,
                self.face_mass_flux,
                self._invalid_mass_face_count,
                self.axis,
                int(self.periodic),
                *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            geometric_advection_kernels.apply_plic_axis_mass_flux_kernel,
            dim=self.res,
            inputs=[
                mass,
                self.face_mass_flux,
                self.updated_mass,
                self._invalid_mass_cell_count,
                self.axis,
                self.bound_tolerance,
            ],
            device=self.device,
        )
        if synchronize:
            self.finalize_diagnostics()
        return self.updated_mass

    def advect_momentum(
        self,
        momentum: wp.array,
        mass: wp.array,
        *,
        synchronize: bool = True,
    ) -> wp.array:
        """Advect ``M*u`` with the already constructed liquid-mass flux."""

        self._validate_input(momentum, self.res, wp.vec3, "momentum")
        self._validate_input(mass, self.res, wp.float32, "mass")
        self._invalid_momentum_face_count.zero_()
        self._invalid_momentum_cell_count.zero_()
        wp.launch(
            geometric_advection_kernels.plic_axis_momentum_flux_kernel,
            dim=self.face_shape,
            inputs=[
                momentum,
                mass,
                self.face_mass_flux,
                self.face_momentum_flux,
                self._invalid_momentum_face_count,
                self.axis,
                int(self.periodic),
                *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            geometric_advection_kernels.apply_plic_axis_momentum_flux_kernel,
            dim=self.res,
            inputs=[
                momentum,
                self.face_momentum_flux,
                self.updated_momentum,
                self._invalid_momentum_cell_count,
                self.axis,
            ],
            device=self.device,
        )
        if synchronize:
            self.finalize_diagnostics()
        return self.updated_momentum

    def finalize_diagnostics(self) -> PlicAxisAdvectionDiagnostics:
        """Synchronize and validate all fields launched for the current sweep."""

        wp.synchronize_device(self.device)
        diagnostics = PlicAxisAdvectionDiagnostics(
            invalid_face_count=int(self._invalid_face_count.numpy()[0]),
            invalid_cell_count=int(self._invalid_cell_count.numpy()[0]),
            bounded_face_count=int(self._bounded_face_count.numpy()[0]),
            maximum_bound_correction=float(
                self._maximum_bound_correction.numpy()[0]
            ),
            invalid_mass_face_count=int(self._invalid_mass_face_count.numpy()[0]),
            invalid_mass_cell_count=int(self._invalid_mass_cell_count.numpy()[0]),
            invalid_momentum_face_count=int(
                self._invalid_momentum_face_count.numpy()[0]
            ),
            invalid_momentum_cell_count=int(
                self._invalid_momentum_cell_count.numpy()[0]
            ),
        )
        self.last_diagnostics = diagnostics
        if diagnostics.invalid_face_count:
            examples = self._invalid_face_examples()
            raise RuntimeError(
                "geometric PLIC face-flux construction found "
                f"{diagnostics.invalid_face_count} invalid faces; bounded faces "
                f"{diagnostics.bounded_face_count}, maximum correction "
                f"{diagnostics.maximum_bound_correction:.9g}; first entries "
                "are (face, error, donor, fill, courant, normal, offset, raw, "
                f"lower_bound, upper_bound, correction): {examples}"
            )
        if diagnostics.invalid_cell_count:
            values = self.updated_fill.numpy()
            invalid = (~np.isfinite(values)) | (
                values < -self.bound_tolerance
            ) | (values > 1.0 + self.bound_tolerance)
            examples = [
                (tuple(int(value) for value in index), float(values[tuple(index)]))
                for index in np.argwhere(invalid)[:8]
            ]
            raise RuntimeError(
                "geometric PLIC update found "
                f"{diagnostics.invalid_cell_count} cells outside physical fill bounds; "
                f"first values {examples}"
            )
        if diagnostics.invalid_mass_face_count or diagnostics.invalid_mass_cell_count:
            raise RuntimeError(
                "geometric PLIC mass transport found "
                f"{diagnostics.invalid_mass_face_count} invalid faces and "
                f"{diagnostics.invalid_mass_cell_count} invalid cells"
            )
        if (
            diagnostics.invalid_momentum_face_count
            or diagnostics.invalid_momentum_cell_count
        ):
            raise RuntimeError(
                "geometric PLIC momentum transport found "
                f"{diagnostics.invalid_momentum_face_count} invalid faces and "
                f"{diagnostics.invalid_momentum_cell_count} invalid cells"
            )
        return diagnostics

    def _invalid_face_examples(self) -> list[tuple[object, ...]]:
        if (
            self._diagnostic_normal is None
            or self._diagnostic_plane_offset is None
            or self._diagnostic_face_courant is None
        ):
            return []
        errors = self.face_error.numpy()
        fill = self._fill_before.numpy()
        normal = self._diagnostic_normal.numpy()
        offset = self._diagnostic_plane_offset.numpy()
        courant = self._diagnostic_face_courant.numpy()
        raw = self.raw_face_volume.numpy()
        examples: list[tuple[object, ...]] = []
        for raw_face in np.argwhere(errors != 0)[:8]:
            face_index = tuple(int(value) for value in raw_face)
            signed_courant = float(courant[face_index])
            donor = list(face_index)
            face = face_index[self.axis]
            if signed_courant > 0.0:
                donor[self.axis] = face - 1
            else:
                donor[self.axis] = face
            if self.periodic:
                donor[self.axis] %= self.res[self.axis]
            donor_index = tuple(donor)
            donor_valid = all(
                0 <= donor_index[axis] < self.res[axis] for axis in range(3)
            )
            donor_fill = float(fill[donor_index]) if donor_valid else math.nan
            width = abs(signed_courant)
            lower = max(0.0, width - (1.0 - donor_fill))
            upper = min(width, donor_fill)
            raw_volume = float(raw[face_index])
            examples.append(
                (
                    face_index,
                    int(errors[face_index]),
                    donor_index,
                    donor_fill,
                    signed_courant,
                    tuple(float(value) for value in normal[donor_index])
                    if donor_valid
                    else None,
                    float(offset[donor_index]) if donor_valid else math.nan,
                    raw_volume,
                    lower,
                    upper,
                    abs(min(max(raw_volume, lower), upper) - raw_volume),
                )
            )
        return examples

    def _validate_input(
        self,
        array: wp.array,
        shape: tuple[int, int, int],
        dtype: object,
        name: str,
    ) -> None:
        if tuple(array.shape) != shape or array.device != self.device:
            raise ValueError(f"{name} must match the PLIC sweep shape and device")
        if array.dtype != dtype:
            raise TypeError(f"{name} has the wrong Warp dtype")


def _plic_swept_slab_volume_unbounded(
    fill_level: float,
    normal: np.ndarray,
    plane_offset: float,
    *,
    axis: int,
    signed_courant: float,
) -> float:
    fill = float(fill_level)
    courant = float(signed_courant)
    if axis not in (0, 1, 2):
        raise ValueError("PLIC sweep axis must be 0, 1, or 2")
    if not math.isfinite(fill) or not 0.0 <= fill <= 1.0:
        raise ValueError("PLIC donor fill must be finite and in [0, 1]")
    if not math.isfinite(courant) or abs(courant) > 1.0:
        raise ValueError("PLIC signed Courant number must be finite and in [-1, 1]")
    width = abs(courant)
    if width == 0.0 or fill == 0.0:
        return 0.0
    if fill == 1.0:
        return width

    n = np.asarray(normal, dtype=np.float64)
    if n.shape != (3,) or not np.isfinite(n).all():
        raise ValueError("PLIC donor normal must be a finite 3-vector")
    length = float(np.linalg.norm(n))
    if length <= 0.0 or not math.isfinite(plane_offset):
        raise ValueError("fractional PLIC donors require a valid plane")
    threshold = 1.0e-4 * float(np.max(np.abs(n)))
    n[np.abs(n) <= threshold] = 0.0
    reduced_length = float(np.linalg.norm(n))
    if reduced_length <= 0.0:
        raise ValueError("fractional PLIC donors require an active normal component")
    n = n / reduced_length
    offset = float(plane_offset) / reduced_length

    face_sign = 1.0 if courant > 0.0 else -1.0
    slab_center = face_sign * (0.5 - 0.5 * width)
    transformed_normal = n.copy()
    transformed_normal[axis] *= width
    transformed_offset = offset - n[axis] * slab_center
    return width * plane_volume_fraction(
        transformed_offset, transformed_normal
    )


def plic_swept_slab_volume(
    fill_level: float,
    normal: np.ndarray,
    plane_offset: float,
    *,
    axis: int,
    signed_courant: float,
) -> float:
    """Return the geometrically bounded liquid volume swept through a face.

    ``signed_courant`` selects the donor face: positive values use the upper
    face and negative values the lower face.  The returned volume is unsigned;
    callers attach the face-flux sign.  The PLIC cube uses coordinates
    ``[-1/2, 1/2]^3`` and liquid occupancy ``normal dot x <= plane_offset``.
    """

    fill = float(fill_level)
    width = abs(float(signed_courant))
    raw_volume = _plic_swept_slab_volume_unbounded(
        fill,
        normal,
        plane_offset,
        axis=axis,
        signed_courant=signed_courant,
    )
    lower_bound = max(0.0, width - (1.0 - fill))
    upper_bound = min(width, fill)
    return min(max(raw_volume, lower_bound), upper_bound)


def advect_plic_axis(
    fill_level: np.ndarray,
    normal: np.ndarray,
    plane_offset: np.ndarray,
    face_courant: np.ndarray,
    *,
    axis: int,
    periodic: bool,
    bound_tolerance: float = 2.0e-12,
    flux_bound_correction_tolerance: float = 4.0e-12,
    compression: np.ndarray | None = None,
) -> PlicAxisAdvectionResult:
    """Advect cell liquid volume through shared faces for one axis.

    ``face_courant`` has one extra entry along ``axis``.  Face ``f`` lies
    between cells ``f-1`` and ``f``.  Periodic sweeps require the first and last
    face Courant numbers to agree; closed nonperiodic sweeps require both to be
    zero.  A positive face value takes its PLIC slab from the lower-index donor,
    while a negative value takes it from the upper-index donor.
    """

    fill = np.asarray(fill_level, dtype=np.float64)
    normals = np.asarray(normal, dtype=np.float64)
    offsets = np.asarray(plane_offset, dtype=np.float64)
    courant = np.asarray(face_courant, dtype=np.float64)
    if fill.ndim != 3:
        raise ValueError("PLIC fill field must be three-dimensional")
    if normals.shape != fill.shape + (3,):
        raise ValueError("PLIC normal field must have grid shape + (3,)")
    if offsets.shape != fill.shape:
        raise ValueError("PLIC offset field must match the fill grid")
    if axis not in (0, 1, 2):
        raise ValueError("PLIC sweep axis must be 0, 1, or 2")
    expected_faces = list(fill.shape)
    expected_faces[axis] += 1
    if courant.shape != tuple(expected_faces):
        raise ValueError("PLIC face Courant field has the wrong shape")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("PLIC fill field must be finite and in [0, 1]")
    if not np.isfinite(courant).all() or np.any(np.abs(courant) > 1.0):
        raise ValueError("PLIC face Courant numbers must be finite and in [-1, 1]")
    if (
        not math.isfinite(flux_bound_correction_tolerance)
        or flux_bound_correction_tolerance < 0.0
    ):
        raise ValueError(
            "PLIC flux-bound correction tolerance must be finite and nonnegative"
        )
    compression_field: np.ndarray | None = None
    if compression is not None:
        compression_field = np.asarray(compression, dtype=np.int32)
        if compression_field.shape != fill.shape or np.any(
            (compression_field != 0) & (compression_field != 1)
        ):
            raise ValueError("Weymouth-Yue compression must be a binary cell field")
    lower_face = np.take(courant, 0, axis=axis)
    upper_face = np.take(courant, courant.shape[axis] - 1, axis=axis)
    if periodic:
        if not np.array_equal(lower_face, upper_face):
            raise ValueError("periodic PLIC boundary-face Courant numbers must match")
    elif np.any(lower_face != 0.0) or np.any(upper_face != 0.0):
        raise ValueError("closed nonperiodic PLIC boundary-face Courant numbers must be zero")

    flux = np.zeros_like(courant)
    bounded_face_count = 0
    maximum_bound_correction = 0.0
    face_shape = list(courant.shape)
    face_shape[axis] -= 1 if periodic else 0
    for face_index in np.ndindex(tuple(face_shape)):
        face = face_index[axis]
        value = float(courant[face_index])
        if value == 0.0:
            continue
        donor = list(face_index)
        if value > 0.0:
            donor[axis] = face - 1
        else:
            donor[axis] = face
        if periodic:
            donor[axis] %= fill.shape[axis]
        elif donor[axis] < 0 or donor[axis] >= fill.shape[axis]:
            raise RuntimeError("closed PLIC sweep selected a donor outside the domain")
        donor_index = tuple(donor)
        raw_volume = _plic_swept_slab_volume_unbounded(
            fill[donor_index],
            normals[donor_index],
            offsets[donor_index],
            axis=axis,
            signed_courant=value,
        )
        volume = plic_swept_slab_volume(
            fill[donor_index],
            normals[donor_index],
            offsets[donor_index],
            axis=axis,
            signed_courant=value,
        )
        correction = abs(volume - raw_volume)
        if correction > 0.0:
            bounded_face_count += 1
            maximum_bound_correction = max(maximum_bound_correction, correction)
        if correction > flux_bound_correction_tolerance:
            raise RuntimeError(
                "geometric PLIC face flux required excessive bound correction: "
                f"{correction:.9g} > {flux_bound_correction_tolerance:.9g}"
            )
        flux[face_index] = math.copysign(volume, value)
    if periodic:
        upper_index = [slice(None)] * 3
        upper_index[axis] = courant.shape[axis] - 1
        lower_index = [slice(None)] * 3
        lower_index[axis] = 0
        flux[tuple(upper_index)] = flux[tuple(lower_index)]

    lower_flux = np.take(flux, np.arange(fill.shape[axis]), axis=axis)
    upper_flux = np.take(flux, np.arange(1, fill.shape[axis] + 1), axis=axis)
    updated = fill + lower_flux - upper_flux
    if compression_field is not None:
        lower_courant = np.take(
            courant, np.arange(fill.shape[axis]), axis=axis
        )
        upper_courant = np.take(
            courant, np.arange(1, fill.shape[axis] + 1), axis=axis
        )
        updated += compression_field * (upper_courant - lower_courant)
    minimum = float(np.min(updated))
    maximum = float(np.max(updated))
    if minimum < -bound_tolerance or maximum > 1.0 + bound_tolerance:
        raise RuntimeError(
            "geometric PLIC sweep left physical fill bounds: "
            f"minimum={minimum:.9g}, maximum={maximum:.9g}"
        )
    updated = np.where(updated < 0.0, 0.0, np.where(updated > 1.0, 1.0, updated))
    volume_before = float(np.sum(fill, dtype=np.float64))
    volume_after = float(np.sum(updated, dtype=np.float64))
    if compression_field is None and periodic and not math.isclose(
        volume_before, volume_after, rel_tol=0.0, abs_tol=5.0e-13
    ):
        raise RuntimeError("periodic geometric PLIC sweep did not conserve liquid volume")
    return PlicAxisAdvectionResult(
        fill_level=updated,
        face_flux=flux,
        volume_before=volume_before,
        volume_after=volume_after,
        maximum_courant=float(np.max(np.abs(courant))),
        bounded_face_count=bounded_face_count,
        maximum_bound_correction=maximum_bound_correction,
    )
