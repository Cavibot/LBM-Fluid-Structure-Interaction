# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp owner for one isolated PLIC directional sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from . import advection_kernels
from .domain import PlicDomain
from .geometry import PlicGeometryState

if TYPE_CHECKING:
    from ..core import HomeCoreModel
    from ..free_surface_fsl import FslWallMask


@dataclass(frozen=True)
class PlicAxisAdvectionDiagnostics:
    volume_before: float
    volume_after: float
    relative_volume_drift: float
    maximum_courant: float
    maximum_bound_correction: float
    endpoint_fallback_face_count: int = 0
    wetting_cell_count: int = 0
    parallel_wall_interface_cell_count: int = 0


class PlicAxisAdvector:
    def __init__(
        self,
        model: HomeCoreModel | PlicDomain,
        walls: FslWallMask | None = None,
        *,
        axis: int,
    ) -> None:
        if walls is None:
            if not isinstance(model, PlicDomain):
                raise TypeError("standalone PLIC advection requires a PlicDomain")
            walls = model
        if walls.model is not model:
            raise ValueError("PLIC advection and walls must share one HOME model")
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        self.model = model
        self.walls = walls
        self.axis = int(axis)
        self.periodic = not walls.closed_axes[axis]
        self.res = walls.res
        face_shape = list(self.res)
        face_shape[axis] += 1
        self.face_shape = tuple(face_shape)
        self.face_flux = wp.zeros(self.face_shape, dtype=float, device=model._device)
        self.updated_fill = wp.zeros(self.res, dtype=float, device=model._device)
        self.face_mass_flux = wp.zeros(
            self.face_shape, dtype=float, device=model._device
        )
        self.updated_mass = wp.zeros(self.res, dtype=float, device=model._device)
        self.face_momentum_flux = wp.zeros(
            self.face_shape, dtype=wp.vec3, device=model._device
        )
        self.updated_momentum = wp.zeros(
            self.res, dtype=wp.vec3, device=model._device
        )
        self._invalid_count = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._endpoint_fallback_count = wp.zeros(
            1, dtype=wp.int32, device=model._device
        )
        self._maximum_correction = wp.zeros(1, dtype=float, device=model._device)
        self._unused_compression = wp.zeros(
            self.res, dtype=wp.int32, device=model._device
        )
        self.last_diagnostics: PlicAxisAdvectionDiagnostics | None = None

    def advect(
        self,
        fill_level: wp.array,
        geometry: PlicGeometryState,
        face_courant: wp.array,
        compression: wp.array | None = None,
        *,
        interface_epsilon: float = 0.0,
        endpoint_fallback_tolerance: float | None = None,
    ) -> tuple[wp.array, PlicAxisAdvectionDiagnostics]:
        if geometry.model is not self.model:
            raise ValueError("PLIC geometry and advection must share one model")
        if not 0.0 <= interface_epsilon < 0.5:
            raise ValueError("interface_epsilon must be in [0, 0.5)")
        if endpoint_fallback_tolerance is None:
            endpoint_fallback_tolerance = interface_epsilon
        if not interface_epsilon <= endpoint_fallback_tolerance < 0.5:
            raise ValueError(
                "endpoint_fallback_tolerance must be in "
                "[interface_epsilon, 0.5)"
            )
        if tuple(fill_level.shape) != self.res or tuple(face_courant.shape) != self.face_shape:
            raise ValueError("PLIC fill or face Courant shape does not match the sweep")
        compression_field = self._unused_compression
        if compression is not None:
            if tuple(compression.shape) != self.res or compression.dtype != wp.int32:
                raise ValueError("PLIC compression field must be int32 on the cell grid")
            compression_field = compression
        self._invalid_count.zero_()
        self._endpoint_fallback_count.zero_()
        self._maximum_correction.zero_()
        wp.launch(
            advection_kernels.face_flux_kernel,
            dim=self.face_shape,
            inputs=[
                fill_level,
                geometry.normal,
                geometry.plane_offset,
                geometry.valid,
                self.walls.device,
                face_courant,
                self.face_flux,
                self._invalid_count,
                self._endpoint_fallback_count,
                self._maximum_correction,
                self.axis,
                int(self.periodic),
                float(interface_epsilon),
                float(endpoint_fallback_tolerance),
                self.res[0],
                self.res[1],
                self.res[2],
            ],
            device=self.model._device,
        )
        if self.periodic:
            self._copy_periodic_boundary_face()
        wp.launch(
            advection_kernels.update_fill_kernel,
            dim=self.res,
            inputs=[
                fill_level,
                self.walls.device,
                self.face_flux,
                face_courant,
                compression_field,
                self.updated_fill,
                self._invalid_count,
                self._maximum_correction,
                self.axis,
                int(compression is not None),
            ],
            device=self.model._device,
        )
        invalid_count = int(self._invalid_count.numpy()[0])
        before = fill_level.numpy()
        after = self.updated_fill.numpy()
        active = ~self.walls.host
        volume_before = float(np.sum(before[active], dtype=np.float64))
        volume_after = float(np.sum(after[active], dtype=np.float64))
        relative_drift = (volume_after - volume_before) / max(volume_before, 1.0)
        diagnostics = PlicAxisAdvectionDiagnostics(
            volume_before,
            volume_after,
            relative_drift,
            float(np.max(np.abs(face_courant.numpy()), initial=0.0)),
            float(self._maximum_correction.numpy()[0]),
            int(self._endpoint_fallback_count.numpy()[0]),
        )
        self.last_diagnostics = diagnostics
        if invalid_count:
            raise FloatingPointError(
                f"PLIC axis-{self.axis} sweep found {invalid_count} invalid faces or cells"
            )
        if compression is None and abs(relative_drift) > 2.0e-6:
            raise FloatingPointError(
                f"PLIC axis-{self.axis} volume drift {relative_drift} exceeds 2e-6"
            )
        return self.updated_fill, diagnostics

    def advect_mass(self, mass: wp.array, fill_level: wp.array) -> wp.array:
        if tuple(mass.shape) != self.res or tuple(fill_level.shape) != self.res:
            raise ValueError("PLIC mass and fill fields must match the cell grid")
        self._invalid_count.zero_()
        wp.launch(
            advection_kernels.mass_flux_kernel,
            dim=self.face_shape,
            inputs=[
                mass,
                fill_level,
                self.face_flux,
                self.face_mass_flux,
                self._invalid_count,
                self.axis,
                int(self.periodic),
                *self.res,
            ],
            device=self.model._device,
        )
        wp.launch(
            advection_kernels.update_mass_kernel,
            dim=self.res,
            inputs=[
                mass,
                self.walls.device,
                self.face_mass_flux,
                self.updated_mass,
                self._invalid_count,
                self.axis,
            ],
            device=self.model._device,
        )
        invalid = int(self._invalid_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                f"PLIC axis-{self.axis} mass transport found {invalid} invalid entries"
            )
        before = float(np.sum(mass.numpy(), dtype=np.float64))
        after = float(np.sum(self.updated_mass.numpy(), dtype=np.float64))
        if abs(after - before) > 2.0e-6 * max(before, 1.0):
            raise FloatingPointError("PLIC mass transport is not conservative")
        return self.updated_mass

    def advect_momentum(self, momentum: wp.array, mass: wp.array) -> wp.array:
        if tuple(momentum.shape) != self.res or tuple(mass.shape) != self.res:
            raise ValueError("PLIC momentum and mass fields must match the cell grid")
        self._invalid_count.zero_()
        wp.launch(
            advection_kernels.momentum_flux_kernel,
            dim=self.face_shape,
            inputs=[
                momentum,
                mass,
                self.face_mass_flux,
                self.face_momentum_flux,
                self._invalid_count,
                self.axis,
                int(self.periodic),
                *self.res,
            ],
            device=self.model._device,
        )
        wp.launch(
            advection_kernels.update_momentum_kernel,
            dim=self.res,
            inputs=[
                momentum,
                self.walls.device,
                self.face_momentum_flux,
                self.updated_momentum,
                self._invalid_count,
                self.axis,
            ],
            device=self.model._device,
        )
        invalid = int(self._invalid_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                f"PLIC axis-{self.axis} momentum transport found {invalid} invalid entries"
            )
        before = np.sum(momentum.numpy(), axis=(0, 1, 2), dtype=np.float64)
        after = np.sum(
            self.updated_momentum.numpy(), axis=(0, 1, 2), dtype=np.float64
        )
        tolerance = 2.0e-7 + 2.0e-6 * max(float(np.max(np.abs(before))), 1.0)
        if float(np.max(np.abs(after - before))) > tolerance:
            raise FloatingPointError("PLIC momentum transport is not conservative")
        return self.updated_momentum

    def _copy_periodic_boundary_face(self) -> None:
        flux = self.face_flux.numpy()
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[self.axis] = 0
        upper[self.axis] = -1
        flux[tuple(upper)] = flux[tuple(lower)]
        self.face_flux.assign(flux)
