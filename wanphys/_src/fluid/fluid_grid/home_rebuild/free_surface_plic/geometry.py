# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional Warp owner for isolated PLIC reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from . import geometry_kernels
from .domain import PlicDomain

if TYPE_CHECKING:
    from ..core import HomeCoreModel
    from ..free_surface_fsl import FslState, FslWallMask


@dataclass(frozen=True)
class PlicGeometryDiagnostics:
    interface_cell_count: int
    invalid_interface_count: int
    maximum_interface_area: float
    endpoint_fallback_cell_count: int = 0
    wetting_cell_count: int = 0
    invalid_wall_normal_count: int = 0
    parallel_wall_interface_cell_count: int = 0


class PlicGeometryState:
    def __init__(self, model: HomeCoreModel | PlicDomain) -> None:
        self.model = model
        self.res = (
            model.res
            if isinstance(model, PlicDomain)
            else (int(model.nx), int(model.ny), int(model.nz))
        )
        self.normal = wp.zeros(self.res, dtype=wp.vec3, device=model._device)
        self.plane_offset = wp.zeros(self.res, dtype=float, device=model._device)
        self.interface_area = wp.zeros(self.res, dtype=float, device=model._device)
        self.valid = wp.zeros(self.res, dtype=wp.int32, device=model._device)


class PlicGeometryReconstructor:
    def __init__(
        self,
        model: HomeCoreModel | PlicDomain,
        walls: FslWallMask | None = None,
        *,
        contact_angle_degrees: float | None = None,
    ) -> None:
        if walls is None:
            if not isinstance(model, PlicDomain):
                raise TypeError("standalone PLIC reconstruction requires a PlicDomain")
            walls = model
        if walls.model is not model:
            raise ValueError("PLIC geometry and walls must share one HOME model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        if contact_angle_degrees is not None and (
            not math.isfinite(contact_angle_degrees)
            or not 0.0 < contact_angle_degrees < 180.0
        ):
            raise ValueError(
                "contact_angle_degrees must be finite and in (0, 180)"
            )
        self.contact_angle_degrees = (
            None
            if contact_angle_degrees is None
            else float(contact_angle_degrees)
        )
        self._invalid_count = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._endpoint_fallback_count = wp.zeros(
            1, dtype=wp.int32, device=model._device
        )
        self._wetting_count = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._invalid_wall_normal_count = wp.zeros(
            1, dtype=wp.int32, device=model._device
        )
        self._parallel_wall_interface_count = wp.zeros(
            1, dtype=wp.int32, device=model._device
        )
        self._unused_flags = wp.zeros(self.res, dtype=wp.int32, device=model._device)
        self.last_diagnostics: PlicGeometryDiagnostics | None = None

    def reconstruct(
        self, fsl_state: FslState, geometry: PlicGeometryState
    ) -> PlicGeometryDiagnostics:
        if fsl_state.model is not self.model or geometry.model is not self.model:
            raise ValueError("PLIC geometry, FSL state, and HOME model must match")
        return self._reconstruct(
            fsl_state.fill_level,
            fsl_state.flags,
            geometry,
            derive_interface=False,
            interface_epsilon=0.0,
            endpoint_fallback_tolerance=0.0,
        )

    def reconstruct_fill(
        self,
        fill_level: wp.array,
        geometry: PlicGeometryState,
        *,
        interface_epsilon: float = 0.0,
        endpoint_fallback_tolerance: float | None = None,
    ) -> PlicGeometryDiagnostics:
        if tuple(fill_level.shape) != self.res or geometry.model is not self.model:
            raise ValueError("PLIC fill reconstruction fields must match the model")
        if not 0.0 <= interface_epsilon < 0.5:
            raise ValueError("interface_epsilon must be in [0, 0.5)")
        if endpoint_fallback_tolerance is None:
            endpoint_fallback_tolerance = interface_epsilon
        if not interface_epsilon <= endpoint_fallback_tolerance < 0.5:
            raise ValueError(
                "endpoint_fallback_tolerance must be in "
                "[interface_epsilon, 0.5)"
            )
        return self._reconstruct(
            fill_level,
            self._unused_flags,
            geometry,
            derive_interface=True,
            interface_epsilon=interface_epsilon,
            endpoint_fallback_tolerance=endpoint_fallback_tolerance,
        )

    def _reconstruct(
        self,
        fill_level: wp.array,
        flags: wp.array,
        geometry: PlicGeometryState,
        *,
        derive_interface: bool,
        interface_epsilon: float,
        endpoint_fallback_tolerance: float,
    ) -> PlicGeometryDiagnostics:
        self._invalid_count.zero_()
        self._endpoint_fallback_count.zero_()
        self._wetting_count.zero_()
        self._invalid_wall_normal_count.zero_()
        self._parallel_wall_interface_count.zero_()
        closed = tuple(int(value) for value in self.walls.closed_axes)
        theta = (
            0.0
            if self.contact_angle_degrees is None
            else math.radians(self.contact_angle_degrees)
        )
        wp.launch(
            geometry_kernels.reconstruct_geometry_kernel,
            dim=self.res,
            inputs=[
                fill_level,
                flags,
                self.walls.device,
                geometry.normal,
                geometry.plane_offset,
                geometry.interface_area,
                geometry.valid,
                self._invalid_count,
                self._endpoint_fallback_count,
                self._wetting_count,
                self._invalid_wall_normal_count,
                self._parallel_wall_interface_count,
                closed[0],
                closed[1],
                closed[2],
                self.res[0],
                self.res[1],
                self.res[2],
                int(derive_interface),
                interface_epsilon,
                endpoint_fallback_tolerance,
                int(self.contact_angle_degrees is not None),
                math.cos(theta),
                math.sin(theta),
            ],
            device=self.model._device,
        )
        invalid_count = int(self._invalid_count.numpy()[0])
        endpoint_fallback_count = int(self._endpoint_fallback_count.numpy()[0])
        wetting_count = int(self._wetting_count.numpy()[0])
        invalid_wall_normal_count = int(
            self._invalid_wall_normal_count.numpy()[0]
        )
        parallel_wall_interface_count = int(
            self._parallel_wall_interface_count.numpy()[0]
        )
        valid = geometry.valid.numpy()
        areas = geometry.interface_area.numpy()
        diagnostics = PlicGeometryDiagnostics(
            interface_cell_count=int(np.count_nonzero(valid)),
            invalid_interface_count=invalid_count,
            maximum_interface_area=float(np.max(areas, initial=0.0)),
            endpoint_fallback_cell_count=endpoint_fallback_count,
            wetting_cell_count=wetting_count,
            invalid_wall_normal_count=invalid_wall_normal_count,
            parallel_wall_interface_cell_count=parallel_wall_interface_count,
        )
        self.last_diagnostics = diagnostics
        if invalid_wall_normal_count:
            raise FloatingPointError(
                "PLIC wetting found "
                f"{invalid_wall_normal_count} undefined wall normals"
            )
        if invalid_count:
            fill = fill_level.numpy()
            undefined = np.argwhere(
                (fill > interface_epsilon)
                & (fill < 1.0 - interface_epsilon)
                & (valid == 0)
                & (~self.walls.host)
            )
            examples = [
                (tuple(int(value) for value in index), float(fill[tuple(index)]))
                for index in undefined[:4]
            ]
            raise FloatingPointError(
                f"PLIC reconstruction found {invalid_count} undefined interface normals; "
                f"examples={examples}"
            )
        return diagnostics
