# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated curvature owner for reconstructed PLIC interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from . import curvature_kernels
from .domain import PlicDomain
from .geometry import PlicGeometryState

if TYPE_CHECKING:
    from ..core import HomeCoreModel
    from ..free_surface_fsl import FslWallMask


@dataclass(frozen=True)
class PlicCurvatureDiagnostics:
    required_cell_count: int
    valid_cell_count: int
    insufficient_neighbor_count: int
    invalid_curvature_count: int
    ill_conditioned_count: int = 0
    wall_contact_skipped_count: int = 0
    wall_contact_fitted_count: int = 0


class PlicCurvatureState:
    def __init__(self, model: HomeCoreModel | PlicDomain) -> None:
        res = (
            model.res
            if isinstance(model, PlicDomain)
            else (int(model.nx), int(model.ny), int(model.nz))
        )
        self.model = model
        self.curvature = wp.zeros(res, dtype=float, device=model._device)
        self.valid = wp.zeros(res, dtype=wp.int32, device=model._device)
        self.required = wp.zeros(res, dtype=wp.int32, device=model._device)


class PlicCurvatureEstimator:
    """Fit mean curvature for a 2-D interface extruded through periodic depth."""

    def __init__(
        self,
        model: HomeCoreModel | PlicDomain,
        walls: FslWallMask | None = None,
    ) -> None:
        if walls is None:
            if not isinstance(model, PlicDomain):
                raise TypeError("standalone PLIC curvature requires a PlicDomain")
            walls = model
        if walls.model is not model:
            raise ValueError("PLIC curvature and walls must share one model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        if self.res[2] != 1 or walls.closed_axes[2]:
            raise ValueError(
                "extruded PLIC curvature requires nz=1 with periodic depth"
            )
        self._counts = wp.zeros(3, dtype=wp.int32, device=model._device)
        self.last_diagnostics: PlicCurvatureDiagnostics | None = None

    def reconstruct(
        self,
        fill_level: wp.array,
        flags: wp.array,
        geometry: PlicGeometryState,
        curvature: PlicCurvatureState,
    ) -> PlicCurvatureDiagnostics:
        if geometry.model is not self.model or curvature.model is not self.model:
            raise ValueError("PLIC geometry and curvature must share one model")
        if tuple(fill_level.shape) != self.res or tuple(flags.shape) != self.res:
            raise ValueError("PLIC curvature fields must match the model")
        if flags.dtype != wp.int32:
            raise TypeError("PLIC curvature flags must use int32")
        self._counts.zero_()
        wp.launch(
            curvature_kernels.reconstruct_extruded_curvature_kernel,
            dim=self.res,
            inputs=[
                fill_level,
                flags,
                self.walls.device,
                geometry.normal,
                geometry.plane_offset,
                geometry.valid,
                curvature.curvature,
                curvature.valid,
                curvature.required,
                self._counts,
                int(self.walls.closed_axes[0]),
                int(self.walls.closed_axes[1]),
                self.res[0],
                self.res[1],
            ],
            device=self.model._device,
        )
        counts = self._counts.numpy()
        valid_count = int(np.count_nonzero(curvature.valid.numpy()))
        diagnostics = PlicCurvatureDiagnostics(
            required_cell_count=int(counts[0]),
            valid_cell_count=valid_count,
            insufficient_neighbor_count=int(counts[1]),
            invalid_curvature_count=int(counts[2]),
        )
        self.last_diagnostics = diagnostics
        if diagnostics.insufficient_neighbor_count:
            raise FloatingPointError(
                "PLIC curvature found "
                f"{diagnostics.insufficient_neighbor_count} interfaces without "
                "a resolved tangential neighbor"
            )
        if diagnostics.invalid_curvature_count:
            raise FloatingPointError(
                "PLIC curvature found "
                f"{diagnostics.invalid_curvature_count} invalid values"
            )
        return diagnostics


class PlicCurvatureEstimator3D:
    """Fit bulk 3-D curvature while leaving wall contact to wetting support."""

    def __init__(
        self,
        model: HomeCoreModel | PlicDomain,
        walls: FslWallMask | None = None,
        *,
        max_condition_number: float = 1.0e3,
        strict_bulk: bool = True,
        fit_wall_contact: bool = False,
    ) -> None:
        if walls is None:
            if not isinstance(model, PlicDomain):
                raise TypeError("standalone PLIC curvature requires a PlicDomain")
            walls = model
        if walls.model is not model:
            raise ValueError("PLIC curvature and walls must share one model")
        if not np.isfinite(max_condition_number) or max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be greater than one")
        self.model = model
        self.walls = walls
        self.res = walls.res
        if any(value < 3 for value in self.res):
            raise ValueError("3-D PLIC curvature requires at least 3 cells per axis")
        self.max_condition_number = float(max_condition_number)
        self.strict_bulk = bool(strict_bulk)
        self.fit_wall_contact = bool(fit_wall_contact)
        self._counts = wp.zeros(6, dtype=wp.int32, device=model._device)
        self.last_diagnostics: PlicCurvatureDiagnostics | None = None

    def reconstruct(
        self,
        fill_level: wp.array,
        flags: wp.array,
        geometry: PlicGeometryState,
        curvature: PlicCurvatureState,
    ) -> PlicCurvatureDiagnostics:
        if geometry.model is not self.model or curvature.model is not self.model:
            raise ValueError("PLIC geometry and curvature must share one model")
        if tuple(fill_level.shape) != self.res or tuple(flags.shape) != self.res:
            raise ValueError("PLIC curvature fields must match the model")
        if flags.dtype != wp.int32:
            raise TypeError("PLIC curvature flags must use int32")
        self._counts.zero_()
        closed = tuple(int(value) for value in self.walls.closed_axes)
        wp.launch(
            curvature_kernels.reconstruct_bulk_curvature_3d_kernel,
            dim=self.res,
            inputs=[
                fill_level,
                flags,
                self.walls.device,
                geometry.normal,
                geometry.plane_offset,
                geometry.valid,
                curvature.curvature,
                curvature.valid,
                curvature.required,
                self._counts,
                self.max_condition_number,
                closed[0],
                closed[1],
                closed[2],
                int(self.fit_wall_contact),
                self.res[0],
                self.res[1],
                self.res[2],
            ],
            device=self.model._device,
        )
        counts = self._counts.numpy()
        diagnostics = PlicCurvatureDiagnostics(
            required_cell_count=int(counts[0]),
            valid_cell_count=int(np.count_nonzero(curvature.valid.numpy())),
            insufficient_neighbor_count=int(counts[1]),
            invalid_curvature_count=int(counts[3]),
            ill_conditioned_count=int(counts[2]),
            wall_contact_skipped_count=int(counts[4]),
            wall_contact_fitted_count=int(counts[5]),
        )
        self.last_diagnostics = diagnostics
        unresolved_bulk = (
            diagnostics.insufficient_neighbor_count
            + diagnostics.ill_conditioned_count
        )
        if self.strict_bulk and unresolved_bulk:
            raise FloatingPointError(
                "3-D PLIC curvature left "
                f"{unresolved_bulk} bulk interfaces unresolved"
            )
        if diagnostics.invalid_curvature_count:
            raise FloatingPointError(
                "3-D PLIC curvature found "
                f"{diagnostics.invalid_curvature_count} invalid values"
            )
        return diagnostics
