# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Directionally split geometric VOF transport, isolated from HOME coupling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .advection import PlicAxisAdvectionDiagnostics, PlicAxisAdvector
from .domain import PlicDomain
from .geometry import PlicGeometryReconstructor, PlicGeometryState

if TYPE_CHECKING:
    from ..core import HomeCoreModel
    from ..free_surface_fsl import FslWallMask


@dataclass(frozen=True)
class PlicTransportDiagnostics:
    split_order: tuple[int, ...]
    axis_diagnostics: tuple[PlicAxisAdvectionDiagnostics, ...]
    volume_before: float
    volume_after: float
    relative_volume_drift: float
    endpoint_snapped_cell_count: int
    endpoint_snapped_volume_delta: float


@dataclass(frozen=True)
class PlicConservativeTransportDiagnostics:
    split_order: tuple[int, ...]
    axis_diagnostics: tuple[PlicAxisAdvectionDiagnostics, ...]
    volume_drift: float
    mass_drift: float
    momentum_drift: tuple[float, float, float]


@dataclass(frozen=True)
class PlicVolumeMassTransportDiagnostics:
    split_order: tuple[int, ...]
    axis_diagnostics: tuple[PlicAxisAdvectionDiagnostics, ...]
    volume_drift: float
    mass_drift: float


class PlicGeometricTransport:
    def __init__(
        self,
        model: HomeCoreModel | PlicDomain,
        walls: FslWallMask | None = None,
        *,
        endpoint_tolerance: float = 4.0e-7,
        contact_angle_degrees: float | None = None,
    ) -> None:
        if walls is None:
            if not isinstance(model, PlicDomain):
                raise TypeError("standalone PLIC transport requires a PlicDomain")
            walls = model
        if walls.model is not model:
            raise ValueError("PLIC transport and walls must share one HOME model")
        self.model = model
        self.walls = walls
        if not 0.0 <= endpoint_tolerance < 0.5:
            raise ValueError("endpoint_tolerance must be in [0, 0.5)")
        self.endpoint_tolerance = float(endpoint_tolerance)
        self.endpoint_fallback_tolerance = max(
            4.0 * self.endpoint_tolerance,
            32.0 * float(np.finfo(np.float32).eps),
        )
        if self.endpoint_fallback_tolerance >= 0.5:
            raise ValueError(
                "endpoint_tolerance is too large for the bounded PLIC fallback"
            )
        self.geometry = PlicGeometryState(model)
        self.compression = wp.zeros(
            walls.res, dtype=wp.int32, device=model._device
        )
        self._source_copy = wp.zeros(walls.res, dtype=float, device=model._device)
        self._mass_source_copy = wp.zeros(
            walls.res, dtype=float, device=model._device
        )
        self._momentum_source_copy = wp.zeros(
            walls.res, dtype=wp.vec3, device=model._device
        )
        self.reconstructor = PlicGeometryReconstructor(
            model,
            walls,
            contact_angle_degrees=contact_angle_degrees,
        )
        self.advectors = tuple(PlicAxisAdvector(model, walls, axis=axis) for axis in range(3))
        self.last_diagnostics: PlicTransportDiagnostics | None = None

    def transport(
        self,
        fill_level: wp.array,
        face_courant: tuple[wp.array, wp.array, wp.array],
        *,
        split_order: tuple[int, ...],
    ) -> tuple[wp.array, PlicTransportDiagnostics]:
        self._validate_split_order(split_order)
        source = fill_level.numpy()
        active = ~self.walls.host
        volume_before = float(np.sum(source[active], dtype=np.float64))
        self._source_copy.assign(source)
        self.compression.assign((source >= 0.5).astype(np.int32))
        current = self._source_copy
        axis_diagnostics: list[PlicAxisAdvectionDiagnostics] = []
        snapped_cell_count = 0
        snapped_volume_delta = 0.0
        for axis in split_order:
            geometry_diagnostics = self.reconstructor.reconstruct_fill(
                current,
                self.geometry,
                interface_epsilon=self.endpoint_tolerance,
                endpoint_fallback_tolerance=self.endpoint_fallback_tolerance,
            )
            current, diagnostics = self.advectors[axis].advect(
                current,
                self.geometry,
                face_courant[axis],
                self.compression,
                interface_epsilon=self.endpoint_tolerance,
                endpoint_fallback_tolerance=self.endpoint_fallback_tolerance,
            )
            diagnostics = replace(
                diagnostics,
                wetting_cell_count=geometry_diagnostics.wetting_cell_count,
                parallel_wall_interface_cell_count=(
                    geometry_diagnostics.parallel_wall_interface_cell_count
                ),
            )
            axis_diagnostics.append(diagnostics)
            current_values = current.numpy()
            canonical = current_values.copy()
            snap_empty = (canonical > 0.0) & (
                canonical <= self.endpoint_tolerance
            )
            snap_full = (canonical < 1.0) & (
                canonical >= 1.0 - self.endpoint_tolerance
            )
            snapped_cell_count += int(np.count_nonzero(snap_empty | snap_full))
            before_snap = float(np.sum(canonical[active], dtype=np.float64))
            canonical[snap_empty] = 0.0
            canonical[snap_full] = 1.0
            after_snap = float(np.sum(canonical[active], dtype=np.float64))
            snapped_volume_delta += after_snap - before_snap
            current.assign(canonical)
        volume_after = float(np.sum(current.numpy()[active], dtype=np.float64))
        relative_drift = (volume_after - volume_before) / max(volume_before, 1.0)
        diagnostics = PlicTransportDiagnostics(
            split_order,
            tuple(axis_diagnostics),
            volume_before,
            volume_after,
            relative_drift,
            snapped_cell_count,
            snapped_volume_delta,
        )
        self.last_diagnostics = diagnostics
        if abs(relative_drift) > 4.0e-6:
            raise FloatingPointError(
                f"split PLIC transport volume drift {relative_drift} exceeds 4e-6"
            )
        return current, diagnostics

    def transport_conservative(
        self,
        fill_level: wp.array,
        mass: wp.array,
        momentum: wp.array,
        face_courant: tuple[wp.array, wp.array, wp.array],
        *,
        split_order: tuple[int, ...],
    ) -> tuple[
        wp.array,
        wp.array,
        wp.array,
        PlicConservativeTransportDiagnostics,
    ]:
        """Transport volume, mass, and momentum through identical PLIC face slabs."""

        self._validate_split_order(split_order)
        if tuple(fill_level.shape) != self.walls.res:
            raise ValueError("PLIC fill must match the cell grid")
        if tuple(mass.shape) != self.walls.res or tuple(momentum.shape) != self.walls.res:
            raise ValueError("PLIC mass and momentum must match the cell grid")
        if momentum.dtype != wp.vec3:
            raise TypeError("PLIC momentum must use wp.vec3")

        active = ~self.walls.host
        initial_fill = fill_level.numpy()
        initial_mass = mass.numpy()
        initial_momentum = momentum.numpy()
        volume_before = float(np.sum(initial_fill[active], dtype=np.float64))
        mass_before = float(np.sum(initial_mass[active], dtype=np.float64))
        momentum_before = np.sum(
            initial_momentum[active], axis=0, dtype=np.float64
        )
        self._source_copy.assign(initial_fill)
        self._mass_source_copy.assign(initial_mass)
        self._momentum_source_copy.assign(initial_momentum)
        self.compression.assign((initial_fill >= 0.5).astype(np.int32))

        current_fill = self._source_copy
        current_mass = self._mass_source_copy
        current_momentum = self._momentum_source_copy
        axis_diagnostics: list[PlicAxisAdvectionDiagnostics] = []
        for axis in split_order:
            # Sub-tolerance tails retain their ledgers between split sweeps, but
            # are not geometrically reconstructable until topology commit.
            geometry_diagnostics = self.reconstructor.reconstruct_fill(
                current_fill,
                self.geometry,
                interface_epsilon=self.endpoint_tolerance,
                endpoint_fallback_tolerance=self.endpoint_fallback_tolerance,
            )
            advector = self.advectors[axis]
            next_fill, diagnostics = advector.advect(
                current_fill,
                self.geometry,
                face_courant[axis],
                self.compression,
                interface_epsilon=self.endpoint_tolerance,
                endpoint_fallback_tolerance=self.endpoint_fallback_tolerance,
            )
            diagnostics = replace(
                diagnostics,
                wetting_cell_count=geometry_diagnostics.wetting_cell_count,
                parallel_wall_interface_cell_count=(
                    geometry_diagnostics.parallel_wall_interface_cell_count
                ),
            )
            next_mass = advector.advect_mass(current_mass, current_fill)
            next_momentum = advector.advect_momentum(
                current_momentum, current_mass
            )
            current_fill = next_fill
            current_mass = next_mass
            current_momentum = next_momentum
            axis_diagnostics.append(diagnostics)

        final_fill = current_fill.numpy()
        final_mass = current_mass.numpy()
        final_momentum = current_momentum.numpy()
        volume_drift = (
            float(np.sum(final_fill[active], dtype=np.float64)) - volume_before
        )
        mass_drift = (
            float(np.sum(final_mass[active], dtype=np.float64)) - mass_before
        )
        momentum_drift_array = (
            np.sum(final_momentum[active], axis=0, dtype=np.float64)
            - momentum_before
        )
        diagnostics = PlicConservativeTransportDiagnostics(
            tuple(split_order),
            tuple(axis_diagnostics),
            volume_drift,
            mass_drift,
            tuple(float(value) for value in momentum_drift_array),
        )
        volume_scale = max(volume_before, 1.0)
        mass_scale = max(abs(mass_before), 1.0)
        momentum_scale = max(float(np.max(np.abs(momentum_before))), 1.0)
        if abs(volume_drift) > 4.0e-6 * volume_scale:
            raise FloatingPointError("conservative PLIC volume drift exceeds 4e-6")
        if abs(mass_drift) > 2.0e-6 * mass_scale:
            raise FloatingPointError("conservative PLIC mass drift exceeds 2e-6")
        if float(np.max(np.abs(momentum_drift_array))) > 2.0e-6 * momentum_scale:
            raise FloatingPointError("conservative PLIC momentum drift exceeds 2e-6")
        return current_fill, current_mass, current_momentum, diagnostics

    def transport_volume_mass(
        self,
        fill_level: wp.array,
        mass: wp.array,
        face_courant: tuple[wp.array, wp.array, wp.array],
        *,
        split_order: tuple[int, ...],
        use_compression: bool = True,
    ) -> tuple[wp.array, wp.array, PlicVolumeMassTransportDiagnostics]:
        """Transport volume and mass without introducing a momentum ledger."""

        self._validate_split_order(split_order)
        if tuple(fill_level.shape) != self.walls.res:
            raise ValueError("PLIC fill must match the cell grid")
        if tuple(mass.shape) != self.walls.res:
            raise ValueError("PLIC mass must match the cell grid")

        active = ~self.walls.host
        initial_fill = fill_level.numpy()
        initial_mass = mass.numpy()
        volume_before = float(np.sum(initial_fill[active], dtype=np.float64))
        mass_before = float(np.sum(initial_mass[active], dtype=np.float64))
        self._source_copy.assign(initial_fill)
        self._mass_source_copy.assign(initial_mass)
        self.compression.assign((initial_fill >= 0.5).astype(np.int32))

        current_fill = self._source_copy
        current_mass = self._mass_source_copy
        axis_diagnostics: list[PlicAxisAdvectionDiagnostics] = []
        for axis in split_order:
            geometry_diagnostics = self.reconstructor.reconstruct_fill(
                current_fill,
                self.geometry,
                interface_epsilon=self.endpoint_tolerance,
                endpoint_fallback_tolerance=self.endpoint_fallback_tolerance,
            )
            advector = self.advectors[axis]
            next_fill, diagnostics = advector.advect(
                current_fill,
                self.geometry,
                face_courant[axis],
                self.compression if use_compression else None,
                interface_epsilon=self.endpoint_tolerance,
                endpoint_fallback_tolerance=self.endpoint_fallback_tolerance,
            )
            diagnostics = replace(
                diagnostics,
                wetting_cell_count=geometry_diagnostics.wetting_cell_count,
                parallel_wall_interface_cell_count=(
                    geometry_diagnostics.parallel_wall_interface_cell_count
                ),
            )
            current_mass = advector.advect_mass(current_mass, current_fill)
            current_fill = next_fill
            axis_diagnostics.append(diagnostics)

        volume_drift = (
            float(np.sum(current_fill.numpy()[active], dtype=np.float64))
            - volume_before
        )
        mass_drift = (
            float(np.sum(current_mass.numpy()[active], dtype=np.float64))
            - mass_before
        )
        diagnostics = PlicVolumeMassTransportDiagnostics(
            tuple(split_order),
            tuple(axis_diagnostics),
            volume_drift,
            mass_drift,
        )
        if abs(volume_drift) > 4.0e-6 * max(volume_before, 1.0):
            raise FloatingPointError("PLIC volume transport drift exceeds 4e-6")
        if abs(mass_drift) > 2.0e-6 * max(abs(mass_before), 1.0):
            raise FloatingPointError("PLIC mass transport drift exceeds 2e-6")
        return current_fill, current_mass, diagnostics

    @staticmethod
    def _validate_split_order(split_order: tuple[int, ...]) -> None:
        if len(split_order) < 2 or len(set(split_order)) != len(split_order):
            raise ValueError("split_order must contain at least two distinct axes")
        if any(axis not in (0, 1, 2) for axis in split_order):
            raise ValueError("split_order axes must be 0, 1, or 2")
