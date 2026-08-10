# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only-coupled PLIC-VOF shadow state for incremental A/B experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    HomeCourantProjector,
    HomeFaceCourantBuilder,
    PlicGeometricTransport,
    PlicTransportDiagnostics,
)


@dataclass(frozen=True)
class ShadowPlicVofDiagnostics:
    step: int
    split_order: tuple[int, ...]
    maximum_courant: float
    maximum_active_divergence: float
    projected_maximum_divergence: float | None
    projection_iteration_count: int
    maximum_face_correction: float
    step_relative_volume_drift: float
    cumulative_relative_volume_drift: float
    drift_limit_exceeded: bool
    drift_violation_count: int


class ShadowPlicVof:
    """Advance a PLIC fill field without writing to HOME or FSL state."""

    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        initial_fill: wp.array,
        *,
        endpoint_tolerance: float = 4.0e-7,
        project_courant: bool = False,
        projection_maximum_iterations: int = 320,
    ) -> None:
        if walls.model is not model:
            raise ValueError("shadow PLIC and walls must share one HOME model")
        if tuple(initial_fill.shape) != walls.res or initial_fill.device != model._device:
            raise ValueError("initial shadow fill must match the HOME grid and device")
        self.model = model
        self.walls = walls
        self.fill_level = wp.clone(initial_fill)
        self.builder = HomeFaceCourantBuilder(model, walls)
        self.projector = (
            HomeCourantProjector(
                model,
                walls,
                maximum_iterations=projection_maximum_iterations,
                relative_tolerance=5.0e-6,
                absolute_divergence_tolerance=5.0e-7,
            )
            if project_courant
            else None
        )
        self.transport = PlicGeometricTransport(
            model, walls, endpoint_tolerance=endpoint_tolerance
        )
        self.step_index = 0
        self.drift_violation_count = 0
        initial = self.fill_level.numpy()
        self._active = ~walls.host
        self.initial_volume = float(
            np.sum(initial[self._active], dtype=np.float64)
        )
        self.last_diagnostics: ShadowPlicVofDiagnostics | None = None

    def advance(
        self, fluid: HomeCoreState, fsl: FslState
    ) -> ShadowPlicVofDiagnostics:
        faces = self.builder.build(fluid, fsl)
        if self.projector is not None:
            faces = self.projector.project(faces, fsl)
        split_order = (
            (0, 1, 2) if self.step_index % 2 == 0 else (2, 1, 0)
        )
        drift_limit_exceeded = False
        try:
            updated, transport = self.transport.transport(
                self.fill_level, faces, split_order=split_order
            )
        except FloatingPointError as error:
            transport = self.transport.last_diagnostics
            if (
                transport is None
                or not str(error).startswith("split PLIC transport volume drift")
            ):
                raise
            # B2 is observational: preserve the bounded candidate field and
            # expose the rejected drift instead of feeding it back or hiding it.
            updated = self.transport.advectors[split_order[-1]].updated_fill
            drift_limit_exceeded = True
            self.drift_violation_count += 1
        assert isinstance(transport, PlicTransportDiagnostics)
        self.fill_level.assign(updated)
        self.step_index += 1
        courant = self.builder.last_diagnostics
        assert courant is not None
        projection = (
            self.projector.last_diagnostics
            if self.projector is not None
            else None
        )
        diagnostics = ShadowPlicVofDiagnostics(
            step=self.step_index,
            split_order=split_order,
            maximum_courant=courant.maximum_courant,
            maximum_active_divergence=courant.maximum_active_divergence,
            projected_maximum_divergence=(
                projection.projected_maximum_divergence
                if projection is not None
                else None
            ),
            projection_iteration_count=(
                projection.iteration_count if projection is not None else 0
            ),
            maximum_face_correction=(
                projection.maximum_face_correction
                if projection is not None
                else 0.0
            ),
            step_relative_volume_drift=transport.relative_volume_drift,
            cumulative_relative_volume_drift=(
                transport.volume_after - self.initial_volume
            )
            / max(self.initial_volume, 1.0),
            drift_limit_exceeded=drift_limit_exceeded,
            drift_violation_count=self.drift_violation_count,
        )
        self.last_diagnostics = diagnostics
        return diagnostics
