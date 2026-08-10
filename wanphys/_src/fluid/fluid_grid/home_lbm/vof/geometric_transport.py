# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared frozen-velocity geometric VOF transaction."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import geometric_advection_kernels
from .courant_projection import (
    HomeFreeCourantProjectionDiagnostics,
    HomeFreeCourantProjector,
)
from .fsl_courant import FslFaceCourantDiagnostics, HomeFreeFslFaceCourantBuilder
from .geometric_advection import (
    HomeFreeGeometricAxisAdvector,
    PlicAxisAdvectionDiagnostics,
)
from .geometric_topology import (
    GeometricPlicTopologyDiagnostics,
    HomeFreeGeometricTopologyResolver,
)
from .geometry import HomeFreeInterfaceGeometry
from .state import HomeFreeState


@dataclass(frozen=True)
class HomeFreeGeometricTransportResult:
    """Committed intermediate state and diagnostics for one split VOF step."""

    final_state: HomeFreeState
    final_momentum: wp.array
    face_courant_by_axis: tuple[tuple[int, FslFaceCourantDiagnostics], ...]
    advection_by_axis: tuple[tuple[int, PlicAxisAdvectionDiagnostics], ...]
    topology_by_axis: tuple[tuple[int, GeometricPlicTopologyDiagnostics], ...]
    split_order: tuple[int, ...]
    projection: HomeFreeCourantProjectionDiagnostics | None


class HomeFreeGeometricTransport:
    """Advect PLIC volume with one frozen velocity field for all split axes."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        axes: tuple[int, ...],
        contact_angle_degrees: float | None = None,
        project_courant: bool = False,
        projection_max_iterations: int | None = None,
    ) -> None:
        if not axes or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("geometric transport axes must use 0, 1, or 2")
        if len(set(axes)) != len(axes):
            raise ValueError("geometric transport axes must not repeat")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.device = model._device
        self.axes = tuple(int(axis) for axis in axes)
        if project_courant and set(self.axes) != {0, 1, 2}:
            raise ValueError("Courant projection requires all three transport axes")
        self.geometry = HomeFreeInterfaceGeometry(
            model, contact_angle_degrees=contact_angle_degrees
        )
        self.advectors = {
            axis: HomeFreeGeometricAxisAdvector(model, axis=axis)
            for axis in self.axes
        }
        self.face_courant_builders = {
            axis: HomeFreeFslFaceCourantBuilder(model, axis=axis)
            for axis in self.axes
        }
        self.topology_resolver = HomeFreeGeometricTopologyResolver(model)
        self.scratch_states = (HomeFreeState(model), HomeFreeState(model))
        self.initial_momentum = wp.zeros(
            self.res, dtype=wp.vec3, device=self.device
        )
        self.scratch_momentum = (
            wp.zeros(self.res, dtype=wp.vec3, device=self.device),
            wp.zeros(self.res, dtype=wp.vec3, device=self.device),
        )
        self._invalid_momentum = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self.compression = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        maximum_projection_iterations = (
            max(160, 8 * max(self.res))
            if projection_max_iterations is None
            else int(projection_max_iterations)
        )
        self.courant_projector = (
            HomeFreeCourantProjector(
                model,
                max_iterations=maximum_projection_iterations,
                require_projected_divergence_limit=False,
            )
            if project_courant
            else None
        )
        self.courant_refinement_projector = (
            HomeFreeCourantProjector(
                model,
                max_iterations=maximum_projection_iterations,
                relative_tolerance=2.0e-6,
            )
            if project_courant
            else None
        )

    def transport(
        self,
        fluid: HomeLbmState,
        source: HomeFreeState,
        active: wp.array,
        *,
        split_order: tuple[int, ...],
        face_courant: wp.array | None = None,
    ) -> HomeFreeGeometricTransportResult:
        """Run all directional sweeps without changing the caller's source state."""

        if tuple(split_order) not in (self.axes, tuple(reversed(self.axes))):
            raise ValueError("split_order must be the configured axes or its reverse")
        if tuple(active.shape) != self.res or active.device != self.device:
            raise ValueError("geometric transport active mask must match the model")
        if active.dtype != wp.int32:
            raise TypeError("geometric transport active mask must use wp.int32")
        if face_courant is not None and len(split_order) != 1:
            raise ValueError(
                "explicit face_courant is only valid for a single-axis transport"
            )
        if face_courant is not None and self.courant_projector is not None:
            raise ValueError("explicit face_courant cannot be projected")

        self.geometry.update(
            source, solid_phi=fluid.solid_phi, compute_curvature=False
        )
        self._invalid_momentum.zero_()
        wp.launch(
            geometric_advection_kernels.initialize_liquid_momentum_kernel,
            dim=self.res,
            inputs=[
                fluid.moments,
                source.mass,
                source.flags,
                self.initial_momentum,
                self._invalid_momentum,
                int(self.model.ny),
                int(self.model.nz),
                fluid.cell_count,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid_momentum = int(self._invalid_momentum.numpy()[0])
        if invalid_momentum:
            raise RuntimeError(
                "geometric transport momentum initialization found "
                f"{invalid_momentum} invalid cells"
            )
        wp.launch(
            geometric_advection_kernels.classify_weymouth_yue_compression_kernel,
            dim=self.res,
            inputs=[source.fill_level, self.compression],
            device=self.device,
        )
        frozen_courant_by_axis: dict[int, wp.array] = {}
        courant_diagnostics: list[tuple[int, FslFaceCourantDiagnostics]] = []
        if face_courant is not None:
            frozen_courant_by_axis[split_order[0]] = face_courant
        else:
            for axis in split_order:
                builder = self.face_courant_builders[axis]
                frozen_courant_by_axis[axis] = builder.build(
                    fluid, active, source.fill_level, source.flags
                )
                assert builder.last_diagnostics is not None
                courant_diagnostics.append((axis, builder.last_diagnostics))
        projection_diagnostics = None
        if self.courant_projector is not None:
            projected = self.courant_projector.project(
                (
                    frozen_courant_by_axis[0],
                    frozen_courant_by_axis[1],
                    frozen_courant_by_axis[2],
                ),
                source.flags,
            )
            if (
                projected.diagnostics.projected_max_divergence
                > self.courant_projector.absolute_divergence_tolerance
            ):
                assert self.courant_refinement_projector is not None
                primary_diagnostics = projected.diagnostics
                projected = self.courant_refinement_projector.project(
                    projected.face_courant,
                    source.flags,
                )
                refinement_diagnostics = projected.diagnostics
                projection_diagnostics = HomeFreeCourantProjectionDiagnostics(
                    active_cell_count=primary_diagnostics.active_cell_count,
                    corrected_face_count=primary_diagnostics.corrected_face_count,
                    iteration_count=(
                        primary_diagnostics.iteration_count
                        + refinement_diagnostics.iteration_count
                    ),
                    initial_max_divergence=(
                        primary_diagnostics.initial_max_divergence
                    ),
                    projected_max_divergence=(
                        refinement_diagnostics.projected_max_divergence
                    ),
                    relative_residual=primary_diagnostics.relative_residual,
                    maximum_face_correction=(
                        primary_diagnostics.maximum_face_correction
                        + refinement_diagnostics.maximum_face_correction
                    ),
                    refinement_iteration_count=(
                        refinement_diagnostics.iteration_count
                    ),
                )
            else:
                projection_diagnostics = projected.diagnostics
            frozen_courant_by_axis = {
                axis: projected.face_courant[axis] for axis in self.axes
            }

        topology_diagnostics: list[
            tuple[int, GeometricPlicTopologyDiagnostics]
        ] = []
        advection_diagnostics: list[
            tuple[int, PlicAxisAdvectionDiagnostics]
        ] = []
        current = source
        current_momentum = self.initial_momentum
        for sweep_index, axis in enumerate(split_order):
            if sweep_index:
                self.geometry.update(
                    current, solid_phi=fluid.solid_phi, compute_curvature=False
                )
            updated_fill = self.advectors[axis].advect(
                current.fill_level,
                self.geometry.normal,
                self.geometry.plane_offset,
                frozen_courant_by_axis[axis],
                self.compression,
                synchronize=False,
            )
            updated_mass = self.advectors[axis].advect_mass(
                current.mass,
                current.fill_level,
                synchronize=False,
            )
            updated_momentum = self.advectors[axis].advect_momentum(
                current_momentum,
                current.mass,
                synchronize=False,
            )
            axis_diagnostics = self.advectors[axis].finalize_diagnostics()
            advection_diagnostics.append((axis, axis_diagnostics))
            scratch = self.scratch_states[sweep_index % 2]
            try:
                topology = self.topology_resolver.resolve(
                    fluid,
                    current,
                    updated_fill,
                    scratch,
                    transported_mass=updated_mass,
                )
            except Exception as error:
                raise RuntimeError(
                    "geometric VOF topology failed after "
                    f"axis {axis} sweep {sweep_index + 1}/{len(split_order)}"
                ) from error
            topology_diagnostics.append((axis, topology))
            next_momentum = self.scratch_momentum[sweep_index % 2]
            self._invalid_momentum.zero_()
            wp.launch(
                geometric_advection_kernels.reconcile_transported_momentum_kernel,
                dim=self.res,
                inputs=[
                    updated_momentum,
                    updated_mass,
                    scratch.mass,
                    scratch.flags,
                    fluid.moments,
                    next_momentum,
                    self._invalid_momentum,
                    int(self.model.ny),
                    int(self.model.nz),
                    fluid.cell_count,
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            invalid_momentum = int(self._invalid_momentum.numpy()[0])
            if invalid_momentum:
                raise RuntimeError(
                    "geometric transport momentum reconciliation found "
                    f"{invalid_momentum} invalid cells after axis {axis}"
                )
            current = scratch
            current_momentum = next_momentum

        return HomeFreeGeometricTransportResult(
            final_state=current,
            final_momentum=current_momentum,
            face_courant_by_axis=tuple(courant_diagnostics),
            advection_by_axis=tuple(advection_diagnostics),
            topology_by_axis=tuple(topology_diagnostics),
            split_order=tuple(split_order),
            projection=projection_diagnostics,
        )
