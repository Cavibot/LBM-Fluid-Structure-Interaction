# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSL with projected PLIC volume/mass transport and baseline topology."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import HomeCoreDiagnostics, HomeCoreModel, HomeCoreState
from ..free_surface_fsl import (
    FslState,
    FslWallActiveCollider,
    FslWallMask,
    FslWallOnlyMissingStreamer,
    FslWallStreamDiagnostics,
    FslWallTopologyUpdater,
    FslWarpTopologyDiagnostics,
)
from .courant import (
    CourantProjectionDiagnostics,
    FaceCourantDiagnostics,
    HomeCourantProjector,
    HomeFaceCourantBuilder,
)
from .transport import (
    PlicGeometricTransport,
    PlicVolumeMassTransportDiagnostics,
)


@dataclass(frozen=True)
class ProjectedGvofFslWallDiagnostics:
    stream: FslWallStreamDiagnostics
    fluid: HomeCoreDiagnostics
    face_courant: FaceCourantDiagnostics
    projection: CourantProjectionDiagnostics
    transport: PlicVolumeMassTransportDiagnostics
    topology: FslWarpTopologyDiagnostics


class ProjectedGvofFslWallStepper:
    """Add only face-Courant projection to the GVOF-only composition."""

    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        axes: tuple[int, ...] = (0, 1),
        gas_density: float = 1.0,
        projection_maximum_iterations: int = 320,
        projection_relative_tolerance: float = 5.0e-6,
        projection_absolute_divergence_tolerance: float = 5.0e-7,
        transition_tolerance: float = 0.0,
        relative_mass_drift_tolerance: float = 5.0e-6,
        interface_roundoff_tolerance: float = 2.0e-6,
    ) -> None:
        if walls.model is not model:
            raise ValueError("projected GVOF stepper and walls must share one model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.axes = tuple(axes)
        self.builder = HomeFaceCourantBuilder(model, walls)
        self.projector = HomeCourantProjector(
            model,
            walls,
            maximum_iterations=projection_maximum_iterations,
            relative_tolerance=projection_relative_tolerance,
            absolute_divergence_tolerance=projection_absolute_divergence_tolerance,
        )
        self.transport = PlicGeometricTransport(
            model,
            walls,
            endpoint_tolerance=interface_roundoff_tolerance,
        )
        self.streamer = FslWallOnlyMissingStreamer(
            model, walls, gas_density=gas_density
        )
        self.collider = FslWallActiveCollider(model, walls)
        self.topology = FslWallTopologyUpdater(
            model,
            walls,
            transition_tolerance=transition_tolerance,
            relative_mass_drift_tolerance=relative_mass_drift_tolerance,
        )
        self._collided = HomeCoreState(model)
        self._candidate_fluid = HomeCoreState(model)
        self._candidate_fsl = FslState(model)
        self._step_index = 0
        self.last_diagnostics: ProjectedGvofFslWallDiagnostics | None = None

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
    ) -> ProjectedGvofFslWallDiagnostics:
        states = (fluid_in, fsl_in, fluid_out, fsl_out)
        if any(state.model is not self.model for state in states):
            raise ValueError("projected GVOF stepper and states must share one model")
        self.model.validate_step(dt)
        self.last_diagnostics = None
        split_order = (
            self.axes
            if self._step_index % 2 == 0
            else tuple(reversed(self.axes))
        )
        faces = self.builder.build(fluid_in, fsl_in)
        projected = self.projector.project(faces, fsl_in)
        _, transported_mass, transport = self.transport.transport_volume_mass(
            fsl_in.fill_level,
            fsl_in.mass,
            projected,
            split_order=split_order,
            use_compression=True,
        )
        streamed = self.streamer.stream(fluid_in, fsl_in)
        self.collider.collide(streamed, fsl_in, self._collided, dt)
        topology = self.topology.resolve(
            self._collided,
            fsl_in,
            transported_mass,
            self._candidate_fluid,
            self._candidate_fsl,
        )
        stream = self.streamer.collect_diagnostics()
        fluid = self.collider.collect_diagnostics()
        if stream.invalid_cell_count or fluid.invalid_cell_count:
            raise FloatingPointError(
                "projected GVOF HOME-FSL transaction contains invalid cells"
            )
        if stream.direct_liquid_gas_link_count:
            raise RuntimeError(
                "projected GVOF HOME-FSL transaction contains liquid-gas links"
            )
        if fluid.max_speed > self.model.max_lattice_speed:
            raise FloatingPointError(
                f"projected GVOF speed {fluid.max_speed} exceeds "
                f"{self.model.max_lattice_speed}"
            )
        diagnostics = ProjectedGvofFslWallDiagnostics(
            stream,
            fluid,
            self.builder.last_diagnostics,
            self.projector.last_diagnostics,
            transport,
            topology,
        )
        fluid_out.copy_from(self._candidate_fluid)
        fsl_out.copy_from(self._candidate_fsl)
        self._step_index += 1
        self.last_diagnostics = diagnostics
        return diagnostics
