# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSL composition with raw-Courant PLIC volume and mass transport."""

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
from .courant import FaceCourantDiagnostics, HomeFaceCourantBuilder
from .transport import (
    PlicGeometricTransport,
    PlicVolumeMassTransportDiagnostics,
)


@dataclass(frozen=True)
class GvofFslWallDiagnostics:
    stream: FslWallStreamDiagnostics
    fluid: HomeCoreDiagnostics
    face_courant: FaceCourantDiagnostics
    transport: PlicVolumeMassTransportDiagnostics
    topology: FslWarpTopologyDiagnostics


class GvofFslWallStepper:
    """Replace link-wise mass exchange while retaining baseline FSL topology."""

    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        axes: tuple[int, ...] = (0, 1),
        gas_density: float = 1.0,
        transition_tolerance: float = 0.0,
        relative_mass_drift_tolerance: float = 5.0e-6,
        interface_roundoff_tolerance: float = 2.0e-6,
    ) -> None:
        if walls.model is not model:
            raise ValueError("GVOF stepper and walls must share one HOME model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.axes = tuple(axes)
        self.builder = HomeFaceCourantBuilder(model, walls)
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
        self.last_diagnostics: GvofFslWallDiagnostics | None = None

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
    ) -> GvofFslWallDiagnostics:
        states = (fluid_in, fsl_in, fluid_out, fsl_out)
        if any(state.model is not self.model for state in states):
            raise ValueError("GVOF stepper and states must share one HOME model")
        self.model.validate_step(dt)
        self.last_diagnostics = None
        split_order = (
            self.axes
            if self._step_index % 2 == 0
            else tuple(reversed(self.axes))
        )
        faces = self.builder.build(fluid_in, fsl_in)
        _, transported_mass, transport = self.transport.transport_volume_mass(
            fsl_in.fill_level,
            fsl_in.mass,
            faces,
            split_order=split_order,
            use_compression=False,
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
            raise FloatingPointError("GVOF HOME-FSL transaction contains invalid cells")
        if stream.direct_liquid_gas_link_count:
            raise RuntimeError("GVOF HOME-FSL transaction contains liquid-gas links")
        if fluid.max_speed > self.model.max_lattice_speed:
            raise FloatingPointError(
                f"GVOF HOME-FSL speed {fluid.max_speed} exceeds "
                f"{self.model.max_lattice_speed}"
            )
        diagnostics = GvofFslWallDiagnostics(
            stream,
            fluid,
            self.builder.last_diagnostics,
            transport,
            topology,
        )
        fluid_out.copy_from(self._candidate_fluid)
        fsl_out.copy_from(self._candidate_fsl)
        self._step_index += 1
        self.last_diagnostics = diagnostics
        return diagnostics
