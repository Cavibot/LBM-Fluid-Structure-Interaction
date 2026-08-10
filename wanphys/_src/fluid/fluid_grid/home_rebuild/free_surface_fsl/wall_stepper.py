# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollback-safe closed-wall, gravity, and dynamic-topology transaction."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..core.diagnostics import HomeCoreDiagnostics
from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from .state import FslState
from .topology import FslWarpTopologyDiagnostics
from .wall_pipeline import (
    FslWallActiveCollider,
    FslWallMassAdvector,
    FslWallMassDiagnostics,
    FslWallOnlyMissingStreamer,
    FslWallStreamDiagnostics,
)
from .wall_topology import FslWallTopologyUpdater
from .walls import FslWallMask


@dataclass(frozen=True)
class FslWallDynamicDiagnostics:
    stream: FslWallStreamDiagnostics
    mass_exchange: FslWallMassDiagnostics
    fluid: HomeCoreDiagnostics
    topology: FslWarpTopologyDiagnostics


class FslWallDynamicTopologyStepper:
    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        gas_density: float = 1.0,
        transition_tolerance: float = 0.0,
        relative_mass_drift_tolerance: float = 5.0e-6,
    ) -> None:
        if walls.model is not model:
            raise ValueError("wall stepper and mask must share one HomeCoreModel")
        self.model = model
        self.walls = walls
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.streamer = FslWallOnlyMissingStreamer(
            model, walls, gas_density=gas_density
        )
        self.advector = FslWallMassAdvector(model, walls)
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
        self.last_diagnostics: FslWallDynamicDiagnostics | None = None

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
        *,
        gas_density_field: wp.array | None = None,
    ) -> FslWallDynamicDiagnostics:
        states = (fluid_in, fsl_in, fluid_out, fsl_out)
        if any(state.model is not self.model for state in states):
            raise ValueError("wall stepper and states must share one HomeCoreModel")
        self.model.validate_step(dt)
        self.last_diagnostics = None
        self.advector.advect(fluid_in, fsl_in)
        streamed = self.streamer.stream(
            fluid_in, fsl_in, gas_density_field=gas_density_field
        )
        self.collider.collide(streamed, fsl_in, self._collided, dt)
        topology = self.topology.resolve(
            self._collided,
            fsl_in,
            self.advector.advected_mass,
            self._candidate_fluid,
            self._candidate_fsl,
        )
        stream = self.streamer.collect_diagnostics()
        mass = self.advector.collect_diagnostics()
        fluid = self.collider.collect_diagnostics()
        if stream.invalid_cell_count or mass.invalid_cell_count or fluid.invalid_cell_count:
            raise FloatingPointError("wall HOME-Free transaction contains invalid cells")
        if stream.direct_liquid_gas_link_count or mass.direct_liquid_gas_link_count:
            raise RuntimeError("wall HOME-Free transaction contains liquid-gas links")
        if fluid.max_speed > self.model.max_lattice_speed:
            raise FloatingPointError(
                f"wall HOME-Free speed {fluid.max_speed} exceeds "
                f"{self.model.max_lattice_speed}"
            )
        diagnostics = FslWallDynamicDiagnostics(
            stream=stream,
            mass_exchange=mass,
            fluid=fluid,
            topology=topology,
        )
        fluid_out.copy_from(self._candidate_fluid)
        fsl_out.copy_from(self._candidate_fsl)
        self.last_diagnostics = diagnostics
        return diagnostics
