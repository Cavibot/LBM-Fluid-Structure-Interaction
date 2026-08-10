# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollback-safe fixed-topology HOME-Free composition transaction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..core.diagnostics import HomeCoreDiagnostics
from ..core.model import HomeCoreModel
from ..core.solver import HomeCoreSolver
from ..core.state import HomeCoreState
from . import kernels
from .advection import FslMassAdvector, FslMassExchangeDiagnostics
from .pressure import FslOnlyMissingDiagnostics, FslOnlyMissingStreamer
from .state import FslState
from .topology import FslTopologyUpdater, FslWarpTopologyDiagnostics


@dataclass(frozen=True)
class FslFixedTopologyDiagnostics:
    pressure: FslOnlyMissingDiagnostics
    mass_exchange: FslMassExchangeDiagnostics
    fluid: HomeCoreDiagnostics
    invalid_commit_cell_count: int
    phase_crossing_cell_count: int
    total_liquid_mass_normalization: float
    max_abs_liquid_mass_normalization: float
    initial_total_mass: float
    candidate_total_mass: float
    relative_total_mass_drift: float
    max_fill_change: float


@dataclass(frozen=True)
class FslDynamicTopologyDiagnostics:
    pressure: FslOnlyMissingDiagnostics
    mass_exchange: FslMassExchangeDiagnostics
    fluid: HomeCoreDiagnostics
    topology: FslWarpTopologyDiagnostics


class FslFixedTopologyStepper:
    """Compose mass exchange, only-missing stream, and collision before commit."""

    def __init__(
        self,
        model: HomeCoreModel,
        *,
        gas_density: float = 1.0,
        fill_epsilon: float = 1.0e-4,
        mass_density_tolerance: float = 2.0e-6,
        relative_mass_drift_tolerance: float = 5.0e-6,
        solver: HomeCoreSolver | None = None,
    ) -> None:
        if solver is not None and solver.model is not model:
            raise ValueError("an injected HOME core solver must own the same model")
        for value, name in (
            (fill_epsilon, "fill_epsilon"),
            (mass_density_tolerance, "mass_density_tolerance"),
            (relative_mass_drift_tolerance, "relative_mass_drift_tolerance"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if fill_epsilon >= 0.5:
            raise ValueError("fill_epsilon must be below 0.5")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.fill_epsilon = float(fill_epsilon)
        self.mass_density_tolerance = float(mass_density_tolerance)
        self.relative_mass_drift_tolerance = float(relative_mass_drift_tolerance)
        self.solver = solver or HomeCoreSolver(model)
        self.streamer = FslOnlyMissingStreamer(model, gas_density=gas_density)
        self.advector = FslMassAdvector(model)
        self._candidate_fluid = HomeCoreState(model)
        self._candidate_fsl = FslState(model)
        self._invalid_commit_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._phase_crossing_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._liquid_mass_normalization = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._total_liquid_mass_normalization = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._max_abs_liquid_mass_normalization = wp.zeros(1, dtype=float, device=self.device)
        self._initial_total_mass = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._candidate_total_mass = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._initial_total_mass_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._candidate_total_mass_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._max_fill_change = wp.zeros(1, dtype=float, device=self.device)
        self.last_diagnostics: FslFixedTopologyDiagnostics | None = None

    def _validate_states(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
    ) -> None:
        states = (fluid_in, fsl_in, fluid_out, fsl_out)
        if any(state.model is not self.model for state in states):
            raise ValueError("stepper and all states must share one HomeCoreModel")
        if any(state.res != self.res for state in states):
            raise ValueError("stepper and all states must share one resolution")

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
    ) -> FslFixedTopologyDiagnostics:
        self._validate_states(fluid_in, fsl_in, fluid_out, fsl_out)
        self.model.validate_step(dt)
        self.last_diagnostics = None
        self._invalid_commit_cell_count.zero_()
        self._phase_crossing_cell_count.zero_()
        self._max_abs_liquid_mass_normalization.zero_()
        self._max_fill_change.zero_()

        self.advector.advect(fluid_in, fsl_in)
        streamed = self.streamer.stream(fluid_in, fsl_in)
        self.solver.collide(streamed, self._candidate_fluid, dt)
        wp.launch(
            kernels.fixed_topology_commit_kernel,
            dim=self.res,
            inputs=[
                self.advector.advected_mass,
                fsl_in.mass,
                fsl_in.fill_level,
                fsl_in.excess_mass,
                fsl_in.flags,
                self._candidate_fluid.moments,
                self._candidate_fsl.mass,
                self._candidate_fsl.fill_level,
                self._candidate_fsl.excess_mass,
                self._candidate_fsl.flags,
                self._invalid_commit_cell_count,
                self._phase_crossing_cell_count,
                self._liquid_mass_normalization,
                self._max_abs_liquid_mass_normalization,
                self._initial_total_mass_cells,
                self._candidate_total_mass_cells,
                self._max_fill_change,
                self.fill_epsilon,
                self.mass_density_tolerance,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.utils.array_sum(
            self._liquid_mass_normalization,
            out=self._total_liquid_mass_normalization,
        )
        wp.utils.array_sum(self._initial_total_mass_cells, out=self._initial_total_mass)
        wp.utils.array_sum(self._candidate_total_mass_cells, out=self._candidate_total_mass)

        fluid = self.solver.validate_state(self._candidate_fluid)
        pressure = self.streamer.collect_diagnostics(synchronize=False)
        mass_exchange = self.advector.collect_diagnostics(synchronize=False)
        if pressure.invalid_cell_count:
            raise FloatingPointError(
                "FSL only-missing stream encountered "
                f"{pressure.invalid_cell_count} invalid cells"
            )
        if pressure.direct_liquid_gas_link_count:
            raise RuntimeError(
                "FSL only-missing stream encountered "
                f"{pressure.direct_liquid_gas_link_count} direct liquid-gas links"
            )
        if mass_exchange.invalid_cell_count:
            raise FloatingPointError(
                "FSL mass exchange encountered "
                f"{mass_exchange.invalid_cell_count} invalid cells"
            )
        if mass_exchange.direct_liquid_gas_link_count:
            raise RuntimeError(
                "FSL mass exchange encountered "
                f"{mass_exchange.direct_liquid_gas_link_count} direct liquid-gas links"
            )
        invalid_commit = int(self._invalid_commit_cell_count.numpy()[0])
        phase_crossing = int(self._phase_crossing_cell_count.numpy()[0])
        total_normalization = float(self._total_liquid_mass_normalization.numpy()[0])
        max_normalization = float(self._max_abs_liquid_mass_normalization.numpy()[0])
        if invalid_commit:
            raise FloatingPointError(
                f"fixed-topology commit contains {invalid_commit} invalid cells"
            )
        if phase_crossing:
            raise RuntimeError(
                "fixed-topology step requires a topology transition in "
                f"{phase_crossing} cells"
            )

        initial_total = float(self._initial_total_mass.numpy()[0])
        candidate_total = float(self._candidate_total_mass.numpy()[0])
        relative_drift = (
            (candidate_total - initial_total) / initial_total if initial_total != 0.0 else 0.0
        )
        if abs(relative_drift) > self.relative_mass_drift_tolerance:
            raise FloatingPointError(
                f"fixed-topology relative mass drift {relative_drift} exceeds "
                f"tolerance {self.relative_mass_drift_tolerance}"
            )
        max_fill_change = float(self._max_fill_change.numpy()[0])
        diagnostics = FslFixedTopologyDiagnostics(
            pressure=pressure,
            mass_exchange=mass_exchange,
            fluid=fluid,
            invalid_commit_cell_count=invalid_commit,
            phase_crossing_cell_count=phase_crossing,
            total_liquid_mass_normalization=total_normalization,
            max_abs_liquid_mass_normalization=max_normalization,
            initial_total_mass=initial_total,
            candidate_total_mass=candidate_total,
            relative_total_mass_drift=relative_drift,
            max_fill_change=max_fill_change,
        )
        fluid_out.copy_from(self._candidate_fluid)
        fsl_out.copy_from(self._candidate_fsl)
        self.last_diagnostics = diagnostics
        return diagnostics


class FslDynamicTopologyStepper:
    """Advance HOME-Free with topology changes as one rollback-safe transaction."""

    def __init__(
        self,
        model: HomeCoreModel,
        *,
        gas_density: float = 1.0,
        transition_tolerance: float = 0.0,
        relative_mass_drift_tolerance: float = 5.0e-6,
        solver: HomeCoreSolver | None = None,
    ) -> None:
        if solver is not None and solver.model is not model:
            raise ValueError("an injected HOME core solver must own the same model")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.solver = solver or HomeCoreSolver(model)
        self.streamer = FslOnlyMissingStreamer(model, gas_density=gas_density)
        self.advector = FslMassAdvector(model)
        self.topology = FslTopologyUpdater(
            model,
            transition_tolerance=transition_tolerance,
            relative_mass_drift_tolerance=relative_mass_drift_tolerance,
        )
        self._collided_fluid = HomeCoreState(model)
        self._candidate_fluid = HomeCoreState(model)
        self._candidate_fsl = FslState(model)
        self.last_diagnostics: FslDynamicTopologyDiagnostics | None = None

    def _validate_states(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
    ) -> None:
        states = (fluid_in, fsl_in, fluid_out, fsl_out)
        if any(state.model is not self.model for state in states):
            raise ValueError("stepper and all states must share one HomeCoreModel")
        if any(state.res != self.res for state in states):
            raise ValueError("stepper and all states must share one resolution")

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
    ) -> FslDynamicTopologyDiagnostics:
        self._validate_states(fluid_in, fsl_in, fluid_out, fsl_out)
        self.model.validate_step(dt)
        self.last_diagnostics = None
        self.advector.advect(fluid_in, fsl_in)
        streamed = self.streamer.stream(fluid_in, fsl_in)
        self.solver.collide(streamed, self._collided_fluid, dt)
        topology = self.topology.resolve(
            self._collided_fluid,
            fsl_in,
            self.advector.advected_mass,
            self._candidate_fluid,
            self._candidate_fsl,
        )
        fluid = self.solver.validate_state(self._candidate_fluid)
        pressure = self.streamer.collect_diagnostics(synchronize=False)
        mass_exchange = self.advector.collect_diagnostics(synchronize=False)
        if pressure.invalid_cell_count or mass_exchange.invalid_cell_count:
            raise FloatingPointError("dynamic HOME-Free pre-topology stage contains invalid cells")
        if pressure.direct_liquid_gas_link_count or mass_exchange.direct_liquid_gas_link_count:
            raise RuntimeError("dynamic HOME-Free pre-topology stage contains liquid-gas links")
        diagnostics = FslDynamicTopologyDiagnostics(
            pressure=pressure,
            mass_exchange=mass_exchange,
            fluid=fluid,
            topology=topology,
        )
        fluid_out.copy_from(self._candidate_fluid)
        fsl_out.copy_from(self._candidate_fsl)
        self.last_diagnostics = diagnostics
        return diagnostics
