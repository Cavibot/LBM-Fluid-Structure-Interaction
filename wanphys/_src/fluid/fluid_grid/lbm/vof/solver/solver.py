"""VOF lifecycle and transaction owner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..contracts import VofMassScheme, VofRuntimeProfile
from ..diagnostics.device import VofDeviceDiagnostics
from ..diagnostics.host import (
    VofDiagnostics,
    collect_vof_diagnostics,
    validate_vof_diagnostics,
)
from ..initial_conditions import (
    initialize_vof_fields,
    prepare_initial_vof,
    validate_no_solid_cells,
)
from ..model import VofModel
from .advection import VofMassTransport
from .geometry import VofInterfaceGeometry
from .kinetic_init import VofKineticInitializer
from .surface import VofSurfaceBoundary
from .transition import VofTopologyTransition

if TYPE_CHECKING:
    from ...state import FullFLbmState, HomeLbmState, LbmStateBase


class VofSolver:
    """Own VOF scratch and expose the three fixed lifecycle operations."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
        *,
        model: VofModel,
        runtime_profile: VofRuntimeProfile,
        validation_interval: int,
        max_lattice_speed: float,
        gravity: tuple[float, float, float],
        enforce_population_positivity: bool,
        population_floor: float,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)
        self.model = VofModel(
            mass_scheme=VofMassScheme(model.mass_scheme),
            atmosphere_pressure=float(model.atmosphere_pressure),
            surface_tension=float(model.surface_tension),
            transition_epsilon=float(model.transition_epsilon),
        )
        self.runtime_profile = VofRuntimeProfile(runtime_profile)
        self.validation_interval = int(validation_interval)
        self.max_lattice_speed = float(max_lattice_speed)
        self.gravity = tuple(float(value) for value in gravity)
        self.enforce_population_positivity = bool(enforce_population_positivity)
        self.population_floor = float(population_floor)
        self.kinetic_initializer = VofKineticInitializer(
            self.shape,
            self.device,
            self.periodic,
        )

        self.mass_transport = VofMassTransport(
            self.shape, self.device, self.periodic
        )
        self.surface_boundary = VofSurfaceBoundary(
            self.shape,
            self.device,
            self.periodic,
            float(self.model.atmosphere_pressure),
            float(self.model.surface_tension),
        )
        self.interface_geometry = VofInterfaceGeometry(
            self.shape, self.device, self.periodic
        )
        self.topology_transition = VofTopologyTransition(
            self.shape,
            self.device,
            self.periodic,
            float(self.model.transition_epsilon),
        )
        self.device_diagnostics = None
        if self.runtime_profile is VofRuntimeProfile.DEVICE:
            self.device_diagnostics = VofDeviceDiagnostics(
                self.shape,
                self.device,
                self.periodic,
                atmosphere_pressure=float(model.atmosphere_pressure),
                surface_tension=float(model.surface_tension),
            )
        self._prepared_source: LbmStateBase | None = None
        self._prepared_target: LbmStateBase | None = None
        self._prepared_epoch: int | None = None
        self._prepared_validation = False
        self._last_diagnostics: VofDiagnostics | None = None

    @property
    def last_diagnostics(self) -> VofDiagnostics | None:
        """Return the latest read-only diagnostics result."""

        return self._last_diagnostics

    @property
    def prepared_proposed_type(self) -> wp.array | None:
        """Expose transition scratch only to the internal diagnostics path."""

        if self.topology_transition.result is None:
            return None
        return self.topology_transition.result.proposed_type

    def initialize(self, state: LbmStateBase, phi0: np.ndarray) -> None:
        """Initialize one authoritative VOF state and clear transaction state."""

        if state.vof is None:
            raise ValueError("supplied state has no authoritative VOF storage")
        validate_no_solid_cells(state.solid_phi)
        prepared_phi, cell_type = prepare_initial_vof(
            phi0,
            shape=self.shape,
            periodic=tuple(bool(value) for value in self.periodic),
        )
        self.initialize_prepared(state, prepared_phi, cell_type)
        self._prepared_source = None
        self._prepared_target = None
        self._prepared_epoch = None
        self._last_diagnostics = None

    def initialize_prepared(
        self,
        state: LbmStateBase,
        prepared_phi: np.ndarray,
        cell_type: np.ndarray,
    ) -> None:
        """Initialize already host-validated VOF input for Domain coordination."""

        if state.vof is None:
            raise ValueError("supplied state has no authoritative VOF storage")
        validate_no_solid_cells(state.solid_phi)
        initialize_vof_fields(state.density, state.vof, prepared_phi, cell_type)
        state.vof.epoch = 0
        self.interface_geometry.compute(state.vof)

    def complete_streaming(
        self,
        state_in: LbmStateBase,
        state_out: LbmStateBase,
        logical_f_post: wp.array,
        streamed_populations: wp.array,
    ) -> None:
        """Prepare VOF mass and complete current streaming without writing target."""

        self._check_prepare_pair(state_in, state_out)
        full_validation = self._full_validation_required(state_in)
        self.mass_transport.compute(
            state_in,
            logical_f_post,
            validate=full_validation,
        )
        self.surface_boundary.complete_populations(
            state_in,
            logical_f_post,
            streamed_populations,
            validate=full_validation,
        )
        assert state_in.vof is not None
        self._last_diagnostics = None
        self._prepared_source = state_in
        self._prepared_target = state_out
        self._prepared_epoch = int(state_in.vof.epoch)
        self._prepared_validation = full_validation

    def finish_step(self, state_in: LbmStateBase, state_out: LbmStateBase) -> None:
        """Commit topology, publish new active cells, then run diagnostics."""

        from ...state import FullFLbmState, HomeLbmState

        self._check_finish_pair(state_in, state_out)
        assert state_in.vof is not None and state_out.vof is not None
        validate = self._prepared_validation

        self.surface_boundary.restore_gas(state_in, state_out)
        transition = self.topology_transition.compute(
            self.mass_transport.result.mass_tmp,
            state_out.density,
            state_in.vof.cell_type,
            validate=validate,
        )
        initializer_kwargs = {
            "gravity": self.gravity,
            "max_lattice_speed": self.max_lattice_speed,
            "enforce_population_positivity": self.enforce_population_positivity,
            "population_floor": self.population_floor,
            "validate": validate,
        }
        if isinstance(state_out, FullFLbmState):
            self.kinetic_initializer.initialize_fullf(
                state_out,
                state_in.vof.cell_type,
                transition,
                **initializer_kwargs,
            )
        elif isinstance(state_out, HomeLbmState):
            self.kinetic_initializer.initialize_home(
                state_out,
                state_in.vof.cell_type,
                transition,
                **initializer_kwargs,
            )
        else:
            raise TypeError(f"Unsupported VOF encoding: {type(state_out).__name__}")
        self.topology_transition.commit(
            state_out.vof,
            source_epoch=state_in.vof.epoch,
        )
        state_out.vof.reference_mass = state_in.vof.reference_mass
        self.interface_geometry.compute(state_out.vof, validate=validate)
        self._validate_target(state_in, state_out)

        if validate:
            diagnostics = collect_vof_diagnostics(
                state_out,
                initial_mass=state_in.vof.reference_mass,
                periodic=tuple(bool(value) for value in self.periodic),
            )
            validate_vof_diagnostics(
                diagnostics,
                max_lattice_speed=self.max_lattice_speed,
            )
            self._last_diagnostics = diagnostics
        elif self.runtime_profile is VofRuntimeProfile.DEVICE:
            if self.device_diagnostics is None:
                raise RuntimeError("device VOF diagnostics were not allocated")
            diagnostics = self.device_diagnostics.collect(
                state_out,
                previous_cell_type=state_in.vof.cell_type,
                proposed_cell_type=transition.proposed_type,
            )
            validate_vof_diagnostics(
                diagnostics,
                max_lattice_speed=self.max_lattice_speed,
            )
            self._last_diagnostics = diagnostics

        self._prepared_source = None
        self._prepared_target = None
        self._prepared_epoch = None

    def _full_validation_required(self, state_in: LbmStateBase) -> bool:
        if self.runtime_profile is VofRuntimeProfile.STRICT:
            return True
        if self.runtime_profile is not VofRuntimeProfile.SAMPLED:
            return False
        if state_in.vof is None:
            raise ValueError("sampled VOF validation requires state.vof")
        next_epoch = int(state_in.vof.epoch) + 1
        return next_epoch % self.validation_interval == 0

    def _check_prepare_pair(
        self,
        state_in: LbmStateBase,
        state_out: LbmStateBase,
    ) -> None:
        if state_in is state_out:
            raise RuntimeError("VOF transaction requires distinct state buffers")
        if self._prepared_source is not None:
            raise RuntimeError("previous VOF step was not finished")
        if type(state_in) is not type(state_out):
            raise RuntimeError("VOF transaction requires matching state types")
        if state_in.vof is None or state_out.vof is None:
            raise RuntimeError("VOF transaction requires authoritative VOF storage")
        if state_in.vof.epoch < 0:
            raise RuntimeError("VOF transaction source must be initialized")

    def _check_finish_pair(
        self,
        state_in: LbmStateBase,
        state_out: LbmStateBase,
    ) -> None:
        if self._prepared_source is not state_in:
            raise RuntimeError("finish_step source does not match prepared source")
        if self._prepared_target is not state_out:
            raise RuntimeError("finish_step target does not match prepared target")
        if state_in.vof is None or self._prepared_epoch != state_in.vof.epoch:
            raise RuntimeError("finish_step epoch does not match prepared epoch")

    @staticmethod
    def _validate_target(state_in: LbmStateBase, state_out: LbmStateBase) -> None:
        assert state_in.vof is not None and state_out.vof is not None
        if state_out.vof.epoch != state_in.vof.epoch + 1:
            raise RuntimeError("VOF transaction target epoch did not advance once")
        if state_out.vof.geometry_epoch != state_out.vof.epoch:
            raise RuntimeError("VOF transaction target geometry is stale")
        if state_out.vof.reference_mass != state_in.vof.reference_mass:
            raise RuntimeError("VOF transaction changed reference mass")


__all__ = ["VofSolver"]
