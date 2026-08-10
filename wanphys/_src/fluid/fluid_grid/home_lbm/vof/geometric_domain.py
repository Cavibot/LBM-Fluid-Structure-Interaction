# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-buffered projected geometric HOME-Free domain."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from ..solver import HomeLbmSolver
from .domain import HomeFreeDomain, HomeFreeDomainState
from .geometric_advection_kernels import (
    initialize_planar_mass_weighted_hydrostatic_kernel,
)
from .om_stepper import (
    HomeFreeOmResearchStepDiagnostics,
    HomeFreeOmResearchStepper,
    HomeFreeOmTransactionHistory,
)
from .reference import HomeFreeStateDiagnostics
from .surface_tension import HomeFreeSurfaceTension


class HomeFreeGeometricDomain(HomeFreeDomain):
    """Advance HOME moments and PLIC VOF through one shared transaction.

    The inherited state and initialization API intentionally preserves the
    ownership contract expected by rigid cut-link interfaces.  Time advancement
    is replaced by :class:`HomeFreeOmResearchStepper`; the legacy link-wise mass
    advector remains owned only by :class:`HomeFreeDomain`.
    """

    uses_independent_geometric_mass = True

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        gas_density: float = 1.0,
        surface_tension: float = 0.0,
        contact_angle_degrees: float | None = None,
        advection_axes: tuple[int, ...] = (0, 1, 2),
        project_courant: bool = True,
        projection_max_iterations: int | None = None,
        solver: HomeLbmSolver | None = None,
    ) -> None:
        super().__init__(
            model,
            gas_density=gas_density,
            surface_tension=0.0,
            solver=solver,
        )
        self._geometric_stepper = HomeFreeOmResearchStepper(
            model,
            advection_axes=advection_axes,
            gas_density=gas_density,
            surface_tension=surface_tension,
            contact_angle_degrees=contact_angle_degrees,
            project_courant=project_courant,
            projection_max_iterations=projection_max_iterations,
            solver=self._solver,
        )
        self._last_geometric_diagnostics: (
            HomeFreeOmResearchStepDiagnostics | None
        ) = None

    @property
    def name(self) -> str:
        return "fluid_grid_home_free_geometric_lbm"

    @property
    def geometric_stepper(self) -> HomeFreeOmResearchStepper:
        return self._geometric_stepper

    @property
    def surface_tension(self) -> HomeFreeSurfaceTension:
        return self._geometric_stepper.surface_tension

    @property
    def last_geometric_diagnostics(
        self,
    ) -> HomeFreeOmResearchStepDiagnostics | None:
        return self._last_geometric_diagnostics

    def create_state(self) -> HomeFreeDomainState:
        state = super().create_state()
        if hasattr(self, "_geometric_stepper"):
            self._reset_geometric_history()
        return state

    def initialize_uniform_lattice(
        self,
        fill_level: np.ndarray,
        *,
        rho: float = 1.0,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        validate_topology: bool = True,
    ) -> HomeFreeStateDiagnostics:
        diagnostics = super().initialize_uniform_lattice(
            fill_level,
            rho=rho,
            velocity=velocity,
            validate_topology=validate_topology,
        )
        self._reset_geometric_history()
        return diagnostics

    def initialize_planar_hydrostatic_lattice(
        self,
        fill_level: np.ndarray,
        *,
        interface_axis: int,
        interface_index: int,
        gas_direction: int,
        validate_topology: bool = True,
    ) -> HomeFreeStateDiagnostics:
        diagnostics = super().initialize_planar_hydrostatic_lattice(
            fill_level,
            interface_axis=interface_axis,
            interface_index=interface_index,
            gas_direction=gas_direction,
            validate_topology=validate_topology,
        )
        interface_fill = float(
            np.take(np.asarray(fill_level), interface_index, axis=interface_axis).flat[0]
        )
        state = self.state
        wp.launch(
            initialize_planar_mass_weighted_hydrostatic_kernel,
            dim=state.fluid.res,
            inputs=[
                state.fluid.moments,
                state.free_surface.mass,
                state.free_surface.fill_level,
                wp.vec3(*self._model.lattice_acceleration),
                self._gas_density,
                interface_fill,
                interface_axis,
                interface_index,
                gas_direction,
                int(self._model.ny),
                int(self._model.nz),
                state.fluid.cell_count,
            ],
            device=self._model._device,
        )
        self._reset_geometric_history()
        return diagnostics

    def initialize_anchored_hydrostatic_lattice(
        self,
        fill_level: np.ndarray,
        *,
        reference_coordinate: tuple[float, float, float],
        reference_density: float | None = None,
        validate_topology: bool = True,
    ) -> HomeFreeStateDiagnostics:
        diagnostics = super().initialize_anchored_hydrostatic_lattice(
            fill_level,
            reference_coordinate=reference_coordinate,
            reference_density=reference_density,
            validate_topology=validate_topology,
        )
        self._reset_geometric_history()
        return diagnostics

    def step(self, dt: float, contacts: object = None) -> None:
        self._model.validate_step(float(dt))
        if contacts is not None:
            raise ValueError("geometric HOME-Free does not accept contact data")
        if self._state_in is None or self._state_out is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None

        diagnostics = self._geometric_stepper.step(
            self._state_in.fluid,
            self._state_in.free_surface,
            self._state_out.fluid,
            self._state_out.free_surface,
        )
        self._state_in, self._state_out = self._state_out, self._state_in
        self._last_diagnostics = diagnostics.solver
        self._last_surface_tension_diagnostics = diagnostics.surface_tension
        self._last_geometric_diagnostics = diagnostics

    def create_transaction_history(self) -> HomeFreeOmTransactionHistory:
        return self._geometric_stepper.create_transaction_history()

    def capture_transaction_history(
        self, history: HomeFreeOmTransactionHistory
    ) -> HomeFreeOmResearchStepDiagnostics | None:
        self._geometric_stepper.capture_transaction_history(history)
        return self._last_geometric_diagnostics

    def restore_transaction_history(
        self,
        history: HomeFreeOmTransactionHistory,
        diagnostics: HomeFreeOmResearchStepDiagnostics | None,
    ) -> None:
        self._geometric_stepper.restore_transaction_history(history)
        self._last_geometric_diagnostics = diagnostics

    def _reset_geometric_history(self) -> None:
        self._geometric_stepper.reset_transaction_history()
        self._last_geometric_diagnostics = None
