# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-buffered HOME-Free advancement built on the HOME moment solver."""

from __future__ import annotations

import math

import numpy as np

from wanphys._src.core.domain import Domain

from ..diagnostics import HomeLbmDiagnostics
from ..model import HomeLbmModel
from ..solver import HomeLbmSolver
from ..state import HomeLbmState
from .advection import HomeFreeMassAdvector
from .plic import plic_plane_offset
from .reference import HomeFreeStateDiagnostics
from .state import HomeFreeState
from .surface_tension import (
    HomeFreeSurfaceTension,
    HomeFreeSurfaceTensionDiagnostics,
)
from .topology import HomeFreeTopologyUpdater


class HomeFreeDomainState:
    """One committed HOME-Free time level."""

    def __init__(self, model: HomeLbmModel) -> None:
        self.fluid = HomeLbmState(model)
        self.free_surface = HomeFreeState(model)

    def clear_forces(self) -> None:
        self.fluid.clear_forces()

    def copy_from(self, other: "HomeFreeDomainState") -> None:
        self.fluid.copy_from(other.fluid)
        self.free_surface.copy_from(other.free_surface)


class HomeFreeDomain(Domain):
    """Advance mass, HOME moments, and sharp topology as one transaction."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        gas_density: float = 1.0,
        surface_tension: float = 0.0,
        fill_epsilon: float = 1.0e-4,
        solver: HomeLbmSolver | None = None,
    ) -> None:
        if not math.isfinite(gas_density) or gas_density <= 0.0:
            raise ValueError("gas_density must be finite and positive")
        if solver is not None and solver.model is not model:
            raise ValueError("an injected HOME solver must own the same model")
        self._model = model
        self._solver = solver or HomeLbmSolver(model)
        self._advector = HomeFreeMassAdvector(model)
        self._topology = HomeFreeTopologyUpdater(model, fill_epsilon=fill_epsilon)
        self._gas_density = float(gas_density)
        self._surface_tension = (
            HomeFreeSurfaceTension(
                model,
                ambient_gas_density=gas_density,
                surface_tension=surface_tension,
            )
            if surface_tension != 0.0
            else None
        )
        self._state_in: HomeFreeDomainState | None = None
        self._state_out: HomeFreeDomainState | None = None
        self._last_diagnostics: HomeLbmDiagnostics | None = None
        self._last_surface_tension_diagnostics: (
            HomeFreeSurfaceTensionDiagnostics | None
        ) = None

    @property
    def name(self) -> str:
        return "fluid_grid_home_free_lbm"

    @property
    def model(self) -> HomeLbmModel:
        return self._model

    @property
    def solver(self) -> HomeLbmSolver:
        return self._solver

    @property
    def advector(self) -> HomeFreeMassAdvector:
        return self._advector

    @property
    def topology(self) -> HomeFreeTopologyUpdater:
        return self._topology

    @property
    def gas_density(self) -> float:
        return self._gas_density

    @property
    def surface_tension(self) -> HomeFreeSurfaceTension | None:
        return self._surface_tension

    @property
    def last_diagnostics(self) -> HomeLbmDiagnostics | None:
        return self._last_diagnostics

    @property
    def last_surface_tension_diagnostics(
        self,
    ) -> HomeFreeSurfaceTensionDiagnostics | None:
        return self._last_surface_tension_diagnostics

    @property
    def state(self) -> HomeFreeDomainState:
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None
        return self._state_in

    @property
    def fluid_state(self) -> HomeLbmState:
        return self.state.fluid

    @property
    def free_surface_state(self) -> HomeFreeState:
        return self.state.free_surface

    def create_state(self) -> HomeFreeDomainState:
        self._state_in = HomeFreeDomainState(self._model)
        self._state_out = HomeFreeDomainState(self._model)
        self._last_diagnostics = None
        self._last_surface_tension_diagnostics = None
        return self._state_in

    def initialize_uniform_lattice(
        self,
        fill_level: np.ndarray,
        *,
        rho: float = 1.0,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        validate_topology: bool = True,
    ) -> HomeFreeStateDiagnostics:
        """Initialize both parts of the active time level from one fill field."""

        state = self.state
        self._solver.initialize_uniform_lattice(state.fluid, rho=rho, velocity=velocity)
        diagnostics = state.free_surface.initialize_from_fill_level(
            state.fluid,
            fill_level,
            validate_topology=validate_topology,
        )
        self._last_diagnostics = None
        self._last_surface_tension_diagnostics = None
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
        """Anchor a planar free-surface hydrostatic state to gas pressure."""

        if interface_axis not in (0, 1, 2):
            raise ValueError("interface_axis must be 0, 1, or 2")
        if gas_direction not in (-1, 1):
            raise ValueError("gas_direction must be -1 or 1")
        fill = np.asarray(fill_level, dtype=np.float32)
        expected_shape = (int(self._model.nx), int(self._model.ny), int(self._model.nz))
        if fill.shape != expected_shape:
            raise ValueError("fill_level shape must match the HOME-Free grid")
        if not 0 <= interface_index < expected_shape[interface_axis]:
            raise ValueError("interface_index is outside the selected axis")
        gas_index = interface_index + gas_direction
        liquid_index = interface_index - gas_direction
        if not 0 <= gas_index < expected_shape[interface_axis]:
            raise ValueError("the gas-side neighbor must lie inside the domain")
        if not 0 <= liquid_index < expected_shape[interface_axis]:
            raise ValueError("the liquid-side neighbor must lie inside the domain")
        interface_plane = np.take(fill, interface_index, axis=interface_axis)
        gas_plane = np.take(fill, gas_index, axis=interface_axis)
        liquid_plane = np.take(fill, liquid_index, axis=interface_axis)
        interface_fill = float(interface_plane.flat[0])
        if not 0.0 < interface_fill < 1.0 or not np.allclose(
            interface_plane, interface_fill, rtol=0.0, atol=1.0e-7
        ):
            raise ValueError("the selected interface plane must have one fractional fill")
        if not np.all(gas_plane == 0.0) or not np.all(liquid_plane == 1.0):
            raise ValueError("selected interface must separate a full liquid and gas plane")
        acceleration = np.asarray(self._model.lattice_acceleration, dtype=np.float64)
        transverse = np.delete(acceleration, interface_axis)
        if np.any(transverse != 0.0):
            raise ValueError("planar hydrostatic initialization requires axis-aligned acceleration")

        normal = np.zeros(3, dtype=np.float64)
        normal[interface_axis] = float(gas_direction)
        offset = plic_plane_offset(interface_fill, normal)
        reference_coordinate = 0.5 * np.asarray(expected_shape, dtype=np.float64)
        reference_coordinate[interface_axis] = (
            interface_index
            + 0.5
            + gas_direction * (offset + 0.5)
        )
        return self.initialize_anchored_hydrostatic_lattice(
            fill,
            reference_coordinate=tuple(float(value) for value in reference_coordinate),
            validate_topology=validate_topology,
        )

    def initialize_anchored_hydrostatic_lattice(
        self,
        fill_level: np.ndarray,
        *,
        reference_coordinate: tuple[float, float, float],
        reference_density: float | None = None,
        validate_topology: bool = True,
    ) -> HomeFreeStateDiagnostics:
        """Initialize a free surface from a prescribed hydrostatic pressure anchor.

        ``reference_coordinate`` uses continuous lattice coordinates with cell
        centers at ``i+0.5``. The default pressure is the configured ambient gas
        density; callers constructing a non-planar initial surface must choose
        the physically relevant atmospheric reference location explicitly.
        """

        state = self.state
        self._solver.initialize_anchored_hydrostatic_lattice(
            state.fluid,
            reference_coordinate,
            self._gas_density if reference_density is None else reference_density,
        )
        diagnostics = state.free_surface.initialize_from_fill_level(
            state.fluid,
            fill_level,
            validate_topology=validate_topology,
        )
        self._last_diagnostics = None
        self._last_surface_tension_diagnostics = None
        return diagnostics

    def step(self, dt: float, contacts: object = None) -> None:
        if self._state_in is None or self._state_out is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None

        source = self._state_in
        destination = self._state_out
        gas_density_field = None
        surface_diagnostics = None
        if self._surface_tension is not None:
            surface_diagnostics = self._surface_tension.update(source.free_surface)
            gas_density_field = self._surface_tension.gas_density
        advected_mass = self._advector.advect(
            source.fluid,
            source.free_surface,
            pressure_reference_density=self._gas_density,
        )
        self._solver.step(
            source.fluid,
            destination.fluid,
            dt,
            contacts=contacts,
            free_surface_flags=source.free_surface.flags,
            gas_density=self._gas_density,
            gas_density_field=gas_density_field,
        )
        self._advector.apply_queue_momentum_correction(destination.fluid)
        self._topology.classify(
            destination.fluid,
            source.free_surface,
            advected_mass,
        )
        self._topology.commit(
            destination.fluid,
            source.free_surface,
            advected_mass,
            destination.free_surface,
        )
        diagnostics = self._solver.validate_state(
            destination.fluid,
            free_surface_flags=destination.free_surface.flags,
            gas_density_field=gas_density_field,
        )
        self._state_in, self._state_out = destination, source
        self._last_diagnostics = diagnostics
        self._last_surface_tension_diagnostics = surface_diagnostics
