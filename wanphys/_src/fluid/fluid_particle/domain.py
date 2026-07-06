# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared base domain for particle-based fluid domains (PBF/WCSPH).

This module defines FluidParticleDomain which provides a minimal common
implementation expected by PBFDomain and WCSPHDomain:
- optional model/solver construction when classes are supplied
- create_state helper that uses subclass _state_cls when available
- default step/pre_step implementations and particle getters
"""

from __future__ import annotations
import numpy as np
import warp as wp
from typing import TYPE_CHECKING, Any

from wanphys._src.core import Domain
from wanphys._src.fluid.fluid_particle.model import ParticleFluidModel
from wanphys._src.fluid.fluid_particle.pbf import PBFSolver, PBFModel, PBFState

from wanphys._src.fluid.fluid_particle.solver import ParticleFluidSolverBase
from wanphys._src.fluid.fluid_particle.wcsph import WCSPHSolver, WCSPHState, WCSPHModel
from wanphys._src.fluid.fluid_particle.dfsph import DFSPHSolver, DFSPHState, DFSPHModel

if TYPE_CHECKING:
    from newton import Contacts
    from wanphys._src.rigid.model import RigidModel
    from wanphys._src.rigid.state import RigidState



class FluidParticleStateManager:
    """Factory for creating particle fluid states.

    This class provides static methods to create state instances based on the
    Newton state. For example, it can wrap the raw Newton state in a PBFState
    or WCSPHState that also holds a reference to the domain model.
    """

    @staticmethod
    def create_state(solver_type: str, particle_q_init, particle_qd_init, model: Any) -> Any:
        """Create a particle fluid state based on the specified type."""
        solver_type = solver_type.upper()
        if solver_type == "PBF":
            return PBFState.from_arrays(particle_q_init, particle_qd_init, model)
        if solver_type == "WCSPH":
            return WCSPHState.from_arrays(particle_q_init, particle_qd_init, model)
        if solver_type == "DFSPH":
            return DFSPHState.from_arrays(particle_q_init, particle_qd_init, model)
        raise ValueError(f"Unsupported solver_type: {solver_type}")




class FluidParticleModelManager:
    """Factory for creating particle fluid models.

    This class provides static methods to create domain models based on the
    Newton model. For example, it can extract particle radius and other
    parameters to construct a PBFModel or WCSPHModel.
    """

    @staticmethod
    def create_model(solver_type : str, parameters: dict[str, Any]) -> Any:
        """Create a particle fluid model based on the solver type and parameters."""
        solver_type = solver_type.upper()
        if solver_type == "PBF":
            return PBFModel(
                particle_count=parameters["particle_count"],
                particle_mass=parameters["particle_mass"],
                particle_flags=parameters["particle_flags"],
                particle_radius=parameters["particle_radius"],
                rest_density=parameters["rest_density"],
                iterations=parameters["iterations"],
                relaxation_parameter=parameters["relaxation_parameter"],
                vorticity_coefficient=parameters["vorticity_coefficient"],
                xsph_c=parameters["xsph_c"],
                use_graph=parameters.get("use_graph", True),
                device=parameters["device"],
            )
        if solver_type == "WCSPH":
            return WCSPHModel(
                h=parameters["h"],
                rest_density=parameters["rest_density"],
                viscosity=parameters["viscosity"],
                use_graph=parameters.get("use_graph", True),
                device=parameters["device"],
                particle_count=parameters["particle_count"],
                particle_mass=parameters["particle_mass"],
                particle_flags=parameters["particle_flags"],
                particle_radius=parameters["particle_radius"],
            )
        if solver_type == "DFSPH":
            return DFSPHModel(
                rest_density=parameters["rest_density"],
                viscosity=parameters["viscosity"],
                divergence_max_iterations=parameters["divergence_max_iterations"],
                pressure_max_iterations=parameters["pressure_max_iterations"],
                density_error_tolerance=parameters["density_error_tolerance"],
                fixed_dt=parameters["fixed_dt"],
                device=parameters["device"],
                use_graph=parameters.get("use_graph", True),
                particle_radius_scalar=parameters["particle_radius_scalar"],
                particle_count=parameters["particle_count"],
                particle_flags=parameters["particle_flags"],
                particle_radius=parameters["particle_radius"],
            )
        raise ValueError(f"Unsupported solver_type: {solver_type}")

class FluidParticleSolverManager:
    """Factory for creating particle fluid domains, states, and solvers.

    This class provides static methods to create domain, state, and solver
    instances for different particle fluid simulation methods (e.g. WCSPH, PBF).
    """

    @staticmethod
    def create_solver(solver_type: str,
                      model: Any | None = None) -> ParticleFluidSolverBase:
        """Create a particle fluid solver based on the specified type."""
        solver_type = solver_type.upper()
        if solver_type == "PBF":
            return PBFSolver(model)
        if solver_type == "WCSPH":
            return WCSPHSolver(model)
        if solver_type == "DFSPH":
            return DFSPHSolver(model)
        raise ValueError(f"Unsupported solver_type: {solver_type}")

class ParticleFluidDomain(Domain):
    """Common base class for particle fluid domains.

    Subclasses may set _state_cls to a wrapper around Newton State that also
    accepts the domain model in its constructor (e.g. WCSPHState, PBFState).
    """

    # Optional state class used by create_state. Subclasses should override.
    _state_cls = None

    def __init__(
        self,
        particle_q: wp.array,
        particle_qd: wp.array,
        parameters: dict[str, Any],
        model: Any | None = None,
        solver: ParticleFluidSolverBase | None = None,
        model_init_kwargs: dict | None = None,
    ) -> None:

        if "solver_type" not in parameters:
            raise ValueError("solver_type must be specified in parameters")
        self._solver_type = str(parameters["solver_type"]).upper()

        if model is None:
            # Extract particle radius from Newton model
            model = FluidParticleModelManager.create_model(self._solver_type, parameters)
        self._model = model
        # CollisionPipeline.collide_particles() expects an object with shape_count/collide.
        self._model_adapter = self
        self._particle_q_init = particle_q
        self._particle_qd_init = particle_qd

        # Use provided or create default solver
        if solver is None:
            solver = FluidParticleSolverManager.create_solver(self._solver_type, self._model)
        self._solver = solver

        # Double-buffered state (lazily created)
        self._state_in = None
        self._state_out = None

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "fluid_particle"

    @property
    def model_adapter(self) -> Any:
        return self._model_adapter

    @property
    def shape_count(self) -> int:
        """Number of collision shapes available for particle-shape collision."""
        return int(getattr(self._model, "shape_count", 0) or 0)

    def collide(self, state: Any) -> Contacts | None:
        """Run model collision if supported by the underlying model."""
        collide_fn = getattr(self._model, "collide", None)
        if collide_fn is None:
            return None
        return collide_fn(state)

    @property
    def model(self) -> ParticleFluidModel:
        return self._model

    @property
    def solver(self) -> ParticleFluidSolverBase:
        return self._solver

    @property
    def state(self) -> Any:
        """Return a new state for this domain.

        This implements the abstract method required by wanphys._src.core.Domain.
        Most examples and the Newton viewer expect domain.state() to exist.
        """
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> None:
        """Create a new state for this domain.

        If a subclass provides _state_cls it will be constructed with the
        Newton state and the domain model: _state_cls(newton_state, self._model).
        Otherwise the raw Newton state from the adapter is returned.
        """
        self._state_in = FluidParticleStateManager.create_state(self._solver_type, self._particle_q_init, self._particle_qd_init, self._model)
        self._state_out = FluidParticleStateManager.create_state(self._solver_type, self._particle_q_init, self._particle_qd_init, self._model)

    def step(
        self,
        dt: float,
        contacts: Contacts | None = None,
        rigid_state: RigidState | None = None,
        rigid_model: RigidModel | None = None,
    ) -> None:
        """Advance particle-fluid simulation by one timestep.

        Dispatches to the configured solver (PBF/WCSPH/DFSPH).

        Args:
            state_in: Current state.
            state_out: State to write results into.
            dt: Timestep in seconds.
        """
        raw = contacts if contacts is not None else None

        if self._solver_type == "PBF":
            self._solver.step(self._state_in, self._state_out, dt, contacts=raw, rigid_state=rigid_state, rigid_model=rigid_model)
        elif self._solver_type == "WCSPH":
            self._solver.step(self._state_in, self._state_out, dt, contacts=raw)
        elif self._solver_type == "DFSPH":
            self._solver.step(self._state_in, self._state_out, dt, contacts=raw)
        else:
            raise ValueError(f"Unsupported solver type: {self._solver_type}")
        self._state_in, self._state_out = self._state_out, self._state_in

    def pre_step(self, dt: float) -> None:
        """Default pre-step: clear forces if the state supports it."""
        self.state.clear_forces()

    # Generic getters that use common state attribute names
    def get_particle_positions(self, state: Any) -> np.ndarray:
        return state.particle_q.numpy()

    def get_particle_velocities(self, state: Any) -> np.ndarray:
        return state.particle_qd.numpy()

    def get_particle_densities(self, state: Any) -> np.ndarray:
        rho = getattr(state, "rho", None)
        if rho is None:
            raise AttributeError("State does not provide `rho` attribute")
        return rho.numpy()

    def get_particle_pressures(self, state: Any) -> np.ndarray:
        pressure = getattr(state, "pressure", None)
        if pressure is None:
            raise AttributeError("State does not provide `pressure` attribute")
        return pressure.numpy()
