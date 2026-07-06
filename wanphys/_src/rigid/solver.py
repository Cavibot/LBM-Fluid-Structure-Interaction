# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body solver - isolates Newton solver dependency."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING, Any, Type

from newton.solvers import SolverBase as NewtonSolverBase
from newton.solvers import SolverXPBD

if TYPE_CHECKING:
    from .builder import RigidModelBuilder
    from .model import RigidModel
    from .state import RigidState


__all__: list[str] = [
    "RigidSolver",
    "create_mujoco_solver",
    "create_semiimplicit_solver",
    "create_si_solver",
    "create_vbd_solver",
    "create_xpbd_solver",
    "register_mujoco_solver_attributes",
]


class RigidSolver(ABC):
    """Abstract base class for rigid body physics solvers.

    Solvers perform time integration of rigid body dynamics, including:
    - Constraint solving (joints, contacts)
    - Force/torque integration
    - Velocity and position updates

    Different solver implementations offer different accuracy/performance tradeoffs:
    - XPBD (Extended Position-Based Dynamics) - fast, stable
    - VBD (Velocity-Based Dynamics) - more accurate
    - Semi-implicit Euler - simple, fast
    - Featherstone - exact for articulated systems
    """

    @abstractmethod
    def step(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        """Advance simulation by one timestep.

        Args:
            state_in: Current state (input, read-only).
            state_out: Next state (output, written to).
            control: Control inputs (joint targets, actuators).
            contacts: Collision contact data from collider.
            dt: Timestep in seconds.
        """
        pass

    def update_contacts(self, contacts: Any, state: RigidState | None = None) -> None:
        """Update contact force/details from the solver's internal contact state."""
        raise NotImplementedError(f"{type(self).__name__} does not support contact updates")

    def get_max_contact_count(self, default: int | None = None) -> int | None:
        """Return solver contact capacity when the backend exposes one."""
        return default


class _NewtonSolverAdapter(RigidSolver):
    """Internal adapter wrapping Newton's solver implementations.

    Provides WanPhys RigidSolver interface while delegating to Newton solvers.
    This allows future replacement of Newton without changing client code.

    Args:
        model: RigidModel instance.
        newton_solver_cls: Newton solver class (e.g., SolverXPBD, SolverVBD).
        **solver_kwargs: Additional arguments passed to Newton solver constructor.

    Client code should use public WanPhys factories such as
    :func:`create_xpbd_solver` instead of constructing this adapter directly.
    """

    def __init__(
        self,
        model: RigidModel,
        newton_solver_cls: Type[NewtonSolverBase] = SolverXPBD,
        **solver_kwargs: Any,
    ) -> None:
        """Initialize Newton solver adapter.

        Args:
            model: RigidModel instance.
            newton_solver_cls: Newton solver class to wrap.
            **solver_kwargs: Solver-specific parameters.
        """
        # Extract Newton backend to pass to solver
        self._backend = newton_solver_cls(model._newton_backend, **solver_kwargs)
        self._update_contacts_accepts_state: bool = (
            len(inspect.signature(self._backend.update_contacts).parameters) > 1
        )

    def step(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        """Advance simulation using Newton solver.

        Args:
            state_in: Current state.
            state_out: Next state.
            control: Control inputs.
            contacts: Contact data.
            dt: Timestep in seconds.
        """
        # Unwrap states to pass Newton's internal State.
        newton_state_in = state_in.as_newton_state()
        newton_state_out = state_out.as_newton_state()
        self._backend.step(
            newton_state_in,
            newton_state_out,
            control,
            contacts,
            dt,
        )
        state_out._sync_from_newton_state()

    def update_contacts(self, contacts: Any, state: RigidState | None = None) -> None:
        """Update contacts through the wrapped Newton solver."""
        if state is not None and self._update_contacts_accepts_state:
            self._backend.update_contacts(contacts, state.as_newton_state())
        else:
            self._backend.update_contacts(contacts)

    def get_max_contact_count(self, default: int | None = None) -> int | None:
        """Return Newton solver contact capacity without exposing the backend."""
        get_max_contact_count = getattr(self._backend, "get_max_contact_count", None)
        if get_max_contact_count is None:
            return default
        try:
            return int(get_max_contact_count())
        except NotImplementedError:
            return default

    @property
    def _newton_backend(self) -> NewtonSolverBase:
        """Access underlying Newton solver (for migration/debugging).

        Warning:
            Temporary escape hatch. Will break when Newton is replaced.
        """
        return self._backend


# Convenience factory functions for common solvers
def create_xpbd_solver(model: RigidModel, **kwargs: Any) -> RigidSolver:
    """Create WanPhys-native XPBD solver (fast, stable, default choice).

    Args:
        model: RigidModel instance.
        **kwargs: XPBD parameters (iterations, relaxation, etc.).

    Returns:
        WanPhys-native XPBD solver.
    """
    from .xpbd import SolverXPBD as WanPhysSolverXPBD

    return WanPhysSolverXPBD(model, **kwargs)


def create_vbd_solver(model: RigidModel, **kwargs: Any) -> RigidSolver:
    """Create VBD solver (more accurate, slower).

    Args:
        model: RigidModel instance.
        **kwargs: VBD parameters.

    Returns:
        RigidSolver backed by VBD.
    """
    from newton.solvers import SolverVBD

    return _NewtonSolverAdapter(model, SolverVBD, **kwargs)


def create_semiimplicit_solver(model: RigidModel, **kwargs: Any) -> RigidSolver:
    """Create semi-implicit Euler solver (simple, fast).

    Args:
        model: RigidModel instance.
        **kwargs: Solver parameters.

    Returns:
        WanPhys-native semi-implicit Euler solver.
    """
    from .SemiImplicit import SymplecticEulerSolver

    return SymplecticEulerSolver(model, **kwargs)


def create_si_solver(model: RigidModel, **kwargs: Any) -> RigidSolver:
    """Create a WanPhys-native sequential impulse solver."""
    from .SI import WanPhysSequentialImpulseSolver

    return WanPhysSequentialImpulseSolver(model, **kwargs)


def create_mujoco_solver(model: RigidModel, **kwargs: Any) -> RigidSolver:
    """Create a MuJoCo-backed rigid solver without exposing Newton classes."""
    from .Mujoco import WanPhysMujocoSolver

    return WanPhysMujocoSolver(model, **kwargs)


def register_mujoco_solver_attributes(builder: RigidModelBuilder) -> None:
    """Register MuJoCo-specific custom attributes on a rigid builder."""
    from .Mujoco import register_mujoco_custom_attributes

    register_mujoco_custom_attributes(builder)
