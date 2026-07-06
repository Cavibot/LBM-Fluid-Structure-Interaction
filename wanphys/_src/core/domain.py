# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Base protocols and abstract classes for simulation domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import warp as wp


@runtime_checkable
class DomainModel(Protocol):
    """Protocol for static domain configuration.

    A DomainModel contains all time-invariant data for a simulation domain:
    geometry, material properties, constraints, etc.
    """

    @property
    def device(self) -> wp.Device:
        """Device where model data is allocated."""
        ...


@runtime_checkable
class DomainState(Protocol):
    """Protocol for time-varying domain state.

    A DomainState contains all dynamic data that evolves during simulation:
    positions, velocities, forces, etc.
    """

    def clear_forces(self) -> None:
        """Reset accumulated forces to zero."""
        ...


@runtime_checkable
class DomainSolver(Protocol):
    """Protocol for domain solver.

    A DomainSolver advances the simulation state by one timestep.
    """

    def step(
        self,
        state_in: DomainState,
        state_out: DomainState,
        dt: float,
        **kwargs: Any,
    ) -> None:
        """Advance simulation by dt seconds.

        Args:
            state_in: Current state (read-only).
            state_out: Next state (write target).
            dt: Timestep in seconds.
            **kwargs: Additional solver-specific arguments.
        """
        ...


class Domain(ABC):
    """Abstract base class for a simulation domain.

    A Domain encapsulates a complete simulation subsystem with its own
    Model (static config), State (dynamic data), and Solver (time integration).
    The domain owns and manages its internal double-buffered state.

    Each physics team implements their own Domain subclass in their
    wanphys/[topic]/ directory.

    Example:
        >>> class MyFluidDomain(Domain):
        ...     @property
        ...     def name(self) -> str:
        ...         return "fluid"
        ...
        ...     @property
        ...     def state(self) -> FluidState:
        ...         return self._state_in
        ...
        ...     def create_state(self) -> None:
        ...         self._state_in = FluidState(self.model)
        ...         self._state_out = FluidState(self.model)
        ...
        ...     def step(self, dt, contacts=None):
        ...         self.solver.step(self._state_in, self._state_out, dt)
        ...         self._state_in, self._state_out = self._state_out, self._state_in
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this domain (e.g., 'rigid', 'fluid')."""
        ...

    @property
    @abstractmethod
    def model(self) -> DomainModel:
        """The domain's static configuration."""
        ...

    @property
    @abstractmethod
    def state(self) -> DomainState:
        """Current simulation state (active buffer)."""
        ...

    @abstractmethod
    def create_state(self) -> None:
        """Initialize internal double-buffered states from the model.

        Call this before the first step, or to reset the simulation.
        After this, the current state is accessible via the ``state`` property.
        """
        ...

    @abstractmethod
    def step(
        self,
        dt: float,
        contacts=None,
    ) -> None:
        """Advance the domain simulation by one timestep.

        The domain reads from its current state buffer, writes to the back
        buffer, and swaps them internally.

        Args:
            dt: Timestep in seconds.
            contacts: Optional DomainContacts from CollisionPipeline.  When
                provided, the domain uses the pre-computed contacts rather than
                running its own collision detection.  Pass None to let the
                domain fall back to its built-in collision detection (default,
                backward-compatible behaviour).
        """
        ...

    def pre_step(self, dt: float) -> None:
        """Hook called before step().

        Override to apply external forces, update controls, etc.

        Args:
            dt: Upcoming timestep.
        """
        pass

    def post_step(self, dt: float) -> None:
        """Hook called after step().

        Override to compute derived quantities, update visualizations, etc.

        Args:
            dt: Timestep that was taken.
        """
        pass
