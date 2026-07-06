from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wanphys._src.fluid.fluid_particle.state import ParticleFluidState


class ParticleFluidSolverBase(ABC):
    """Abstract base class for particle fluid solvers.

    Provides the standard step() interface for time integration.
    """

    @abstractmethod
    def step(
        self,
        state_in: ParticleFluidState,
        state_out: ParticleFluidState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance simulation by dt seconds.

        Args:
            state_in: Current state (read-only).
            state_out: Next state (write target).
            dt: Timestep in seconds.
            contacts: Optional collision contacts from Newton.
            control: Optional control inputs.
        """
        ...