from __future__ import annotations

from abc import ABC
from typing import Any

import warp as wp


class ParticleFluidState(ABC):
    """Abstract base class for particle fluid state.

    A ParticleFluidState contains all time-varying data for particle-based
    fluid simulation: positions, velocities, forces, and scratch buffers.

    Subclasses can either own particle buffers directly, or adapt an external state.
    """

    def __init__(self, n: int, device: str | wp.Device):
        """Initialize common particle buffers shared by particle-fluid states."""
        self._n = n
        self._particle_q = wp.zeros(n, dtype=wp.vec3, device=device)
        self._particle_qd = wp.zeros(n, dtype=wp.vec3, device=device)
        self._particle_f = wp.zeros(n, dtype=wp.vec3, device=device)

    @classmethod
    def from_arrays(
        cls,
        q: wp.array,
        qd: wp.array,
        model: Any,
    ) -> ParticleFluidState:
        """Create a WCSPHState initialized from position/velocity arrays.

        The arrays are copied so the state owns its data independently.

        Args:
            q: Initial particle positions, shape (N,), dtype vec3.
            qd: Initial particle velocities, shape (N,), dtype vec3.
            model: WCSPHModel with configuration parameters.

        Returns:
            A new WCSPHState with positions and velocities copied from the inputs.
        """
        n = q.shape[0]
        device = q.device
        state = cls(n, device, model)
        wp.copy(state._particle_q, q)
        wp.copy(state._particle_qd, qd)
        return state

    @property
    def particle_count(self) -> int:
        """Number of particles."""
        return self._n

    @property
    def particle_q(self) -> wp.array:
        """Particle positions array, shape (N,), dtype vec3."""
        return self._particle_q

    @particle_q.setter
    def particle_q(self, value: wp.array):
        self._particle_q = value

    @property
    def particle_qd(self) -> wp.array:
        """Particle velocities array, shape (N,), dtype vec3."""
        return self._particle_qd

    @particle_qd.setter
    def particle_qd(self, value: wp.array):
        self._particle_qd = value

    @property
    def particle_f(self) -> wp.array:
        """Particle forces array, shape (N,), dtype vec3."""
        return self._particle_f

    @particle_f.setter
    def particle_f(self, value: wp.array):
        self._particle_f = value

    def clear_forces(self) -> None:
        """Reset accumulated forces to zero."""
        self._particle_f.zero_()
