# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""PBF state - time-varying data for Position Based Fluids simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..state import ParticleFluidState

if TYPE_CHECKING:
    from .model import PBFModel


class PBFState(ParticleFluidState):
    """Time-varying state for Position Based Fluids simulation.

    Wraps Newton State and adds PBF-specific scratch buffers:
    - particle_lambdas: Constraint multipliers (Eq. 9)
    - particle_position_deltas: Position corrections (Eq. 12)
    - particle_vorticities: Vorticity vectors (Eq. 15)
    - particle_q_init: Initial positions for velocity update

    The state follows the double-buffering pattern where step() reads from
    state_in and writes to state_out.

    Example:
        >>> # Direct construction from particle arrays
        >>> state = PBFState.from_arrays(fluid_model.particle_q, fluid_model.particle_qd, model)
        >>>
        >>> # In typical use (dam-break), state is managed by PBFDomain
        >>> pbf_domain = PBFDomain(fluid_model.particle_q, fluid_model.particle_qd, model)
        >>> pbf_domain.create_state()
        >>> state = pbf_domain.state
    """

    def __init__(self, n: int, device: str | wp.Device, model: PBFModel):
        """Initialize PBF state.

        Args:
            n: Number of particles.
            device: Device for tensor allocation.
            model: PBFModel with configuration parameters.
        """
        super().__init__(n, device)
        self._model = model

        # PBF-specific scratch buffers
        self._particle_lambdas = wp.zeros(n, dtype=wp.float32, device=device)
        self._particle_position_deltas = wp.zeros(n, dtype=wp.vec3, device=device)
        self._particle_vorticities = wp.zeros(n, dtype=wp.vec3, device=device)
        self._particle_q_init = wp.zeros(n, dtype=wp.vec3, device=device)

    @classmethod
    def from_arrays(
        cls,
        q: wp.array,
        qd: wp.array,
        model: PBFModel,
    ) -> PBFState:
        """Create a PBFState initialized from position/velocity arrays.

        The arrays are copied so the state owns its data independently.

        Args:
            q: Initial particle positions, shape (N,), dtype vec3.
            qd: Initial particle velocities, shape (N,), dtype vec3.
            model: PBFModel with configuration parameters.

        Returns:
            A new PBFState with positions and velocities copied from the inputs.
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
        """Particle positions array."""
        return self._particle_q

    @property
    def particle_qd(self) -> wp.array:
        """Particle velocities array."""
        return self._particle_qd

    @property
    def particle_f(self) -> wp.array:
        """Particle force accumulator array."""
        return self._particle_f

    @property
    def particle_lambdas(self) -> wp.array:
        """Constraint multipliers array, shape (N,), dtype float32."""
        return self._particle_lambdas

    @property
    def particle_position_deltas(self) -> wp.array:
        """Position corrections array, shape (N,), dtype vec3."""
        return self._particle_position_deltas

    @property
    def particle_vorticities(self) -> wp.array:
        """Vorticity vectors array, shape (N,), dtype vec3."""
        return self._particle_vorticities

    @property
    def particle_q_init(self) -> wp.array:
        """Initial positions backup array, shape (N,), dtype vec3."""
        return self._particle_q_init

    def zero_scratch_buffers(self) -> None:
        """Zero all PBF scratch buffers."""
        self._particle_lambdas.zero_()
        self._particle_position_deltas.zero_()
        self._particle_vorticities.zero_()

    def clear_forces(self) -> None:
        """Reset accumulated forces to zero."""
        self._particle_f.zero_()

    def backup_positions(self) -> None:
        """Copy current positions to backup array for velocity update."""
        wp.copy(self._particle_q_init, self.particle_q)
