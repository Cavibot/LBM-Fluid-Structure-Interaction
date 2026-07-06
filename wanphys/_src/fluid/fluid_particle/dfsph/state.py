# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""DFSPH state - time-varying data for Divergence-Free SPH simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..state import ParticleFluidState

if TYPE_CHECKING:
    from .model import DFSPHModel

class DFSPHState(ParticleFluidState):
    """Time-varying state for DFSPH simulation.

    Owns particle data directly as wp.arrays (independent of Newton State),
    and stores DFSPH-specific solver buffers.
    """

    def __init__(self, n: int, device: str | wp.Device, model: DFSPHModel):
        super().__init__(n, device)
        self._model = model

        # Physical / DFSPH arrays
        self._rho = wp.zeros(n, dtype=wp.float32, device=device)
        self._pressure = wp.zeros(n, dtype=wp.float32, device=device)
        self._alpha = wp.zeros(n, dtype=wp.float32, device=device)
        self._kappa = wp.zeros(n, dtype=wp.float32, device=device)
        self._kappa_v = wp.zeros(n, dtype=wp.float32, device=device)
        self._adv_rho = wp.zeros(n, dtype=wp.float32, device=device)

        # Scalar reduction buffers
        self._avg_density_err = wp.zeros(1, dtype=wp.float32, device=device)
        self._density_err_prev = wp.zeros(1, dtype=wp.float32, device=device)
        self._density_err_curr = wp.zeros(1, dtype=wp.float32, device=device)
        self._divergence_err_prev = wp.zeros(1, dtype=wp.float32, device=device)
        self._divergence_err_curr = wp.zeros(1, dtype=wp.float32, device=device)

        # Boundary particles
        self._boundary_q = wp.empty(shape=(0,), dtype=wp.vec3, device=device)
        self._boundary_psi = wp.empty(shape=(0,), dtype=wp.float32, device=device)

        # Fixed dt container for kernels expecting dt_arr
        self._deltaT = wp.zeros(1, dtype=wp.float32, device=device)
        wp.copy(self._deltaT, wp.array([model.fixed_dt], dtype=wp.float32, device=device))

    @classmethod
    def from_arrays(
        cls,
        particle_q: wp.array,
        particle_qd: wp.array,
        model: DFSPHModel,
    ) -> DFSPHState:
        n = particle_q.shape[0]
        device = particle_q.device
        state = cls(n, device, model)
        wp.copy(state._particle_q, particle_q)
        wp.copy(state._particle_qd, particle_qd)
        return state

    @property
    def particle_count(self) -> int:
        return self._n

    @property
    def particle_q(self) -> wp.array:
        return self._particle_q

    @particle_q.setter
    def particle_q(self, value: wp.array):
        self._particle_q = value

    @property
    def particle_qd(self) -> wp.array:
        return self._particle_qd

    @particle_qd.setter
    def particle_qd(self, value: wp.array):
        self._particle_qd = value

    @property
    def particle_f(self) -> wp.array:
        return self._particle_f

    @particle_f.setter
    def particle_f(self, value: wp.array):
        self._particle_f = value

    def clear_forces(self) -> None:
        self._particle_f.zero_()

    @property
    def rho(self) -> wp.array:
        return self._rho

    @property
    def pressure(self) -> wp.array:
        return self._pressure

    @property
    def alpha(self) -> wp.array:
        return self._alpha

    @property
    def kappa(self) -> wp.array:
        return self._kappa

    @property
    def kappa_v(self) -> wp.array:
        return self._kappa_v

    @property
    def adv_rho(self) -> wp.array:
        return self._adv_rho

    @property
    def avg_density_err(self) -> wp.array:
        return self._avg_density_err

    @property
    def density_err_prev(self) -> wp.array:
        return self._density_err_prev

    @property
    def density_err_curr(self) -> wp.array:
        return self._density_err_curr

    @property
    def divergence_err_prev(self) -> wp.array:
        return self._divergence_err_prev

    @property
    def divergence_err_curr(self) -> wp.array:
        return self._divergence_err_curr

    @property
    def boundary_q(self) -> wp.array:
        return self._boundary_q

    @boundary_q.setter
    def boundary_q(self, value: wp.array):
        self._boundary_q = value

    @property
    def boundary_psi(self) -> wp.array:
        return self._boundary_psi

    @boundary_psi.setter
    def boundary_psi(self, value: wp.array):
        self._boundary_psi = value

    @property
    def deltaT(self) -> wp.array:
        return self._deltaT

    def set_fixed_dt(self, dt: float) -> None:
        wp.copy(self._deltaT, wp.array([dt], dtype=wp.float32, device=self._deltaT.device))

    def zero_scratch_buffers(self) -> None:
        self._avg_density_err.zero_()
        self._density_err_prev.zero_()
        self._density_err_curr.zero_()
        self._divergence_err_prev.zero_()
        self._divergence_err_curr.zero_()
