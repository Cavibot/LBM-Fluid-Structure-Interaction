# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WCSPH state - time-varying data for Weakly Compressible SPH simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..state import ParticleFluidState

if TYPE_CHECKING:
    from .model import WCSPHModel


class WCSPHState(ParticleFluidState):
    """Time-varying state for Weakly Compressible SPH simulation.

    Owns particle data (positions, velocities, forces) directly as wp.arrays,
    independent of Newton State.  Also holds WCSPH-specific scratch buffers:
    - rho: Particle densities
    - pressure: Particle pressures (from Tait EOS)
    - f_sph: SPH forces (pressure + viscosity)
    - f_total: Total forces (external + SPH)
    - dv_xsph: XSPH velocity correction

    The state follows the double-buffering pattern where step() reads from
    state_in and writes to state_out.

    Example:
        >>> state = WCSPHState(n=8000, device="cuda:0", model=wcsph_model)
        >>> print(f"Particles: {state.particle_count}")
    """

    def __init__(self, n: int, device: str | wp.Device, model: WCSPHModel):
        """Initialize WCSPH state.

        Args:
            n: Number of particles.
            device: Warp device for array allocation.
            model: WCSPHModel with configuration parameters.
        """
        super().__init__(n, device)
        self._model = model

        # WCSPH-specific scratch buffers
        self._rho = wp.zeros(n, dtype=wp.float32, device=device)
        self._pressure = wp.zeros(n, dtype=wp.float32, device=device)
        self._f_sph = wp.zeros(n, dtype=wp.vec3, device=device)
        self._f_total = wp.zeros(n, dtype=wp.vec3, device=device)
        self._dv_xsph = wp.zeros(n, dtype=wp.vec3, device=device)

    @property
    def rho(self) -> wp.array:
        """Particle densities array, shape (N,), dtype float32."""
        return self._rho

    @property
    def pressure(self) -> wp.array:
        """Particle pressures array, shape (N,), dtype float32."""
        return self._pressure

    @property
    def f_sph(self) -> wp.array:
        """SPH forces array, shape (N,), dtype vec3."""
        return self._f_sph

    @property
    def f_total(self) -> wp.array:
        """Total forces array, shape (N,), dtype vec3."""
        return self._f_total

    @property
    def dv_xsph(self) -> wp.array:
        """XSPH velocity correction array, shape (N,), dtype vec3."""
        return self._dv_xsph

    def zero_scratch_buffers(self) -> None:
        """Zero all WCSPH scratch buffers."""
        self._rho.zero_()
        self._pressure.zero_()
        self._f_sph.zero_()
        self._f_total.zero_()
        self._dv_xsph.zero_()


