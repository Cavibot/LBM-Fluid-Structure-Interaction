# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WCSPH model - static configuration for Weakly Compressible SPH simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import warp as wp

from ..model import ParticleFluidModel


@dataclass
class WCSPHModel(ParticleFluidModel):
    """Static configuration for Weakly Compressible SPH simulation.

    Contains all time-invariant parameters for WCSPH:
    - Physical properties (smoothing length, density, viscosity)
    - Tait equation of state parameters (c0, gamma)
    - XSPH velocity smoothing coefficient
    -
    - Pre-computed kernel coefficients

    Example:
        >>> model = WCSPHModel(
        ...     h=0.08,
        ...     rest_density=1000.0,
        ...     viscosity=2.0,
        ...     bounds_min=(-1.0, 0.0, -1.0),
        ...     bounds_max=(1.0, 2.0, 1.0),
        ... )
    """

    # SPH parameters
    h: float = 0.08
    """Smoothing length (support radius) in world units."""

    rest_density: float = 1000.0
    """Rest density of fluid in kg/m³ (water = 1000)."""

    viscosity: float = 2.0
    """Dynamic viscosity coefficient."""

    # Tait equation of state parameters
    c0: float = 20.0
    """Speed of sound for Tait EOS (controls stiffness)."""

    gamma: float = 7.0
    """Exponent for Tait EOS (typically 7 for water)."""

    # Gravity
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """Gravity vector (x, y, z). Newton uses Z-up coordinate system."""

    # XSPH velocity smoothing
    xsph_c: float = 0.1
    """XSPH smoothing coefficient (0 = disabled)."""

    # Unified rigid-fluid contact response parameters
    contact_margin: float = 0.002 #接触边界阈值
    contact_max_push_frac: float = 0.30 #最大推力比例，防止流体推力过大导致刚体被异常推飞
    contact_vel_damp: float = 0.015 #接触速度阻尼
    contact_friction: float = 0.05 #接触摩擦系数

    # Solver options
    clear_external_forces: bool = False
    """Whether to clear external forces each step."""

    use_graph: bool = True
    """Whether to use CUDA graph capture for performance."""

    # Stability
    max_velocity: float = 100.0
    """Maximum particle velocity to prevent instability."""

    # Particle data (owned by WCSPHModel, not Newton)
    particle_count: int = 0
    """Number of particles."""

    particle_mass: wp.array | None = None
    """Particle mass array, shape (N,), dtype float32."""

    particle_flags: wp.array | None = None
    """Particle flags array, shape (N,), dtype uint32."""

    particle_radius: wp.array | None = None
    """Particle radius array, shape (N,), dtype float32. Used by inter-domain collision."""

    # Device
    device: str | wp.Device | None = None
    """Warp device for GPU arrays."""

    # Computed fields (set in __post_init__)
    _device: wp.Device = field(init=False, repr=False, default=None)
    _poly6_coef: float = field(init=False, repr=False, default=None)
    _spiky_grad_coef: float = field(init=False, repr=False, default=None)
    _visc_lap_coef: float = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize computed fields."""
        super().__post_init__()

        # Pre-compute kernel coefficients
        h = self.h
        h6 = h ** 6
        h9 = h6 * (h ** 3)
        self._poly6_coef = 315.0 / (64.0 * math.pi * h9)
        self._spiky_grad_coef = -45.0 / (math.pi * h6)
        self._visc_lap_coef = 45.0 / (math.pi * h6)


