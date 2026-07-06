# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""PBF model - static configuration for Position Based Fluids simulation."""

from __future__ import annotations
import math

from dataclasses import dataclass, field

import warp as wp

from ..model import ParticleFluidModel


@dataclass
class PBFModel(ParticleFluidModel):
    """Static configuration for Position Based Fluids simulation.

    Contains all time-invariant parameters for PBF:
    - Physical properties (radius, density, mass)
    - Solver parameters (iterations, relaxation)
    - Artificial pressure parameters (anti-clustering)
    - Vorticity and viscosity coefficients
    - Boundary friction

    The support radius (smoothing length) is automatically computed as
    4x the particle radius, which is typical for SPH simulations.

    Example:
        >>> model = PBFModel(
        ...     particle_count=fluid_model.particle_count,
        ...     particle_mass=wp.clone(fluid_model.particle_mass),
        ...     particle_flags=wp.clone(fluid_model.particle_flags),
        ...     particle_radius=wp.clone(fluid_model.particle_radius),
        ...     rest_density=1000.0,
        ...     iterations=4,
        ...     relaxation_parameter=0.01,
        ...     vorticity_coefficient=0.0001,
        ...     xsph_c=0.3,
        ... )
    """

    # Particle data (owned by PBFModel, not Newton)
    particle_count: int = 0
    """Number of particles."""

    particle_mass: wp.array | None = None
    """Particle mass array, shape (N,), dtype float32."""

    particle_flags: wp.array | None = None
    """Particle flags array, shape (N,), dtype uint32."""

    particle_radius: wp.array | None = None
    """Particle radius (scalar or array), used by inter-domain collision."""

    rest_density: float = 1000.0
    """Rest density of fluid in kg/m³ (water = 1000)."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """Gravity vector (x, y, z). Newton uses Z-up coordinate system."""

    # PBF solver parameters
    iterations: int = 4
    """Number of constraint solver iterations per timestep."""

    relaxation_parameter: float = 0.01
    """Relaxation parameter epsilon in lambda calculation (Eq. 11)."""

    # Artificial pressure parameters (anti-clustering, Eq. 13)
    artificial_pressure_k: float = 0.0001
    """Coefficient k for artificial pressure term."""

    artificial_pressure_n: float = 4.0
    """Exponent n for artificial pressure term."""

    artificial_pressure_dq_factor: float = 0.3
    """Delta(q) as fraction of support radius for artificial pressure."""

    # Vorticity confinement (Eq. 16)
    vorticity_coefficient: float = 0.0001
    """Vorticity confinement strength epsilon."""

    # XSPH viscosity (Eq. 17)
    xsph_c: float = 0.3
    """XSPH artificial viscosity coefficient c."""

    # Boundary interaction
    boundary_friction: float = 0.01
    """Friction coefficient for particle-boundary contacts."""

    # AABB boundary constraints
    bounds_min: tuple[float, float, float] | None = None
    """Minimum corner of AABB boundary (None = no boundary)."""

    bounds_max: tuple[float, float, float] | None = None
    """Maximum corner of AABB boundary (None = no boundary)."""

    bounds_restitution: float = 0.1
    """Restitution coefficient for boundary collisions."""

    bounds_damping: float = 0.02
    """Velocity damping on boundary collision."""

    # Stability
    max_velocity: float = 100.0
    """Maximum particle velocity to prevent instability."""

    # Solver options
    use_graph: bool = True
    """Whether to use CUDA graph capture for performance."""

    # Device
    device: str | wp.Device | None = None
    """Warp device for GPU arrays."""

    # Computed fields (set in __post_init__)
    _device: wp.Device = field(init=False, repr=False, default=None)
    _support_radius: float = field(init=False, repr=False, default=None)
    _support_radius_sq: float = field(init=False, repr=False, default=None)
    _poly6_coef: float = field(init=False, repr=False, default=None)
    _spiky_grad_coef: float = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize computed fields."""
        super().__post_init__()

        # Computed physical properties
        if self.particle_radius is None:
            raise ValueError("particle_radius must be provided")

        if isinstance(self.particle_radius, wp.array):
            radius_np = self.particle_radius.numpy()
            if radius_np.size == 0:
                raise ValueError("particle_radius array is empty")
            radius_value = float(radius_np[0])
        else:
            radius_value = float(self.particle_radius)

        if radius_value <= 0.0:
            raise ValueError("particle_radius must be positive")

        # Pre-compute kernel coefficients
        self._support_radius = radius_value * 4.0
        self._support_radius_sq = self._support_radius * self._support_radius
        h_6 = self._support_radius_sq * self._support_radius_sq * self._support_radius_sq
        h_9 = h_6 * self._support_radius_sq * self._support_radius
        self._poly6_coef = 315.0 / (64.0 * math.pi * h_9)
        self._spiky_grad_coef = -45.0 / (math.pi * h_6)


    @property
    def support_radius(self) -> float:
        """SPH kernel support radius (4x particle radius)."""
        return self._support_radius
    
    @property
    def support_radius_sq(self) -> float:
        """Squared SPH kernel support radius."""
        return self._support_radius_sq
    
    @property
    def poly6_coef(self) -> float:
        """Pre-computed Poly6 kernel coefficient."""
        return self._poly6_coef

    @property
    def spiky_grad_coef(self) -> float:
        """Pre-computed Spiky gradient coefficient."""
        return self._spiky_grad_coef

    @property
    def d_q(self) -> float:
        """Delta(q) distance for artificial pressure calculation."""
        return self._support_radius * self.artificial_pressure_dq_factor

