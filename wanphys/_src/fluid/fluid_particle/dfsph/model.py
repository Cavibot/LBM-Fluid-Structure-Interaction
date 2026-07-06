# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""DFSPH model - static configuration for Divergence-Free SPH simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import warp as wp

from ..model import ParticleFluidModel


@dataclass
class DFSPHModel(ParticleFluidModel):
    """Static configuration for Divergence-Free SPH simulation.

    Contains all time-invariant parameters for DFSPH:
    - Physical properties (particle radius, rest density, gravity)
    - Solver controls (neighbor count, iterations, error tolerances)
    - Viscosity and boundary interaction parameters
    - Fixed timestep setting
    - Pre-computed derived quantities (support radius, mass, volume)
    - Optional particle constant arrays for inter-domain/collision pipeline
    """

    # Physical parameters
    particle_radius_scalar: float = 0.025
    """Particle radius scalar used by DFSPH internal parameterization."""

    rest_density: float = 1000.0
    """Rest density of fluid in kg/m³ (water = 1000)."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """Gravity vector (x, y, z). Newton uses Z-up coordinate system."""

    # Solver parameters
    max_neighbors: int = 256
    """Maximum neighbor buffer size."""

    # Viscosity and boundary interaction
    viscosity: float = 0.05
    """Viscosity coefficient."""

    viscosity_error_tolerance: float = 0.2
    """Error tolerance for viscosity solve."""

    viscosity_max_iterations: int = 4
    """Maximum iterations for viscosity solve."""

    # DFSPH solver settings
    divergence_max_iterations: int = 3
    """Maximum iterations for divergence correction."""

    pressure_max_iterations: int = 10
    """Maximum iterations for pressure (density) correction."""

    density_error_tolerance: float = 0.1
    """Density error tolerance for pressure solve."""

    # Fixed timestep
    fixed_dt: float = 0.002
    """Fixed timestep in seconds."""

    use_graph: bool = True
    """Whether to use CUDA graph capture for performance."""

    # Optional particle constant data (for collision bridge compatibility)
    particle_count: int = 0
    """Number of fluid particles (optional, mainly for bridge/collision pipeline)."""

    particle_flags: wp.array | None = None
    """Particle flags array, shape (N,), dtype int32 (optional)."""

    particle_radius: wp.array | None = None
    """Per-particle radius array, shape (N,), dtype float32 (for collision pipeline)."""

    # Device
    device: str | wp.Device | None = None
    """Warp device for GPU arrays."""

    # Computed fields (set in __post_init__)
    _device: wp.Device = field(init=False, repr=False, default=None)
    _support_radius: float = field(init=False, repr=False, default=None)
    _particle_mass: float = field(init=False, repr=False, default=None)
    _particle_volume: float = field(init=False, repr=False, default=None)
    _kernel_mk: float = field(init=False, repr=False, default=None)
    _kernel_ml: float = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize computed fields."""
        super().__post_init__()

        # Basic checks
        if self.particle_radius_scalar <= 0.0:
            raise ValueError("particle_radius_scalar must be > 0")
        if self.rest_density <= 0.0:
            raise ValueError("rest_density must be > 0")
        if self.fixed_dt <= 0.0:
            raise ValueError("fixed_dt must be > 0")
        if self.particle_count < 0:
            raise ValueError("particle_count must be >= 0")

        self._support_radius = self.particle_radius_scalar * 4.0
        self._particle_volume = ((self.particle_radius_scalar * 2.0) ** 3.0) * 0.8
        self._particle_mass = self._particle_volume * self.rest_density

        # Kernel pre-factors
        self._kernel_mk = 8.0 / math.pi
        self._kernel_ml = 48.0 / math.pi

        # Optional array shape checks
        if self.particle_flags is not None:
            if self.particle_flags.shape[0] != self.particle_count:
                raise ValueError("particle_flags shape mismatch with particle_count")

        if self.particle_radius is not None:
            if self.particle_radius.shape[0] != self.particle_count:
                raise ValueError("particle_radius shape mismatch with particle_count")

        if self.particle_count > 0:
            if self.particle_flags is None:
                raise ValueError("particle_flags must be provided when particle_count > 0")
                
            if self.particle_radius is None:
                self.particle_radius = wp.full(
                    self.particle_count,
                    float(self.particle_radius_scalar),
                    dtype=wp.float32,
                    device=self._device,
                    )
            elif self.particle_radius.shape[0] != self.particle_count:
                raise ValueError("particle_radius shape mismatch with particle_count")

    @property
    def support_radius(self) -> float:
        """Support radius used in neighbor queries and kernels."""
        return self._support_radius

    @property
    def particle_mass(self) -> float:
        """Per-particle mass derived from particle volume and rest density."""
        return self._particle_mass

    @property
    def particle_volume(self) -> float:
        """Per-particle volume approximation."""
        return self._particle_volume

    @property
    def kernel_mk(self) -> float:
        """Cubic spline kernel coefficient mk = 8/pi."""
        return self._kernel_mk

    @property
    def kernel_ml(self) -> float:
        """Cubic spline kernel gradient coefficient ml = 48/pi."""
        return self._kernel_ml

    def get_gravity_vec3(self) -> wp.vec3:
        """Get gravity as wp.vec3."""
        return wp.vec3(*self.gravity)
