from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import warp as wp

from wanphys._src.fluid.fluid_particle.state import ParticleFluidState


@dataclass
class ParticleFluidModel(ABC):
    """Abstract base class for particle fluid model configuration.

    A ParticleFluidModel contains all time-invariant parameters for particle-based
    fluid simulation: physical properties, solver parameters, and boundary settings.

    Subclasses (PBFModel, WCSPHModel) add method-specific parameters.
    """

    def __init__(self):
        # Physical parameters
        self.particle_radius: float = 0.01
        """Particle radius in world units."""

        self.rest_density: float = 1000.0
        """Rest density of fluid in kg/m³ (water = 1000)."""

        self.gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        """Gravity vector (x, y, z)."""

        # Device
        self.device: str | wp.Device | None = None
        """Warp device for GPU arrays."""


        # Newton model particle parameters
        self.requires_grad = False
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        self.num_worlds = 0
        """Number of worlds added to the ModelBuilder."""

        self.particle_q = None
        """Particle positions, shape [particle_count, 3], float."""
        self.particle_qd = None
        """Particle velocities, shape [particle_count, 3], float."""
        self.particle_mass = None
        """Particle mass, shape [particle_count], float."""
        self.particle_inv_mass = None
        """Particle inverse mass, shape [particle_count], float."""
        self.particle_radius = None
        """Particle radius, shape [particle_count], float."""
        self.particle_max_radius = 0.0
        """Maximum particle radius (useful for HashGrid construction)."""
        self.particle_ke = 1.0e3
        """Particle normal contact stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kd = 1.0e2
        """Particle normal contact damping (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kf = 1.0e2
        """Particle friction force stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_mu = 0.5
        """Particle friction coefficient."""
        self.particle_cohesion = 0.0
        """Particle cohesion strength."""
        self.particle_adhesion = 0.0
        """Particle adhesion strength."""
        self.particle_grid: wp.HashGrid | None = None
        """HashGrid instance for accelerated simulation of particle interactions."""
        self.particle_flags: wp.array | None = None
        """Particle enabled state, shape [particle_count], int."""
        self.particle_max_velocity: float = 1e5
        """Maximum particle velocity (to prevent instability)."""
        self.particle_world_ids: wp.array | None = None
        """World index for each particle, shape [particle_count], int. -1 for global."""

        self.particle_count = 0
        """Total number of particles in the system."""

        # indices of particles sharing the same color
        self.particle_color_groups = []
        """Coloring of all particles for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of particles sharing the same color."""
        self.particle_colors = None
        """Color assignment for every particle."""

    def state(self, requires_grad: bool | None = None) -> ParticleFluidState:
        """
        Create and return a new :class:`State` object for this model.

        The returned state is initialized with the initial configuration from the model description.

        Args:
            requires_grad (bool, optional): Whether the state variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.

        Returns:
            State: The state object
        """

        s = ParticleFluidState()
        if requires_grad is None:
            requires_grad = self.requires_grad

        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        return s

    def __post_init__(self):
        """Initialize computed fields."""
        if self.device is None:
            self._device = wp.get_device()
        elif isinstance(self.device, str):
            self._device = wp.get_device(self.device)
        else:
            self._device = self.device

        # Allocate particle_world_ids array needed by rigid-fluid collision kernels.
        # Default world index 0 is compatible with single-world scenarios.
        if not hasattr(self, "particle_world_ids") or self.particle_world_ids is None:
            if self.particle_count > 0:
                self.particle_world_ids = wp.zeros(self.particle_count, dtype=wp.int32, device=self._device)
            else:
                self.particle_world_ids = None


    @property
    def poly6_coef(self) -> float:
        """Pre-computed Poly6 kernel coefficient."""
        return self._poly6_coef

    @property
    def spiky_grad_coef(self) -> float:
        """Pre-computed Spiky gradient coefficient."""
        return self._spiky_grad_coef

    @property
    def visc_lap_coef(self) -> float:
        """Pre-computed viscosity Laplacian coefficient."""
        return self._visc_lap_coef

    @property
    def bounds_enabled(self) -> bool:
        """Whether AABB boundaries are enabled."""
        return self.bounds_min is not None and self.bounds_max is not None

    def get_bounds_min_vec3(self) -> wp.vec3:
        """Get bounds_min as wp.vec3."""
        if self.bounds_min is None:
            return wp.vec3(0.0, 0.0, 0.0)
        return wp.vec3(*self.bounds_min)

    def get_bounds_max_vec3(self) -> wp.vec3:
        """Get bounds_max as wp.vec3."""
        if self.bounds_max is None:
            return wp.vec3(0.0, 0.0, 0.0)
        return wp.vec3(*self.bounds_max)

    def get_gravity_vec3(self) -> wp.vec3:
        """Get gravity as wp.vec3."""
        return wp.vec3(*self.gravity)
