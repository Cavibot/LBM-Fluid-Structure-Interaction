# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle-based fluid simulation domains (Lagrangian methods).

This module provides particle-based fluid simulation using SPH (Smoothed Particle
Hydrodynamics) methods:

- **PBF (Position Based Fluids)**: Constraint-based method for incompressible fluids
- **WCSPH (Weakly Compressible SPH)**: Force-based method using Tait equation of state

Both methods support rigid-fluid coupling via Newton's collision detection system.
"""

from .domain import ParticleFluidDomain
from .model import (
    ParticleFluidModel,
)
from .solver import ParticleFluidSolverBase
from .state import (
    ParticleFluidState,
)

from .builder import (
    ParticleFluidBuilder,
    ParticleFluidData,
)

__all__ = [
    # Base classes
    "ParticleFluidDomain",
    "ParticleFluidModel",
    "ParticleFluidState",
    "ParticleFluidSolverBase",
    # Builder
    "ParticleFluidBuilder",
    "ParticleFluidData",
]
