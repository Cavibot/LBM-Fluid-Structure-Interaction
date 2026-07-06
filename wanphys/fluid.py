# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Grid-based Eulerian fluid simulation domain."""

from wanphys._src.fluid import (
    FluidGridDomain,
    FluidGridModel,
    FluidGridSolver,
    FluidGridState,
    
    ParticleFluidDomain,
    ParticleFluidModel,
    ParticleFluidSolverBase,
    ParticleFluidState,
)

__all__ = [
    # Grid-based (Eulerian)
    "FluidGridModel",
    "FluidGridState",
    "FluidGridSolver",
    "FluidGridDomain",
    # Particle-based
    "ParticleFluidModel",
    "ParticleFluidState",
    "ParticleFluidSolverBase",
    "ParticleFluidDomain",
]
