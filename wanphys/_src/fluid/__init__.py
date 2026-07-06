# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fluid simulation domains.

This module provides fluid simulation using multiple methods:

**Eulerian (Grid-based):**
- `FluidModel`, `FluidState`, `FluidSolver`, `FluidDomain` - MAC grid-based
  incompressible Navier-Stokes simulation

**Lagrangian (Particle-based):**
- `PBFModel`, `PBFState`, `PBFSolver`, `PBFDomain` - Position Based Fluids
- `WCSPHModel`, `WCSPHState`, `WCSPHSolver`, `WCSPHDomain` - Weakly Compressible SPH

Particle-based methods are available via the `fluid_particle` submodule.
"""

# Custom viewer
from .viewer_gl import FluidViewerGL

# Lagrangian (particle-based) fluids
from .fluid_particle import (
    # Base classes
    ParticleFluidDomain,
    ParticleFluidModel,
    ParticleFluidState,
    ParticleFluidSolverBase,
)

# Eulerian (grid-based) fluids
from .fluid_grid import (
    FluidGridMacSolverBase,
    FluidGridModelBase,
    FluidGridSolverBase,
    FluidGridStateBase,
    PressureLinearSolver,
    JacobiPressureSolver,
    PcgPressureSolver,
    MgpcgPressureSolver,
    build_pressure_solver,
    FluidGridModel,
    FluidGridState,
    FluidGridSolver,
    FluidGridDomain,
    FluidGridLiquidDomain,
    FluidGridLiquidModel,
    FluidGridLiquidSolver,
    FluidGridLiquidState,
    LbmDomain,
    LbmModel,
    LbmSolver,
    LbmState,
)

__all__ = [
    "FluidViewerGL",
    # particle
    "ParticleFluidDomain",
    "ParticleFluidModel",
    "ParticleFluidState",
    "ParticleFluidSolverBase",
    # grid base
    "FluidGridMacSolverBase",
    "FluidGridModelBase",
    "FluidGridSolverBase",
    "FluidGridStateBase",
    "PressureLinearSolver",
    "JacobiPressureSolver",
    "PcgPressureSolver",
    "MgpcgPressureSolver",
    "build_pressure_solver",
    # basic vortex
    "FluidGridModel",
    "FluidGridState",
    "FluidGridSolver",
    "FluidGridDomain",
    # liquid
    "FluidGridLiquidModel",
    "FluidGridLiquidDomain",
    "FluidGridLiquidSolver",
    "FluidGridLiquidState",
    # LBM
    "LbmModel",
    "LbmDomain",
    "LbmSolver",
    "LbmState",
]





