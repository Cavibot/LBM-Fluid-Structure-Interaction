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
    HomeLbmModel,
    HomeLbmDomain,
    HomeLbmSolver,
    HomeLbmState,
    HomeFreeBackend,
    HomeFreeDomain,
    HomeFreeLegacyDomain,
    create_home_free_domain,
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
    # HOME-LBM and geometric HOME-Free
    "HomeLbmModel",
    "HomeLbmDomain",
    "HomeLbmSolver",
    "HomeLbmState",
    "HomeFreeBackend",
    "HomeFreeDomain",
    "HomeFreeLegacyDomain",
    "create_home_free_domain",
]
