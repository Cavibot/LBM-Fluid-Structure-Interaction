# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fluid coupling modules for cross-domain interactions."""

from wanphys._src.fluid.fluid_grid.coupling.archimedes_buoyancy import (
    ArchimedesBuoyancy,
    ArchimedesBuoyancyConfig,
    ArchimedesBuoyancyResult,
    apply_archimedes_buoyancy,
)
from wanphys._src.fluid.fluid_grid.coupling.grid_lbm_rigid_coupling import (
    GridLbmRigidCoupling,
    LbmFeedbackMode,
    lattice_gravity_to_world,
    open_me_force_conversion,
    recommended_me_force_scale,
)
from wanphys._src.fluid.fluid_grid.coupling.grid_liquid_rigid_coupling import GridLiquidRigidCoupling

__all__: list[str] = [
    "ArchimedesBuoyancy",
    "ArchimedesBuoyancyConfig",
    "ArchimedesBuoyancyResult",
    "GridLbmRigidCoupling",
    "GridLiquidRigidCoupling",
    "LbmFeedbackMode",
    "apply_archimedes_buoyancy",
    "lattice_gravity_to_world",
    "open_me_force_conversion",
    "recommended_me_force_scale",
]
