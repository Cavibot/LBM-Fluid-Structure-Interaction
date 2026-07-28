# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fluid coupling modules for cross-domain interactions."""

from wanphys._src.fluid.fluid_grid.coupling.grid_lbm_rigid_coupling import (
    GridLbmRigidCoupling,
    LbmFeedbackMode,
    recommended_me_force_scale,
)
from wanphys._src.fluid.fluid_grid.coupling.grid_liquid_rigid_coupling import GridLiquidRigidCoupling

__all__: list[str] = [
    "GridLbmRigidCoupling",
    "GridLiquidRigidCoupling",
    "LbmFeedbackMode",
    "recommended_me_force_scale",
]
