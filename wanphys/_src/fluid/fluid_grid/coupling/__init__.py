# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fluid coupling modules for cross-domain interactions."""

from .grid_lbm_rigid_coupling import GridLbmRigidCoupling
from .grid_liquid_rigid_coupling import GridLiquidRigidCoupling

__all__: list[str] = [
    "GridLbmRigidCoupling",
    "GridLiquidRigidCoupling",
]
