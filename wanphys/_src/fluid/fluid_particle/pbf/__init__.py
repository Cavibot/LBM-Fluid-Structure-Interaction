# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Position Based Fluids (PBF) domain implementation.

PBF is a constraint-based particle fluid method that enforces incompressibility
through iterative position corrections. Key features:
- Stable and robust for real-time applications
- Supports vorticity confinement for swirling effects
- XSPH viscosity for smooth velocity fields
- Rigid-fluid coupling via Newton collision detection

Reference:
    Macklin, M., & Müller, M. (2013). Position based fluids.
    ACM Transactions on Graphics (TOG), 32(4), 1-12.
"""

from .model import PBFModel
from .state import PBFState
from .solver import PBFSolver

__all__ = [
    "PBFModel",
    "PBFState",
    "PBFSolver",
]
