# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Weakly Compressible SPH (WCSPH) domain implementation.

WCSPH is a force-based particle fluid method that uses the Tait equation of state
to compute pressure forces. Key features:
- Simple and efficient force-based method
- Tait EOS for weakly compressible behavior
- XSPH velocity smoothing
- AABB boundary constraints
- Rigid-fluid coupling via Newton collision detection

Reference:
    Becker, M., & Teschner, M. (2007). Weakly compressible SPH for free surface flows.
    Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation.
"""

from .model import WCSPHModel
from .state import WCSPHState
from .solver import WCSPHSolver

__all__ = [
    "WCSPHModel",
    "WCSPHState",
    "WCSPHSolver",
]
