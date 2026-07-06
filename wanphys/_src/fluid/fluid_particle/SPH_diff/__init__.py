# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH_diff particle fluid domain (WCSPH-first differentiable baseline)."""

from .diff_composite import DiffCompositeSimulation
from .domain import SPHDiffDomain
from .DFSPH_diff import DFSPHDiffDomain, DFSPHDiffSolver, DFSPHDiffState
from .kernels import MaterialMarks, MaterialType
from .model import SPHDiffModel
from .solver import SPHDiffSolver
from .state import SPHDiffState
from .time_manager import estimate_cfl_dt
from .tasks import SkipStoneTask

__all__ = [
    "SPHDiffModel",
    "SPHDiffState",
    "SPHDiffSolver",
    "SPHDiffDomain",
    "DFSPHDiffState",
    "DFSPHDiffSolver",
    "DFSPHDiffDomain",
    "DiffCompositeSimulation",
    "estimate_cfl_dt",
    "MaterialMarks",
    "MaterialType",
    "SkipStoneTask",
]
