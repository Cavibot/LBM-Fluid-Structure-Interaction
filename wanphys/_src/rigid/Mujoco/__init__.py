# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys MuJoCo solver package."""

from .attributes import register_mujoco_custom_attributes
from .config import MujocoConfig
from .flags import SolverNotifyFlags
from .solver_mujoco import WanPhysMujocoSolver

__all__ = ["MujocoConfig", "SolverNotifyFlags", "WanPhysMujocoSolver", "register_mujoco_custom_attributes"]
