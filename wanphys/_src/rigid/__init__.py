# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body domain (internal implementation)."""

from .builder import RigidModelBuilder, ShapeConfig
from .domain import RigidDomain
from .model import RigidModel
from .solver import (
    RigidSolver,
    create_mujoco_solver,
    create_semiimplicit_solver,
    create_si_solver,
    create_vbd_solver,
    create_xpbd_solver,
    register_mujoco_solver_attributes,
)
from .state import RigidState
from .Mujoco import MujocoConfig, SolverNotifyFlags, WanPhysMujocoSolver, register_mujoco_custom_attributes
from .SemiImplicit import SymplecticEulerSolver
from .SI import WanPhysSequentialImpulseSolver
from .xpbd import SolverXPBD

__all__: list[str] = [
    "RigidDomain",
    "RigidModel",
    "RigidModelBuilder",
    "RigidSolver",
    "RigidState",
    "MujocoConfig",
    "ShapeConfig",
    "SolverNotifyFlags",
    "SolverXPBD",
    "SymplecticEulerSolver",
    "WanPhysMujocoSolver",
    "WanPhysSequentialImpulseSolver",
    "create_mujoco_solver",
    "create_semiimplicit_solver",
    "create_si_solver",
    "create_vbd_solver",
    "create_xpbd_solver",
    "register_mujoco_custom_attributes",
    "register_mujoco_solver_attributes",
]
