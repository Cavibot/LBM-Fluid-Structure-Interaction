# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body and articulation domain with Newton isolation layer.

This module provides a stable WanPhys API for rigid body simulation,
isolating the Newton dependency to allow future migration to custom
implementations.

Core abstractions:
- RigidModelBuilder: WanPhys-native scene builder
- RigidModel: Scene configuration (bodies, joints, shapes)
- RigidState: Time-varying simulation state owned by WanPhys
- RigidSolver: Physics integration algorithm
- RigidDomain: Domain orchestrator for CompositeSimulation

Example (new API — fully WanPhys):
    >>> from wanphys.rigid import RigidModelBuilder
    >>> builder = RigidModelBuilder(up_axis=2)
    >>> builder.add_ground_plane()
    >>> body = builder.add_body(position=(0, 3, 0), label="box")
    >>> builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
    >>> model = builder.finalize()
    >>> state = model.state()

New code should construct rigid scenes through :class:`RigidModelBuilder`
and pass :class:`RigidModel` / :class:`RigidState` objects across WanPhys
APIs.
"""

from wanphys._src.rigid import (
    MujocoConfig,
    RigidDomain,
    RigidModel,
    RigidModelBuilder,
    RigidSolver,
    RigidState,
    ShapeConfig,
    SolverNotifyFlags,
    SolverXPBD,
    SymplecticEulerSolver,
    WanPhysMujocoSolver,
    WanPhysSequentialImpulseSolver,
    create_mujoco_solver,
    create_semiimplicit_solver,
    create_si_solver,
    create_vbd_solver,
    create_xpbd_solver,
    register_mujoco_custom_attributes,
    register_mujoco_solver_attributes,
)

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
