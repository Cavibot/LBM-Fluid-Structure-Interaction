# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 Lattice Boltzmann Method fluid solver."""

from .contracts import (
    BoundaryModel,
    CapabilityStatus,
    Collision,
    CollisionContext,
    CollisionSpace,
    Encoding,
    ForceModel,
    InterfaceModel,
)
from .domain import LbmDomain
from .model import LbmModel
from .solver import LbmSolver
from .state import FullFLbmState, HomeLbmState, LbmState, LbmStateBase
from .vof import (
    DebugMockScToVofState,
    DebugVofView,
    InterfaceGeometry,
    InterfaceVisualData,
    VofCellType,
    VofGridState,
    VofInterfaceGeometry,
    VofInterfaceVisualizer,
    VofKineticInitializationResult,
    VofKineticInitializer,
    VofMassScheme,
    VofMassTransport,
    VofMassTransportResult,
    VofSurfaceBoundary,
    VofTopologyTransition,
    VofTransitionResult,
    plic_cube_volume,
    plic_plane_offset,
    validate_authoritative_geometry,
)

__all__ = [
    "BoundaryModel",
    "CapabilityStatus",
    "Collision",
    "CollisionContext",
    "CollisionSpace",
    "DebugMockScToVofState",
    "DebugVofView",
    "Encoding",
    "ForceModel",
    "FullFLbmState",
    "HomeLbmState",
    "InterfaceGeometry",
    "InterfaceModel",
    "InterfaceVisualData",
    "LbmDomain",
    "LbmModel",
    "LbmSolver",
    "LbmState",
    "LbmStateBase",
    "VofCellType",
    "VofGridState",
    "VofInterfaceVisualizer",
    "VofInterfaceGeometry",
    "VofKineticInitializationResult",
    "VofKineticInitializer",
    "VofMassScheme",
    "VofMassTransport",
    "VofMassTransportResult",
    "VofSurfaceBoundary",
    "VofTopologyTransition",
    "VofTransitionResult",
    "plic_cube_volume",
    "plic_plane_offset",
    "validate_authoritative_geometry",
]
