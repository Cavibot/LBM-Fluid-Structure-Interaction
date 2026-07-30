# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Volume-of-fluid state, geometry, and visualization building blocks."""

from .advection import VofMassTransport, VofMassTransportResult
from .contracts import VofCellType, VofMassScheme
from .diagnostics import (
    VofDiagnostics,
    collect_vof_diagnostics,
    validate_vof_diagnostics,
)
from .geometry import (
    InterfaceGeometry,
    VofInterfaceGeometry,
    plic_cube_volume,
    plic_plane_offset,
    validate_authoritative_geometry,
)
from .kinetic_init import (
    VofKineticInitializationResult,
    VofKineticInitializer,
)
from .state import DebugMockScToVofState, VofGridState
from .surface import VofSurfaceBoundary
from .transition import VofTopologyTransition, VofTransitionResult
from .visualization import DebugVofView, InterfaceVisualData, VofInterfaceVisualizer

__all__ = [
    "DebugMockScToVofState",
    "DebugVofView",
    "InterfaceGeometry",
    "InterfaceVisualData",
    "VofCellType",
    "VofDiagnostics",
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
    "collect_vof_diagnostics",
    "validate_vof_diagnostics",
]
