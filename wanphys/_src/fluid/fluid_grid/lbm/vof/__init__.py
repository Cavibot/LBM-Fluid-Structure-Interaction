# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Volume-of-fluid state, geometry, and visualization building blocks."""

from .advection import VofMassTransport, VofMassTransportResult
from .contracts import VofCellType, VofMassScheme
from .geometry import InterfaceGeometry
from .state import DebugMockScToVofState, VofGridState
from .surface import VofSurfaceBoundary
from .visualization import DebugVofView, InterfaceVisualData, VofInterfaceVisualizer

__all__ = [
    "DebugMockScToVofState",
    "DebugVofView",
    "InterfaceGeometry",
    "InterfaceVisualData",
    "VofCellType",
    "VofGridState",
    "VofInterfaceVisualizer",
    "VofMassScheme",
    "VofMassTransport",
    "VofMassTransportResult",
    "VofSurfaceBoundary",
]
