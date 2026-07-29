# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Volume-of-fluid state, geometry, and visualization building blocks."""

from .contracts import VofCellType
from .geometry import InterfaceGeometry
from .state import GeometryState, VofGridState
from .visualization import InterfaceVisualData, VofInterfaceVisualizer

__all__ = [
    "GeometryState",
    "InterfaceGeometry",
    "InterfaceVisualData",
    "VofCellType",
    "VofGridState",
    "VofInterfaceVisualizer",
]
