# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys core infrastructure for multi-domain physics simulation."""

from wanphys._src.core import (
    CompositeSimulation,
    Domain,
    DomainModel,
    DomainSolver,
    DomainState,
)

__all__ = [
    "Domain",
    "DomainModel",
    "DomainState",
    "DomainSolver",
    "CompositeSimulation",
]
