# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Health metrics emitted by the minimal HOME core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HomeCoreDiagnostics:
    invalid_cell_count: int
    min_density: float
    max_density: float
    max_speed: float
    max_nonequilibrium_stress: float
