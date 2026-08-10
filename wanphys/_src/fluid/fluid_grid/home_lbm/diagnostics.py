# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Runtime health metrics for HOME-LBM states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HomeLbmDiagnostics:
    invalid_cell_count: int
    direct_liquid_gas_link_count: int
    invalid_gas_density_count: int
    missing_cut_link_count: int
    unsupported_interpolation_count: int
    wall_confined_interpolation_count: int
    min_density: float
    max_density: float
    max_speed: float
    max_nonequilibrium_stress: float
    gas_boundary_impulse_lattice: tuple[float, float, float]
    cut_link_frame_correction_lattice: tuple[float, float, float]
