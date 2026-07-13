# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Performance and validation metrics for LBM benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.pipeline import StepStats

if TYPE_CHECKING:
    from ..state import LbmState


@dataclass
class PerfMetrics:
    """Aggregated performance measurements."""

    ms_per_lbm_step: float = 0.0
    mlups: float = 0.0
    bytes_per_cell: float = 0.0
    num_cells: int = 0
    lattice: str = "D3Q19"
    variant: str = ""
    kernel_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class ValidationMetrics:
    """Physics / stability snapshots for regression comparison."""

    max_velocity: float = 0.0
    max_density: float = 0.0
    min_density: float = 0.0
    water_cell_count: int = 0
    water_mass: float = 0.0
    front_position_x: float = 0.0
    sim_time: float = 0.0
    step_index: int = 0
    finite_fields: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def bytes_per_cell(state: LbmState) -> float:
    """Estimate persistent GPU bytes per grid cell for an :class:`LbmState`."""
    num_cells = int(state.model.nx * state.model.ny * state.model.nz)
    if num_cells <= 0:
        return 0.0

    total_bytes = 0
    for name in (
        "f",
        "density",
        "pressure",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "vel_u",
        "vel_v",
        "vel_w",
        "vel_solid_u",
        "vel_solid_v",
        "vel_solid_w",
        "solid_phi",
        "solid_body_id",
        "force_x",
        "force_y",
        "force_z",
    ):
        arr = getattr(state, name, None)
        if arr is None:
            continue
        dtype = getattr(arr, "dtype", None)
        itemsize = int(np.dtype(dtype).itemsize) if dtype is not None else 8
        total_bytes += int(arr.size) * itemsize

    return float(total_bytes) / float(num_cells)


def perf_metrics_from_step_stats(
    stats: StepStats,
    *,
    variant: str,
    lattice: str,
    num_cells: int,
    bytes_per_cell_value: float,
) -> PerfMetrics:
    """Build :class:`PerfMetrics` from a single-step :class:`StepStats`."""
    stats_with_cells = stats.with_num_cells(num_cells)
    return PerfMetrics(
        ms_per_lbm_step=float(stats_with_cells.ms_total),
        mlups=float(stats_with_cells.mlups),
        bytes_per_cell=float(bytes_per_cell_value),
        num_cells=int(num_cells),
        lattice=str(lattice),
        variant=str(variant),
        kernel_breakdown={
            "ms_moments": float(stats.ms_moments),
            "ms_phase": float(stats.ms_phase),
            "ms_regularization": float(stats.ms_regularization),
            "ms_collision": float(stats.ms_collision),
            "ms_bc": float(stats.ms_bc),
            "ms_body_force": float(stats.ms_body_force),
            "ms_export": float(stats.ms_export),
        },
    )


def collect_validation_metrics(
    state: LbmState,
    *,
    water_density_threshold: float = 0.7,
    sim_time: float = 0.0,
    step_index: int = 0,
) -> ValidationMetrics:
    """Sample macroscopic fields for dam-break style regression checks."""
    rho_np = np.asarray(state.density.numpy(), dtype=np.float64)
    ux_np = np.asarray(state.velocity_x.numpy(), dtype=np.float64)
    uy_np = np.asarray(state.velocity_y.numpy(), dtype=np.float64)
    uz_np = np.asarray(state.velocity_z.numpy(), dtype=np.float64)

    speed = np.sqrt(ux_np * ux_np + uy_np * uy_np + uz_np * uz_np)
    finite = (
        np.all(np.isfinite(rho_np))
        and np.all(np.isfinite(speed))
    )

    water_mask = rho_np > float(water_density_threshold)
    front_x = 0.0
    if water_mask.any():
        front_x = float(np.max(np.argwhere(water_mask)[:, 0]))

    return ValidationMetrics(
        max_velocity=float(np.max(speed)) if speed.size else 0.0,
        max_density=float(np.max(rho_np)) if rho_np.size else 0.0,
        min_density=float(np.min(rho_np)) if rho_np.size else 0.0,
        water_cell_count=int(water_mask.sum()),
        water_mass=float(rho_np[water_mask].sum()) if water_mask.any() else 0.0,
        front_position_x=front_x,
        sim_time=float(sim_time),
        step_index=int(step_index),
        finite_fields=bool(finite),
    )
