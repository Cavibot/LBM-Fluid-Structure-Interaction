# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a multi-step R3.4 fixed-topology pressure-wave audit."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslFixedTopologyStepper,
    FslState,
)


@dataclass(frozen=True)
class FixedTopologyPressureWaveConfig:
    resolution_x: int = 128
    resolution_y: int = 48
    slab_start: int = 32
    slab_end: int = 96
    interface_fill: float = 0.5
    gas_density: float = 1.002
    viscosity: float = 0.02
    sample_steps: tuple[int, ...] = (0, 1, 5, 15, 30, 60, 100)
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.resolution_x < 32 or self.resolution_y < 4:
            raise ValueError("pressure-wave audit requires at least a 32x4 grid")
        if not 2 <= self.slab_start < self.slab_end <= self.resolution_x - 2:
            raise ValueError("slab must leave at least two gas cells on each periodic side")
        if not 0.0 < self.interface_fill < 1.0:
            raise ValueError("interface_fill must be strictly fractional")
        if self.gas_density <= 0.0 or self.viscosity <= 0.0:
            raise ValueError("gas density and viscosity must be positive")
        if not self.sample_steps or self.sample_steps[0] != 0:
            raise ValueError("sample_steps must begin at zero")
        if any(b <= a for a, b in zip(self.sample_steps, self.sample_steps[1:])):
            raise ValueError("sample_steps must be strictly increasing")


def _initial_fill(config: FixedTopologyPressureWaveConfig) -> np.ndarray:
    shape = (config.resolution_x, config.resolution_y, 1)
    fill = np.zeros(shape, dtype=np.float32)
    fill[config.slab_start : config.slab_end, :, :] = 1.0
    fill[config.slab_start - 1, :, :] = config.interface_fill
    fill[config.slab_end, :, :] = config.interface_fill
    return fill


def _download_moments(state: object) -> np.ndarray:
    return state.moments.numpy().reshape(10, -1).T.reshape(state.res + (10,))


def _snapshot(fluid: object, fsl: FslState) -> dict[str, np.ndarray]:
    return {
        "moments": _download_moments(fluid).astype(np.float64),
        "mass": fsl.mass.numpy().astype(np.float64),
        "fill": fsl.fill_level.numpy().astype(np.float64),
        "excess": fsl.excess_mass.numpy().astype(np.float64),
        "flags": fsl.flags.numpy(),
    }


def _render_frame(
    output_path: Path,
    step: int,
    snapshot: dict[str, np.ndarray],
    density_limit: float,
    momentum_limit: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    moments = snapshot["moments"]
    fill = snapshot["fill"]
    flags = snapshot["flags"]
    active = flags != int(FslCellFlag.GAS)
    density = np.where(active, moments[..., 0] - 1.0, np.nan)
    momentum = np.where(active, moments[..., 1], np.nan)
    x = np.arange(fill.shape[0])
    active_profile = active[:, 0, 0]
    density_profile = np.mean(moments[:, :, 0, 0] - 1.0, axis=1)
    momentum_profile = np.mean(moments[:, :, 0, 1], axis=1)
    fill_profile = np.mean(fill[:, :, 0], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=160)
    density_axis, momentum_axis, profile_axis, fill_axis = axes.ravel()
    density_image = density_axis.imshow(
        density[:, :, 0].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-density_limit,
        vmax=density_limit,
        interpolation="nearest",
        aspect="auto",
    )
    density_axis.set_title("Active density perturbation")
    density_axis.set_xlabel("x lattice cell")
    density_axis.set_ylabel("y lattice cell")
    fig.colorbar(density_image, ax=density_axis, label="rho - 1")
    momentum_image = momentum_axis.imshow(
        momentum[:, :, 0].T,
        origin="lower",
        cmap="PuOr_r",
        vmin=-momentum_limit,
        vmax=momentum_limit,
        interpolation="nearest",
        aspect="auto",
    )
    momentum_axis.set_title("Normal momentum")
    momentum_axis.set_xlabel("x lattice cell")
    momentum_axis.set_ylabel("y lattice cell")
    fig.colorbar(momentum_image, ax=momentum_axis, label="j_x")
    profile_axis.plot(
        x[active_profile], density_profile[active_profile],
        color="#00798c", linewidth=1.8, label="rho - 1"
    )
    profile_axis.plot(
        x[active_profile], momentum_profile[active_profile],
        color="#d1495b", linewidth=1.8, label="j_x"
    )
    profile_axis.axhline(0.0, color="#888888", linewidth=0.8)
    profile_axis.set_title("Centerline pressure-wave profiles")
    profile_axis.set_xlabel("x lattice cell")
    profile_axis.grid(alpha=0.22)
    profile_axis.legend(loc="best")
    fill_axis.plot(x, fill_profile, color="#164e63", linewidth=2.0)
    fill_axis.set_ylim(-0.05, 1.05)
    fill_axis.set_title("Committed fixed-topology fill profile")
    fill_axis.set_xlabel("x lattice cell")
    fill_axis.set_ylabel("fill level")
    fill_axis.grid(alpha=0.22)
    fig.suptitle(f"R3.4 fixed-topology pressure wave, step {step}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _render_overview(
    output_path: Path,
    snapshots: dict[int, dict[str, np.ndarray]],
    momentum_limit: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(snapshots)
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.2), dpi=160)
    for axis, step in zip(axes.ravel(), steps):
        snapshot = snapshots[step]
        active = snapshot["flags"] != int(FslCellFlag.GAS)
        momentum = np.where(active, snapshot["moments"][..., 1], np.nan)
        axis.imshow(
            momentum[:, :, 0].T,
            origin="lower",
            cmap="PuOr_r",
            vmin=-momentum_limit,
            vmax=momentum_limit,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(f"step {step}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    for axis in axes.ravel()[len(steps) :]:
        axis.axis("off")
    fig.suptitle("R3.4 inward pressure-wave propagation (j_x)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_pressure_wave(
    config: FixedTopologyPressureWaveConfig, output_dir: Path
) -> dict[str, object]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    shape = (config.resolution_x, config.resolution_y, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=config.device,
        kinematic_viscosity=config.viscosity,
    )
    domain = HomeCoreDomain(model)
    fluid_a = domain.create_state()
    fluid_b = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid_a)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fsl_a.initialize_from_fill_level(fluid_a, _initial_fill(config))
    stepper = FslFixedTopologyStepper(model, gas_density=config.gas_density)
    snapshots = {0: _snapshot(fluid_a, fsl_a)}
    diagnostics_by_step: dict[int, object] = {}
    started = time.perf_counter()
    sample_set = set(config.sample_steps[1:])
    for step in range(1, config.sample_steps[-1] + 1):
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        if step in sample_set:
            snapshots[step] = _snapshot(fluid_a, fsl_a)
            diagnostics_by_step[step] = diagnostics
    elapsed = time.perf_counter() - started
    density_limit = max(
        1.0e-8,
        max(float(np.nanmax(np.abs(s["moments"][..., 0] - 1.0))) for s in snapshots.values()),
    )
    momentum_limit = max(
        1.0e-8,
        max(float(np.nanmax(np.abs(s["moments"][..., 1]))) for s in snapshots.values()),
    )
    frames: list[str] = []
    metrics: dict[str, dict[str, float | int]] = {}
    initial_total = float(
        np.sum(snapshots[0]["mass"], dtype=np.float64)
        + np.sum(snapshots[0]["excess"], dtype=np.float64)
    )
    for step, snapshot in snapshots.items():
        frame = f"frames/step-{step:06d}.png"
        _render_frame(
            output_dir / frame, step, snapshot, density_limit, momentum_limit
        )
        frames.append(frame)
        active = snapshot["flags"] != int(FslCellFlag.GAS)
        rho = snapshot["moments"][..., 0]
        speed = np.linalg.norm(snapshot["moments"][..., 1:4] / rho[..., None], axis=-1)
        total = float(
            np.sum(snapshot["mass"], dtype=np.float64)
            + np.sum(snapshot["excess"], dtype=np.float64)
        )
        interface = snapshot["flags"] == int(FslCellFlag.INTERFACE)
        metrics[str(step)] = {
            "relative_total_mass_drift": (total - initial_total) / initial_total,
            "min_active_density": float(np.min(rho[active])),
            "max_active_density": float(np.max(rho[active])),
            "max_active_speed": float(np.max(speed[active])),
            "min_interface_fill": float(np.min(snapshot["fill"][interface])),
            "max_interface_fill": float(np.max(snapshot["fill"][interface])),
            "total_excess_mass": float(np.sum(snapshot["excess"], dtype=np.float64)),
            "invalid_cells": (
                diagnostics_by_step[step].fluid.invalid_cell_count if step else 0
            ),
            "phase_crossing_cells": (
                diagnostics_by_step[step].phase_crossing_cell_count if step else 0
            ),
        }
    overview = "pressure-wave-overview.png"
    _render_overview(output_dir / overview, snapshots, momentum_limit)
    manifest: dict[str, object] = {
        "scene": "r3.4-fixed-topology-pressure-wave",
        "profile": "r3.4-warp-validated-transaction",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": config.device,
        },
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "frames": frames,
        "overview": overview,
        "limitations": [
            "fixed LIQUID/INTERFACE/GAS topology",
            "no topology conversion or excess-mass redistribution",
            "pressure-wave audit, not a Dam-break scene",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gas-density", type=float, default=1.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FixedTopologyPressureWaveConfig(
        gas_density=args.gas_density, device=args.device
    )
    print(json.dumps(run_pressure_wave(config, args.output), indent=2))


if __name__ == "__main__":
    main()
