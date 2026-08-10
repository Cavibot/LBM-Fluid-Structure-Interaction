# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the R3.5 dynamic-topology translating-slab audit."""

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
    FslDynamicTopologyStepper,
    FslState,
)


@dataclass(frozen=True)
class DynamicTopologyTranslationConfig:
    resolution_x: int = 128
    resolution_y: int = 48
    slab_start: int = 32
    slab_end: int = 80
    velocity_x: float = 0.05
    viscosity: float = 0.02
    sample_steps: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40)
    device: str = "cuda:0"


def _initial_fill(config: DynamicTopologyTranslationConfig) -> np.ndarray:
    shape = (config.resolution_x, config.resolution_y, 1)
    fill = np.zeros(shape, dtype=np.float32)
    fill[config.slab_start : config.slab_end] = 1.0
    y = np.arange(config.resolution_y, dtype=np.float64)
    modulation = 0.5 + 0.12 * np.sin(2.0 * np.pi * y / config.resolution_y)
    fill[config.slab_start - 1, :, 0] = modulation
    fill[config.slab_end, :, 0] = 1.0 - modulation
    return fill


def _snapshot(fluid: object, fsl: FslState) -> dict[str, np.ndarray]:
    moments = fluid.moments.numpy().reshape(10, -1).T.reshape(fluid.res + (10,))
    return {
        "moments": moments.astype(np.float64),
        "mass": fsl.mass.numpy().astype(np.float64),
        "fill": fsl.fill_level.numpy().astype(np.float64),
        "excess": fsl.excess_mass.numpy().astype(np.float64),
        "flags": fsl.flags.numpy(),
    }


def _render_frame(path: Path, step: int, snapshot: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fill = snapshot["fill"][:, :, 0]
    flags = snapshot["flags"][:, :, 0]
    moments = snapshot["moments"][:, :, 0]
    rho = moments[..., 0]
    velocity_x = moments[..., 1] / rho
    active = flags != int(FslCellFlag.GAS)
    x = np.arange(fill.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), dpi=170)
    fill_axis, flag_axis, profile_axis, velocity_axis = axes.ravel()
    fill_image = fill_axis.imshow(
        fill.T, origin="lower", vmin=0.0, vmax=1.0, cmap="Blues",
        interpolation="nearest", aspect="auto"
    )
    fill_axis.contour(fill.T, levels=[0.001, 0.999], colors=["#d1495b"], linewidths=0.7)
    fill_axis.set_title("Liquid fill fraction")
    fill_axis.set_xlabel("x lattice cell")
    fill_axis.set_ylabel("y lattice cell")
    fig.colorbar(fill_image, ax=fill_axis, label="fill")
    flag_axis.imshow(
        flags.T, origin="lower", vmin=0, vmax=2,
        cmap=ListedColormap(["#f3f4f6", "#f59e0b", "#0369a1"]),
        interpolation="nearest", aspect="auto"
    )
    flag_axis.set_title("Committed topology: gas / interface / liquid")
    flag_axis.set_xlabel("x lattice cell")
    flag_axis.set_ylabel("y lattice cell")
    profile_axis.plot(x, np.mean(fill, axis=1), color="#0369a1", linewidth=2.0)
    profile_axis.set_ylim(-0.05, 1.05)
    profile_axis.set_title("Mean fill profile")
    profile_axis.set_xlabel("x lattice cell")
    profile_axis.set_ylabel("fill")
    profile_axis.grid(alpha=0.22)
    velocity_axis.plot(
        x, np.mean(np.where(active, velocity_x, np.nan), axis=1),
        color="#b91c1c", linewidth=1.8
    )
    velocity_axis.axhline(0.05, color="#666666", linewidth=0.8, linestyle="--")
    velocity_axis.set_title("Active mean x velocity")
    velocity_axis.set_xlabel("x lattice cell")
    velocity_axis.set_ylabel("u_x")
    velocity_axis.grid(alpha=0.22)
    fig.suptitle(f"R3.5 dynamic topology translation, step {step}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_overview(path: Path, snapshots: dict[int, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.0), dpi=170)
    for axis, (step, snapshot) in zip(axes.ravel(), snapshots.items()):
        axis.imshow(
            snapshot["fill"][:, :, 0].T, origin="lower", vmin=0.0, vmax=1.0,
            cmap="Blues", interpolation="nearest", aspect="auto"
        )
        axis.set_title(f"step {step}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    for axis in axes.ravel()[len(snapshots) :]:
        axis.axis("off")
    fig.suptitle("R3.5 translating slab with committed topology changes", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_scene(
    config: DynamicTopologyTranslationConfig, output_dir: Path
) -> dict[str, object]:
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
    domain.solver.initialize_uniform_lattice(
        fluid_a, velocity=(config.velocity_x, 0.0, 0.0)
    )
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fsl_a.initialize_from_fill_level(fluid_a, _initial_fill(config))
    stepper = FslDynamicTopologyStepper(model)
    snapshots = {0: _snapshot(fluid_a, fsl_a)}
    transition_totals = {"interface_to_liquid": 0, "interface_to_gas": 0}
    started = time.perf_counter()
    sample_set = set(config.sample_steps[1:])
    for step in range(1, config.sample_steps[-1] + 1):
        diagnostics = stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
        transition_totals["interface_to_liquid"] += (
            diagnostics.topology.interface_to_liquid_count
        )
        transition_totals["interface_to_gas"] += diagnostics.topology.interface_to_gas_count
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        if step in sample_set:
            snapshots[step] = _snapshot(fluid_a, fsl_a)
    wp.synchronize_device(config.device)
    elapsed = time.perf_counter() - started

    initial_total = float(np.sum(snapshots[0]["mass"]) + np.sum(snapshots[0]["excess"]))
    metrics: dict[str, dict[str, float | int]] = {}
    frames: list[str] = []
    for step, snapshot in snapshots.items():
        frame = f"frames/step-{step:06d}.png"
        _render_frame(output_dir / frame, step, snapshot)
        frames.append(frame)
        total = float(np.sum(snapshot["mass"]) + np.sum(snapshot["excess"]))
        x = np.arange(config.resolution_x, dtype=np.float64)[:, None, None]
        metrics[str(step)] = {
            "relative_total_mass_drift": (total - initial_total) / initial_total,
            "liquid_center_x": float(np.sum(x * snapshot["mass"]) / np.sum(snapshot["mass"])),
            "liquid_cells": int(np.count_nonzero(snapshot["flags"] == int(FslCellFlag.LIQUID))),
            "interface_cells": int(np.count_nonzero(snapshot["flags"] == int(FslCellFlag.INTERFACE))),
            "queued_excess_mass": float(np.sum(snapshot["excess"])),
        }
    overview = "dynamic-topology-overview.png"
    _render_overview(output_dir / overview, snapshots)
    manifest: dict[str, object] = {
        "scene": "r3.5-dynamic-topology-translation",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": config.device,
        },
        "elapsed_seconds": elapsed,
        "transition_totals": transition_totals,
        "metrics": metrics,
        "frames": frames,
        "overview": overview,
        "limitations": [
            "topology translation audit, not a gravity-driven Dam-break",
            "periodic HOME core without wall boundary composition",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = DynamicTopologyTranslationConfig(device=args.device)
    print(json.dumps(run_scene(config, args.output), indent=2))


if __name__ == "__main__":
    main()
