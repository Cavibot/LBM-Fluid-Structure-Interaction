# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the R3.3 only-missing gas-pressure boundary response."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslOnlyMissingStreamer,
    FslState,
)


@dataclass(frozen=True)
class PressureBoundaryConfig:
    resolution_x: int = 96
    resolution_y: int = 32
    slab_start: int = 24
    slab_end: int = 72
    interface_fill: float = 0.5
    gas_densities: tuple[float, ...] = (0.98, 1.0, 1.02)
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.resolution_x < 16 or self.resolution_y < 4:
            raise ValueError("pressure audit requires at least a 16x4 grid")
        if not 2 <= self.slab_start < self.slab_end <= self.resolution_x - 2:
            raise ValueError("slab must leave at least two gas cells on each periodic side")
        if not 0.0 < self.interface_fill < 1.0:
            raise ValueError("interface_fill must be strictly fractional")
        if len(self.gas_densities) < 2 or any(rho <= 0.0 for rho in self.gas_densities):
            raise ValueError("at least two positive gas densities are required")


def _initial_fill(config: PressureBoundaryConfig) -> np.ndarray:
    shape = (config.resolution_x, config.resolution_y, 1)
    fill = np.zeros(shape, dtype=np.float32)
    fill[config.slab_start : config.slab_end, :, :] = 1.0
    fill[config.slab_start - 1, :, :] = config.interface_fill
    fill[config.slab_end, :, :] = config.interface_fill
    return fill


def _download_moments(state: object) -> np.ndarray:
    return state.moments.numpy().reshape(10, -1).T.reshape(state.res + (10,))


def _render(
    output_path: Path,
    fill: np.ndarray,
    flags: np.ndarray,
    responses: dict[float, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), dpi=170)
    topology_axis, density_axis, momentum_axis, field_axis = axes.ravel()
    topology = flags[:, :, 0].T
    topology_image = topology_axis.imshow(
        topology,
        origin="lower",
        cmap=ListedColormap(("#f4f4f4", "#37a6b8", "#164e63")),
        vmin=0,
        vmax=2,
        interpolation="nearest",
        aspect="auto",
    )
    topology_axis.set_title("Fixed FSL topology: gas / interface / liquid")
    topology_axis.set_xlabel("x lattice cell")
    topology_axis.set_ylabel("y lattice cell")
    colorbar = fig.colorbar(topology_image, ax=topology_axis, ticks=(0, 1, 2))
    colorbar.ax.set_yticklabels(("gas", "interface", "liquid"))

    colors = ("#d1495b", "#4d4d4d", "#00798c", "#edae49")
    x = np.arange(fill.shape[0])
    active = flags[:, 0, 0] != int(FslCellFlag.GAS)
    for color, (gas_density, moments) in zip(colors, responses.items()):
        density = np.mean(moments[:, :, 0, 0], axis=1)
        momentum = np.mean(moments[:, :, 0, 1], axis=1)
        density_axis.plot(
            x[active], density[active] - 1.0, color=color, linewidth=1.8,
            marker="o", markersize=2.5, label=f"rho_g={gas_density:.2f}"
        )
        momentum_axis.plot(
            x[active], momentum[active], color=color, linewidth=1.8,
            marker="o", markersize=2.5, label=f"rho_g={gas_density:.2f}"
        )
    density_axis.axhline(0.0, color="#999999", linewidth=0.8)
    density_axis.set_title("One-step active density response")
    density_axis.set_ylabel("rho - 1")
    density_axis.set_xlabel("x lattice cell")
    density_axis.grid(alpha=0.22)
    density_axis.legend(loc="best")
    momentum_axis.axhline(0.0, color="#999999", linewidth=0.8)
    momentum_axis.set_title("One-step normal momentum response")
    momentum_axis.set_ylabel("j_x")
    momentum_axis.set_xlabel("x lattice cell")
    momentum_axis.grid(alpha=0.22)
    momentum_axis.legend(loc="best")

    highest_density = max(responses)
    field = responses[highest_density][..., 1].squeeze(axis=2).T.copy()
    field[:, flags[:, 0, 0] == int(FslCellFlag.GAS)] = np.nan
    limit = float(np.nanmax(np.abs(field)))
    image = field_axis.imshow(
        field,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        aspect="auto",
    )
    field_axis.set_title(f"High-pressure inward impulse: rho_g={highest_density:.2f}")
    field_axis.set_xlabel("x lattice cell")
    field_axis.set_ylabel("y lattice cell")
    fig.colorbar(image, ax=field_axis, label="j_x")
    fig.suptitle("R3.3 HOME-Free only-missing pressure boundary", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_pressure_boundary(config: PressureBoundaryConfig, output_dir: Path) -> dict[str, object]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (config.resolution_x, config.resolution_y, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=config.device)
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid)
    fsl = FslState(model)
    fill = _initial_fill(config)
    fsl.initialize_from_fill_level(fluid, fill)
    flags = fsl.flags.numpy()
    left = config.slab_start - 1
    right = config.slab_end
    responses: dict[float, np.ndarray] = {}
    cases: dict[str, dict[str, float | int]] = {}
    for gas_density in config.gas_densities:
        streamer = FslOnlyMissingStreamer(model, gas_density=gas_density)
        output = streamer.stream(fluid, fsl)
        diagnostics = streamer.validate_result()
        moments = _download_moments(output).astype(np.float64)
        responses[gas_density] = moments
        cases[f"{gas_density:.6g}"] = {
            "gas_link_count": diagnostics.gas_link_count,
            "invalid_cell_count": diagnostics.invalid_cell_count,
            "direct_liquid_gas_link_count": diagnostics.direct_liquid_gas_link_count,
            "min_active_density": diagnostics.min_active_density,
            "max_active_density": diagnostics.max_active_density,
            "max_active_speed": diagnostics.max_active_speed,
            "left_interface_mean_density": float(np.mean(moments[left, :, 0, 0])),
            "right_interface_mean_density": float(np.mean(moments[right, :, 0, 0])),
            "left_interface_mean_momentum_x": float(np.mean(moments[left, :, 0, 1])),
            "right_interface_mean_momentum_x": float(np.mean(moments[right, :, 0, 1])),
        }
    _render(output_dir / "pressure-response.png", fill, flags, responses)
    manifest: dict[str, object] = {
        "scene": "r3.3-planar-interface-only-missing-pressure",
        "profile": "r3.3-warp-only-missing",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": config.device,
        },
        "cases": cases,
        "frames": ["pressure-response.png"],
        "limitations": [
            "single only-missing pull-stream step",
            "fixed planar topology",
            "no collision, topology transition, or mass-commit transaction",
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
    parser.add_argument("--resolution-x", type=int, default=96)
    parser.add_argument("--resolution-y", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PressureBoundaryConfig(
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        device=args.device,
    )
    print(json.dumps(run_pressure_boundary(config, args.output), indent=2))


if __name__ == "__main__":
    main()
