# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless nonlinear double-shear-layer audit for the pure HOME core."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys.examples.lbm.home_rebuild.home_core.taylor_green import (
    download_moments,
    equilibrium_moments,
    make_contact_sheet,
    upload_moments,
    vorticity_field,
)


@dataclass(frozen=True)
class ShearLayerConfig:
    resolution: int = 128
    viscosity: float = 0.001
    stream_speed: float = 0.08
    perturbation_ratio: float = 0.05
    layer_thickness: float = 0.04
    sample_steps: tuple[int, ...] = (0, 500, 1200, 2200, 3600, 5000)
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.resolution < 32:
            raise ValueError("resolution must be at least 32")
        if not math.isfinite(self.viscosity) or self.viscosity <= 0.0:
            raise ValueError("viscosity must be finite and positive")
        if not math.isfinite(self.stream_speed) or not 0.0 < self.stream_speed < 0.2:
            raise ValueError("stream_speed must be in (0, 0.2)")
        if not math.isfinite(self.perturbation_ratio) or not 0.0 < self.perturbation_ratio < 0.5:
            raise ValueError("perturbation_ratio must be in (0, 0.5)")
        minimum_thickness = 2.0 / self.resolution
        if not math.isfinite(self.layer_thickness) or not minimum_thickness <= self.layer_thickness < 0.2:
            raise ValueError(
                f"layer_thickness must be in [{minimum_thickness}, 0.2) so the interface is resolved"
            )
        if not self.sample_steps or self.sample_steps[0] != 0:
            raise ValueError("sample_steps must start at zero")
        if tuple(sorted(set(self.sample_steps))) != self.sample_steps:
            raise ValueError("sample_steps must be strictly increasing")


def shear_layer_initial_state(config: ShearLayerConfig) -> np.ndarray:
    """Create two periodic counter-flowing streams with a transverse seed."""

    config.validate()
    coordinates = (np.arange(config.resolution) + 0.5) / config.resolution
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    lower = np.tanh((y - 0.25) / config.layer_thickness)
    upper = np.tanh((0.75 - y) / config.layer_thickness)

    velocity = np.zeros((config.resolution, config.resolution, 1, 3), dtype=np.float64)
    velocity[:, :, 0, 0] = config.stream_speed * np.where(y <= 0.5, lower, upper)
    velocity[:, :, 0, 1] = (
        config.stream_speed
        * config.perturbation_ratio
        * np.sin(2.0 * np.pi * (x + 0.25))
    )
    rho = np.ones((config.resolution, config.resolution, 1), dtype=np.float64)
    return equilibrium_moments(rho, velocity)


def frame_metrics(
    *, step: int, moments: np.ndarray, initial_mass: float, initial_energy: float
) -> dict[str, float | int]:
    rho = moments[:, :, 0, 0]
    velocity = moments[:, :, 0, 1:3] / rho[..., None]
    speed_squared = np.sum(velocity * velocity, axis=-1)
    vorticity = vorticity_field(moments)
    mass = float(np.sum(rho, dtype=np.float64))
    kinetic_energy = float(0.5 * np.sum(rho * speed_squared, dtype=np.float64))
    return {
        "step": step,
        "mass": mass,
        "relative_mass_drift": (mass - initial_mass) / initial_mass,
        "kinetic_energy": kinetic_energy,
        "relative_energy_change": (kinetic_energy - initial_energy) / initial_energy,
        "enstrophy": float(0.5 * np.mean(vorticity * vorticity, dtype=np.float64)),
        "max_speed": float(np.max(np.sqrt(speed_squared))),
        "min_density": float(np.min(rho)),
        "max_density": float(np.max(rho)),
        "max_abs_vorticity": float(np.max(np.abs(vorticity))),
    }


def render_vorticity_frame(
    *,
    output_path: Path,
    moments: np.ndarray,
    metrics: dict[str, float | int],
    color_limit: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = moments[:, :, 0, 0]
    ux = moments[:, :, 0, 1] / rho
    uy = moments[:, :, 0, 2] / rho
    vorticity = vorticity_field(moments)
    resolution = rho.shape[0]
    spacing = max(1, resolution // 20)
    coordinates = np.arange(resolution)

    fig, axis = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    image = axis.imshow(
        vorticity.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="bilinear",
    )
    axis.quiver(
        coordinates[::spacing],
        coordinates[::spacing],
        ux[::spacing, ::spacing].T,
        uy[::spacing, ::spacing].T,
        color="black",
        alpha=0.48,
        scale=2.0,
        width=0.0018,
    )
    axis.set_title(
        f"HOME double shear layer | step {int(metrics['step'])}\n"
        f"energy change={float(metrics['relative_energy_change']):+.2%}, "
        f"mass drift={float(metrics['relative_mass_drift']):+.2e}"
    )
    axis.set_xlabel("x lattice cell")
    axis.set_ylabel("y lattice cell")
    fig.colorbar(image, ax=axis, label="lattice vorticity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_shear_layer(config: ShearLayerConfig, output_dir: Path) -> list[dict[str, float | int]]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    model = HomeCoreModel(
        fluid_grid_res=(config.resolution, config.resolution, 1),
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        kinematic_viscosity=config.viscosity,
        periodic=(True, True, True),
        device=config.device,
    )
    domain = HomeCoreDomain(model)
    state = domain.create_state()
    initial = shear_layer_initial_state(config)
    upload_moments(state, initial)
    initial_mass = float(np.sum(initial[..., 0], dtype=np.float64))
    initial_velocity = initial[..., 1:3] / initial[..., 0, None]
    initial_energy = float(
        0.5 * np.sum(initial[..., 0] * np.sum(initial_velocity * initial_velocity, axis=-1))
    )
    color_limit = float(np.max(np.abs(vorticity_field(initial))))

    records: list[dict[str, float | int]] = []
    frame_paths: list[Path] = []
    current_step = 0
    start = time.perf_counter()
    for target_step in config.sample_steps:
        for _ in range(target_step - current_step):
            domain.step(1.0)
        current_step = target_step
        wp.synchronize_device(config.device)
        moments = download_moments(domain.state)
        diagnostics = domain.solver.validate_state(domain.state)
        record = frame_metrics(
            step=target_step,
            moments=moments,
            initial_mass=initial_mass,
            initial_energy=initial_energy,
        )
        record["invalid_cell_count"] = diagnostics.invalid_cell_count
        records.append(record)
        frame_path = frames_dir / f"step-{target_step:06d}.png"
        render_vorticity_frame(
            output_path=frame_path,
            moments=moments,
            metrics=record,
            color_limit=color_limit,
        )
        frame_paths.append(frame_path)

    elapsed = time.perf_counter() - start
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "scene": "home-core-double-shear-layer",
        "profile": "audit-candidate",
        "backend": "home_rebuild.core",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": str(wp.get_device(config.device)),
        },
        "elapsed_seconds": elapsed,
        "simulated_steps": config.sample_steps[-1],
        "steps_per_second_including_output": (
            config.sample_steps[-1] / elapsed if elapsed > 0.0 else None
        ),
        "million_lattice_updates_per_second_including_output": (
            config.sample_steps[-1] * config.resolution * config.resolution / elapsed / 1.0e6
            if elapsed > 0.0
            else None
        ),
        "frames": [str(path.relative_to(output_dir)) for path in frame_paths],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    make_contact_sheet(frame_paths, output_dir / "contact_sheet.png")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--stream-speed", type=float, default=0.08)
    parser.add_argument("--perturbation-ratio", type=float, default=0.05)
    parser.add_argument("--layer-thickness", type=float, default=0.04)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 500, 1200, 2200, 3600, 5000])
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ShearLayerConfig(
        resolution=args.resolution,
        viscosity=args.viscosity,
        stream_speed=args.stream_speed,
        perturbation_ratio=args.perturbation_ratio,
        layer_thickness=args.layer_thickness,
        sample_steps=tuple(args.steps),
        device=args.device,
    )
    records = run_shear_layer(config, args.output)
    final = records[-1]
    print(
        f"Double shear layer completed: step={final['step']} "
        f"mass_drift={final['relative_mass_drift']:+.3e} "
        f"energy_change={final['relative_energy_change']:+.3e} "
        f"max_speed={final['max_speed']:.6f}"
    )


if __name__ == "__main__":
    main()
