# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless Taylor-Green audit with metrics and fixed-scale vorticity frames."""

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


@dataclass(frozen=True)
class TaylorGreenConfig:
    resolution: int = 96
    viscosity: float = 0.04
    initial_amplitude: float = 0.08
    sample_steps: tuple[int, ...] = (0, 100, 250, 500, 1000, 1500)
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.resolution < 8:
            raise ValueError("resolution must be at least 8")
        if not math.isfinite(self.viscosity) or self.viscosity <= 0.0:
            raise ValueError("viscosity must be finite and positive")
        if not math.isfinite(self.initial_amplitude) or not 0.0 < self.initial_amplitude < 0.2:
            raise ValueError("initial_amplitude must be in (0, 0.2)")
        if not self.sample_steps or self.sample_steps[0] != 0:
            raise ValueError("sample_steps must start at zero")
        if any(step < 0 for step in self.sample_steps):
            raise ValueError("sample_steps must be nonnegative")
        if tuple(sorted(set(self.sample_steps))) != self.sample_steps:
            raise ValueError("sample_steps must be strictly increasing")


def equilibrium_moments(rho: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    moments = np.zeros(rho.shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * velocity
    moments[..., 4] = rho * velocity[..., 0] ** 2
    moments[..., 5] = rho * velocity[..., 1] ** 2
    moments[..., 6] = rho * velocity[..., 2] ** 2
    moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
    moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
    return moments


def taylor_green_initial_state(config: TaylorGreenConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config.validate()
    coordinates = 2.0 * np.pi * (np.arange(config.resolution) + 0.5) / config.resolution
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    basis_x = np.sin(x) * np.cos(y)
    basis_y = -np.cos(x) * np.sin(y)
    velocity = np.zeros((config.resolution, config.resolution, 1, 3), dtype=np.float64)
    velocity[:, :, 0, 0] = config.initial_amplitude * basis_x
    velocity[:, :, 0, 1] = config.initial_amplitude * basis_y
    rho = np.ones((config.resolution, config.resolution, 1), dtype=np.float64)
    return equilibrium_moments(rho, velocity), basis_x, basis_y


def upload_moments(state: object, moments: np.ndarray) -> None:
    soa = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(soa, dtype=float, device=state.device))


def download_moments(state: object) -> np.ndarray:
    nx, ny, nz = state.res
    return state.moments.numpy().reshape(10, -1).T.reshape(nx, ny, nz, 10).astype(np.float64)


def projected_amplitude(
    moments: np.ndarray, basis_x: np.ndarray, basis_y: np.ndarray
) -> float:
    rho = moments[:, :, 0, 0]
    ux = moments[:, :, 0, 1] / rho
    uy = moments[:, :, 0, 2] / rho
    numerator = np.sum(ux * basis_x) + np.sum(uy * basis_y)
    denominator = np.sum(basis_x * basis_x) + np.sum(basis_y * basis_y)
    return float(numerator / denominator)


def vorticity_field(moments: np.ndarray) -> np.ndarray:
    rho = moments[:, :, 0, 0]
    ux = moments[:, :, 0, 1] / rho
    uy = moments[:, :, 0, 2] / rho
    duy_dx = 0.5 * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0))
    dux_dy = 0.5 * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1))
    return duy_dx - dux_dy


def frame_metrics(
    *,
    step: int,
    moments: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    config: TaylorGreenConfig,
    initial_mass: float,
) -> dict[str, float | int]:
    rho = moments[:, :, 0, 0]
    velocity = moments[:, :, 0, 1:3] / rho[..., None]
    amplitude = projected_amplitude(moments, basis_x, basis_y)
    wave_number = 2.0 * np.pi / config.resolution
    expected = config.initial_amplitude * math.exp(
        -2.0 * config.viscosity * wave_number * wave_number * step
    )
    mass = float(np.sum(rho, dtype=np.float64))
    return {
        "step": step,
        "mass": mass,
        "relative_mass_drift": (mass - initial_mass) / initial_mass,
        "projected_amplitude": amplitude,
        "expected_amplitude": expected,
        "relative_amplitude_error": (amplitude - expected) / expected,
        "max_speed": float(np.max(np.linalg.norm(velocity, axis=-1))),
        "min_density": float(np.min(rho)),
        "max_density": float(np.max(rho)),
        "max_abs_vorticity": float(np.max(np.abs(vorticity_field(moments)))),
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
    spacing = max(1, resolution // 16)
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
        alpha=0.55,
        scale=1.6,
        width=0.002,
    )
    axis.set_title(
        f"HOME Taylor-Green | step {int(metrics['step'])}\n"
        f"A={float(metrics['projected_amplitude']):.5f}, "
        f"mass drift={float(metrics['relative_mass_drift']):+.2e}"
    )
    axis.set_xlabel("x lattice cell")
    axis.set_ylabel("y lattice cell")
    fig.colorbar(image, ax=axis, label="lattice vorticity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    from PIL import Image, ImageOps

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    try:
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        columns = 3
        rows = math.ceil(len(images) / columns)
        sheet = Image.new("RGB", (columns * width, rows * height), "white")
        for index, image in enumerate(images):
            fitted = ImageOps.contain(image, (width, height))
            x = (index % columns) * width + (width - fitted.width) // 2
            y = (index // columns) * height + (height - fitted.height) // 2
            sheet.paste(fitted, (x, y))
        sheet.save(output_path)
    finally:
        for image in images:
            image.close()


def run_taylor_green(config: TaylorGreenConfig, output_dir: Path) -> list[dict[str, float | int]]:
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
    initial, basis_x, basis_y = taylor_green_initial_state(config)
    upload_moments(state, initial)
    initial_mass = float(np.sum(initial[..., 0], dtype=np.float64))
    initial_vorticity_limit = float(np.max(np.abs(vorticity_field(initial))))

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
            basis_x=basis_x,
            basis_y=basis_y,
            config=config,
            initial_mass=initial_mass,
        )
        record["invalid_cell_count"] = diagnostics.invalid_cell_count
        records.append(record)
        frame_path = frames_dir / f"step-{target_step:06d}.png"
        render_vorticity_frame(
            output_path=frame_path,
            moments=moments,
            metrics=record,
            color_limit=initial_vorticity_limit,
        )
        frame_paths.append(frame_path)

    elapsed = time.perf_counter() - start
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "scene": "home-core-taylor-green",
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
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--viscosity", type=float, default=0.04)
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 100, 250, 500, 1000, 1500])
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TaylorGreenConfig(
        resolution=args.resolution,
        viscosity=args.viscosity,
        initial_amplitude=args.amplitude,
        sample_steps=tuple(args.steps),
        device=args.device,
    )
    records = run_taylor_green(config, args.output)
    final = records[-1]
    print(
        f"Taylor-Green completed: step={final['step']} "
        f"mass_drift={final['relative_mass_drift']:+.3e} "
        f"amplitude_error={final['relative_amplitude_error']:+.3e} "
        f"max_speed={final['max_speed']:.6f}"
    )


if __name__ == "__main__":
    main()
