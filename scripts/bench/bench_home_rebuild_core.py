# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the isolated periodic HOME core with the audited general solver."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeLbmDomain, HomeLbmModel
from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel


def initial_moments(resolution: int, amplitude: float) -> np.ndarray:
    coordinates = 2.0 * np.pi * (np.arange(resolution) + 0.5) / resolution
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    velocity = np.zeros((resolution, resolution, 1, 3), dtype=np.float64)
    velocity[:, :, 0, 0] = amplitude * np.sin(x) * np.cos(y)
    velocity[:, :, 0, 1] = -amplitude * np.cos(x) * np.sin(y)
    rho = np.ones((resolution, resolution, 1), dtype=np.float64)
    moments = np.zeros(rho.shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * velocity
    moments[..., 4] = rho * velocity[..., 0] ** 2
    moments[..., 5] = rho * velocity[..., 1] ** 2
    moments[..., 6] = rho * velocity[..., 2] ** 2
    moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    return moments


def upload(state: object, moments: np.ndarray) -> None:
    values = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(values, dtype=float, device=state.device))


def run_domain(domain: object, warmup: int, steps: int, device: str) -> float:
    for _ in range(warmup):
        domain.step(1.0)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(steps):
        domain.step(1.0)
    wp.synchronize_device(device)
    return time.perf_counter() - start


def benchmark(
    *,
    resolution: int,
    viscosity: float,
    amplitude: float,
    warmup: int,
    steps: int,
    repeats: int,
    device: str,
) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    common = dict(
        fluid_grid_res=(resolution, resolution, 1),
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        kinematic_viscosity=viscosity,
        periodic=(True, True, True),
        device=device,
    )
    old_domain = HomeLbmDomain(HomeLbmModel(**common))
    new_domain = HomeCoreDomain(HomeCoreModel(**common))
    initial = initial_moments(resolution, amplitude)
    upload(old_domain.create_state(), initial)
    upload(new_domain.create_state(), initial)

    old_samples: list[float] = []
    new_samples: list[float] = []
    for repeat in range(repeats):
        upload(old_domain.state, initial)
        upload(new_domain.state, initial)
        if repeat % 2 == 0:
            old_samples.append(run_domain(old_domain, warmup, steps, device))
            new_samples.append(run_domain(new_domain, warmup, steps, device))
        else:
            new_samples.append(run_domain(new_domain, warmup, steps, device))
            old_samples.append(run_domain(old_domain, warmup, steps, device))
    old_seconds = float(np.median(old_samples))
    new_seconds = float(np.median(new_samples))
    old_values = old_domain.state.moments.numpy()
    new_values = new_domain.state.moments.numpy()
    difference = new_values.astype(np.float64) - old_values.astype(np.float64)
    max_abs_difference = float(np.max(np.abs(difference)))
    relative_l2_difference = float(
        np.linalg.norm(difference) / np.linalg.norm(old_values.astype(np.float64))
    )
    cell_count = resolution * resolution
    relative_mass_difference = float(
        np.sum(difference[:cell_count], dtype=np.float64)
        / np.sum(old_values[:cell_count], dtype=np.float64)
    )
    cell_updates = resolution * resolution * steps

    # Two state buffers. The old periodic path also allocates solid fields and
    # per-cell placeholders for optional cut-link, free-surface, and force inputs.
    new_bytes_per_cell = 2 * 10 * 4
    old_bytes_per_cell = 2 * (10 * 4 + 4 + 4) + (4 + 4 + 4 + 4 + 3 * 4)
    return {
        "config": {
            "resolution": [resolution, resolution, 1],
            "viscosity": viscosity,
            "amplitude": amplitude,
            "warmup_steps": warmup,
            "timed_steps": steps,
            "repeats": repeats,
            "device": device,
        },
        "old_general_solver": {
            "seconds": old_seconds,
            "mlups": cell_updates / old_seconds / 1.0e6,
            "seconds_samples": old_samples,
            "estimated_bytes_per_cell": old_bytes_per_cell,
        },
        "isolated_periodic_core": {
            "seconds": new_seconds,
            "mlups": cell_updates / new_seconds / 1.0e6,
            "seconds_samples": new_samples,
            "estimated_bytes_per_cell": new_bytes_per_cell,
        },
        "speedup": old_seconds / new_seconds,
        "estimated_per_cell_memory_reduction": 1.0 - new_bytes_per_cell / old_bytes_per_cell,
        "max_abs_moment_difference": max_abs_difference,
        "relative_l2_moment_difference": relative_l2_difference,
        "relative_mass_difference": relative_mass_difference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--viscosity", type=float, default=0.0001)
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(
        resolution=args.resolution,
        viscosity=args.viscosity,
        amplitude=args.amplitude,
        warmup=args.warmup,
        steps=args.steps,
        repeats=args.repeats,
        device=args.device,
    )
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
