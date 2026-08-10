# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the isolated Warp FSL link-wise mass exchange kernel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import FslMassAdvector, FslState


def benchmark(
    *, resolution: int, warmup: int, steps: int, repeats: int, device: str
) -> dict[str, object]:
    if min(resolution, steps, repeats) < 1 or warmup < 0:
        raise ValueError("resolution, steps, and repeats must be positive; warmup must be nonnegative")
    shape = (resolution, resolution, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    pipeline_domain = HomeCoreDomain(model)
    fluid = pipeline_domain.create_state()
    pipeline_domain.solver.initialize_uniform_lattice(fluid, velocity=(0.05, 0.0, 0.0))
    core_model = HomeCoreModel(fluid_grid_res=shape, device=device)
    core_domain = HomeCoreDomain(core_model)
    core_domain.solver.initialize_uniform_lattice(
        core_domain.create_state(), velocity=(0.05, 0.0, 0.0)
    )
    fill = np.zeros(shape, dtype=np.float32)
    start = resolution // 4
    end = 3 * resolution // 4
    fill[start:end, :, :] = 1.0
    fill[start - 1, :, :] = 0.5
    fill[end, :, :] = 0.5
    fsl = FslState(model)
    fsl.initialize_from_fill_level(fluid, fill)
    advector = FslMassAdvector(model)

    mass_samples: list[float] = []
    core_samples: list[float] = []
    combined_samples: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            advector.advect(fluid, fsl)
        wp.synchronize_device(device)
        begin = time.perf_counter()
        for _ in range(steps):
            advector.advect(fluid, fsl)
        wp.synchronize_device(device)
        mass_samples.append(time.perf_counter() - begin)

        for _ in range(warmup):
            core_domain.step(1.0)
        wp.synchronize_device(device)
        begin = time.perf_counter()
        for _ in range(steps):
            core_domain.step(1.0)
        wp.synchronize_device(device)
        core_samples.append(time.perf_counter() - begin)

        for _ in range(warmup):
            pipeline_domain.step(1.0)
            advector.advect(pipeline_domain.state, fsl)
        wp.synchronize_device(device)
        begin = time.perf_counter()
        for _ in range(steps):
            pipeline_domain.step(1.0)
            advector.advect(pipeline_domain.state, fsl)
        wp.synchronize_device(device)
        combined_samples.append(time.perf_counter() - begin)
    mass_seconds = float(np.median(mass_samples))
    core_seconds = float(np.median(core_samples))
    combined_seconds = float(np.median(combined_samples))
    diagnostics = advector.validate_result()
    cell_updates = resolution * resolution * steps
    return {
        "config": {
            "resolution": list(shape),
            "warmup_steps": warmup,
            "timed_steps": steps,
            "repeats": repeats,
            "device": device,
            "liquid_fraction": float(np.mean(fill)),
        },
        "mass_exchange_only": {
            "median_seconds": mass_seconds,
            "seconds_samples": mass_samples,
            "mlups": cell_updates / mass_seconds / 1.0e6,
        },
        "home_core_only": {
            "median_seconds": core_seconds,
            "seconds_samples": core_samples,
            "mlups": cell_updates / core_seconds / 1.0e6,
        },
        "home_plus_mass_exchange": {
            "median_seconds": combined_seconds,
            "seconds_samples": combined_samples,
            "mlups": cell_updates / combined_seconds / 1.0e6,
            "relative_time_overhead_vs_home": combined_seconds / core_seconds - 1.0,
        },
        "diagnostics": {
            "relative_mass_drift": diagnostics.relative_mass_drift,
            "max_abs_cell_delta": diagnostics.max_abs_cell_delta,
            "invalid_cell_count": diagnostics.invalid_cell_count,
            "direct_liquid_gas_link_count": diagnostics.direct_liquid_gas_link_count,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(
        resolution=args.resolution,
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
