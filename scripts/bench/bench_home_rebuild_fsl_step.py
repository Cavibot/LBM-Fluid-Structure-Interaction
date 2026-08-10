# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the validated R3.4 fixed-topology HOME-Free transaction."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslFixedTopologyStepper,
    FslState,
)


def _fill(shape: tuple[int, int, int]) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    start = shape[0] // 4
    end = 3 * shape[0] // 4
    fill[start:end, :, :] = 1.0
    fill[start - 1, :, :] = 0.5
    fill[end, :, :] = 0.5
    return fill


def _timed_core(
    model: HomeCoreModel, *, warmup: int, steps: int, repeats: int, validate: bool
) -> list[float]:
    domain = HomeCoreDomain(model)
    state_a = domain.create_state()
    state_b = domain.create_state()
    domain.solver.initialize_uniform_lattice(state_a)
    for _ in range(warmup):
        domain.solver.step(state_a, state_b, model.time_step)
        if validate:
            domain.solver.validate_state(state_b)
        state_a, state_b = state_b, state_a
    timings: list[float] = []
    for _ in range(repeats):
        wp.synchronize_device(model._device)
        start = time.perf_counter()
        for _ in range(steps):
            domain.solver.step(state_a, state_b, model.time_step)
            if validate:
                domain.solver.validate_state(state_b)
            state_a, state_b = state_b, state_a
        wp.synchronize_device(model._device)
        timings.append(time.perf_counter() - start)
    return timings


def _timed_transaction(
    model: HomeCoreModel, *, warmup: int, steps: int, repeats: int
) -> tuple[list[float], object]:
    domain = HomeCoreDomain(model)
    fluid_a = domain.create_state()
    fluid_b = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid_a)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fsl_a.initialize_from_fill_level(fluid_a, _fill(fluid_a.res))
    stepper = FslFixedTopologyStepper(model)
    diagnostics = None
    for _ in range(warmup):
        diagnostics = stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    timings: list[float] = []
    for _ in range(repeats):
        wp.synchronize_device(model._device)
        start = time.perf_counter()
        for _ in range(steps):
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
            fluid_a, fluid_b = fluid_b, fluid_a
            fsl_a, fsl_b = fsl_b, fsl_a
        wp.synchronize_device(model._device)
        timings.append(time.perf_counter() - start)
    assert diagnostics is not None
    return timings, diagnostics


def benchmark(
    *, resolution: int, depth: int, warmup: int, steps: int, repeats: int, device: str
) -> dict[str, object]:
    shape = (resolution, resolution, depth)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=0.02,
    )
    core_raw = _timed_core(
        model, warmup=warmup, steps=steps, repeats=repeats, validate=False
    )
    core_validated = _timed_core(
        model, warmup=warmup, steps=steps, repeats=repeats, validate=True
    )
    transaction, diagnostics = _timed_transaction(
        model, warmup=warmup, steps=steps, repeats=repeats
    )
    cells = int(np.prod(shape))

    def summarize(samples: list[float]) -> dict[str, object]:
        median = statistics.median(samples)
        return {
            "samples_seconds": samples,
            "median_seconds": median,
            "steps_per_second": steps / median,
            "million_lattice_updates_per_second": cells * steps / median / 1.0e6,
        }

    raw_summary = summarize(core_raw)
    validated_summary = summarize(core_validated)
    transaction_summary = summarize(transaction)
    return {
        "scene": "r3.4-fixed-topology-transaction-benchmark",
        "config": {
            "resolution": list(shape),
            "warmup_steps": warmup,
            "timed_steps": steps,
            "repeats": repeats,
            "device": device,
        },
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": device,
        },
        "core_raw": raw_summary,
        "core_validated": validated_summary,
        "fixed_topology_transaction": transaction_summary,
        "relative_transaction_time_vs_raw_core": (
            float(transaction_summary["median_seconds"])
            / float(raw_summary["median_seconds"])
        ),
        "relative_transaction_time_vs_validated_core": (
            float(transaction_summary["median_seconds"])
            / float(validated_summary["median_seconds"])
        ),
        "last_diagnostics": {
            "relative_total_mass_drift": diagnostics.relative_total_mass_drift,
            "max_fill_change": diagnostics.max_fill_change,
            "max_speed": diagnostics.fluid.max_speed,
            "max_abs_liquid_mass_normalization": (
                diagnostics.max_abs_liquid_mass_normalization
            ),
            "invalid_cells": diagnostics.fluid.invalid_cell_count,
            "phase_crossing_cells": diagnostics.phase_crossing_cell_count,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(
        resolution=args.resolution,
        depth=args.depth,
        warmup=args.warmup,
        steps=args.steps,
        repeats=args.repeats,
        device=args.device,
    )
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
