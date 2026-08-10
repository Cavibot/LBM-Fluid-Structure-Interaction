# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-resolved GPU benchmark for the independent HOME-LBM FSI path.

Example::

    python scripts/bench/bench_home_lbm_fsi.py --steps 100
    python scripts/bench/bench_home_lbm_fsi.py --strong-iterations 12 --steps 50
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidCoupling,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


class StageTimer:
    """Synchronize around selected calls to obtain exclusive GPU stage costs."""

    def __init__(self, device: wp.DeviceLike) -> None:
        self.device = wp.get_device(device)
        self.samples: dict[str, list[float]] = defaultdict(list)

    def wrap(self, owner: object, method_name: str, stage: str) -> None:
        original = getattr(owner, method_name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            wp.synchronize_device(self.device)
            start = time.perf_counter()
            result = original(*args, **kwargs)
            wp.synchronize_device(self.device)
            self.samples[stage].append((time.perf_counter() - start) * 1000.0)
            return result

        setattr(owner, method_name, measured)

    def summary(self, physical_steps: int) -> dict[str, dict[str, float]]:
        report: dict[str, dict[str, float]] = {}
        for stage, samples in sorted(self.samples.items()):
            values = np.asarray(samples, dtype=np.float64)
            report[stage] = {
                "calls": float(len(samples)),
                "calls_per_step": len(samples) / physical_steps,
                "mean_call_ms": float(np.mean(values)),
                "mean_step_ms": float(np.sum(values) / physical_steps),
                "p50_call_ms": float(np.percentile(values, 50)),
                "p95_call_ms": float(np.percentile(values, 95)),
            }
        return report


def _build_case(
    resolution: tuple[int, int, int],
    radius: float,
    velocity: float,
    strong_iterations: int,
    device: str,
) -> HomeLbmRigidCoupling:
    model = HomeLbmModel(
        fluid_grid_res=resolution,
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        reference_density=1.0,
        kinematic_viscosity=0.012,
        periodic=(False, False, False),
        device=device,
    )
    fluid = HomeLbmDomain(model)
    fluid.create_state()
    fluid.solver.initialize_uniform_lattice(fluid.state)

    center = (0.5 * resolution[0], 0.5 * resolution[1], 0.7 * resolution[2])
    builder = RigidModelBuilder(gravity=0.0)
    body = builder.add_body(position=center, label="home_fsi_benchmark_sphere")
    builder.add_shape_sphere(
        body,
        radius=radius,
        cfg=ShapeConfig(density=1.15, has_shape_collision=False),
    )
    rigid_model = builder.finalize(device=device)
    rigid = RigidDomain(
        rigid_model,
        solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
    )
    rigid.create_state()
    rigid.state.body_qd.assign([[0.0, 0.0, -velocity, 0.0, 0.0, 0.0]])

    return HomeLbmRigidCoupling(
        fluid,
        rigid,
        rigid_substeps=2,
        strong_coupling_max_iterations=strong_iterations,
        strong_coupling_tolerance=2.0e-6,
        strong_coupling_relaxation=0.8,
    )


def _instrument(coupling: HomeLbmRigidCoupling) -> StageTimer:
    timer = StageTimer(coupling.fluid_domain.model._device)
    timer.wrap(coupling.predictor, "validate_lattice_displacement", "motion_cfl")
    timer.wrap(coupling.predictor, "sample", "motion_sample")
    timer.wrap(coupling.interface.geometry, "rasterize_incremental", "sdf_update")
    timer.wrap(coupling.interface.remapper, "remap", "transition_remap")
    timer.wrap(coupling.interface, "_build_links", "cut_link_rebuild")
    timer.wrap(coupling.fluid_domain, "step", "fluid_step")
    timer.wrap(coupling.fluid_domain.solver, "validate_state", "fluid_diagnostics")
    timer.wrap(coupling.interface, "collect_wrench", "wrench_reduction")
    timer.wrap(coupling.rigid_domain, "step", "rigid_step")
    return timer


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    resolution = tuple(int(value) for value in args.resolution.split("x"))
    if len(resolution) != 3 or any(value <= 0 for value in resolution):
        raise ValueError("resolution must be formatted as NXxNYxNZ with positive values")
    coupling = _build_case(
        resolution,
        args.radius,
        args.velocity,
        args.strong_iterations,
        args.device,
    )
    for _ in range(args.warmup):
        coupling.step()
    wp.synchronize_device(args.device)

    timer = None if args.total_only else _instrument(coupling)
    step_times: list[float] = []
    iteration_counts: list[int] = []
    for _ in range(args.steps):
        wp.synchronize_device(args.device)
        start = time.perf_counter()
        diagnostics = coupling.step()
        wp.synchronize_device(args.device)
        step_times.append((time.perf_counter() - start) * 1000.0)
        iteration_counts.append(diagnostics.strong_coupling_iterations)

    values = np.asarray(step_times, dtype=np.float64)
    stage_report = {} if timer is None else timer.summary(args.steps)
    measured_stage_ms = sum(stage["mean_step_ms"] for stage in stage_report.values())
    mean_step_ms = float(np.mean(values))
    return {
        "device": str(wp.get_device(args.device)),
        "resolution": resolution,
        "cells": int(np.prod(resolution)),
        "radius_cells": args.radius,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "stage_timing_enabled": timer is not None,
        "strong_iteration_limit": args.strong_iterations,
        "mean_strong_iterations": statistics.mean(iteration_counts),
        "mean_step_ms": mean_step_ms,
        "p50_step_ms": float(np.percentile(values, 50)),
        "p95_step_ms": float(np.percentile(values, 95)),
        "steps_per_second": 1000.0 / mean_step_ms,
        "measured_stage_ms": measured_stage_ms,
        "unattributed_python_and_copy_ms": mean_step_ms - measured_stage_ms,
        "warp_mempool_current_mib": int(wp.get_mempool_used_mem_current(args.device)) / 2**20,
        "warp_mempool_peak_mib": int(wp.get_mempool_used_mem_high(args.device)) / 2**20,
        "stages": stage_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="54x54x86")
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--velocity", type=float, default=0.02)
    parser.add_argument("--strong-iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--total-only",
        action="store_true",
        help="measure production throughput without per-stage synchronization",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = run_benchmark(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"HOME-LBM FSI benchmark failed: {error}", file=sys.stderr)
        raise
