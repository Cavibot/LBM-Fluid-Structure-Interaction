# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Production-throughput benchmark for the HOME-Free rigid FSI path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeDomain,
    HomeFreeLegacyDomain,
    HomeFreeRigidCoupling,
    HomeLbmModel,
    HomeMomentQuantizationAuditor,
)
from wanphys._src.fluid.fluid_grid.home_lbm.constants import MOMENT_NAMES
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


def _represented_mass(fluid: HomeFreeDomain | HomeFreeLegacyDomain) -> float:
    state = fluid.free_surface_state
    recipients = (
        np.zeros(state.res, dtype=np.float64)
        if getattr(fluid, "uses_independent_geometric_mass", False)
        else fluid.topology.active_neighbor_count.numpy()
    )
    return float(
        np.sum(
            state.mass.numpy() + state.excess_mass.numpy() * recipients,
            dtype=np.float64,
        )
    )


def _build_case(
    resolution: tuple[int, int, int],
    radius: float,
    velocity: float,
    strong_iterations: int,
    backend: str,
    project_courant: bool,
    projection_max_iterations: int | None,
    contact_angle_degrees: float,
    device: str,
) -> tuple[HomeFreeRigidCoupling, int]:
    model = HomeLbmModel(
        fluid_grid_res=resolution,
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        reference_density=1.0,
        kinematic_viscosity=0.2,
        max_lattice_speed=0.15,
        periodic=(True, True, False),
        device=device,
    )
    if backend == "legacy":
        if project_courant:
            raise ValueError("Courant projection is only available for geometric HOME-Free")
        fluid = HomeFreeLegacyDomain(model)
    elif backend == "geometric":
        fluid = HomeFreeDomain(
            model,
            project_courant=project_courant,
            projection_max_iterations=projection_max_iterations,
            contact_angle_degrees=contact_angle_degrees,
        )
    else:
        raise ValueError(f"unsupported HOME-Free backend: {backend}")
    interface_index = int(0.55 * resolution[2])
    center = (
        0.5 * resolution[0],
        0.5 * resolution[1],
        interface_index + radius + 0.8,
    )
    builder = RigidModelBuilder(gravity=0.0)
    body = builder.add_body(position=center, label="home_free_benchmark_sphere")
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
    coupling = HomeFreeRigidCoupling(
        fluid,
        rigid,
        rigid_substeps=2,
        strong_coupling_max_iterations=strong_iterations,
        strong_coupling_tolerance=2.0e-6,
        strong_coupling_relaxation=0.8,
    )
    fill = np.zeros(resolution, dtype=np.float32)
    fill[:, :, :interface_index] = 1.0
    fill[:, :, interface_index] = 0.5
    fluid.initialize_uniform_lattice(fill)
    return coupling, body


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    resolution = tuple(int(value) for value in args.resolution.split("x"))
    if len(resolution) != 3 or any(value <= 0 for value in resolution):
        raise ValueError("resolution must be formatted as NXxNYxNZ with positive values")
    coupling, body = _build_case(
        resolution,
        args.radius,
        args.velocity,
        args.strong_iterations,
        args.backend,
        args.project_courant,
        args.projection_max_iterations,
        args.contact_angle,
        args.device,
    )
    initial_mass = _represented_mass(coupling.fluid_domain)
    initial_body_position = np.asarray(
        coupling.rigid_domain.state.get_body_position(body), dtype=np.float64
    )
    quantization_auditor = (
        HomeMomentQuantizationAuditor(coupling.fluid_domain.state.fluid.model)
        if args.audit_quantization
        else None
    )
    maximum_saturation_counts = np.zeros(10, dtype=np.int64)
    for _ in range(args.warmup):
        coupling.step()
    wp.synchronize_device(args.device)

    step_times: list[float] = []
    transition_step_times: list[float] = []
    steady_step_times: list[float] = []
    iteration_counts: list[int] = []
    residuals: list[float] = []
    projection_iterations: list[int] = []
    projected_divergence: list[float] = []
    fresh_cells = 0
    dead_cells = 0
    maximum_wet_links = 0
    maximum_dry_links = 0
    maximum_mass_drift = 0.0
    launch_counts = {"kernel": 0, "graph": 0}
    original_launch = wp.launch
    original_capture_launch = wp.capture_launch
    if args.count_launches:
        def counted_launch(*launch_args: Any, **launch_kwargs: Any) -> Any:
            launch_counts["kernel"] += 1
            return original_launch(*launch_args, **launch_kwargs)

        def counted_capture_launch(*launch_args: Any, **launch_kwargs: Any) -> Any:
            launch_counts["graph"] += 1
            return original_capture_launch(*launch_args, **launch_kwargs)

        wp.launch = counted_launch
        wp.capture_launch = counted_capture_launch
    try:
        for _ in range(args.steps):
            wp.synchronize_device(args.device)
            start = time.perf_counter()
            diagnostics = coupling.step()
            wp.synchronize_device(args.device)
            step_time = (time.perf_counter() - start) * 1000.0
            step_times.append(step_time)
            iteration_counts.append(diagnostics.strong_coupling_iterations)
            residuals.append(diagnostics.strong_coupling_residual)
            geometric = getattr(
                coupling.fluid_domain, "last_geometric_diagnostics", None
            )
            if geometric is not None and geometric.projection is not None:
                projection_iterations.append(
                    geometric.projection.iteration_count
                )
                projected_divergence.append(
                    geometric.projection.projected_max_divergence
                )
            if quantization_auditor is not None:
                state = coupling.fluid_domain.state
                quantization = quantization_auditor.audit(
                    state.fluid.moments,
                    active_flags=state.free_surface.flags,
                    raise_on_saturation=False,
                )
                maximum_saturation_counts = np.maximum(
                    maximum_saturation_counts,
                    np.asarray(quantization.saturation_counts, dtype=np.int64),
                )
            transition = diagnostics.interface.transitions
            load = diagnostics.interface.load
            if transition.fresh_cell_count or transition.dead_cell_count:
                transition_step_times.append(step_time)
            else:
                steady_step_times.append(step_time)
            fresh_cells += transition.fresh_cell_count
            dead_cells += transition.dead_cell_count
            maximum_wet_links = max(maximum_wet_links, load.wet_link_count)
            maximum_dry_links = max(maximum_dry_links, load.dry_link_count)
            mass_drift = abs(
                _represented_mass(coupling.fluid_domain) - initial_mass
            ) / initial_mass
            maximum_mass_drift = max(maximum_mass_drift, mass_drift)
    finally:
        wp.launch = original_launch
        wp.capture_launch = original_capture_launch

    values = np.asarray(step_times, dtype=np.float64)
    mean_step_ms = float(np.mean(values))
    mlups = int(np.prod(resolution)) / (mean_step_ms * 1000.0)
    final_body_position = np.asarray(
        coupling.rigid_domain.state.get_body_position(body), dtype=np.float64
    )
    final_body_velocity = np.asarray(
        coupling.rigid_domain.state.get_body_velocity(body), dtype=np.float64
    )
    return {
        "device": str(wp.get_device(args.device)),
        "backend": args.backend,
        "domain_type": type(coupling.fluid_domain).__name__,
        "project_courant": args.project_courant,
        "projection_max_iterations": args.projection_max_iterations,
        "contact_angle_degrees": args.contact_angle,
        "resolution": resolution,
        "cells": int(np.prod(resolution)),
        "radius_cells": args.radius,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "strong_iteration_limit": args.strong_iterations,
        "mean_strong_iterations": statistics.mean(iteration_counts),
        "maximum_strong_residual": max(residuals),
        "mean_projection_iterations": (
            statistics.mean(projection_iterations) if projection_iterations else 0.0
        ),
        "maximum_projection_iterations": max(projection_iterations, default=0),
        "maximum_projected_divergence": max(projected_divergence, default=0.0),
        "quantization_audited": quantization_auditor is not None,
        "quantization_maximum_saturation_counts": {
            name: int(count)
            for name, count in zip(MOMENT_NAMES, maximum_saturation_counts)
        },
        "quantization_maximum_roundtrip_error": (
            {
                name: error
                for name, error in zip(
                    MOMENT_NAMES,
                    quantization_auditor.spec.maximum_roundtrip_error,
                )
            }
            if quantization_auditor is not None
            else {}
        ),
        "mean_step_ms": mean_step_ms,
        "p50_step_ms": float(np.percentile(values, 50)),
        "p95_step_ms": float(np.percentile(values, 95)),
        "steps_per_second": 1000.0 / mean_step_ms,
        "mlups": mlups,
        "home_moment_minimum_bytes_per_lup": 80,
        "home_moment_minimum_bandwidth_gbps": mlups * 0.08,
        "kernel_launch_counting_enabled": args.count_launches,
        "kernel_launches_per_step": launch_counts["kernel"] / args.steps,
        "cuda_graph_launches_per_step": launch_counts["graph"] / args.steps,
        "transition_steps": len(transition_step_times),
        "mean_transition_step_ms": (
            statistics.mean(transition_step_times) if transition_step_times else 0.0
        ),
        "steady_steps": len(steady_step_times),
        "mean_steady_step_ms": (
            statistics.mean(steady_step_times) if steady_step_times else 0.0
        ),
        "fresh_cells": fresh_cells,
        "dead_cells": dead_cells,
        "maximum_wet_links": maximum_wet_links,
        "maximum_dry_links": maximum_dry_links,
        "maximum_mass_drift": maximum_mass_drift,
        "initial_body_position": initial_body_position.tolist(),
        "final_body_position": final_body_position.tolist(),
        "body_displacement": (final_body_position - initial_body_position).tolist(),
        "final_body_velocity": final_body_velocity.tolist(),
        "warp_mempool_current_mib": int(
            wp.get_mempool_used_mem_current(args.device)
        )
        / 2**20,
        "warp_mempool_peak_mib": int(wp.get_mempool_used_mem_high(args.device))
        / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="54x54x86")
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--velocity", type=float, default=0.04)
    parser.add_argument("--strong-iterations", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("legacy", "geometric"),
        default="geometric",
    )
    parser.add_argument(
        "--project-courant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--projection-max-iterations", type=int)
    parser.add_argument("--contact-angle", type=float, default=90.0)
    parser.add_argument("--audit-quantization", action="store_true")
    parser.add_argument("--count-launches", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
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
        print(f"HOME-Free FSI benchmark failed: {error}", file=sys.stderr)
        raise
