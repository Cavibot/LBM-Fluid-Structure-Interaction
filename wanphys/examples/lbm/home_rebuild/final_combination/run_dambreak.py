# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run and archive the final fast/balanced HOME-Free dam-break comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from wanphys.examples.lbm.home_free_offline.capture import atomic_write_json

from .dambreak import (
    CombinationBackend,
    CombinationDambreakMetrics,
    build_combination_dambreak,
    make_combination_config,
)


def _atomic_snapshot(path: Path, scene: Any, metrics: CombinationDambreakMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    state = scene.fluid.free_surface_state
    metadata = {
        "config": scene.config.to_dict(),
        "metrics": metrics.to_dict(),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            fill=state.fill_level.numpy(),
            flags=state.flags.numpy(),
            mass=state.mass.numpy(),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _qualification_issues(
    metrics: CombinationDambreakMetrics,
    *,
    max_lattice_speed: float,
    relative_mass_tolerance: float,
) -> list[str]:
    issues: list[str] = []
    numeric = (
        metrics.represented_mass,
        metrics.relative_mass_error,
        metrics.liquid_volume,
        metrics.maximum_speed,
        metrics.minimum_density,
        metrics.maximum_density,
    )
    if not all(np.isfinite(value) for value in numeric):
        issues.append("non-finite primary metric")
    if metrics.relative_mass_error >= relative_mass_tolerance:
        issues.append(
            "relative represented-mass error reached "
            f"{relative_mass_tolerance:.6g}"
        )
    if metrics.maximum_speed >= max_lattice_speed:
        issues.append("configured lattice-speed limit was reached")
    if metrics.invalid_fluid_cells:
        issues.append("invalid fluid cells were reported")
    if metrics.direct_liquid_gas_links:
        issues.append("direct liquid-gas links were reported")
    if metrics.invalid_surface_tension_cells:
        issues.append("invalid capillary cells were reported")
    if metrics.detached_bulk_columns:
        issues.append("bulk liquid columns became detached from the floor")
    return issues


def run(
    *,
    backend: CombinationBackend | str,
    scale: str,
    device: str,
    output: str | Path,
    steps: int | None = None,
    contact_angle_degrees: float | None = None,
    surface_tension: float | None = None,
    periodic_depth: bool = False,
    advection_axes: tuple[int, ...] | None = None,
    max_lattice_speed: float | None = None,
    continue_on_instability: bool = False,
) -> dict[str, Any]:
    config = make_combination_config(backend, scale=scale, device=device)
    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be positive")
        samples = tuple(step for step in config.sample_steps if step <= steps)
        if steps not in samples:
            samples = (*samples, steps)
        config = replace(config, steps=steps, sample_steps=samples)
    if contact_angle_degrees is not None:
        if not 0.0 < contact_angle_degrees < 180.0:
            raise ValueError("contact angle must lie in (0, 180)")
        config = replace(config, contact_angle_degrees=contact_angle_degrees)
    if surface_tension is not None:
        if surface_tension < 0.0 or not np.isfinite(surface_tension):
            raise ValueError("surface tension must be finite and nonnegative")
        config = replace(config, surface_tension=surface_tension)
    if periodic_depth:
        config = replace(config, periodic=(False, True, False))
    if advection_axes is not None:
        if not advection_axes or any(axis not in (0, 1, 2) for axis in advection_axes):
            raise ValueError("advection axes must be a nonempty subset of 0, 1, 2")
        config = replace(config, advection_axes=advection_axes)
    if max_lattice_speed is not None:
        if not 0.0 < max_lattice_speed < 1.0 / np.sqrt(3.0):
            raise ValueError("maximum lattice speed must lie below lattice sound speed")
        config = replace(config, max_lattice_speed=max_lattice_speed)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene = build_combination_dambreak(config)
    trace_path = output / "metrics.jsonl"
    sample_steps = set(config.sample_steps)
    maximum_step_ms = 0.0
    total_step_ms = 0.0
    failure: dict[str, Any] | None = None

    with trace_path.open("w", encoding="utf-8") as trace:
        for step in range(config.steps + 1):
            metrics = scene.measure()
            trace.write(json.dumps(metrics.to_dict(), sort_keys=True) + "\n")
            trace.flush()
            if step in sample_steps:
                _atomic_snapshot(output / f"step_{step:06d}.npz", scene, metrics)
            if step == config.steps:
                break

            issues = [] if step == 0 else _qualification_issues(
                metrics,
                max_lattice_speed=config.max_lattice_speed,
                relative_mass_tolerance=config.relative_mass_tolerance,
            )
            blocking_issues = issues
            if continue_on_instability:
                blocking_issues = [
                    issue
                    for issue in issues
                    if issue != "bulk liquid columns became detached from the floor"
                ]
            if blocking_issues:
                _atomic_snapshot(output / f"failure_step_{step:06d}.npz", scene, metrics)
                failure = {"step": step, "issues": blocking_issues}
                break
            started = time.perf_counter()
            try:
                scene.step()
                scene.synchronize()
            except Exception as error:
                try:
                    failure_metrics = scene.measure()
                    _atomic_snapshot(
                        output / f"failure_step_{scene.step_index:06d}.npz",
                        scene,
                        failure_metrics,
                    )
                except Exception:
                    pass
                failure = {
                    "step": step,
                    "exception": f"{type(error).__name__}: {error}",
                }
                break
            step_ms = (time.perf_counter() - started) * 1000.0
            total_step_ms += step_ms
            maximum_step_ms = max(maximum_step_ms, step_ms)

    completed_steps = scene.step_index
    summary = {
        "config": config.to_dict(),
        "completed_steps": completed_steps,
        "qualified": failure is None and completed_steps == config.steps,
        "strictly_qualified": (
            failure is None
            and completed_steps == config.steps
            and config.max_lattice_speed <= 0.2
            and not continue_on_instability
        ),
        "failure": failure,
        "average_step_ms": (
            0.0 if completed_steps == 0 else total_step_ms / completed_steps
        ),
        "maximum_step_ms": maximum_step_ms,
        "trace": trace_path.name,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=[item.value for item in CombinationBackend], default="balanced"
    )
    parser.add_argument(
        "--scale", choices=("gate", "preview", "viewer-default"), default="gate"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--contact-angle", type=float)
    parser.add_argument("--surface-tension", type=float)
    parser.add_argument("--periodic-depth", action="store_true")
    parser.add_argument("--advection-axes", choices=("xyz", "xz"), default="xyz")
    parser.add_argument("--max-lattice-speed", type=float)
    parser.add_argument(
        "--device", default="cuda:0" if wp.is_cuda_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-on-instability", action="store_true")
    args = parser.parse_args()
    summary = run(
        backend=args.backend,
        scale=args.scale,
        steps=args.steps,
        contact_angle_degrees=args.contact_angle,
        surface_tension=args.surface_tension,
        periodic_depth=args.periodic_depth,
        advection_axes=(0, 1, 2) if args.advection_axes == "xyz" else (0, 2),
        max_lattice_speed=args.max_lattice_speed,
        device=args.device,
        output=args.output,
        continue_on_instability=args.continue_on_instability,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
