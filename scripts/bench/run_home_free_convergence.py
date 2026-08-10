#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one traceable HOME-Free long-time convergence case."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import warp as wp

from wanphys.examples.lbm.home_free_offline.acceptance import evaluate_scene_trace
from wanphys.examples.lbm.home_free_offline.capture import atomic_write_json
from wanphys.examples.lbm.home_free_offline.config import (
    OfflineSceneLevel,
    OfflineSceneName,
)
from wanphys.examples.lbm.home_free_offline.convergence import (
    make_convergence_config,
)
from wanphys.examples.lbm.home_free_offline.factory import build_offline_scene


def _spanwise_fill_range(scene) -> float:
    fill = scene.fluid.free_surface_state.fill_level.numpy()
    return float(np.max(np.ptp(fill, axis=1)))


def _capture_failure_state(scene, output: Path) -> dict[str, object]:
    stepper_candidate = getattr(
        getattr(scene.fluid, "geometric_stepper", None),
        "candidate_fluid",
        None,
    )
    candidate = getattr(scene.fluid, "_state_out", None)
    if stepper_candidate is not None:
        fluid = stepper_candidate
        free = scene.fluid.free_surface_state
        buffer_name = "om_candidate_fluid_with_committed_free_surface"
    elif candidate is None:
        fluid = scene.fluid.fluid_state
        free = scene.fluid.free_surface_state
        buffer_name = "committed"
    else:
        fluid = candidate.fluid
        free = candidate.free_surface
        buffer_name = "candidate_output"
    moments = fluid.moments.numpy().copy()
    fill = free.fill_level.numpy().copy()
    flags = free.flags.numpy().copy()
    solid_phi = fluid.solid_phi.numpy().copy()
    shape = fill.shape
    stride = int(np.prod(shape))
    moment_fields = moments.reshape(10, stride)
    density = moment_fields[0]
    momentum = moment_fields[1:4]
    speed_squared = np.sum(
        np.square(momentum / density[None, :]), axis=0, dtype=np.float64
    )
    active = (flags.reshape(-1) != 0) & (solid_phi.reshape(-1) >= 0.0)
    speed_squared[~active] = -np.inf
    maximum_cell = int(np.argmax(speed_squared))
    maximum_index = tuple(
        int(value) for value in np.unravel_index(maximum_cell, shape)
    )
    i, j, k = maximum_index
    velocity = momentum[:, maximum_cell] / density[maximum_cell]
    np.savez_compressed(
        output / "failure_state.npz",
        moments=moments,
        fill_level=fill,
        flags=flags,
        solid_phi=solid_phi,
    )
    return {
        "state_path": "failure_state.npz",
        "buffer": buffer_name,
        "maximum_speed_cell": maximum_index,
        "maximum_speed": float(np.sqrt(speed_squared[maximum_cell])),
        "density": float(density[maximum_cell]),
        "momentum": tuple(float(value) for value in momentum[:, maximum_cell]),
        "velocity": tuple(float(value) for value in velocity),
        "fill_level": float(fill[i, j, k]),
        "flag": int(flags[i, j, k]),
        "solid_phi": float(solid_phi[i, j, k]),
        "moments": tuple(
            float(value) for value in moment_fields[:, maximum_cell]
        ),
        "fill_neighborhood": fill[
            max(0, i - 1) : min(shape[0], i + 2),
            max(0, j - 1) : min(shape[1], j + 2),
            max(0, k - 1) : min(shape[2], k + 2),
        ].tolist(),
        "flag_neighborhood": flags[
            max(0, i - 1) : min(shape[0], i + 2),
            max(0, j - 1) : min(shape[1], j + 2),
            max(0, k - 1) : min(shape[2], k + 2),
        ].tolist(),
    }


def run_case(
    scene_name: OfflineSceneName,
    level: OfflineSceneLevel,
    *,
    device: str,
    output: Path,
) -> dict[str, object]:
    config = make_convergence_config(scene_name, level, device=device)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.jsonl"
    trace_temporary = trace_path.with_name(trace_path.name + ".tmp")
    manifest_path = output / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "long-time-convergence",
        "scene": scene_name.value,
        "level": level.value,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "derived_parameters": config.derived_parameters,
        "complete": False,
        "numerically_qualified": False,
    }
    atomic_write_json(manifest_path, manifest)

    scene = build_offline_scene(config)
    trace = []
    maximum_spanwise_fill_range = 0.0
    started = time.perf_counter()
    failure: dict[str, object] | None = None
    with trace_temporary.open("w", encoding="utf-8") as stream:
        try:
            for step in range(config.steps + 1):
                metrics = scene.measure()
                trace.append(metrics)
                spanwise_range = (
                    _spanwise_fill_range(scene)
                    if scene_name is OfflineSceneName.DAMBREAK
                    else 0.0
                )
                maximum_spanwise_fill_range = max(
                    maximum_spanwise_fill_range,
                    spanwise_range,
                )
                record = asdict(metrics)
                record["spanwise_maximum_fill_range"] = spanwise_range
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                if step % config.output_every_steps == 0 or step == config.steps:
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "physical_time": metrics.physical_time,
                                "mass_error": metrics.relative_mass_error,
                                "maximum_speed": metrics.maximum_speed,
                                "front": metrics.occupied_maximum[0],
                                "body_position": metrics.body_position,
                                "spanwise_fill_range": spanwise_range,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if step < config.steps:
                    scene.step()
                    scene.synchronize()
        except Exception as error:
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "step": scene.step_index,
                "physical_time": scene.physical_time,
                "traceback": traceback.format_exc(),
                "state": _capture_failure_state(scene, output),
            }
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    os.replace(trace_temporary, trace_path)

    acceptance = None
    acceptance_failure = None
    if failure is None:
        try:
            acceptance = evaluate_scene_trace(config, trace).to_dict()
        except Exception as error:
            acceptance_failure = {
                "type": type(error).__name__,
                "message": str(error),
            }
    elapsed = time.perf_counter() - started
    final = trace[-1]
    summary: dict[str, object] = {
        "schema_version": 1,
        "scene": scene_name.value,
        "level": level.value,
        "device": str(wp.get_device(device)),
        "config_hash": config.config_hash,
        "requested_steps": config.steps,
        "completed_steps": final.step,
        "physical_end_time": final.physical_time,
        "elapsed_seconds": elapsed,
        "steps_per_second": final.step / elapsed if elapsed > 0.0 else 0.0,
        "simulation_complete": failure is None,
        "acceptance_passed": acceptance is not None,
        "acceptance": acceptance,
        "acceptance_failure": acceptance_failure,
        "failure": failure,
        "maximum_relative_mass_error": max(
            item.relative_mass_error for item in trace
        ),
        "maximum_projected_divergence": max(
            item.projected_max_divergence for item in trace
        ),
        "maximum_speed": max(item.maximum_speed for item in trace),
        "minimum_density": min(
            (item.minimum_density for item in trace[1:]),
            default=0.0,
        ),
        "maximum_density": max(item.maximum_density for item in trace),
        "maximum_spanwise_fill_range": maximum_spanwise_fill_range,
        "initial": asdict(trace[0]),
        "final": asdict(final),
    }
    atomic_write_json(output / "summary.json", summary)
    manifest.update(
        {
            "complete": True,
            "simulation_complete": failure is None,
            "acceptance_passed": acceptance is not None,
            "trace": trace_path.name,
            "summary": "summary.json",
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=[item.value for item in OfflineSceneName], required=True)
    parser.add_argument("--level", choices=[item.value for item in OfflineSceneLevel], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_case(
        OfflineSceneName(args.scene),
        OfflineSceneLevel(args.level),
        device=args.device,
        output=args.output,
    )
    if not summary["simulation_complete"] or not summary["acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
