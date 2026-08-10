#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare completed HOME-Free convergence traces in physical units."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from wanphys.examples.lbm.home_free_offline.capture import atomic_write_json
from wanphys.examples.lbm.home_free_offline.config import (
    OfflineSceneLevel,
    OfflineSceneName,
)
from wanphys.examples.lbm.home_free_offline.convergence import (
    make_convergence_config,
)


def _read_trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _physical_series(
    records: list[dict[str, object]], cell_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray([record["physical_time"] for record in records], dtype=np.float64)
    centroid = np.asarray(
        [record["liquid_centroid"] for record in records], dtype=np.float64
    )
    centroid = (centroid + 0.5) * cell_size
    front = np.asarray(
        [record["occupied_maximum"][0] for record in records], dtype=np.float64
    )
    front = (front + 1.0) * cell_size
    return time, centroid, front


def _aligned_error(
    coarse: tuple[np.ndarray, np.ndarray, np.ndarray],
    fine: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    coarse_time, coarse_centroid, coarse_front = coarse
    fine_time, fine_centroid, fine_front = fine
    fine_indices = np.searchsorted(fine_time, coarse_time)
    if np.any(fine_indices >= len(fine_time)) or not np.allclose(
        fine_time[fine_indices], coarse_time, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("convergence traces do not share aligned physical times")
    centroid_delta = coarse_centroid[:, (0, 2)] - fine_centroid[fine_indices][:, (0, 2)]
    front_delta = coarse_front - fine_front[fine_indices]
    return {
        "centroid_xz_rms": float(np.sqrt(np.mean(np.sum(centroid_delta**2, axis=1)))),
        "front_rms": float(np.sqrt(np.mean(front_delta**2))),
        "final_centroid_xz": float(np.linalg.norm(centroid_delta[-1])),
        "final_front": float(abs(front_delta[-1])),
    }


def _observed_order(coarse_error: float, fine_error: float) -> float | None:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    return math.log(coarse_error / fine_error, 2.0)


def compare(root: Path, scene: OfflineSceneName, suffix: str) -> dict[str, object]:
    traces = {}
    levels = {}
    for level in OfflineSceneLevel:
        output = root / f"{level.value}-{suffix}"
        manifest = json.loads((output / "manifest.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        if not manifest.get("numerically_qualified") or not summary.get(
            "acceptance_passed"
        ):
            raise RuntimeError(f"{level.value} is not numerically qualified")
        config = make_convergence_config(scene, level)
        records = _read_trace(output / "trace.jsonl")
        traces[level] = _physical_series(records, config.fluid.cell_size)
        acceptance = summary["acceptance"]
        levels[level.value] = {
            "steps": summary["completed_steps"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "maximum_physical_speed": (
                acceptance["maximum_speed"]
                * config.fluid.cell_size
                / config.fluid.time_step
            ),
            "maximum_relative_mass_error": acceptance[
                "maximum_relative_mass_error"
            ],
            "maximum_projected_divergence": acceptance[
                "maximum_projected_divergence"
            ],
            "maximum_spanwise_fill_range": summary[
                "maximum_spanwise_fill_range"
            ],
            "final_physical_centroid": traces[level][1][-1].tolist(),
            "final_physical_front": float(traces[level][2][-1]),
        }

    coarse_error = _aligned_error(
        traces[OfflineSceneLevel.L0], traces[OfflineSceneLevel.L1]
    )
    fine_error = _aligned_error(
        traces[OfflineSceneLevel.L1], traces[OfflineSceneLevel.L2]
    )
    observed_order = {
        key: _observed_order(coarse_error[key], fine_error[key])
        for key in coarse_error
    }
    return {
        "schema_version": 1,
        "scene": scene.value,
        "levels": levels,
        "pairwise_error": {"L0_L1": coarse_error, "L1_L2": fine_error},
        "observed_order": observed_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", choices=[item.value for item in OfflineSceneName], required=True)
    parser.add_argument("--suffix", default="workflow-final")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.root, OfflineSceneName(args.scene), args.suffix)
    output = args.output or args.root / "convergence-summary.json"
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
