# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless geometric HOME-Free liquid-column example."""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from wanphys.examples.lbm.home_free_offline import (
    OfflineSceneName,
    build_dambreak_fill,
    build_offline_scene,
    make_scene_config,
)


def build_column_fill(shape: tuple[int, int, int]) -> np.ndarray:
    return build_dambreak_fill(shape)


def run(steps: int, device: str) -> dict[str, float]:
    config = make_scene_config(OfflineSceneName.DAMBREAK, device=device)
    scene = build_offline_scene(config)
    scene.run(steps)
    scene.synchronize()
    metrics = scene.measure()
    return {
        "steps": float(steps),
        "relative_mass_error": metrics.relative_mass_error,
        "liquid_volume": metrics.liquid_volume,
        "maximum_fill": metrics.maximum_fill,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda:0" if wp.is_cuda_available() else "cpu"
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    for key, value in run(args.steps, args.device).items():
        print(f"{key}: {value:.9g}")


if __name__ == "__main__":
    main()
