# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare fused on-demand HOME against an equivalent split D3Q27 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_WEIGHTS,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm import kernels


@wp.kernel
def _periodic_pull_populations(
    source: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    destination: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    direction, i, j, k = wp.tid()
    c = directions[direction]
    si = (i - int(c[0]) + nx) % nx
    sj = (j - int(c[1]) + ny) % ny
    sk = (k - int(c[2]) + nz) % nz
    cell = i * ny * nz + j * nz + k
    source_cell = si * ny * nz + sj * nz + sk
    destination[direction * stride + cell] = source[
        direction * stride + source_cell
    ]


class SplitHomeStepper:
    def __init__(self, model: HomeLbmModel) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        self.weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self.post_collision_populations = wp.zeros(
            27 * self.stride, dtype=float, device=self.device
        )
        self.streamed_populations = wp.zeros_like(self.post_collision_populations)
        self.streamed_moments = wp.zeros(
            10 * self.stride, dtype=float, device=self.device
        )
        self.active = wp.ones(self.res, dtype=wp.int32, device=self.device)
        self.invalid = wp.zeros(1, dtype=wp.int32, device=self.device)

    def step(self, source: HomeLbmState, destination: HomeLbmState) -> None:
        self.invalid.zero_()
        wp.launch(
            kernels.reconstruct_populations_kernel,
            dim=(27, self.stride),
            inputs=[
                source.moments,
                self.directions,
                self.weights,
                self.post_collision_populations,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            _periodic_pull_populations,
            dim=(27, *self.res),
            inputs=[
                self.post_collision_populations,
                self.directions,
                self.streamed_populations,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.extract_moments_kernel,
            dim=self.stride,
            inputs=[
                self.streamed_populations,
                self.directions,
                self.streamed_moments,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.collide_force_free_moments_kernel,
            dim=self.res,
            inputs=[
                self.streamed_moments,
                self.active,
                destination.moments,
                self.invalid,
                float(self.model.shear_omega),
                int(self.model.ny),
                int(self.model.nz),
                self.stride,
            ],
            device=self.device,
        )


def _time_steps(step, source, destination, steps: int, device: str) -> float:
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(steps):
        step(source, destination)
        source, destination = destination, source
    wp.synchronize_device(device)
    return 1000.0 * (time.perf_counter() - start) / steps


def run(args: argparse.Namespace) -> dict[str, object]:
    resolution = tuple(int(value) for value in args.resolution.split("x"))
    if len(resolution) != 3 or any(value <= 0 for value in resolution):
        raise ValueError("resolution must be NXxNYxNZ")
    model = HomeLbmModel(
        fluid_grid_res=resolution,
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        reference_density=1.0,
        kinematic_viscosity=0.2,
        periodic=(True, True, True),
        device=args.device,
    )
    fused_solver = HomeLbmSolver(model)
    fused = (HomeLbmState(model), HomeLbmState(model))
    split = (HomeLbmState(model), HomeLbmState(model))
    for state in (fused[0], split[0]):
        fused_solver.initialize_uniform_lattice(
            state, velocity=(0.02, -0.01, 0.005)
        )
    wp.copy(fused[1].moments, fused[0].moments)
    wp.copy(split[1].moments, split[0].moments)
    split_stepper = SplitHomeStepper(model)

    fused_step = lambda source, destination: fused_solver.step(
        source, destination, model.time_step
    )
    for _ in range(args.warmup):
        fused_step(fused[0], fused[1])
        split_stepper.step(split[0], split[1])
    wp.synchronize_device(args.device)

    fused_ms = _time_steps(fused_step, *fused, args.steps, args.device)
    split_ms = _time_steps(split_stepper.step, *split, args.steps, args.device)

    fused_check = (HomeLbmState(model), HomeLbmState(model))
    split_check = (HomeLbmState(model), HomeLbmState(model))
    fused_solver.initialize_uniform_lattice(
        fused_check[0], velocity=(0.02, -0.01, 0.005)
    )
    wp.copy(split_check[0].moments, fused_check[0].moments)
    fused_step(*fused_check)
    split_stepper.step(*split_check)
    wp.synchronize_device(args.device)
    maximum_difference = float(
        np.max(
            np.abs(
                fused_check[1].moments.numpy()
                - split_check[1].moments.numpy()
            )
        )
    )
    cells = int(np.prod(resolution))
    return {
        "device": str(wp.get_device(args.device)),
        "resolution": resolution,
        "cells": cells,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "fused_mean_step_ms": fused_ms,
        "split_mean_step_ms": split_ms,
        "fused_mlups": cells / (fused_ms * 1000.0),
        "split_mlups": cells / (split_ms * 1000.0),
        "split_over_fused_time": split_ms / fused_ms,
        "fused_persistent_bytes_per_cell": 40,
        "split_pdf_scratch_bytes_per_cell": 216,
        "split_total_scratch_bytes_per_cell": 256,
        "one_step_maximum_moment_difference": maximum_difference,
        "warp_mempool_peak_mib": wp.get_mempool_used_mem_high(args.device) / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="54x54x86")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
