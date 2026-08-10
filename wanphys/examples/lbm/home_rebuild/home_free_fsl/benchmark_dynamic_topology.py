# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the validated R3.5 dynamic HOME-Free transaction."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslDynamicTopologyStepper,
    FslState,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", nargs=3, type=int, default=(128, 128, 64))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    shape = tuple(args.resolution)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=args.device,
        kinematic_viscosity=0.02,
    )
    domain = HomeCoreDomain(model)
    fluid_a = domain.create_state()
    fluid_b = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid_a, velocity=(0.05, 0.0, 0.0))
    fill = np.zeros(shape, dtype=np.float32)
    start = shape[0] // 4
    end = 3 * shape[0] // 4
    fill[start] = 0.5
    fill[start + 1 : end] = 1.0
    fill[end] = 0.5
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fsl_a.initialize_from_fill_level(fluid_a, fill)
    stepper = FslDynamicTopologyStepper(model)
    for _ in range(args.warmup):
        stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    wp.synchronize_device(args.device)
    started = time.perf_counter()
    transitions = 0
    max_drift = 0.0
    for _ in range(args.steps):
        diagnostics = stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
        transitions += (
            diagnostics.topology.interface_to_liquid_count
            + diagnostics.topology.interface_to_gas_count
        )
        max_drift = max(max_drift, abs(diagnostics.topology.relative_total_mass_drift))
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    wp.synchronize_device(args.device)
    elapsed = time.perf_counter() - started
    cells = int(np.prod(shape))
    print(
        json.dumps(
            {
                "resolution": shape,
                "steps": args.steps,
                "elapsed_seconds": elapsed,
                "mlups": cells * args.steps / elapsed / 1.0e6,
                "topology_transitions": transitions,
                "max_step_relative_mass_drift": max_drift,
                "device": args.device,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
