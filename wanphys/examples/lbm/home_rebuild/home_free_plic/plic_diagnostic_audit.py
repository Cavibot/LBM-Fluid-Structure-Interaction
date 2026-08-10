# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit read-only PLIC reconstruction against the frozen Viewer baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    _maximum_lattice_speed,
    _simulation_cell_size,
    make_viewer_config,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    _column_fill,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.plic_diagnostic_viewer import (
    ReadOnlyPlicObserver,
)


def _state_fingerprint(fluid: HomeCoreState, fsl: FslState) -> str:
    digest = hashlib.sha256()
    for array in (
        fluid.moments,
        fsl.mass,
        fsl.fill_level,
        fsl.excess_mass,
        fsl.flags,
    ):
        digest.update(np.ascontiguousarray(array.numpy()).tobytes())
    return digest.hexdigest()


def run_audit(
    *, steps: int, observer_interval: int, device: str
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if observer_interval < 1:
        raise ValueError("observer_interval must be positive")
    config = make_viewer_config("viewer-default", device)
    shape = (
        config.resolution_x,
        config.resolution_y,
        config.resolution_z,
    )
    model = HomeCoreModel(
        fluid_grid_res=shape,
        fluid_grid_cell_size=_simulation_cell_size(config),
        device=device,
        kinematic_viscosity=config.physical_viscosity,
        max_lattice_speed=_maximum_lattice_speed(config),
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
    )
    walls = FslWallMask.closed_box(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fill = _column_fill(config, walls)
    walls.initialize_hydrostatic(
        fluid_a,
        fsl_a,
        fill,
        gas_density=config.gas_density,
        gravity_axis=1,
        surface_coordinate=float(config.column_end_y + 1),
    )
    fluid_b.copy_from(fluid_a)
    fsl_b.copy_from(fsl_a)
    stepper = FslWallDynamicTopologyStepper(
        model, walls, gas_density=config.gas_density
    )
    observer = ReadOnlyPlicObserver(model, walls)
    initial_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    observations = 0
    observation_seconds = 0.0
    maximum_interface_count = 0
    maximum_interface_area = 0.0
    maximum_speed = 0.0
    state_unchanged = True
    verification_seconds = 0.0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        maximum_speed = max(maximum_speed, diagnostics.fluid.max_speed)
        if step % observer_interval == 0 or step == steps:
            verify_state = step == observer_interval or step == steps
            verification_started = time.perf_counter()
            before = _state_fingerprint(fluid_a, fsl_a) if verify_state else None
            verification_seconds += time.perf_counter() - verification_started
            plic = observer.observe(fsl_a)
            verification_started = time.perf_counter()
            after = _state_fingerprint(fluid_a, fsl_a) if verify_state else None
            verification_seconds += time.perf_counter() - verification_started
            state_unchanged &= before == after
            observations += 1
            observation_seconds += observer.last_elapsed_ms / 1000.0
            maximum_interface_count = max(
                maximum_interface_count, plic.interface_cell_count
            )
            maximum_interface_area = max(
                maximum_interface_area, plic.maximum_interface_area
            )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - started
    final_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    cells = int(np.prod(shape))
    operational_seconds = elapsed - verification_seconds
    return {
        "stage": "B1-read-only-PLIC",
        "configuration": {
            "resolution": shape,
            "steps": steps,
            "observer_interval": observer_interval,
            "physical_viscosity": config.physical_viscosity,
            "physical_gravity_y": config.physical_gravity_y,
            "lattice_viscosity": model.lattice_viscosity,
            "lattice_acceleration": model.lattice_acceleration,
        },
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": device,
        },
        "state_unchanged_by_plic": state_unchanged,
        "observation_count": observations,
        "invalid_interface_count": int(
            observer.last_diagnostics.invalid_interface_count
            if observer.last_diagnostics is not None
            else 0
        ),
        "maximum_interface_cell_count": maximum_interface_count,
        "maximum_interface_area": maximum_interface_area,
        "maximum_active_speed": maximum_speed,
        "relative_total_mass_drift": (final_mass - initial_mass) / initial_mass,
        "elapsed_seconds": elapsed,
        "verification_seconds": verification_seconds,
        "operational_seconds": operational_seconds,
        "observation_seconds": observation_seconds,
        "solve_estimated_seconds": operational_seconds - observation_seconds,
        "combined_ms_per_step": operational_seconds * 1000.0 / steps,
        "observer_ms_per_observation": (
            observation_seconds * 1000.0 / observations
        ),
        "combined_mlups": cells * steps / operational_seconds / 1.0e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--observer-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(
        steps=args.steps,
        observer_interval=args.observer_interval,
        device=args.device,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
