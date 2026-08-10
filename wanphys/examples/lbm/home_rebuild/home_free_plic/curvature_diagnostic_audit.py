# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit read-only bulk 3-D PLIC curvature on the frozen Viewer baseline."""

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
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicCurvatureEstimator3D,
    PlicCurvatureState,
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
    geometry_observer = ReadOnlyPlicObserver(model, walls)
    curvature_state = PlicCurvatureState(model)
    curvature_estimator = PlicCurvatureEstimator3D(
        model, walls, strict_bulk=False
    )
    initial_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    observations = 0
    geometry_seconds = 0.0
    curvature_seconds = 0.0
    verification_seconds = 0.0
    maximum_speed = 0.0
    maximum_required = 0
    maximum_valid = 0
    maximum_wall_skipped = 0
    maximum_insufficient = 0
    maximum_ill_conditioned = 0
    maximum_invalid = 0
    maximum_absolute_curvature = 0.0
    minimum_bulk_completeness = 1.0
    state_unchanged = True
    started = time.perf_counter()
    for step in range(1, steps + 1):
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        maximum_speed = max(maximum_speed, diagnostics.fluid.max_speed)
        if step % observer_interval != 0 and step != steps:
            continue
        verify_state = step == observer_interval or step == steps
        verification_started = time.perf_counter()
        before = _state_fingerprint(fluid_a, fsl_a) if verify_state else None
        verification_seconds += time.perf_counter() - verification_started
        geometry_observer.observe(fsl_a)
        geometry_seconds += geometry_observer.last_elapsed_ms / 1000.0
        curvature_started = time.perf_counter()
        curvature = curvature_estimator.reconstruct(
            fsl_a.fill_level,
            fsl_a.flags,
            geometry_observer.geometry,
            curvature_state,
        )
        wp.synchronize_device(device)
        curvature_seconds += time.perf_counter() - curvature_started
        verification_started = time.perf_counter()
        after = _state_fingerprint(fluid_a, fsl_a) if verify_state else None
        verification_seconds += time.perf_counter() - verification_started
        state_unchanged &= before == after
        observations += 1
        maximum_required = max(maximum_required, curvature.required_cell_count)
        maximum_valid = max(maximum_valid, curvature.valid_cell_count)
        maximum_wall_skipped = max(
            maximum_wall_skipped, curvature.wall_contact_skipped_count
        )
        maximum_insufficient = max(
            maximum_insufficient, curvature.insufficient_neighbor_count
        )
        maximum_ill_conditioned = max(
            maximum_ill_conditioned, curvature.ill_conditioned_count
        )
        maximum_invalid = max(
            maximum_invalid, curvature.invalid_curvature_count
        )
        bulk_required = (
            curvature.required_cell_count
            - curvature.wall_contact_skipped_count
        )
        if bulk_required > 0:
            minimum_bulk_completeness = min(
                minimum_bulk_completeness,
                curvature.valid_cell_count / bulk_required,
            )
        values = curvature_state.curvature.numpy()
        valid = curvature_state.valid.numpy() != 0
        if np.any(valid):
            maximum_absolute_curvature = max(
                maximum_absolute_curvature,
                float(np.max(np.abs(values[valid]))),
            )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - started
    final_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    cells = int(np.prod(shape))
    operational_seconds = elapsed - verification_seconds
    observation_seconds = geometry_seconds + curvature_seconds
    return {
        "stage": "B5-read-only-bulk-3D-curvature",
        "configuration": {
            "resolution": shape,
            "steps": steps,
            "observer_interval": observer_interval,
            "physical_viscosity": config.physical_viscosity,
            "physical_gravity_y": config.physical_gravity_y,
            "lattice_viscosity": model.lattice_viscosity,
            "lattice_acceleration": model.lattice_acceleration,
            "wall_contact_policy": "reported-and-skipped-until-wetting-stage",
        },
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": device,
        },
        "state_unchanged_by_curvature": state_unchanged,
        "observation_count": observations,
        "maximum_required_cell_count": maximum_required,
        "maximum_valid_cell_count": maximum_valid,
        "maximum_wall_contact_skipped_count": maximum_wall_skipped,
        "maximum_insufficient_neighbor_count": maximum_insufficient,
        "maximum_ill_conditioned_count": maximum_ill_conditioned,
        "maximum_invalid_curvature_count": maximum_invalid,
        "minimum_bulk_completeness": minimum_bulk_completeness,
        "maximum_absolute_curvature": maximum_absolute_curvature,
        "maximum_active_speed": maximum_speed,
        "relative_total_mass_drift": (final_mass - initial_mass) / initial_mass,
        "elapsed_seconds": elapsed,
        "verification_seconds": verification_seconds,
        "operational_seconds": operational_seconds,
        "geometry_seconds": geometry_seconds,
        "curvature_seconds": curvature_seconds,
        "curvature_ms_per_observation": (
            curvature_seconds * 1000.0 / observations
        ),
        "combined_ms_per_step": operational_seconds * 1000.0 / steps,
        "combined_mlups": cells * steps / operational_seconds / 1.0e6,
        "solve_estimated_seconds": operational_seconds - observation_seconds,
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
