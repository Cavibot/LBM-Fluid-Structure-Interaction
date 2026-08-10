# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit raw-Courant PLIC-VOF as a non-feedback shadow of Viewer Baseline V1."""

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
    _simulation_cell_size,
    make_viewer_config,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    _column_fill,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.shadow_vof import (
    ShadowPlicVof,
)


def _fingerprint(fluid: HomeCoreState, fsl: FslState) -> str:
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
    *,
    steps: int,
    device: str,
    project_courant: bool = False,
    projection_maximum_iterations: int = 320,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive")
    config = make_viewer_config("viewer-default", device)
    shape = (
        config.resolution_x,
        config.resolution_y,
        config.resolution_z,
    )
    model = HomeCoreModel(
        fluid_grid_res=shape,
        fluid_grid_cell_size=_simulation_cell_size(config),
        kinematic_viscosity=config.physical_viscosity,
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
        device=device,
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
    stepper = FslWallDynamicTopologyStepper(model, walls)
    shadow = ShadowPlicVof(
        model,
        walls,
        fsl_a.fill_level,
        project_courant=project_courant,
        projection_maximum_iterations=projection_maximum_iterations,
    )
    initial_baseline_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    baseline_seconds = 0.0
    shadow_seconds = 0.0
    verification_seconds = 0.0
    maximum_speed = 0.0
    maximum_divergence = 0.0
    maximum_projected_divergence = 0.0
    maximum_projection_iterations = 0
    maximum_face_correction = 0.0
    maximum_step_drift = 0.0
    state_unchanged = True
    shadow_failure_step: int | None = None
    shadow_error: str | None = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        phase = time.perf_counter()
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        baseline_seconds += time.perf_counter() - phase
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        maximum_speed = max(maximum_speed, diagnostics.fluid.max_speed)
        if shadow_failure_step is not None:
            continue
        verify = step in (1, steps)
        phase = time.perf_counter()
        before = _fingerprint(fluid_a, fsl_a) if verify else None
        verification_seconds += time.perf_counter() - phase
        phase = time.perf_counter()
        try:
            shadow_diagnostics = shadow.advance(fluid_a, fsl_a)
        except Exception as error:
            shadow_failure_step = step
            shadow_error = f"{type(error).__name__}: {error}"
            shadow_seconds += time.perf_counter() - phase
            continue
        shadow_seconds += time.perf_counter() - phase
        phase = time.perf_counter()
        after = _fingerprint(fluid_a, fsl_a) if verify else None
        verification_seconds += time.perf_counter() - phase
        state_unchanged &= before == after
        maximum_divergence = max(
            maximum_divergence,
            shadow_diagnostics.maximum_active_divergence,
        )
        if shadow_diagnostics.projected_maximum_divergence is not None:
            maximum_projected_divergence = max(
                maximum_projected_divergence,
                shadow_diagnostics.projected_maximum_divergence,
            )
        maximum_projection_iterations = max(
            maximum_projection_iterations,
            shadow_diagnostics.projection_iteration_count,
        )
        maximum_face_correction = max(
            maximum_face_correction,
            shadow_diagnostics.maximum_face_correction,
        )
        maximum_step_drift = max(
            maximum_step_drift,
            abs(shadow_diagnostics.step_relative_volume_drift),
        )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - started
    final_baseline_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    shadow_values = shadow.fill_level.numpy()
    active_values = shadow_values[~walls.host]
    last = shadow.last_diagnostics
    return {
        "stage": (
            "B2P-projected-shadow-PLIC-VOF"
            if project_courant
            else "B2-raw-Courant-shadow-PLIC-VOF"
        ),
        "configuration": {
            "resolution": shape,
            "requested_steps": steps,
            "completed_shadow_steps": shadow.step_index,
            "split_orders": [[0, 1, 2], [2, 1, 0]],
            "project_courant": project_courant,
            "projection_maximum_iterations": projection_maximum_iterations,
        },
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": device,
        },
        "baseline_state_unchanged_by_shadow": state_unchanged,
        "shadow_failure_step": shadow_failure_step,
        "shadow_error": shadow_error,
        "drift_violation_count": shadow.drift_violation_count,
        "maximum_active_divergence": maximum_divergence,
        "maximum_projected_divergence": (
            maximum_projected_divergence if project_courant else None
        ),
        "maximum_projection_iterations": maximum_projection_iterations,
        "maximum_face_correction": maximum_face_correction,
        "maximum_absolute_step_volume_drift": maximum_step_drift,
        "final_cumulative_shadow_volume_drift": (
            last.cumulative_relative_volume_drift if last is not None else None
        ),
        "shadow_minimum_fill": float(np.min(active_values, initial=0.0)),
        "shadow_maximum_fill": float(np.max(active_values, initial=0.0)),
        "baseline_maximum_speed": maximum_speed,
        "baseline_relative_total_mass_drift": (
            final_baseline_mass - initial_baseline_mass
        )
        / initial_baseline_mass,
        "elapsed_seconds": elapsed,
        "verification_seconds": verification_seconds,
        "baseline_seconds": baseline_seconds,
        "shadow_seconds": shadow_seconds,
        "baseline_ms_per_step": baseline_seconds * 1000.0 / steps,
        "shadow_ms_per_completed_step": (
            shadow_seconds * 1000.0 / max(shadow.step_index, 1)
        ),
        "combined_ms_per_step": (
            baseline_seconds + shadow_seconds
        )
        * 1000.0
        / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-courant", action="store_true")
    parser.add_argument("--projection-maximum-iterations", type=int, default=320)
    args = parser.parse_args()
    result = run_audit(
        steps=args.steps,
        device=args.device,
        project_courant=args.project_courant,
        projection_maximum_iterations=args.projection_maximum_iterations,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
