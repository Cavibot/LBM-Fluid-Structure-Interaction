# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay the first geometric HOME-FSL speed failure with stage-local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    GeometricHomeFslStepper,
    ProjectedGeometricFslStepper,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)


def _moments(state: HomeCoreState) -> np.ndarray:
    return state.moments.numpy().reshape(10, *state.res)


def _speed_field(state: HomeCoreState, fsl: FslState, walls: FslWallMask) -> np.ndarray:
    moments = _moments(state)
    active = (fsl.flags.numpy() != 0) & ~walls.host
    speed = np.full(state.res, np.nan, dtype=np.float64)
    rho = moments[0]
    velocity = np.moveaxis(moments[1:4], 0, -1) / rho[..., None]
    speed[active] = np.linalg.norm(velocity[active], axis=-1)
    return speed


def _patch(field: np.ndarray, coordinate: tuple[int, int, int]) -> list:
    i, j, k = coordinate
    x0, x1 = max(0, i - 1), min(field.shape[0], i + 2)
    y0, y1 = max(0, j - 1), min(field.shape[1], j + 2)
    if field.ndim == 3:
        return field[x0:x1, y0:y1, k].tolist()
    return field[x0:x1, y0:y1, k, ...].tolist()


def _peak(
    state: HomeCoreState,
    fsl: FslState,
    walls: FslWallMask,
) -> dict[str, object]:
    moments = _moments(state)
    speed = _speed_field(state, fsl, walls)
    coordinate = tuple(int(value) for value in np.unravel_index(np.nanargmax(speed), speed.shape))
    i, j, k = coordinate
    rho = float(moments[0, i, j, k])
    momentum = moments[1:4, i, j, k].astype(np.float64)
    return {
        "coordinate": coordinate,
        "speed": float(speed[coordinate]),
        "flag": int(fsl.flags.numpy()[coordinate]),
        "fill": float(fsl.fill_level.numpy()[coordinate]),
        "mass": float(fsl.mass.numpy()[coordinate]),
        "density": rho,
        "momentum": momentum.tolist(),
        "velocity": (momentum / rho).tolist(),
        "speed_patch": _patch(speed, coordinate),
        "fill_patch": _patch(fsl.fill_level.numpy(), coordinate),
        "flag_patch": _patch(fsl.flags.numpy(), coordinate),
    }


def _cell_face_velocity(
    faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    coordinate: tuple[int, int, int],
) -> np.ndarray:
    i, j, k = coordinate
    return np.asarray(
        (
            0.5 * (faces[0][i, j, k] + faces[0][i + 1, j, k]),
            0.5 * (faces[1][i, j, k] + faces[1][i, j + 1, k]),
            0.5 * (faces[2][i, j, k] + faces[2][i, j, k + 1]),
        ),
        dtype=np.float64,
    )


def _render_stage_speeds(
    path: Path,
    stages: tuple[tuple[str, HomeCoreState, FslState], ...],
    walls: FslWallMask,
    step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [(name, _speed_field(fluid, fsl, walls)) for name, fluid, fsl in stages]
    maximum = max(float(np.nanmax(field)) for _, field in fields)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), dpi=180)
    for axis, (name, field) in zip(axes.ravel(), fields, strict=False):
        image = axis.imshow(
            field[:, :, 0].T,
            origin="lower",
            vmin=0.0,
            vmax=maximum,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        coordinate = np.unravel_index(np.nanargmax(field), field.shape)
        axis.plot(coordinate[0], coordinate[1], marker="x", color="cyan")
        axis.set_title(f"{name}: max {np.nanmax(field):.6f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    for axis in axes.ravel()[len(fields) :]:
        axis.set_visible(False)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="lattice speed")
    fig.suptitle(f"Geometric HOME-FSL stage speeds, failed attempt {step}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_replay(
    output_dir: Path,
    *,
    maximum_steps: int = 500,
    endpoint_tolerance: float = 5.0e-6,
    device: str = "cuda:0",
    mode: str = "geometric",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (96, 56, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=2.0e-4,
        body_acceleration=(0.0, -5.0e-6, 0.0),
    )
    walls = FslWallMask.periodic_depth_channel(model)
    fluid_a, fluid_b = HomeCoreState(model), HomeCoreState(model)
    fsl_a, fsl_b = FslState(model), FslState(model)
    scene = GravityColumnConfig(
        resolution_x=shape[0],
        resolution_y=shape[1],
        resolution_z=shape[2],
        periodic_depth=True,
        column_end_x=24,
        column_end_y=34,
        physical_viscosity=2.0e-4,
        physical_gravity_y=-5.0e-6,
        sample_steps=(0, maximum_steps),
        device=device,
    )
    fill = _column_fill(scene, walls)
    walls.initialize_hydrostatic(
        fluid_a,
        fsl_a,
        fill,
        gas_density=1.0,
        gravity_axis=1,
        surface_coordinate=35.0,
    )
    if mode == "geometric":
        stepper = GeometricHomeFslStepper(
            model,
            walls,
            interface_roundoff_tolerance=endpoint_tolerance,
        )
    elif mode == "projected-geometric":
        stepper = ProjectedGeometricFslStepper(
            model,
            walls,
            interface_roundoff_tolerance=endpoint_tolerance,
        )
    else:
        raise ValueError("mode must be geometric or projected-geometric")

    failed_step = 0
    failure: Exception | None = None
    for step in range(1, maximum_steps + 1):
        try:
            stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
        except Exception as error:
            failed_step = step
            failure = error
            break
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    if failure is None:
        raise RuntimeError(f"geometric replay did not fail within {maximum_steps} steps")
    if stepper.last_attempt_diagnostics is None:
        raise RuntimeError("failed geometric attempt did not retain stage diagnostics")
    if stepper.last_transported_fill is None or stepper.last_transported_mass is None:
        raise RuntimeError("failed geometric attempt did not retain transported ledgers")

    candidate_fsl = stepper._candidate_fsl
    stage_list = [
        ("source", fluid_a, fsl_a),
        ("collided", stepper._collided, fsl_a),
    ]
    if mode == "geometric":
        stage_list.append(
            ("pre-momentum", stepper._pre_momentum_fluid, candidate_fsl)
        )
    stage_list.append(("committed", stepper._candidate_fluid, candidate_fsl))
    stages = tuple(stage_list)
    peaks = {
        name: _peak(fluid, fsl, walls)
        for name, fluid, fsl in stages
    }
    coordinate = tuple(peaks["committed"]["coordinate"])
    i, j, k = coordinate
    raw_faces = tuple(field.numpy() for field in stepper.builder.face_courant)
    projected_faces = tuple(field.numpy() for field in stepper.projector.projected_faces)
    raw_velocity = _cell_face_velocity(raw_faces, coordinate)
    projected_velocity = _cell_face_velocity(projected_faces, coordinate)
    transported_mass = stepper.last_transported_mass.numpy()
    local_mass = float(transported_mass[i, j, k])
    committed_velocity = np.asarray(peaks["committed"]["velocity"], dtype=np.float64)
    attempt = stepper.last_attempt_diagnostics
    report: dict[str, object] = {
        "failed_step": failed_step,
        "mode": mode,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure),
        "endpoint_tolerance": endpoint_tolerance,
        "stage_maxima": peaks,
        "failed_attempt_diagnostics": {
            "collided_maximum_speed": attempt.collided_fluid.max_speed,
            "committed_maximum_speed": attempt.fluid.max_speed,
            "raw_maximum_divergence": attempt.projection.initial_maximum_divergence,
            "projected_maximum_divergence": (
                attempt.projection.projected_maximum_divergence
            ),
            "maximum_face_correction": attempt.projection.maximum_face_correction,
            "endpoint_promoted_liquid_count": (
                attempt.topology.endpoint_promoted_liquid_count
            ),
            "separation_repair_count": attempt.topology.separation_repair_count,
            "endpoint_removed_gas_count": (
                attempt.topology.endpoint_removed_gas_count
            ),
            "endpoint_redistributed_volume": (
                attempt.topology.endpoint_redistributed_volume
            ),
            "endpoint_redistributed_mass": (
                attempt.topology.endpoint_redistributed_mass
            ),
        },
        "committed_peak_projection": {
            "coordinate": coordinate,
            "pressure": float(stepper.projector.pressure.numpy()[coordinate]),
            "raw_cell_face_velocity": raw_velocity.tolist(),
            "projected_cell_face_velocity": projected_velocity.tolist(),
            "projection_velocity_correction": (
                projected_velocity - raw_velocity
            ).tolist(),
            "transported_mass": local_mass,
            "committed_velocity": committed_velocity.tolist(),
        },
    }
    transported_momentum = None
    if mode == "geometric":
        transported_momentum = stepper.last_transported_momentum.numpy()
        local_transport_velocity = (
            transported_momentum[i, j, k].astype(np.float64) / local_mass
            if local_mass > 1.0e-12
            else np.zeros(3, dtype=np.float64)
        )
        pre_velocity = np.asarray(
            peaks["pre-momentum"]["velocity"], dtype=np.float64
        )
        report["committed_peak_coupling"] = {
            **report["committed_peak_projection"],
            "transported_liquid_velocity": local_transport_velocity.tolist(),
            "pre_momentum_velocity": pre_velocity.tolist(),
            "applied_velocity_delta": (
                committed_velocity - pre_velocity
            ).tolist(),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    arrays = {
        "source_moments": _moments(fluid_a),
        "source_fill": fsl_a.fill_level.numpy(),
        "source_mass": fsl_a.mass.numpy(),
        "source_flags": fsl_a.flags.numpy(),
        "collided_moments": _moments(stepper._collided),
        "committed_moments": _moments(stepper._candidate_fluid),
        "committed_fill": candidate_fsl.fill_level.numpy(),
        "committed_mass": candidate_fsl.mass.numpy(),
        "committed_flags": candidate_fsl.flags.numpy(),
        "transported_fill": stepper.last_transported_fill.numpy(),
        "transported_mass": transported_mass,
        "projection_pressure": stepper.projector.pressure.numpy(),
    }
    if transported_momentum is not None:
        arrays["pre_momentum_moments"] = _moments(
            stepper._pre_momentum_fluid
        )
        arrays["transported_momentum"] = transported_momentum
    np.savez_compressed(output_dir / "failed-state.npz", **arrays)
    _render_stage_speeds(
        output_dir / "stage-speed-comparison.png",
        stages,
        walls,
        failed_step,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-steps", type=int, default=500)
    parser.add_argument("--endpoint-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("geometric", "projected-geometric"),
        default="geometric",
    )
    args = parser.parse_args()
    report = run_replay(
        args.output,
        maximum_steps=args.maximum_steps,
        endpoint_tolerance=args.endpoint_tolerance,
        device=args.device,
        mode=args.mode,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
