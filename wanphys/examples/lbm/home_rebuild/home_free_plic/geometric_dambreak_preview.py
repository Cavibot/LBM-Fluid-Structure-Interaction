# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a matched link-wise/geometric HOME-FSL Dam-break preview."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    GeometricHomeFslStepper,
    GvofFslWallStepper,
    ProjectedGeometricFslStepper,
    ProjectedGvofFslWallStepper,
    ProjectedVelocityGeometricHomeFslStepper,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
    _impact_audit,
    _render_frame,
    _render_overview,
    _snapshot,
    _wall_film_metrics,
)
from wanphys.examples.lbm.home_rebuild.volume_artifacts import (
    SOLVER_AXIS_ORDER,
    sha256_file,
)


@dataclass(frozen=True)
class GeometricDambreakConfig:
    mode: str = "geometric"
    resolution_x: int = 96
    resolution_y: int = 56
    resolution_z: int = 1
    cell_size: float = 0.1
    time_step: float = 1.0
    column_end_x: int = 24
    column_end_y: int = 34
    column_offset_x: int = 0
    physical_viscosity: float = 2.0e-4
    physical_gravity_y: float = -5.0e-6
    gas_density: float = 1.0
    interface_roundoff_tolerance: float = 2.0e-6
    contact_angle_degrees: float | None = None
    sample_steps: tuple[int, ...] = (0, 10, 25, 50, 100, 200)
    device: str = "cuda:0"
    allow_incomplete: bool = False


def run_scene(config: GeometricDambreakConfig, output_dir: Path) -> dict[str, object]:
    if config.mode not in (
        "link-wise",
        "gvof",
        "projected-gvof",
        "projected-geometric",
        "projected-momentum",
        "geometric",
    ):
        raise ValueError(
            "mode must be link-wise, gvof, projected-gvof, "
            "projected-geometric, projected-momentum, or geometric"
        )
    if config.contact_angle_degrees is not None and config.mode not in (
        "projected-geometric",
        "projected-momentum",
        "geometric",
    ):
        raise ValueError("contact angle is only available for geometric PLIC modes")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    shape = (config.resolution_x, config.resolution_y, config.resolution_z)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        fluid_grid_cell_size=config.cell_size,
        time_step=config.time_step,
        device=config.device,
        kinematic_viscosity=config.physical_viscosity,
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
    )
    walls = FslWallMask.periodic_depth_channel(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    legacy_config = GravityColumnConfig(
        profile=f"r4.1-{config.mode}-dambreak-preview",
        resolution_x=config.resolution_x,
        resolution_y=config.resolution_y,
        resolution_z=config.resolution_z,
        periodic_depth=True,
        column_end_x=config.column_end_x,
        column_end_y=config.column_end_y,
        column_offset_x=config.column_offset_x,
        physical_viscosity=config.physical_viscosity,
        physical_gravity_y=config.physical_gravity_y,
        gas_density=config.gas_density,
        sample_steps=config.sample_steps,
        device=config.device,
    )
    fill = _column_fill(legacy_config, walls)
    walls.initialize_hydrostatic(
        fluid_a,
        fsl_a,
        fill,
        gas_density=config.gas_density,
        gravity_axis=1,
        surface_coordinate=float(config.column_end_y + 1),
    )
    if config.mode == "projected-momentum":
        stepper = ProjectedVelocityGeometricHomeFslStepper(
            model,
            walls,
            gas_density=config.gas_density,
            interface_roundoff_tolerance=config.interface_roundoff_tolerance,
            contact_angle_degrees=config.contact_angle_degrees,
        )
    elif config.mode == "geometric":
        stepper = GeometricHomeFslStepper(
            model,
            walls,
            gas_density=config.gas_density,
            interface_roundoff_tolerance=config.interface_roundoff_tolerance,
            contact_angle_degrees=config.contact_angle_degrees,
        )
    elif config.mode == "gvof":
        stepper = GvofFslWallStepper(
            model,
            walls,
            gas_density=config.gas_density,
            interface_roundoff_tolerance=config.interface_roundoff_tolerance,
        )
    elif config.mode == "projected-gvof":
        stepper = ProjectedGvofFslWallStepper(
            model,
            walls,
            gas_density=config.gas_density,
            interface_roundoff_tolerance=config.interface_roundoff_tolerance,
        )
    elif config.mode == "projected-geometric":
        stepper = ProjectedGeometricFslStepper(
            model,
            walls,
            gas_density=config.gas_density,
            interface_roundoff_tolerance=config.interface_roundoff_tolerance,
            contact_angle_degrees=config.contact_angle_degrees,
        )
    else:
        stepper = FslWallDynamicTopologyStepper(
            model, walls, gas_density=config.gas_density
        )
    snapshots = {0: _snapshot(fluid_a, fsl_a)}
    samples = set(config.sample_steps[1:])
    maximum_speed = 0.0
    maximum_collided_speed = 0.0
    maximum_projection_iterations = 0
    maximum_raw_divergence = 0.0
    maximum_projected_divergence = 0.0
    maximum_face_correction = 0.0
    maximum_sliver_count = 0
    maximum_endpoint_promoted_count = 0
    maximum_separation_repair_count = 0
    maximum_endpoint_removed_count = 0
    maximum_endpoint_fallback_face_count = 0
    maximum_wetting_cell_count = 0
    maximum_parallel_wall_interface_count = 0
    total_endpoint_promoted_count = 0
    total_separation_repair_count = 0
    total_endpoint_removed_count = 0
    total_endpoint_fallback_face_count = 0
    total_wetting_cell_count = 0
    total_parallel_wall_interface_count = 0
    total_endpoint_redistributed_volume = 0.0
    total_endpoint_redistributed_mass = 0.0
    completed_step = 0
    failure: dict[str, object] | None = None
    started = time.perf_counter()
    for step in range(1, config.sample_steps[-1] + 1):
        try:
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
        except Exception as error:
            if not config.allow_incomplete:
                raise RuntimeError(
                    f"{config.mode} Dam-break failed at step {step}: {error}"
                ) from error
            failure = {
                "step": step,
                "type": type(error).__name__,
                "message": str(error),
            }
            attempt = getattr(stepper, "last_attempt_diagnostics", None)
            if attempt is not None:
                failure["committed_maximum_speed"] = attempt.fluid.max_speed
                failure["collided_maximum_speed"] = (
                    attempt.collided_fluid.max_speed
                )
            if completed_step not in snapshots:
                snapshots[completed_step] = _snapshot(fluid_a, fsl_a)
            break
        maximum_speed = max(maximum_speed, diagnostics.fluid.max_speed)
        maximum_collided_speed = max(
            maximum_collided_speed,
            getattr(diagnostics, "collided_fluid", diagnostics.fluid).max_speed,
        )
        if config.mode in (
            "projected-gvof",
            "projected-geometric",
            "projected-momentum",
            "geometric",
        ):
            maximum_projection_iterations = max(
                maximum_projection_iterations,
                diagnostics.projection.iteration_count,
            )
            maximum_raw_divergence = max(
                maximum_raw_divergence,
                diagnostics.projection.initial_maximum_divergence,
            )
            maximum_projected_divergence = max(
                maximum_projected_divergence,
                diagnostics.projection.projected_maximum_divergence,
            )
            maximum_face_correction = max(
                maximum_face_correction,
                diagnostics.projection.maximum_face_correction,
            )
        if config.mode in (
            "projected-geometric",
            "projected-momentum",
            "geometric",
        ):
            endpoint_fallback_faces = sum(
                axis.endpoint_fallback_face_count
                for axis in diagnostics.transport.axis_diagnostics
            )
            wetting_cells = sum(
                axis.wetting_cell_count
                for axis in diagnostics.transport.axis_diagnostics
            )
            parallel_wall_interfaces = sum(
                axis.parallel_wall_interface_cell_count
                for axis in diagnostics.transport.axis_diagnostics
            )
            maximum_endpoint_fallback_face_count = max(
                maximum_endpoint_fallback_face_count,
                endpoint_fallback_faces,
            )
            total_endpoint_fallback_face_count += endpoint_fallback_faces
            maximum_wetting_cell_count = max(
                maximum_wetting_cell_count,
                wetting_cells,
            )
            total_wetting_cell_count += wetting_cells
            maximum_parallel_wall_interface_count = max(
                maximum_parallel_wall_interface_count,
                parallel_wall_interfaces,
            )
            total_parallel_wall_interface_count += parallel_wall_interfaces
            maximum_sliver_count = max(
                maximum_sliver_count,
                diagnostics.topology.sub_tolerance_interface_count,
            )
            maximum_endpoint_promoted_count = max(
                maximum_endpoint_promoted_count,
                diagnostics.topology.endpoint_promoted_liquid_count,
            )
            maximum_separation_repair_count = max(
                maximum_separation_repair_count,
                diagnostics.topology.separation_repair_count,
            )
            maximum_endpoint_removed_count = max(
                maximum_endpoint_removed_count,
                diagnostics.topology.endpoint_removed_gas_count,
            )
            total_endpoint_promoted_count += (
                diagnostics.topology.endpoint_promoted_liquid_count
            )
            total_separation_repair_count += (
                diagnostics.topology.separation_repair_count
            )
            total_endpoint_removed_count += (
                diagnostics.topology.endpoint_removed_gas_count
            )
            total_endpoint_redistributed_volume += (
                diagnostics.topology.endpoint_redistributed_volume
            )
            total_endpoint_redistributed_mass += (
                diagnostics.topology.endpoint_redistributed_mass
            )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        completed_step = step
        if step in samples:
            snapshots[step] = _snapshot(fluid_a, fsl_a)
    wp.synchronize_device(config.device)
    elapsed = time.perf_counter() - started

    initial_volume = float(np.sum(snapshots[0]["fill"], dtype=np.float64))
    initial_mass = float(
        np.sum(snapshots[0]["mass"], dtype=np.float64)
        + np.sum(snapshots[0]["excess"], dtype=np.float64)
    )
    metrics: dict[str, dict[str, float | int]] = {}
    frames: list[str] = []
    volume_snapshots: list[dict[str, object]] = []
    for step, snapshot in snapshots.items():
        frame = f"frames/step-{step:06d}.png"
        _render_frame(output_dir / frame, step, snapshot, legacy_config.profile)
        frames.append(frame)
        snapshot_name = f"step-{step:06d}.npz"
        snapshot_path = snapshot_dir / snapshot_name
        np.savez_compressed(
            snapshot_path,
            **snapshot,
            excess_mass=snapshot["excess"],
            solver_axis_order=np.asarray(SOLVER_AXIS_ORDER),
        )
        volume_snapshots.append(
            {
                "step": step,
                "file": f"snapshots/{snapshot_name}",
                "sha256": sha256_file(snapshot_path),
                "bytes": snapshot_path.stat().st_size,
                "shape": [int(value) for value in snapshot["fill"].shape],
                "solver_axis_order": list(SOLVER_AXIS_ORDER),
            }
        )
        center = config.resolution_z // 2
        center_fill = snapshot["fill"][:, :, center]
        wet = center_fill > 0.01
        center_mass = snapshot["mass"][:, :, center]
        center_total = float(np.sum(center_mass, dtype=np.float64))
        x = np.arange(center_mass.shape[0], dtype=np.float64)[:, None]
        y = np.arange(center_mass.shape[1], dtype=np.float64)[None, :]
        total_mass_drift = float(
            np.sum(snapshot["mass"], dtype=np.float64)
            + np.sum(snapshot["excess"], dtype=np.float64)
            - initial_mass
        )
        metrics[str(step)] = {
            "volume_drift": float(np.sum(snapshot["fill"]) - initial_volume),
            "total_mass_drift": total_mass_drift,
            "relative_total_mass_drift": total_mass_drift / initial_mass,
            "queued_excess_mass": float(
                np.sum(snapshot["excess"], dtype=np.float64)
            ),
            "front_x": int(np.max(np.nonzero(wet)[0])),
            "wet_height": int(np.max(np.nonzero(wet)[1])),
            "center_of_mass_x": float(
                np.sum(x * center_mass, dtype=np.float64) / center_total
            ),
            "center_of_mass_y": float(
                np.sum(y * center_mass, dtype=np.float64) / center_total
            ),
            "right_wall_wet_cells": int(np.count_nonzero(wet[-2, :])),
            "interface_cells": int(np.count_nonzero((center_fill > 0.0) & (center_fill < 1.0))),
            "interface_flag_cells": int(
                np.count_nonzero(
                    snapshot["flags"][:, :, center] == 1
                )
            ),
            "geometric_interface_cells": int(
                np.count_nonzero(
                    (center_fill > config.interface_roundoff_tolerance)
                    & (
                        center_fill
                        < 1.0 - config.interface_roundoff_tolerance
                    )
                )
            ),
            **_wall_film_metrics(
                snapshot["fill"], periodic_depth=True
            ),
        }
    overview = "dambreak-overview.png"
    _render_overview(output_dir / overview, snapshots, legacy_config.profile)
    impact_audit = _impact_audit(
        metrics,
        interior_right_x=config.resolution_x - 2,
        max_active_speed=maximum_speed,
    )
    final_metrics = metrics[str(max(snapshots))]
    initial_metrics = metrics[str(min(snapshots))]
    manifest: dict[str, object] = {
        "scene": legacy_config.profile,
        "config": asdict(config),
        "lattice_viscosity": model.lattice_viscosity,
        "lattice_acceleration": model.lattice_acceleration,
        "completed": failure is None,
        "completed_step": completed_step,
        "failure": failure,
        "elapsed_seconds": elapsed,
        "steps_per_second": completed_step / elapsed,
        "maximum_speed": maximum_speed,
        "maximum_collided_speed": maximum_collided_speed,
        "maximum_projection_iterations": maximum_projection_iterations,
        "maximum_raw_divergence": maximum_raw_divergence,
        "maximum_projected_divergence": maximum_projected_divergence,
        "maximum_face_correction": maximum_face_correction,
        "maximum_sub_tolerance_interface_count": maximum_sliver_count,
        "maximum_endpoint_promoted_liquid_count": maximum_endpoint_promoted_count,
        "maximum_separation_repair_count": maximum_separation_repair_count,
        "maximum_endpoint_removed_gas_count": maximum_endpoint_removed_count,
        "maximum_endpoint_fallback_face_count": (
            maximum_endpoint_fallback_face_count
        ),
        "maximum_wetting_cell_count": maximum_wetting_cell_count,
        "maximum_parallel_wall_interface_cell_count": (
            maximum_parallel_wall_interface_count
        ),
        "total_endpoint_promoted_liquid_count": total_endpoint_promoted_count,
        "total_separation_repair_count": total_separation_repair_count,
        "total_endpoint_removed_gas_count": total_endpoint_removed_count,
        "total_endpoint_fallback_face_count": total_endpoint_fallback_face_count,
        "total_wetting_cell_count": total_wetting_cell_count,
        "total_parallel_wall_interface_cell_count": (
            total_parallel_wall_interface_count
        ),
        "total_endpoint_redistributed_volume": (
            total_endpoint_redistributed_volume
        ),
        "total_endpoint_redistributed_mass": total_endpoint_redistributed_mass,
        "impact_audit": impact_audit,
        "initial_column_audit": {
            "configured_left_offset_cells": config.column_offset_x,
            "initial_liquid_volume": initial_volume,
            "initial_left_wall_fill_volume": float(
                initial_metrics["left_wall_fill_volume"]
            ),
            "initial_left_wall_half_full_height": int(
                initial_metrics["left_wall_max_half_full_height"]
            ),
        },
        "wall_film_audit": {
            "final_bulk_reference_height": int(
                final_metrics["bulk_reference_height"]
            ),
            "final_wall_film_fill_volume_above_bulk": float(
                final_metrics["wall_film_fill_volume_above_bulk"]
            ),
            "final_left_wall_max_half_full_height": int(
                final_metrics["left_wall_max_half_full_height"]
            ),
            "final_right_wall_max_half_full_height": int(
                final_metrics["right_wall_max_half_full_height"]
            ),
        },
        "metrics": metrics,
        "frames": frames,
        "volume_snapshots": volume_snapshots,
        "overview": overview,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "link-wise",
            "gvof",
            "projected-gvof",
            "projected-geometric",
            "projected-momentum",
            "geometric",
        ),
        default="geometric",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution-x", type=int, default=96)
    parser.add_argument("--resolution-y", type=int, default=56)
    parser.add_argument("--resolution-z", type=int, default=1)
    parser.add_argument("--cell-size", type=float, default=0.1)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--column-end-x", type=int, default=24)
    parser.add_argument("--column-end-y", type=int, default=34)
    parser.add_argument("--interface-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--contact-angle", type=float)
    parser.add_argument("--column-offset-x", type=int, default=0)
    parser.add_argument(
        "--sample-steps",
        type=int,
        nargs="+",
        default=(0, 10, 25, 50, 100, 200),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = GeometricDambreakConfig(
        mode=args.mode,
        device=args.device,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        resolution_z=args.resolution_z,
        cell_size=args.cell_size,
        time_step=args.time_step,
        column_end_x=args.column_end_x,
        column_end_y=args.column_end_y,
        interface_roundoff_tolerance=args.interface_tolerance,
        contact_angle_degrees=args.contact_angle,
        column_offset_x=args.column_offset_x,
        sample_steps=tuple(args.sample_steps),
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(run_scene(config, args.output), indent=2))


if __name__ == "__main__":
    main()
