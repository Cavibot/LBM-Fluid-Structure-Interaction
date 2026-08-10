# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run and render a HOME-FSL gravity-column benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
)
from wanphys.examples.lbm.home_rebuild.volume_artifacts import save_volume_snapshot


@dataclass(frozen=True)
class GravityColumnConfig:
    profile: str = "r3.6-collapse-preview"
    resolution_x: int = 160
    resolution_y: int = 96
    resolution_z: int = 8
    periodic_depth: bool = False
    column_end_x: int = 40
    column_end_y: int = 60
    column_offset_x: int = 0
    physical_viscosity: float = 2.0e-4
    physical_gravity_y: float = -2.0e-5
    gas_density: float = 1.0
    sample_steps: tuple[int, ...] = (0, 50, 100, 200, 400, 800, 1200, 2000, 3000)
    save_volume_snapshots: bool = False
    device: str = "cuda:0"


def _column_fill(config: GravityColumnConfig, walls: FslWallMask) -> np.ndarray:
    fill = np.zeros(walls.res, dtype=np.float32)
    depth = slice(None) if config.periodic_depth else slice(1, -1)
    if config.column_offset_x < 0:
        raise ValueError("column_offset_x must be nonnegative")
    if config.column_end_y < 1 or config.column_end_y + 1 >= walls.res[1] - 1:
        raise ValueError("column height must leave one interface row inside the tank")
    if config.column_offset_x == 0:
        if config.column_end_x < 1 or config.column_end_x + 1 >= walls.res[0] - 1:
            raise ValueError("column width must leave one interface column inside the tank")
        fill[1 : config.column_end_x + 1, 1 : config.column_end_y + 1, depth] = 1.0
        fill[config.column_end_x + 1, 1 : config.column_end_y + 2, depth] = 0.5
        fill[1 : config.column_end_x + 2, config.column_end_y + 1, depth] = 0.5
    else:
        left_interface = 1 + config.column_offset_x
        first_full = left_interface + 1
        last_full = config.column_end_x + config.column_offset_x + 1
        right_interface = last_full + 1
        if right_interface >= walls.res[0] - 1:
            raise ValueError("offset column must remain inside the tank")

        # Two quarter-filled side interfaces preserve the legacy column volume
        # while inserting a topology-safe gas gap next to the left wall.
        x_fraction = np.zeros(walls.res[0], dtype=np.float32)
        x_fraction[left_interface] = 0.25
        x_fraction[first_full : last_full + 1] = 1.0
        x_fraction[right_interface] = 0.25
        y_fraction = np.zeros(walls.res[1], dtype=np.float32)
        y_fraction[1 : config.column_end_y + 1] = 1.0
        y_fraction[config.column_end_y + 1] = 0.5
        cross_section = np.minimum(x_fraction[:, None], y_fraction[None, :])
        fill[:, :, depth] = cross_section[:, :, None]
    fill[walls.host] = 0.0
    return fill


def _snapshot(fluid: HomeCoreState, fsl: FslState) -> dict[str, np.ndarray]:
    moments = fluid.moments.numpy().reshape(10, -1).T.reshape((*fluid.res, 10))
    return {
        "moments": moments.astype(np.float64),
        "mass": fsl.mass.numpy().astype(np.float64),
        "fill": fsl.fill_level.numpy().astype(np.float64),
        "excess": fsl.excess_mass.numpy().astype(np.float64),
        "flags": fsl.flags.numpy(),
    }


def _render_frame(
    path: Path, step: int, snapshot: dict[str, np.ndarray], profile: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center_z = snapshot["fill"].shape[2] // 2
    fill = snapshot["fill"][:, :, center_z]
    flags = snapshot["flags"][:, :, center_z]
    moments = snapshot["moments"][:, :, center_z]
    active = flags != int(FslCellFlag.GAS)
    rho = moments[..., 0]
    speed = np.zeros(fill.shape, dtype=np.float64)
    speed[active] = np.linalg.norm(moments[..., 1:3][active] / rho[active, None], axis=-1)
    fig, (fill_axis, speed_axis) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=180)
    image = fill_axis.imshow(
        fill.T, origin="lower", vmin=0.0, vmax=1.0, cmap="Blues",
        interpolation="bilinear", aspect="equal"
    )
    fill_axis.contour(fill.T, levels=[0.01, 0.5, 0.99], colors="#075985", linewidths=0.55)
    fill_axis.plot([0.5, fill.shape[0] - 1.5], [0.5, 0.5], color="#111827", linewidth=2.2)
    fill_axis.plot([0.5, 0.5], [0.5, fill.shape[1] - 1.5], color="#111827", linewidth=2.2)
    fill_axis.plot(
        [fill.shape[0] - 1.5, fill.shape[0] - 1.5],
        [0.5, fill.shape[1] - 1.5], color="#111827", linewidth=2.2
    )
    fill_axis.plot(
        [0.5, fill.shape[0] - 1.5],
        [fill.shape[1] - 1.5, fill.shape[1] - 1.5],
        color="#111827", linewidth=2.2,
    )
    fill_axis.set_title("Liquid fill in tank cross-section")
    fill_axis.set_xlabel("x lattice cell")
    fill_axis.set_ylabel("y lattice cell")
    fig.colorbar(image, ax=fill_axis, label="fill fraction")
    speed_image = speed_axis.imshow(
        np.where(active, speed, np.nan).T, origin="lower", cmap="magma",
        interpolation="bilinear", aspect="equal", vmin=0.0
    )
    speed_axis.set_title("Active lattice speed")
    speed_axis.set_xlabel("x lattice cell")
    speed_axis.set_ylabel("y lattice cell")
    fig.colorbar(speed_image, ax=speed_axis, label="|u|")
    fig.suptitle(f"{profile} gravity column, step {step}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_overview(
    path: Path, snapshots: dict[int, dict[str, np.ndarray]], profile: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 9.0), dpi=180)
    for axis, (step, snapshot) in zip(
        axes.ravel(), snapshots.items(), strict=False
    ):
        center_z = snapshot["fill"].shape[2] // 2
        fill = snapshot["fill"][:, :, center_z]
        axis.imshow(
            fill.T, origin="lower", vmin=0.0, vmax=1.0, cmap="Blues",
            interpolation="bilinear", aspect="equal"
        )
        axis.contour(fill.T, levels=[0.01, 0.5], colors="#075985", linewidths=0.45)
        axis.plot([0.5, fill.shape[0] - 1.5], [0.5, 0.5], color="#111827", linewidth=1.5)
        axis.plot([0.5, 0.5], [0.5, fill.shape[1] - 1.5], color="#111827", linewidth=1.5)
        axis.plot(
            [fill.shape[0] - 1.5, fill.shape[0] - 1.5],
            [0.5, fill.shape[1] - 1.5], color="#111827", linewidth=1.5
        )
        axis.plot(
            [0.5, fill.shape[0] - 1.5],
            [fill.shape[1] - 1.5, fill.shape[1] - 1.5],
            color="#111827", linewidth=1.5,
        )
        axis.set_title(f"step {step}")
        axis.set_xlim(0, fill.shape[0] - 1)
        axis.set_ylim(0, fill.shape[1] - 1)
    fig.suptitle(f"{profile} gravity-column evolution", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _trapped_gas_metrics(
    flags: np.ndarray, solid: np.ndarray
) -> tuple[int, int]:
    gas = (flags == int(FslCellFlag.GAS)) & ~solid
    exterior = np.zeros(gas.shape, dtype=bool)
    stack = [(x, gas.shape[1] - 2) for x in range(1, gas.shape[0] - 1) if gas[x, -2]]
    while stack:
        x, y = stack.pop()
        if exterior[x, y] or not gas[x, y]:
            continue
        exterior[x, y] = True
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < gas.shape[0] and 0 <= ny < gas.shape[1]:
                if gas[nx, ny] and not exterior[nx, ny]:
                    stack.append((nx, ny))
    trapped = gas & ~exterior
    visited = np.zeros(gas.shape, dtype=bool)
    components = 0
    for seed in zip(*np.nonzero(trapped), strict=True):
        if visited[seed]:
            continue
        components += 1
        pending = [seed]
        while pending:
            x, y = pending.pop()
            if visited[x, y] or not trapped[x, y]:
                continue
            visited[x, y] = True
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < gas.shape[0] and 0 <= ny < gas.shape[1]:
                    if trapped[nx, ny] and not visited[nx, ny]:
                        pending.append((nx, ny))
    return components, int(np.count_nonzero(trapped))


def _wall_film_metrics(
    fill: np.ndarray, *, periodic_depth: bool = False
) -> dict[str, float | int]:
    depth = slice(None) if periodic_depth else slice(1, -1)
    core_depth = slice(None) if periodic_depth or fill.shape[2] <= 4 else slice(2, -2)
    core = fill[2:-2, 1:-1, core_depth]
    row_fill = np.sum(core, axis=(0, 2), dtype=np.float64)
    row_area = core.shape[0] * core.shape[2]
    bulk_rows = np.nonzero(row_fill > 0.05 * row_area)[0]
    bulk_height = int(np.max(bulk_rows)) + 1 if bulk_rows.size else 0

    result: dict[str, float | int] = {}
    wall_slabs = [
        ("left", fill[1, 1:-1, depth]),
        ("right", fill[-2, 1:-1, depth]),
    ]
    if not periodic_depth:
        wall_slabs.extend(
            (
                ("front", fill[1:-1, 1:-1, 1].T),
                ("back", fill[1:-1, 1:-1, -2].T),
            )
        )
    for name, slab in wall_slabs:
        wet = slab > 0.01
        half = slab >= 0.5
        wet_rows = np.nonzero(np.any(wet, axis=1))[0]
        half_rows = np.nonzero(np.any(half, axis=1))[0]
        result[f"{name}_wall_wet_cells_3d"] = int(np.count_nonzero(wet))
        result[f"{name}_wall_half_full_cells_3d"] = int(np.count_nonzero(half))
        result[f"{name}_wall_max_wet_height"] = (
            int(np.max(wet_rows)) + 1 if wet_rows.size else 0
        )
        result[f"{name}_wall_max_half_full_height"] = (
            int(np.max(half_rows)) + 1 if half_rows.size else 0
        )
        result[f"{name}_wall_fill_volume"] = float(
            np.sum(slab, dtype=np.float64)
        )
        result[f"{name}_wall_fill_volume_above_bulk"] = float(
            np.sum(slab[bulk_height:], dtype=np.float64)
        )

    wall_band = np.zeros(fill.shape, dtype=bool)
    wall_band[1, 1:-1, depth] = True
    wall_band[-2, 1:-1, depth] = True
    if not periodic_depth:
        wall_band[1:-1, 1:-1, 1] = True
        wall_band[1:-1, 1:-1, -2] = True
    above_bulk = np.zeros(fill.shape, dtype=bool)
    above_bulk[:, bulk_height + 1 :, :] = True
    film = wall_band & above_bulk
    result["bulk_reference_height"] = bulk_height
    result["wall_film_cells_above_bulk"] = int(np.count_nonzero(film & (fill > 0.01)))
    result["wall_film_half_full_cells_above_bulk"] = int(
        np.count_nonzero(film & (fill >= 0.5))
    )
    result["wall_film_fill_volume_above_bulk"] = float(
        np.sum(fill[film], dtype=np.float64)
    )
    return result


def _impact_audit(
    metrics: dict[str, dict[str, float | int]],
    *,
    interior_right_x: int,
    max_active_speed: float,
) -> dict[str, float | int | bool | None]:
    ordered = [(int(step), values) for step, values in metrics.items()]
    contacts = [(step, values) for step, values in ordered if values["right_wall_wet_cells"] > 0]
    first_contact_step = contacts[0][0] if contacts else None
    post_contact = contacts if contacts else []
    if post_contact:
        peak_step, peak_values = max(
            post_contact, key=lambda item: float(item[1]["center_of_mass_x"])
        )
        after_peak = [item for item in post_contact if item[0] >= peak_step]
        minimum_step, minimum_values = min(
            after_peak, key=lambda item: float(item[1]["center_of_mass_x"])
        )
        reflected_displacement = float(peak_values["center_of_mass_x"]) - float(
            minimum_values["center_of_mass_x"]
        )
    else:
        peak_step = minimum_step = None
        reflected_displacement = 0.0
    max_mass_drift = max(
        abs(float(values["relative_total_mass_drift"])) for _, values in ordered
    )
    no_solid_penetration = all(
        int(values["front_x"]) <= interior_right_x for _, values in ordered
    )
    return {
        "first_sampled_wall_contact_step": first_contact_step,
        "peak_forward_center_of_mass_step": peak_step,
        "post_impact_minimum_center_of_mass_step": minimum_step,
        "reflected_center_of_mass_displacement": reflected_displacement,
        "maximum_right_wall_wet_cells": max(
            int(values["right_wall_wet_cells"]) for _, values in ordered
        ),
        "maximum_absolute_mass_drift": max_mass_drift,
        "no_solid_penetration": no_solid_penetration,
        "stable_low_mach": max_active_speed <= 0.2,
        "reflection_observed": reflected_displacement >= 5.0,
        "mass_drift_within_1e-3": max_mass_drift <= 1.0e-3,
    }


def run_scene(config: GravityColumnConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    if config.save_volume_snapshots:
        snapshot_dir.mkdir(exist_ok=True)
    shape = (config.resolution_x, config.resolution_y, config.resolution_z)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=config.device,
        kinematic_viscosity=config.physical_viscosity,
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
    )
    walls = (
        FslWallMask.periodic_depth_channel(model)
        if config.periodic_depth
        else FslWallMask.closed_box(model)
    )
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
    stepper = FslWallDynamicTopologyStepper(
        model, walls, gas_density=config.gas_density
    )
    snapshots = {0: _snapshot(fluid_a, fsl_a)}
    samples = set(config.sample_steps[1:])
    transition_totals = {
        "interface_to_liquid": 0,
        "interface_to_gas": 0,
        "gas_to_interface": 0,
        "liquid_to_interface": 0,
    }
    max_speed = 0.0
    started = time.perf_counter()
    for step in range(1, config.sample_steps[-1] + 1):
        try:
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
        except FloatingPointError as error:
            raise FloatingPointError(
                f"{config.profile} failed at step {step}: {error}"
            ) from error
        topology = diagnostics.topology
        transition_totals["interface_to_liquid"] += topology.interface_to_liquid_count
        transition_totals["interface_to_gas"] += topology.interface_to_gas_count
        transition_totals["gas_to_interface"] += topology.gas_to_interface_count
        transition_totals["liquid_to_interface"] += topology.liquid_to_interface_count
        max_speed = max(max_speed, diagnostics.fluid.max_speed)
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        if step in samples:
            snapshots[step] = _snapshot(fluid_a, fsl_a)
    wp.synchronize_device(config.device)
    elapsed = time.perf_counter() - started
    initial_total = float(np.sum(snapshots[0]["mass"]) + np.sum(snapshots[0]["excess"]))
    metrics: dict[str, dict[str, float | int]] = {}
    frames: list[str] = []
    volume_snapshots: list[dict[str, object]] = []
    for step, snapshot in snapshots.items():
        frame = f"frames/step-{step:06d}.png"
        _render_frame(output_dir / frame, step, snapshot, config.profile)
        frames.append(frame)
        if config.save_volume_snapshots:
            snapshot_name = f"step-{step:06d}.npz"
            artifact = save_volume_snapshot(
                snapshot_dir / snapshot_name,
                fill=snapshot["fill"],
                mass=snapshot["mass"],
                flags=snapshot["flags"],
                moments=snapshot["moments"],
                excess_mass=snapshot["excess"],
            )
            artifact["step"] = step
            artifact["file"] = f"snapshots/{snapshot_name}"
            volume_snapshots.append(artifact)
        total = float(np.sum(snapshot["mass"]) + np.sum(snapshot["excess"]))
        center_z = snapshot["fill"].shape[2] // 2
        wet = snapshot["fill"][:, :, center_z] > 0.01
        center_mass = snapshot["mass"][:, :, center_z]
        x = np.arange(center_mass.shape[0], dtype=np.float64)[:, None]
        y = np.arange(center_mass.shape[1], dtype=np.float64)[None, :]
        center_total = float(np.sum(center_mass, dtype=np.float64))
        center_flags = snapshot["flags"][:, :, center_z]
        center_rho = snapshot["moments"][:, :, center_z, 0]
        center_active = center_flags != int(FslCellFlag.GAS)
        trapped_components, trapped_cells = _trapped_gas_metrics(
            center_flags, walls.host[:, :, center_z]
        )
        metrics[str(step)] = {
            "relative_total_mass_drift": (total - initial_total) / initial_total,
            "front_x": int(np.max(np.nonzero(wet)[0])),
            "wet_height": int(np.max(np.nonzero(wet)[1])),
            "center_of_mass_x": float(np.sum(x * center_mass) / center_total),
            "center_of_mass_y": float(np.sum(y * center_mass) / center_total),
            "min_active_density": float(np.min(center_rho[center_active])),
            "max_active_density": float(np.max(center_rho[center_active])),
            "queued_excess_mass": float(np.sum(snapshot["excess"], dtype=np.float64)),
            "max_abs_queued_excess": float(np.max(np.abs(snapshot["excess"]))),
            "right_wall_wet_cells": int(np.count_nonzero(wet[-2, :])),
            "trapped_gas_components": trapped_components,
            "trapped_gas_cells": trapped_cells,
            "liquid_cells": int(np.count_nonzero(snapshot["flags"] == int(FslCellFlag.LIQUID))),
            "interface_cells": int(np.count_nonzero(snapshot["flags"] == int(FslCellFlag.INTERFACE))),
            **_wall_film_metrics(
                snapshot["fill"], periodic_depth=config.periodic_depth
            ),
        }
    overview = "gravity-column-overview.png"
    _render_overview(output_dir / overview, snapshots, config.profile)
    impact_audit = _impact_audit(
        metrics,
        interior_right_x=config.resolution_x - 2,
        max_active_speed=max_speed,
    )
    initial_metrics = metrics[str(min(snapshots))]
    final_metrics = metrics[str(max(snapshots))]
    wall_film_audit = {
        "final_left_wall_fill_fraction_of_initial": float(
            final_metrics["left_wall_fill_volume"]
        )
        / max(float(initial_metrics["left_wall_fill_volume"]), 1.0),
        "final_right_wall_fill_volume": float(
            final_metrics["right_wall_fill_volume"]
        ),
        "final_left_wall_max_half_full_height": int(
            final_metrics["left_wall_max_half_full_height"]
        ),
        "final_right_wall_max_half_full_height": int(
            final_metrics["right_wall_max_half_full_height"]
        ),
        "peak_right_wall_fill_volume": max(
            float(values["right_wall_fill_volume"]) for values in metrics.values()
        ),
        "peak_right_wall_max_half_full_height": max(
            int(values["right_wall_max_half_full_height"])
            for values in metrics.values()
        ),
        "final_bulk_reference_height": int(
            final_metrics["bulk_reference_height"]
        ),
        "final_wall_film_fill_volume_above_bulk": float(
            final_metrics["wall_film_fill_volume_above_bulk"]
        ),
        "final_wall_film_half_full_cells_above_bulk": int(
            final_metrics["wall_film_half_full_cells_above_bulk"]
        ),
    }
    manifest: dict[str, object] = {
        "scene": config.profile,
        "config": asdict(config),
        "lattice_acceleration": model.lattice_acceleration,
        "lattice_viscosity": model.lattice_viscosity,
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": config.device,
        },
        "elapsed_seconds": elapsed,
        "max_active_speed": max_speed,
        "transition_totals": transition_totals,
        "metrics": metrics,
        "impact_audit": impact_audit,
        "wall_film_audit": wall_film_audit,
        "frames": frames,
        "volume_snapshots": volume_snapshots,
        "overview": overview,
        "limitations": [
            "algorithm benchmark, not final Dam-break acceptance",
            "halfway bounce-back walls without cut-link geometry",
            "no PLIC, curvature, surface tension, or FSI",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run_scene(GravityColumnConfig(device=args.device), args.output), indent=2))


if __name__ == "__main__":
    main()
