# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit curvature-driven Laplace pressure on a sphere or Viewer dam break."""

from __future__ import annotations

import argparse
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
    FslCellFlag,
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicCapillaryWallStepper,
    PlicSurfaceTension,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    _maximum_lattice_speed,
    _simulation_cell_size,
    make_viewer_config,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    _column_fill,
)


def _sphere_fill(shape: tuple[int, int, int], radius: float) -> np.ndarray:
    center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    coordinates = np.indices(shape, dtype=np.float64)
    distance = np.sqrt(
        sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
    )


def _wall_droplet_fill(shape: tuple[int, int, int], radius: float) -> np.ndarray:
    center = np.asarray(
        (0.5 * (shape[0] - 1), 1.0, 0.5 * (shape[2] - 1)),
        dtype=np.float64,
    )
    coordinates = np.indices(shape, dtype=np.float64)
    distance = np.sqrt(
        sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
    )
    return np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0).astype(
        np.float32
    )
    return np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0).astype(
        np.float32
    )


def _shape_metrics(fill: np.ndarray, solid: np.ndarray) -> dict[str, float]:
    active_fill = np.where(solid, 0.0, fill).astype(np.float64)
    volume = float(np.sum(active_fill))
    coordinates = np.moveaxis(np.indices(fill.shape, dtype=np.float64), 0, -1)
    centroid = np.sum(coordinates * active_fill[..., None], axis=(0, 1, 2)) / volume
    displacement = coordinates - centroid
    flat_displacement = displacement.reshape(-1, 3)
    flat_fill = active_fill.reshape(-1)
    covariance = (
        (flat_displacement.T * flat_fill) @ flat_displacement / volume
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "volume": volume,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "shape_anisotropy": float(eigenvalues[-1] / eigenvalues[0] - 1.0),
    }


def run_audit(
    *,
    scenario: str,
    steps: int,
    lattice_surface_tension: float,
    device: str,
    contact_angle_degrees: float | None = None,
) -> dict[str, object]:
    if scenario not in ("sphere", "wall-droplet", "dambreak"):
        raise ValueError("scenario must be sphere, wall-droplet, or dambreak")
    if steps < 1:
        raise ValueError("steps must be positive")
    if lattice_surface_tension < 0.0:
        raise ValueError("lattice surface tension must be nonnegative")
    if scenario in ("sphere", "wall-droplet"):
        shape = (48, 48, 48)
        model = HomeCoreModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            device=device,
            kinematic_viscosity=1.0 / 6.0,
            max_lattice_speed=0.2,
        )
        gas_density = 1.0
    else:
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
        gas_density = config.gas_density
    walls = FslWallMask.closed_box(model)
    if scenario == "sphere":
        fill = _sphere_fill(shape, radius=10.0)
        surface_coordinate = 0.5 * (shape[1] - 1)
    elif scenario == "wall-droplet":
        if contact_angle_degrees is None:
            raise ValueError("wall-droplet requires a contact angle")
        fill = _wall_droplet_fill(shape, radius=10.0)
        surface_coordinate = 1.0
    else:
        fill = _column_fill(config, walls)
        surface_coordinate = float(config.column_end_y + 1)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    if scenario in ("sphere", "wall-droplet"):
        fill[walls.host] = 0.0
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[fill >= 1.0] = int(FslCellFlag.LIQUID)
        flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
        moments = np.zeros(shape + (10,), dtype=np.float32)
        moments[..., 0] = gas_density
        fluid_a.moments.assign(
            np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1))
        )
        fsl_a.fill_level.assign(fill)
        fsl_a.mass.assign(fill)
        fsl_a.flags.assign(flags)
    else:
        walls.initialize_hydrostatic(
            fluid_a,
            fsl_a,
            fill,
            gas_density=gas_density,
            gravity_axis=1,
            surface_coordinate=surface_coordinate,
        )
    fluid_b.copy_from(fluid_a)
    fsl_b.copy_from(fsl_a)
    baseline = FslWallDynamicTopologyStepper(
        model, walls, gas_density=gas_density
    )
    surface = PlicSurfaceTension(
        model,
        walls,
        ambient_gas_density=gas_density,
        surface_tension=model.scaling.surface_tension_to_physical(
            lattice_surface_tension
        ),
        contact_angle_degrees=contact_angle_degrees,
    )
    stepper = PlicCapillaryWallStepper(baseline, surface)
    initial_shape = _shape_metrics(fsl_a.fill_level.numpy(), walls.host)
    initial_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    maximum_speed = 0.0
    maximum_wall_skipped = 0
    maximum_wall_fitted = 0
    maximum_wetting_cells = 0
    minimum_gas_density = float("inf")
    maximum_gas_density = float("-inf")
    last_valid_step = 0
    failure: str | None = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
        except (FloatingPointError, RuntimeError) as error:
            failure = str(error)
            break
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        last_valid_step = step
        maximum_speed = max(
            maximum_speed, diagnostics.baseline.fluid.max_speed
        )
        capillary = diagnostics.surface_tension
        maximum_wall_skipped = max(
            maximum_wall_skipped,
            capillary.curvature.wall_contact_skipped_count,
        )
        maximum_wall_fitted = max(
            maximum_wall_fitted,
            capillary.curvature.wall_contact_fitted_count,
        )
        maximum_wetting_cells = max(
            maximum_wetting_cells,
            capillary.geometry.wetting_cell_count,
        )
        minimum_gas_density = min(
            minimum_gas_density, capillary.minimum_interface_gas_density
        )
        maximum_gas_density = max(
            maximum_gas_density, capillary.maximum_interface_gas_density
        )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - started
    final_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    final_shape = _shape_metrics(fsl_a.fill_level.numpy(), walls.host)
    return {
        "stage": (
            "B7-contact-angle-wetting"
            if contact_angle_degrees is not None
            else "B6-capillary-pressure"
        ),
        "scenario": scenario,
        "configuration": {
            "resolution": shape,
            "requested_steps": steps,
            "lattice_surface_tension": lattice_surface_tension,
            "physical_surface_tension": surface.surface_tension,
            "lattice_viscosity": model.lattice_viscosity,
            "lattice_acceleration": model.lattice_acceleration,
            "wall_contact_policy": (
                "configured-contact-angle"
                if contact_angle_degrees is not None
                else "ambient-pressure-until-B7"
            ),
            "contact_angle_degrees": contact_angle_degrees,
        },
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": device,
        },
        "last_valid_step": last_valid_step,
        "failure": failure,
        "maximum_active_speed": maximum_speed,
        "minimum_interface_gas_density": minimum_gas_density,
        "maximum_interface_gas_density": maximum_gas_density,
        "maximum_wall_contact_skipped_count": maximum_wall_skipped,
        "maximum_wall_contact_fitted_count": maximum_wall_fitted,
        "maximum_wetting_cell_count": maximum_wetting_cells,
        "relative_total_mass_drift": (final_mass - initial_mass) / initial_mass,
        "initial_shape": initial_shape,
        "final_shape": final_shape,
        "elapsed_seconds": elapsed,
        "ms_per_completed_step": (
            elapsed * 1000.0 / last_valid_step
            if last_valid_step > 0
            else float("inf")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("sphere", "wall-droplet", "dambreak"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lattice-surface-tension", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-angle-degrees", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(
        scenario=args.scenario,
        steps=args.steps,
        lattice_surface_tension=args.lattice_surface_tension,
        device=args.device,
        contact_angle_degrees=args.contact_angle_degrees,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
