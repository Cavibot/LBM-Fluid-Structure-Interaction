# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Scan HOME-Free gravity-column stability while changing only viscosity."""

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
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)


@dataclass(frozen=True)
class StabilityScanConfig:
    resolution_x: int = 64
    resolution_y: int = 56
    resolution_z: int = 16
    column_end_x: int = 16
    column_end_y: int = 35
    cell_size: float = 0.1
    time_step: float = 1.0
    lattice_gravity_y: float = -5.0e-5
    lattice_viscosities: tuple[float, ...] = (0.02, 0.01, 0.005)
    steps: int = 2000
    sample_every: int = 100
    gas_density: float = 1.0
    max_lattice_speed: float = 0.2
    device: str = "cuda:0"

    def validate(self) -> None:
        if min(self.resolution_x, self.resolution_y, self.resolution_z) < 5:
            raise ValueError("all scan dimensions must be at least five")
        if self.steps < 1 or self.sample_every < 1:
            raise ValueError("steps and sample_every must be positive")
        if not self.lattice_viscosities or any(
            value <= 0.0 for value in self.lattice_viscosities
        ):
            raise ValueError("lattice viscosities must be positive")


def make_scan_model(config: StabilityScanConfig, lattice_viscosity: float) -> HomeCoreModel:
    """Convert an explicit lattice scan point to the model's physical units."""

    return HomeCoreModel(
        fluid_grid_res=(
            config.resolution_x,
            config.resolution_y,
            config.resolution_z,
        ),
        fluid_grid_cell_size=config.cell_size,
        time_step=config.time_step,
        kinematic_viscosity=(
            lattice_viscosity * config.cell_size**2 / config.time_step
        ),
        body_acceleration=(
            0.0,
            config.lattice_gravity_y * config.cell_size / config.time_step**2,
            0.0,
        ),
        max_lattice_speed=config.max_lattice_speed,
        device=config.device,
    )


def _represented_mass(state: FslState) -> float:
    return float(
        np.sum(state.mass.numpy(), dtype=np.float64)
        + np.sum(state.excess_mass.numpy(), dtype=np.float64)
    )


def run_stability_case(
    config: StabilityScanConfig, lattice_viscosity: float
) -> dict[str, object]:
    config.validate()
    model = make_scan_model(config, lattice_viscosity)
    walls = FslWallMask.closed_box(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    fill_config = GravityColumnConfig(
        profile="low-viscosity-scan",
        resolution_x=config.resolution_x,
        resolution_y=config.resolution_y,
        resolution_z=config.resolution_z,
        column_end_x=config.column_end_x,
        column_end_y=config.column_end_y,
        physical_viscosity=model.kinematic_viscosity,
        physical_gravity_y=model.body_acceleration[1],
        gas_density=config.gas_density,
        sample_steps=(0,),
        device=config.device,
    )
    fill = _column_fill(fill_config, walls)
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
    initial_mass = _represented_mass(fsl_a)
    maximum_speed = 0.0
    samples: list[dict[str, float | int]] = []
    failure: str | None = None
    completed_steps = 0
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        try:
            diagnostics = stepper.step(
                fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
            )
        except (FloatingPointError, RuntimeError) as error:
            failure = str(error)
            break
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
        completed_steps = step
        maximum_speed = max(maximum_speed, diagnostics.fluid.max_speed)
        if step % config.sample_every == 0 or step == config.steps:
            mass = _represented_mass(fsl_a)
            samples.append(
                {
                    "step": step,
                    "relative_mass_drift": (mass - initial_mass) / initial_mass,
                    "max_speed_so_far": maximum_speed,
                }
            )
    wp.synchronize_device(model._device)
    elapsed = time.perf_counter() - started
    final_mass = _represented_mass(fsl_a)
    relative_mass_drift = (final_mass - initial_mass) / initial_mass
    completed = failure is None and completed_steps == config.steps
    return {
        "lattice_viscosity": model.lattice_viscosity,
        "shear_omega": model.shear_omega,
        "lattice_gravity_y": model.lattice_acceleration[1],
        "completed": completed,
        "completed_steps": completed_steps,
        "failure": failure,
        "maximum_speed": maximum_speed,
        "relative_mass_drift": relative_mass_drift,
        "elapsed_seconds": elapsed,
        "steps_per_second": completed_steps / elapsed if elapsed > 0.0 else 0.0,
        "accepted": (
            completed
            and maximum_speed <= config.max_lattice_speed
            and abs(relative_mass_drift) <= 1.0e-3
        ),
        "samples": samples,
    }


def run_scan(config: StabilityScanConfig) -> dict[str, object]:
    return {
        "scene": "home-free-linkwise-low-viscosity-scan",
        "config": asdict(config),
        "cases": [
            run_stability_case(config, viscosity)
            for viscosity in config.lattice_viscosities
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument(
        "--lattice-viscosities", type=float, nargs="+", default=[0.02, 0.01, 0.005]
    )
    args = parser.parse_args()
    config = StabilityScanConfig(
        lattice_viscosities=tuple(args.lattice_viscosities),
        steps=args.steps,
        sample_every=args.sample_every,
        device=args.device,
    )
    result = run_scan(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
