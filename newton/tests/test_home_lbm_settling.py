# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Ten Cate E1 free-settling benchmark for the HOME rigid coupling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.collision.pipeline import CollisionPipeline
from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidCoupling,
    HomeLbmRigidInterface,
    SphereWallLubricationConfig,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


TEN_CATE_E1_FLUID_DENSITY = 970.0
TEN_CATE_E1_SPHERE_DENSITY = 1120.0
TEN_CATE_E1_DYNAMIC_VISCOSITY = 0.373
TEN_CATE_E1_DIAMETER = 0.015
TEN_CATE_E1_GRAVITY = 9.81
TEN_CATE_E1_INFINITE_VELOCITY = 0.038
TEN_CATE_E1_REYNOLDS = 1.5
TEN_CATE_E1_EXPERIMENTAL_MAX_RATIO = 0.947


@dataclass(frozen=True)
class _SettlingScaling:
    diameter: int
    width: int
    height: int
    release_height: float
    density_ratio: float
    lattice_viscosity: float
    lattice_gravity: float
    infinite_velocity: float
    galileo_number: float


@dataclass(frozen=True)
class _SettlingResult:
    maximum_velocity_ratio: float
    plateau_velocity_ratio: float
    plateau_relative_std: float
    buoyancy_relative_error: float
    maximum_lateral_velocity_ratio: float
    maximum_density_variation: float
    travelled_diameters: float
    maximum_strong_iterations: int
    maximum_strong_residual: float
    first_transition_step: int
    maximum_transition_count: int
    pretransition_density_variation: float
    warmup_buoyancy_relative_std: float
    minimum_fluid_force_ratio: float
    maximum_fluid_force_ratio: float
    final_velocity_ratio: float
    minimum_gap_cells: float
    maximum_rebound_velocity_ratio: float
    contact_step_count: int
    maximum_wall_confined_interpolation_count: int


def _ten_cate_e1_scaling(
    diameter: int,
    lattice_viscosity: float,
) -> _SettlingScaling:
    if diameter < 4:
        raise ValueError("settling benchmark requires at least two cells per radius")
    if lattice_viscosity <= 0.0:
        raise ValueError("lattice_viscosity must be positive")

    density_ratio = TEN_CATE_E1_SPHERE_DENSITY / TEN_CATE_E1_FLUID_DENSITY
    physical_viscosity = TEN_CATE_E1_DYNAMIC_VISCOSITY / TEN_CATE_E1_FLUID_DENSITY
    galileo_number = math.sqrt(
        (density_ratio - 1.0)
        * TEN_CATE_E1_GRAVITY
        * TEN_CATE_E1_DIAMETER**3
    ) / physical_viscosity
    lattice_gravity = (
        galileo_number**2
        * lattice_viscosity**2
        / ((density_ratio - 1.0) * diameter**3)
    )
    infinite_velocity = TEN_CATE_E1_REYNOLDS * lattice_viscosity / diameter
    return _SettlingScaling(
        diameter=diameter,
        width=round(100.0 * diameter / 15.0),
        height=round(160.0 * diameter / 15.0),
        release_height=120.0 * diameter / 15.0,
        density_ratio=density_ratio,
        lattice_viscosity=lattice_viscosity,
        lattice_gravity=lattice_gravity,
        infinite_velocity=infinite_velocity,
        galileo_number=galileo_number,
    )


def _run_ten_cate_e1(
    diameter: int,
    lattice_viscosity: float,
    device: str,
    diffusion_times: float = 0.1,
    advection_times: float = 3.0,
    lubrication: SphereWallLubricationConfig | None = None,
) -> _SettlingResult:
    scaling = _ten_cate_e1_scaling(diameter, lattice_viscosity)
    model = HomeLbmModel(
        fluid_grid_res=(scaling.width, scaling.width, scaling.height),
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        reference_density=1.0,
        kinematic_viscosity=scaling.lattice_viscosity,
        body_acceleration=(0.0, 0.0, -scaling.lattice_gravity),
        periodic=(False, False, False),
        device=device,
    )
    fluid = HomeLbmDomain(model)
    fluid.create_state()
    acceleration_z = -scaling.lattice_gravity
    density_ratio_z = (1.0 + 1.5 * acceleration_z) / (
        1.0 - 1.5 * acceleration_z
    )
    centered_z = (
        np.arange(scaling.height, dtype=np.float64)
        + 0.5
        - 0.5 * scaling.height
    )
    density_factors = np.exp(np.log(density_ratio_z) * centered_z)
    release_density_factor = math.exp(
        math.log(density_ratio_z)
        * (scaling.release_height - 0.5 * scaling.height)
    )
    fluid.solver.initialize_hydrostatic_lattice(
        fluid.state,
        mean_rho=float(np.mean(density_factors) / release_density_factor),
    )

    radius = 0.5 * diameter
    center = (0.5 * scaling.width, 0.5 * scaling.width, scaling.release_height)
    builder = RigidModelBuilder(gravity=-scaling.lattice_gravity)
    body = builder.add_body(position=center, label="ten_cate_e1_sphere")
    builder.add_shape_sphere(
        body,
        radius=radius,
        cfg=ShapeConfig(
            density=scaling.density_ratio,
            has_shape_collision=False,
        ),
    )
    rigid_model = builder.finalize(device=device)
    rigid = RigidDomain(
        rigid_model,
        solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
    )
    rigid.create_state()
    rigid.state.body_qd.zero_()

    static_interface = HomeLbmRigidInterface(fluid, rigid)
    warmup_steps = math.ceil(
        diffusion_times * diameter**2 / scaling.lattice_viscosity
    )
    warmup_force_samples: list[float] = []
    warmup_sample_start = max(0, warmup_steps - max(20, warmup_steps // 10))
    for warmup_step in range(warmup_steps):
        fluid.step(model.time_step)
        if warmup_step >= warmup_sample_start:
            static_interface.collect_wrench(rigid.state.body_q)
            warmup_force_samples.append(
                float(static_interface.wrench.force_lattice.numpy()[body, 2])
            )

    hydro_force = float(np.mean(warmup_force_samples))
    expected_buoyancy = 4.0 * math.pi * radius**3 * scaling.lattice_gravity / 3.0
    buoyancy_relative_error = abs(hydro_force - expected_buoyancy) / expected_buoyancy
    warmup_buoyancy_relative_std = float(
        np.std(warmup_force_samples) / expected_buoyancy
    )

    coupling = HomeLbmRigidCoupling(
        fluid,
        rigid,
        rigid_substeps=2,
        strong_coupling_max_iterations=20,
        strong_coupling_tolerance=2.0e-6,
        strong_coupling_relaxation=0.8,
        lubrication=lubrication,
    )
    contacts = CollisionPipeline.collide_rigid(rigid)
    settling_steps = math.ceil(
        advection_times * diameter / scaling.infinite_velocity
    )
    sample_interval = max(1, settling_steps // 300)
    velocities: list[float] = []
    lateral_velocities: list[float] = []
    positions: list[float] = []
    maximum_density_variation = 0.0
    maximum_strong_iterations = 0
    maximum_strong_residual = 0.0
    first_transition_step = -1
    maximum_transition_count = 0
    pretransition_density_variation = 0.0
    fluid_force_ratios: list[float] = []
    minimum_gap_cells = float("inf")
    maximum_rebound_velocity = 0.0
    contact_step_count = 0
    maximum_wall_confined_interpolation_count = 0
    for step in range(settling_steps):
        diagnostics = coupling.step(contacts=contacts)
        transition_count = (
            diagnostics.transitions.fresh_cell_count
            + diagnostics.transitions.dead_cell_count
        )
        maximum_transition_count = max(maximum_transition_count, transition_count)
        if transition_count and first_transition_step < 0:
            first_transition_step = step
        maximum_strong_iterations = max(
            maximum_strong_iterations,
            diagnostics.strong_coupling_iterations,
        )
        maximum_strong_residual = max(
            maximum_strong_residual,
            diagnostics.strong_coupling_residual,
        )
        maximum_wall_confined_interpolation_count = max(
            maximum_wall_confined_interpolation_count,
            diagnostics.fluid.wall_confined_interpolation_count,
        )
        maximum_density_variation = max(
            maximum_density_variation,
            abs(diagnostics.fluid.min_density - 1.0),
            abs(diagnostics.fluid.max_density - 1.0),
        )
        if first_transition_step < 0:
            pretransition_density_variation = maximum_density_variation
        fluid_force_z = float(coupling.interface.wrench.force_lattice.numpy()[body, 2])
        fluid_force_ratios.append(fluid_force_z / expected_buoyancy)
        current_velocity = rigid.state.body_qd.numpy()[body, :3].astype(np.float64)
        current_position = float(rigid.state.body_q.numpy()[body, 2])
        minimum_gap_cells = min(minimum_gap_cells, current_position - radius)
        maximum_rebound_velocity = max(maximum_rebound_velocity, float(current_velocity[2]))
        if coupling.lubrication is not None:
            contact_step_count += int(
                coupling.lubrication.contact_count.numpy()[body] > 0
            )
        if step % sample_interval == 0 or step + 1 == settling_steps:
            velocities.append(-float(current_velocity[2]))
            lateral_velocities.append(float(np.linalg.norm(current_velocity[:2])))
            positions.append(current_position)

    velocities_array = np.asarray(velocities)
    positions_array = np.asarray(positions)
    travelled = (center[2] - positions_array) / diameter
    plateau_start = max(0, len(velocities_array) * 2 // 3)
    plateau = velocities_array[plateau_start:]
    plateau_mean = float(np.mean(plateau))
    plateau_std = float(np.std(plateau))
    plateau_relative_std = (
        plateau_std / plateau_mean
        if abs(plateau_mean) > 1.0e-12
        else (0.0 if plateau_std <= 1.0e-12 else float("inf"))
    )
    return _SettlingResult(
        maximum_velocity_ratio=float(np.max(velocities_array) / scaling.infinite_velocity),
        plateau_velocity_ratio=plateau_mean / scaling.infinite_velocity,
        plateau_relative_std=plateau_relative_std,
        buoyancy_relative_error=buoyancy_relative_error,
        maximum_lateral_velocity_ratio=max(lateral_velocities) / scaling.infinite_velocity,
        maximum_density_variation=maximum_density_variation,
        travelled_diameters=float(travelled[-1]),
        maximum_strong_iterations=maximum_strong_iterations,
        maximum_strong_residual=maximum_strong_residual,
        first_transition_step=first_transition_step,
        maximum_transition_count=maximum_transition_count,
        pretransition_density_variation=pretransition_density_variation,
        warmup_buoyancy_relative_std=warmup_buoyancy_relative_std,
        minimum_fluid_force_ratio=min(fluid_force_ratios),
        maximum_fluid_force_ratio=max(fluid_force_ratios),
        final_velocity_ratio=float(velocities_array[-1] / scaling.infinite_velocity),
        minimum_gap_cells=minimum_gap_cells,
        maximum_rebound_velocity_ratio=(
            maximum_rebound_velocity / scaling.infinite_velocity
        ),
        contact_step_count=contact_step_count,
        maximum_wall_confined_interpolation_count=(
            maximum_wall_confined_interpolation_count
        ),
    )


class TestHomeLbmSettling(unittest.TestCase):
    def test_ten_cate_e1_dimensionless_mapping(self) -> None:
        scaling = _ten_cate_e1_scaling(diameter=6, lattice_viscosity=0.012)
        recovered_reynolds = (
            scaling.infinite_velocity * scaling.diameter / scaling.lattice_viscosity
        )
        recovered_galileo = math.sqrt(
            (scaling.density_ratio - 1.0)
            * scaling.lattice_gravity
            * scaling.diameter**3
        ) / scaling.lattice_viscosity

        self.assertAlmostEqual(recovered_reynolds, TEN_CATE_E1_REYNOLDS, places=14)
        self.assertAlmostEqual(recovered_galileo, scaling.galileo_number, places=13)
        self.assertEqual((scaling.width, scaling.height), (40, 64))
        self.assertEqual(scaling.release_height, 48.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_ten_cate_e1_settling_converges_in_space_and_time(self) -> None:
        coarse = _run_ten_cate_e1(8, 0.012, "cuda:0")
        intermediate = _run_ten_cate_e1(12, 0.012, "cuda:0")
        fine = _run_ten_cate_e1(16, 0.012, "cuda:0")
        time_refined = _run_ten_cate_e1(8, 0.009, "cuda:0")

        for result in (coarse, intermediate, fine, time_refined):
            self.assertLess(result.maximum_lateral_velocity_ratio, 0.002)
            self.assertLess(result.maximum_density_variation, 0.015)
            self.assertGreater(result.travelled_diameters, 2.0)
            self.assertLess(result.plateau_relative_std, 0.03)
            self.assertLess(result.buoyancy_relative_error, 0.06)
            self.assertLess(result.warmup_buoyancy_relative_std, 0.003)
            self.assertGreater(result.first_transition_step, 0)
            self.assertGreater(result.maximum_transition_count, 0)
            self.assertLessEqual(result.maximum_strong_iterations, 4)
            self.assertLessEqual(result.maximum_strong_residual, 2.0e-6)

        coarse_error = abs(
            coarse.maximum_velocity_ratio - TEN_CATE_E1_EXPERIMENTAL_MAX_RATIO
        )
        fine_error = abs(
            fine.maximum_velocity_ratio - TEN_CATE_E1_EXPERIMENTAL_MAX_RATIO
        )
        # Curved cut-link phase errors need not decrease at every intermediate grid.
        # Require the fine-grid error envelope to improve without claiming false order.
        self.assertLess(fine_error, coarse_error)
        self.assertLess(fine_error, 0.03)
        self.assertLess(
            abs(
                intermediate.maximum_velocity_ratio
                - TEN_CATE_E1_EXPERIMENTAL_MAX_RATIO
            ),
            0.12,
        )
        self.assertLess(
            abs(time_refined.maximum_velocity_ratio - fine.maximum_velocity_ratio),
            0.2,
        )
        self.assertLess(
            abs(time_refined.maximum_velocity_ratio - coarse.maximum_velocity_ratio),
            0.01,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_ten_cate_e1_bottom_approach_uses_lubrication_and_stops(self) -> None:
        result = _run_ten_cate_e1(
            8,
            0.012,
            "cuda:0",
            advection_times=16.0,
            lubrication=SphereWallLubricationConfig(
                cutoff_cells=1.0,
                contact_gap_cells=0.5,
            ),
        )

        self.assertGreater(result.contact_step_count, 0, result)
        self.assertGreaterEqual(result.minimum_gap_cells, 0.5 - 2.0e-5, result)
        self.assertLessEqual(result.minimum_gap_cells, 0.5 + 2.0e-4, result)
        self.assertLess(result.maximum_rebound_velocity_ratio, 0.01, result)
        self.assertLess(abs(result.final_velocity_ratio), 0.01, result)
        self.assertLess(result.maximum_lateral_velocity_ratio, 0.002, result)
        self.assertLess(result.maximum_density_variation, 0.03, result)
        self.assertLessEqual(result.maximum_strong_residual, 2.0e-6, result)


if __name__ == "__main__":
    unittest.main()
