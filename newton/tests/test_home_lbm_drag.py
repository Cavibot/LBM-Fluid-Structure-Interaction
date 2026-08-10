# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Low-Re periodic-sphere drag verification against Hasimoto's solution."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmRigidInterface,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder


@wp.kernel
def _sum_fluid_mass_momentum_x(
    moments: wp.array(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    totals: wp.array(dtype=float),
    stride: int,
    ny: int,
    nz: int,
):
    cell = wp.tid()
    i = cell // (ny * nz)
    remainder = cell - i * ny * nz
    j = remainder // nz
    k = remainder - j * nz
    if solid_phi[i, j, k] >= 0.0:
        wp.atomic_add(totals, 0, moments[cell])
        wp.atomic_add(totals, 1, moments[stride + cell])


@dataclass(frozen=True)
class _DragResult:
    relative_error: float
    relative_force_std: float
    relative_velocity_error: float
    reynolds_number: float


def _hasimoto_drag(
    radius: float,
    box_length: float,
    dynamic_viscosity: float,
    intrinsic_velocity: float,
) -> float:
    """Dilute simple-cubic drag through O(phi^2), Hasimoto/Sangani form."""

    solid_fraction = 4.0 * np.pi * radius**3 / (3.0 * box_length**3)
    denominator = (
        1.0
        - 1.7601 * solid_fraction ** (1.0 / 3.0)
        + solid_fraction
        - 1.5593 * solid_fraction**2
    )
    return (
        6.0
        * np.pi
        * dynamic_viscosity
        * radius
        * intrinsic_velocity
        * (1.0 - solid_fraction)
        / denominator
    )


def _run_constant_flux_drag(
    resolution: int,
    radius: float,
    target_reynolds: float,
    steps: int,
    sample_steps: int,
    device: str,
) -> _DragResult:
    viscosity = 0.2
    target_velocity = target_reynolds * viscosity / (2.0 * radius)
    expected_drag = _hasimoto_drag(
        radius,
        float(resolution),
        viscosity,
        target_velocity,
    )
    model = HomeLbmModel(
        fluid_grid_res=(resolution, resolution, resolution),
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        kinematic_viscosity=viscosity,
        body_acceleration=(expected_drag / resolution**3, 0.0, 0.0),
        periodic=(True, True, True),
        device=device,
    )
    fluid = HomeLbmDomain(model)
    fluid.create_state()

    builder = RigidModelBuilder(gravity=0.0)
    body = builder.add_body(
        position=(resolution / 2.0, resolution / 2.0, resolution / 2.0),
        label="fixed_drag_sphere",
    )
    builder.add_shape_sphere(body, radius=radius)
    rigid = RigidDomain(builder.finalize(device=device))
    rigid.create_state()
    interface = HomeLbmRigidInterface(fluid, rigid)
    fluid.solver.initialize_uniform_lattice(
        fluid.state,
        velocity=(target_velocity, 0.0, 0.0),
    )

    totals = wp.zeros(2, dtype=float, device=device)
    force_samples: list[float] = []
    velocity_samples: list[float] = []
    for step in range(steps):
        fluid.step(model.time_step)
        interface.collect_wrench(rigid.state.body_q)
        totals.zero_()
        wp.launch(
            _sum_fluid_mass_momentum_x,
            dim=fluid.state.cell_count,
            inputs=[
                fluid.state.moments,
                fluid.state.solid_phi,
                totals,
                fluid.state.cell_count,
                fluid.state.res[1],
                fluid.state.res[2],
            ],
            device=device,
        )
        mass, momentum_x = totals.numpy()
        measured_velocity = float(momentum_x / mass)
        measured_force = float(interface.wrench.force_lattice.numpy()[body, 0])
        momentum_error = mass * target_velocity - momentum_x
        next_acceleration = (measured_force + 0.5 * momentum_error) / mass
        model.body_acceleration = (float(next_acceleration), 0.0, 0.0)
        if step >= steps - sample_steps:
            force_samples.append(measured_force)
            velocity_samples.append(measured_velocity)

    mean_force = float(np.mean(force_samples))
    mean_velocity = float(np.mean(velocity_samples))
    return _DragResult(
        relative_error=abs(mean_force - expected_drag) / expected_drag,
        relative_force_std=float(np.std(force_samples) / mean_force),
        relative_velocity_error=abs(mean_velocity - target_velocity) / target_velocity,
        reynolds_number=2.0 * radius * mean_velocity / viscosity,
    )


class TestHomeLbmDrag(unittest.TestCase):
    def test_hasimoto_reference_uses_intrinsic_velocity_and_porosity(self) -> None:
        drag = _hasimoto_drag(3.2, 24.0, 0.2, 7.0e-4)
        self.assertAlmostEqual(drag, 0.013240082924793, places=14)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_periodic_sphere_drag_converges_toward_hasimoto(self) -> None:
        target_reynolds = 0.0224
        coarse = _run_constant_flux_drag(
            resolution=24,
            radius=3.2,
            target_reynolds=target_reynolds,
            steps=600,
            sample_steps=100,
            device="cuda:0",
        )
        fine = _run_constant_flux_drag(
            resolution=36,
            radius=4.8,
            target_reynolds=target_reynolds,
            steps=1000,
            sample_steps=150,
            device="cuda:0",
        )

        for result in (coarse, fine):
            self.assertLess(result.relative_force_std, 5.0e-4)
            self.assertLess(result.relative_velocity_error, 5.0e-4)
            self.assertAlmostEqual(result.reynolds_number, target_reynolds, delta=2.0e-5)
        self.assertLess(coarse.relative_error, 0.12)
        self.assertLess(fine.relative_error, coarse.relative_error)
        self.assertLess(fine.relative_error, 0.08)


if __name__ == "__main__":
    unittest.main()
