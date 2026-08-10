# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Dynamic geometric-VOF Laplace acceptance for the HOME-FSL path."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.fsl_stepper import (
    HomeFreeFslResearchStepper,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.geometry import (
    HomeFreeInterfaceGeometry,
)


def _sampled_sphere_vof(
    shape: tuple[int, int, int],
    radius: float,
    *,
    samples_per_axis: int = 32,
    center: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return a sharp sphere/cell volume fraction with topology-safe quadrature."""

    sphere_center = (
        0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        if center is None
        else np.asarray(center, dtype=np.float64)
    )
    if sphere_center.shape != (3,) or not np.isfinite(sphere_center).all():
        raise ValueError("sphere center must contain three finite coordinates")
    coordinates = np.indices(shape).transpose(1, 2, 3, 0)
    delta = np.abs(coordinates - sphere_center)
    minimum_distance = np.linalg.norm(np.maximum(delta - 0.5, 0.0), axis=-1)
    maximum_distance = np.linalg.norm(delta + 0.5, axis=-1)
    inside = maximum_distance <= radius
    boundary = (minimum_distance < radius) & ~inside
    fill = np.zeros(shape, dtype=np.float32)
    fill[inside] = 1.0
    coordinate = (np.arange(samples_per_axis) + 0.5) / samples_per_axis - 0.5
    offsets = np.stack(
        np.meshgrid(coordinate, coordinate, coordinate, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    quadrature_floor = 1.0 / samples_per_axis**3
    for index in map(tuple, np.argwhere(boundary)):
        fraction = float(
            np.mean(
                np.sum((np.asarray(index) - sphere_center + offsets) ** 2, axis=1)
                <= radius * radius
            )
        )
        fill[index] = np.clip(
            fraction, quadrature_floor, 1.0 - quadrature_floor
        )
    return fill


class TestHomeFreeDynamicLaplace(unittest.TestCase):
    def _run_case(self, device: str) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
        shape = (17, 17, 17)
        radius = 5.25
        surface_tension = 0.001
        fill = _sampled_sphere_vof(shape, radius)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.5,
            periodic=(False, False, False),
            device=device,
        )
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        initial_density = 1.0 + 6.0 * surface_tension / radius
        HomeLbmSolver(model).initialize_uniform_lattice(
            fluid_in, rho=initial_density
        )
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        free_in.initialize_from_fill_level(fluid_in, fill)
        geometry = HomeFreeInterfaceGeometry(model)
        geometry_diagnostics = geometry.update(free_in)
        curvature_required = geometry.curvature_required.numpy().astype(bool)
        curvature_error = abs(
            float(np.mean(geometry.curvature.numpy()[curvature_required]))
            + 1.0 / radius
        )
        initial_volume = float(np.sum(fill, dtype=np.float64))
        analytic_volume = 4.0 * math.pi * radius**3 / 3.0
        initial_rho = fluid_in.moments.numpy().reshape(10, *shape)[0]
        initial_mass = float(np.sum(initial_rho * fill, dtype=np.float64))
        stepper = HomeFreeFslResearchStepper(
            model,
            advection_axes=(0, 1, 2),
            surface_tension=surface_tension,
            no_support_policy="only_missing",
        )
        peak_speed = 0.0
        maximum_fallback_links = 0
        for _ in range(20):
            diagnostics = stepper.step(
                fluid_in, free_in, fluid_out, free_out
            )
            moments = fluid_out.moments.numpy().reshape(10, -1)
            active = np.isin(free_out.flags.numpy(), (1, 2)).reshape(-1)
            speed = np.linalg.norm(
                moments[1:4, active].T / moments[0, active, None], axis=1
            )
            peak_speed = max(peak_speed, float(np.max(speed)))
            maximum_fallback_links = max(
                maximum_fallback_links, diagnostics.fallback_link_count
            )
            self.assertEqual(diagnostics.invalid_geometry_count, 0)
            self.assertEqual(diagnostics.invalid_stream_count, 0)
            self.assertEqual(diagnostics.invalid_collision_count, 0)
            fluid_in, fluid_out = fluid_out, fluid_in
            free_in, free_out = free_out, free_in

        final_fill = free_in.fill_level.numpy().copy()
        final_moments = fluid_in.moments.numpy().copy()
        final_rho = final_moments.reshape(10, *shape)[0]
        final_volume = float(np.sum(final_fill, dtype=np.float64))
        final_mass = float(np.sum(final_rho * final_fill, dtype=np.float64))
        metrics = (
            abs(initial_volume - analytic_volume) / analytic_volume,
            abs(final_volume - initial_volume) / initial_volume,
            abs(final_mass - initial_mass) / initial_mass,
            peak_speed,
            curvature_error,
            float(maximum_fallback_links),
            float(geometry_diagnostics.curvature_required_cell_count),
        )
        self.assertLess(metrics[0], 1.0e-4)
        self.assertLess(metrics[1], 1.5e-4)
        self.assertLess(metrics[2], 1.5e-4)
        self.assertLess(metrics[3], 1.5e-4)
        self.assertLess(metrics[4], 5.0e-3)
        self.assertEqual(maximum_fallback_links, 192)
        self.assertEqual(geometry_diagnostics.curvature_required_cell_count, 530)
        return final_fill, final_moments, metrics

    def test_cpu_dynamic_laplace_acceptance(self) -> None:
        self._run_case("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_dynamic_laplace_matches_cpu(self) -> None:
        expected_fill, expected_moments, expected_metrics = self._run_case("cpu")
        actual_fill, actual_moments, actual_metrics = self._run_case("cuda:0")
        np.testing.assert_allclose(actual_fill, expected_fill, rtol=0.0, atol=3.0e-6)
        np.testing.assert_allclose(
            actual_moments, expected_moments, rtol=3.0e-5, atol=3.0e-6
        )
        np.testing.assert_allclose(
            actual_metrics, expected_metrics, rtol=2.0e-3, atol=3.0e-7
        )


if __name__ == "__main__":
    unittest.main()
