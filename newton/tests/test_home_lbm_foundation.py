# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Foundation tests for the independent HOME-LBM implementation."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_OPPOSITE,
    D3Q27_WEIGHTS,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    LatticeScaling,
    equilibrium_convective_populations,
    extract_moments,
    periodic_home_convective_density_momentum_increment,
    periodic_home_convective_momentum_increment,
    reconstruct_populations,
)
from wanphys._src.fluid.fluid_grid.home_lbm import kernels


class TestHomeLbmFoundation(unittest.TestCase):
    def _run_warp_convective_increment(self, device: str) -> None:
        shape = (5, 4, 3)
        stride = int(np.prod(shape))
        coordinates = np.indices(shape, dtype=np.float32)
        rho = 1.0 + 0.015 * np.sin(2.0 * np.pi * coordinates[0] / shape[0])
        velocity = np.stack(
            (
                0.035 + 0.012 * np.cos(2.0 * np.pi * coordinates[1] / shape[1]),
                -0.018 + 0.007 * np.sin(2.0 * np.pi * coordinates[2] / shape[2]),
                0.005 * np.cos(2.0 * np.pi * coordinates[0] / shape[0]),
            ),
            axis=-1,
        )
        moments = np.zeros(shape + (10,), dtype=np.float32)
        moments[..., 0] = rho
        moments[..., 1:4] = rho[..., None] * velocity
        moments_soa = np.ascontiguousarray(np.moveaxis(moments, -1, 0).reshape(-1))
        moments_wp = wp.array(moments_soa, dtype=float, device=device)
        directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device
        )
        weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=device
        )
        density_increment = wp.zeros(shape, dtype=float, device=device)
        increment = wp.zeros(3 * stride, dtype=float, device=device)

        wp.launch(
            kernels.periodic_convective_momentum_increment_kernel,
            dim=shape,
            inputs=[
                moments_wp,
                directions,
                weights,
                density_increment,
                increment,
                *shape,
                stride,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        actual = np.moveaxis(increment.numpy().reshape(3, *shape), 0, -1)
        expected = periodic_home_convective_density_momentum_increment(moments)
        np.testing.assert_allclose(
            density_increment.numpy(), expected[..., 0], rtol=3.0e-5, atol=2.0e-8
        )
        np.testing.assert_allclose(actual, expected[..., 1:4], rtol=3.0e-5, atol=2.0e-8)

    def test_d3q27_quadrature_is_isotropic(self) -> None:
        np.testing.assert_allclose(D3Q27_WEIGHTS.sum(), 1.0, atol=1.0e-15)
        np.testing.assert_allclose(D3Q27_WEIGHTS @ D3Q27_DIRECTIONS, np.zeros(3), atol=1.0e-15)
        second = np.einsum("q,qa,qb->ab", D3Q27_WEIGHTS, D3Q27_DIRECTIONS, D3Q27_DIRECTIONS)
        np.testing.assert_allclose(second, np.eye(3) / 3.0, atol=1.0e-15)
        for direction, opposite in enumerate(D3Q27_OPPOSITE):
            np.testing.assert_array_equal(D3Q27_DIRECTIONS[opposite], -D3Q27_DIRECTIONS[direction])
            self.assertEqual(int(D3Q27_OPPOSITE[opposite]), direction)

    def test_third_order_reconstruction_preserves_all_ten_moments(self) -> None:
        rng = np.random.default_rng(41027)
        count = 32
        rho = rng.uniform(0.8, 1.2, size=count)
        velocity = rng.uniform(-0.08, 0.08, size=(count, 3))
        moments = np.zeros((count, 10), dtype=np.float64)
        moments[:, 0] = rho
        moments[:, 1:4] = rho[:, None] * velocity
        moments[:, 4] = rho * (velocity[:, 0] ** 2 + rng.uniform(-0.01, 0.01, count))
        moments[:, 5] = rho * (velocity[:, 1] ** 2 + rng.uniform(-0.01, 0.01, count))
        moments[:, 6] = rho * (velocity[:, 2] ** 2 + rng.uniform(-0.01, 0.01, count))
        moments[:, 7] = rho * (velocity[:, 0] * velocity[:, 1] + rng.uniform(-0.01, 0.01, count))
        moments[:, 8] = rho * (velocity[:, 0] * velocity[:, 2] + rng.uniform(-0.01, 0.01, count))
        moments[:, 9] = rho * (velocity[:, 1] * velocity[:, 2] + rng.uniform(-0.01, 0.01, count))

        recovered = extract_moments(reconstruct_populations(moments))
        np.testing.assert_allclose(recovered, moments, rtol=2.0e-13, atol=2.0e-13)

    def test_equilibrium_convective_populations_have_exact_moments(self) -> None:
        rng = np.random.default_rng(7319)
        rho = rng.uniform(0.8, 1.2, size=19)
        velocity = rng.uniform(-0.09, 0.09, size=(19, 3))
        moments = np.zeros((19, 10), dtype=np.float64)
        moments[:, 0] = rho
        moments[:, 1:4] = rho[:, None] * velocity

        convective = equilibrium_convective_populations(moments)

        np.testing.assert_allclose(
            np.sum(convective, axis=-1), 0.0, rtol=0.0, atol=2.0e-15
        )
        np.testing.assert_allclose(
            np.einsum("nq,qa->na", convective, D3Q27_DIRECTIONS),
            0.0,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.einsum(
                "nq,qa,qb->nab",
                convective,
                D3Q27_DIRECTIONS,
                D3Q27_DIRECTIONS,
            ),
            rho[:, None, None]
            * np.einsum("na,nb->nab", velocity, velocity),
            rtol=2.0e-13,
            atol=2.0e-15,
        )

    def test_periodic_home_convective_increment_is_exact_stream_difference(self) -> None:
        shape = (5, 4, 3)
        coordinates = np.indices(shape, dtype=np.float64)
        rho = 1.0 + 0.01 * np.sin(2.0 * np.pi * coordinates[0] / shape[0])
        velocity = np.stack(
            (
                0.04 + 0.01 * np.cos(2.0 * np.pi * coordinates[1] / shape[1]),
                -0.02 + 0.008 * np.sin(2.0 * np.pi * coordinates[2] / shape[2]),
                0.006 * np.cos(2.0 * np.pi * coordinates[0] / shape[0]),
            ),
            axis=-1,
        )
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = rho
        moments[..., 1:4] = rho[..., None] * velocity
        moments[..., 4] = rho * velocity[..., 0] ** 2
        moments[..., 5] = rho * velocity[..., 1] ** 2
        moments[..., 6] = rho * velocity[..., 2] ** 2
        moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
        moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
        moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
        populations = reconstruct_populations(moments)
        convective = equilibrium_convective_populations(moments)

        def stream(values: np.ndarray) -> np.ndarray:
            streamed = np.empty_like(values)
            for q, c in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
                streamed[..., q] = np.roll(
                    values[..., q],
                    shift=tuple(int(component) for component in c),
                    axis=(0, 1, 2),
                )
            return streamed

        full_momentum = np.einsum(
            "...q,qa->...a", stream(populations), D3Q27_DIRECTIONS
        )
        without_momentum = np.einsum(
            "...q,qa->...a", stream(populations - convective), D3Q27_DIRECTIONS
        )
        increment = periodic_home_convective_momentum_increment(moments)
        density_momentum_increment = (
            periodic_home_convective_density_momentum_increment(moments)
        )

        full_density = np.sum(stream(populations), axis=-1)
        without_density = np.sum(stream(populations - convective), axis=-1)
        np.testing.assert_allclose(
            density_momentum_increment[..., 0],
            full_density - without_density,
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            increment, full_momentum - without_momentum, rtol=2.0e-13, atol=2.0e-15
        )
        np.testing.assert_allclose(
            np.sum(increment, axis=(0, 1, 2)), 0.0, rtol=0.0, atol=2.0e-14
        )

        uniform = np.broadcast_to(moments[0, 0, 0], moments.shape).copy()
        np.testing.assert_allclose(
            periodic_home_convective_momentum_increment(uniform),
            0.0,
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_cpu_warp_convective_increment_matches_reference(self) -> None:
        self._run_warp_convective_increment("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_warp_convective_increment_matches_reference(self) -> None:
        self._run_warp_convective_increment("cuda:0")

    def test_warp_projection_matches_reference(self) -> None:
        device = "cpu"
        stride = 4
        moments_by_cell = np.asarray(
            [
                [1.0, 0.03, -0.02, 0.01, 0.001, 0.0004, 0.0001, -0.0006, 0.0003, -0.0002],
                [0.9, -0.02, 0.01, 0.00, 0.0008, 0.0002, 0.0003, 0.0001, -0.0002, 0.0004],
                [1.1, 0.00, 0.04, -0.03, 0.0002, 0.0015, 0.0009, -0.0001, 0.0002, -0.0005],
                [1.0, 0.00, 0.00, 0.00, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            dtype=np.float32,
        )
        moment_soa = np.ascontiguousarray(moments_by_cell.T.reshape(-1))
        moments = wp.array(moment_soa, dtype=float, device=device)
        directions = wp.array(D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device)
        weights = wp.array(D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=device)
        populations = wp.zeros(27 * stride, dtype=float, device=device)
        recovered = wp.zeros(10 * stride, dtype=float, device=device)

        wp.launch(
            kernels.reconstruct_populations_kernel,
            dim=(27, stride),
            inputs=[moments, directions, weights, populations, stride],
            device=device,
        )
        wp.launch(
            kernels.extract_moments_kernel,
            dim=stride,
            inputs=[populations, directions, recovered, stride],
            device=device,
        )
        wp.synchronize_device(device)

        populations_by_cell = populations.numpy().reshape(27, stride).T
        reference_populations = reconstruct_populations(moments_by_cell)
        np.testing.assert_allclose(populations_by_cell, reference_populations, rtol=2.0e-5, atol=2.0e-6)
        recovered_by_cell = recovered.numpy().reshape(10, stride).T
        np.testing.assert_allclose(recovered_by_cell, moments_by_cell, rtol=2.0e-5, atol=2.0e-6)

    def test_state_has_only_ten_persistent_fluid_scalars_per_cell(self) -> None:
        model = HomeLbmModel(fluid_grid_res=(3, 4, 5), device="cpu")
        state = HomeLbmState(model)
        self.assertEqual(state.persistent_scalar_count, 10 * 3 * 4 * 5)
        self.assertEqual(state.moments.size, state.persistent_scalar_count)
        self.assertFalse(hasattr(state, "f"))

    def test_scaling_is_round_trip_and_model_rejects_silent_dt_change(self) -> None:
        scaling = LatticeScaling(cell_size=0.02, time_step=0.001, reference_density=1000.0)
        self.assertAlmostEqual(scaling.velocity_to_physical(scaling.velocity_to_lattice(1.25)), 1.25)
        self.assertAlmostEqual(scaling.viscosity_to_physical(scaling.viscosity_to_lattice(1.0e-6)), 1.0e-6)
        self.assertAlmostEqual(scaling.force_to_physical(scaling.force_to_lattice(3.5)), 3.5)
        self.assertAlmostEqual(scaling.pressure_to_physical(scaling.pressure_to_lattice(2.4)), 2.4)
        self.assertAlmostEqual(
            scaling.surface_tension_to_physical(
                scaling.surface_tension_to_lattice(0.072)
            ),
            0.072,
        )
        self.assertAlmostEqual(
            scaling.surface_tension_unit, scaling.pressure_unit * scaling.cell_size
        )

        model = HomeLbmModel(
            fluid_grid_res=(4, 4, 4),
            fluid_grid_cell_size=0.02,
            time_step=0.001,
            kinematic_viscosity=1.0e-4,
            device="cpu",
        )
        model.validate_step(0.001)
        with self.assertRaises(ValueError):
            model.validate_step(0.002)

    def test_fused_step_diagnostics_match_full_grid_reduction(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=(5, 4, 3),
            time_step=0.01,
            kinematic_viscosity=0.03,
            periodic=(True, True, True),
            device="cpu",
        )
        solver = HomeLbmSolver(model)
        state_in = HomeLbmState(model)
        state_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(state_in, rho=1.03, velocity=(0.025, -0.01, 0.015))

        solver.step(state_in, state_out, model.time_step)
        fused = solver.collect_diagnostics(state_out)
        scanned = solver.collect_diagnostics(state_out, force_recompute=True)

        self.assertEqual(fused.invalid_cell_count, scanned.invalid_cell_count)
        self.assertEqual(fused.missing_cut_link_count, scanned.missing_cut_link_count)
        self.assertEqual(
            fused.unsupported_interpolation_count, scanned.unsupported_interpolation_count
        )
        self.assertEqual(
            fused.wall_confined_interpolation_count, scanned.wall_confined_interpolation_count
        )
        self.assertAlmostEqual(fused.min_density, scanned.min_density, places=7)
        self.assertAlmostEqual(fused.max_density, scanned.max_density, places=7)
        self.assertAlmostEqual(fused.max_speed, scanned.max_speed, places=7)
        self.assertAlmostEqual(
            fused.max_nonequilibrium_stress,
            scanned.max_nonequilibrium_stress,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
