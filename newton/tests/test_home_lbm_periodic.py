# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Periodic streaming and central-moment collision tests for HOME-LBM."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    HomeLbmDomain,
    HomeLbmModel,
    HomeLbmState,
    collide_central_moments,
    extract_moments,
    reconstruct_populations,
)


def _equilibrium_moments(rho: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    moments = np.zeros(rho.shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * velocity
    moments[..., 4] = rho * velocity[..., 0] * velocity[..., 0]
    moments[..., 5] = rho * velocity[..., 1] * velocity[..., 1]
    moments[..., 6] = rho * velocity[..., 2] * velocity[..., 2]
    moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
    moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
    return moments


def _periodic_reference_step(
    moments: np.ndarray,
    shear_omega: float,
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    nx, ny, nz, _ = moments.shape
    reconstructed = reconstruct_populations(moments)
    streamed = np.empty_like(reconstructed)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for direction, c in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
                    source = ((i - c[0]) % nx, (j - c[1]) % ny, (k - c[2]) % nz)
                    streamed[i, j, k, direction] = reconstructed[source][direction]
    streamed_moments = extract_moments(streamed)
    force = streamed_moments[..., 0, None] * np.asarray(acceleration, dtype=np.float64)
    return collide_central_moments(streamed_moments, shear_omega, force)


def _upload_moments(state, moments: np.ndarray) -> None:
    soa = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(soa, dtype=float, device=state.device))


def _download_moments(state) -> np.ndarray:
    nx, ny, nz = state.res
    return state.moments.numpy().reshape(10, -1).T.reshape(nx, ny, nz, 10).astype(np.float64)


class TestHomeLbmPeriodic(unittest.TestCase):
    def test_closed_form_collision_conserves_and_relaxes_expected_modes(self) -> None:
        rho = np.asarray([1.2])
        velocity = np.asarray([[0.04, -0.03, 0.02]])
        moments = _equilibrium_moments(rho, velocity)
        moments[0, 4:10] += rho[0] * np.asarray([0.030, -0.010, 0.020, 0.012, -0.006, 0.009])
        omega = 0.4

        post = collide_central_moments(moments, omega)
        np.testing.assert_allclose(post[:, :4], moments[:, :4], atol=0.0)

        post_second = np.asarray(
            [
                [post[0, 4], post[0, 7], post[0, 8]],
                [post[0, 7], post[0, 5], post[0, 9]],
                [post[0, 8], post[0, 9], post[0, 6]],
            ]
        ) / rho[0]
        equilibrium = np.outer(velocity[0], velocity[0])
        self.assertAlmostEqual(float(np.trace(post_second)), float(np.trace(equilibrium)), delta=1.0e-14)

        pre_second = np.asarray(
            [
                [moments[0, 4], moments[0, 7], moments[0, 8]],
                [moments[0, 7], moments[0, 5], moments[0, 9]],
                [moments[0, 8], moments[0, 9], moments[0, 6]],
            ]
        ) / rho[0]
        identity = np.eye(3)
        pre_dev = pre_second - np.trace(pre_second) * identity / 3.0
        eq_dev = equilibrium - np.trace(equilibrium) * identity / 3.0
        post_dev = post_second - np.trace(post_second) * identity / 3.0
        np.testing.assert_allclose(post_dev - eq_dev, (1.0 - omega) * (pre_dev - eq_dev), atol=1.0e-14)

    def test_warp_step_matches_cellwise_reference(self) -> None:
        shape = (3, 4, 2)
        rng = np.random.default_rng(1729)
        rho = rng.uniform(0.97, 1.03, size=shape)
        velocity = rng.uniform(-0.025, 0.025, size=shape + (3,))
        moments = _equilibrium_moments(rho, velocity)
        moments[..., 4:10] += rng.uniform(-0.002, 0.002, size=shape + (6,))

        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=0.5,
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        _upload_moments(state, moments)
        expected = _periodic_reference_step(moments, model.shear_omega)

        domain.step(1.0)
        wp.synchronize_device("cpu")
        actual = _download_moments(domain.state)
        np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-6)

    def test_periodic_domain_conserves_global_mass_and_momentum(self) -> None:
        shape = (5, 4, 3)
        rng = np.random.default_rng(314159)
        rho = rng.uniform(0.99, 1.01, size=shape)
        velocity = rng.uniform(-0.015, 0.015, size=shape + (3,))
        moments = _equilibrium_moments(rho, velocity)
        moments[..., 4:10] += rng.uniform(-0.001, 0.001, size=shape + (6,))

        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=0.2,
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        _upload_moments(state, moments)
        initial_totals = moments[..., :4].sum(axis=(0, 1, 2))

        for _ in range(12):
            domain.step(1.0)
        wp.synchronize_device("cpu")
        final = _download_moments(domain.state)
        final_totals = final[..., :4].sum(axis=(0, 1, 2))
        np.testing.assert_allclose(final_totals, initial_totals, rtol=2.0e-6, atol=2.0e-5)
        self.assertTrue(np.isfinite(final).all())

    def test_uniform_acceleration_adds_exact_lattice_impulse(self) -> None:
        shape = (4, 3, 2)
        acceleration = (2.0e-4, -1.0e-4, 0.5e-4)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=acceleration,
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_uniform_lattice(state)

        for step in range(1, 9):
            before = _download_moments(domain.state)
            expected = _periodic_reference_step(before, model.shear_omega, acceleration)
            domain.step(1.0)
            wp.synchronize_device("cpu")
            actual = _download_moments(domain.state)
            np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-6)
            mean_velocity = actual[..., 1:4].sum(axis=(0, 1, 2)) / actual[..., 0].sum()
            np.testing.assert_allclose(mean_velocity, step * np.asarray(acceleration), rtol=2.0e-5, atol=2.0e-7)

    def test_explicit_force_density_overrides_model_acceleration(self) -> None:
        shape = (4, 3, 2)
        model_acceleration = (9.0e-4, -7.0e-4, 5.0e-4)
        applied_acceleration = (2.0e-4, -1.0e-4, 0.5e-4)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=model_acceleration,
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state_in = domain.create_state()
        state_out = HomeLbmState(model)
        domain.solver.initialize_uniform_lattice(state_in)
        before = _download_moments(state_in)
        expected = _periodic_reference_step(
            before,
            model.shear_omega,
            applied_acceleration,
        )
        force = np.empty(shape + (3,), dtype=np.float32)
        force[...] = np.asarray(applied_acceleration, dtype=np.float32)

        domain.solver.step(
            state_in,
            state_out,
            1.0,
            force_density=wp.array(force, dtype=wp.vec3, device="cpu"),
        )
        wp.synchronize_device("cpu")

        np.testing.assert_allclose(
            _download_moments(state_out),
            expected,
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    def test_uniform_flow_remains_exactly_uniform(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=(4, 3, 5),
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=0.12,
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_uniform_lattice(state, rho=1.0, velocity=(0.04, -0.02, 0.01))
        for _ in range(20):
            domain.step(1.0)
        wp.synchronize_device("cpu")

        final = _download_moments(domain.state)
        expected = _equilibrium_moments(
            np.ones((4, 3, 5), dtype=np.float64),
            np.broadcast_to(np.asarray([0.04, -0.02, 0.01]), (4, 3, 5, 3)),
        )
        np.testing.assert_allclose(final, expected, rtol=3.0e-5, atol=3.0e-6)
        diagnostics = domain.solver.validate_state(domain.state)
        self.assertEqual(diagnostics.invalid_cell_count, 0)
        self.assertAlmostEqual(diagnostics.min_density, 1.0, delta=3.0e-6)
        self.assertAlmostEqual(diagnostics.max_density, 1.0, delta=3.0e-6)
        self.assertAlmostEqual(diagnostics.max_speed, np.linalg.norm([0.04, -0.02, 0.01]), delta=3.0e-6)
        self.assertLess(diagnostics.max_nonequilibrium_stress, 3.0e-6)

    def test_diagnostics_reject_nonfinite_state(self) -> None:
        model = HomeLbmModel(fluid_grid_res=(2, 2, 2), device="cpu")
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_uniform_lattice(state)
        moments = _download_moments(state)
        moments[0, 0, 0, 0] = np.nan
        _upload_moments(state, moments)

        diagnostics = domain.solver.collect_diagnostics(state)
        self.assertEqual(diagnostics.invalid_cell_count, 1)
        with self.assertRaises(FloatingPointError):
            domain.solver.validate_state(state)

    def test_taylor_green_mode_recovers_configured_viscosity(self) -> None:
        resolution = 32
        steps = 20
        viscosity = 0.08
        initial_amplitude = 0.01
        coordinates = 2.0 * np.pi * (np.arange(resolution) + 0.5) / resolution
        x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
        basis_x = np.sin(x) * np.cos(y)
        basis_y = -np.cos(x) * np.sin(y)
        velocity = np.zeros((resolution, resolution, 1, 3), dtype=np.float64)
        velocity[:, :, 0, 0] = initial_amplitude * basis_x
        velocity[:, :, 0, 1] = initial_amplitude * basis_y
        rho = np.ones((resolution, resolution, 1), dtype=np.float64)

        model = HomeLbmModel(
            fluid_grid_res=(resolution, resolution, 1),
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=viscosity,
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        _upload_moments(state, _equilibrium_moments(rho, velocity))
        for _ in range(steps):
            domain.step(1.0)
        wp.synchronize_device("cpu")

        final = _download_moments(domain.state)
        ux = final[:, :, 0, 1] / final[:, :, 0, 0]
        uy = final[:, :, 0, 2] / final[:, :, 0, 0]
        measured_amplitude = (
            np.sum(ux * basis_x) + np.sum(uy * basis_y)
        ) / (np.sum(basis_x * basis_x) + np.sum(basis_y * basis_y))
        wave_number = 2.0 * np.pi / resolution
        expected_amplitude = initial_amplitude * np.exp(
            -2.0 * viscosity * wave_number * wave_number * steps
        )

        self.assertAlmostEqual(measured_amplitude / expected_amplitude, 1.0, delta=0.01)
        self.assertLess(abs(float(final[..., 0].sum()) - resolution * resolution), 2.0e-4)

    def test_taylor_green_exhibits_second_order_grid_convergence(self) -> None:
        domain_length = 1.0
        physical_viscosity = 0.01
        lattice_viscosity = 0.08
        final_time = 1.0
        initial_speed = 0.02

        def velocity_error(resolution: int) -> float:
            cell_size = domain_length / resolution
            time_step = lattice_viscosity * cell_size * cell_size / physical_viscosity
            step_count = int(round(final_time / time_step))
            actual_final_time = step_count * time_step
            coordinates = (np.arange(resolution) + 0.5) * cell_size
            x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
            basis_x = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
            basis_y = -np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
            velocity_physical = np.zeros((resolution, resolution, 1, 3), dtype=np.float64)
            velocity_physical[:, :, 0, 0] = initial_speed * basis_x
            velocity_physical[:, :, 0, 1] = initial_speed * basis_y
            velocity_scale = time_step / cell_size
            velocity_lattice = velocity_physical * velocity_scale
            rho = np.ones((resolution, resolution, 1), dtype=np.float64)

            model = HomeLbmModel(
                fluid_grid_res=(resolution, resolution, 1),
                fluid_grid_cell_size=cell_size,
                time_step=time_step,
                kinematic_viscosity=physical_viscosity,
                device="cpu",
            )
            domain = HomeLbmDomain(model)
            state = domain.create_state()
            _upload_moments(state, _equilibrium_moments(rho, velocity_lattice))
            for _ in range(step_count):
                domain.step(time_step)
            wp.synchronize_device("cpu")

            final = _download_moments(domain.state)
            ux = final[:, :, 0, 1] / final[:, :, 0, 0] / velocity_scale
            uy = final[:, :, 0, 2] / final[:, :, 0, 0] / velocity_scale
            decay = np.exp(-2.0 * physical_viscosity * (2.0 * np.pi) ** 2 * actual_final_time)
            expected_x = initial_speed * decay * basis_x
            expected_y = initial_speed * decay * basis_y
            return float(
                np.sqrt(
                    np.sum((ux - expected_x) ** 2 + (uy - expected_y) ** 2)
                    / np.sum(expected_x**2 + expected_y**2)
                )
            )

        coarse_error = velocity_error(16)
        fine_error = velocity_error(32)
        self.assertLess(coarse_error, 0.025)
        self.assertLess(fine_error, 0.0065)
        self.assertGreater(coarse_error / fine_error, 3.5)


if __name__ == "__main__":
    unittest.main()
