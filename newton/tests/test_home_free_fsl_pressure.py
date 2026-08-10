# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bogner FSL pressure/shear closure against its analytic coefficients."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_OPPOSITE,
    D3Q27_WEIGHTS,
    HomeFreeInterfaceGeometry,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    collide_central_moments,
    extract_moments,
    fsl_boundary_coefficients,
    fsl_boundary_population,
    fsl_extrapolated_boundary_velocity,
    fsl_extrapolated_boundary_strain,
    fsl_stream_moments,
    gas_pressure_boundary_populations,
    only_missing_boundary_residual,
    plic_fsl_link_coverage,
    plic_plane_offset,
    reconstruct_populations,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof import kernels as vof_kernels
from wanphys._src.fluid.fluid_grid.home_lbm import kernels as home_kernels


def _equilibrium_moments(rho: float, velocity: np.ndarray) -> np.ndarray:
    u = np.asarray(velocity, dtype=np.float64)
    moments = np.empty(10, dtype=np.float64)
    moments[0] = rho
    moments[1:4] = rho * u
    moments[4] = rho * u[0] * u[0]
    moments[5] = rho * u[1] * u[1]
    moments[6] = rho * u[2] * u[2]
    moments[7] = rho * u[0] * u[1]
    moments[8] = rho * u[0] * u[2]
    moments[9] = rho * u[1] * u[2]
    return moments


def _only_missing_stream_collide(
    moments: np.ndarray,
    flags: np.ndarray,
    gas_density: np.ndarray,
    shear_omega: float,
) -> np.ndarray:
    shape = flags.shape
    reconstructed = reconstruct_populations(moments)
    boundary = gas_pressure_boundary_populations(moments, gas_density)
    streamed = np.full(shape + (27,), np.nan, dtype=np.float64)
    active = np.isin(flags, (1, 2))
    for index in map(tuple, np.argwhere(active)):
        for q, c in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
            source = tuple(index[axis] - int(c[axis]) for axis in range(3))
            if any(source[axis] < 0 or source[axis] >= shape[axis] for axis in range(3)):
                raise ValueError("OM reference active domain touched a non-periodic wall")
            if int(flags[source]) == 0:
                streamed[index + (q,)] = boundary[index + (q,)]
            elif active[source]:
                streamed[index + (q,)] = reconstructed[source + (q,)]
            else:
                raise ValueError("OM reference encountered a solid link")
    result = moments.copy()
    result[active] = collide_central_moments(
        extract_moments(streamed[active]), shear_omega
    )
    return result


class TestHomeFreeFslPressure(unittest.TestCase):
    def _warp_strain_projection_matches_reference(self, device: str) -> None:
        shape = (8, 3, 2)
        stride = int(np.prod(shape))
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 1
        flags[1:4] = 2
        flags[4] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[0, ..., 0] = -1.0
        normals[4, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[0] = 0.25
        offsets[4] = -0.25
        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(True, True, True),
        )
        base = np.asarray(
            ((0.02, 0.006, -0.004), (0.006, -0.01, 0.003), (-0.004, 0.003, 0.005)),
            dtype=np.float64,
        )
        slope = np.asarray(
            ((0.004, -0.002, 0.001), (-0.002, 0.003, -0.001), (0.001, -0.001, 0.002)),
            dtype=np.float64,
        )
        bulk = np.empty(shape + (3, 3), dtype=np.float64)
        for index in np.ndindex(shape):
            bulk[index] = base + slope * float(index[0])
        target = 0.0123
        expected = fsl_extrapolated_boundary_strain(
            bulk,
            normals,
            coverage,
            target,
            periodic=(True, True, True),
        )
        bulk_components = np.stack(
            (
                bulk[..., 0, 0],
                bulk[..., 1, 1],
                bulk[..., 2, 2],
                bulk[..., 0, 1],
                bulk[..., 0, 2],
                bulk[..., 1, 2],
            ),
            axis=0,
        )
        output = wp.zeros(6 * 27 * stride, dtype=float, device=device)
        invalid = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            vof_kernels.extrapolate_fsl_boundary_strain_kernel,
            dim=27 * stride,
            inputs=[
                wp.array(
                    np.ascontiguousarray(bulk_components.reshape(-1), dtype=np.float32),
                    dtype=float,
                    device=device,
                ),
                wp.array(normals.astype(np.float32), dtype=wp.vec3, device=device),
                wp.array(
                    coverage.hydrodynamic_active.astype(np.int32),
                    dtype=wp.int32,
                    device=device,
                ),
                wp.array(
                    np.ascontiguousarray(coverage.status.reshape(-1), dtype=np.int32),
                    dtype=wp.int32,
                    device=device,
                ),
                wp.array(
                    np.ascontiguousarray(coverage.plane_owner.reshape(-1), dtype=np.int32),
                    dtype=wp.int32,
                    device=device,
                ),
                wp.array(
                    np.ascontiguousarray(coverage.fraction.reshape(-1), dtype=np.float32),
                    dtype=float,
                    device=device,
                ),
                wp.full(27 * stride, target, dtype=float, device=device),
                wp.array(
                    D3Q27_DIRECTIONS.astype(np.float32),
                    dtype=wp.vec3,
                    device=device,
                ),
                output,
                invalid,
                0,
                1,
                1,
                1,
                *shape,
                stride,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(invalid.numpy()[0]), 0)
        actual = np.moveaxis(
            output.numpy().reshape(6, *shape, 27), 0, -1
        )
        expected_components = np.stack(
            (
                expected[..., 0, 0],
                expected[..., 1, 1],
                expected[..., 2, 2],
                expected[..., 0, 1],
                expected[..., 0, 2],
                expected[..., 1, 2],
            ),
            axis=-1,
        )
        applicable = np.isin(coverage.status, (1, 2))
        np.testing.assert_allclose(
            actual[applicable],
            expected_components[applicable],
            rtol=3.0e-5,
            atol=3.0e-7,
        )

    def _warp_grid_hybrid_matches_reference(self, device: str) -> None:
        shape = (11, 11, 11)
        stride = int(np.prod(shape))
        center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        coordinates = np.indices(shape).transpose(1, 2, 3, 0)
        radius = 3.25
        distance = np.linalg.norm(coordinates - center, axis=-1)
        fill = np.clip(
            0.5 + (radius - distance) / 2.0, 0.0, 1.0
        ).astype(np.float32)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            kinematic_viscosity=0.5,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)
        geometry = HomeFreeInterfaceGeometry(model)
        geometry.update(state)
        flags = state.flags.numpy()
        coverage = plic_fsl_link_coverage(
            geometry.normal.numpy(),
            geometry.plane_offset.numpy(),
            flags,
            periodic=model.periodic,
            tolerance=1.0e-6,
        )
        rng = np.random.default_rng(32416190071)
        rho = rng.uniform(0.985, 1.015, size=shape)
        velocity = rng.uniform(-0.008, 0.008, size=shape + (3,))
        moments = np.asarray(
            [
                _equilibrium_moments(cell_rho, cell_velocity)
                for cell_rho, cell_velocity in zip(
                    rho.reshape(-1), velocity.reshape(-1, 3), strict=True
                )
            ]
        ).reshape(shape + (10,))
        moments[..., 4:10] += rng.uniform(
            -2.0e-4, 2.0e-4, size=shape + (6,)
        )
        gas_density = np.ones(shape, dtype=np.float64)
        interface = flags == 1
        gas_density[interface] -= (
            0.018 * geometry.curvature.numpy()[interface].astype(np.float64)
        )
        expected_velocity = fsl_extrapolated_boundary_velocity(
            moments,
            coverage,
            periodic=model.periodic,
            no_support_policy="only_missing",
        )
        expected_stream = fsl_stream_moments(
            moments,
            flags,
            coverage,
            gas_density,
            expected_velocity,
            model.shear_omega,
            periodic=model.periodic,
            no_support_policy="only_missing",
        )
        expected = expected_stream.moments.copy()
        active = coverage.hydrodynamic_active
        expected[active] = collide_central_moments(
            expected[active], model.shear_omega
        )

        device_moments = wp.array(
            np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32),
            dtype=float,
            device=device,
        )
        device_active = wp.array(
            active.astype(np.int32), dtype=wp.int32, device=device
        )
        device_status = wp.array(
            np.ascontiguousarray(coverage.status.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        device_fraction = wp.array(
            np.ascontiguousarray(coverage.fraction.reshape(-1), dtype=np.float32),
            dtype=float,
            device=device,
        )
        device_owner = wp.array(
            np.ascontiguousarray(coverage.plane_owner.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device
        )
        weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=device
        )
        opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=device)
        device_velocity = wp.zeros(27 * stride, dtype=wp.vec3, device=device)
        invalid_velocity = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            vof_kernels.extrapolate_fsl_boundary_velocity_kernel,
            dim=27 * stride,
            inputs=[
                device_moments,
                device_active,
                device_status,
                device_fraction,
                directions,
                device_velocity,
                invalid_velocity,
                1,
                0,
                0,
                0,
                *shape,
                stride,
            ],
            device=device,
        )
        raw_moments = wp.zeros(10 * stride, dtype=float, device=device)
        collided_moments = wp.zeros(10 * stride, dtype=float, device=device)
        invalid_stream = wp.zeros(1, dtype=wp.int32, device=device)
        fallback_count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            vof_kernels.fsl_stream_moments_kernel,
            dim=shape,
            inputs=[
                device_moments,
                device_active,
                device_status,
                device_owner,
                device_fraction,
                wp.array(gas_density.astype(np.float32), dtype=float, device=device),
                device_velocity,
                wp.zeros(6 * 27 * stride, dtype=float, device=device),
                directions,
                weights,
                opposites,
                raw_moments,
                invalid_stream,
                fallback_count,
                model.shear_omega,
                0,
                1,
                0,
                0,
                0,
                *shape,
                stride,
            ],
            device=device,
        )
        invalid_collision = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            home_kernels.collide_force_free_moments_kernel,
            dim=shape,
            inputs=[
                raw_moments,
                device_active,
                collided_moments,
                invalid_collision,
                model.shear_omega,
                shape[1],
                shape[2],
                stride,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(invalid_velocity.numpy()[0]), 0)
        self.assertEqual(int(invalid_stream.numpy()[0]), 0)
        self.assertEqual(int(invalid_collision.numpy()[0]), 0)
        self.assertEqual(
            int(fallback_count.numpy()[0]), coverage.no_support_links
        )
        applicable = np.isin(coverage.status, (1, 2, -7))
        np.testing.assert_allclose(
            device_velocity.numpy().reshape(shape + (27, 3))[applicable],
            expected_velocity[applicable],
            rtol=4.0e-5,
            atol=4.0e-7,
        )
        actual = (
            collided_moments.numpy()
            .reshape(10, -1)
            .T.reshape(shape + (10,))
            .astype(np.float64)
        )
        np.testing.assert_allclose(actual, expected, rtol=8.0e-5, atol=8.0e-7)

    def _warp_matches_random_reference(self, device: str) -> None:
        rng = np.random.default_rng(982451653)
        count = 512
        local_rho = rng.uniform(0.96, 1.04, size=count)
        local_velocity = rng.uniform(-0.025, 0.025, size=(count, 3))
        interior_rho = rng.uniform(0.96, 1.04, size=count)
        interior_velocity = rng.uniform(-0.025, 0.025, size=(count, 3))
        local = np.asarray(
            [_equilibrium_moments(rho, velocity) for rho, velocity in zip(local_rho, local_velocity, strict=True)]
        )
        interior = np.asarray(
            [
                _equilibrium_moments(rho, velocity)
                for rho, velocity in zip(interior_rho, interior_velocity, strict=True)
            ]
        )
        local[:, 4:10] += rng.uniform(-8.0e-4, 8.0e-4, size=(count, 6))
        interior[:, 4:10] += rng.uniform(-8.0e-4, 8.0e-4, size=(count, 6))
        direction = rng.integers(1, 27, size=count, dtype=np.int32)
        fraction = rng.uniform(0.0, 1.0, size=count)
        boundary_density = rng.uniform(0.97, 1.03, size=count)
        boundary_velocity = rng.uniform(-0.02, 0.02, size=(count, 3))
        strain_components = rng.uniform(-0.006, 0.006, size=(count, 6))
        strain = np.zeros((count, 3, 3), dtype=np.float64)
        strain[:, 0, 0] = strain_components[:, 0]
        strain[:, 1, 1] = strain_components[:, 1]
        strain[:, 2, 2] = strain_components[:, 2]
        strain[:, 0, 1] = strain[:, 1, 0] = strain_components[:, 3]
        strain[:, 0, 2] = strain[:, 2, 0] = strain_components[:, 4]
        strain[:, 1, 2] = strain[:, 2, 1] = strain_components[:, 5]
        omega = 1.27
        expected = np.asarray(
            [
                fsl_boundary_population(
                    local[index],
                    interior[index],
                    int(direction[index]),
                    float(fraction[index]),
                    float(boundary_density[index]),
                    omega,
                    boundary_velocity=boundary_velocity[index],
                    boundary_strain=strain[index],
                )
                for index in range(count)
            ]
        )

        output = wp.zeros(count, dtype=float, device=device)
        status = wp.zeros(count, dtype=wp.int32, device=device)
        wp.launch(
            vof_kernels.fsl_boundary_populations_kernel,
            dim=count,
            inputs=[
                wp.array(np.ascontiguousarray(local.T.reshape(-1), dtype=np.float32), dtype=float, device=device),
                wp.array(np.ascontiguousarray(interior.T.reshape(-1), dtype=np.float32), dtype=float, device=device),
                wp.array(direction, dtype=wp.int32, device=device),
                wp.array(fraction.astype(np.float32), dtype=float, device=device),
                wp.array(boundary_density.astype(np.float32), dtype=float, device=device),
                wp.array(boundary_velocity.astype(np.float32), dtype=wp.vec3, device=device),
                wp.array(
                    np.ascontiguousarray(strain_components.T.reshape(-1), dtype=np.float32),
                    dtype=float,
                    device=device,
                ),
                wp.array(D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device),
                wp.array(D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=device),
                wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=device),
                output,
                status,
                omega,
                count,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(status.numpy(), 1)
        np.testing.assert_allclose(output.numpy(), expected, rtol=5.0e-5, atol=2.0e-7)

    def test_coefficients_map_bogner_negative_lambda_to_positive_omega(self) -> None:
        delta = 0.25
        omega = 1.25
        weight = 2.0 / 27.0
        coefficients = fsl_boundary_coefficients(delta, omega, weight)

        self.assertAlmostEqual(coefficients.a0, 0.25)
        self.assertAlmostEqual(coefficients.opposite_a0, 0.5)
        self.assertAlmostEqual(coefficients.a1, -0.75)
        self.assertAlmostEqual(coefficients.nonequilibrium, 1.5625)
        self.assertAlmostEqual(
            coefficients.strain,
            -3.0 * (1.0 / omega - 0.5) * weight,
        )

    def test_planar_grid_stream_preserves_uniform_moving_equilibrium(self) -> None:
        shape = (8, 3, 2)
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 1
        flags[1:4] = 2
        flags[4] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[0, ..., 0] = -1.0
        normals[4, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[0] = plic_plane_offset(0.75, np.asarray((-1.0, 0.0, 0.0)))
        offsets[4] = plic_plane_offset(0.25, np.asarray((1.0, 0.0, 0.0)))
        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(True, True, True),
        )
        rho = 1.017
        velocity = np.asarray((0.013, -0.007, 0.004), dtype=np.float64)
        moments = np.empty(shape + (10,), dtype=np.float64)
        moments[...] = _equilibrium_moments(rho, velocity)
        boundary_velocity = fsl_extrapolated_boundary_velocity(
            moments,
            coverage,
            periodic=(True, True, True),
        )

        result = fsl_stream_moments(
            moments,
            flags,
            coverage,
            rho,
            boundary_velocity,
            1.1,
            periodic=(True, True, True),
        )

        self.assertEqual(result.boundary_link_count, coverage.usable_links)
        self.assertEqual(result.exact_link_count, coverage.exact_links)
        self.assertEqual(result.extrapolated_link_count, 0)
        np.testing.assert_allclose(
            result.moments[coverage.hydrodynamic_active],
            moments[coverage.hydrodynamic_active],
            rtol=2.0e-14,
            atol=2.0e-15,
        )

    def test_grid_stream_rejects_missing_interior_support(self) -> None:
        shape = (3, 1, 1)
        flags = np.asarray([1, 0, 0], dtype=np.int32).reshape(shape)
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[0, 0, 0] = (-1.0, 0.0, 0.0)
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[0, 0, 0] = 0.25
        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(True, True, True),
        )
        moments = np.empty(shape + (10,), dtype=np.float64)
        moments[...] = _equilibrium_moments(1.0, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "NO_SUPPORT"):
            fsl_stream_moments(
                moments,
                flags,
                coverage,
                1.0,
                np.zeros(shape + (27, 3), dtype=np.float64),
                1.0,
                periodic=(True, True, True),
            )

    def test_spherical_laplace_one_step_reports_om_and_fsl_residuals(self) -> None:
        shape = (11, 11, 11)
        center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        coordinates = np.indices(shape).transpose(1, 2, 3, 0)
        radius = 3.25
        gamma = 0.003
        distance = np.linalg.norm(coordinates - center, axis=-1)
        fill = np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0).astype(
            np.float32
        )
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.5,
            periodic=(False, False, False),
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)
        geometry = HomeFreeInterfaceGeometry(model)
        geometry.update(state)
        flags = state.flags.numpy()
        normals = geometry.normal.numpy().astype(np.float64)
        offsets = geometry.plane_offset.numpy().astype(np.float64)
        curvature = geometry.curvature.numpy().astype(np.float64)
        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=model.periodic,
            tolerance=1.0e-6,
        )
        gas_density = np.ones(shape, dtype=np.float64)
        interface = flags == 1
        gas_density[interface] -= 6.0 * gamma * curvature[interface]
        initial_density = 1.0 + 6.0 * gamma / radius
        moments = np.empty(shape + (10,), dtype=np.float64)
        moments[...] = _equilibrium_moments(initial_density, np.zeros(3))

        om = only_missing_boundary_residual(
            moments,
            flags,
            gas_density,
            periodic=model.periodic,
        )
        om_next = moments[..., :4] + om.total
        om_active = np.isin(flags, (1, 2))
        om_speed = np.linalg.norm(
            om_next[..., 1:4] / om_next[..., 0, None], axis=-1
        )
        om_max = float(np.max(om_speed[om_active]))

        self.assertTrue(np.isfinite(om_max))
        self.assertLess(om_max, 0.01)
        self.assertGreater(coverage.no_support_links, 0)
        with self.assertRaisesRegex(ValueError, "NO_SUPPORT"):
            fsl_stream_moments(
                moments,
                flags,
                coverage,
                gas_density,
                np.zeros(shape + (27, 3), dtype=np.float64),
                model.shear_omega,
                periodic=model.periodic,
            )
        hybrid = fsl_stream_moments(
            moments,
            flags,
            coverage,
            gas_density,
            fsl_extrapolated_boundary_velocity(
                moments,
                coverage,
                periodic=model.periodic,
                no_support_policy="only_missing",
            ),
            model.shear_omega,
            periodic=model.periodic,
            no_support_policy="only_missing",
        )
        hybrid_speed = np.linalg.norm(
            hybrid.moments[..., 1:4] / hybrid.moments[..., 0, None], axis=-1
        )
        hybrid_max = float(
            np.max(hybrid_speed[coverage.hydrodynamic_active])
        )
        self.assertEqual(
            hybrid.only_missing_fallback_count, coverage.no_support_links
        )
        self.assertEqual(hybrid.boundary_link_count, coverage.total_boundary_links)
        self.assertTrue(np.isfinite(hybrid_max))
        self.assertLess(hybrid_max, 0.01)
        self.assertLess(hybrid_max, 0.4 * om_max)

    def test_linkwise_velocity_extrapolation_is_exact_for_affine_flow(self) -> None:
        shape = (8, 3, 2)
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 1
        flags[1:4] = 2
        flags[4] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[0, ..., 0] = -1.0
        normals[4, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[0] = 0.25
        offsets[4] = -0.25
        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(True, True, True),
        )
        base = np.asarray((0.01, -0.006, 0.003), dtype=np.float64)
        gradient = np.asarray((0.004, -0.002, 0.001), dtype=np.float64)
        moments = np.empty(shape + (10,), dtype=np.float64)
        for index in np.ndindex(shape):
            moments[index] = _equilibrium_moments(
                1.0, base + gradient * float(index[0])
            )

        actual = fsl_extrapolated_boundary_velocity(
            moments,
            coverage,
            periodic=(True, True, True),
        )
        for index in map(tuple, np.argwhere(coverage.hydrodynamic_active)):
            for q, c in enumerate(D3Q27_DIRECTIONS):
                if int(coverage.status[index + (q,)]) not in (1, 2):
                    continue
                boundary_x = float(index[0]) - coverage.fraction[index + (q,)] * c[0]
                np.testing.assert_allclose(
                    actual[index + (q,)],
                    base + gradient * boundary_x,
                    rtol=0.0,
                    atol=2.0e-16,
                )

    def test_linkwise_strain_extrapolation_enforces_free_surface_components(self) -> None:
        shape = (8, 3, 2)
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 1
        flags[1:4] = 2
        flags[4] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[0, ..., 0] = -1.0
        normals[4, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[0] = 0.25
        offsets[4] = -0.25
        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(True, True, True),
        )
        base = np.asarray(
            ((0.02, 0.006, -0.004), (0.006, -0.01, 0.003), (-0.004, 0.003, 0.005)),
            dtype=np.float64,
        )
        slope = np.asarray(
            ((0.004, -0.002, 0.001), (-0.002, 0.003, -0.001), (0.001, -0.001, 0.002)),
            dtype=np.float64,
        )
        bulk = np.empty(shape + (3, 3), dtype=np.float64)
        for index in np.ndindex(shape):
            bulk[index] = base + slope * float(index[0])
        target_normal = 0.0123

        actual = fsl_extrapolated_boundary_strain(
            bulk,
            normals,
            coverage,
            target_normal,
            periodic=(True, True, True),
        )
        for index in map(tuple, np.argwhere(coverage.hydrodynamic_active)):
            for q, c in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
                if int(coverage.status[index + (q,)]) not in (1, 2):
                    continue
                owner = int(coverage.plane_owner[index + (q,)])
                source = tuple((index[axis] - int(c[axis])) % shape[axis] for axis in range(3))
                owner_index = index if owner == 1 else source
                n = normals[owner_index].copy()
                n /= np.linalg.norm(n)
                tensor = actual[index + (q,)]
                normal_vector = tensor @ n
                np.testing.assert_allclose(
                    normal_vector,
                    target_normal * n,
                    rtol=0.0,
                    atol=3.0e-17,
                )
                boundary_x = float(index[0]) - coverage.fraction[index + (q,)] * c[0]
                raw = base + slope * boundary_x
                tangent = np.eye(3) - np.outer(n, n)
                np.testing.assert_allclose(
                    tangent @ tensor @ tangent,
                    tangent @ raw @ tangent,
                    rtol=0.0,
                    atol=3.0e-17,
                )

    def test_spherical_fsl_support_resolution_audit(self) -> None:
        audited: list[tuple[int, int, int, int]] = []
        for side, radius in ((11, 3.25), (17, 5.25), (23, 7.25)):
            shape = (side, side, side)
            center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
            coordinates = np.indices(shape).transpose(1, 2, 3, 0)
            distance = np.linalg.norm(coordinates - center, axis=-1)
            fill = np.clip(
                0.5 + (radius - distance) / 2.0, 0.0, 1.0
            ).astype(np.float32)
            model = HomeLbmModel(
                fluid_grid_res=shape,
                periodic=(False, False, False),
                device="cpu",
            )
            fluid = HomeLbmState(model)
            HomeLbmSolver(model).initialize_uniform_lattice(fluid)
            state = HomeFreeState(model)
            state.initialize_from_fill_level(fluid, fill)
            geometry = HomeFreeInterfaceGeometry(model)
            geometry.update(state)
            coverage = plic_fsl_link_coverage(
                geometry.normal.numpy(),
                geometry.plane_offset.numpy(),
                state.flags.numpy(),
                periodic=model.periodic,
                tolerance=1.0e-6,
            )
            self.assertEqual(coverage.not_facing_links, 0)
            self.assertEqual(coverage.behind_links, 0)
            self.assertEqual(coverage.outside_links, 0)
            self.assertEqual(coverage.no_plane_links, 0)
            self.assertGreater(coverage.no_support_links, 0)
            self.assertLess(
                coverage.no_support_links / coverage.total_boundary_links,
                0.2,
            )
            audited.append(
                (
                    side,
                    coverage.total_boundary_links,
                    coverage.no_support_links,
                    coverage.extrapolated_links,
                )
            )

        support_gaps = [missing / total for _, total, missing, _ in audited]
        self.assertLess(support_gaps[-1], support_gaps[0])
        self.assertEqual(
            audited,
            [(11, 1226, 168, 48), (17, 3242, 192, 0), (23, 6098, 192, 48)],
        )

    def test_fixed_spherical_laplace_multistep_hybrid_vs_om(self) -> None:
        shape = (11, 11, 11)
        center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        coordinates = np.indices(shape).transpose(1, 2, 3, 0)
        radius = 3.25
        gamma = 0.003
        distance = np.linalg.norm(coordinates - center, axis=-1)
        fill = np.clip(
            0.5 + (radius - distance) / 2.0, 0.0, 1.0
        ).astype(np.float32)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.5,
            periodic=(False, False, False),
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)
        geometry = HomeFreeInterfaceGeometry(model)
        geometry.update(state)
        flags = state.flags.numpy()
        coverage = plic_fsl_link_coverage(
            geometry.normal.numpy(),
            geometry.plane_offset.numpy(),
            flags,
            periodic=model.periodic,
            tolerance=1.0e-6,
        )
        gas_density = np.ones(shape, dtype=np.float64)
        interface = flags == 1
        curvature = geometry.curvature.numpy().astype(np.float64)
        gas_density[interface] -= 6.0 * gamma * curvature[interface]
        initial_density = 1.0 + 6.0 * gamma / radius
        initial = np.empty(shape + (10,), dtype=np.float64)
        initial[...] = _equilibrium_moments(initial_density, np.zeros(3))
        om = initial.copy()
        hybrid = initial.copy()
        om_active = np.isin(flags, (1, 2))
        hybrid_active = coverage.hydrodynamic_active
        om_peak = 0.0
        hybrid_peak = 0.0

        for _ in range(20):
            om = _only_missing_stream_collide(
                om, flags, gas_density, model.shear_omega
            )
            boundary_velocity = fsl_extrapolated_boundary_velocity(
                hybrid,
                coverage,
                periodic=model.periodic,
                no_support_policy="only_missing",
            )
            streamed = fsl_stream_moments(
                hybrid,
                flags,
                coverage,
                gas_density,
                boundary_velocity,
                model.shear_omega,
                periodic=model.periodic,
                no_support_policy="only_missing",
            )
            hybrid_next = hybrid.copy()
            hybrid_next[hybrid_active] = collide_central_moments(
                streamed.moments[hybrid_active], model.shear_omega
            )
            hybrid = hybrid_next
            om_speed = np.linalg.norm(om[..., 1:4] / om[..., 0, None], axis=-1)
            hybrid_speed = np.linalg.norm(
                hybrid[..., 1:4] / hybrid[..., 0, None], axis=-1
            )
            om_peak = max(om_peak, float(np.max(om_speed[om_active])))
            hybrid_peak = max(
                hybrid_peak, float(np.max(hybrid_speed[hybrid_active]))
            )

        self.assertTrue(np.isfinite(om_peak))
        self.assertTrue(np.isfinite(hybrid_peak))
        self.assertLess(om_peak, 0.02)
        self.assertLess(hybrid_peak, 0.02)
        self.assertLess(hybrid_peak, 0.7 * om_peak)

    def test_uniform_home_equilibrium_is_fixed_for_every_link_distance(self) -> None:
        rho = 1.03
        velocity = np.asarray((0.017, -0.011, 0.008), dtype=np.float64)
        moments = _equilibrium_moments(rho, velocity)
        populations = reconstruct_populations(moments)

        for q in range(1, 27):
            for delta in (0.0, 0.17, 0.5, 0.83, 1.0):
                actual = fsl_boundary_population(
                    moments,
                    moments,
                    q,
                    delta,
                    rho,
                    1.1,
                    boundary_velocity=velocity,
                )
                self.assertAlmostEqual(actual, populations[q], places=14)

    def test_full_closure_matches_expanded_table_one_formula(self) -> None:
        local = _equilibrium_moments(1.07, np.asarray((0.02, -0.01, 0.015)))
        interior = _equilibrium_moments(0.98, np.asarray((-0.01, 0.025, 0.005)))
        local[4:10] += np.asarray((0.013, -0.009, 0.004, 0.006, -0.003, 0.002))
        interior[4:10] += np.asarray((-0.005, 0.008, -0.002, 0.001, 0.004, -0.006))
        q = 19
        opposite = int(D3Q27_OPPOSITE[q])
        delta = 0.37
        omega = 1.3
        rho_boundary = 1.012
        velocity_boundary = np.asarray((0.006, -0.004, 0.009))
        strain = np.asarray(
            ((0.003, -0.002, 0.001), (-0.002, -0.004, 0.0005), (0.001, 0.0005, 0.002))
        )

        local_populations = reconstruct_populations(local)
        interior_populations = reconstruct_populations(interior)
        c = D3Q27_DIRECTIONS[q]
        weight = float(D3Q27_WEIGHTS[q])
        local_velocity = local[1:4] / local[0]
        even_equilibrium = weight * local[0] * (
            1.0
            + 4.5 * np.dot(c, local_velocity) ** 2
            - 1.5 * np.dot(local_velocity, local_velocity)
        )
        even_nonequilibrium = (
            0.5 * (local_populations[q] + local_populations[opposite])
            - even_equilibrium
        )
        boundary_even = weight * rho_boundary * (
            1.0
            + 4.5 * np.dot(c, velocity_boundary) ** 2
            - 1.5 * np.dot(velocity_boundary, velocity_boundary)
        )
        coefficients = fsl_boundary_coefficients(delta, omega, weight)
        expected = (
            coefficients.a0 * local_populations[opposite]
            + coefficients.opposite_a0 * local_populations[q]
            + coefficients.a1 * interior_populations[opposite]
            + coefficients.nonequilibrium * even_nonequilibrium
            + boundary_even
            + coefficients.strain * float(c @ strain @ c)
        )

        actual = fsl_boundary_population(
            local,
            interior,
            q,
            delta,
            rho_boundary,
            omega,
            boundary_velocity=velocity_boundary,
            boundary_strain=strain,
        )
        self.assertAlmostEqual(actual, expected, places=14)

    def test_strain_term_is_disabled_only_when_explicitly_omitted(self) -> None:
        moments = _equilibrium_moments(1.0, np.zeros(3))
        q = 7
        strain = np.asarray(((0.01, 0.004, 0.0), (0.004, -0.003, 0.0), (0.0, 0.0, 0.0)))
        without_strain = fsl_boundary_population(moments, moments, q, 0.4, 1.0, 1.2)
        with_strain = fsl_boundary_population(
            moments,
            moments,
            q,
            0.4,
            1.0,
            1.2,
            boundary_strain=strain,
        )
        coefficients = fsl_boundary_coefficients(0.4, 1.2, D3Q27_WEIGHTS[q])
        expected_delta = coefficients.strain * float(
            D3Q27_DIRECTIONS[q] @ strain @ D3Q27_DIRECTIONS[q]
        )
        self.assertAlmostEqual(with_strain - without_strain, expected_delta, places=15)

    def test_invalid_inputs_are_rejected(self) -> None:
        moments = _equilibrium_moments(1.0, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "fraction"):
            fsl_boundary_coefficients(1.01, 1.0, D3Q27_WEIGHTS[1])
        with self.assertRaisesRegex(ValueError, "shear_omega"):
            fsl_boundary_coefficients(0.5, 2.0, D3Q27_WEIGHTS[1])
        with self.assertRaisesRegex(ValueError, "non-rest"):
            fsl_boundary_population(moments, moments, 0, 0.5, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            fsl_boundary_population(
                moments,
                moments,
                1,
                0.5,
                1.0,
                1.0,
                boundary_strain=np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
            )

    def test_warp_cpu_matches_random_reference(self) -> None:
        self._warp_matches_random_reference("cpu")

    def test_warp_cpu_grid_hybrid_matches_reference(self) -> None:
        self._warp_grid_hybrid_matches_reference("cpu")

    def test_warp_cpu_strain_projection_matches_reference(self) -> None:
        self._warp_strain_projection_matches_reference("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_matches_random_reference(self) -> None:
        self._warp_matches_random_reference("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_grid_hybrid_matches_reference(self) -> None:
        self._warp_grid_hybrid_matches_reference("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_strain_projection_matches_reference(self) -> None:
        self._warp_strain_projection_matches_reference("cuda:0")


if __name__ == "__main__":
    unittest.main()
