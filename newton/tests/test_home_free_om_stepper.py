# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Official only-missing comparator for the shared geometric HOME-Free path."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeFslResearchStepper,
    HomeFreeOmResearchStepper,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof import (
    geometric_advection_kernels,
)

from newton.tests.test_home_free_dynamic_laplace import _sampled_sphere_vof


@dataclass(frozen=True)
class _DynamicLaplaceResult:
    fill: np.ndarray
    moments: np.ndarray
    relative_volume_error: float
    relative_mass_error: float
    peak_speed: float


@dataclass(frozen=True)
class _MovingLaplaceResult:
    fill: np.ndarray
    moments: np.ndarray
    flags_by_step: tuple[np.ndarray, ...]
    fill_by_step: tuple[np.ndarray, ...]
    moments_by_step: tuple[np.ndarray, ...]
    relative_volume_error: float
    relative_mass_error: float
    centroid_error: float
    shape_l1_error: float
    peak_velocity_error: float
    separation_closure_count: int
    absolute_snapped_volume_delta: float
    maximum_initial_divergence: float
    maximum_projected_divergence: float
    maximum_projection_iterations: int
    bounded_face_count: int
    maximum_flux_bound_correction: float


def _run_dynamic_laplace(
    device: str,
    *,
    backend: str,
    steps: int = 20,
) -> _DynamicLaplaceResult:
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
    initial_volume = float(np.sum(fill, dtype=np.float64))
    initial_mass = float(np.sum(free_in.mass.numpy(), dtype=np.float64))
    if backend == "om":
        stepper = HomeFreeOmResearchStepper(
            model,
            advection_axes=(0, 1, 2),
            surface_tension=surface_tension,
        )
    elif backend == "fsl":
        stepper = HomeFreeFslResearchStepper(
            model,
            advection_axes=(0, 1, 2),
            surface_tension=surface_tension,
            no_support_policy="only_missing",
        )
    else:
        raise ValueError(f"unknown HOME-Free backend {backend!r}")

    peak_speed = 0.0
    for step in range(steps):
        diagnostics = stepper.step(fluid_in, free_in, fluid_out, free_out)
        expected_order = (0, 1, 2) if step % 2 == 0 else (2, 1, 0)
        if diagnostics.split_order != expected_order:
            raise AssertionError(
                f"{backend} split order {diagnostics.split_order} != {expected_order}"
            )
        if backend == "om":
            if diagnostics.invalid_active_count:
                raise AssertionError("OM active ownership became invalid")
            if diagnostics.solver.invalid_cell_count:
                raise AssertionError("OM solver produced an invalid cell")
            if diagnostics.solver.direct_liquid_gas_link_count:
                raise AssertionError("OM topology exposed a direct liquid-gas link")
            if diagnostics.solver.invalid_gas_density_count:
                raise AssertionError("OM capillary gas density became invalid")
        else:
            invalid = (
                diagnostics.invalid_geometry_count
                + diagnostics.invalid_velocity_count
                + diagnostics.invalid_stream_count
                + diagnostics.invalid_collision_count
                + diagnostics.invalid_commit_count
            )
            if invalid:
                raise AssertionError(f"FSL transaction produced {invalid} errors")
        moments = fluid_out.moments.numpy().reshape(10, -1)
        active = np.isin(free_out.flags.numpy(), (1, 2)).reshape(-1)
        speed = np.linalg.norm(
            moments[1:4, active].T / moments[0, active, None], axis=1
        )
        peak_speed = max(peak_speed, float(np.max(speed)))
        free_out.validate(fluid_out, require_mass_fill_consistency=False)
        fluid_in, fluid_out = fluid_out, fluid_in
        free_in, free_out = free_out, free_in

    final_fill = free_in.fill_level.numpy().copy()
    final_moments = fluid_in.moments.numpy().copy()
    final_volume = float(np.sum(final_fill, dtype=np.float64))
    final_mass = float(np.sum(free_in.mass.numpy(), dtype=np.float64))
    return _DynamicLaplaceResult(
        fill=final_fill,
        moments=final_moments,
        relative_volume_error=abs(final_volume - initial_volume) / initial_volume,
        relative_mass_error=abs(final_mass - initial_mass) / initial_mass,
        peak_speed=peak_speed,
    )


def _run_moving_laplace(
    device: str,
    *,
    backend: str,
    steps: int = 20,
    endpoint_tolerance: float = 4.0e-7,
    surface_tension: float = 0.001,
    project_courant: bool = True,
) -> _MovingLaplaceResult:
    shape = (17, 17, 17)
    radius = 5.25
    velocity = np.asarray((0.008, -0.005, 0.003), dtype=np.float64)
    initial_center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    fill = _sampled_sphere_vof(shape, radius, center=initial_center)
    model = HomeLbmModel(
        fluid_grid_res=shape,
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        reference_density=1.0,
        kinematic_viscosity=0.5,
        periodic=(True, True, True),
        device=device,
    )
    fluid_in = HomeLbmState(model)
    fluid_out = HomeLbmState(model)
    initial_density = 1.0 + 6.0 * surface_tension / radius
    HomeLbmSolver(model).initialize_uniform_lattice(
        fluid_in,
        rho=initial_density,
        velocity=tuple(float(component) for component in velocity),
    )
    free_in = HomeFreeState(model)
    free_out = HomeFreeState(model)
    free_in.initialize_from_fill_level(fluid_in, fill)
    initial_volume = float(np.sum(fill, dtype=np.float64))
    initial_mass = float(np.sum(free_in.mass.numpy(), dtype=np.float64))
    if backend == "om":
        stepper = HomeFreeOmResearchStepper(
            model,
            advection_axes=(0, 1, 2),
            surface_tension=surface_tension,
            project_courant=project_courant,
        )
    elif backend == "fsl":
        stepper = HomeFreeFslResearchStepper(
            model,
            advection_axes=(0, 1, 2),
            surface_tension=surface_tension,
            no_support_policy="only_missing",
            project_courant=project_courant,
        )
    else:
        raise ValueError(f"unknown HOME-Free backend {backend!r}")
    stepper.topology_resolver.endpoint_tolerance = endpoint_tolerance

    peak_velocity_error = 0.0
    separation_closure_count = 0
    absolute_snapped_volume_delta = 0.0
    maximum_initial_divergence = 0.0
    maximum_projected_divergence = 0.0
    maximum_projection_iterations = 0
    bounded_face_count = 0
    maximum_flux_bound_correction = 0.0
    flags_by_step: list[np.ndarray] = []
    fill_by_step: list[np.ndarray] = []
    moments_by_step: list[np.ndarray] = []
    for step in range(steps):
        try:
            diagnostics = stepper.step(fluid_in, free_in, fluid_out, free_out)
        except Exception as error:
            raise RuntimeError(
                f"{backend} moving Laplace transaction failed at step {step + 1}"
            ) from error
        expected_order = (0, 1, 2) if step % 2 == 0 else (2, 1, 0)
        if diagnostics.split_order != expected_order:
            raise AssertionError(
                f"{backend} split order {diagnostics.split_order} != {expected_order}"
            )
        for _, topology in diagnostics.topology_by_axis:
            separation_closure_count += topology.separation_closure_cell_count
            absolute_snapped_volume_delta += abs(topology.snapped_volume_delta)
        for _, advection in diagnostics.advection_by_axis:
            bounded_face_count += advection.bounded_face_count
            maximum_flux_bound_correction = max(
                maximum_flux_bound_correction,
                advection.maximum_bound_correction,
            )
        if project_courant:
            if diagnostics.projection is None:
                raise AssertionError("requested Courant projection was not executed")
            maximum_initial_divergence = max(
                maximum_initial_divergence,
                diagnostics.projection.initial_max_divergence,
            )
            maximum_projected_divergence = max(
                maximum_projected_divergence,
                diagnostics.projection.projected_max_divergence,
            )
            maximum_projection_iterations = max(
                maximum_projection_iterations,
                diagnostics.projection.iteration_count,
            )
        moments = fluid_out.moments.numpy().reshape(10, -1)
        active = np.isin(free_out.flags.numpy(), (1, 2)).reshape(-1)
        actual_velocity = moments[1:4, active].T / moments[0, active, None]
        peak_velocity_error = max(
            peak_velocity_error,
            float(np.max(np.linalg.norm(actual_velocity - velocity, axis=1))),
        )
        free_out.validate(fluid_out, require_mass_fill_consistency=False)
        flags_by_step.append(free_out.flags.numpy().copy())
        fill_by_step.append(free_out.fill_level.numpy().copy())
        moments_by_step.append(fluid_out.moments.numpy().copy())
        fluid_in, fluid_out = fluid_out, fluid_in
        free_in, free_out = free_out, free_in

    final_fill = free_in.fill_level.numpy().copy()
    final_moments = fluid_in.moments.numpy().copy()
    final_volume = float(np.sum(final_fill, dtype=np.float64))
    final_mass = float(np.sum(free_in.mass.numpy(), dtype=np.float64))
    expected_center = initial_center + velocity * steps
    expected_fill = _sampled_sphere_vof(
        shape, radius, center=expected_center
    )
    coordinates = np.indices(shape, dtype=np.float64)
    actual_center = np.asarray(
        [
            np.sum(coordinates[axis] * final_fill, dtype=np.float64)
            / final_volume
            for axis in range(3)
        ]
    )
    return _MovingLaplaceResult(
        fill=final_fill,
        moments=final_moments,
        flags_by_step=tuple(flags_by_step),
        fill_by_step=tuple(fill_by_step),
        moments_by_step=tuple(moments_by_step),
        relative_volume_error=abs(final_volume - initial_volume) / initial_volume,
        relative_mass_error=abs(final_mass - initial_mass) / initial_mass,
        centroid_error=float(np.linalg.norm(actual_center - expected_center)),
        shape_l1_error=float(
            np.sum(np.abs(final_fill - expected_fill), dtype=np.float64)
            / initial_volume
        ),
        peak_velocity_error=peak_velocity_error,
        separation_closure_count=separation_closure_count,
        absolute_snapped_volume_delta=absolute_snapped_volume_delta,
        maximum_initial_divergence=maximum_initial_divergence,
        maximum_projected_divergence=maximum_projected_divergence,
        maximum_projection_iterations=maximum_projection_iterations,
        bounded_face_count=bounded_face_count,
        maximum_flux_bound_correction=maximum_flux_bound_correction,
    )


class TestHomeFreeOmStepper(unittest.TestCase):
    def _run_endpoint_momentum_conditioning(self, device: str) -> None:
        shape = (3, 2, 2)
        endpoint = np.float32(4.0e-7)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        source = HomeLbmState(model)
        candidate = HomeLbmState(model)
        solver = HomeLbmSolver(model)
        solver.initialize_uniform_lattice(source, velocity=(0.02, 0.0, 0.0))
        candidate.copy_from(source)

        source_mass = np.zeros(shape, dtype=np.float32)
        transported_mass = np.zeros(shape, dtype=np.float32)
        transported_fill = np.zeros(shape, dtype=np.float32)
        transported_momentum = np.zeros(shape + (3,), dtype=np.float32)
        flags = np.zeros(shape, dtype=np.int32)
        endpoint_cell = (0, 0, 0)
        regular_cell = (1, 0, 0)
        for index, fill in ((endpoint_cell, endpoint), (regular_cell, 0.25)):
            source_mass[index] = fill
            transported_mass[index] = fill
            transported_fill[index] = fill
            flags[index] = 1
        transported_momentum[endpoint_cell] = endpoint * np.asarray(
            (0.02, 2.6, 0.0), dtype=np.float32
        )
        transported_momentum[regular_cell] = 0.25 * np.asarray(
            (0.02, 0.03, 0.0), dtype=np.float32
        )
        invalid = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            geometric_advection_kernels.apply_geometric_momentum_transport_kernel,
            dim=shape,
            inputs=[
                source.moments,
                wp.array(source_mass, dtype=float, device=device),
                candidate.moments,
                wp.array(transported_mass, dtype=float, device=device),
                wp.array(transported_fill, dtype=float, device=device),
                wp.array(transported_momentum, dtype=wp.vec3, device=device),
                wp.array(flags, dtype=wp.int32, device=device),
                invalid,
                shape[1],
                shape[2],
                int(np.prod(shape)),
            ],
            device=device,
        )
        self.assertEqual(int(invalid.numpy()[0]), 0)
        moments = candidate.moments.numpy().reshape(10, *shape)
        endpoint_velocity = moments[1:4, *endpoint_cell] / moments[(0, *endpoint_cell)]
        regular_velocity = moments[1:4, *regular_cell] / moments[(0, *regular_cell)]
        np.testing.assert_allclose(
            endpoint_velocity,
            (0.02, float(endpoint) * 2.6, 0.0),
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            regular_velocity, (0.02, 0.0075, 0.0), rtol=0.0, atol=2.0e-7
        )

    def test_endpoint_interface_does_not_amplify_velocity(self) -> None:
        self._run_endpoint_momentum_conditioning("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_endpoint_interface_does_not_amplify_velocity(self) -> None:
        self._run_endpoint_momentum_conditioning("cuda:0")

    def _run_mass_weighted_body_force(self, device: str) -> None:
        shape = (9, 9, 9)
        acceleration = np.asarray((1.0e-5, -2.0e-5, 3.0e-5), dtype=np.float32)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            body_acceleration=tuple(float(value) for value in acceleration),
            periodic=(True, True, True),
            device=device,
        )
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid_in)
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        free_in.initialize_from_fill_level(
            fluid_in,
            _sampled_sphere_vof(shape, 2.75),
        )
        stepper = HomeFreeOmResearchStepper(
            model,
            advection_axes=(0, 1, 2),
        )

        stepper.step(fluid_in, free_in, fluid_out, free_out)

        lattice_acceleration = np.asarray(
            model.lattice_acceleration, dtype=np.float32
        )
        expected = free_in.mass.numpy()[..., None] * lattice_acceleration
        np.testing.assert_array_equal(
            stepper._body_force_density.numpy(), expected
        )
        free_out.validate(fluid_out, require_mass_fill_consistency=False)

    def _assert_moving_laplace_acceptance(
        self, result: _MovingLaplaceResult
    ) -> None:
        self.assertLess(result.relative_volume_error, 5.0e-7)
        self.assertLess(result.relative_mass_error, 2.0e-5)
        self.assertLess(result.centroid_error, 2.0e-3)
        self.assertLess(result.shape_l1_error, 2.5e-3)
        self.assertLess(result.peak_velocity_error, 2.5e-4)
        self.assertGreater(result.separation_closure_count, 0)
        self.assertLess(result.absolute_snapped_volume_delta, 5.0e-4)
        self.assertGreater(result.maximum_initial_divergence, 2.0e-7)
        self.assertLessEqual(result.maximum_projected_divergence, 2.0e-8)
        self.assertGreater(result.maximum_projection_iterations, 0)
        self.assertLessEqual(result.maximum_flux_bound_correction, 4.0e-6)

    def test_failed_sweep_is_fully_transactional(self) -> None:
        shape = (12, 3, 2)
        speed = 0.02
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device="cpu",
        )
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(
            fluid_in, rho=1.0, velocity=(speed, 0.0, 0.0)
        )
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.75
        fill[1:5] = 1.0
        fill[5] = 0.25
        free_in.initialize_from_fill_level(fluid_in, fill)
        stepper = HomeFreeOmResearchStepper(model, advection_axis=0)
        for _ in range(13):
            stepper.step(fluid_in, free_in, fluid_out, free_out)
            fluid_in, fluid_out = fluid_out, fluid_in
            free_in, free_out = free_out, free_in

        source_moments = fluid_in.moments.numpy().copy()
        source_fill = free_in.fill_level.numpy().copy()
        source_flags = free_in.flags.numpy().copy()
        destination_moments = fluid_out.moments.numpy().copy()
        destination_fill = free_out.fill_level.numpy().copy()
        destination_flags = free_out.flags.numpy().copy()
        self.assertIsNotNone(stepper._remapper)
        previous_active = stepper._remapper.previous_active.numpy().copy()
        split_parity = stepper._split_parity
        invalid_courant = np.full(
            (shape[0] + 1, shape[1], shape[2]), speed, dtype=np.float32
        )
        invalid_courant[-1] = speed * 2.0
        with self.assertRaisesRegex(RuntimeError, "invalid faces"):
            stepper.step(
                fluid_in,
                free_in,
                fluid_out,
                free_out,
                wp.array(invalid_courant, dtype=float, device="cpu"),
            )

        np.testing.assert_array_equal(fluid_in.moments.numpy(), source_moments)
        np.testing.assert_array_equal(free_in.fill_level.numpy(), source_fill)
        np.testing.assert_array_equal(free_in.flags.numpy(), source_flags)
        np.testing.assert_array_equal(fluid_out.moments.numpy(), destination_moments)
        np.testing.assert_array_equal(free_out.fill_level.numpy(), destination_fill)
        np.testing.assert_array_equal(free_out.flags.numpy(), destination_flags)
        self.assertIsNotNone(stepper._remapper)
        np.testing.assert_array_equal(
            stepper._remapper.previous_active.numpy(), previous_active
        )
        self.assertEqual(stepper._split_parity, split_parity)
        diagnostics = stepper.step(fluid_in, free_in, fluid_out, free_out)
        self.assertEqual(diagnostics.split_order, (0,))

    def test_cpu_dynamic_laplace_acceptance(self) -> None:
        result = _run_dynamic_laplace("cpu", backend="om")
        self.assertLess(result.relative_volume_error, 1.5e-4)
        self.assertLess(result.relative_mass_error, 1.5e-4)
        self.assertLess(result.peak_speed, 1.5e-4)

    def test_cpu_body_force_uses_transported_liquid_mass(self) -> None:
        self._run_mass_weighted_body_force("cpu")

    def test_cpu_om_fsl_same_initial_comparison(self) -> None:
        om = _run_dynamic_laplace("cpu", backend="om")
        fsl = _run_dynamic_laplace("cpu", backend="fsl")
        self.assertLess(om.relative_volume_error, fsl.relative_volume_error)
        self.assertLess(om.peak_speed, fsl.peak_speed)

    def test_cpu_moving_laplace_acceptance_and_tradeoff(self) -> None:
        om = _run_moving_laplace("cpu", backend="om")
        fsl = _run_moving_laplace("cpu", backend="fsl")
        self._assert_moving_laplace_acceptance(om)
        self._assert_moving_laplace_acceptance(fsl)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_dynamic_laplace_matches_cpu(self) -> None:
        expected = _run_dynamic_laplace("cpu", backend="om")
        actual = _run_dynamic_laplace("cuda:0", backend="om")
        np.testing.assert_allclose(
            actual.fill, expected.fill, rtol=0.0, atol=3.0e-6
        )
        np.testing.assert_allclose(
            actual.moments, expected.moments, rtol=3.0e-5, atol=3.0e-6
        )
        np.testing.assert_allclose(
            (
                actual.relative_volume_error,
                actual.relative_mass_error,
                actual.peak_speed,
            ),
            (
                expected.relative_volume_error,
                expected.relative_mass_error,
                expected.peak_speed,
            ),
            rtol=2.0e-3,
            atol=3.0e-7,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_body_force_uses_transported_liquid_mass(self) -> None:
        self._run_mass_weighted_body_force("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_moving_laplace_matches_cpu(self) -> None:
        for backend in ("om", "fsl"):
            expected = _run_moving_laplace("cpu", backend=backend)
            actual = _run_moving_laplace("cuda:0", backend=backend)
            repeated = _run_moving_laplace("cuda:0", backend=backend)
            self._assert_moving_laplace_acceptance(actual)
            np.testing.assert_array_equal(repeated.fill, actual.fill)
            np.testing.assert_array_equal(repeated.moments, actual.moments)
            self.assertEqual(
                repeated.separation_closure_count,
                actual.separation_closure_count,
            )

            fill_error = np.abs(actual.fill - expected.fill)
            moment_error = np.abs(actual.moments - expected.moments)
            first_flag_difference = -1
            first_flag_difference_count = 0
            first_fill_difference = 0.0
            first_moment_difference = 0.0
            for step, (expected_flags, actual_flags) in enumerate(
                zip(expected.flags_by_step, actual.flags_by_step, strict=True),
                start=1,
            ):
                mismatch_count = int(np.count_nonzero(expected_flags != actual_flags))
                if mismatch_count:
                    first_flag_difference = step
                    first_flag_difference_count = mismatch_count
                    first_fill_difference = float(
                        np.max(
                            np.abs(
                                expected.fill_by_step[step - 1]
                                - actual.fill_by_step[step - 1]
                            )
                        )
                    )
                    first_moment_difference = float(
                        np.max(
                            np.abs(
                                expected.moments_by_step[step - 1]
                                - actual.moments_by_step[step - 1]
                            )
                        )
                    )
                    break
            self.assertLess(float(np.max(fill_error)), 1.5e-4)
            self.assertLess(float(np.mean(fill_error)), 5.0e-7)
            self.assertLess(
                float(np.max(moment_error)),
                3.0e-4,
                (
                    f"backend={backend}, first_flag_step={first_flag_difference}, "
                    f"flag_mismatches={first_flag_difference_count}, "
                    f"first_fill_max={first_fill_difference:.9e}, "
                    f"first_moment_max={first_moment_difference:.9e}"
                ),
            )
            self.assertLess(float(np.mean(moment_error)), 1.0e-6)
            self.assertLess(
                abs(actual.relative_volume_error - expected.relative_volume_error),
                2.0e-8,
            )
            self.assertLess(
                abs(actual.relative_mass_error - expected.relative_mass_error),
                1.0e-6,
            )
            self.assertLess(
                abs(actual.centroid_error - expected.centroid_error), 1.0e-5
            )
            self.assertLess(
                abs(actual.shape_l1_error - expected.shape_l1_error), 1.0e-6
            )
            self.assertLess(
                abs(actual.peak_velocity_error - expected.peak_velocity_error),
                1.0e-6,
            )
            self.assertLess(
                abs(
                    actual.absolute_snapped_volume_delta
                    - expected.absolute_snapped_volume_delta
                ),
                5.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
