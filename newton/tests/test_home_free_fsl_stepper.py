# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-topology transactions joining geometric PLIC and Bogner FSL."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.fsl_stepper import (
    HomeFreeFslResearchStepper,
)


class TestHomeFreeFslStepper(unittest.TestCase):
    def _run_multiaxis_dynamic_topology(
        self, device: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (12, 12, 2)
        velocity = (0.02, 0.01, 0.0)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device=device,
        )
        solver = HomeLbmSolver(model)
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(
            fluid_in, rho=1.0, velocity=velocity
        )
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        interface_values = {0: 0.01, 1: 0.5, 6: 0.99, 7: 0.5}
        for i in range(shape[0]):
            for j in range(shape[1]):
                phase = (i + j) % shape[0]
                fill[i, j] = interface_values.get(
                    phase, 1.0 if 2 <= phase <= 5 else 0.0
                )
        free_in.initialize_from_fill_level(fluid_in, fill)
        stepper = HomeFreeFslResearchStepper(model, advection_axes=(0, 1))

        diagnostics = stepper.step(fluid_in, free_in, fluid_out, free_out)

        self.assertEqual(diagnostics.split_order, (0, 1))
        self.assertEqual(
            tuple(axis for axis, _ in diagnostics.topology_by_axis), (0, 1)
        )
        first_topology = diagnostics.topology_by_axis[0][1]
        phase_layer_size = shape[0] * shape[2]
        self.assertEqual(first_topology.gas_to_interface_count, phase_layer_size)
        self.assertEqual(
            first_topology.liquid_to_interface_count, phase_layer_size
        )
        self.assertEqual(diagnostics.topology_by_axis[1][1].snapped_cell_count, 0)
        self.assertEqual(
            diagnostics.topology_by_axis[1][1].snapped_volume_delta, 0.0
        )
        self.assertLess(
            abs(
                float(np.sum(free_out.fill_level.numpy(), dtype=np.float64))
                - float(np.sum(fill, dtype=np.float64))
            ),
            2.0e-5,
        )
        free_out.validate(fluid_out, require_mass_fill_consistency=False)
        return (
            free_out.fill_level.numpy().copy(),
            free_out.flags.numpy().copy(),
            fluid_out.moments.numpy().copy(),
        )

    def _run_dynamic_topology_translation(
        self, device: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (12, 3, 2)
        speed = 0.02
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device=device,
        )
        solver = HomeLbmSolver(model)
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(
            fluid_in, rho=1.0, velocity=(speed, 0.0, 0.0)
        )
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.01
        fill[1:5] = 1.0
        fill[5] = 0.99
        free_in.initialize_from_fill_level(fluid_in, fill)
        stepper = HomeFreeFslResearchStepper(model, advection_axis=0)

        diagnostics = stepper.step(fluid_in, free_in, fluid_out, free_out)

        self.assertEqual(len(diagnostics.topology_by_axis), 1)
        axis, topology = diagnostics.topology_by_axis[0]
        self.assertEqual(axis, 0)
        cross_section = shape[1] * shape[2]
        self.assertEqual(topology.gas_to_interface_count, cross_section)
        self.assertEqual(topology.liquid_to_interface_count, cross_section)
        self.assertEqual(topology.interface_to_gas_count, cross_section)
        self.assertEqual(topology.interface_to_liquid_count, cross_section)
        self.assertEqual(topology.snapped_cell_count, 0)
        expected_fill = np.zeros(shape, dtype=np.float32)
        expected_fill[1] = 0.99
        expected_fill[2:6] = 1.0
        expected_fill[6] = 0.01
        np.testing.assert_allclose(
            free_out.fill_level.numpy(), expected_fill, rtol=0.0, atol=2.0e-7
        )
        expected_flags = np.zeros(shape, dtype=np.int32)
        expected_flags[1] = int(HomeFreeCellFlag.INTERFACE)
        expected_flags[2:6] = int(HomeFreeCellFlag.LIQUID)
        expected_flags[6] = int(HomeFreeCellFlag.INTERFACE)
        np.testing.assert_array_equal(free_out.flags.numpy(), expected_flags)
        moments = fluid_out.moments.numpy().reshape(10, -1)
        rho = moments[0].reshape(shape)
        velocity_x = (moments[1] / moments[0]).reshape(shape)
        np.testing.assert_allclose(rho[6], 1.0, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(
            velocity_x[6], speed, rtol=0.0, atol=4.0e-7
        )
        np.testing.assert_allclose(
            free_out.mass.numpy(),
            free_out.fill_level.numpy(),
            rtol=0.0,
            atol=2.0e-7,
        )
        self.assertAlmostEqual(
            float(np.sum(free_out.mass.numpy(), dtype=np.float64)),
            float(np.sum(fill, dtype=np.float64)),
            delta=2.0e-7,
        )
        free_out.validate(fluid_out, require_mass_fill_consistency=False)
        return (
            free_out.fill_level.numpy().copy(),
            free_out.flags.numpy().copy(),
            fluid_out.moments.numpy().copy(),
        )

    def _run_multiaxis_oblique_translation(
        self, device: str
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (16, 16, 4)
        velocity = (0.01, 0.007, 0.004)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device=device,
        )
        solver = HomeLbmSolver(model)
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(fluid_in, rho=1.0, velocity=velocity)
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        for i in range(shape[0]):
            for j in range(shape[1]):
                phase = (i + j) % shape[0]
                value = 0.0
                if phase == 2 or phase == 11:
                    value = 0.25
                elif phase == 3 or phase == 10:
                    value = 0.75
                elif 4 <= phase <= 9:
                    value = 1.0
                fill[i, j] = value
        initial_fill = fill.copy()
        free_in.initialize_from_fill_level(fluid_in, fill)
        initial_flags = free_in.flags.numpy().copy()
        initial_volume = float(np.sum(fill, dtype=np.float64))
        stepper = HomeFreeFslResearchStepper(
            model,
            advection_axes=(0, 1, 2),
            gas_density=1.0,
        )

        for step in range(4):
            diagnostics = stepper.step(fluid_in, free_in, fluid_out, free_out)
            expected_order = (0, 1, 2) if step % 2 == 0 else (2, 1, 0)
            self.assertEqual(diagnostics.split_order, expected_order)
            self.assertEqual(
                tuple(axis for axis, _ in diagnostics.face_courant_by_axis),
                expected_order,
            )
            self.assertEqual(diagnostics.invalid_geometry_count, 0)
            self.assertEqual(diagnostics.invalid_velocity_count, 0)
            self.assertEqual(diagnostics.invalid_stream_count, 0)
            self.assertEqual(diagnostics.invalid_collision_count, 0)
            self.assertEqual(diagnostics.invalid_commit_count, 0)
            self.assertEqual(diagnostics.fallback_link_count, 0)
            np.testing.assert_array_equal(free_out.flags.numpy(), initial_flags)
            self.assertLess(
                abs(
                    float(np.sum(free_out.fill_level.numpy(), dtype=np.float64))
                    - initial_volume
                ),
                8.0e-5,
            )
            free_out.validate(fluid_out, require_mass_fill_consistency=False)
            fluid_in, fluid_out = fluid_out, fluid_in
            free_in, free_out = free_out, free_in

        final_fill = free_in.fill_level.numpy().copy()
        self.assertGreater(float(np.max(np.abs(final_fill - initial_fill))), 1.0e-3)
        return final_fill, fluid_in.moments.numpy().copy()

    def _run_planar_translation(self, device: str) -> None:
        shape = (12, 3, 2)
        rho = 1.017
        speed = 0.02
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device=device,
        )
        solver = HomeLbmSolver(model)
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(
            fluid_in, rho=rho, velocity=(speed, 0.0, 0.0)
        )
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.75
        fill[1:5] = 1.0
        fill[5] = 0.25
        free_in.initialize_from_fill_level(fluid_in, fill)
        stepper = HomeFreeFslResearchStepper(
            model, advection_axis=0, gas_density=rho
        )
        initial_volume = float(np.sum(fill, dtype=np.float64))
        fresh_total = 0
        dead_total = 0
        maximum_face_courant_error = 0.0
        cumulative_trailing_courant = 0.0
        cumulative_leading_courant = 0.0

        for step in range(1, 21):
            diagnostics = stepper.step(
                fluid_in, free_in, fluid_out, free_out
            )
            fresh_total += diagnostics.transition.fresh_cell_count
            dead_total += diagnostics.transition.dead_cell_count
            self.assertEqual(diagnostics.invalid_geometry_count, 0)
            self.assertEqual(diagnostics.invalid_velocity_count, 0)
            self.assertEqual(diagnostics.invalid_stream_count, 0)
            self.assertEqual(diagnostics.invalid_collision_count, 0)
            self.assertEqual(diagnostics.invalid_commit_count, 0)
            self.assertEqual(diagnostics.fallback_link_count, 0)
            self.assertEqual(diagnostics.stress.invalid_normal_strain_count, 0)
            self.assertEqual(diagnostics.stress.invalid_boundary_strain_count, 0)
            self.assertEqual(diagnostics.stress.no_support_link_count, 0)
            self.assertIsNotNone(diagnostics.face_courant)
            actual_courant = stepper.face_courant_builder.face_courant.numpy()
            transported_faces = np.abs(actual_courant) > 0.0
            transported_courant = actual_courant[transported_faces]
            trailing_courant = actual_courant[1]
            leading_courant = actual_courant[5]
            self.assertLess(
                float(np.max(trailing_courant) - np.min(trailing_courant)),
                2.0e-7,
            )
            self.assertLess(
                float(np.max(leading_courant) - np.min(leading_courant)),
                2.0e-7,
            )
            cumulative_trailing_courant += float(np.mean(trailing_courant))
            cumulative_leading_courant += float(np.mean(leading_courant))
            maximum_face_courant_error = max(
                maximum_face_courant_error,
                float(np.max(np.abs(transported_courant - speed))),
            )
            actual_fill = free_out.fill_level.numpy()
            np.testing.assert_allclose(
                actual_fill[0],
                0.75 - cumulative_trailing_courant,
                rtol=0.0,
                atol=8.0e-7,
            )
            np.testing.assert_allclose(
                actual_fill[5],
                0.25 + cumulative_leading_courant,
                rtol=0.0,
                atol=8.0e-7,
            )
            self.assertLess(
                abs(float(np.sum(actual_fill, dtype=np.float64)) - initial_volume),
                2.0e-5,
            )
            moments = fluid_out.moments.numpy().reshape(10, -1)
            active = stepper.active.numpy().reshape(-1).astype(bool)
            np.testing.assert_allclose(
                moments[0, active], rho, rtol=3.0e-5, atol=3.0e-6
            )
            np.testing.assert_allclose(
                moments[1, active] / moments[0, active],
                speed,
                rtol=0.0,
                atol=4.0e-7,
            )
            free_out.validate(fluid_out, require_mass_fill_consistency=False)
            fluid_in, fluid_out = fluid_out, fluid_in
            free_in, free_out = free_out, free_in

        self.assertEqual(fresh_total, shape[1] * shape[2])
        self.assertEqual(dead_total, shape[1] * shape[2])
        self.assertLess(maximum_face_courant_error, 1.0e-6)

    def test_cpu_stable_classification_transaction(self) -> None:
        self._run_planar_translation("cpu")

    def test_cpu_dynamic_topology_transaction(self) -> None:
        self._run_dynamic_topology_translation("cpu")

    def test_failed_sweep_restores_transition_history_and_source_state(self) -> None:
        shape = (12, 3, 2)
        speed = 0.02
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device="cpu",
        )
        solver = HomeLbmSolver(model)
        fluid_in = HomeLbmState(model)
        fluid_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(
            fluid_in, rho=1.0, velocity=(speed, 0.0, 0.0)
        )
        free_in = HomeFreeState(model)
        free_out = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.75
        fill[1:5] = 1.0
        fill[5] = 0.25
        free_in.initialize_from_fill_level(fluid_in, fill)
        stepper = HomeFreeFslResearchStepper(model, advection_axis=0)
        for _ in range(13):
            stepper.step(fluid_in, free_in, fluid_out, free_out)
            fluid_in, fluid_out = fluid_out, fluid_in
            free_in, free_out = free_out, free_in

        moments_before = fluid_in.moments.numpy().copy()
        fill_before = free_in.fill_level.numpy().copy()
        flags_before = free_in.flags.numpy().copy()
        destination_moments_before = fluid_out.moments.numpy().copy()
        destination_fill_before = free_out.fill_level.numpy().copy()
        destination_flags_before = free_out.flags.numpy().copy()
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
        np.testing.assert_array_equal(fluid_in.moments.numpy(), moments_before)
        np.testing.assert_array_equal(free_in.fill_level.numpy(), fill_before)
        np.testing.assert_array_equal(free_in.flags.numpy(), flags_before)
        np.testing.assert_array_equal(
            fluid_out.moments.numpy(), destination_moments_before
        )
        np.testing.assert_array_equal(
            free_out.fill_level.numpy(), destination_fill_before
        )
        np.testing.assert_array_equal(
            free_out.flags.numpy(), destination_flags_before
        )

        diagnostics = stepper.step(
            fluid_in, free_in, fluid_out, free_out
        )
        self.assertEqual(diagnostics.transition.fresh_cell_count, shape[1] * shape[2])
        self.assertEqual(diagnostics.transition.dead_cell_count, shape[1] * shape[2])

    def test_cpu_multiaxis_oblique_translation(self) -> None:
        self._run_multiaxis_oblique_translation("cpu")

    def test_cpu_multiaxis_dynamic_topology(self) -> None:
        self._run_multiaxis_dynamic_topology("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_multiaxis_oblique_translation_matches_cpu(self) -> None:
        expected_fill, expected_moments = self._run_multiaxis_oblique_translation(
            "cpu"
        )
        actual_fill, actual_moments = self._run_multiaxis_oblique_translation(
            "cuda:0"
        )
        np.testing.assert_allclose(
            actual_fill, expected_fill, rtol=2.0e-5, atol=3.0e-6
        )
        np.testing.assert_allclose(
            actual_moments, expected_moments, rtol=3.0e-5, atol=3.0e-6
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_stable_classification_transaction(self) -> None:
        self._run_planar_translation("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_dynamic_topology_transaction_matches_cpu(self) -> None:
        expected = self._run_dynamic_topology_translation("cpu")
        actual = self._run_dynamic_topology_translation("cuda:0")
        np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=2.0e-7)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(
            actual[2], expected[2], rtol=3.0e-5, atol=3.0e-6
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_multiaxis_dynamic_topology_matches_cpu(self) -> None:
        expected = self._run_multiaxis_dynamic_topology("cpu")
        actual = self._run_multiaxis_dynamic_topology("cuda:0")
        np.testing.assert_allclose(actual[0], expected[0], rtol=2.0e-5, atol=3.0e-6)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(
            actual[2], expected[2], rtol=3.0e-5, atol=3.0e-6
        )


if __name__ == "__main__":
    unittest.main()
