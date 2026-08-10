# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Domain-level transaction for projected geometric HOME-Free VOF."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeGeometricDomain,
    HomeFreeRigidCoupling,
    HomeLbmModel,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


class TestHomeFreeGeometricDomain(unittest.TestCase):
    @staticmethod
    def _liquid_momentum_x(domain: HomeFreeGeometricDomain) -> float:
        moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        fill = domain.free_surface_state.fill_level.numpy().reshape(-1)
        committed = fill * moments[1]
        queued = domain.free_surface_state.excess_momentum.numpy().reshape(3, -1)[0]
        lattice = float(np.sum(committed + queued, dtype=np.float64))
        return lattice * domain.model.scaling.momentum_unit

    @staticmethod
    def _oblique_band(shape: tuple[int, int, int]) -> np.ndarray:
        fill = np.zeros(shape, dtype=np.float32)
        for i in range(shape[0]):
            for j in range(shape[1]):
                phase = (i + j) % shape[0]
                if phase == 2 or phase == 11:
                    value = 0.25
                elif phase == 3 or phase == 10:
                    value = 0.75
                elif 4 <= phase <= 9:
                    value = 1.0
                else:
                    value = 0.0
                fill[i, j] = value
        return fill

    def _run_oblique_transaction(
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
        domain = HomeFreeGeometricDomain(model, project_courant=True)
        self.assertIs(domain.solver, domain.geometric_stepper.solver)
        fill = self._oblique_band(shape)
        initial_volume = float(np.sum(fill, dtype=np.float64))
        domain.initialize_uniform_lattice(fill, velocity=velocity)

        for step in range(4):
            domain.step(model.time_step)
            diagnostics = domain.last_geometric_diagnostics
            assert diagnostics is not None
            expected_order = (0, 1, 2) if step % 2 == 0 else (2, 1, 0)
            self.assertEqual(diagnostics.split_order, expected_order)
            self.assertIsNotNone(diagnostics.projection)
            self.assertLessEqual(
                diagnostics.projection.projected_max_divergence, 2.0e-8
            )
            domain.free_surface_state.validate(
                domain.fluid_state, require_mass_fill_consistency=False
            )
            volume = float(
                np.sum(domain.free_surface_state.fill_level.numpy(), dtype=np.float64)
            )
            self.assertLess(abs(volume - initial_volume), 8.0e-5)

        final_fill = domain.free_surface_state.fill_level.numpy().copy()
        final_moments = domain.fluid_state.moments.numpy().copy()
        self.assertGreater(float(np.max(np.abs(final_fill - fill))), 1.0e-3)

        domain.initialize_uniform_lattice(fill, velocity=velocity)
        domain.step(model.time_step)
        reset_diagnostics = domain.last_geometric_diagnostics
        assert reset_diagnostics is not None
        self.assertEqual(reset_diagnostics.split_order, (0, 1, 2))
        return final_fill, final_moments

    def test_cpu_oblique_transaction(self) -> None:
        self._run_oblique_transaction("cpu")

    def _run_submerged_rigid_impulse(self, device: str) -> None:
        shape = (20, 12, 18)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=0.1,
            time_step=0.02,
            reference_density=1.0,
            kinematic_viscosity=0.1,
            max_lattice_speed=0.2,
            periodic=(True, True, False),
            device=device,
        )
        fluid = HomeFreeGeometricDomain(model, project_courant=True)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.8627, 0.6, 0.6), label="sphere")
        builder.add_shape_sphere(
            body,
            radius=0.2,
            cfg=ShapeConfig(density=1.2, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        initial_velocity = 0.05
        rigid.state._body_qd = wp.array(
            [[initial_velocity, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeFreeRigidCoupling(fluid, rigid, rigid_substeps=2)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :13] = 1.0
        fill[:, :, 13] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=13,
            gas_direction=1,
        )
        body_mass = rigid.model.get_body_mass(body)
        initial_total_x = (
            self._liquid_momentum_x(fluid) + body_mass * initial_velocity
        )
        initial_liquid_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )

        diagnostics = coupling.step()

        final_velocity = float(rigid.state.body_qd.numpy()[body, 0])
        final_total_x = self._liquid_momentum_x(fluid) + body_mass * final_velocity
        final_liquid_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        geometric = fluid.last_geometric_diagnostics
        assert geometric is not None and geometric.projection is not None
        self.assertGreater(diagnostics.interface.transitions.dead_cell_count, 0)
        self.assertGreater(diagnostics.interface.transitions.queued_mass, 0.0)
        self.assertGreater(geometric.queue.queue_source_count, 0)
        self.assertGreater(geometric.queue.materialized_cell_count, 0)
        self.assertLessEqual(geometric.queue.relative_mass_error, 4.0e-6)
        self.assertLessEqual(geometric.queue.relative_momentum_error, 4.0e-6)
        self.assertLessEqual(
            abs(
                geometric.queue.represented_mass
                - diagnostics.interface.transitions.final_represented_mass
            )
            / diagnostics.interface.transitions.final_represented_mass,
            4.0e-6,
        )
        total_momentum_error = abs(final_total_x - initial_total_x) / abs(
            initial_total_x
        )
        self.assertLess(
            total_momentum_error,
            3.0e-4,
            (
                f"initial_total={initial_total_x:.9e}, "
                f"final_total={final_total_x:.9e}, "
                f"final_velocity={final_velocity:.9e}, "
                f"remap_old={diagnostics.interface.transitions.old_represented_momentum}, "
                f"remap_final={diagnostics.interface.transitions.final_represented_momentum}, "
                f"queue_represented={geometric.queue.represented_momentum}, "
                f"queue_committed={geometric.queue.committed_momentum}"
            ),
        )
        self.assertLess(
            abs(final_liquid_mass - initial_liquid_mass) / initial_liquid_mass,
            2.0e-5,
        )
        self.assertEqual(diagnostics.interface.load.dry_link_count, 0)
        self.assertEqual(diagnostics.interface.fluid.missing_cut_link_count, 0)
        self.assertLessEqual(
            geometric.projection.projected_max_divergence, 2.0e-8
        )

    def test_cpu_submerged_rigid_impulse(self) -> None:
        self._run_submerged_rigid_impulse("cpu")

    def _make_geometric_strong_case(
        self, device: str, *, maximum_iterations: int, tolerance: float
    ) -> tuple[HomeFreeGeometricDomain, RigidDomain, HomeFreeRigidCoupling]:
        shape = (20, 12, 18)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=0.1,
            time_step=0.02,
            reference_density=1.0,
            kinematic_viscosity=0.1,
            max_lattice_speed=0.2,
            periodic=(True, True, False),
            device=device,
        )
        fluid = HomeFreeGeometricDomain(model, project_courant=True)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.8627, 0.6, 0.6), label="sphere")
        builder.add_shape_sphere(
            body,
            radius=0.2,
            cfg=ShapeConfig(density=1.2, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        rigid.state._body_qd = wp.array(
            [[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=maximum_iterations,
            strong_coupling_tolerance=tolerance,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :13] = 1.0
        fill[:, :, 13] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=13,
            gas_direction=1,
        )
        return fluid, rigid, coupling

    def _assert_geometric_strong_retries_restore_split_history(
        self, device: str
    ) -> None:
        fluid, _, coupling = self._make_geometric_strong_case(
            device,
            maximum_iterations=12,
            tolerance=1.0e-7,
        )
        split_orders: list[tuple[int, ...]] = []
        original_step = fluid.step

        def traced_step(dt: float, contacts: object = None) -> None:
            original_step(dt, contacts)
            assert fluid.last_geometric_diagnostics is not None
            split_orders.append(fluid.last_geometric_diagnostics.split_order)

        fluid.step = traced_step  # type: ignore[method-assign]
        diagnostics = coupling.step()

        self.assertGreater(diagnostics.strong_coupling_iterations, 1)
        self.assertEqual(
            len(split_orders), diagnostics.strong_coupling_iterations
        )
        self.assertEqual(set(split_orders), {(0, 1, 2)})
        self.assertEqual(fluid.geometric_stepper._split_parity, 1)
        self.assertIsNotNone(fluid.geometric_stepper._remapper)
        assert fluid.last_geometric_diagnostics is not None
        self.assertGreater(
            fluid.last_geometric_diagnostics.queue.queue_source_count, 0
        )

    def test_cpu_geometric_strong_retries_restore_split_history(self) -> None:
        self._assert_geometric_strong_retries_restore_split_history("cpu")

    def _assert_failed_geometric_strong_step_restores_transaction_history(
        self, device: str
    ) -> None:
        fluid, rigid, coupling = self._make_geometric_strong_case(
            device,
            maximum_iterations=2,
            tolerance=1.0e-20,
        )
        fluid_before = (
            fluid.fluid_state.moments.numpy().copy(),
            fluid.free_surface_state.mass.numpy().copy(),
            fluid.free_surface_state.fill_level.numpy().copy(),
            fluid.free_surface_state.excess_mass.numpy().copy(),
            fluid.free_surface_state.excess_momentum.numpy().copy(),
            fluid.free_surface_state.flags.numpy().copy(),
        )
        rigid_before = (
            rigid.state.body_q.numpy().copy(),
            rigid.state.body_qd.numpy().copy(),
            rigid.state.body_f.numpy().copy(),
        )
        split_orders: list[tuple[int, ...]] = []
        original_step = fluid.step

        def traced_step(dt: float, contacts: object = None) -> None:
            original_step(dt, contacts)
            assert fluid.last_geometric_diagnostics is not None
            split_orders.append(fluid.last_geometric_diagnostics.split_order)

        fluid.step = traced_step  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            coupling.step()

        self.assertEqual(split_orders, [(0, 1, 2), (0, 1, 2)])
        self.assertEqual(fluid.geometric_stepper._split_parity, 0)
        self.assertIsNone(fluid.geometric_stepper._remapper)
        self.assertIsNone(fluid.last_geometric_diagnostics)
        fluid_after = (
            fluid.fluid_state.moments.numpy(),
            fluid.free_surface_state.mass.numpy(),
            fluid.free_surface_state.fill_level.numpy(),
            fluid.free_surface_state.excess_mass.numpy(),
            fluid.free_surface_state.excess_momentum.numpy(),
            fluid.free_surface_state.flags.numpy(),
        )
        rigid_after = (
            rigid.state.body_q.numpy(),
            rigid.state.body_qd.numpy(),
            rigid.state.body_f.numpy(),
        )
        for before, after in zip(fluid_before, fluid_after, strict=True):
            np.testing.assert_array_equal(after, before)
        for before, after in zip(rigid_before, rigid_after, strict=True):
            np.testing.assert_array_equal(after, before)
        self.assertIs(fluid.solver.cut_links, coupling.interface.cut_links)
        self.assertIsNone(coupling.interface._prepared)

    def test_cpu_failed_geometric_strong_step_restores_transaction_history(
        self,
    ) -> None:
        self._assert_failed_geometric_strong_step_restores_transaction_history(
            "cpu"
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_geometric_strong_retries_restore_split_history(self) -> None:
        self._assert_geometric_strong_retries_restore_split_history("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_failed_geometric_strong_step_restores_transaction_history(
        self,
    ) -> None:
        self._assert_failed_geometric_strong_step_restores_transaction_history(
            "cuda:0"
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_oblique_transaction_matches_cpu(self) -> None:
        expected_fill, expected_moments = self._run_oblique_transaction("cpu")
        actual_fill, actual_moments = self._run_oblique_transaction("cuda:0")
        np.testing.assert_allclose(actual_fill, expected_fill, rtol=0.0, atol=3.0e-6)
        np.testing.assert_allclose(
            actual_moments,
            expected_moments,
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_submerged_rigid_impulse(self) -> None:
        self._run_submerged_rigid_impulse("cuda:0")


if __name__ == "__main__":
    unittest.main()
