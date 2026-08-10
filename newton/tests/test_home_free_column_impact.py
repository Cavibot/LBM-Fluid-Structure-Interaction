# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Momentum-ledger acceptance case for a liquid column striking a sphere."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
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


def _represented_mass_and_momentum(
    fluid: HomeFreeLegacyDomain,
) -> tuple[float, np.ndarray]:
    free = fluid.free_surface_state
    shape = free.res
    recipients = (
        np.zeros(shape, dtype=np.float64)
        if getattr(fluid, "uses_independent_geometric_mass", False)
        else fluid.topology.active_neighbor_count.numpy().astype(np.float64)
    )
    mass = free.mass.numpy().astype(np.float64)
    excess_mass = free.excess_mass.numpy().astype(np.float64)
    excess_momentum = (
        free.excess_momentum.numpy()
        .reshape(3, *shape)
        .transpose(1, 2, 3, 0)
        .astype(np.float64)
    )
    moments = fluid.fluid_state.moments.numpy().reshape(10, *shape).astype(np.float64)
    density = moments[0]
    velocity = np.zeros(shape + (3,), dtype=np.float64)
    active_mass = mass != 0.0
    velocity[active_mass] = (
        np.moveaxis(moments[1:4], 0, -1)[active_mass]
        / density[active_mass, None]
    )
    represented_mass = np.sum(
        mass + recipients * excess_mass,
        dtype=np.float64,
    )
    represented_momentum = np.sum(
        mass[..., None] * velocity
        + recipients[..., None] * excess_momentum,
        axis=(0, 1, 2),
        dtype=np.float64,
    )
    return float(represented_mass), represented_momentum


class TestHomeFreeColumnImpact(unittest.TestCase):
    def _run_geometric_column_impact(self, device: str) -> None:
        shape = (28, 14, 14)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            max_lattice_speed=0.18,
            periodic=(True, True, True),
            device=device,
        )
        fluid = HomeFreeGeometricDomain(
            model,
            contact_angle_degrees=90.0,
            project_courant=True,
        )
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(16.0, 7.0, 7.0), label="impact_sphere")
        builder.add_shape_sphere(
            body,
            radius=3.0,
            cfg=ShapeConfig(density=1.0, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=10,
            strong_coupling_tolerance=2.0e-6,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[2] = 0.5
        fill[3:11] = 1.0
        fill[11] = 0.5
        fluid.initialize_uniform_lattice(fill, velocity=(0.06, 0.0, 0.0))

        initial_mass, initial_fluid_momentum = _represented_mass_and_momentum(fluid)
        body_mass = rigid.model.get_body_mass(body)
        initial_total_momentum = initial_fluid_momentum.copy()
        accumulated_gas_impulse = np.zeros(3, dtype=np.float64)
        accumulated_ambient_correction = np.zeros(3, dtype=np.float64)
        accumulated_frame_correction = np.zeros(3, dtype=np.float64)
        maximum_cut_link_residual = 0.0
        maximum_ledger_residual = 0.0
        maximum_projection_divergence = 0.0
        maximum_queue_mass_error = 0.0
        maximum_queue_momentum_error = 0.0
        fresh_total = 0
        dead_total = 0
        wet_counts: list[int] = []

        for _ in range(40):
            rigid_before = body_mass * rigid.state.body_qd.numpy()[
                body, :3
            ].astype(np.float64)
            diagnostics = coupling.step()
            rigid_after = body_mass * rigid.state.body_qd.numpy()[
                body, :3
            ].astype(np.float64)
            expected_rigid_impulse = (
                coupling.interface.wrench.lattice_numpy()[0][body].astype(np.float64)
                * model.scaling.force_unit
                * model.time_step
            )
            maximum_cut_link_residual = max(
                maximum_cut_link_residual,
                float(
                    np.linalg.norm(
                        rigid_after - rigid_before - expected_rigid_impulse
                    )
                ),
            )
            accumulated_gas_impulse += (
                np.asarray(
                    diagnostics.interface.fluid.gas_boundary_impulse_lattice,
                    dtype=np.float64,
                )
                * model.scaling.momentum_unit
            )
            accumulated_ambient_correction += (
                np.asarray(
                    diagnostics.interface.load.ambient_pressure_correction_lattice,
                    dtype=np.float64,
                )
                * model.scaling.momentum_unit
            )
            accumulated_frame_correction += (
                np.asarray(
                    diagnostics.interface.fluid.cut_link_frame_correction_lattice,
                    dtype=np.float64,
                )
                * model.scaling.momentum_unit
            )
            _, fluid_momentum = _represented_mass_and_momentum(fluid)
            ledger = (
                fluid_momentum
                + rigid_after
                - initial_total_momentum
                - accumulated_gas_impulse
                - accumulated_ambient_correction
                - accumulated_frame_correction
            )
            maximum_ledger_residual = max(
                maximum_ledger_residual, float(np.linalg.norm(ledger))
            )
            geometric = fluid.last_geometric_diagnostics
            assert geometric is not None and geometric.projection is not None
            maximum_projection_divergence = max(
                maximum_projection_divergence,
                geometric.projection.projected_max_divergence,
            )
            maximum_queue_mass_error = max(
                maximum_queue_mass_error, geometric.queue.relative_mass_error
            )
            maximum_queue_momentum_error = max(
                maximum_queue_momentum_error,
                geometric.queue.relative_momentum_error,
            )
            fresh_total += diagnostics.interface.transitions.fresh_cell_count
            dead_total += diagnostics.interface.transitions.dead_cell_count
            wet_counts.append(diagnostics.interface.load.wet_link_count)
            self.assertLess(
                diagnostics.strong_coupling_residual,
                coupling.strong_coupling_tolerance,
            )

        final_mass, final_fluid_momentum = _represented_mass_and_momentum(fluid)
        final_velocity = rigid.state.body_qd.numpy()[body, :3].astype(np.float64)
        momentum_scale = max(float(np.linalg.norm(initial_total_momentum)), 1.0)
        self.assertEqual(wet_counts[0], 0)
        self.assertGreater(max(wet_counts), 300)
        self.assertGreater(final_velocity[0], 0.04)
        self.assertLess(final_fluid_momentum[0], initial_fluid_momentum[0] - 0.5)
        self.assertGreater(fresh_total, 0)
        self.assertGreater(dead_total, 0)
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 2.0e-6)
        self.assertLess(maximum_cut_link_residual / momentum_scale, 2.0e-5)
        self.assertLess(maximum_ledger_residual / momentum_scale, 9.0e-3)
        self.assertLess(maximum_projection_divergence, 1.0e-8)
        self.assertLess(maximum_queue_mass_error, 4.0e-6)
        self.assertLess(maximum_queue_momentum_error, 4.0e-6)
        np.testing.assert_array_equal(
            fluid.free_surface_state.excess_mass.numpy(), 0.0
        )

    def _run_column_impact(self, device: str) -> None:
        shape = (28, 14, 14)
        initial_liquid_speed = 0.06
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            max_lattice_speed=0.18,
            periodic=(True, True, True),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(16.0, 7.0, 7.0), label="impact_sphere")
        builder.add_shape_sphere(
            body,
            radius=3.0,
            cfg=ShapeConfig(density=1.0, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=device)
        rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        rigid.create_state()
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=10,
            strong_coupling_tolerance=2.0e-6,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[2] = 0.5
        fill[3:11] = 1.0
        fill[11] = 0.5
        fluid.initialize_uniform_lattice(
            fill,
            velocity=(initial_liquid_speed, 0.0, 0.0),
        )
        fluid.advector.enable_momentum_defect_diagnostics()

        initial_mass, initial_fluid_momentum = _represented_mass_and_momentum(fluid)
        body_mass = rigid.model.get_body_mass(body)
        initial_rigid_momentum = (
            body_mass * rigid.state.body_qd.numpy()[body, :3].astype(np.float64)
        )
        initial_total_momentum = initial_fluid_momentum + initial_rigid_momentum
        accumulated_gas_impulse = np.zeros(3, dtype=np.float64)
        accumulated_ambient_correction = np.zeros(3, dtype=np.float64)
        accumulated_frame_correction = np.zeros(3, dtype=np.float64)
        accumulated_exact_link_defect = np.zeros(3, dtype=np.float64)
        maximum_cut_link_residual = 0.0
        maximum_remap_residual = 0.0
        maximum_total_ledger_residual = 0.0
        maximum_exact_link_ledger_residual = 0.0
        fresh_total = 0
        dead_total = 0
        wet_counts: list[int] = []

        for _ in range(40):
            rigid_momentum_before = (
                body_mass * rigid.state.body_qd.numpy()[body, :3].astype(np.float64)
            )
            diagnostics = coupling.step()
            rigid_momentum_after = (
                body_mass * rigid.state.body_qd.numpy()[body, :3].astype(np.float64)
            )
            fluid_force_lattice = coupling.interface.wrench.lattice_numpy()[0][body]
            expected_rigid_impulse = (
                fluid_force_lattice.astype(np.float64)
                * model.scaling.force_unit
                * model.time_step
            )
            cut_link_residual = np.linalg.norm(
                rigid_momentum_after - rigid_momentum_before - expected_rigid_impulse
            )
            maximum_cut_link_residual = max(
                maximum_cut_link_residual,
                float(cut_link_residual),
            )

            transition = diagnostics.interface.transitions
            remap_residual = np.linalg.norm(
                np.asarray(transition.final_represented_momentum)
                - np.asarray(transition.old_represented_momentum)
            )
            maximum_remap_residual = max(maximum_remap_residual, float(remap_residual))
            gas_impulse = (
                np.asarray(
                    diagnostics.interface.fluid.gas_boundary_impulse_lattice,
                    dtype=np.float64,
                )
                * model.scaling.momentum_unit
            )
            accumulated_gas_impulse += gas_impulse
            ambient_correction = (
                np.asarray(
                    diagnostics.interface.load.ambient_pressure_correction_lattice,
                    dtype=np.float64,
                )
                * model.scaling.momentum_unit
            )
            accumulated_ambient_correction += ambient_correction
            frame_correction = (
                np.asarray(
                    diagnostics.interface.fluid.cut_link_frame_correction_lattice,
                    dtype=np.float64,
                )
                * model.scaling.momentum_unit
            )
            accumulated_frame_correction += frame_correction
            advector = fluid.advector
            assert advector.home_destination_momentum is not None
            assert advector.internal_link_advected_momentum is not None
            assert advector.home_internal_link_momentum is not None
            assert advector.incoming_excess_momentum is not None
            home_momentum = np.moveaxis(
                advector.home_destination_momentum.numpy().reshape(3, *shape),
                0,
                -1,
            ).astype(np.float64)
            _, fluid_momentum = _represented_mass_and_momentum(fluid)
            internal_liquid_momentum = np.moveaxis(
                advector.internal_link_advected_momentum.numpy().reshape(
                    3, *shape
                ),
                0,
                -1,
            ).astype(np.float64)
            home_internal_momentum = np.moveaxis(
                advector.home_internal_link_momentum.numpy().reshape(3, *shape),
                0,
                -1,
            ).astype(np.float64)
            incoming_queue_momentum = np.moveaxis(
                advector.incoming_excess_momentum.numpy().reshape(3, *shape),
                0,
                -1,
            ).astype(np.float64)
            source_free_state = advector._advected_free_surface_state
            assert source_free_state is not None
            source_active = np.isin(source_free_state.flags.numpy(), (1, 2))
            target_fluid_momentum = np.sum(
                (
                    internal_liquid_momentum
                    + incoming_queue_momentum
                    + home_momentum
                    - home_internal_momentum
                )[source_active],
                axis=0,
                dtype=np.float64,
            )
            accumulated_exact_link_defect += (
                target_fluid_momentum - fluid_momentum
            )
            total_momentum = fluid_momentum + rigid_momentum_after
            momentum_ledger = (
                total_momentum
                - initial_total_momentum
                - accumulated_gas_impulse
                - accumulated_ambient_correction
                - accumulated_frame_correction
            )
            total_ledger_residual = np.linalg.norm(momentum_ledger)
            exact_link_ledger_residual = np.linalg.norm(
                momentum_ledger + accumulated_exact_link_defect
            )
            maximum_total_ledger_residual = max(
                maximum_total_ledger_residual,
                float(total_ledger_residual),
            )
            maximum_exact_link_ledger_residual = max(
                maximum_exact_link_ledger_residual,
                float(exact_link_ledger_residual),
            )
            fresh_total += transition.fresh_cell_count
            dead_total += transition.dead_cell_count
            wet_counts.append(diagnostics.interface.load.wet_link_count)
            self.assertEqual(
                diagnostics.interface.fluid.direct_liquid_gas_link_count,
                0,
            )
            self.assertLess(
                diagnostics.strong_coupling_residual,
                coupling.strong_coupling_tolerance,
            )

        final_mass, final_fluid_momentum = _represented_mass_and_momentum(fluid)
        final_velocity = rigid.state.body_qd.numpy()[body, :3].astype(np.float64)
        momentum_scale = max(float(np.linalg.norm(initial_total_momentum)), 1.0)
        self.assertEqual(wet_counts[0], 0)
        self.assertGreater(max(wet_counts), 0)
        self.assertGreater(final_velocity[0], 0.03)
        self.assertLess(final_fluid_momentum[0], initial_fluid_momentum[0] - 2.0)
        self.assertGreater(fresh_total, 0)
        self.assertGreater(dead_total, 0)
        self.assertGreater(np.linalg.norm(accumulated_gas_impulse), 1.0)
        self.assertGreater(np.linalg.norm(accumulated_ambient_correction), 1.0)
        self.assertGreater(np.linalg.norm(accumulated_frame_correction), 0.1)
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 2.0e-5)
        self.assertLess(maximum_cut_link_residual / momentum_scale, 2.0e-5)
        self.assertLess(maximum_remap_residual / momentum_scale, 2.0e-5)
        # The fast production state uses mass times HOME velocity, so its
        # represented momentum carries a bounded link-replacement defect. The
        # exact link ledger below remains the conservation oracle.
        self.assertLess(maximum_total_ledger_residual / momentum_scale, 1.25e-2)
        self.assertLess(
            maximum_exact_link_ledger_residual / momentum_scale,
            3.0e-5,
        )

    def test_cpu_column_strikes_sphere_with_closed_momentum_ledger(self) -> None:
        self._run_column_impact("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_column_strikes_sphere_with_closed_momentum_ledger(self) -> None:
        self._run_column_impact("cuda:0")

    def test_cpu_geometric_column_strikes_sphere(self) -> None:
        self._run_geometric_column_impact("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_geometric_column_strikes_sphere(self) -> None:
        self._run_geometric_column_impact("cuda:0")


if __name__ == "__main__":
    unittest.main()
