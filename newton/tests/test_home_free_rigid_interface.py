# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase-aware cut-link loads for a frozen, partially wetted rigid body."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeFreeRigidInterface,
    HomeLbmModel,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder


class TestHomeFreeRigidInterface(unittest.TestCase):
    @staticmethod
    def _represented_mass(fluid: HomeFreeLegacyDomain) -> float:
        recipients = fluid.topology.active_neighbor_count.numpy()
        state = fluid.free_surface_state
        return float(
            np.sum(
                state.mass.numpy() + state.excess_mass.numpy() * recipients,
                dtype=np.float64,
            )
        )

    def _run_partially_wetted_sphere(self, device: str) -> None:
        shape = (18, 18, 18)
        cell_size = 0.1
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=cell_size,
            time_step=0.02,
            reference_density=1.0,
            kinematic_viscosity=0.1,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.9, 0.9, 0.9), label="half_wet_sphere")
        builder.add_shape_sphere(body, radius=0.34)
        rigid = RigidDomain(builder.finalize(device=device))
        interface = HomeFreeRigidInterface(fluid, rigid)

        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :9] = 1.0
        fill[:, :, 9] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=9,
            gas_direction=1,
        )
        diagnostics = interface.step_static()
        force, torque = interface.wrench.lattice_numpy()

        self.assertGreater(interface.cut_links.link_count, 0)
        self.assertGreater(diagnostics.load.wet_link_count, 0)
        self.assertGreater(diagnostics.load.dry_link_count, 0)
        self.assertEqual(diagnostics.load.invalid_flag_count, 0)
        self.assertEqual(diagnostics.load.dry_nonzero_impulse_count, 0)
        self.assertEqual(diagnostics.fluid.missing_cut_link_count, 0)
        self.assertEqual(diagnostics.fluid.direct_liquid_gas_link_count, 0)
        np.testing.assert_allclose(force[body], 0.0, rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(torque[body], 0.0, rtol=0.0, atol=2.0e-5)

    def test_cpu_partially_wetted_sphere_has_zero_ambient_gauge_load(self) -> None:
        self._run_partially_wetted_sphere("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_partially_wetted_sphere_has_zero_ambient_gauge_load(self) -> None:
        self._run_partially_wetted_sphere("cuda:0")

    def _buoyancy_error(self, radius: float, device: str) -> float:
        shape = (28, 28, 28)
        acceleration = -2.0e-4
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=(0.0, 0.0, acceleration),
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(14.0, 14.0, 10.0), label="submerged_sphere")
        builder.add_shape_sphere(body, radius=radius)
        rigid = RigidDomain(builder.finalize(device=device))
        interface = HomeFreeRigidInterface(fluid, rigid)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :22] = 1.0
        fill[:, :, 22] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=22,
            gas_direction=1,
        )

        diagnostics = interface.step_static()
        force, torque = interface.wrench.lattice_numpy()
        expected = 4.0 * np.pi * radius**3 * abs(model.lattice_acceleration[2]) / 3.0
        self.assertGreater(diagnostics.load.wet_link_count, 0)
        self.assertEqual(diagnostics.load.dry_link_count, 0)
        self.assertLess(float(np.linalg.norm(force[body, :2])) / expected, 1.0e-4)
        self.assertGreater(float(force[body, 2]), 0.0)
        self.assertLess(float(np.linalg.norm(torque[body])) / expected, 1.0e-4)
        return abs(float(force[body, 2]) - expected) / expected

    def test_cpu_hydrostatic_buoyancy_improves_with_sphere_resolution(self) -> None:
        coarse_error = self._buoyancy_error(4.0, "cpu")
        fine_error = self._buoyancy_error(6.0, "cpu")
        self.assertLess(coarse_error, 0.16)
        self.assertLess(fine_error, 0.09)
        self.assertLess(fine_error, coarse_error)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_hydrostatic_buoyancy_improves_with_sphere_resolution(self) -> None:
        coarse_error = self._buoyancy_error(4.0, "cuda:0")
        fine_error = self._buoyancy_error(6.0, "cuda:0")
        self.assertLess(coarse_error, 0.16)
        self.assertLess(fine_error, 0.09)
        self.assertLess(fine_error, coarse_error)

    def test_motion_is_rejected_until_vof_transition_remap_exists(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.1,
            time_step=0.02,
            device="cpu",
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.4, 0.4, 0.4), label="moving_sphere")
        builder.add_shape_sphere(body, radius=0.15)
        rigid = RigidDomain(builder.finalize(device="cpu"))
        interface = HomeFreeRigidInterface(fluid, rigid)
        fill = np.ones((8, 8, 8), dtype=np.float32)
        fluid.initialize_uniform_lattice(fill)
        rigid.state._body_qd = wp.array(
            [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device="cpu",
        )
        with self.assertRaisesRegex(RuntimeError, "transition remapper"):
            interface.step_static()

    def _run_prepared_moving_sphere(self, device: str) -> None:
        shape = (18, 18, 18)
        cell_size = 0.1
        time_step = 0.02
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=cell_size,
            time_step=time_step,
            kinematic_viscosity=0.1,
            max_lattice_speed=0.25,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeFreeLegacyDomain(model)
        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=(0.83, 0.9, 0.9), label="moving_half_wet_sphere")
        builder.add_shape_sphere(body, radius=0.24)
        rigid = RigidDomain(builder.finalize(device=device))
        interface = HomeFreeRigidInterface(fluid, rigid)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :9] = 1.0
        fill[:, :, 9] = 0.5
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=9,
            gas_direction=1,
        )
        initial_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        sampled_q = wp.array(
            [wp.transform(wp.vec3(0.84, 0.9, 0.9), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        sampled_qd = wp.array(
            [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=device,
        )

        prepared = interface.prepare_step(sampled_q, sampled_qd)
        diagnostics = interface.step_prepared(sampled_q)
        force, torque = interface.wrench.lattice_numpy()
        final_mass = self._represented_mass(fluid)

        self.assertGreater(prepared.transitions.fresh_cell_count, 0)
        self.assertGreater(prepared.transitions.dead_cell_count, 0)
        self.assertLess(prepared.transitions.relative_mass_error, 1.0e-5)
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 1.0e-5)
        self.assertGreater(diagnostics.load.wet_link_count, 0)
        self.assertGreater(diagnostics.load.dry_link_count, 0)
        self.assertEqual(diagnostics.load.dry_nonzero_impulse_count, 0)
        self.assertEqual(diagnostics.fluid.missing_cut_link_count, 0)
        self.assertEqual(diagnostics.fluid.direct_liquid_gas_link_count, 0)
        self.assertLess(diagnostics.fluid.max_speed, model.max_lattice_speed)
        np.testing.assert_array_equal(
            fluid.free_surface_state.flags.numpy() == 3,
            fluid.fluid_state.solid_phi.numpy() < 0.0,
        )
        self.assertLess(float(force[body, 0]), 0.0)
        self.assertTrue(np.isfinite(force[body]).all())
        self.assertTrue(np.isfinite(torque[body]).all())

    def test_cpu_prepared_motion_remaps_vof_and_produces_drag(self) -> None:
        self._run_prepared_moving_sphere("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_prepared_motion_remaps_vof_and_produces_drag(self) -> None:
        self._run_prepared_moving_sphere("cuda:0")


if __name__ == "__main__":
    unittest.main()
