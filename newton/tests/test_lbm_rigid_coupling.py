# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused smoke tests for WanPhys LBM rigid coupling."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.coupling import coupling_kernels as ck
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys.rigid import RigidDomain, RigidModelBuilder


@wp.kernel
def _seed_single_lbm_feedback_torque_case(
    density: wp.array3d(dtype=float),
    velocity_x: wp.array3d(dtype=float),
    velocity_y: wp.array3d(dtype=float),
    velocity_z: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
):
    density[5, 6, 4] = 1.0
    velocity_x[5, 6, 4] = -0.2
    velocity_y[5, 6, 4] = 0.0
    velocity_z[5, 6, 4] = 0.0
    solid_phi[4, 6, 4] = -1.0
    solid_body_id[4, 6, 4] = 0


class TestLbmRigidCoupling(unittest.TestCase):
    def test_static_sphere_rasterizes_into_lbm_solid_fields(self) -> None:
        """A zero-velocity rigid sphere should become a static LBM obstacle."""
        nx: int = 12
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        sphere_center: tuple[float, float, float] = (0.6, 0.6, 0.6)
        sphere_radius: float = 0.26
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=sphere_center, label="static_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state.body_qd.zero_()

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=sphere_radius)
        coupling.set_rigid_dynamics_enabled(False)

        coupling.step(dt=1.0 / 60.0)
        wp.synchronize_device(fluid_model._device)

        solid_phi: np.ndarray = fluid_domain.state.solid_phi.numpy()
        solid_body_id: np.ndarray = fluid_domain.state.solid_body_id.numpy()

        self.assertTrue(np.any(solid_phi < 0.0))
        self.assertTrue(np.any(solid_body_id == body_id))
        self.assertTrue(hasattr(fluid_domain.state, "vel_solid_u"))
        self.assertTrue(hasattr(fluid_domain.state, "vel_solid_v"))
        self.assertTrue(hasattr(fluid_domain.state, "vel_solid_w"))

    def test_moving_sphere_stamps_surface_velocity_and_moves_fluid(self) -> None:
        """A moving rigid sphere should stamp wall velocity and perturb LBM velocity."""
        nx: int = 12
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        sphere_center: tuple[float, float, float] = (0.6, 0.6, 0.6)
        sphere_radius: float = 0.26
        sphere_speed: float = 0.04
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=sphere_center, label="moving_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state._body_qd = wp.array(
            [[sphere_speed, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=fluid_model._device,
        )

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=sphere_radius)
        coupling.set_rigid_dynamics_enabled(False)

        for _ in range(3):
            coupling.step(dt=1.0 / 60.0)
        wp.synchronize_device(fluid_model._device)

        solid_u: np.ndarray = fluid_domain.state.vel_solid_u.numpy()
        fluid_ux: np.ndarray = fluid_domain.state.velocity_x.numpy()

        expected_lbm_speed: float = sphere_speed * (1.0 / 60.0) / dh
        self.assertGreater(float(np.max(np.abs(solid_u))), expected_lbm_speed * 0.5)
        self.assertLess(float(np.max(np.abs(solid_u))), expected_lbm_speed * 1.5)
        self.assertGreater(float(np.max(np.abs(fluid_ux))), 1.0e-5)

    def test_moving_sphere_stamps_lbm_wall_velocity_from_world_speed(self) -> None:
        """Rigid body_qd is world velocity; vel_solid_* stores LBM wall velocity."""
        nx: int = 12
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        dt: float = 1.0 / 60.0
        world_speed: float = 0.3
        expected_lbm_speed: float = world_speed * dt / dh
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=(0.6, 0.6, 0.6), label="scaled_velocity_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=0.26)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state._body_qd = wp.array(
            [[world_speed, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=fluid_model._device,
        )

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=0.26)
        coupling.set_rigid_dynamics_enabled(False)

        coupling.step(dt=dt)
        wp.synchronize_device(fluid_model._device)

        solid_u: np.ndarray = np.asarray(fluid_domain.state.vel_solid_u.numpy(), dtype=np.float64)
        nonzero_u: np.ndarray = np.abs(solid_u[np.abs(solid_u) > 1.0e-8])

        self.assertGreater(int(nonzero_u.size), 0)
        self.assertAlmostEqual(float(nonzero_u.max()), expected_lbm_speed, delta=expected_lbm_speed * 0.15)

    def test_rotating_sphere_warns_when_lbm_wall_velocity_is_large(self) -> None:
        """Angular rigid speed contributes to the LBM wall-velocity warning."""
        nx: int = 12
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        dt: float = 1.0 / 60.0
        sphere_radius: float = 0.26
        angular_speed: float = 4.0
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=(0.6, 0.6, 0.6), label="rotating_warning_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state._body_qd = wp.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, angular_speed]],
            dtype=wp.spatial_vector,
            device=fluid_model._device,
        )

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=sphere_radius)
        coupling.set_rigid_dynamics_enabled(False)

        with self.assertWarns(RuntimeWarning):
            coupling.step(dt=dt)

    def test_uncoupled_fast_body_does_not_trigger_lbm_wall_velocity_warning(self) -> None:
        """The LBM wall-velocity warning should only inspect registered bodies."""
        nx: int = 12
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        dt: float = 1.0 / 60.0
        coupled_body_id: int = 0
        uncoupled_body_id: int = 1

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        coupled_body: int = rigid_builder.add_body(position=(0.6, 0.6, 0.6), label="coupled_sphere")
        uncoupled_body: int = rigid_builder.add_body(position=(0.9, 0.6, 0.6), label="uncoupled_fast_sphere")
        self.assertEqual(coupled_body, coupled_body_id)
        self.assertEqual(uncoupled_body, uncoupled_body_id)
        rigid_builder.add_shape_sphere(coupled_body, radius=0.2)
        rigid_builder.add_shape_sphere(uncoupled_body, radius=0.2)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state._body_qd = wp.array(
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=wp.spatial_vector,
            device=fluid_model._device,
        )

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=coupled_body_id, radius=0.2)
        coupling.set_rigid_dynamics_enabled(False)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            coupling.step(dt=dt)

        runtime_warnings: list[warnings.WarningMessage] = [
            warning for warning in caught_warnings if issubclass(warning.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])

    def test_fast_prescribed_sphere_keeps_lbm_fields_finite(self) -> None:
        """A fast world-space sphere should remain finite after dt/dh conversion."""
        nx: int = 16
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        dt: float = 1.0 / 60.0
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.6,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=(0.55, 0.6, 0.6), label="fast_prescribed_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=0.24)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state._body_qd = wp.array(
            [[0.75, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=fluid_model._device,
        )

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=0.24)
        coupling.set_rigid_dynamics_enabled(False)

        for _step_index in range(5):
            coupling.step(dt=dt)

        wp.synchronize_device(fluid_model._device)
        density: np.ndarray = np.asarray(fluid_domain.state.density.numpy(), dtype=np.float64)
        velocity_x: np.ndarray = np.asarray(fluid_domain.state.velocity_x.numpy(), dtype=np.float64)

        self.assertTrue(np.isfinite(density).all())
        self.assertTrue(np.isfinite(velocity_x).all())

    def test_advancing_sphere_moves_solid_rasterization_and_fluid(self) -> None:
        """An advancing rigid pose should move the rasterized obstacle and perturb fluid."""
        nx: int = 16
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        sphere_center: tuple[float, float, float] = (0.55, 0.6, 0.6)
        sphere_radius: float = 0.24
        sphere_speed: float = 0.3
        body_id: int = 0
        dt: float = 1.0 / 30.0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=sphere_center, label="advancing_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state._body_qd = wp.array(
            [[sphere_speed, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=fluid_model._device,
        )

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=sphere_radius)
        coupling.set_rigid_dynamics_enabled(True)

        coupling.step(dt=dt)
        wp.synchronize_device(fluid_model._device)
        initial_solid_phi: np.ndarray = fluid_domain.state.solid_phi.numpy()
        initial_solid_body_id: np.ndarray = fluid_domain.state.solid_body_id.numpy()
        initial_mask: np.ndarray = (initial_solid_phi < 0.0) & (initial_solid_body_id == body_id)
        initial_com: np.ndarray = (np.argwhere(initial_mask).mean(axis=0) + 0.5) * dh

        for _step_index in range(8):
            coupling.step(dt=dt)
        wp.synchronize_device(fluid_model._device)

        final_solid_phi: np.ndarray = fluid_domain.state.solid_phi.numpy()
        final_solid_body_id: np.ndarray = fluid_domain.state.solid_body_id.numpy()
        final_mask: np.ndarray = (final_solid_phi < 0.0) & (final_solid_body_id == body_id)
        final_com: np.ndarray = (np.argwhere(final_mask).mean(axis=0) + 0.5) * dh
        fluid_ux: np.ndarray = fluid_domain.state.velocity_x.numpy()

        self.assertTrue(np.any(initial_mask))
        self.assertTrue(np.any(final_mask))
        self.assertGreater(float(final_com[0] - initial_com[0]), 0.25 * dh)
        self.assertGreater(float(rigid_domain.state.get_body_position(body_id)[0]), sphere_center[0])
        self.assertGreater(float(np.max(np.abs(fluid_ux))), 1.0e-5)

    def test_two_way_feedback_accumulates_body_force_when_enabled(self) -> None:
        """Two-way mode should accumulate boundary feedback into body_f."""
        nx: int = 12
        ny: int = 12
        nz: int = 12
        dh: float = 0.1
        sphere_center: tuple[float, float, float] = (0.6, 0.6, 0.6)
        sphere_radius: float = 0.26
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=1.0,
            u0=(0.05, 0.0, 0.0),
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=sphere_center, label="two_way_sphere")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state.body_qd.zero_()

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
        coupling.add_body_sphere(body_idx=body_id, radius=sphere_radius)
        coupling.set_rigid_dynamics_enabled(False)

        coupling.step(dt=1.0 / 60.0)
        wp.synchronize_device(fluid_model._device)
        one_way_force: np.ndarray = rigid_domain.state.get_body_force(body_id)
        self.assertEqual(float(np.linalg.norm(one_way_force)), 0.0)

        rigid_domain.state.clear_forces()
        coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
        coupling.step(dt=1.0 / 60.0)
        wp.synchronize_device(fluid_model._device)
        two_way_force: np.ndarray = rigid_domain.state.get_body_force(body_id)

        self.assertGreater(float(np.linalg.norm(two_way_force[:3])), 0.0)

    def test_two_way_feedback_accumulates_torque_for_off_center_face(self) -> None:
        """An off-center boundary impulse should accumulate torque as well as force."""
        nx: int = 8
        ny: int = 8
        nz: int = 8
        dh: float = 0.1
        body_id: int = 0
        body_center: tuple[float, float, float] = (0.5, 0.5, 0.45)

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.state.clear()

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(position=body_center, label="torque_probe")
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=0.1)
        rigid_domain: RigidDomain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
        rigid_domain.create_state()
        rigid_domain.state.clear_forces()

        wp.launch(
            _seed_single_lbm_feedback_torque_case,
            dim=1,
            inputs=[
                fluid_domain.state.density,
                fluid_domain.state.velocity_x,
                fluid_domain.state.velocity_y,
                fluid_domain.state.velocity_z,
                fluid_domain.state.solid_phi,
                fluid_domain.state.solid_body_id,
            ],
        )
        rigid_backend: object = rigid_domain.model._newton_backend
        wp.launch(
            ck.accumulate_lbm_boundary_feedback_all_bodies,
            dim=(nx, ny, nz),
            inputs=[
                fluid_domain.state.density,
                fluid_domain.state.velocity_x,
                fluid_domain.state.velocity_y,
                fluid_domain.state.velocity_z,
                fluid_domain.state.solid_phi,
                fluid_domain.state.solid_body_id,
                fluid_domain.state.vel_solid_u,
                fluid_domain.state.vel_solid_v,
                fluid_domain.state.vel_solid_w,
                dh,
                nx,
                ny,
                nz,
                rigid_domain.state.body_q,
                rigid_backend.body_com,
                rigid_domain.state.body_f,
                1.0,
            ],
        )
        wp.synchronize_device(fluid_model._device)

        body_force: np.ndarray = rigid_domain.state.get_body_force(body_id)

        self.assertLess(float(body_force[0]), 0.0)
        self.assertGreater(float(abs(body_force[5])), 1.0e-8)


# =======================================================================
# Momentum-Exchange Feedback Tests (strict Ladd 1994 formulation)
# =======================================================================


class TestLbmMomentumExchangeFeedback(unittest.TestCase):
    """Smoke tests for the strict momentum-exchange feedback path."""

    def _run_momentum_exchange_cycle(
        self,
        nx: int,
        ny: int,
        nz: int,
        dh: float,
        sphere_center: tuple[float, float, float],
        sphere_radius: float,
        n_steps: int,
        body_qd: list[float] | None = None,
        initial_rho: float = 1.0,
        initial_u: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, LbmDomain, RigidDomain, GridLbmRigidCoupling]:
        """Helper: set up and run a momentum-exchange coupling scenario.

        Returns:
            (body_force_array, fluid_domain, rigid_domain, coupling)
        """
        dt: float = 1.0 / 60.0
        body_id: int = 0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.solver.initialize_equilibrium(
            fluid_domain.state,
            rho0=initial_rho,
            u0=initial_u,
        )

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(
            position=sphere_center, label="momentum_exchange_sphere"
        )
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(
            rigid_builder.finalize(device=fluid_model._device)
        )
        rigid_domain.create_state()

        if body_qd is not None:
            rigid_domain.state._body_qd = wp.array(
                [body_qd],
                dtype=wp.spatial_vector,
                device=fluid_model._device,
            )
        else:
            rigid_domain.state.body_qd.zero_()

        coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(
            fluid_domain, rigid_domain
        )
        coupling.add_body_sphere(body_idx=body_id, radius=sphere_radius)
        coupling.set_rigid_dynamics_enabled(False)
        coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
        coupling.set_feedback_mode("momentum_exchange")

        for _ in range(n_steps):
            coupling.step(dt=dt)

        wp.synchronize_device(fluid_model._device)
        body_force: np.ndarray = rigid_domain.state.get_body_force(body_id)
        return body_force, fluid_domain, rigid_domain, coupling

    def test_static_sphere_static_fluid_near_zero_force(self) -> None:
        """A static sphere in a static, equilibrium fluid should produce ~zero force.

        In equilibrium, incoming and outgoing distributions cancel at each
        boundary link, so the net momentum exchange should be near zero.
        An exact zero is not expected because the sphere rasterisation
        is stair-stepped, but the residual must be finite and small.
        """
        nx: int = 16
        ny: int = 16
        nz: int = 16
        dh: float = 0.1
        sphere_center: tuple[float, float, float] = (0.8, 0.8, 0.8)
        sphere_radius: float = 0.3

        body_force: np.ndarray
        body_force, _, _, _ = self._run_momentum_exchange_cycle(
            nx=nx, ny=ny, nz=nz, dh=dh,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            n_steps=5,
            initial_rho=1.0,
            initial_u=(0.0, 0.0, 0.0),
        )

        force_magnitude: float = float(np.linalg.norm(body_force[:3]))
        # Residual should be finite and small relative to expected scale
        self.assertTrue(np.isfinite(force_magnitude),
                        msg=f"Momentum-exchange force is not finite: {body_force[:3]}")
        self.assertFalse(np.isnan(force_magnitude),
                         msg=f"Momentum-exchange force is NaN: {body_force[:3]}")
        self.assertLess(force_magnitude, 1.0e-2,
                        msg=f"Static sphere in static fluid should have near-zero "
                            f"force, got |F|={force_magnitude}")

    def test_moving_sphere_static_fluid_reaction_force(self) -> None:
        """A sphere moving rightward in static fluid should feel a leftward reaction.

        The momentum-exchange method should produce a drag-like force
        opposite to the sphere motion.  The sign check is a minimal
        directional correctness test.
        """
        nx: int = 16
        ny: int = 16
        nz: int = 16
        dh: float = 0.1
        sphere_center: tuple[float, float, float] = (0.8, 0.8, 0.8)
        sphere_radius: float = 0.3
        sphere_speed: float = 0.1  # world units

        body_force: np.ndarray
        body_force, _, _, _ = self._run_momentum_exchange_cycle(
            nx=nx, ny=ny, nz=nz, dh=dh,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            n_steps=5,
            body_qd=[sphere_speed, 0.0, 0.0, 0.0, 0.0, 0.0],
            initial_rho=1.0,
            initial_u=(0.0, 0.0, 0.0),
        )

        force_x: float = float(body_force[0])
        force_magnitude: float = float(np.linalg.norm(body_force[:3]))

        # Directional correctness: sphere moves +x, fluid reaction force on sphere is -x
        self.assertTrue(np.isfinite(force_magnitude),
                        msg=f"Force not finite: {body_force[:3]}")
        self.assertFalse(np.isnan(force_magnitude),
                         msg=f"Force is NaN: {body_force[:3]}")
        self.assertLess(
            force_x, 0.0,
            msg=f"Sphere moving +x should receive -x momentum-exchange reaction, "
                f"got Fx={force_x}",
        )
        # The magnitude should be non-negligible (some reasonable nonzero)
        self.assertGreater(force_magnitude, 1.0e-6,
                           msg=f"Reaction force magnitude too small: {force_magnitude}")

    def test_eccentric_link_produces_torque(self) -> None:
        """Off-center boundary links should produce non-zero torque.

        Use a manually seeded fluid-solid pair that mimics a single
        off-center link to verify torque accumulation.
        """
        nx: int = 8
        ny: int = 8
        nz: int = 8
        dh: float = 0.1
        body_center: tuple[float, float, float] = (0.5, 0.5, 0.45)
        sphere_radius: float = 0.1
        body_id: int = 0
        dt: float = 1.0 / 60.0

        fluid_model: LbmModel = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=dh,
            tau=0.55,
        )
        fluid_domain: LbmDomain = LbmDomain(fluid_model)
        fluid_domain.create_state()
        fluid_domain.state.clear()

        rigid_builder: RigidModelBuilder = RigidModelBuilder(gravity=0.0)
        rigid_body: int = rigid_builder.add_body(
            position=body_center, label="torque_probe_momentum"
        )
        self.assertEqual(rigid_body, body_id)
        rigid_builder.add_shape_sphere(rigid_body, radius=sphere_radius)
        rigid_domain: RigidDomain = RigidDomain(
            rigid_builder.finalize(device=fluid_model._device)
        )
        rigid_domain.create_state()
        rigid_domain.state.clear_forces()

        # Manually seed one fluid cell next to one solid cell
        wp.launch(
            _seed_single_lbm_feedback_torque_case,
            dim=1,
            inputs=[
                fluid_domain.state.density,
                fluid_domain.state.velocity_x,
                fluid_domain.state.velocity_y,
                fluid_domain.state.velocity_z,
                fluid_domain.state.solid_phi,
                fluid_domain.state.solid_body_id,
            ],
        )

        # Manually set up f for the momentum-exchange kernel
        # Manually set up f for the momentum-exchange kernel
        stride: int = nx * ny * nz
        idx_fluid: int = 5 * ny * nz + 6 * nz + 4

        # We set f_pre (state_out after swap) and f_post (state_in after swap)
        # with non-equilibrium values to trigger momentum exchange
        fluid_domain._state_in.f.zero_()
        fluid_domain._state_out.f.zero_()

        # Set pre-stream f at the fluid cell for direction 2 (x-axis, -x)
        wp_idx_f_pre_2: int = 2 * stride + idx_fluid
        f_pre_val: float = 1.0 / 18.0  # use weight as base value
        wp.copy(
            fluid_domain._state_out.f,
            wp.array([f_pre_val], dtype=float, device=fluid_model._device),
            dest_offset=wp_idx_f_pre_2,
            count=1,
        )

        # Set post-stream f (direction 1 = +x) which includes bounce-back effect
        wp_idx_f_post_1: int = 1 * stride + idx_fluid
        f_post_val: float = 1.0 / 18.0 + 0.02  # small perturbation
        wp.copy(
            fluid_domain._state_in.f,
            wp.array([f_post_val], dtype=float, device=fluid_model._device),
            dest_offset=wp_idx_f_post_1,
            count=1,
        )

        rigid_backend: object = rigid_domain.model._newton_backend
        rigid_state: RigidState = rigid_domain._state_in
        rigid_state.clear_forces()

        wp.launch(
            ck.accumulate_lbm_momentum_exchange_all_bodies,
            dim=(nx, ny, nz),
            inputs=[
                fluid_domain._state_in.f,    # post-stream
                fluid_domain._state_out.f,   # pre-stream
                fluid_domain.state.solid_phi,
                fluid_domain.state.solid_body_id,
                dh,
                nx,
                ny,
                nz,
                stride,
                rigid_state.body_q,
                rigid_backend.body_com,
                rigid_state.body_f,
                1.0,
            ],
        )
        wp.synchronize_device(fluid_model._device)

        body_force: np.ndarray = rigid_domain.state.get_body_force(body_id)

        # Force in x should be non-zero (sign depends on perturbation)
        force_magnitude: float = float(np.linalg.norm(body_force[:3]))

        self.assertTrue(np.isfinite(force_magnitude),
                        msg=f"Force not finite: {body_force[:3]}")
        self.assertFalse(np.isnan(force_magnitude),
                         msg=f"Force is NaN: {body_force[:3]}")
        self.assertGreater(
            abs(force_magnitude), 1.0e-10,
            msg=f"Seeded eccentric link should produce non-zero force, got {force_magnitude}",
        )

        # Torque should be non-zero due to eccentricity
        torque_magnitude: float = float(np.linalg.norm(body_force[3:]))
        self.assertGreater(
            torque_magnitude, 1.0e-10,
            msg=f"Eccentric link should produce non-zero torque, got {torque_magnitude}",
        )

    def test_all_force_torque_finite(self) -> None:
        """Under any valid coupling condition, all force/torque components are finite."""
        random_seed: int = 42
        np.random.seed(random_seed)

        for _trial in range(3):
            nx: int = 12
            ny: int = 12
            nz: int = 12
            dh: float = 0.1
            offset_x: float = float(np.random.uniform(0.4, 0.9))
            offset_y: float = float(np.random.uniform(0.4, 0.9))
            offset_z: float = float(np.random.uniform(0.4, 0.9))
            sphere_center: tuple[float, float, float] = (offset_x, offset_y, offset_z)
            sphere_radius: float = float(np.random.uniform(0.15, 0.3))
            speed: float = float(np.random.uniform(0.0, 0.15))

            body_qd: list[float] | None = None
            if speed > 1.0e-6:
                body_qd = [speed, 0.0, 0.0, 0.0, 0.0, 0.0]

            body_force: np.ndarray
            body_force, _, _, _ = self._run_momentum_exchange_cycle(
                nx=nx, ny=ny, nz=nz, dh=dh,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                n_steps=3,
                body_qd=body_qd,
                initial_rho=1.0,
                initial_u=(0.0, 0.0, 0.0),
            )

            self.assertTrue(
                np.isfinite(body_force).all(),
                msg=f"Trial {_trial}: body_force has non-finite values: {body_force}",
            )
            self.assertFalse(
                np.isnan(body_force).any(),
                msg=f"Trial {_trial}: body_force has NaN values: {body_force}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
