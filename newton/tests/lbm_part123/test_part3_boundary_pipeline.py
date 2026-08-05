"""Part 3 acceptance tests for completion-before-collection boundaries."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import FullFLbmState, LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.solver import kernels, streaming
from wanphys._src.fluid.fluid_grid.lbm.constants import (
    BC_BOUNCE_BACK,
    BC_OUTFLOW,
    BC_PRESSURE,
    BC_VELOCITY_INLET,
)


class TestPart3BoundaryPipeline(unittest.TestCase):
    @staticmethod
    def _model(encoding: str = "fullf", collision: str = "srt") -> LbmModel:
        return LbmModel(
            fluid_grid_res=(6, 5, 5),
            device="cpu",
            encoding=encoding,
            collision=collision,
            tau=0.8,
            bc_types=(
                BC_VELOCITY_INLET,
                BC_OUTFLOW,
                BC_BOUNCE_BACK,
                BC_BOUNCE_BACK,
                BC_BOUNCE_BACK,
                BC_BOUNCE_BACK,
            ),
            bc_velocity=(
                (0.02, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            bc_convective_speed=(0.5, 0.25, 0.5, 0.5, 0.5, 0.5),
        )

    def test_boundary_completion_precedes_moment_collection(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            domain = LbmDomain(self._model())
        state = domain.create_state()
        domain.solver.initialize_equilibrium(state, rho0=1.0)
        domain.step(1.0)

        solver = domain.solver
        f_star = solver._f_star.numpy().reshape(19, -1)
        i, j, k = 0, 2, 2  # x-min face interior, not an edge/corner
        idx = i * solver.ny * solver.nz + j * solver.nz + k
        collected_density = float(solver._moments[0].numpy()[i, j, k])
        collected_jx = float(solver._moments[1].numpy()[i, j, k])
        expected_density = float(np.sum(f_star[:, idx], dtype=np.float64))
        expected_jx = float(
            f_star[1, idx] - f_star[2, idx]
            + f_star[7, idx] - f_star[8, idx]
            + f_star[9, idx] - f_star[10, idx]
            + f_star[11, idx] - f_star[12, idx]
            + f_star[13, idx] - f_star[14, idx]
        )
        self.assertAlmostEqual(collected_density, expected_density, places=6)
        self.assertAlmostEqual(collected_jx, expected_jx, places=6)
        self.assertAlmostEqual(collected_jx / collected_density, 0.02, places=5)

    def test_zou_he_closes_the_prescribed_normal_velocity_on_all_six_faces(self) -> None:
        face_cells = (
            (0, (0, 2, 2), 0, 0.01),
            (1, (4, 2, 2), 0, -0.01),
            (2, (2, 0, 2), 1, 0.01),
            (3, (2, 4, 2), 1, -0.01),
            (4, (2, 2, 0), 2, 0.01),
            (5, (2, 2, 4), 2, -0.01),
        )
        for face, cell, axis, prescribed in face_cells:
            with self.subTest(face=face):
                types = [BC_BOUNCE_BACK] * 6
                types[face] = BC_VELOCITY_INLET
                velocities = [(0.0, 0.0, 0.0)] * 6
                velocity = [0.0, 0.0, 0.0]
                velocity[axis] = prescribed
                velocities[face] = tuple(velocity)
                model = LbmModel(
                    fluid_grid_res=(5, 5, 5),
                    device="cpu",
                    bc_types=tuple(types),
                    bc_velocity=tuple(velocities),
                    tau=0.8,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    domain = LbmDomain(model)
                state = domain.create_state()
                domain.solver.initialize_equilibrium(state, rho0=1.0)
                domain.step(1.0)
                rho = float(domain.solver._moments[0].numpy()[cell])
                momentum = float(domain.solver._moments[1 + axis].numpy()[cell])
                self.assertAlmostEqual(momentum / rho, prescribed, places=5)

    def test_convective_outlet_owns_previous_boundary_state(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            domain = LbmDomain(self._model())
        state = domain.create_state()
        domain.solver.initialize_equilibrium(state, rho0=1.0)
        self.assertFalse(domain.solver._boundary_history_ready)
        domain.step(1.0)
        self.assertTrue(domain.solver._boundary_history_ready)
        np.testing.assert_allclose(
            domain.solver._boundary_history.numpy(),
            domain.solver._f_star.numpy(),
            atol=0.0,
            rtol=0.0,
        )

    def test_pressure_completion_sets_prescribed_density_before_collection(self) -> None:
        model = self._model()
        model.bc_types = (
            BC_PRESSURE,
            BC_OUTFLOW,
            BC_BOUNCE_BACK,
            BC_BOUNCE_BACK,
            BC_BOUNCE_BACK,
            BC_BOUNCE_BACK,
        )
        model.bc_density = (1.05, 1.0, 1.0, 1.0, 1.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            domain = LbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_equilibrium(state, rho0=1.0)
        domain.step(1.0)
        self.assertAlmostEqual(float(domain.solver._moments[0].numpy()[0, 2, 2]), 1.05, places=5)

    def test_corner_conflicts_warn_once_per_boundary_resolution(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            LbmDomain(self._model())
        messages = [str(item.message) for item in caught if item.category is RuntimeWarning]
        self.assertEqual(len(messages), 1)
        self.assertIn("forced to static bounce-back", messages[0])
        self.assertIn("representative cells", messages[0])

    def test_population_completion_is_reused_by_home_and_mrt_paths(self) -> None:
        paths = (("home", "srt"), ("home", "nocm_mrt"), ("fullf", "raw_mrt"), ("fullf", "nocm_mrt"))
        for encoding, collision in paths:
            with self.subTest(encoding=encoding, collision=collision):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    domain = LbmDomain(self._model(encoding, collision))
                state = domain.create_state()
                domain.solver.initialize_equilibrium(state, rho0=1.0)
                domain.step(1.0)
                self.assertTrue(np.all(np.isfinite(domain.state.density.numpy())))

    def test_cut_link_interpolates_the_reflected_population(self) -> None:
        model = LbmModel(
            fluid_grid_res=(3, 3, 3),
            device="cpu",
            use_cut_link=True,
            bc_periodic=(True, True, True),
        )
        state = FullFLbmState(model)
        stride = 27
        f_post = np.zeros((19, 3, 3, 3), dtype=np.float32)
        # q=1 pulls from x-1.  At x=1 that source is solid; q=2 is reflected.
        f_post[2, 1, 1, 1] = 2.0
        f_post[2, 2, 1, 1] = 4.0
        state.f_post.assign(f_post.reshape(-1))
        phi = np.full((3, 3, 3), 10.0, dtype=np.float32)
        phi[1, 1, 1] = 0.25
        phi[0, 1, 1] = -1.0  # link fraction = 0.2
        state.solid_phi.assign(phi)
        f_star = wp.zeros(19 * stride, dtype=float, device="cpu")
        bc_types = wp.zeros(6, dtype=wp.int32, device="cpu")
        wp.launch(
            streaming.stream_fullf_to_populations_kernel,
            dim=(3, 3, 3),
            inputs=[state.f_post, state.solid_phi, f_star, 1, 1, 1, 3, 3, 3, stride],
            device="cpu",
        )
        wp.launch(
            streaming.apply_fullf_cut_link_transport_kernel,
            dim=(3, 3, 3),
            inputs=[state.f_post, state.solid_phi, f_star, bc_types, 1, 1, 1, 3, 3, 3, stride],
            device="cpu",
        )
        actual = f_star.numpy().reshape(19, 3, 3, 3)
        # 2*q*f_opp(x) + (1-2*q)*f_opp(x+c) = 0.4*2 + 0.6*4.
        self.assertAlmostEqual(float(actual[1, 1, 1, 1]), 3.2, places=6)

    def test_moving_wall_correction_is_applied_during_transport(self) -> None:
        model = LbmModel(fluid_grid_res=(3, 3, 3), device="cpu", has_moving_walls=True)
        state = FullFLbmState(model)
        LbmDomain(model).solver.initialize_equilibrium(state, rho0=1.0)
        wall_u = np.zeros((4, 3, 3), dtype=np.float32)
        wall_u[0, :, :] = 0.1
        state.vel_solid_u.assign(wall_u)
        f_star = wp.zeros(19 * 27, dtype=float, device="cpu")
        wp.launch(
            streaming.stream_fullf_to_populations_kernel,
            dim=(3, 3, 3),
            inputs=[state.f_post, state.solid_phi, f_star, 0, 0, 0, 3, 3, 3, 27],
            device="cpu",
        )
        baseline = float(f_star.numpy().reshape(19, 3, 3, 3)[1, 0, 1, 1])
        bc_types = wp.zeros(6, dtype=wp.int32, device="cpu")
        wp.launch(
            kernels.apply_moving_wall_transport_kernel,
            dim=(3, 3, 3),
            inputs=[
                f_star,
                state.density,
                state.solid_phi,
                state.vel_solid_u,
                state.vel_solid_v,
                state.vel_solid_w,
                bc_types,
                0,
                0,
                0,
                0,
                3,
                3,
                3,
                27,
            ],
            device="cpu",
        )
        corrected = float(f_star.numpy().reshape(19, 3, 3, 3)[1, 0, 1, 1])
        self.assertAlmostEqual(corrected - baseline, 2.0 * (1.0 / 18.0) * 3.0 * 0.1, places=6)

    def test_open_corner_forces_static_wall_even_when_moving_wall_is_enabled(self) -> None:
        model = LbmModel(fluid_grid_res=(3, 3, 3), device="cpu", has_moving_walls=True)
        state = FullFLbmState(model)
        LbmDomain(model).solver.initialize_equilibrium(state, rho0=1.0)
        wall_u = np.zeros((4, 3, 3), dtype=np.float32)
        wall_u[0, :, :] = 0.1
        state.vel_solid_u.assign(wall_u)
        f_star = wp.zeros(19 * 27, dtype=float, device="cpu")
        wp.launch(
            streaming.stream_fullf_to_populations_kernel,
            dim=(3, 3, 3),
            inputs=[state.f_post, state.solid_phi, f_star, 0, 0, 0, 3, 3, 3, 27],
            device="cpu",
        )
        before = float(f_star.numpy().reshape(19, 3, 3, 3)[1, 0, 0, 1])
        bc_types = wp.array(
            np.array([BC_VELOCITY_INLET, BC_OUTFLOW, 0, 0, 0, 0], dtype=np.int32),
            dtype=wp.int32,
            device="cpu",
        )
        wp.launch(
            kernels.apply_moving_wall_transport_kernel,
            dim=(3, 3, 3),
            inputs=[
                f_star,
                state.density,
                state.solid_phi,
                state.vel_solid_u,
                state.vel_solid_v,
                state.vel_solid_w,
                bc_types,
                0,
                0,
                0,
                0,
                3,
                3,
                3,
                27,
            ],
            device="cpu",
        )
        after = float(f_star.numpy().reshape(19, 3, 3, 3)[1, 0, 0, 1])
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
