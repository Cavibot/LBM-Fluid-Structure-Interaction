# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Moving cut-link wall Couette characterization for HOME-LBM."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    CutLinkBuffer,
    HomeLbmDomain,
    HomeLbmModel,
)


class TestHomeLbmCouette(unittest.TestCase):
    def _run_case(self, fraction: float, device: str = "cpu") -> None:
        shape = (8, 12, 3)
        wall_speed = 0.03
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=1.0 / 6.0,
            periodic=(True, True, True),
            device=device,
        )
        domain = HomeLbmDomain(model)
        domain.create_state()
        domain.solver.initialize_uniform_lattice(domain.state)

        phi = np.ones(shape, dtype=np.float32)
        body_id = np.full(shape, -1, dtype=np.int32)
        phi[:, 0, :] = -(1.0 - fraction)
        phi[:, 1, :] = fraction
        phi[:, -2, :] = fraction
        phi[:, -1, :] = -(1.0 - fraction)
        body_id[:, 0, :] = 0
        body_id[:, -1, :] = 1
        wp.copy(domain.state.solid_phi, wp.array(phi, dtype=float, device=device))
        wp.copy(domain.state.solid_body_id, wp.array(body_id, dtype=wp.int32, device=device))

        links = CutLinkBuffer.build_from_sdf(domain.state)
        body_q = wp.array(
            [
                wp.transform(wp.vec3(), wp.quat_identity()),
                wp.transform(wp.vec3(), wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_qd = wp.array(
            [
                [wall_speed, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-wall_speed, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_com = wp.zeros(2, dtype=wp.vec3, device=device)
        links.set_rigid_wall_velocity(body_q, body_qd, body_com, 1.0, 1.0)
        links.validate_body_ids()
        domain.solver.set_cut_links(links)

        for _ in range(1800):
            domain.step(model.time_step)
        domain.solver.validate_state(domain.state)
        wp.synchronize_device(device)

        moments = domain.state.moments.numpy().reshape(10, -1)
        density = moments[0].reshape(shape)
        velocity_x = (moments[1] / moments[0]).reshape(shape)
        measured = velocity_x[:, 1:-1, :].mean(axis=(0, 2))
        cell_centers = np.arange(1, shape[1] - 1, dtype=np.float64) + 0.5
        lower_wall = 1.5 - fraction
        upper_wall = shape[1] - 1.5 + fraction
        expected = wall_speed - 2.0 * wall_speed * (
            (cell_centers - lower_wall) / (upper_wall - lower_wall)
        )
        np.testing.assert_allclose(measured, expected, atol=7.5e-4)
        np.testing.assert_allclose(density[:, 1:-1, :], 1.0, atol=2.0e-5)

        impulse = links.impulse.numpy()
        bodies = links.body_id.numpy()
        lower_force = impulse[bodies == 0].sum(axis=0)
        upper_force = impulse[bodies == 1].sum(axis=0)
        self.assertLess(lower_force[0], 0.0)
        self.assertGreater(upper_force[0], 0.0)
        np.testing.assert_allclose(lower_force + upper_force, 0.0, atol=2.0e-5)

    def test_interpolated_boundary_tracks_sdf_wall_and_balances_shear(self) -> None:
        for fraction in (0.25, 0.5, 0.75):
            with self.subTest(fraction=fraction):
                self._run_case(fraction)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_interpolated_boundary_runs_on_cuda(self) -> None:
        self._run_case(0.25, device="cuda:0")


if __name__ == "__main__":
    unittest.main()
