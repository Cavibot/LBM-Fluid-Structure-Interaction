# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Wall-bounded validation cases for HOME-LBM."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeLbmDomain, HomeLbmModel

from .test_home_lbm_periodic import _download_moments


class TestHomeLbmChannel(unittest.TestCase):
    def test_halfway_wall_poiseuille_matches_analytic_profile(self) -> None:
        height = 24
        viscosity = 0.1
        acceleration = 1.0e-5
        model = HomeLbmModel(
            fluid_grid_res=(1, height, 1),
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=viscosity,
            body_acceleration=(acceleration, 0.0, 0.0),
            periodic=(True, False, True),
            device="cpu",
        )
        domain = HomeLbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_uniform_lattice(state)
        initial_mass = float(height)

        for _ in range(4000):
            domain.step(1.0)
        wp.synchronize_device("cpu")

        moments = _download_moments(domain.state)
        density = moments[0, :, 0, 0]
        velocity_x = moments[0, :, 0, 1] / density
        velocity_y = moments[0, :, 0, 2] / density
        velocity_z = moments[0, :, 0, 3] / density
        wall_distance = np.arange(height, dtype=np.float64) + 0.5
        analytic = acceleration * wall_distance * (height - wall_distance) / (2.0 * viscosity)
        relative_l2 = np.linalg.norm(velocity_x - analytic) / np.linalg.norm(analytic)

        self.assertLess(relative_l2, 3.0e-3)
        self.assertLess(abs(float(density.sum()) - initial_mass), 7.0e-4)
        self.assertLess(float(np.max(np.abs(velocity_y))), 1.0e-6)
        self.assertLess(float(np.max(np.abs(velocity_z))), 1.0e-6)
        domain.solver.validate_state(domain.state)


if __name__ == "__main__":
    unittest.main()
