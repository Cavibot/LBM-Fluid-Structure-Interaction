# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Discrete hydrostatic initialization for forced HOME-LBM domains."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeLbmDomain, HomeLbmModel


class TestHomeLbmHydrostatic(unittest.TestCase):
    def test_hydrostatic_initializer_preserves_rest_state(self) -> None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        acceleration = -2.0e-4
        model = HomeLbmModel(
            fluid_grid_res=(12, 10, 16),
            kinematic_viscosity=0.04,
            body_acceleration=(0.0, 0.0, acceleration),
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmDomain(model)
        fluid.create_state()
        fluid.solver.initialize_hydrostatic_lattice(fluid.state)
        lattice_acceleration = model.lattice_acceleration[2]
        initial = fluid.state.moments.numpy().reshape(10, -1)
        initial_density = initial[0].copy()

        np.testing.assert_allclose(
            initial[3] / initial[0],
            0.5 * lattice_acceleration,
            atol=2.0e-8,
        )
        self.assertAlmostEqual(float(np.mean(initial_density)), 1.0, places=6)
        ratio = (1.0 + 1.5 * lattice_acceleration) / (
            1.0 - 1.5 * lattice_acceleration
        )
        density = initial_density.reshape(model.fluid_grid_res)
        np.testing.assert_allclose(density[:, :, 1:] / density[:, :, :-1], ratio, atol=2.0e-7)

        for _ in range(200):
            fluid.step(model.time_step)

        final = fluid.state.moments.numpy().reshape(10, -1)
        np.testing.assert_allclose(final[0], initial_density, rtol=4.0e-4, atol=4.0e-5)
        np.testing.assert_allclose(
            final[3] / final[0],
            0.5 * lattice_acceleration,
            atol=2.0e-5,
        )

    def test_hydrostatic_initializer_rejects_forced_periodic_axis(self) -> None:
        model = HomeLbmModel(
            fluid_grid_res=(4, 4, 4),
            body_acceleration=(0.0, 0.0, -1.0e-4),
            periodic=(True, True, True),
            device="cpu",
        )
        fluid = HomeLbmDomain(model)
        fluid.create_state()
        with self.assertRaisesRegex(ValueError, "periodic"):
            fluid.solver.initialize_hydrostatic_lattice(fluid.state)


if __name__ == "__main__":
    unittest.main()
