# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""H5: LbmSolver home_fp32 backend hook smoke tests."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.backends.moment import home_fp32_ref


class TestHomeFp32SolverHook(unittest.TestCase):
    def test_next_increment(self) -> None:
        self.assertIn("H7", home_fp32_ref.NEXT_INCREMENT)

    def test_model_rejects_home_without_vof(self) -> None:
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(8, 8, 8),
                fluid_grid_cell_size=0.1,
                tau=0.6,
                phase_mode="none",
                lbm_backend="home_fp32",
            )

    def test_solver_home_fp32_dambreak_smoke(self) -> None:
        n = 16
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            lattice="D3Q27",
            tau=0.7,
            G=0.0,
            phase_mode="vof_sharp",
            lbm_backend="home_fp32",
            vof_rho_gas=1.0,
            vof_gamma=0.0,
            initial_density=1.0,
            gravity_z=-0.0002,
        )
        self.assertEqual(model.lbm_backend, "home_fp32")
        domain = LbmDomain(model)
        domain.create_state()
        self.assertIsNotNone(domain.solver._home_fp32)
        state = domain.state
        domain.solver._home_fp32.seed_dam_break(
            state, dam_x=n // 4, fill_z=n // 2, rho_liquid=1.0,
        )
        domain.solver._home_fp32.sync_to_state(domain._state_out)
        phi0 = float(state.phi.numpy().sum())
        for _ in range(20):
            domain.step(1.0)
        wp.synchronize_device(model._device)
        state = domain.state
        phi = state.phi.numpy()
        ctype = state.cell_type.numpy()
        self.assertTrue(np.isfinite(phi).all())
        self.assertTrue(np.isfinite(state.density.numpy()).all())
        self.assertLess(abs(float(phi.sum()) - phi0) / phi0, 0.1)
        self.assertGreater(int((ctype == 1).sum()), 0)
        self.assertGreater(int((ctype == 2).sum()), 0)

    def test_home_fp32_larger_grid_smoke(self) -> None:
        """H6: Warp path should handle ~32³ without hanging."""
        n = 32
        model = LbmModel(
            fluid_grid_res=(n, n, n),
            fluid_grid_cell_size=0.02,
            lattice="D3Q27",
            tau=0.7,
            G=0.0,
            phase_mode="vof_sharp",
            lbm_backend="home_fp32",
            vof_gamma=0.0,
            initial_density=1.0,
            gravity_z=-0.0003,
        )
        domain = LbmDomain(model)
        domain.create_state()
        domain.solver._home_fp32.seed_dam_break(
            domain.state, dam_x=n // 4, fill_z=n // 2, rho_liquid=1.0,
        )
        domain.solver._home_fp32.sync_to_state(domain._state_out)
        for _ in range(10):
            domain.step(1.0)
        wp.synchronize_device(model._device)
        self.assertTrue(np.isfinite(domain.state.phi.numpy()).all())


if __name__ == "__main__":
    unittest.main()
