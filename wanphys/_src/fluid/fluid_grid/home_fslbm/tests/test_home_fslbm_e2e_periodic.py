# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 end-to-end test: periodic boundary small grid.

Covers criteria:
    B.8  — Minimal periodic grid (8,8,8) runs 100 steps without crash/NaN
    B.9  — Domain double-buffer consistency (odd/even step check)
    B.10 — create_state() returns independent states
    B.11 — Solver skeleton correctly wired through domain.step()
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFSLbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFSLbmState
from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFSLbmDomain
from wanphys._src.fluid.fluid_grid.home_fslbm.constants import TYPE_F, TYPE_G, TYPE_S


class TestEndToEndPeriodic(unittest.TestCase):
    """Acceptance criteria B.8, B.9, B.10, B.11."""

    def setUp(self):
        wp.init()
        self.res = (8, 8, 8)
        self.model = HomeFSLbmModel(
            fluid_grid_res=self.res,
            fluid_grid_cell_size=1.0,
            tau=0.55,
            bc_types=("periodic",) * 6,
        )

    def _init_uniform_fluid(self, state: HomeFSLbmState):
        """Set all cells to TYPE_F with rho=1, u=0, S=0."""
        nx, ny, nz = self.res
        tn = state.total_num

        # Set flag: TYPE_F everywhere
        flag_np = np.full((nx, ny, nz), TYPE_F, dtype=np.int32)
        wp.copy(state.flag, wp.array(flag_np, dtype=wp.int32, device=self.model._device))

        # Set moments: rho=1, u=0, S=0
        f_np = np.zeros(10 * tn, dtype=np.float32)
        f_np[0 * tn:1 * tn] = 1.0
        wp.copy(state.f_mom, wp.array(f_np, dtype=float, device=self.model._device))

        # Also copy to f_mom_post to avoid uninitialised reads
        wp.copy(state.f_mom_post, wp.array(f_np, dtype=float, device=self.model._device))

        # Set mass/phi for fluid cells
        mass_np = np.ones((nx, ny, nz), dtype=np.float32)
        phi_np = np.ones((nx, ny, nz), dtype=np.float32)
        wp.copy(state.mass, wp.array(mass_np, dtype=float, device=self.model._device))
        wp.copy(state.phi, wp.array(phi_np, dtype=float, device=self.model._device))
        wp.synchronize()

    def test_minimal_grid_runs(self):
        """B.8: (8,8,8) periodic grid, 100 steps, no crash, no NaN, max|u|<1e-10."""
        domain = HomeFSLbmDomain(self.model)
        domain.create_state()
        self._init_uniform_fluid(domain.state)

        for step in range(100):
            domain.step(dt=1.0)

        # Verify final state
        wp.synchronize()
        st = domain.state
        f_np = st.f_mom.numpy()
        tn = st.total_num

        rho = f_np[0 * tn:1 * tn]
        ux = f_np[1 * tn:2 * tn]
        uy = f_np[2 * tn:3 * tn]
        uz = f_np[3 * tn:4 * tn]

        # No NaN
        self.assertFalse(np.any(np.isnan(rho)), "NaN in rho")
        self.assertFalse(np.any(np.isnan(ux)), "NaN in ux")
        self.assertFalse(np.any(np.isnan(uy)), "NaN in uy")
        self.assertFalse(np.any(np.isnan(uz)), "NaN in uz")

        # No Inf
        self.assertFalse(np.any(np.isinf(rho)), "Inf in rho")

        # Density non-negative
        self.assertTrue(np.all(rho >= 0), "Negative density detected")

        # Velocity near zero (at rest, no forces)
        max_u = max(np.max(np.abs(ux)), np.max(np.abs(uy)), np.max(np.abs(uz)))
        self.assertLess(max_u, 1e-10, f"max|u| = {max_u}, expected < 1e-10")

    def test_double_buffer_consistency(self):
        """B.9: Domain double-buffer flips correctly each step."""
        domain = HomeFSLbmDomain(self.model)
        domain.create_state()
        self._init_uniform_fluid(domain.state)

        # Record initial state_in identity
        state_ids = []
        for step in range(6):
            sid = id(domain.state)
            state_ids.append(sid)
            domain.step(dt=1.0)

        # Odd steps should have different state_in than even steps
        for i in range(2, len(state_ids), 2):
            self.assertEqual(state_ids[i], state_ids[i - 2],
                             f"Step {i}: buffer should match step {i-2}")
        for i in range(3, len(state_ids), 2):
            self.assertEqual(state_ids[i], state_ids[i - 2],
                             f"Step {i}: buffer should match step {i-2}")
        self.assertNotEqual(state_ids[0], state_ids[1],
                            "Step 0 and 1 should use different state objects")

    def test_create_state_independent(self):
        """B.10: Two calls to create_state() return independent states."""
        domain = HomeFSLbmDomain(self.model)
        s1 = domain.create_state()
        s2 = domain.create_state()

        # Different objects
        self.assertIsNot(s1, s2)
        self.assertIsNot(s1.f_mom, s2.f_mom)

        # Modify s1, verify s2 is unaffected
        f_np = s1.f_mom.numpy()
        f_np[0] = 999.0
        wp.copy(s1.f_mom, wp.array(f_np, dtype=float, device=self.model._device))
        wp.synchronize()

        s2_np = s2.f_mom.numpy()
        self.assertEqual(s2_np[0], 0.0,
                         "s2.f_mom should be independent of s1")

    def test_solver_wired_correctly(self):
        """B.11: solver.step() is called by domain.step() without exception."""
        domain = HomeFSLbmDomain(self.model)
        domain.create_state()
        self._init_uniform_fluid(domain.state)

        # Run a single step — should not raise
        try:
            domain.step(dt=1.0)
        except Exception as exc:
            self.fail(f"domain.step() raised {exc}")

        # Verify state is accessible after step
        st = domain.state
        self.assertIsNotNone(st)
        self.assertEqual(st.total_num, 512)

    def test_mass_conservation_through_steps(self):
        """Verify that mass stays close to initial through LBM steps.

        Stage 2 stream-collide produces float32 rounding drift ~1e-3.
        """
        domain = HomeFSLbmDomain(self.model)
        domain.initialize(rho0=1.0)

        mass_before = domain.solver.total_mass(domain.state)
        cell_count = 8 * 8 * 8
        self.assertAlmostEqual(mass_before, float(cell_count), delta=1e-6)

        for _ in range(10):
            domain.step(dt=1.0)

        mass_after = domain.solver.total_mass(domain.state)
        drift = abs(mass_after - mass_before)
        self.assertLess(
            drift, 2e-3,
            f"Mass drift {drift:.3e} exceeds float32 tolerance",
        )


if __name__ == "__main__":
    unittest.main()
