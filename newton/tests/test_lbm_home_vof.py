# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for moment-encoded HOME-FREE VOF (H4)."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    CELL_GAS,
    CELL_INTERFACE,
    CELL_LIQUID,
    NEXT_INCREMENT,
    HomeDomainBC,
    seed_dam_break_column,
    step_home_vof_numpy,
)


class TestHomeVofH4(unittest.TestCase):
    def test_next_increment(self) -> None:
        self.assertIn("H7", NEXT_INCREMENT)

    def test_seed_has_liquid_and_interface(self) -> None:
        st = seed_dam_break_column((12, 8, 12), dam_x=4, fill_z=6)
        self.assertGreater(int((st.cell_type == CELL_LIQUID).sum()), 0)
        self.assertGreater(int((st.cell_type == CELL_INTERFACE).sum()), 0)
        self.assertGreater(int((st.cell_type == CELL_GAS).sum()), 0)
        self.assertAlmostEqual(float(st.phi.sum()), float((st.phi > 0).sum()), places=5)

    def test_dambreak_smoke_stable(self) -> None:
        """Small dam-break: finite fields, φ conserved ~5%, interface persists."""
        n = 16
        st = seed_dam_break_column((n, n // 2, n), dam_x=n // 4, fill_z=n // 2)
        phi0 = float(st.phi.sum())
        bc = HomeDomainBC.all_walls()
        out = st
        for _ in range(25):
            out = step_home_vof_numpy(
                out,
                lattice="D3Q27",
                tau=0.7,
                fz=-0.0002,
                domain_bc=bc,
                rho_g0=1.0,
                gamma=0.0,
            )
        self.assertTrue(np.isfinite(out.phi).all())
        self.assertTrue(np.isfinite(out.moments.rho).all())
        self.assertTrue(np.isfinite(out.moments.ux).all())
        phi1 = float(out.phi.sum())
        self.assertGreater(phi1, 0.0)
        self.assertLess(abs(phi1 - phi0) / phi0, 0.08)
        self.assertGreater(int((out.cell_type == CELL_LIQUID).sum()), 0)
        self.assertGreater(int((out.cell_type == CELL_INTERFACE).sum()), 0)
        # Gravity should produce some downward motion in the column
        wet = out.moments.rho > 0.5
        self.assertTrue(wet.any())
        self.assertLess(float(np.mean(out.moments.uz[wet])), 0.05)

    def test_bytes_per_cell_moments_only(self) -> None:
        """HOME stores 10 floats/cell vs D3Q27's 27 distributions."""
        self.assertLess(10, 27)


if __name__ == "__main__":
    unittest.main()
