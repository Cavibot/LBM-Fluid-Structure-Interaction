# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HOME-LBM H2: walls (Eq. 24) + Zou–He domain BC."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    NEXT_INCREMENT,
    HomeDomainBC,
    HomeFaceBC,
    HomeFaceKind,
    make_uniform_equilibrium,
    solid_moments_eq24,
    step_domain_numpy,
    step_periodic_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.lattice import get_lattice_spec
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    zou_he_velocity_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.hermite import reconstruct_f_numpy


class TestHomeSolidEq24(unittest.TestCase):
    def test_stationary_wall_keeps_neq(self) -> None:
        """S^p = 0 + (S − uu) when u^p = 0."""
        rho, ux, uy, uz = 1.0, 0.1, 0.0, 0.0
        sxx = ux * ux + 0.01
        m = solid_moments_eq24(
            rho, ux, uy, uz, sxx, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        )
        self.assertAlmostEqual(m[0], rho)
        self.assertAlmostEqual(m[1], 0.0)
        self.assertAlmostEqual(m[4], 0.01)  # sxx_p = neq only


class TestHomeWallBox(unittest.TestCase):
    def test_closed_box_rest_stable(self) -> None:
        field = make_uniform_equilibrium((5, 5, 5), rho0=1.0)
        bc = HomeDomainBC.all_walls()
        out = field
        for _ in range(8):
            out = step_domain_numpy(out, lattice="D3Q27", tau=0.7, domain_bc=bc)
        fluid = out.rho > 0.0
        self.assertTrue(np.allclose(out.rho[fluid], 1.0, atol=1e-8))
        self.assertTrue(np.allclose(out.ux[fluid], 0.0, atol=1e-8))
        self.assertTrue(np.allclose(out.uy[fluid], 0.0, atol=1e-8))

    def test_lid_driven_cavity_develops_flow(self) -> None:
        """Moving top wall (HOME Eq. 24) should induce non-zero bulk velocity."""
        field = make_uniform_equilibrium((8, 8, 4), rho0=1.0)
        bc = HomeDomainBC.all_walls(lid_face="ymax", lid_ux=0.08)
        out = field
        for _ in range(40):
            out = step_domain_numpy(out, lattice="D3Q27", tau=0.8, domain_bc=bc)
        # Interior kinetic energy
        ke = float(np.mean(out.ux[1:-1, 1:-1, 1:-1] ** 2))
        self.assertGreater(ke, 1e-6)
        self.assertFalse(np.any(np.isnan(out.ux)))
        # Top fluid row (under moving ymax lid) should have positive ux
        self.assertGreater(float(np.mean(out.ux[:, -1, :])), 0.01)

    def test_interior_solid_block(self) -> None:
        field = make_uniform_equilibrium((6, 6, 6), rho0=1.0)
        solid = np.zeros((6, 6, 6), dtype=bool)
        solid[2:4, 2:4, 2:4] = True
        bc = HomeDomainBC.all_walls()
        out = step_domain_numpy(
            field, lattice="D3Q27", tau=0.6, domain_bc=bc, solid=solid,
        )
        self.assertTrue(np.allclose(out.rho[solid], 0.0))
        self.assertTrue(np.all(out.rho[~solid] > 0.5))


class TestHomeZouHe(unittest.TestCase):
    def test_zou_he_recovers_prescribed_velocity(self) -> None:
        """After ZH on a face-like population, macro u matches BC (approx)."""
        spec = get_lattice_spec("D3Q27")
        cx = np.asarray(spec.cx, dtype=np.float64)
        cy = np.asarray(spec.cy, dtype=np.float64)
        cz = np.asarray(spec.cz, dtype=np.float64)
        w = np.asarray(spec.weights, dtype=np.float64)
        # Start from rest eq, then ZH xmin with ux=0.05
        f = reconstruct_f_numpy(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cx, cy, cz, w,
        )
        # Zero unknown (+x) to simulate missing pull from outside
        for d in range(spec.num_dirs):
            if int(spec.cx[d]) > 0:
                f[d] = 0.0
        f2 = zou_he_velocity_numpy(f, spec, 1, 0, 0, 0.05, 0.0, 0.0)
        rho = float(np.sum(f2))
        ux = float(np.dot(cx, f2) / rho)
        self.assertAlmostEqual(ux, 0.05, places=5)

    def test_channel_zou_he_smoke(self) -> None:
        field = make_uniform_equilibrium((8, 4, 4), rho0=1.0, ux0=0.02)
        bc = HomeDomainBC.channel_x(0.02, periodic_y=True, periodic_z=True)
        out = field
        for _ in range(15):
            out = step_domain_numpy(out, lattice="D3Q27", tau=0.7, domain_bc=bc)
        self.assertFalse(np.any(np.isnan(out.rho)))
        self.assertGreater(float(np.mean(out.ux)), 0.0)
        # Inlet face cells should stay near prescribed u
        self.assertAlmostEqual(float(np.mean(out.ux[0, :, :])), 0.02, places=2)

    def test_next_increment(self) -> None:
        self.assertIn("H7", NEXT_INCREMENT)

    def test_periodic_wrapper_unchanged(self) -> None:
        field = make_uniform_equilibrium((4, 4, 4), rho0=1.0)
        a = step_periodic_numpy(field, lattice="D3Q27", tau=0.6)
        b = step_domain_numpy(
            field, lattice="D3Q27", tau=0.6, domain_bc=HomeDomainBC(),
        )
        self.assertTrue(np.allclose(a.rho, b.rho))
        self.assertTrue(np.allclose(a.ux, b.ux))


if __name__ == "__main__":
    unittest.main()
