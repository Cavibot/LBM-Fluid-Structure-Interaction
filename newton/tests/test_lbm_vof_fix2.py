# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""FIX2 isolated-interface closure and retained residual acceptance."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    LbmDomain,
    LbmModel,
    VofCellType,
    VofTopologyTransition,
    validate_vof_diagnostics,
)


class TestVofFix2TopologyClosure(unittest.TestCase):
    def test_epoch_832_pattern_retires_the_last_interface_pair(self) -> None:
        shape = (5, 5, 5)
        density = np.ones(shape, dtype=np.float32)
        mass = np.zeros(shape, dtype=np.float32)
        cell_type = np.full(shape, int(VofCellType.GAS), dtype=np.uint8)
        center = (2, 2, 2)
        q4_neighbor = (2, 3, 2)
        cell_type[center] = int(VofCellType.INTERFACE)
        cell_type[q4_neighbor] = int(VofCellType.INTERFACE)
        mass[center] = np.float32(-4.45738478e-5)
        mass[q4_neighbor] = np.float32(-2.0e-4)
        transition = VofTopologyTransition(
            shape,
            wp.get_device("cpu"),
            (0, 0, 0),
            1.0e-4,
        )

        result = transition.compute(
            wp.array(mass, dtype=float, device="cpu"),
            wp.array(density, dtype=float, device="cpu"),
            wp.array(cell_type, dtype=wp.uint8, device="cpu"),
        )

        for cell in (center, q4_neighbor):
            self.assertEqual(
                int(result.proposed_type.numpy()[cell]),
                int(VofCellType.GAS),
            )
            self.assertEqual(
                int(result.final_type.numpy()[cell]),
                int(VofCellType.GAS),
            )
            self.assertEqual(int(result.receiver_count.numpy()[cell]), 0)
        self.assertAlmostEqual(
            float(result.excess.numpy()[center]),
            float(mass[center]),
            places=10,
        )
        np.testing.assert_array_equal(result.phi_final.numpy(), 0.0)


class TestVofFix2ResidualLifetime(unittest.TestCase):
    def test_material_zero_receiver_residual_survives_two_steps(self) -> None:
        shape = (7, 3, 3)
        model = LbmModel(
            fluid_grid_res=shape,
            device="cpu",
            encoding="fullf",
            collision="srt",
            interface_model="vof",
            vof_runtime_profile="device",
            gravity_z=0.0,
            enforce_population_positivity=False,
        )
        domain = LbmDomain(model)
        phi = np.zeros(shape, dtype=np.float32)
        phi[:2] = 1.0
        phi[2] = 0.5
        state = domain.initialize_vof(phi)
        assert state.vof is not None
        source = (6, 1, 1)
        retained = np.float32(-4.45738478e-5)
        pending = state.vof.pending_excess.numpy().copy()
        pending[source] = retained
        state.vof.pending_excess.assign(pending)
        state.vof.reference_mass += float(retained)
        assert domain._state_out is not None
        assert domain._state_out.vof is not None
        domain._state_out.vof.reference_mass = state.vof.reference_mass

        for expected_epoch in (1, 2):
            domain.step(1.0)
            assert domain.state.vof is not None
            self.assertEqual(domain.state.vof.epoch, expected_epoch)
            self.assertAlmostEqual(
                float(domain.state.vof.pending_excess.numpy()[source]),
                float(retained),
                places=10,
            )
            self.assertEqual(
                int(domain.state.vof.pending_receiver_count.numpy()[source]),
                0,
            )
            report = domain.solver.last_vof_diagnostics
            assert report is not None
            self.assertGreaterEqual(report.pending_zero_receiver_count, 1)
            self.assertEqual(report.invalid_pending_excess_count, 0)
            self.assertLessEqual(report.relative_mass_error, 5.0e-6)
            validate_vof_diagnostics(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
