# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl.flags import (
    FslCellFlag,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl.topology_reference import (
    _receive_queued_excess,
    resolve_fsl_topology,
)


def _equilibrium_moments(
    shape: tuple[int, int, int],
    rho: float = 1.0,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    moments = np.zeros(shape + (10,), dtype=np.float64)
    u = np.asarray(velocity, dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho * u
    moments[..., 4] = rho * u[0] * u[0]
    moments[..., 5] = rho * u[1] * u[1]
    moments[..., 6] = rho * u[2] * u[2]
    moments[..., 7] = rho * u[0] * u[1]
    moments[..., 8] = rho * u[0] * u[2]
    moments[..., 9] = rho * u[1] * u[2]
    return moments


def _layered_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (4, 3, 3)
    moments = _equilibrium_moments(shape)
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[0] = int(FslCellFlag.LIQUID)
    flags[1] = int(FslCellFlag.INTERFACE)
    mass = np.zeros(shape, dtype=np.float64)
    mass[0] = 1.0
    mass[1] = 0.5
    excess = np.zeros(shape, dtype=np.float64)
    return moments, flags, mass, excess


class TestFslTopologyReference(unittest.TestCase):
    def test_advancing_interface_creates_fresh_interface_shell(self) -> None:
        moments, flags, mass, excess = _layered_fields()
        mass[1] = 1.1
        result = resolve_fsl_topology(
            moments, flags, mass, excess, periodic=(False, False, False)
        )

        np.testing.assert_array_equal(result.flags[1], int(FslCellFlag.LIQUID))
        np.testing.assert_array_equal(result.flags[2], int(FslCellFlag.INTERFACE))
        np.testing.assert_array_equal(result.fill_level[2], 0.0)
        self.assertEqual(result.diagnostics.interface_to_liquid_count, 9)
        self.assertEqual(result.diagnostics.gas_to_interface_count, 9)
        self.assertEqual(result.diagnostics.direct_liquid_gas_link_count, 0)
        self.assertAlmostEqual(result.diagnostics.relative_total_mass_drift, 0.0, places=14)

    def test_receding_interface_demotes_neighbor_liquid(self) -> None:
        moments, flags, mass, excess = _layered_fields()
        mass[1] = -0.1
        result = resolve_fsl_topology(
            moments, flags, mass, excess, periodic=(False, False, False)
        )

        np.testing.assert_array_equal(result.flags[0], int(FslCellFlag.INTERFACE))
        np.testing.assert_array_equal(result.flags[1], int(FslCellFlag.GAS))
        np.testing.assert_array_equal(result.fill_level[0], 1.0)
        self.assertEqual(result.diagnostics.interface_to_gas_count, 9)
        self.assertEqual(result.diagnostics.liquid_to_interface_count, 9)
        self.assertAlmostEqual(result.diagnostics.relative_total_mass_drift, 0.0, places=14)

    def test_growth_cancels_adjacent_interface_erosion(self) -> None:
        shape = (4, 1, 1)
        moments = _equilibrium_moments(shape)
        flags = np.asarray(
            [FslCellFlag.LIQUID, FslCellFlag.INTERFACE, FslCellFlag.INTERFACE, FslCellFlag.GAS],
            dtype=np.int32,
        ).reshape(shape)
        mass = np.asarray([1.0, 1.1, -0.1, 0.0], dtype=np.float64).reshape(shape)
        result = resolve_fsl_topology(
            moments, flags, mass, periodic=(False, False, False)
        )

        self.assertEqual(result.flags[1, 0, 0], int(FslCellFlag.LIQUID))
        self.assertEqual(result.flags[2, 0, 0], int(FslCellFlag.INTERFACE))
        self.assertEqual(result.flags[3, 0, 0], int(FslCellFlag.GAS))
        self.assertEqual(result.diagnostics.cancelled_interface_to_gas_count, 1)
        self.assertEqual(result.diagnostics.interface_to_gas_count, 0)

    def test_fresh_moments_use_average_density_and_velocity(self) -> None:
        moments, flags, mass, excess = _layered_fields()
        moments[1, ..., 0] = np.linspace(0.9, 1.1, 9).reshape(3, 3)
        velocity_x = np.linspace(0.01, 0.09, 9).reshape(3, 3)
        moments[1, ..., 1] = moments[1, ..., 0] * velocity_x
        mass[1] = 1.2
        result = resolve_fsl_topology(
            moments, flags, mass, excess, periodic=(False, False, False)
        )

        center = result.moments[2, 1, 1]
        self.assertAlmostEqual(center[0], 1.0, places=14)
        self.assertAlmostEqual(center[1] / center[0], 0.05, places=14)
        self.assertAlmostEqual(center[4], center[0] * 0.05**2, places=14)

    def test_queued_donor_total_is_delivered_once(self) -> None:
        shape = (3, 1, 1)
        flags = np.asarray(
            [FslCellFlag.LIQUID, FslCellFlag.INTERFACE, FslCellFlag.GAS], dtype=np.int32
        ).reshape(shape)
        mass = np.asarray([1.0, 0.4, 0.0], dtype=np.float64).reshape(shape)
        queued = np.asarray([0.2, 0.0, 0.0], dtype=np.float64).reshape(shape)

        received, stranded = _receive_queued_excess(
            mass, queued, flags, (False, False, False)
        )

        self.assertAlmostEqual(received[1, 0, 0], 0.6, places=14)
        self.assertAlmostEqual(float(np.sum(received) + np.sum(stranded)), 1.6, places=14)
        np.testing.assert_array_equal(stranded, 0.0)

    def test_stranded_excess_remains_in_the_donor_queue(self) -> None:
        shape = (2, 1, 1)
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        mass = np.zeros(shape, dtype=np.float64)
        queued = np.zeros(shape, dtype=np.float64)
        queued[0, 0, 0] = -0.3

        received, stranded = _receive_queued_excess(
            mass, queued, flags, (False, False, False)
        )

        np.testing.assert_array_equal(received, mass)
        self.assertAlmostEqual(stranded[0, 0, 0], -0.3, places=14)

    def test_growth_then_recession_restores_layered_topology(self) -> None:
        moments, flags, mass, excess = _layered_fields()
        mass[1] = 1.1
        grown = resolve_fsl_topology(
            moments, flags, mass, excess, periodic=(False, False, False)
        )
        receding_mass = grown.mass.copy()
        receding_mass[2] = -0.1
        restored = resolve_fsl_topology(
            grown.moments,
            grown.flags,
            receding_mass,
            grown.excess_mass,
            periodic=(False, False, False),
        )

        np.testing.assert_array_equal(restored.flags, flags)
        self.assertEqual(restored.diagnostics.interface_to_gas_count, 9)
        self.assertEqual(restored.diagnostics.liquid_to_interface_count, 9)
        self.assertAlmostEqual(grown.diagnostics.relative_total_mass_drift, 0.0, places=14)
        self.assertAlmostEqual(restored.diagnostics.relative_total_mass_drift, 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
