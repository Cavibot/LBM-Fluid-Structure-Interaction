# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import FslCellFlag
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    resolve_plic_topology_reference,
)


GAS = int(FslCellFlag.GAS)
INTERFACE = int(FslCellFlag.INTERFACE)
LIQUID = int(FslCellFlag.LIQUID)


def _moments(shape: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros(shape + (10,), dtype=np.float64)
    x = np.arange(shape[0], dtype=np.float64)[:, None, None]
    result[..., 0] = 1.0 + 0.01 * x
    result[..., 1] = result[..., 0] * 0.02
    result[..., 2] = result[..., 0] * -0.01
    result[..., 4] = result[..., 0] * 0.02**2
    result[..., 5] = result[..., 0] * 0.01**2
    result[..., 7] = result[..., 0] * -0.0002
    return result


def _case() -> tuple[np.ndarray, ...]:
    shape = (5, 3, 3)
    source = np.full(shape, GAS, dtype=np.int32)
    source[0] = LIQUID
    source[1] = INTERFACE
    fill = np.zeros(shape, dtype=np.float64)
    fill[0] = 1.0
    fill[1] = 0.8
    fill[2] = 0.2
    mass = fill * (0.9 + 0.02 * np.arange(shape[0])[:, None, None])
    momentum = mass[..., None] * np.asarray((0.03, -0.02, 0.01))
    solid = np.zeros(shape, dtype=bool)
    return _moments(shape), source, fill, mass, momentum, solid


class TestPlicTopologyReference(unittest.TestCase):
    def test_advancing_front_preserves_three_ledgers_and_initializes_fresh_cells(self) -> None:
        moments, source, fill, mass, momentum, solid = _case()
        moments[2] = -9.0
        result = resolve_plic_topology_reference(
            moments, source, fill, mass, momentum, solid
        )
        np.testing.assert_array_equal(result.flags[0], LIQUID)
        np.testing.assert_array_equal(result.flags[1:3], INTERFACE)
        np.testing.assert_array_equal(result.flags[3:], GAS)
        np.testing.assert_array_equal(result.fill_level, fill)
        np.testing.assert_array_equal(result.mass, mass)
        np.testing.assert_array_equal(result.momentum, momentum)
        self.assertEqual(result.diagnostics.fresh_interface_count, 9)
        self.assertTrue(np.all(result.donor_count[2] > 0))
        self.assertTrue(np.all(result.moments[2, ..., 0] > 0.0))

    def test_sub_tolerance_tail_is_conservatively_closed_into_interface(self) -> None:
        moments, source, fill, mass, momentum, solid = _case()
        fill[1, 1, 1] = 1.0e-7
        mass[1, 1, 1] = 7.0e-8
        momentum[1, 1, 1] = (2.0e-9, 3.0e-9, -1.0e-9)
        result = resolve_plic_topology_reference(
            moments, source, fill, mass, momentum, solid
        )
        self.assertEqual(result.flags[1, 1, 1], GAS)
        self.assertEqual(result.fill_level[1, 1, 1], 0.0)
        self.assertEqual(result.mass[1, 1, 1], 0.0)
        np.testing.assert_array_equal(result.momentum[1, 1, 1], 0.0)
        self.assertEqual(result.diagnostics.endpoint_removed_gas_count, 1)
        self.assertEqual(
            result.diagnostics.endpoint_redistributed_volume,
            fill[1, 1, 1],
        )
        self.assertEqual(
            result.diagnostics.endpoint_redistributed_mass,
            mass[1, 1, 1],
        )
        self.assertAlmostEqual(result.diagnostics.volume_drift, 0.0, places=14)
        self.assertAlmostEqual(result.diagnostics.mass_drift, 0.0, places=14)
        np.testing.assert_allclose(
            result.diagnostics.momentum_drift, 0.0, atol=1.0e-15
        )

    def test_near_full_tail_is_liquid_without_changing_ledgers(self) -> None:
        moments, source, fill, mass, momentum, solid = _case()
        fill[0, 1, 1] = 1.0 - 1.0e-7
        mass[0, 1, 1] = 0.93
        result = resolve_plic_topology_reference(
            moments,
            source,
            fill,
            mass,
            momentum,
            solid,
            endpoint_tolerance=2.0e-6,
        )
        self.assertEqual(result.flags[0, 1, 1], LIQUID)
        self.assertEqual(result.fill_level[0, 1, 1], fill[0, 1, 1])
        self.assertEqual(result.mass[0, 1, 1], mass[0, 1, 1])
        self.assertGreater(
            result.diagnostics.endpoint_promoted_liquid_count, 0
        )

    def test_new_liquid_next_to_gas_is_repaired_to_interface(self) -> None:
        shape = (5, 5, 1)
        source = np.full(shape, GAS, dtype=np.int32)
        source[2, 2, 0] = INTERFACE
        fill = np.zeros(shape, dtype=np.float64)
        fill[2, 2, 0] = 1.0
        mass = fill.copy()
        result = resolve_plic_topology_reference(
            _moments(shape),
            source,
            fill,
            mass,
            np.zeros(shape + (3,)),
            np.zeros(shape, bool),
        )
        self.assertEqual(result.flags[2, 2, 0], INTERFACE)
        self.assertEqual(result.fill_level[2, 2, 0], 1.0)
        self.assertEqual(result.diagnostics.separation_repair_count, 1)
        self.assertEqual(result.diagnostics.direct_liquid_gas_link_count, 0)

    def test_rejects_cfl_phase_jump(self) -> None:
        moments, source, fill, mass, momentum, solid = _case()
        source[2] = GAS
        fill[2] = 1.0
        with self.assertRaisesRegex(RuntimeError, "gas-to-liquid"):
            resolve_plic_topology_reference(
                moments, source, fill, mass, momentum, solid
            )

    def test_rejects_direct_liquid_gas_link(self) -> None:
        shape = (3, 3, 3)
        source = np.full(shape, GAS, dtype=np.int32)
        source[1, 1, 1] = LIQUID
        fill = (source == LIQUID).astype(np.float64)
        with self.assertRaisesRegex(RuntimeError, "direct liquid-gas"):
            resolve_plic_topology_reference(
                _moments(shape), source, fill, fill, np.zeros(shape + (3,)), np.zeros(shape, bool)
            )

    def test_rejects_fresh_interface_without_donor(self) -> None:
        shape = (3, 3, 3)
        source = np.full(shape, GAS, dtype=np.int32)
        fill = np.zeros(shape, dtype=np.float64)
        fill[1, 1, 1] = 0.2
        with self.assertRaisesRegex(RuntimeError, "no persistent fluid donor"):
            resolve_plic_topology_reference(
                _moments(shape), source, fill, fill, np.zeros(shape + (3,)), np.zeros(shape, bool)
            )


if __name__ == "__main__":
    unittest.main()
