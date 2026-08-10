# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""State-contract tests for the sharp HOME-FREE VOF path."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof import (
    link_mass_delta,
    normalize_advected_mass_momentum_and_excess,
    normalize_mass_and_excess,
    normalize_mass_momentum_and_excess,
    validate_state_fields,
)


class TestHomeFreeVofState(unittest.TestCase):
    def _fluid_state(self) -> tuple[HomeLbmModel, HomeLbmState]:
        model = HomeLbmModel(fluid_grid_res=(4, 3, 2), periodic=(False, False, False), device="cpu")
        state = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(state, rho=1.05)
        return model, state

    def test_initialization_preserves_mass_density_fill_contract(self) -> None:
        model, fluid = self._fluid_state()
        fill = np.zeros((4, 3, 2), dtype=np.float32)
        fill[0] = 1.0
        fill[1] = 0.35
        free = HomeFreeState(model)

        diagnostics = free.initialize_from_fill_level(fluid, fill)

        self.assertEqual(diagnostics.liquid_cells, 6)
        self.assertEqual(diagnostics.interface_cells, 6)
        self.assertEqual(diagnostics.gas_cells, 12)
        np.testing.assert_allclose(free.mass.numpy(), 1.05 * fill, rtol=0.0, atol=2.0e-7)
        self.assertEqual(free.persistent_scalar_count, 7 * 4 * 3 * 2)
        np.testing.assert_array_equal(free.excess_momentum.numpy(), 0.0)
        self.assertEqual(free.validate(fluid), diagnostics)

    def test_solid_cells_override_fill_and_carry_no_liquid_mass(self) -> None:
        model, fluid = self._fluid_state()
        solid_phi = np.full((4, 3, 2), 1000.0, dtype=np.float32)
        solid_phi[0, 0, 0] = -0.2
        fluid.solid_phi.assign(solid_phi)
        fill = np.ones((4, 3, 2), dtype=np.float32)
        free = HomeFreeState(model)

        diagnostics = free.initialize_from_fill_level(fluid, fill)

        self.assertEqual(diagnostics.solid_cells, 1)
        self.assertEqual(int(free.flags.numpy()[0, 0, 0]), int(HomeFreeCellFlag.SOLID))
        self.assertEqual(float(free.fill_level.numpy()[0, 0, 0]), 0.0)
        self.assertEqual(float(free.mass.numpy()[0, 0, 0]), 0.0)

    def test_direct_liquid_gas_neighbors_are_rejected(self) -> None:
        density = np.ones((3, 1, 1), dtype=np.float32)
        fill = np.asarray([1.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1, 1)
        flags = np.asarray(
            [HomeFreeCellFlag.LIQUID, HomeFreeCellFlag.GAS, HomeFreeCellFlag.GAS],
            dtype=np.int32,
        ).reshape(3, 1, 1)
        with self.assertRaisesRegex(ValueError, "direct liquid-gas"):
            validate_state_fields(
                density,
                fill,
                fill,
                np.zeros_like(fill),
                np.zeros(fill.shape + (3,), dtype=np.float32),
                flags,
            )

    def test_invalid_fraction_and_inconsistent_committed_mass_fail(self) -> None:
        model, fluid = self._fluid_state()
        free = HomeFreeState(model)
        invalid = np.zeros((4, 3, 2), dtype=np.float32)
        invalid[1, 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            free.initialize_from_fill_level(fluid, invalid)

        fill = np.zeros((4, 3, 2), dtype=np.float32)
        fill[1] = 0.5
        free.initialize_from_fill_level(fluid, fill)
        mass = free.mass.numpy()
        mass[1, 0, 0] += 0.1
        free.mass.assign(mass)
        with self.assertRaisesRegex(ValueError, r"density\*fill_level"):
            free.validate(fluid)

    def test_copy_preserves_all_committed_and_queued_fields(self) -> None:
        model, fluid = self._fluid_state()
        fill = np.zeros((4, 3, 2), dtype=np.float32)
        fill[1] = 0.4
        source = HomeFreeState(model)
        source.initialize_from_fill_level(fluid, fill)
        excess = np.zeros((4, 3, 2), dtype=np.float32)
        excess[1, 0, 0] = 0.02
        source.excess_mass.assign(excess)
        excess_momentum = np.zeros((3, source.cell_count), dtype=np.float32)
        excess_momentum[:, 6] = (0.001, -0.002, 0.003)
        source.excess_momentum.assign(excess_momentum.reshape(-1))
        target = HomeFreeState(model)

        target.copy_from(source)

        np.testing.assert_array_equal(target.mass.numpy(), source.mass.numpy())
        np.testing.assert_array_equal(target.fill_level.numpy(), source.fill_level.numpy())
        np.testing.assert_array_equal(target.excess_mass.numpy(), source.excess_mass.numpy())
        np.testing.assert_array_equal(
            target.excess_momentum.numpy(), source.excess_momentum.numpy()
        )
        np.testing.assert_array_equal(target.flags.numpy(), source.flags.numpy())

    def test_link_mass_exchange_uses_paper_weights(self) -> None:
        neighbor_flags = np.asarray(
            [
                HomeFreeCellFlag.LIQUID,
                HomeFreeCellFlag.INTERFACE,
                HomeFreeCellFlag.GAS,
                HomeFreeCellFlag.SOLID,
            ],
            dtype=np.int32,
        )
        neighbor_fill = np.asarray([1.0, 0.25, 0.0, 0.0])
        incoming = np.asarray([0.12, 0.09, 0.50, 0.40])
        outgoing = np.asarray([0.10, 0.05, 0.01, 0.02])

        liquid_delta = link_mass_delta(
            HomeFreeCellFlag.LIQUID,
            1.0,
            neighbor_flags,
            neighbor_fill,
            incoming,
            outgoing,
        )
        interface_delta = link_mass_delta(
            HomeFreeCellFlag.INTERFACE,
            0.75,
            neighbor_flags,
            neighbor_fill,
            incoming,
            outgoing,
        )

        self.assertAlmostEqual(liquid_delta, (0.12 - 0.10) + (0.09 - 0.05))
        self.assertAlmostEqual(
            interface_delta,
            (0.12 - 0.10) + 0.5 * (0.75 + 0.25) * (0.09 - 0.05),
        )

    def test_mass_normalization_preserves_committed_and_queued_total(self) -> None:
        density = np.asarray([1.0, 1.1, 0.9, 1.0])
        advected = np.asarray([1.08, -0.03, 0.04, 0.0])
        flags = np.asarray(
            [
                HomeFreeCellFlag.LIQUID,
                HomeFreeCellFlag.INTERFACE,
                HomeFreeCellFlag.GAS,
                HomeFreeCellFlag.SOLID,
            ],
            dtype=np.int32,
        )
        recipient_count = np.asarray([2, 3, 1, 0], dtype=np.int32)

        committed, fill, queued_share = normalize_mass_and_excess(
            density, advected, flags, recipient_count
        )

        np.testing.assert_allclose(committed, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(fill, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            float(np.sum(committed + queued_share * recipient_count)),
            float(np.sum(advected)),
        )

        momentum = density[:, None] * np.asarray((0.03, -0.02, 0.01))
        committed, fill, queued_share, queued_momentum = (
            normalize_mass_momentum_and_excess(
                density,
                momentum,
                advected,
                flags,
                recipient_count,
            )
        )
        np.testing.assert_allclose(
            queued_momentum,
            queued_share[:, None] * np.asarray((0.03, -0.02, 0.01)),
        )
        represented_momentum = committed[:, None] * momentum / density[:, None]
        represented_momentum += recipient_count[:, None] * queued_momentum
        np.testing.assert_allclose(
            np.sum(represented_momentum, axis=0),
            np.sum(advected[:, None] * momentum / density[:, None], axis=0),
        )

    def test_exact_mass_momentum_normalization_closes_each_cell(self) -> None:
        density = np.asarray([1.0, 1.1, 0.9, 1.0])
        home_velocity = np.asarray(
            [
                (0.02, -0.01, 0.03),
                (-0.01, 0.04, 0.02),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ]
        )
        home_momentum = density[:, None] * home_velocity
        advected_mass = np.asarray([1.08, -0.03, 0.04, 0.0])
        transported_velocity = np.asarray(
            [
                (0.05, -0.02, 0.01),
                (-0.04, 0.03, 0.02),
                (0.02, 0.01, -0.03),
                (0.0, 0.0, 0.0),
            ]
        )
        advected_momentum = advected_mass[:, None] * transported_velocity
        flags = np.asarray(
            [
                HomeFreeCellFlag.LIQUID,
                HomeFreeCellFlag.INTERFACE,
                HomeFreeCellFlag.GAS,
                HomeFreeCellFlag.SOLID,
            ],
            dtype=np.int32,
        )
        recipients = np.asarray([2, 3, 1, 0], dtype=np.int32)

        committed, fill, excess, excess_j, committed_j = (
            normalize_advected_mass_momentum_and_excess(
                density,
                home_momentum,
                advected_mass,
                advected_momentum,
                flags,
                recipients,
            )
        )

        np.testing.assert_allclose(committed, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(fill, [1.0, 0.0, 0.0, 0.0])
        represented_mass = committed + recipients * excess
        np.testing.assert_allclose(represented_mass, advected_mass)
        velocity = np.zeros_like(committed_j)
        active = np.isin(
            flags,
            (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
        )
        velocity[active] = committed_j[active] / density[active, None]
        velocity[~active & (np.abs(advected_mass) > 0.0)] = transported_velocity[
            ~active & (np.abs(advected_mass) > 0.0)
        ]
        represented_momentum = committed[:, None] * velocity
        represented_momentum += recipients[:, None] * excess_j
        np.testing.assert_allclose(represented_momentum, advected_momentum)

    def test_zero_mass_interface_retains_home_state_and_rejects_momentum(self) -> None:
        density = np.asarray([1.2])
        home_momentum = np.asarray([[0.036, -0.012, 0.024]])
        flags = np.asarray([HomeFreeCellFlag.INTERFACE], dtype=np.int32)
        recipients = np.asarray([1], dtype=np.int32)

        _, _, _, excess_j, committed_j = normalize_advected_mass_momentum_and_excess(
            density,
            home_momentum,
            np.zeros(1),
            np.zeros((1, 3)),
            flags,
            recipients,
        )

        np.testing.assert_array_equal(excess_j, 0.0)
        np.testing.assert_allclose(committed_j, home_momentum)
        with self.assertRaisesRegex(ValueError, "requires transported mass"):
            normalize_advected_mass_momentum_and_excess(
                density,
                home_momentum,
                np.zeros(1),
                np.asarray([[1.0e-4, 0.0, 0.0]]),
                flags,
                recipients,
            )


if __name__ == "__main__":
    unittest.main()
