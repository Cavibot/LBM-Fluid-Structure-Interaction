# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P4 acceptance tests for deterministic topology and mass redistribution."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    LbmDomain,
    LbmModel,
    VofCellType,
    VofTopologyTransition,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ


def _model(
    shape: tuple[int, int, int],
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    epsilon: float = 1.0e-4,
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding="fullf",
        collision="srt",
        interface_model="vof",
        bc_periodic=periodic,
        vof_transition_epsilon=epsilon,
        enforce_population_positivity=False,
    )


def _layered_phi(shape: tuple[int, int, int]) -> np.ndarray:
    phi = np.zeros(shape, dtype=np.float32)
    split = shape[0] // 2
    phi[:split] = 1.0
    phi[split] = 0.5
    return phi


def _neighbor(
    cell: tuple[int, int, int],
    q: int,
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[int, int, int] | None:
    values = [
        cell[0] - CX[q],
        cell[1] - CY[q],
        cell[2] - CZ[q],
    ]
    for axis in range(3):
        if 0 <= values[axis] < shape[axis]:
            continue
        if not periodic[axis]:
            return None
        values[axis] %= shape[axis]
    return values[0], values[1], values[2]


def _oracle(
    mass_pre: np.ndarray,
    density: np.ndarray,
    cell_type: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool],
    epsilon: float,
    mass_tolerance: float = 2.0e-6,
) -> dict[str, np.ndarray]:
    shape = mass_pre.shape
    proposed = cell_type.copy()
    interface = cell_type == int(VofCellType.INTERFACE)
    phi_pre = mass_pre / density
    proposed[interface & (phi_pre >= 1.0 + epsilon)] = int(
        VofCellType.LIQUID
    )
    proposed[interface & (phi_pre <= 0.0 - epsilon)] = int(VofCellType.GAS)

    final_type = proposed.copy()
    for cell in np.ndindex(shape):
        old_kind = int(cell_type[cell])
        proposal = int(proposed[cell])
        opposing: int | None = None
        if old_kind == int(VofCellType.GAS):
            opposing = int(VofCellType.LIQUID)
        elif old_kind == int(VofCellType.LIQUID):
            opposing = int(VofCellType.GAS)
        elif (
            old_kind == int(VofCellType.INTERFACE)
            and proposal == int(VofCellType.GAS)
        ):
            opposing = int(VofCellType.LIQUID)
        if opposing is None:
            continue
        for q in range(1, 19):
            neighbor = _neighbor(cell, q, shape, periodic)
            if (
                neighbor is not None
                and int(proposed[neighbor]) == opposing
                and (
                    old_kind != int(VofCellType.INTERFACE)
                    or int(cell_type[neighbor]) == int(VofCellType.INTERFACE)
                )
            ):
                final_type[cell] = int(VofCellType.INTERFACE)

    mass_base = np.zeros(shape, dtype=np.float32)
    liquid = final_type == int(VofCellType.LIQUID)
    final_interface = final_type == int(VofCellType.INTERFACE)
    mass_base[liquid] = density[liquid]
    roundoff_liquid = liquid & (
        np.abs(mass_pre - density) <= mass_tolerance
    )
    mass_base[roundoff_liquid] = mass_pre[roundoff_liquid]
    mass_base[final_interface] = np.clip(
        mass_pre[final_interface],
        0.0,
        density[final_interface],
    )
    excess = mass_pre - mass_base
    count = np.zeros(shape, dtype=np.uint8)
    share = np.zeros(shape, dtype=np.float32)
    unresolved = np.zeros(shape, dtype=np.float32)
    for cell in np.ndindex(shape):
        receiver_count = 0
        for q in range(1, 19):
            neighbor = _neighbor(cell, q, shape, periodic)
            if (
                neighbor is not None
                and int(final_type[neighbor]) == int(VofCellType.INTERFACE)
            ):
                receiver_count += 1
        count[cell] = receiver_count
        if receiver_count:
            share[cell] = excess[cell] / np.float32(receiver_count)
        elif abs(float(excess[cell])) > mass_tolerance:
            unresolved[cell] = excess[cell]

    mass_final = mass_base.copy()
    for cell in np.ndindex(shape):
        if int(final_type[cell]) != int(VofCellType.INTERFACE):
            continue
        for q in range(1, 19):
            neighbor = _neighbor(cell, q, shape, periodic)
            if neighbor is not None:
                mass_final[cell] += share[neighbor]
    phi_final = np.zeros(shape, dtype=np.float32)
    phi_final[liquid] = mass_final[liquid] / density[liquid]
    phi_final[final_interface] = (
        mass_final[final_interface] / density[final_interface]
    )
    return {
        "proposed_type": proposed,
        "final_type": final_type,
        "mass_base": mass_base,
        "excess": excess,
        "share": share,
        "receiver_count": count,
        "unresolved_excess": unresolved,
        "mass_final": mass_final,
        "phi_final": phi_final,
        "new_interface": (
            (cell_type == int(VofCellType.GAS))
            & (final_type == int(VofCellType.INTERFACE))
        ).astype(np.uint8),
        "retired_active": (
            (cell_type != int(VofCellType.GAS))
            & (final_type == int(VofCellType.GAS))
        ).astype(np.uint8),
        "changed": (cell_type != final_type).astype(np.uint8),
    }


def _transition(
    shape: tuple[int, int, int],
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    epsilon: float = 1.0e-4,
) -> VofTopologyTransition:
    return VofTopologyTransition(
        shape,
        wp.get_device("cpu"),
        tuple(int(value) for value in periodic),
        epsilon,
    )


def _compute(
    transition: VofTopologyTransition,
    mass: np.ndarray,
    density: np.ndarray,
    cell_type: np.ndarray,
):
    return transition.compute(
        wp.array(np.ascontiguousarray(mass), dtype=float, device="cpu"),
        wp.array(np.ascontiguousarray(density), dtype=float, device="cpu"),
        wp.array(
            np.ascontiguousarray(cell_type),
            dtype=wp.uint8,
            device="cpu",
        ),
    )


class TestVofP4ConfigurationAndThresholds(unittest.TestCase):
    def test_epsilon_contract(self) -> None:
        model = _model((3, 3, 3))
        self.assertEqual(model.vof_transition_epsilon, 1.0e-4)
        for epsilon in (-1.0e-4, 0.5, np.nan, np.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    _model((3, 3, 3), epsilon=epsilon)

    def test_threshold_equalities_and_inner_band(self) -> None:
        shape = (3, 3, 3)
        density = np.ones(shape, dtype=np.float32)
        cell_type = np.full(
            shape,
            int(VofCellType.INTERFACE),
            dtype=np.uint8,
        )
        transition = _transition(shape, periodic=(True, True, True))
        center = (1, 1, 1)

        mass = np.full(shape, 0.5, dtype=np.float32)
        mass[center] = np.float32(1.0 + 1.0e-4)
        high = _compute(transition, mass, density, cell_type)
        self.assertEqual(
            int(high.proposed_type.numpy()[center]),
            int(VofCellType.LIQUID),
        )

        mass[center] = np.float32(-1.0e-4)
        low = _compute(transition, mass, density, cell_type)
        self.assertEqual(
            int(low.proposed_type.numpy()[center]),
            int(VofCellType.GAS),
        )

        mass[center] = np.float32(1.0 + 0.5e-4)
        inner = _compute(transition, mass, density, cell_type)
        self.assertEqual(
            int(inner.proposed_type.numpy()[center]),
            int(VofCellType.INTERFACE),
        )


class TestVofP4TopologyAndRedistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (5, 3, 3)
        self.density = np.ones(self.shape, dtype=np.float32)
        phi = _layered_phi(self.shape)
        self.cell_type = np.full(
            self.shape,
            int(VofCellType.INTERFACE),
            dtype=np.uint8,
        )
        self.cell_type[phi == 0.0] = int(VofCellType.GAS)
        self.cell_type[phi == 1.0] = int(VofCellType.LIQUID)
        self.mass = phi.copy()
        self.center = (2, 1, 1)

    def test_interface_to_liquid_promotes_gas_and_splits_positive_excess(
        self,
    ) -> None:
        mass = self.mass.copy()
        mass[self.center] = 1.2
        result = _compute(
            _transition(self.shape),
            mass,
            self.density,
            self.cell_type,
        )
        final_type = result.final_type.numpy()
        self.assertEqual(
            int(final_type[self.center]),
            int(VofCellType.LIQUID),
        )
        self.assertEqual(int(result.new_interface.numpy().sum()), 5)
        self.assertEqual(int(result.receiver_count.numpy()[self.center]), 13)
        self.assertAlmostEqual(
            float(result.share.numpy()[self.center]),
            0.2 / 13.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(np.sum(result.mass_final.numpy(), dtype=np.float64)),
            float(np.sum(mass, dtype=np.float64)),
            places=5,
        )

    def test_interface_to_gas_demotes_liquid_and_splits_negative_excess(
        self,
    ) -> None:
        mass = self.mass.copy()
        mass[self.center] = -0.2
        result = _compute(
            _transition(self.shape),
            mass,
            self.density,
            self.cell_type,
        )
        final_type = result.final_type.numpy()
        self.assertEqual(int(final_type[self.center]), int(VofCellType.GAS))
        self.assertEqual(int(result.retired_active.numpy().sum()), 1)
        self.assertEqual(int(result.receiver_count.numpy()[self.center]), 13)
        self.assertAlmostEqual(
            float(result.share.numpy()[self.center]),
            -0.2 / 13.0,
            places=6,
        )
        self.assertEqual(
            int(
                np.count_nonzero(
                    final_type[1, :, :] == int(VofCellType.INTERFACE)
                )
            ),
            5,
        )
        self.assertAlmostEqual(
            float(np.sum(result.mass_final.numpy(), dtype=np.float64)),
            float(np.sum(mass, dtype=np.float64)),
            places=5,
        )

    def test_opposing_candidates_use_reference_liquid_priority(self) -> None:
        shape = (4, 3, 3)
        density = np.ones(shape, dtype=np.float32)
        cell_type = np.full(
            shape,
            int(VofCellType.INTERFACE),
            dtype=np.uint8,
        )
        mass = np.full(shape, 0.5, dtype=np.float32)
        liquid_candidate = (1, 1, 1)
        gas_candidate = (2, 1, 1)
        mass[liquid_candidate] = 1.2
        mass[gas_candidate] = -0.2
        result = _compute(
            _transition(shape),
            mass,
            density,
            cell_type,
        )
        final_type = result.final_type.numpy()
        self.assertEqual(
            int(final_type[liquid_candidate]),
            int(VofCellType.LIQUID),
        )
        self.assertEqual(
            int(final_type[gas_candidate]),
            int(VofCellType.INTERFACE),
        )

    def test_periodic_seam_uses_the_same_topology_neighborhood(self) -> None:
        shape = (5, 3, 3)
        phi = np.empty(shape, dtype=np.float32)
        phi[0] = 0.0
        phi[1] = 0.5
        phi[2] = 1.0
        phi[3:] = 0.5
        cell_type = np.full(
            shape,
            int(VofCellType.INTERFACE),
            dtype=np.uint8,
        )
        cell_type[phi == 0.0] = int(VofCellType.GAS)
        cell_type[phi == 1.0] = int(VofCellType.LIQUID)
        mass = phi.copy()
        mass[4, 1, 1] = 1.2
        result = _compute(
            _transition(shape, periodic=(True, False, False)),
            mass,
            np.ones(shape, dtype=np.float32),
            cell_type,
        )
        self.assertEqual(
            int(result.final_type.numpy()[0, 1, 1]),
            int(VofCellType.INTERFACE),
        )
        self.assertEqual(int(result.new_interface.numpy()[0, 1, 1]), 1)

    def test_zero_receiver_fails_without_mutating_inputs(self) -> None:
        shape = (3, 3, 3)
        mass = np.zeros(shape, dtype=np.float32)
        density = np.ones(shape, dtype=np.float32)
        cell_type = np.full(shape, int(VofCellType.GAS), dtype=np.uint8)
        center = (1, 1, 1)
        mass[center] = -0.2
        cell_type[center] = int(VofCellType.INTERFACE)
        snapshots = (mass.copy(), density.copy(), cell_type.copy())
        with self.assertRaisesRegex(ValueError, "no final INTERFACE receiver"):
            _compute(_transition(shape), mass, density, cell_type)
        np.testing.assert_array_equal(mass, snapshots[0])
        np.testing.assert_array_equal(density, snapshots[1])
        np.testing.assert_array_equal(cell_type, snapshots[2])


class TestVofP4OracleAndIntegration(unittest.TestCase):
    def test_random_field_matches_independent_oracle_and_is_deterministic(
        self,
    ) -> None:
        shape = (5, 4, 3)
        periodic = (False, True, True)
        rng = np.random.default_rng(404)
        density = rng.uniform(0.92, 1.08, size=shape).astype(np.float32)
        cell_type = np.full(
            shape,
            int(VofCellType.INTERFACE),
            dtype=np.uint8,
        )
        cell_type[0] = int(VofCellType.LIQUID)
        cell_type[4] = int(VofCellType.GAS)
        mass = np.zeros(shape, dtype=np.float32)
        mass[cell_type == int(VofCellType.LIQUID)] = density[
            cell_type == int(VofCellType.LIQUID)
        ]
        interface = cell_type == int(VofCellType.INTERFACE)
        mass[interface] = rng.uniform(
            -0.12,
            1.12,
            size=np.count_nonzero(interface),
        ).astype(np.float32) * density[interface]
        expected = _oracle(
            mass,
            density,
            cell_type,
            periodic=periodic,
            epsilon=1.0e-4,
        )
        transition = _transition(shape, periodic=periodic)
        mass_array = wp.array(mass, dtype=float, device="cpu")
        density_array = wp.array(density, dtype=float, device="cpu")
        type_array = wp.array(cell_type, dtype=wp.uint8, device="cpu")
        first_result = transition.compute(
            mass_array,
            density_array,
            type_array,
        )
        first = {
            name: getattr(first_result, name).numpy().copy()
            for name in expected
        }
        np.testing.assert_array_equal(mass_array.numpy(), mass)
        np.testing.assert_array_equal(density_array.numpy(), density)
        np.testing.assert_array_equal(type_array.numpy(), cell_type)
        second_result = transition.compute(
            mass_array,
            density_array,
            type_array,
        )
        second = {
            name: getattr(second_result, name).numpy().copy()
            for name in expected
        }

        for name, oracle_value in expected.items():
            if np.issubdtype(oracle_value.dtype, np.floating):
                np.testing.assert_allclose(
                    first[name],
                    oracle_value,
                    atol=4.0e-7,
                    rtol=3.0e-6,
                    err_msg=name,
                )
            else:
                np.testing.assert_array_equal(first[name], oracle_value)
            np.testing.assert_array_equal(first[name], second[name])
        self.assertAlmostEqual(
            float(np.sum(first["mass_final"], dtype=np.float64)),
            float(np.sum(mass, dtype=np.float64)),
            places=5,
        )

    def test_transition_without_new_interface_commits_and_swaps(self) -> None:
        shape = (3, 3, 3)
        domain = LbmDomain(_model(shape, periodic=(True, True, True)))
        state_in = domain.initialize_vof(
            np.full(shape, 0.5, dtype=np.float32)
        )
        assert state_in.vof is not None
        center = (1, 1, 1)
        mass = state_in.vof.mass.numpy().copy()
        phi = state_in.vof.phi.numpy().copy()
        mass[center] = 1.2
        phi[center] = 1.2
        state_in.vof.mass.assign(mass)
        state_in.vof.phi.assign(phi)
        initial_total = float(np.sum(mass, dtype=np.float64))

        domain.step(1.0)

        self.assertIsNot(domain.state, state_in)
        assert domain.state.vof is not None
        self.assertEqual(
            int(domain.state.vof.cell_type.numpy()[center]),
            int(VofCellType.LIQUID),
        )
        self.assertAlmostEqual(
            float(np.sum(domain.state.vof.mass.numpy(), dtype=np.float64)),
            initial_total,
            places=5,
        )

    def test_new_interface_waits_for_p5_without_buffer_swap(self) -> None:
        shape = (5, 3, 3)
        domain = LbmDomain(_model(shape))
        state_in = domain.initialize_vof(_layered_phi(shape))
        state_out = domain._state_out
        assert state_in.vof is not None and state_out is not None
        center = (2, 1, 1)
        mass = state_in.vof.mass.numpy().copy()
        phi = state_in.vof.phi.numpy().copy()
        mass[center] = 1.2
        phi[center] = 1.2
        state_in.vof.mass.assign(mass)
        state_in.vof.phi.assign(phi)
        mass_before = state_in.vof.mass.numpy().copy()

        with self.assertRaisesRegex(NotImplementedError, "deferred to P5"):
            domain.step(1.0)

        self.assertIs(domain._state_in, state_in)
        self.assertIs(domain._state_out, state_out)
        np.testing.assert_array_equal(state_in.vof.mass.numpy(), mass_before)


if __name__ == "__main__":
    unittest.main()
