# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P5 acceptance tests for new-interface FullF kinetic initialization."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    FullFLbmState,
    LbmDomain,
    LbmModel,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, W
from wanphys._src.fluid.fluid_grid.lbm.vof import VofCellType
from wanphys._src.fluid.fluid_grid.lbm.vof.solver.kinetic_init import VofKineticInitializer
from wanphys._src.fluid.fluid_grid.lbm.vof.solver.transition import VofTransitionResult


def _model(
    shape: tuple[int, int, int],
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding="fullf",
        collision="srt",
        interface_model="vof",
        bc_periodic=periodic,
        gravity_x=gravity[0],
        gravity_y=gravity[1],
        gravity_z=gravity[2],
        enforce_population_positivity=False,
    )


def _layered_phi(shape: tuple[int, int, int]) -> np.ndarray:
    phi = np.zeros(shape, dtype=np.float32)
    split = shape[0] // 2
    phi[:split] = 1.0
    phi[split] = 0.5
    return phi


def _equilibrium(
    rho: float,
    velocity: tuple[float, float, float],
) -> np.ndarray:
    ux, uy, uz = velocity
    result = np.empty(19, dtype=np.float64)
    u2 = ux * ux + uy * uy + uz * uz
    for q in range(19):
        cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz
        result[q] = W[q] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
        )
    return result


def _array(values: np.ndarray, *, dtype=float) -> wp.array:
    return wp.array(
        np.ascontiguousarray(values),
        dtype=dtype,
        device="cpu",
    )


def _fake_transition(
    final_type: np.ndarray,
    new_interface: np.ndarray,
    *,
    mass_final: np.ndarray | None = None,
) -> VofTransitionResult:
    shape = final_type.shape
    zeros_f = np.zeros(shape, dtype=np.float32)
    zeros_b = np.zeros(shape, dtype=np.uint8)
    mass = (
        np.asarray(mass_final, dtype=np.float32)
        if mass_final is not None
        else zeros_f.copy()
    )
    phi = np.zeros(shape, dtype=np.float32)
    active = final_type != int(VofCellType.GAS)
    phi[active] = mass[active]
    return VofTransitionResult(
        proposed_type=_array(final_type, dtype=wp.uint8),
        final_type=_array(final_type, dtype=wp.uint8),
        mass_base=_array(mass),
        excess=_array(zeros_f),
        share=_array(zeros_f),
        receiver_count=_array(zeros_b, dtype=wp.uint8),
        unresolved_excess=_array(zeros_f),
        mass_final=_array(mass),
        phi_final=_array(phi),
        new_interface=_array(new_interface, dtype=wp.uint8),
        retired_active=_array(zeros_b, dtype=wp.uint8),
        changed=_array(new_interface, dtype=wp.uint8),
    )


def _state(shape: tuple[int, int, int]) -> FullFLbmState:
    domain = LbmDomain(_model(shape))
    state = domain.solver.create_state()
    assert isinstance(state, FullFLbmState)
    domain.solver.initialize_equilibrium(state)
    return state


def _donor_fixture():
    shape = (3, 3, 3)
    target = (1, 1, 1)
    state = _state(shape)
    old_type = np.full(shape, int(VofCellType.GAS), dtype=np.uint8)
    final_type = old_type.copy()
    final_type[target] = int(VofCellType.INTERFACE)
    new_interface = np.zeros(shape, dtype=np.uint8)
    new_interface[target] = 1

    donor_specs = (
        (1, int(VofCellType.INTERFACE), int(VofCellType.INTERFACE), 1.0),
        (2, int(VofCellType.LIQUID), int(VofCellType.LIQUID), 2.0),
        (3, int(VofCellType.INTERFACE), int(VofCellType.LIQUID), 3.0),
        (4, int(VofCellType.LIQUID), int(VofCellType.INTERFACE), 4.0),
        (5, int(VofCellType.INTERFACE), int(VofCellType.GAS), 80.0),
        (6, int(VofCellType.GAS), int(VofCellType.INTERFACE), 90.0),
    )
    density = np.full(shape, 50.0, dtype=np.float32)
    ux = np.full(shape, 0.3, dtype=np.float32)
    uy = np.full(shape, -0.2, dtype=np.float32)
    uz = np.full(shape, 0.1, dtype=np.float32)
    for q, old_kind, final_kind, value in donor_specs:
        neighbor = (
            target[0] - CX[q],
            target[1] - CY[q],
            target[2] - CZ[q],
        )
        old_type[neighbor] = old_kind
        final_type[neighbor] = final_kind
        if (
            old_kind == int(VofCellType.GAS)
            and final_kind == int(VofCellType.INTERFACE)
        ):
            new_interface[neighbor] = 1
        density[neighbor] = value
        ux[neighbor] = 0.01 * value
        uy[neighbor] = -0.005 * value
        uz[neighbor] = 0.002 * value
    state.density.assign(density)
    state.velocity_x.assign(ux)
    state.velocity_y.assign(uy)
    state.velocity_z.assign(uz)
    mass = np.zeros(shape, dtype=np.float32)
    mass[target] = 0.625
    transition = _fake_transition(
        final_type,
        new_interface,
        mass_final=mass,
    )
    initializer = VofKineticInitializer(
        shape,
        wp.get_device("cpu"),
        (0, 0, 0),
    )
    return (
        state,
        old_type,
        transition,
        initializer,
        target,
    )


class TestVofP5Donors(unittest.TestCase):
    def test_all_18_moving_directions_can_donate(self) -> None:
        shape = (3, 3, 3)
        target = (1, 1, 1)
        for q in range(1, 19):
            with self.subTest(q=q):
                state = _state(shape)
                old_type = np.full(
                    shape,
                    int(VofCellType.GAS),
                    dtype=np.uint8,
                )
                final_type = old_type.copy()
                final_type[target] = int(VofCellType.INTERFACE)
                new_interface = np.zeros(shape, dtype=np.uint8)
                new_interface[target] = 1
                donor = (
                    target[0] - CX[q],
                    target[1] - CY[q],
                    target[2] - CZ[q],
                )
                old_type[donor] = int(VofCellType.LIQUID)
                final_type[donor] = int(VofCellType.LIQUID)
                density = state.density.numpy()
                density[donor] = np.float32(1.0 + 0.01 * q)
                state.density.assign(density)
                transition = _fake_transition(final_type, new_interface)
                initializer = VofKineticInitializer(
                    shape,
                    wp.get_device("cpu"),
                    (0, 0, 0),
                )
                result = initializer.prepare(
                    state,
                    _array(old_type, dtype=wp.uint8),
                    transition,
                    max_lattice_speed=0.4,
                    enforce_population_positivity=False,
                    population_floor=0.0,
                )
                self.assertEqual(int(result.donor_count.numpy()[target]), 1)
                self.assertAlmostEqual(
                    float(result.rho_init.numpy()[target]),
                    1.0 + 0.01 * q,
                    places=6,
                )

    def test_donor_eligibility_and_uniform_mean(self) -> None:
        state, old_type, transition, initializer, target = _donor_fixture()
        f_before = state.f_post.numpy().copy()
        density_before = state.density.numpy().copy()
        result = initializer.prepare(
            state,
            _array(old_type, dtype=wp.uint8),
            transition,
            max_lattice_speed=0.4,
            enforce_population_positivity=True,
            population_floor=1.0e-9,
        )
        self.assertEqual(int(result.donor_count.numpy()[target]), 4)
        self.assertAlmostEqual(float(result.rho_init.numpy()[target]), 2.5)
        self.assertAlmostEqual(float(result.ux_init.numpy()[target]), 0.025)
        self.assertAlmostEqual(float(result.uy_init.numpy()[target]), -0.0125)
        self.assertAlmostEqual(float(result.uz_init.numpy()[target]), 0.005)
        np.testing.assert_array_equal(state.f_post.numpy(), f_before)
        np.testing.assert_array_equal(state.density.numpy(), density_before)

    def test_old_gas_and_same_batch_new_interface_garbage_is_ignored(
        self,
    ) -> None:
        state, old_type, transition, initializer, target = _donor_fixture()
        first = initializer.prepare(
            state,
            _array(old_type, dtype=wp.uint8),
            transition,
            max_lattice_speed=0.4,
            enforce_population_positivity=False,
            population_floor=0.0,
        )
        expected = (
            float(first.rho_init.numpy()[target]),
            float(first.ux_init.numpy()[target]),
            float(first.uy_init.numpy()[target]),
            float(first.uz_init.numpy()[target]),
        )
        old_gas = old_type == int(VofCellType.GAS)
        density = state.density.numpy().copy()
        ux = state.velocity_x.numpy().copy()
        uy = state.velocity_y.numpy().copy()
        uz = state.velocity_z.numpy().copy()
        density[old_gas] = 999.0
        ux[old_gas] = 8.0
        uy[old_gas] = -7.0
        uz[old_gas] = 6.0
        state.density.assign(density)
        state.velocity_x.assign(ux)
        state.velocity_y.assign(uy)
        state.velocity_z.assign(uz)
        second = initializer.prepare(
            state,
            _array(old_type, dtype=wp.uint8),
            transition,
            max_lattice_speed=0.4,
            enforce_population_positivity=False,
            population_floor=0.0,
        )
        actual = (
            float(second.rho_init.numpy()[target]),
            float(second.ux_init.numpy()[target]),
            float(second.uy_init.numpy()[target]),
            float(second.uz_init.numpy()[target]),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_periodic_seam_donor(self) -> None:
        shape = (5, 3, 3)
        target = (0, 1, 1)
        donor = (4, 1, 1)
        state = _state(shape)
        old_type = np.full(shape, int(VofCellType.GAS), dtype=np.uint8)
        final_type = old_type.copy()
        old_type[donor] = int(VofCellType.LIQUID)
        final_type[donor] = int(VofCellType.LIQUID)
        final_type[target] = int(VofCellType.INTERFACE)
        new_interface = np.zeros(shape, dtype=np.uint8)
        new_interface[target] = 1
        density = state.density.numpy()
        density[donor] = 1.7
        state.density.assign(density)
        transition = _fake_transition(final_type, new_interface)
        initializer = VofKineticInitializer(
            shape,
            wp.get_device("cpu"),
            (1, 0, 0),
        )
        result = initializer.prepare(
            state,
            _array(old_type, dtype=wp.uint8),
            transition,
            max_lattice_speed=0.4,
            enforce_population_positivity=False,
            population_floor=0.0,
        )
        self.assertEqual(int(result.donor_count.numpy()[target]), 1)
        self.assertAlmostEqual(float(result.rho_init.numpy()[target]), 1.7)

    def test_zero_donor_fails_before_kinetic_write(self) -> None:
        shape = (3, 3, 3)
        target = (1, 1, 1)
        state = _state(shape)
        old_type = np.full(shape, int(VofCellType.GAS), dtype=np.uint8)
        final_type = old_type.copy()
        final_type[target] = int(VofCellType.INTERFACE)
        new_interface = np.zeros(shape, dtype=np.uint8)
        new_interface[target] = 1
        transition = _fake_transition(final_type, new_interface)
        initializer = VofKineticInitializer(
            shape,
            wp.get_device("cpu"),
            (0, 0, 0),
        )
        f_before = state.f_post.numpy().copy()
        density_before = state.density.numpy().copy()
        with self.assertRaisesRegex(ValueError, "no old-active"):
            initializer.initialize_fullf(
                state,
                _array(old_type, dtype=wp.uint8),
                transition,
                gravity=(0.0, 0.0, 0.0),
                max_lattice_speed=0.4,
                enforce_population_positivity=False,
                population_floor=0.0,
            )
        np.testing.assert_array_equal(state.f_post.numpy(), f_before)
        np.testing.assert_array_equal(state.density.numpy(), density_before)


class TestVofP5Equilibrium(unittest.TestCase):
    def test_fullf_equilibrium_macro_force_and_phi_closure(self) -> None:
        state, old_type, transition, initializer, target = _donor_fixture()
        gravity = (0.01, -0.02, 0.03)
        initializer.initialize_fullf(
            state,
            _array(old_type, dtype=wp.uint8),
            transition,
            gravity=gravity,
            max_lattice_speed=0.4,
            enforce_population_positivity=True,
            population_floor=1.0e-9,
        )
        rho = 2.5
        velocity = (0.025, -0.0125, 0.005)
        actual_f = state.f_post.numpy().reshape((19, *state.res))[
            (slice(None),) + target
        ]
        expected_f = _equilibrium(rho, velocity)
        np.testing.assert_allclose(actual_f, expected_f, atol=2.0e-7)
        self.assertAlmostEqual(float(np.sum(actual_f)), rho, places=6)
        momentum = np.array(
            [
                np.dot(actual_f, CX),
                np.dot(actual_f, CY),
                np.dot(actual_f, CZ),
            ]
        )
        np.testing.assert_allclose(
            momentum,
            rho * np.asarray(velocity),
            atol=3.0e-7,
        )
        self.assertGreater(float(np.min(actual_f)), 0.0)
        self.assertAlmostEqual(float(state.density.numpy()[target]), rho)
        self.assertAlmostEqual(
            float(state.force_x.numpy()[target]),
            rho * gravity[0],
        )
        self.assertAlmostEqual(
            float(state.force_y.numpy()[target]),
            rho * gravity[1],
        )
        self.assertAlmostEqual(
            float(state.force_z.numpy()[target]),
            rho * gravity[2],
        )
        self.assertAlmostEqual(
            float(transition.mass_final.numpy()[target]),
            0.625,
        )
        self.assertAlmostEqual(
            float(transition.phi_final.numpy()[target]),
            0.25,
        )

    def test_non_new_kinetic_and_macro_fields_are_unchanged(self) -> None:
        state, old_type, transition, initializer, _ = _donor_fixture()
        mask = transition.new_interface.numpy().astype(bool)
        shape = tuple(int(value) for value in state.res)
        before = (
            state.f_post.numpy().reshape((19, *shape)).copy(),
            state.density.numpy().copy(),
            state.velocity_x.numpy().copy(),
            state.velocity_y.numpy().copy(),
            state.velocity_z.numpy().copy(),
            state.force_x.numpy().copy(),
            state.force_y.numpy().copy(),
            state.force_z.numpy().copy(),
        )
        initializer.initialize_fullf(
            state,
            _array(old_type, dtype=wp.uint8),
            transition,
            gravity=(0.0, 0.0, 0.0),
            max_lattice_speed=0.4,
            enforce_population_positivity=False,
            population_floor=0.0,
        )
        after = (
            state.f_post.numpy().reshape((19, *shape)),
            state.density.numpy(),
            state.velocity_x.numpy(),
            state.velocity_y.numpy(),
            state.velocity_z.numpy(),
            state.force_x.numpy(),
            state.force_y.numpy(),
            state.force_z.numpy(),
        )
        np.testing.assert_array_equal(after[0][:, ~mask], before[0][:, ~mask])
        for actual, expected in zip(after[1:], before[1:], strict=True):
            np.testing.assert_array_equal(actual[~mask], expected[~mask])


class TestVofP5Integration(unittest.TestCase):
    def _moving_domain(self):
        shape = (5, 3, 3)
        domain = LbmDomain(_model(shape))
        state = domain.initialize_vof(_layered_phi(shape))
        assert state.vof is not None
        center = (2, 1, 1)
        mass = state.vof.mass.numpy().copy()
        phi = state.vof.phi.numpy().copy()
        mass[center] = 1.2
        phi[center] = 1.2
        reference_mass = float(np.sum(mass, dtype=np.float64))
        state.vof.mass.assign(mass)
        state.vof.phi.assign(phi)
        state.vof.reference_mass = reference_mass
        assert domain._state_out is not None
        assert domain._state_out.vof is not None
        domain._state_out.vof.reference_mass = reference_mass
        domain.solver._vof_interface_geometry.compute(state.vof)
        return domain, state, center, reference_mass

    def test_positive_front_commits_and_new_interface_runs_next_step(
        self,
    ) -> None:
        domain, state_in, center, total_before = self._moving_domain()
        old_type = state_in.vof.cell_type.numpy().copy()
        domain.step(1.0)
        self.assertIsNot(domain.state, state_in)
        state = domain.state
        assert state.vof is not None
        final_type = state.vof.cell_type.numpy()
        new_interface = (
            (old_type == int(VofCellType.GAS))
            & (final_type == int(VofCellType.INTERFACE))
        )
        self.assertEqual(int(final_type[center]), int(VofCellType.LIQUID))
        self.assertEqual(int(np.count_nonzero(new_interface)), 5)
        active = final_type != int(VofCellType.GAS)
        populations = state.f_post.numpy().reshape((19, *state.res))
        self.assertTrue(np.all(np.isfinite(populations[:, active])))
        self.assertTrue(np.all(state.density.numpy()[active] > 0.0))
        self.assertAlmostEqual(
            float(
                np.sum(state.vof.mass.numpy(), dtype=np.float64)
                + np.sum(
                    state.vof.pending_excess.numpy(),
                    dtype=np.float64,
                )
            ),
            total_before,
            places=5,
        )

        domain.step(1.0)
        state = domain.state
        assert state.vof is not None
        active = state.vof.cell_type.numpy() != int(VofCellType.GAS)
        populations = state.f_post.numpy().reshape((19, *state.res))
        self.assertTrue(np.all(np.isfinite(populations[:, active])))
        self.assertTrue(np.all(np.isfinite(state.vof.mass.numpy())))
        self.assertTrue(np.all(np.isfinite(state.vof.phi.numpy())))

    def test_p5_failure_does_not_swap_or_mutate_current_state(self) -> None:
        domain, state_in, _, _ = self._moving_domain()
        state_out = domain._state_out
        assert state_in.vof is not None and state_out is not None
        mass_before = state_in.vof.mass.numpy().copy()
        initializer = domain.solver._vof_kinetic_initializer
        assert initializer is not None
        with patch.object(
            initializer,
            "prepare",
            side_effect=ValueError("synthetic P5 pre-write failure"),
        ):
            with self.assertRaisesRegex(ValueError, "pre-write failure"):
                domain.step(1.0)
        self.assertIs(domain._state_in, state_in)
        self.assertIs(domain._state_out, state_out)
        np.testing.assert_array_equal(state_in.vof.mass.numpy(), mass_before)

    def test_create_then_retire_interface_remains_finite_and_conservative(
        self,
    ) -> None:
        domain, _, _, total_before = self._moving_domain()
        domain.step(1.0)
        state = domain.state
        assert state.vof is not None
        types = state.vof.cell_type.numpy()
        candidates = np.argwhere(types == int(VofCellType.INTERFACE))
        target = tuple(int(value) for value in candidates[0])
        mass = state.vof.mass.numpy().copy()
        phi = state.vof.phi.numpy().copy()
        density = state.density.numpy()
        mass[target] = np.float32(-0.2 * density[target])
        phi[target] = np.float32(-0.2)
        delta = float(mass[target] - state.vof.mass.numpy()[target])
        state.vof.mass.assign(mass)
        state.vof.phi.assign(phi)
        state.vof.reference_mass = total_before + delta
        assert domain._state_out is not None
        assert domain._state_out.vof is not None
        domain._state_out.vof.reference_mass = total_before + delta
        domain.solver._vof_interface_geometry.compute(state.vof)

        domain.step(1.0)

        state = domain.state
        assert state.vof is not None
        self.assertTrue(np.all(np.isfinite(state.vof.mass.numpy())))
        self.assertTrue(np.all(np.isfinite(state.vof.phi.numpy())))
        active = state.vof.cell_type.numpy() != int(VofCellType.GAS)
        self.assertTrue(np.all(np.isfinite(state.density.numpy()[active])))
        self.assertAlmostEqual(
            float(
                np.sum(state.vof.mass.numpy(), dtype=np.float64)
                + np.sum(
                    state.vof.pending_excess.numpy(),
                    dtype=np.float64,
                )
            ),
            total_before + delta,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
