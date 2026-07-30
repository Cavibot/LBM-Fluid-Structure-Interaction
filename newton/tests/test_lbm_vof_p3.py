# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P3 acceptance tests for zero-surface-tension free-surface completion."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm import (
    LbmDomain,
    LbmModel,
    VofCellType,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, OPPOSITE, W


def _model(
    shape: tuple[int, int, int],
    *,
    encoding: str = "fullf",
    periodic: tuple[bool, bool, bool] = (False, False, False),
    pressure: float = 1.0 / 3.0,
    surface_tension: float = 0.0,
    boundary_models: tuple[str, str, str, str, str, str] | None = None,
) -> LbmModel:
    kwargs: dict[str, object] = {}
    if boundary_models is not None:
        kwargs["boundary_models"] = boundary_models
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding=encoding,
        collision="srt",
        interface_model="vof",
        bc_periodic=periodic,
        vof_atmosphere_pressure=pressure,
        vof_surface_tension=surface_tension,
        enforce_population_positivity=False,
        **kwargs,
    )


def _equilibrium(
    q: int,
    rho: float,
    velocity: tuple[float, float, float],
) -> float:
    ux, uy, uz = velocity
    cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz
    u2 = ux * ux + uy * uy + uz * uz
    return W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def _layered_phi(shape: tuple[int, int, int]) -> np.ndarray:
    phi = np.zeros(shape, dtype=np.float32)
    split = shape[0] // 2
    phi[:split] = 1.0
    phi[split] = 0.5
    return phi


class TestVofP3Configuration(unittest.TestCase):
    def test_pressure_contract_and_zero_surface_tension_gate(self) -> None:
        model = _model((3, 2, 2))
        domain = LbmDomain(model)
        self.assertAlmostEqual(
            domain.solver._vof_surface_boundary.rho_g,
            1.0,
            places=7,
        )
        for pressure in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(pressure=pressure):
                with self.assertRaises(ValueError):
                    _model((3, 2, 2), pressure=pressure)
        with self.assertRaisesRegex(NotImplementedError, "deferred to P6"):
            _model((3, 2, 2), surface_tension=1.0e-3)

    def test_open_vof_boundary_is_deferred(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "static bounce-back"):
            _model(
                (3, 2, 2),
                boundary_models=(
                    "zou_he",
                    "bounce_back",
                    "bounce_back",
                    "bounce_back",
                    "bounce_back",
                    "bounce_back",
                ),
            )

    def test_home_step_fails_before_mutation_or_swap(self) -> None:
        shape = (5, 2, 2)
        domain = LbmDomain(_model(shape, encoding="home"))
        state_in = domain.initialize_vof(_layered_phi(shape))
        state_out = domain._state_out
        assert state_in.vof is not None and state_out is not None
        mass_before = state_in.vof.mass.numpy().copy()
        with self.assertRaisesRegex(NotImplementedError, "deferred to P7"):
            domain.step(1.0)
        self.assertIs(domain._state_in, state_in)
        self.assertIs(domain._state_out, state_out)
        np.testing.assert_array_equal(state_in.vof.mass.numpy(), mass_before)


class TestVofP3Equation11(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (3, 3, 3)
        self.center = (1, 1, 1)
        phi = np.zeros(self.shape, dtype=np.float32)
        phi[self.center] = 0.5
        self.domain = LbmDomain(_model(self.shape))
        self.state = self.domain.initialize_vof(
            phi,
            rho0=1.0,
            u0=(0.0, 0.0, 0.0),
        )

    def test_all_moving_directions_match_independent_equation_11(self) -> None:
        velocity = (0.07, -0.03, 0.02)
        self.state.velocity_x.fill_(velocity[0])
        self.state.velocity_y.fill_(velocity[1])
        self.state.velocity_z.fill_(velocity[2])
        populations = self.state.f_post.numpy().reshape((19, *self.shape)).copy()
        for q in range(19):
            populations[(q,) + self.center] = np.float32(0.011 + 0.003 * q)
        self.state.f_post.assign(populations.reshape(-1))

        actual = (
            self.domain.solver.compute_vof_surface_populations(self.state)
            .numpy()
            .reshape((19, *self.shape))
        )
        rho_g = 1.0
        for q in range(1, 19):
            opposite = OPPOSITE[q]
            expected = (
                _equilibrium(q, rho_g, velocity)
                + _equilibrium(opposite, rho_g, velocity)
                - float(populations[(opposite,) + self.center])
            )
            self.assertAlmostEqual(
                float(actual[(q,) + self.center]),
                expected,
                places=6,
                msg=f"direction {q}",
            )

        self.assertEqual(
            float(actual[(0,) + self.center]),
            float(populations[(0,) + self.center]),
        )

    def test_gas_storage_does_not_change_completed_active_populations(self) -> None:
        rng = np.random.default_rng(310)
        base = self.state.f_post.numpy().reshape((19, *self.shape)).copy()
        cell_type = self.state.vof.cell_type.numpy()
        gas = cell_type == int(VofCellType.GAS)

        first = base.copy()
        second = base.copy()
        for q in range(19):
            first[q][gas] = rng.uniform(2.0, 4.0, size=np.count_nonzero(gas))
            second[q][gas] = rng.uniform(7.0, 9.0, size=np.count_nonzero(gas))

        self.state.f_post.assign(first.reshape(-1))
        completed_a = (
            self.domain.solver.compute_vof_surface_populations(self.state)
            .numpy()
            .reshape((19, *self.shape))[:, self.center[0], self.center[1], self.center[2]]
            .copy()
        )
        self.state.f_post.assign(second.reshape(-1))
        state_f_before = self.state.f_post.numpy().copy()
        state_mass_before = self.state.vof.mass.numpy().copy()
        state_phi_before = self.state.vof.phi.numpy().copy()
        state_type_before = self.state.vof.cell_type.numpy().copy()
        completed_b = (
            self.domain.solver.compute_vof_surface_populations(self.state)
            .numpy()
            .reshape((19, *self.shape))[:, self.center[0], self.center[1], self.center[2]]
            .copy()
        )
        np.testing.assert_array_equal(completed_a, completed_b)
        np.testing.assert_array_equal(self.state.f_post.numpy(), state_f_before)
        np.testing.assert_array_equal(self.state.vof.mass.numpy(), state_mass_before)
        np.testing.assert_array_equal(self.state.vof.phi.numpy(), state_phi_before)
        np.testing.assert_array_equal(self.state.vof.cell_type.numpy(), state_type_before)


class TestVofP3IntegratedStep(unittest.TestCase):
    def test_planar_equilibrium_is_multistep_fixed_point(self) -> None:
        shape = (7, 3, 3)
        phi0 = _layered_phi(shape)
        domain = LbmDomain(_model(shape))
        state0 = domain.initialize_vof(phi0, rho0=1.0)
        assert state0.vof is not None
        initial_mass = float(np.sum(state0.vof.mass.numpy(), dtype=np.float64))
        initial_types = state0.vof.cell_type.numpy().copy()

        for _ in range(12):
            domain.step(1.0)

        state = domain.state
        assert state.vof is not None
        np.testing.assert_allclose(state.vof.phi.numpy(), phi0, atol=1.0e-5)
        np.testing.assert_array_equal(state.vof.cell_type.numpy(), initial_types)
        self.assertAlmostEqual(
            float(np.sum(state.vof.mass.numpy(), dtype=np.float64)),
            initial_mass,
            places=5,
        )
        active = state.vof.cell_type.numpy() != int(VofCellType.GAS)
        self.assertLessEqual(
            float(np.max(np.abs(state.velocity_x.numpy()[active]))), 1.0e-6
        )
        self.assertLessEqual(
            float(np.max(np.abs(state.velocity_y.numpy()[active]))), 1.0e-6
        )
        self.assertLessEqual(
            float(np.max(np.abs(state.velocity_z.numpy()[active]))), 1.0e-6
        )

    def test_p2_mass_scratch_is_committed_and_buffers_swap(self) -> None:
        shape = (4, 3, 3)
        periodic = (True, True, True)
        phi = np.full(shape, 0.5, dtype=np.float32)
        domain = LbmDomain(_model(shape, periodic=periodic))
        state_in = domain.initialize_vof(phi)
        populations = state_in.f_post.numpy().reshape((19, *shape)).copy()
        populations[1, 0, 1, 1] += np.float32(0.01)
        populations[2, 0, 1, 1] -= np.float32(0.01)
        state_in.f_post.assign(populations.reshape(-1))
        expected_mass = (
            domain.solver.compute_vof_mass_transport(state_in)
            .mass_tmp.numpy()
            .copy()
        )

        domain.step(1.0)
        self.assertIsNot(domain.state, state_in)
        assert domain.state.vof is not None
        np.testing.assert_allclose(
            domain.state.vof.mass.numpy(),
            expected_mass,
            atol=2.0e-7,
            rtol=2.0e-6,
        )

    def test_integrated_active_result_is_independent_of_gas_storage(self) -> None:
        shape = (5, 3, 3)
        phi = _layered_phi(shape)
        domains = (LbmDomain(_model(shape)), LbmDomain(_model(shape)))
        rng = np.random.default_rng(811)
        gas_snapshots: list[tuple[np.ndarray, ...]] = []
        for index, domain in enumerate(domains):
            state = domain.initialize_vof(phi)
            assert state.vof is not None
            populations = state.f_post.numpy().reshape((19, *shape)).copy()
            gas = state.vof.cell_type.numpy() == int(VofCellType.GAS)
            for q in range(19):
                populations[q][gas] = rng.uniform(
                    1.0 + 4.0 * index,
                    2.0 + 4.0 * index,
                    size=np.count_nonzero(gas),
                )
            state.f_post.assign(populations.reshape(-1))
            gas_snapshots.append(
                (
                    populations[:, gas].copy(),
                    state.density.numpy()[gas].copy(),
                    state.velocity_x.numpy()[gas].copy(),
                    state.velocity_y.numpy()[gas].copy(),
                    state.velocity_z.numpy()[gas].copy(),
                    state.force_x.numpy()[gas].copy(),
                    state.force_y.numpy()[gas].copy(),
                    state.force_z.numpy()[gas].copy(),
                )
            )

        for domain in domains:
            domain.step(1.0)

        first, second = domains[0].state, domains[1].state
        assert first.vof is not None and second.vof is not None
        active = first.vof.cell_type.numpy() != int(VofCellType.GAS)
        first_f = first.f_post.numpy().reshape((19, *shape))
        second_f = second.f_post.numpy().reshape((19, *shape))
        np.testing.assert_allclose(first_f[:, active], second_f[:, active], atol=1.0e-7)
        np.testing.assert_allclose(
            first.density.numpy()[active],
            second.density.numpy()[active],
            atol=1.0e-7,
        )
        for domain, snapshot in zip(domains, gas_snapshots, strict=True):
            state = domain.state
            assert state.vof is not None
            gas = state.vof.cell_type.numpy() == int(VofCellType.GAS)
            actual = (
                state.f_post.numpy().reshape((19, *shape))[:, gas],
                state.density.numpy()[gas],
                state.velocity_x.numpy()[gas],
                state.velocity_y.numpy()[gas],
                state.velocity_z.numpy()[gas],
                state.force_x.numpy()[gas],
                state.force_y.numpy()[gas],
                state.force_z.numpy()[gas],
            )
            for actual_field, expected_field in zip(actual, snapshot, strict=True):
                np.testing.assert_array_equal(actual_field, expected_field)

    def test_invalid_input_raises_without_current_buffer_swap(self) -> None:
        shape = (3, 3, 3)
        phi = np.full(shape, 0.5, dtype=np.float32)
        domain = LbmDomain(_model(shape, periodic=(True, True, True)))
        state_in = domain.initialize_vof(phi)
        state_out = domain._state_out
        assert state_in.vof is not None and state_out is not None
        mass = state_in.vof.mass.numpy().copy()
        mass[1, 1, 1] = np.float32(np.nan)
        state_in.vof.mass.assign(mass)
        mass_before = state_in.vof.mass.numpy().copy()

        with self.assertRaisesRegex(ValueError, "mass must be finite"):
            domain.step(1.0)

        self.assertIs(domain._state_in, state_in)
        self.assertIs(domain._state_out, state_out)
        np.testing.assert_array_equal(
            state_in.vof.mass.numpy(),
            mass_before,
        )


if __name__ == "__main__":
    unittest.main()
