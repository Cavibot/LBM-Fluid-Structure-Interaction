# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P2 acceptance tests for fixed-topology authoritative VOF mass transport."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm import (
    LbmDomain,
    LbmModel,
    VofCellType,
    VofMassScheme,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, OPPOSITE


def _model(
    shape: tuple[int, int, int],
    *,
    encoding: str = "fullf",
    periodic: tuple[bool, bool, bool] = (False, False, False),
    scheme: str = "fslbm_neighbor",
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding=encoding,
        collision="srt",
        interface_model="vof",
        vof_mass_scheme=scheme,
        bc_periodic=periodic,
        enforce_population_positivity=False,
    )


def _wrap_or_none(
    cell: tuple[int, int, int],
    q: int,
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[int, int, int] | None:
    source = [
        cell[0] - CX[q],
        cell[1] - CY[q],
        cell[2] - CZ[q],
    ]
    for axis in range(3):
        if 0 <= source[axis] < shape[axis]:
            continue
        if not periodic[axis]:
            return None
        source[axis] %= shape[axis]
    return source[0], source[1], source[2]


def _reference_mass_transport(
    populations: np.ndarray,
    density: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    cell_type: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent NumPy oracle; it does not call production helpers."""

    shape = tuple(int(value) for value in density.shape)
    delta = np.zeros(shape, dtype=np.float64)
    for cell in np.ndindex(shape):
        center_type = int(cell_type[cell])
        if center_type == int(VofCellType.GAS):
            continue
        for q in range(1, 19):
            source = _wrap_or_none(cell, q, shape, periodic)
            if source is None:
                continue
            source_type = int(cell_type[source])
            weight = 0.0
            if center_type == int(VofCellType.LIQUID):
                if source_type != int(VofCellType.GAS):
                    weight = 1.0
            elif source_type == int(VofCellType.LIQUID):
                weight = 1.0
            elif source_type == int(VofCellType.INTERFACE):
                weight = 0.5 * (float(phi[cell]) + float(phi[source]))
            if weight == 0.0:
                continue
            delta[cell] += weight * (
                float(populations[(q, *source)])
                - float(populations[(OPPOSITE[q], *cell)])
            )
    mass_tmp = mass.astype(np.float64) + delta
    phi_tmp = mass_tmp / density.astype(np.float64)
    return delta, mass_tmp, phi_tmp


def _domain_from_phi(
    phi: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    encoding: str = "fullf",
) -> LbmDomain:
    domain = LbmDomain(_model(phi.shape, periodic=periodic, encoding=encoding))
    domain.initialize_vof(phi, rho0=1.0)
    return domain


def _assign_consistent_transport_state(
    domain: LbmDomain,
    populations: np.ndarray,
    density: np.ndarray | None = None,
) -> None:
    state = domain.state
    assert state.vof is not None
    if density is None:
        density = np.ones(state.res, dtype=np.float32)
    density = np.asarray(density, dtype=np.float32)
    state.density.assign(density)
    state.vof.mass.assign(density * state.vof.phi.numpy())
    state.f_post.assign(np.asarray(populations, dtype=np.float32).reshape(-1))


class TestVofP2Contracts(unittest.TestCase):
    def test_mass_scheme_is_explicit_and_unknown_values_fail(self) -> None:
        model = _model((3, 2, 2))
        self.assertEqual(
            model.vof_mass_scheme,
            VofMassScheme.FSLBM_NEIGHBOR.value,
        )
        with self.assertRaisesRegex(ValueError, "Unknown VOF mass scheme"):
            _model((3, 2, 2), scheme="paper_center")

    def test_p2_is_fullf_only_and_complete_step_still_fails_before_writes(
        self,
    ) -> None:
        shape = (3, 2, 2)
        phi = np.full(shape, 0.5, dtype=np.float32)
        home = _domain_from_phi(phi, encoding="home")
        with self.assertRaisesRegex(NotImplementedError, "deferred to P7"):
            home.solver.compute_vof_mass_transport(home.state)

        fullf = _domain_from_phi(phi)
        state_before = fullf.state.clone()
        with self.assertRaisesRegex(NotImplementedError, "requires P3"):
            fullf.step(1.0)
        np.testing.assert_array_equal(
            fullf.state.f_post.numpy(), state_before.f_post.numpy()
        )
        assert fullf.state.vof is not None and state_before.vof is not None
        for name in ("mass", "phi", "cell_type"):
            np.testing.assert_array_equal(
                getattr(fullf.state.vof, name).numpy(),
                getattr(state_before.vof, name).numpy(),
            )


class TestVofP2LinkWeights(unittest.TestCase):
    shape = (3, 3, 3)
    center = (1, 1, 1)
    q = 1
    source = (0, 1, 1)

    def _run_link(
        self,
        *,
        center_type: VofCellType,
        source_type: VofCellType,
        center_phi: float,
        source_phi: float,
        incoming: float = 0.17,
        outgoing: float = 0.03,
    ) -> tuple[float, float]:
        phi = np.full(self.shape, 0.4, dtype=np.float32)
        if center_type == VofCellType.LIQUID:
            phi[self.center] = 1.0
        elif center_type == VofCellType.GAS:
            phi[self.center] = 0.0
        else:
            phi[self.center] = center_phi
        if source_type == VofCellType.LIQUID:
            phi[self.source] = 1.0
        elif source_type == VofCellType.GAS:
            phi[self.source] = 0.0
        else:
            phi[self.source] = source_phi

        domain = _domain_from_phi(phi)
        populations = np.zeros((19, *self.shape), dtype=np.float32)
        populations[self.q, self.source[0], self.source[1], self.source[2]] = incoming
        opposite = OPPOSITE[self.q]
        populations[
            opposite,
            self.center[0],
            self.center[1],
            self.center[2],
        ] = outgoing
        _assign_consistent_transport_state(domain, populations)

        result = domain.solver.compute_vof_mass_transport(domain.state)
        actual = float(result.mass_delta.numpy()[self.center])
        raw_flux = incoming - outgoing
        return actual, raw_flux

    def test_liquid_and_interface_link_weights_match_frozen_scheme(self) -> None:
        cases = (
            (VofCellType.LIQUID, VofCellType.LIQUID, 1.0),
            (VofCellType.LIQUID, VofCellType.INTERFACE, 1.0),
            (VofCellType.INTERFACE, VofCellType.LIQUID, 1.0),
            (VofCellType.INTERFACE, VofCellType.INTERFACE, 0.6),
            (VofCellType.INTERFACE, VofCellType.GAS, 0.0),
            (VofCellType.GAS, VofCellType.INTERFACE, 0.0),
        )
        for center_type, source_type, weight in cases:
            with self.subTest(center=center_type, source=source_type):
                actual, raw_flux = self._run_link(
                    center_type=center_type,
                    source_type=source_type,
                    center_phi=0.5,
                    source_phi=0.7,
                )
                self.assertAlmostEqual(actual, weight * raw_flux, places=6)

    def test_rest_population_has_no_mass_flux(self) -> None:
        phi = np.full(self.shape, 0.5, dtype=np.float32)
        domain = _domain_from_phi(phi)
        populations = np.zeros((19, *self.shape), dtype=np.float32)
        populations[0, self.center[0], self.center[1], self.center[2]] = 9.0
        _assign_consistent_transport_state(domain, populations)
        result = domain.solver.compute_vof_mass_transport(domain.state)
        np.testing.assert_array_equal(result.mass_delta.numpy(), 0.0)

    def test_nonperiodic_outside_link_is_impermeable(self) -> None:
        phi = np.full(self.shape, 0.5, dtype=np.float32)
        domain = _domain_from_phi(phi)
        populations = np.zeros((19, *self.shape), dtype=np.float32)
        boundary_cell = (0, 1, 1)
        populations[(OPPOSITE[1],) + boundary_cell] = 0.8
        _assign_consistent_transport_state(domain, populations)
        result = domain.solver.compute_vof_mass_transport(domain.state)
        self.assertAlmostEqual(
            float(result.mass_delta.numpy()[boundary_cell]), 0.0, places=7
        )


class TestVofP2ConservationAndOracle(unittest.TestCase):
    def test_periodic_random_field_matches_oracle_and_conserves_mass(self) -> None:
        shape = (4, 3, 3)
        periodic = (True, True, True)
        rng = np.random.default_rng(20260730)
        phi = rng.uniform(0.2, 0.8, size=shape).astype(np.float32)
        domain = _domain_from_phi(phi, periodic=periodic)
        populations = rng.uniform(0.001, 0.08, size=(19, *shape)).astype(
            np.float32
        )
        density = rng.uniform(0.9, 1.1, size=shape).astype(np.float32)
        _assign_consistent_transport_state(domain, populations, density)

        state = domain.state
        assert state.vof is not None
        expected = _reference_mass_transport(
            populations,
            density,
            state.vof.mass.numpy(),
            state.vof.phi.numpy(),
            state.vof.cell_type.numpy(),
            periodic,
        )
        result = domain.solver.compute_vof_mass_transport(state)

        np.testing.assert_allclose(
            result.mass_delta.numpy(), expected[0], atol=2.0e-7, rtol=2.0e-6
        )
        np.testing.assert_allclose(
            result.mass_tmp.numpy(), expected[1], atol=2.0e-7, rtol=2.0e-6
        )
        np.testing.assert_allclose(
            result.phi_tmp.numpy(), expected[2], atol=2.0e-7, rtol=2.0e-6
        )
        self.assertAlmostEqual(
            float(np.sum(result.mass_delta.numpy(), dtype=np.float64)),
            0.0,
            places=6,
        )

    def test_interface_gas_flux_does_not_read_gas_populations(self) -> None:
        shape = (4, 3, 3)
        phi = np.empty(shape, dtype=np.float32)
        phi[0] = 1.0
        phi[1] = 0.5
        phi[2:] = 0.0
        domain = _domain_from_phi(phi)
        state = domain.state
        assert state.vof is not None
        gas_mask = state.vof.cell_type.numpy() == int(VofCellType.GAS)
        rng = np.random.default_rng(9)
        base = rng.uniform(0.01, 0.05, size=(19, *shape)).astype(np.float32)

        first = base.copy()
        second = base.copy()
        for q in range(19):
            first[q][gas_mask] = rng.uniform(3.0, 8.0, size=np.count_nonzero(gas_mask))
            second[q][gas_mask] = rng.uniform(
                -9.0, -2.0, size=np.count_nonzero(gas_mask)
            )

        _assign_consistent_transport_state(domain, first)
        result_a = (
            domain.solver.compute_vof_mass_transport(state)
            .mass_tmp.numpy()
            .copy()
        )
        _assign_consistent_transport_state(domain, second)
        result_b = (
            domain.solver.compute_vof_mass_transport(state)
            .mass_tmp.numpy()
            .copy()
        )
        np.testing.assert_array_equal(result_a, result_b)

    def test_transport_is_read_only_and_phi_uses_density_n(self) -> None:
        shape = (3, 3, 3)
        phi = np.full(shape, 0.45, dtype=np.float32)
        domain = _domain_from_phi(phi, periodic=(True, True, True))
        rng = np.random.default_rng(71)
        populations = rng.uniform(0.01, 0.04, size=(19, *shape)).astype(np.float32)
        density = np.full(shape, 1.25, dtype=np.float32)
        _assign_consistent_transport_state(domain, populations, density)
        state = domain.state
        assert state.vof is not None
        snapshots = {
            "f_post": state.f_post.numpy().copy(),
            "density": state.density.numpy().copy(),
            "mass": state.vof.mass.numpy().copy(),
            "phi": state.vof.phi.numpy().copy(),
            "cell_type": state.vof.cell_type.numpy().copy(),
        }

        result = domain.solver.compute_vof_mass_transport(state)
        np.testing.assert_allclose(
            result.phi_tmp.numpy(),
            result.mass_tmp.numpy() / density,
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        np.testing.assert_array_equal(state.f_post.numpy(), snapshots["f_post"])
        np.testing.assert_array_equal(state.density.numpy(), snapshots["density"])
        np.testing.assert_array_equal(state.vof.mass.numpy(), snapshots["mass"])
        np.testing.assert_array_equal(state.vof.phi.numpy(), snapshots["phi"])
        np.testing.assert_array_equal(
            state.vof.cell_type.numpy(), snapshots["cell_type"]
        )


if __name__ == "__main__":
    unittest.main()
