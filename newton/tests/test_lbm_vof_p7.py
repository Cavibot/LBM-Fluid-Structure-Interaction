# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P7 HOME, integration-ledger and conditional device acceptance tests."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    HomeLbmState,
    LbmDomain,
    LbmModel,
)
from wanphys._src.fluid.fluid_grid.lbm.vof import VofCellType
from wanphys._src.fluid.fluid_grid.lbm.vof.diagnostics.host import (
    VofDiagnostics,
    collect_vof_diagnostics,
    validate_vof_diagnostics,
)


def _model(
    shape: tuple[int, int, int],
    *,
    encoding: str = "fullf",
    device: str = "cpu",
    periodic: tuple[bool, bool, bool] = (False, False, False),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    surface_tension: float = 0.0,
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device=device,
        encoding=encoding,
        collision="srt",
        interface_model="vof",
        bc_periodic=periodic,
        gravity_x=gravity[0],
        gravity_y=gravity[1],
        gravity_z=gravity[2],
        vof_surface_tension=surface_tension,
        enforce_population_positivity=False,
    )


def _layered_phi(shape: tuple[int, int, int]) -> np.ndarray:
    phi = np.zeros(shape, dtype=np.float32)
    split = shape[0] // 2
    phi[:split] = 1.0
    phi[split] = 0.5
    return phi


def _dam_break_phi(shape: tuple[int, int, int]) -> np.ndarray:
    """Return a liquid column wrapped in one D3Q19 interface layer."""

    liquid = np.zeros(shape, dtype=bool)
    liquid[: shape[0] // 3, :, : 2 * shape[2] // 3] = True
    interface = np.zeros(shape, dtype=bool)
    directions = (
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    )
    for i, j, k in np.argwhere(liquid):
        for di, dj, dk in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            if (
                0 <= ni < shape[0]
                and 0 <= nj < shape[1]
                and 0 <= nk < shape[2]
                and not liquid[ni, nj, nk]
            ):
                interface[ni, nj, nk] = True
    phi = np.zeros(shape, dtype=np.float32)
    phi[interface] = 0.5
    phi[liquid] = 1.0
    return phi


def _domain_pair(
    shape: tuple[int, int, int],
    phi: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    surface_tension: float = 0.0,
) -> tuple[LbmDomain, LbmDomain]:
    domains = tuple(
        LbmDomain(
            _model(
                shape,
                encoding=encoding,
                periodic=periodic,
                gravity=gravity,
                surface_tension=surface_tension,
            )
        )
        for encoding in ("fullf", "home")
    )
    for domain in domains:
        domain.initialize_vof(phi, u0=velocity)
    return domains


def _assert_cross_encoding(
    test: unittest.TestCase,
    fullf: LbmDomain,
    home: LbmDomain,
    *,
    scalar_atol: float = 2.0e-4,
    velocity_atol: float = 3.0e-4,
) -> None:
    first, second = fullf.state, home.state
    assert first.vof is not None and second.vof is not None
    np.testing.assert_array_equal(
        first.vof.cell_type.numpy(),
        second.vof.cell_type.numpy(),
    )
    for name in ("mass", "phi", "pending_excess", "curvature"):
        np.testing.assert_allclose(
            getattr(first.vof, name).numpy(),
            getattr(second.vof, name).numpy(),
            atol=scalar_atol,
            rtol=2.0e-4,
            err_msg=name,
        )
    np.testing.assert_allclose(
        first.density.numpy(),
        second.density.numpy(),
        atol=scalar_atol,
        rtol=2.0e-4,
    )
    for name in ("velocity_x", "velocity_y", "velocity_z"):
        np.testing.assert_allclose(
            getattr(first, name).numpy(),
            getattr(second, name).numpy(),
            atol=velocity_atol,
            rtol=3.0e-4,
            err_msg=name,
        )
    test.assertEqual(first.vof.epoch, second.vof.epoch)
    test.assertEqual(first.vof.geometry_epoch, second.vof.geometry_epoch)


class TestVofP7LogicalPopulationAndHome(unittest.TestCase):
    def test_equilibrium_logical_population_provider_matches_fullf_home(self) -> None:
        shape = (5, 3, 3)
        fullf, home = _domain_pair(
            shape,
            _layered_phi(shape),
            velocity=(0.03, -0.015, 0.01),
        )
        fullf_logical = fullf.solver._vof_postcollision_populations(
            fullf.state
        ).numpy()
        home_logical = home.solver._vof_postcollision_populations(
            home.state
        ).numpy()
        np.testing.assert_allclose(
            fullf_logical,
            home_logical,
            atol=5.0e-6,
            rtol=2.0e-4,
        )

    def test_home_p2_and_eq11_share_fullf_oracles(self) -> None:
        shape = (5, 3, 3)
        fullf, home = _domain_pair(
            shape,
            _layered_phi(shape),
            velocity=(0.02, -0.01, 0.005),
        )
        fullf_mass = (
            fullf.solver.compute_vof_mass_transport(fullf.state)
            .mass_tmp.numpy()
            .copy()
        )
        home_mass = (
            home.solver.compute_vof_mass_transport(home.state)
            .mass_tmp.numpy()
            .copy()
        )
        np.testing.assert_allclose(fullf_mass, home_mass, atol=4.0e-6)
        fullf_surface = (
            fullf.solver.compute_vof_surface_populations(fullf.state)
            .numpy()
            .copy()
        )
        home_surface = (
            home.solver.compute_vof_surface_populations(home.state)
            .numpy()
            .copy()
        )
        np.testing.assert_allclose(fullf_surface, home_surface, atol=6.0e-6)

    def test_home_gas_storage_isolated_from_active_surface_and_mass(self) -> None:
        shape = (5, 3, 3)
        domains = (
            LbmDomain(_model(shape, encoding="home")),
            LbmDomain(_model(shape, encoding="home")),
        )
        rng = np.random.default_rng(707)
        for index, domain in enumerate(domains):
            state = domain.initialize_vof(_layered_phi(shape))
            assert isinstance(state, HomeLbmState) and state.vof is not None
            gas = state.vof.cell_type.numpy() == int(VofCellType.GAS)
            for field in state.kinetic_fields:
                values = field.numpy()
                values[gas] = rng.uniform(
                    2.0 + 4.0 * index,
                    3.0 + 4.0 * index,
                    size=np.count_nonzero(gas),
                )
                field.assign(values)
        masses = [
            domain.solver.compute_vof_mass_transport(domain.state)
            .mass_tmp.numpy()
            .copy()
            for domain in domains
        ]
        surfaces = [
            domain.solver.compute_vof_surface_populations(domain.state)
            .numpy()
            .reshape((19, *shape))
            .copy()
            for domain in domains
        ]
        active = (
            domains[0].state.vof.cell_type.numpy()
            != int(VofCellType.GAS)
        )
        np.testing.assert_array_equal(masses[0], masses[1])
        np.testing.assert_allclose(
            surfaces[0][:, active],
            surfaces[1][:, active],
            atol=2.0e-7,
        )

    def test_home_new_interface_writes_equilibrium_moments(self) -> None:
        shape = (5, 3, 3)
        fullf, home = _domain_pair(shape, _layered_phi(shape))
        old_type = home.state.vof.cell_type.numpy().copy()
        for domain in (fullf, home):
            state = domain.state
            assert state.vof is not None
            center = (2, 1, 1)
            mass = state.vof.mass.numpy()
            phi = state.vof.phi.numpy()
            mass[center] = np.float32(1.2)
            phi[center] = np.float32(1.2)
            state.vof.mass.assign(mass)
            state.vof.phi.assign(phi)
            reference_mass = float(np.sum(mass, dtype=np.float64))
            state.vof.reference_mass = reference_mass
            assert domain._state_out is not None
            assert domain._state_out.vof is not None
            domain._state_out.vof.reference_mass = reference_mass
            domain.solver._vof_interface_geometry.compute(state.vof)
            domain.step(1.0)

        state = home.state
        assert isinstance(state, HomeLbmState) and state.vof is not None
        new_interface = (
            (old_type == int(VofCellType.GAS))
            & (state.vof.cell_type.numpy() == int(VofCellType.INTERFACE))
        )
        self.assertGreater(int(np.count_nonzero(new_interface)), 0)
        rho = state.rho.numpy()
        ux, uy, uz = (
            state.velocity_x.numpy(),
            state.velocity_y.numpy(),
            state.velocity_z.numpy(),
        )
        expected = (
            rho,
            rho * ux,
            rho * uy,
            rho * uz,
            rho * ux * ux,
            rho * uy * uy,
            rho * uz * uz,
            rho * ux * uy,
            rho * ux * uz,
            rho * uy * uz,
        )
        for actual, target in zip(
            state.kinetic_fields, expected, strict=True
        ):
            np.testing.assert_allclose(
                actual.numpy()[new_interface],
                target[new_interface],
                atol=3.0e-6,
            )
        _assert_cross_encoding(self, fullf, home)


class TestVofP7IntegratedAcceptance(unittest.TestCase):
    def test_planar_fullf_home_multistep_differential(self) -> None:
        shape = (9, 4, 4)
        fullf, home = _domain_pair(shape, _layered_phi(shape))
        initial_mass = float(
            np.sum(fullf.state.vof.mass.numpy(), dtype=np.float64)
        )
        for _ in range(20):
            fullf.step(1.0)
            home.step(1.0)
            _assert_cross_encoding(self, fullf, home)
        for domain in (fullf, home):
            diagnostics = collect_vof_diagnostics(
                domain.state,
                initial_mass=initial_mass,
            )
            validate_vof_diagnostics(diagnostics)

    def test_periodic_uniform_fill_translation_is_closed_and_cross_encoded(
        self,
    ) -> None:
        shape = (6, 4, 3)
        phi = np.full(shape, 0.5, dtype=np.float32)
        fullf, home = _domain_pair(
            shape,
            phi,
            periodic=(True, True, True),
            velocity=(0.025, -0.0125, 0.00625),
        )
        initial_mass = float(np.sum(phi, dtype=np.float64))
        for _ in range(12):
            fullf.step(1.0)
            home.step(1.0)
        _assert_cross_encoding(self, fullf, home)
        for domain in (fullf, home):
            diagnostics = collect_vof_diagnostics(
                domain.state,
                initial_mass=initial_mass,
                periodic=(True, True, True),
            )
            validate_vof_diagnostics(diagnostics)

    def test_gravity_and_surface_tension_short_scenarios_are_admissible(
        self,
    ) -> None:
        shape = (9, 4, 4)
        for gravity, gamma in (
            ((0.0, -1.0e-5, 0.0), 0.0),
            ((0.0, 0.0, 0.0), 0.02),
        ):
            with self.subTest(gravity=gravity, gamma=gamma):
                fullf, home = _domain_pair(
                    shape,
                    _layered_phi(shape),
                    gravity=gravity,
                    surface_tension=gamma,
                )
                initial_mass = float(
                    np.sum(fullf.state.vof.mass.numpy(), dtype=np.float64)
                )
                for _ in range(5):
                    fullf.step(1.0)
                    home.step(1.0)
                _assert_cross_encoding(self, fullf, home)
                for domain in (fullf, home):
                    validate_vof_diagnostics(
                        collect_vof_diagnostics(
                            domain.state,
                            initial_mass=initial_mass,
                        )
                    )

    def test_long_closed_domain_mass_ledger(self) -> None:
        shape = (9, 3, 3)
        for encoding in ("fullf", "home"):
            with self.subTest(encoding=encoding):
                domain = LbmDomain(_model(shape, encoding=encoding))
                state = domain.initialize_vof(_layered_phi(shape))
                assert state.vof is not None
                initial_mass = float(
                    np.sum(state.vof.mass.numpy(), dtype=np.float64)
                )
                for _ in range(60):
                    domain.step(1.0)
                diagnostics = collect_vof_diagnostics(
                    domain.state,
                    initial_mass=initial_mass,
                )
                validate_vof_diagnostics(diagnostics)
                self.assertLessEqual(diagnostics.relative_mass_error, 5.0e-6)

    def test_headless_dam_break_quantitative_smoke(self) -> None:
        shape = (12, 3, 9)
        phi = _dam_break_phi(shape)
        for encoding in ("fullf", "home"):
            with self.subTest(encoding=encoding):
                domain = LbmDomain(
                    _model(
                        shape,
                        encoding=encoding,
                        periodic=(False, True, False),
                        gravity=(0.0, 0.0, -2.0e-5),
                    )
                )
                initial = domain.initialize_vof(phi)
                assert initial.vof is not None
                initial_mass = float(
                    np.sum(initial.vof.mass.numpy(), dtype=np.float64)
                )
                initial_interface_count = int(
                    np.count_nonzero(
                        initial.vof.cell_type.numpy()
                        == int(VofCellType.INTERFACE)
                    )
                )
                ledger: list[VofDiagnostics] = []
                for _ in range(15):
                    domain.step(1.0)
                    diagnostics = collect_vof_diagnostics(
                        domain.state,
                        initial_mass=initial_mass,
                        periodic=(False, True, False),
                    )
                    validate_vof_diagnostics(diagnostics)
                    ledger.append(diagnostics)
                self.assertEqual(len(ledger), 15)
                diagnostics = ledger[-1]
                self.assertGreater(diagnostics.max_velocity, 0.0)
                self.assertGreater(
                    max(item.max_velocity for item in ledger),
                    min(item.max_velocity for item in ledger),
                )
                self.assertGreater(initial_interface_count, 0)
                self.assertGreater(diagnostics.interface_cell_count, 0)

    def test_diagnostic_validator_rejects_each_material_invariant(self) -> None:
        valid = VofDiagnostics(
            total_mass=1.0,
            initial_mass=1.0,
            boundary_mass_flux=0.0,
            relative_mass_error=0.0,
            phi_min=0.0,
            phi_max=1.0,
            invalid_liquid_gas_adjacency_count=0,
            non_finite_count=0,
            nonpositive_active_density_count=0,
            max_velocity=0.1,
            interface_cell_count=1,
            epoch=2,
            geometry_epoch=2,
        )
        validate_vof_diagnostics(valid)
        fixtures = (
            ("mass", {"relative_mass_error": 6.0e-6}),
            ("phi", {"phi_max": 1.01}),
            ("topology", {"invalid_liquid_gas_adjacency_count": 1}),
            ("NaN", {"non_finite_count": 1}),
            ("density", {"nonpositive_active_density_count": 1}),
            ("velocity", {"max_velocity": 0.41}),
            ("epoch", {"geometry_epoch": 1}),
        )
        for message, changes in fixtures:
            values = {**valid.__dict__, **changes}
            with self.subTest(message=message):
                with self.assertRaises(ValueError):
                    validate_vof_diagnostics(VofDiagnostics(**values))

    @unittest.skipUnless(
        wp.is_cuda_available(),
        "CUDA unavailable: P7 status must remain CUDA_NOT_ACCEPTED",
    )
    def test_cpu_cuda_fullf_home_smoke_and_topology_match(self) -> None:
        shape = (7, 3, 3)
        phi = _layered_phi(shape)
        for encoding in ("fullf", "home"):
            cpu = LbmDomain(_model(shape, encoding=encoding, device="cpu"))
            cuda = LbmDomain(_model(shape, encoding=encoding, device="cuda:0"))
            cpu.initialize_vof(phi, u0=(0.01, 0.0, 0.0))
            cuda.initialize_vof(phi, u0=(0.01, 0.0, 0.0))
            for _ in range(5):
                cpu.step(1.0)
                cuda.step(1.0)
            assert cpu.state.vof is not None and cuda.state.vof is not None
            np.testing.assert_array_equal(
                cpu.state.vof.cell_type.numpy(),
                cuda.state.vof.cell_type.numpy(),
            )
            for name in ("mass", "phi", "pending_excess", "curvature"):
                np.testing.assert_allclose(
                    getattr(cpu.state.vof, name).numpy(),
                    getattr(cuda.state.vof, name).numpy(),
                    atol=5.0e-5,
                    rtol=5.0e-5,
                )


if __name__ == "__main__":
    unittest.main()
