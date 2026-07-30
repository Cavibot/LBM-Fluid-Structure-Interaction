# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P1 acceptance tests for authoritative VOF state and initialization."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    HomeLbmState,
    LbmDomain,
    LbmModel,
    VofCellType,
    VofGridState,
)
from wanphys._src.fluid.fluid_grid.lbm.vof.initialization import (
    prepare_initial_vof,
    validate_initial_topology,
)
from wanphys._src.fluid.fluid_grid.lbm.vof.validation import (
    validate_empty_vof_state,
    validate_initialized_vof_state,
)


def _vof_model(
    *,
    shape: tuple[int, int, int] = (3, 2, 2),
    encoding: str = "fullf",
    bc_periodic: tuple[bool, bool, bool] = (False, False, False),
    bc_types: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding=encoding,
        collision="srt",
        interface_model="vof",
        bc_periodic=bc_periodic,
        bc_types=bc_types,
    )


def _layered_phi(
    shape: tuple[int, int, int] = (3, 2, 2),
    interface_phi: float = 0.25,
) -> np.ndarray:
    phi = np.empty(shape, dtype=np.float32)
    phi[0, :, :] = 1.0
    phi[1, :, :] = interface_phi
    phi[2:, :, :] = 0.0
    return phi


class TestVofP1ConfigurationAndState(unittest.TestCase):
    def test_vof_model_allocates_only_authoritative_storage(self) -> None:
        for encoding in ("fullf", "home"):
            with self.subTest(encoding=encoding):
                domain = LbmDomain(_vof_model(encoding=encoding))
                state = domain.create_state()
                self.assertIsInstance(state.vof, VofGridState)
                self.assertIsNone(state.debug_mock_sc_to_vof)
                if encoding == "home":
                    self.assertIsInstance(state, HomeLbmState)
                assert state.vof is not None
                self.assertEqual(state.vof.mass.dtype, wp.float32)
                self.assertEqual(state.vof.phi.dtype, wp.float32)
                self.assertEqual(state.vof.cell_type.dtype, wp.uint8)
                validate_empty_vof_state(state.vof)

    def test_configuration_rejects_debug_gradient_and_solid_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires interface_model"):
            LbmModel(
                fluid_grid_res=(3, 2, 2),
                device="cpu",
                interface_model="vof",
                debug_vof_observation=True,
            )
        with self.assertRaisesRegex(ValueError, "non-zero Shan-Chen G"):
            LbmModel(
                fluid_grid_res=(3, 2, 2),
                device="cpu",
                interface_model="vof",
                G=-0.1,
            )
        with self.assertRaisesRegex(NotImplementedError, "moving-wall"):
            _vof_model_with_boundaries("moving_wall")
        with self.assertRaisesRegex(NotImplementedError, "cut-link"):
            _vof_model_with_boundaries("cut_link")

        domain = LbmDomain(_vof_model())
        with self.assertRaisesRegex(NotImplementedError, "requires_grad"):
            domain.solver.create_state(requires_grad=True)

    def test_clear_clone_and_copy_use_independent_vof_storage(self) -> None:
        state = LbmDomain(_vof_model()).create_state()
        assert state.vof is not None
        state.vof.mass.assign(np.full(state.res, 0.25, dtype=np.float32))
        state.vof.phi.assign(np.full(state.res, 0.5, dtype=np.float32))
        state.vof.cell_type.fill_(int(VofCellType.INTERFACE))

        clone = state.clone()
        assert clone.vof is not None
        for name in (
            "mass",
            "phi",
            "cell_type",
            "normal",
            "plic_offset",
            "curvature",
        ):
            source = getattr(state.vof, name)
            target = getattr(clone.vof, name)
            self.assertIsNot(source, target)
            self.assertNotEqual(int(source.ptr), int(target.ptr))
            np.testing.assert_array_equal(source.numpy(), target.numpy())

        clone.clear()
        validate_empty_vof_state(clone.vof)
        self.assertTrue(np.all(state.vof.mass.numpy() == 0.25))

    def test_copy_to_state_without_vof_storage_is_rejected(self) -> None:
        source = LbmDomain(_vof_model()).create_state()
        target = LbmDomain(
            LbmModel(fluid_grid_res=(3, 2, 2), device="cpu")
        ).create_state()
        with self.assertRaisesRegex(ValueError, "without storage"):
            source._copy_common_to(target)


def _vof_model_with_boundaries(boundary: str) -> LbmModel:
    return LbmModel(
        fluid_grid_res=(3, 2, 2),
        device="cpu",
        interface_model="vof",
        boundary_models=(
            boundary,
            "bounce_back",
            "bounce_back",
            "bounce_back",
            "bounce_back",
            "bounce_back",
        ),
    )


class TestVofP1Preparation(unittest.TestCase):
    def test_phi_is_validated_then_canonicalized_and_exactly_classified(self) -> None:
        raw = np.array([0.0, 0.25, 1.0 - 1.0e-10], dtype=np.float64).reshape(
            3, 1, 1
        )
        phi, cell_type = prepare_initial_vof(
            raw,
            shape=(3, 1, 1),
            periodic=(False, False, False),
        )
        self.assertEqual(phi.dtype, np.float32)
        self.assertTrue(phi.flags.c_contiguous)
        np.testing.assert_array_equal(
            cell_type[:, 0, 0],
            np.array(
                [
                    int(VofCellType.GAS),
                    int(VofCellType.INTERFACE),
                    int(VofCellType.LIQUID),
                ],
                dtype=np.uint8,
            ),
        )

    def test_invalid_phi_inputs_fail_before_canonicalization(self) -> None:
        invalid_inputs = (
            np.zeros((3, 2, 2), dtype=np.int32),
            np.full((3, 2, 2), np.nan, dtype=np.float32),
            np.full((3, 2, 2), np.inf, dtype=np.float32),
            np.full((3, 2, 2), -0.1, dtype=np.float32),
            np.full((3, 2, 2), 1.1, dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.float32),
        )
        for phi in invalid_inputs:
            with self.subTest(dtype=phi.dtype, shape=phi.shape, value=phi.flat[0]):
                with self.assertRaises((TypeError, ValueError)):
                    prepare_initial_vof(
                        phi,
                        shape=(3, 2, 2),
                        periodic=(False, False, False),
                    )

    def test_face_edge_and_periodic_seam_liquid_gas_links_are_rejected(self) -> None:
        interface = int(VofCellType.INTERFACE)
        liquid = int(VofCellType.LIQUID)
        gas = int(VofCellType.GAS)

        face = np.full((3, 3, 3), interface, dtype=np.uint8)
        face[0, 1, 1], face[1, 1, 1] = liquid, gas
        with self.assertRaisesRegex(ValueError, "LIQUID-GAS"):
            validate_initial_topology(face, periodic=(False, False, False))

        edge = np.full((3, 3, 3), interface, dtype=np.uint8)
        edge[0, 0, 1], edge[1, 1, 1] = liquid, gas
        with self.assertRaisesRegex(ValueError, "direction="):
            validate_initial_topology(edge, periodic=(False, False, False))

        seam = np.full((4, 2, 2), interface, dtype=np.uint8)
        seam[0, :, :], seam[-1, :, :] = gas, liquid
        validate_initial_topology(seam, periodic=(False, False, False))
        with self.assertRaisesRegex(ValueError, "LIQUID-GAS"):
            validate_initial_topology(seam, periodic=(True, False, False))

    def test_disconnected_gas_regions_are_not_analyzed(self) -> None:
        cell_type = np.full(
            (4, 3, 3),
            int(VofCellType.INTERFACE),
            dtype=np.uint8,
        )
        cell_type[0, 0, 0] = int(VofCellType.GAS)
        cell_type[3, 2, 2] = int(VofCellType.GAS)
        validate_initial_topology(cell_type, periodic=(False, False, False))


class TestVofP1Initialization(unittest.TestCase):
    def test_fullf_and_home_share_density_authority_and_double_buffer_semantics(
        self,
    ) -> None:
        phi0 = _layered_phi()
        for encoding in ("fullf", "home"):
            with self.subTest(encoding=encoding):
                domain = LbmDomain(_vof_model(encoding=encoding))
                state = domain.initialize_vof(
                    phi0,
                    rho0=1.25,
                    u0=(0.02, -0.01, 0.005),
                )
                assert state.vof is not None
                self.assertTrue(np.all(state.density.numpy() == np.float32(1.25)))
                np.testing.assert_allclose(state.vof.phi.numpy(), phi0, atol=0.0)
                np.testing.assert_allclose(
                    state.vof.mass.numpy(),
                    state.density.numpy() * phi0,
                    atol=1.0e-7,
                    rtol=1.0e-6,
                )
                validate_initialized_vof_state(
                    state,
                    periodic=(False, False, False),
                )

                state_out = domain._state_out
                assert state_out is not None and state_out.vof is not None
                for name in ("mass", "phi", "cell_type"):
                    source = getattr(state.vof, name)
                    target = getattr(state_out.vof, name)
                    np.testing.assert_array_equal(source.numpy(), target.numpy())
                    self.assertNotEqual(int(source.ptr), int(target.ptr))
                if encoding == "fullf":
                    np.testing.assert_array_equal(
                        state.f_post.numpy(), state_out.f_post.numpy()
                    )
                else:
                    assert isinstance(state, HomeLbmState)
                    assert isinstance(state_out, HomeLbmState)
                    for source, target in zip(
                        state.kinetic_fields, state_out.kinetic_fields
                    ):
                        np.testing.assert_array_equal(
                            source.numpy(), target.numpy()
                        )

    def test_actual_state_density_not_rho_argument_controls_mass(self) -> None:
        domain = LbmDomain(_vof_model())
        state = domain.solver.create_state()
        domain.solver.initialize_equilibrium(state, rho0=1.0)
        density = np.linspace(0.8, 1.3, num=12, dtype=np.float32).reshape(
            state.res
        )
        state.density.assign(density)
        phi0 = np.full(state.res, 0.25, dtype=np.float32)

        domain.solver.initialize_vof_state(state, phi0)

        assert state.vof is not None
        np.testing.assert_allclose(
            state.vof.mass.numpy(),
            density * phi0,
            atol=1.0e-7,
            rtol=1.0e-6,
        )

    def test_density_precondition_distinguishes_empty_and_initialized_all_gas(
        self,
    ) -> None:
        domain = LbmDomain(_vof_model())
        empty = domain.solver.create_state()
        with self.assertRaisesRegex(ValueError, "density must be positive"):
            domain.solver.initialize_vof_state(
                empty,
                np.zeros(empty.res, dtype=np.float32),
            )

        initialized = domain.initialize_vof(
            np.zeros((3, 2, 2), dtype=np.float32),
            rho0=1.0,
        )
        validate_initialized_vof_state(
            initialized,
            periodic=(False, False, False),
        )

    def test_all_liquid_initialization_is_valid(self) -> None:
        domain = LbmDomain(_vof_model())
        state = domain.initialize_vof(
            np.ones((3, 2, 2), dtype=np.float32),
            rho0=1.15,
        )
        assert state.vof is not None
        np.testing.assert_array_equal(
            state.vof.cell_type.numpy(),
            int(VofCellType.LIQUID),
        )
        np.testing.assert_array_equal(state.vof.mass.numpy(), state.density.numpy())
        validate_initialized_vof_state(
            state,
            periodic=(False, False, False),
        )

    def test_invalid_first_and_reinitialization_are_transactional(self) -> None:
        invalid = np.zeros((3, 2, 2), dtype=np.float32)
        invalid[0, :, :] = 1.0
        fresh = LbmDomain(_vof_model())
        with self.assertRaisesRegex(ValueError, "LIQUID-GAS"):
            fresh.initialize_vof(invalid)
        self.assertIsNone(fresh._state_in)
        self.assertIsNone(fresh._state_out)

        domain = LbmDomain(_vof_model())
        original_in = domain.initialize_vof(_layered_phi(), rho0=1.1)
        original_out = domain._state_out
        assert original_in.vof is not None and original_out is not None
        density_before = original_in.density.numpy().copy()
        mass_before = original_in.vof.mass.numpy().copy()

        with self.assertRaisesRegex(ValueError, "LIQUID-GAS"):
            domain.initialize_vof(invalid, rho0=1.3)

        self.assertIs(domain._state_in, original_in)
        self.assertIs(domain._state_out, original_out)
        np.testing.assert_array_equal(original_in.density.numpy(), density_before)
        np.testing.assert_array_equal(original_in.vof.mass.numpy(), mass_before)

    def test_reinitialization_replaces_both_buffers_after_success(self) -> None:
        domain = LbmDomain(_vof_model())
        old_in = domain.initialize_vof(_layered_phi(interface_phi=0.25))
        old_out = domain._state_out
        new_in = domain.initialize_vof(
            _layered_phi(interface_phi=0.75),
            rho0=1.2,
        )
        self.assertIsNot(new_in, old_in)
        self.assertIsNot(domain._state_out, old_out)
        assert new_in.vof is not None
        self.assertTrue(np.all(new_in.vof.phi.numpy()[1, :, :] == 0.75))

    def test_embedded_solid_is_rejected_but_static_outer_walls_are_allowed(
        self,
    ) -> None:
        domain = LbmDomain(_vof_model())
        state = domain.create_state()
        solid_phi = np.full(state.res, 1000.0, dtype=np.float32)
        solid_phi[1, 0, 0] = -0.1
        state.solid_phi.assign(solid_phi)
        with self.assertRaisesRegex(NotImplementedError, "embedded solid"):
            domain.initialize_vof(_layered_phi())
        self.assertIs(domain._state_in, state)

        ordinary_walls = LbmDomain(_vof_model())
        ordinary_walls.initialize_vof(_layered_phi())

    def test_home_step_preserves_p1_state_contract_after_p7_extension(self) -> None:
        domain = LbmDomain(_vof_model(encoding="home"))
        state_in = domain.initialize_vof(_layered_phi())
        state_out = domain._state_out
        assert state_in.vof is not None and state_out is not None
        mass_before = float(np.sum(state_in.vof.mass.numpy(), dtype=np.float64))
        domain.step(1.0)
        self.assertIs(domain._state_in, state_out)
        self.assertIs(domain._state_out, state_in)
        assert domain.state.vof is not None
        self.assertAlmostEqual(
            float(np.sum(domain.state.vof.mass.numpy(), dtype=np.float64)),
            mass_before,
            places=5,
        )
        self.assertEqual(
            domain.state.vof.epoch,
            domain.state.vof.geometry_epoch,
        )

    def test_uniform_initial_conditions_must_survive_float32_conversion(self) -> None:
        domain = LbmDomain(_vof_model())
        for rho0 in (0.0, -1.0, np.nan, np.inf, 1.0e-50, 1.0e50):
            with self.subTest(rho0=rho0):
                with self.assertRaises(ValueError):
                    domain.initialize_vof(_layered_phi(), rho0=rho0)
        with self.assertRaises(ValueError):
            domain.initialize_vof(_layered_phi(), u0=(0.0, np.nan, 0.0))

    def test_bc_type_periodicity_is_used_by_domain_topology_check(self) -> None:
        domain = LbmDomain(
            _vof_model(
                shape=(4, 2, 2),
                bc_types=(3, 3, 0, 0, 0, 0),
            )
        )
        phi = np.full((4, 2, 2), 0.5, dtype=np.float32)
        phi[0, :, :] = 0.0
        phi[-1, :, :] = 1.0
        with self.assertRaisesRegex(ValueError, "LIQUID-GAS"):
            domain.initialize_vof(phi)


if __name__ == "__main__":
    unittest.main()
