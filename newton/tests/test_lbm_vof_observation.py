# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation-only VOF state, geometry, and visualization."""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    HomeLbmState,
    LbmDomain,
    LbmModel,
    VofCellType,
    VofInterfaceVisualizer,
)
from wanphys.examples.lbm import fluid_grid_lbm_dambreak_trt as dambreak_example


def _debug_model(
    *,
    shape: tuple[int, int, int] = (5, 3, 3),
    encoding: str = "fullf",
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding=encoding,
        collision="nocm_mrt" if encoding == "home" else "srt",
        vof_debug_labels=True,
        vof_debug_rho_gas=0.1,
        vof_debug_rho_liquid=1.1,
        vof_debug_epsilon=0.05,
        bc_periodic=periodic,
    )


class TestLbmVofObservation(unittest.TestCase):
    def test_debug_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                vof_debug_rho_gas=1.0,
                vof_debug_rho_liquid=1.0,
            )
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                vof_debug_epsilon=0.5,
            )

    def test_vof_storage_is_opt_in_for_fullf_and_home(self) -> None:
        plain = LbmDomain(
            LbmModel(fluid_grid_res=(2, 2, 2), device="cpu")
        ).create_state()
        self.assertIsNone(plain.vof)

        for encoding in ("fullf", "home"):
            state = LbmDomain(_debug_model(shape=(2, 2, 2), encoding=encoding)).create_state()
            if encoding == "home":
                self.assertIsInstance(state, HomeLbmState)
            self.assertIsNotNone(state.vof)
            assert state.vof is not None
            self.assertEqual(state.vof.cell_type.dtype, wp.uint8)
            self.assertEqual(state.vof.epoch, -1)
            self.assertEqual(state.vof.geometry.valid_epoch, -1)

    def test_state_clone_and_clear_preserve_vof_lifecycle(self) -> None:
        domain = LbmDomain(_debug_model(shape=(3, 2, 2)))
        state = domain.create_state()
        density = np.full((3, 2, 2), 0.6, dtype=np.float32)
        state.density.assign(density)
        domain.solver.update_debug_vof_labels(state)

        clone = state.clone()
        assert state.vof is not None and clone.vof is not None
        np.testing.assert_array_equal(clone.vof.phi.numpy(), state.vof.phi.numpy())
        np.testing.assert_array_equal(
            clone.vof.cell_type.numpy(), state.vof.cell_type.numpy()
        )
        np.testing.assert_array_equal(
            clone.vof.geometry.normal.numpy(), state.vof.geometry.normal.numpy()
        )
        self.assertEqual(clone.vof.epoch, state.vof.epoch)
        self.assertEqual(
            clone.vof.geometry.valid_epoch, state.vof.geometry.valid_epoch
        )

        clone.clear()
        self.assertEqual(clone.vof.epoch, -1)
        self.assertEqual(clone.vof.geometry.valid_epoch, -1)
        np.testing.assert_array_equal(clone.vof.phi.numpy(), 0.0)
        np.testing.assert_array_equal(
            clone.vof.cell_type.numpy(), int(VofCellType.GAS)
        )
        np.testing.assert_array_equal(clone.vof.geometry.normal.numpy(), 0.0)

    def test_density_classification_and_planar_normal(self) -> None:
        domain = LbmDomain(_debug_model())
        state = domain.create_state()
        phi_by_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        density = 0.1 + phi_by_x[:, None, None] * 1.0
        state.density.assign(np.broadcast_to(density, (5, 3, 3)).copy())
        domain.solver.update_debug_vof_labels(state)

        assert state.vof is not None
        phi = state.vof.phi.numpy()
        types = state.vof.cell_type.numpy()
        normals = state.vof.geometry.normal.numpy()
        np.testing.assert_allclose(phi[:, 1, 1], phi_by_x, atol=1.0e-6)
        self.assertTrue(np.all(types[0] == int(VofCellType.GAS)))
        self.assertTrue(np.all(types[-1] == int(VofCellType.LIQUID)))
        self.assertTrue(
            np.all(types[1:4] == int(VofCellType.INTERFACE))
        )
        np.testing.assert_allclose(normals[1:4, :, :, 0], -1.0, atol=1.0e-6)
        np.testing.assert_allclose(normals[1:4, :, :, 1:], 0.0, atol=1.0e-6)
        self.assertEqual(state.vof.geometry.valid_epoch, state.vof.epoch)

    def test_uniform_interface_has_zero_normal(self) -> None:
        domain = LbmDomain(_debug_model(shape=(3, 3, 3)))
        state = domain.create_state()
        state.density.fill_(0.6)
        domain.solver.update_debug_vof_labels(state)
        assert state.vof is not None
        self.assertTrue(
            np.all(
                state.vof.cell_type.numpy() == int(VofCellType.INTERFACE)
            )
        )
        np.testing.assert_array_equal(state.vof.geometry.normal.numpy(), 0.0)

    def test_periodic_normal_is_continuous_across_seam(self) -> None:
        domain = LbmDomain(
            _debug_model(shape=(6, 3, 3), periodic=(True, False, False))
        )
        state = domain.create_state()
        x = np.arange(6, dtype=np.float32)
        phi = 0.5 + 0.4 * np.sin(2.0 * np.pi * x / 6.0)
        density = 0.1 + phi[:, None, None]
        state.density.assign(np.broadcast_to(density, (6, 3, 3)).copy())
        domain.solver.update_debug_vof_labels(state)

        assert state.vof is not None
        normals = state.vof.geometry.normal.numpy()
        np.testing.assert_allclose(normals[0, :, :, 0], -1.0, atol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(normals)))

    def test_solid_cells_are_gas_with_zero_normal(self) -> None:
        domain = LbmDomain(_debug_model(shape=(3, 3, 3)))
        state = domain.create_state()
        state.density.fill_(0.6)
        solid = np.full((3, 3, 3), 1000.0, dtype=np.float32)
        solid[1, 1, 1] = -1.0
        state.solid_phi.assign(solid)
        domain.solver.update_debug_vof_labels(state)

        assert state.vof is not None
        self.assertEqual(
            int(state.vof.cell_type.numpy()[1, 1, 1]),
            int(VofCellType.GAS),
        )
        np.testing.assert_array_equal(
            state.vof.geometry.normal.numpy()[1, 1, 1], 0.0
        )

    def test_visual_compaction_returns_cell_centers_and_normals(self) -> None:
        domain = LbmDomain(_debug_model(shape=(3, 2, 2)))
        state = domain.create_state()
        density = np.full((3, 2, 2), 0.1, dtype=np.float32)
        density[0, :, :] = 1.1
        density[1, :, :] = 0.6
        state.density.assign(density)
        domain.solver.update_debug_vof_labels(state)

        assert state.vof is not None
        visualizer = VofInterfaceVisualizer(
            (3, 2, 2),
            domain.model._device,
            cell_size=2.0,
            origin=(10.0, 20.0, 30.0),
            normal_length_scale=0.5,
        )
        data = visualizer.compact(state.vof, state.solid_phi)
        self.assertEqual(data.count, 4)
        points = data.points.numpy()
        expected = np.array(
            [
                [13.0, 21.0, 31.0],
                [13.0, 21.0, 33.0],
                [13.0, 23.0, 31.0],
                [13.0, 23.0, 33.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            points[np.lexsort((points[:, 2], points[:, 1]))],
            expected,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            data.normal_ends.numpy() - points,
            np.tile(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                (data.count, 1),
            ),
            atol=1.0e-6,
        )

        class FakeViewer:
            def __init__(self) -> None:
                self.points: dict[str, object] | None = None

            def log_points(self, _name: str, **kwargs: object) -> None:
                self.points = kwargs

            def log_lines(self, _name: str, **_kwargs: object) -> None:
                pass

        viewer = FakeViewer()
        rendered_count = visualizer.render(
            viewer,
            state.vof,
            state.solid_phi,
            show_normals=False,
        )
        self.assertEqual(rendered_count, data.count)
        assert viewer.points is not None
        radii = viewer.points["radii"]
        colors = viewer.points["colors"]
        self.assertIsInstance(radii, wp.array)
        self.assertIsInstance(colors, wp.array)
        self.assertEqual(len(radii), data.count)
        self.assertEqual(len(colors), data.count)
        self.assertEqual(radii.dtype, wp.float32)
        self.assertEqual(colors.dtype, wp.vec3)

    def test_observer_does_not_change_shan_chen_physics(self) -> None:
        common = {
            "fluid_grid_res": (4, 4, 4),
            "device": "cpu",
            "force_model": "shan_chen",
            "G": -0.1,
            "psi_type": 0,
            "sc_force_stride": 1,
            "enforce_population_positivity": False,
        }
        plain = LbmDomain(LbmModel(**common))
        observed = LbmDomain(
            LbmModel(
                **common,
                vof_debug_labels=True,
                vof_debug_rho_gas=0.1,
                vof_debug_rho_liquid=1.1,
            )
        )
        plain_state = plain.create_state()
        observed_state = observed.create_state()
        plain.solver.initialize_equilibrium(plain_state, rho0=0.6)
        observed.solver.initialize_equilibrium(observed_state, rho0=0.6)

        plain.step(1.0)
        observed.step(1.0)
        np.testing.assert_array_equal(
            plain.state.f_post.numpy(), observed.state.f_post.numpy()
        )
        for name in (
            "density",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "force_x",
            "force_y",
            "force_z",
        ):
            np.testing.assert_array_equal(
                getattr(plain.state, name).numpy(),
                getattr(observed.state, name).numpy(),
            )

    def test_debug_dambreak_produces_a_nonempty_interface(self) -> None:
        args = argparse.Namespace(
            enc="fullf",
            col="trt",
            force="gravity+shan_chen",
            res=8,
            vof_debug_labels=True,
            vof_debug_no_normals=False,
        )
        model = dambreak_example.build_model(args)
        domain = LbmDomain(model)
        state = domain.create_state()
        dam_x = 3
        wp.launch(
            dambreak_example._init_fullf_dam,
            dim=(8, 8, 8),
            inputs=[
                state.f_post,
                state.density,
                dam_x,
                dambreak_example.RHO_WATER,
                dambreak_example.RHO_AIR,
                42,
                8,
                8,
                8,
                8 * 8 * 8,
            ],
            device=model._device,
        )
        dambreak_example._mirror_state(domain)
        domain.solver.update_debug_vof_labels(domain.state, domain._state_out)
        assert domain.state.vof is not None and domain._state_out.vof is not None
        self.assertEqual(domain.state.vof.epoch, domain._state_out.vof.epoch)
        self.assertEqual(
            domain.state.vof.geometry.valid_epoch,
            domain._state_out.vof.geometry.valid_epoch,
        )
        for _ in range(8):
            domain.step(1.0)

        assert domain.state.vof is not None
        cell_type = domain.state.vof.cell_type.numpy()
        fluid = domain.state.solid_phi.numpy() >= 0.0
        gas = int(((cell_type == int(VofCellType.GAS)) & fluid).sum())
        interface = int(
            ((cell_type == int(VofCellType.INTERFACE)) & fluid).sum()
        )
        liquid = int(((cell_type == int(VofCellType.LIQUID)) & fluid).sum())
        self.assertEqual(gas + interface + liquid, int(fluid.sum()))
        self.assertGreater(interface, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
