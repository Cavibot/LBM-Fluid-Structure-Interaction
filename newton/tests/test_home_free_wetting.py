# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Static contact-angle closure for geometric HOME-Free PLIC normals."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeInterfaceGeometry,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    contact_angle_interface_normal,
)


class TestHomeFreeWetting(unittest.TestCase):
    _SHAPE = (5, 5, 5)
    _CENTER = (2, 2, 1)

    def _wall_state(
        self,
        device: str,
        *,
        interface_axis: int = 0,
        constant_phi: bool = False,
    ) -> tuple[HomeLbmModel, HomeLbmState, HomeFreeState]:
        model = HomeLbmModel(
            fluid_grid_res=self._SHAPE,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        coordinates = np.indices(self._SHAPE, dtype=np.float32)
        if constant_phi:
            solid_phi = np.ones(self._SHAPE, dtype=np.float32)
        else:
            solid_phi = coordinates[2] - 0.5
        fluid.solid_phi.assign(solid_phi)

        coordinate = coordinates[interface_axis]
        center_coordinate = float(self._CENTER[interface_axis])
        fill = np.where(
            coordinate < center_coordinate,
            1.0,
            np.where(coordinate == center_coordinate, 0.5, 0.0),
        ).astype(np.float32)
        flags = np.where(fill == 1.0, 2, 0).astype(np.int32)
        flags[:, :, 0] = 3
        flags[self._CENTER] = 1

        state = HomeFreeState(model)
        state.fill_level.assign(fill)
        state.flags.assign(flags)
        return model, fluid, state

    def _angle_case(self, device: str, angle: float) -> None:
        model, fluid, state = self._wall_state(device)
        geometry = HomeFreeInterfaceGeometry(
            model, contact_angle_degrees=angle
        )
        diagnostics = geometry.update(
            state, solid_phi=fluid.solid_phi, compute_curvature=False
        )

        expected = contact_angle_interface_normal(
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            angle,
        )
        actual = geometry.normal.numpy()[self._CENTER]
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
        self.assertAlmostEqual(float(np.dot(actual, (0.0, 0.0, 1.0))), np.cos(np.deg2rad(angle)), places=6)
        self.assertEqual(diagnostics.interface_cell_count, 1)
        self.assertEqual(diagnostics.wetting_cell_count, 1)
        self.assertEqual(diagnostics.invalid_wall_normal_count, 0)
        self.assertEqual(diagnostics.undefined_contact_tangent_count, 0)
        self.assertEqual(diagnostics.unconfigured_wetting_cell_count, 0)
        self.assertEqual(int(geometry.valid.numpy()[self._CENTER]), 1)

    def test_reference_contact_angle_convention(self) -> None:
        for angle in (30.0, 60.0, 90.0, 120.0, 150.0):
            with self.subTest(angle=angle):
                normal = contact_angle_interface_normal(
                    np.asarray((1.0, 0.0, 0.2)),
                    np.asarray((0.0, 0.0, 2.0)),
                    angle,
                )
                self.assertAlmostEqual(
                    float(np.dot(normal, (0.0, 0.0, 1.0))),
                    float(np.cos(np.deg2rad(angle))),
                    places=13,
                )
                self.assertGreater(float(normal[0]), 0.0)

    def test_contact_angle_configuration_is_strict(self) -> None:
        model = HomeLbmModel(fluid_grid_res=(3, 3, 3), device="cpu")
        for value in (0.0, 180.0, -1.0, 181.0, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "contact_angle_degrees"
            ):
                HomeFreeInterfaceGeometry(model, contact_angle_degrees=value)

    def test_cpu_planar_wall_angles_match_reference(self) -> None:
        for angle in (60.0, 90.0, 120.0):
            with self.subTest(angle=angle):
                self._angle_case("cpu", angle)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_planar_wall_angles_match_reference(self) -> None:
        for angle in (60.0, 90.0, 120.0):
            with self.subTest(angle=angle):
                self._angle_case("cuda:0", angle)

    def test_solid_contact_requires_explicit_angle(self) -> None:
        model, fluid, state = self._wall_state("cpu")
        with self.assertRaisesRegex(RuntimeError, "explicitly configured contact angle"):
            HomeFreeInterfaceGeometry(model).update(
                state, solid_phi=fluid.solid_phi, compute_curvature=False
            )

    def test_nonperiodic_domain_wall_uses_explicit_neutral_contact_angle(self) -> None:
        shape = (5, 5, 5)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :2] = 1.0
        fill[:, :, 2] = 0.5
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)

        with self.assertRaisesRegex(RuntimeError, "explicitly configured contact angle"):
            HomeFreeInterfaceGeometry(model).update(
                state, solid_phi=fluid.solid_phi, compute_curvature=False
            )

        geometry = HomeFreeInterfaceGeometry(
            model, contact_angle_degrees=90.0
        )
        diagnostics = geometry.update(
            state, solid_phi=fluid.solid_phi, compute_curvature=False
        )
        interface = state.flags.numpy() == 1
        expected = np.zeros((int(np.sum(interface)), 3), dtype=np.float32)
        expected[:, 2] = 1.0
        np.testing.assert_allclose(
            geometry.normal.numpy()[interface], expected, rtol=2.0e-6, atol=2.0e-6
        )
        self.assertEqual(diagnostics.interface_cell_count, 25)
        self.assertEqual(diagnostics.wetting_cell_count, 16)
        self.assertEqual(diagnostics.invalid_stencil_count, 0)

    def test_invalid_wall_normal_is_rejected(self) -> None:
        model, fluid, state = self._wall_state("cpu", constant_phi=True)
        with self.assertRaisesRegex(RuntimeError, "undefined solid_phi wall normals"):
            HomeFreeInterfaceGeometry(
                model, contact_angle_degrees=90.0
            ).update(state, solid_phi=fluid.solid_phi, compute_curvature=False)

    def test_undefined_wall_tangent_is_rejected(self) -> None:
        model, fluid, state = self._wall_state("cpu", interface_axis=2)
        coordinates = np.indices(self._SHAPE, dtype=np.float32)[2]
        fill = np.where(
            coordinates > self._CENTER[2],
            1.0,
            np.where(coordinates == self._CENTER[2], 0.5, 0.0),
        ).astype(np.float32)
        state.fill_level.assign(fill)
        with self.assertRaisesRegex(RuntimeError, "undefined wall-tangent"):
            HomeFreeInterfaceGeometry(
                model, contact_angle_degrees=90.0
            ).update(state, solid_phi=fluid.solid_phi, compute_curvature=False)

    def test_neutral_sessile_hemisphere_has_valid_wall_curvature(self) -> None:
        shape = (25, 25, 20)
        radius = 6.25
        coordinates = np.indices(shape, dtype=np.float64)
        distance = np.sqrt(
            (coordinates[0] - 12.0) ** 2
            + (coordinates[1] - 12.0) ** 2
            + (coordinates[2] + 0.5) ** 2
        )
        fill = np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0).astype(
            np.float32
        )
        model = HomeLbmModel(
            fluid_grid_res=shape,
            kinematic_viscosity=0.5,
            periodic=(False, False, False),
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)
        geometry = HomeFreeInterfaceGeometry(
            model, contact_angle_degrees=90.0
        )

        diagnostics = geometry.update(
            state, solid_phi=fluid.solid_phi, compute_curvature=True
        )

        self.assertGreater(diagnostics.wetting_cell_count, 0)
        self.assertGreater(diagnostics.curvature_required_cell_count, 0)
        self.assertEqual(diagnostics.insufficient_curvature_neighbor_count, 0)
        self.assertEqual(diagnostics.ill_conditioned_curvature_count, 0)
        self.assertEqual(diagnostics.invalid_curvature_count, 0)


if __name__ == "__main__":
    unittest.main()
