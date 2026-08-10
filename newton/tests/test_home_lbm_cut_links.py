# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse cut-link geometry tests for HOME-LBM."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    CutLinkBuffer,
    HomeLbmModel,
    HomeLbmState,
)


def _upload_geometry(state: HomeLbmState, phi: np.ndarray, body_id: np.ndarray) -> None:
    wp.copy(state.solid_phi, wp.array(phi.astype(np.float32), dtype=float, device=state.device))
    wp.copy(state.solid_body_id, wp.array(body_id.astype(np.int32), dtype=wp.int32, device=state.device))


class TestHomeLbmCutLinks(unittest.TestCase):
    def test_planar_interface_builds_exact_sparse_links(self) -> None:
        shape = (4, 4, 3)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        state = HomeLbmState(model)
        phi = np.full(shape, 0.5, dtype=np.float32)
        phi[:2, :, :] = -0.5
        body_id = np.full(shape, -1, dtype=np.int32)
        body_id[:2, :, :] = 7
        _upload_geometry(state, phi, body_id)

        links = CutLinkBuffer.build_from_sdf(state)
        wp.synchronize_device("cpu")
        expected_link_count = (3 * shape[1] - 2) * (3 * shape[2] - 2)
        self.assertEqual(links.link_count, expected_link_count)
        np.testing.assert_allclose(links.fraction.numpy(), 0.5, atol=1.0e-7)
        np.testing.assert_array_equal(links.body_id.numpy(), 7)
        self.assertLess(links.occupancy, 0.1)

        counts = links.counts.numpy().reshape(shape)
        self.assertEqual(int(counts[2, 0, 0]), 4)
        self.assertEqual(int(counts[2, 1, 0]), 6)
        self.assertEqual(int(counts[2, 1, 1]), 9)
        self.assertEqual(int(counts[:2, :, :].sum()), 0)
        self.assertEqual(int(counts[3, :, :].sum()), 0)

    def test_sphere_links_are_fluid_to_solid_and_intersections_are_bounded(self) -> None:
        shape = (12, 12, 12)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        state = HomeLbmState(model)
        coordinates = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0) + 0.5
        center = np.asarray(shape, dtype=np.float64) / 2.0
        phi = np.linalg.norm(coordinates - center, axis=-1) - 3.2
        body_id = np.where(phi < 0.0, 3, -1).astype(np.int32)
        _upload_geometry(state, phi, body_id)

        links = CutLinkBuffer.build_from_sdf(state)
        wp.synchronize_device("cpu")
        self.assertGreater(links.link_count, 0)
        self.assertLess(links.occupancy, 0.05)
        self.assertTrue(np.all((links.fraction.numpy() >= 0.0) & (links.fraction.numpy() <= 1.0)))
        np.testing.assert_array_equal(links.body_id.numpy(), 3)

        cells = links.cell.numpy()
        directions = links.direction.numpy()
        for cell, direction in zip(cells, directions, strict=True):
            i = int(cell) // (shape[1] * shape[2])
            remainder = int(cell) % (shape[1] * shape[2])
            j = remainder // shape[2]
            k = remainder % shape[2]
            c = D3Q27_DIRECTIONS[int(direction)].astype(np.int32)
            source = (i - c[0], j - c[1], k - c[2])
            self.assertGreaterEqual(phi[i, j, k], 0.0)
            self.assertLess(phi[source], 0.0)

    def test_unchanged_mask_updates_link_geometry_in_place(self) -> None:
        shape = (12, 12, 12)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        state = HomeLbmState(model)
        coordinates = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0) + 0.5
        center = np.asarray(shape, dtype=np.float64) / 2.0
        old_phi = np.linalg.norm(coordinates - center, axis=-1) - 3.2
        body_id = np.where(old_phi < 0.0, 2, -1).astype(np.int32)
        _upload_geometry(state, old_phi, body_id)
        links = CutLinkBuffer.build_from_sdf(state)
        old_fraction = links.fraction.numpy().copy()
        old_ptrs = (
            links.cell.ptr,
            links.direction.ptr,
            links.fraction.ptr,
            links.body_id.ptr,
            links.intersection_lattice.ptr,
        )

        new_phi = old_phi + np.where(old_phi < 0.0, -0.03, 0.03)
        self.assertTrue(np.array_equal(new_phi < 0.0, old_phi < 0.0))
        _upload_geometry(state, new_phi, body_id)
        links.update_from_sdf(state)

        self.assertEqual(
            old_ptrs,
            (
                links.cell.ptr,
                links.direction.ptr,
                links.fraction.ptr,
                links.body_id.ptr,
                links.intersection_lattice.ptr,
            ),
        )
        self.assertGreater(float(np.max(np.abs(links.fraction.numpy() - old_fraction))), 0.0)

    def test_rigid_wall_velocity_is_evaluated_at_each_true_intersection(self) -> None:
        shape = (12, 12, 12)
        cell_size = 0.1
        time_step = 0.02
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=cell_size,
            time_step=time_step,
            kinematic_viscosity=0.1,
            device="cpu",
        )
        state = HomeLbmState(model)
        coordinates = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0) + 0.5
        center_lattice = np.asarray(shape, dtype=np.float64) / 2.0
        phi = np.linalg.norm(coordinates - center_lattice, axis=-1) - 3.2
        body_id = np.where(phi < 0.0, 0, -1).astype(np.int32)
        _upload_geometry(state, phi, body_id)
        links = CutLinkBuffer.build_from_sdf(state)

        center_world = center_lattice * cell_size
        linear = np.array([0.2, -0.1, 0.05])
        angular = np.array([0.0, 0.0, 2.0])
        body_q = wp.array(
            [wp.transform(wp.vec3(*center_world), wp.quat_identity())],
            dtype=wp.transform,
            device="cpu",
        )
        body_qd = wp.array(
            [[*linear, *angular]], dtype=wp.spatial_vector, device="cpu"
        )
        body_com = wp.zeros(1, dtype=wp.vec3, device="cpu")

        links.set_rigid_wall_velocity(body_q, body_qd, body_com, cell_size, time_step)
        links.validate_body_ids()

        intersections_world = links.intersection_lattice.numpy() * cell_size
        expected_world = linear + np.cross(angular, intersections_world - center_world)
        expected_lattice = expected_world * time_step / cell_size
        np.testing.assert_allclose(links.wall_velocity.numpy(), expected_lattice, atol=2.0e-7)

if __name__ == "__main__":
    unittest.main()
