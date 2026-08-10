# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified Newton shape-query rasterization tests for HOME-LBM."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    CutLinkBuffer,
    HomeLbmModel,
    HomeLbmRigidGeometry,
    HomeLbmState,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder


class TestHomeLbmRigidGeometry(unittest.TestCase):
    def test_shape_query_rasterizes_physical_sdf_and_builds_cut_links(self) -> None:
        shape = (12, 12, 12)
        cell_size = 0.1
        center = np.array([0.6, 0.6, 0.6])
        radius = 0.26
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=cell_size,
            device="cpu",
        )
        state = HomeLbmState(model)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=tuple(center), label="home_sphere")
        builder.add_shape_sphere(body, radius=radius)
        rigid = RigidDomain(builder.finalize(device="cpu"))
        rigid.create_state()
        geometry = HomeLbmRigidGeometry.from_domain(rigid)

        geometry.rasterize(state, rigid.state.body_q)
        wp.synchronize_device("cpu")

        phi = state.solid_phi.numpy()
        body_id = state.solid_body_id.numpy()
        coordinates = np.indices(shape).transpose(1, 2, 3, 0)
        points = (coordinates + 0.5) * cell_size
        expected = np.linalg.norm(points - center, axis=-1) - radius
        np.testing.assert_allclose(phi, expected, atol=2.0e-6)
        np.testing.assert_array_equal(body_id[phi < 0.0], body)
        np.testing.assert_array_equal(body_id[phi >= 0.0], -1)

        links = CutLinkBuffer.build_from_sdf(state)
        self.assertGreater(links.link_count, 0)
        np.testing.assert_array_equal(links.body_id.numpy(), body)

        radius_bound = geometry.update_radius_bounds(rigid.model.body_com)
        wp.synchronize_device("cpu")
        self.assertGreaterEqual(float(radius_bound.numpy()[body]), radius)
        self.assertLess(float(radius_bound.numpy()[body]), radius * 1.75)

    def test_incremental_update_matches_full_mask_and_surface_band(self) -> None:
        shape = (24, 24, 24)
        cell_size = 0.1
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        center = np.array([1.0, 1.2, 1.2])
        moved_center = center + np.array([0.04, 0.0, 0.0])
        radius = 0.22
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=cell_size,
            device=device,
        )
        incremental_state = HomeLbmState(model)
        reference_state = HomeLbmState(model)

        builder = RigidModelBuilder(gravity=0.0)
        body = builder.add_body(position=tuple(center), label="incremental_sphere")
        builder.add_shape_sphere(body, radius=radius)
        rigid = RigidDomain(builder.finalize(device=device))
        rigid.create_state()

        geometry = HomeLbmRigidGeometry.from_domain(rigid)
        geometry.update_radius_bounds(rigid.model.body_com)
        geometry.rasterize(incremental_state, rigid.state.body_q)
        moved_q = wp.array(
            [wp.transform(wp.vec3(*moved_center), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        update = geometry.rasterize_incremental(incremental_state, moved_q)

        reference = HomeLbmRigidGeometry.from_domain(rigid)
        reference.rasterize(reference_state, moved_q)
        wp.synchronize_device(device)

        incremental_phi = incremental_state.solid_phi.numpy()
        reference_phi = reference_state.solid_phi.numpy()
        np.testing.assert_array_equal(incremental_phi < 0.0, reference_phi < 0.0)
        surface_band = np.abs(reference_phi) <= 2.0 * cell_size
        np.testing.assert_allclose(
            incremental_phi[surface_band], reference_phi[surface_band], atol=2.0e-6
        )
        np.testing.assert_array_equal(
            incremental_state.solid_body_id.numpy()[reference_phi < 0.0], body
        )
        self.assertEqual(update.mode, "incremental")
        self.assertLess(update.launched_cell_capacity, update.full_grid_cell_count)
        self.assertLess(update.capacity_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
