# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp/reference agreement for HOME-Free PLIC geometry."""

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
    fit_plic_curvature,
    plic_fsl_link_coverage,
    plane_interface_area,
    plic_plane_offset,
    plic_pull_link_intersection,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof import geometry_kernels
from wanphys._src.fluid.fluid_grid.home_lbm.constants import D3Q27_DIRECTIONS


class TestHomeFreeGeometry(unittest.TestCase):
    def _pull_link_intersections(self, device: str) -> None:
        rng = np.random.default_rng(32452843)
        count = 512
        normals = rng.normal(size=(count, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        directions = rng.choice((-1.0, 0.0, 1.0), size=(count, 3)).astype(
            np.float32
        )
        zero = np.linalg.norm(directions, axis=1) == 0.0
        directions[zero, 0] = -1.0
        facing = -np.sum(normals * directions, axis=1)
        directions[facing <= 0.0] *= -1.0
        max_component = np.max(np.abs(directions), axis=1)
        fractions = rng.uniform(0.0, 0.49, size=count) / max_component
        points = -fractions[:, None] * directions
        offsets = np.sum(normals * points, axis=1).astype(np.float32)

        expected = [
            plic_pull_link_intersection(
                offset, normal, direction, tolerance=1.0e-6
            )
            for offset, normal, direction in zip(
                offsets, normals, directions, strict=True
            )
        ]
        device_offsets = wp.array(offsets, dtype=float, device=device)
        device_normals = wp.array(normals, dtype=wp.vec3, device=device)
        device_directions = wp.array(directions, dtype=wp.vec3, device=device)
        actual_fraction = wp.zeros(count, dtype=float, device=device)
        actual_point = wp.zeros(count, dtype=wp.vec3, device=device)
        actual_distance = wp.zeros(count, dtype=float, device=device)
        status = wp.zeros(count, dtype=wp.int32, device=device)

        wp.launch(
            geometry_kernels.plic_pull_link_intersections_kernel,
            dim=count,
            inputs=[
                device_offsets,
                device_normals,
                device_directions,
                actual_fraction,
                actual_point,
                actual_distance,
                status,
                1.0e-6,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(status.numpy(), 1)
        np.testing.assert_allclose(
            actual_fraction.numpy(),
            [value.fraction for value in expected],
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            actual_point.numpy(),
            [value.point for value in expected],
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            actual_distance.numpy(),
            [value.distance for value in expected],
            rtol=2.0e-5,
            atol=2.0e-6,
        )

        invalid_offsets = np.asarray([0.2, -0.2, 0.75], dtype=np.float32)
        invalid_normals = np.asarray([[1.0, 0.0, 0.0]] * 3, dtype=np.float32)
        invalid_directions = np.asarray(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        invalid_status = wp.zeros(3, dtype=wp.int32, device=device)
        wp.launch(
            geometry_kernels.plic_pull_link_intersections_kernel,
            dim=3,
            inputs=[
                wp.array(invalid_offsets, dtype=float, device=device),
                wp.array(invalid_normals, dtype=wp.vec3, device=device),
                wp.array(invalid_directions, dtype=wp.vec3, device=device),
                wp.zeros(3, dtype=float, device=device),
                wp.zeros(3, dtype=wp.vec3, device=device),
                wp.zeros(3, dtype=float, device=device),
                invalid_status,
                1.0e-6,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        np.testing.assert_array_equal(invalid_status.numpy(), (-2, -3, -4))

    def _random_offsets(self, device: str) -> None:
        rng = np.random.default_rng(2718281)
        normals = rng.normal(size=(512, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals[:8] = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0e-8, -1.0e-8],
                [1.0e-8, 1.0, 1.0e-8],
                [-1.0e-8, 1.0e-8, -1.0],
                [1.2628e-7, -9.55494e-7, -1.0],
                [2.37961e-7, 1.3139e-7, 1.0],
            ],
            dtype=np.float32,
        )
        diagonal = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        diagonal /= np.linalg.norm(diagonal)
        normals[8] = diagonal
        normals[9] = diagonal
        normals[10] = np.asarray(
            [2.09361e-6, -1.41785e-5, 1.0], dtype=np.float32
        )
        normals[11] = np.asarray(
            [-1.04438511e-4, -1.69280913e-4, 0.99999998],
            dtype=np.float32,
        )
        normals[12] = np.asarray(
            [-1.23348085e-4, -1.23362986e-4, 0.999999985],
            dtype=np.float32,
        )
        fill = rng.uniform(1.0e-5, 1.0 - 1.0e-5, size=512).astype(np.float32)
        fill[8] = 0.0
        fill[9] = 1.0
        fill[11] = 0.499798983335495
        fill[12] = 0.4997641146183014
        expected = np.asarray(
            [plic_plane_offset(value, normal) for value, normal in zip(fill, normals, strict=True)]
        )
        expected_area = np.asarray(
            [
                plane_interface_area(offset, normal)
                for offset, normal in zip(expected, normals, strict=True)
            ]
        )
        device_fill = wp.array(fill, dtype=float, device=device)
        device_normals = wp.array(normals, dtype=wp.vec3, device=device)
        actual = wp.zeros(fill.size, dtype=float, device=device)
        actual_area = wp.zeros(fill.size, dtype=float, device=device)

        wp.launch(
            geometry_kernels.plic_offsets_kernel,
            dim=fill.size,
            inputs=[device_fill, device_normals, actual],
            device=device,
        )
        wp.launch(
            geometry_kernels.plic_areas_kernel,
            dim=fill.size,
            inputs=[actual, device_normals, actual_area],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(actual.numpy(), expected, rtol=4.0e-5, atol=4.0e-6)
        np.testing.assert_allclose(
            actual_area.numpy(), expected_area, rtol=2.0e-4, atol=2.0e-5
        )

    def _closed_surface_case(self, device: str) -> None:
        shape = (11, 11, 11)
        center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        radius = 3.25
        transition_width = 2.0
        fill = np.empty(shape, dtype=np.float32)
        for index in np.ndindex(shape):
            distance = float(np.linalg.norm(np.asarray(index, dtype=np.float64) - center))
            fill[index] = np.clip(
                0.5 + (radius - distance) / transition_width, 0.0, 1.0
            )

        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, fill)
        geometry = HomeFreeInterfaceGeometry(model)

        diagnostics = geometry.update(state)

        flags = state.flags.numpy()
        interface = flags == 1
        self.assertEqual(diagnostics.interface_cell_count, int(np.count_nonzero(interface)))
        all_normals = geometry.normal.numpy()
        actual_normal = all_normals[interface].astype(np.float64)
        interface_indices = np.argwhere(interface).astype(np.float64)
        expected_normal = interface_indices - center
        expected_normal /= np.linalg.norm(expected_normal, axis=1)[:, None]
        alignment = np.sum(actual_normal * expected_normal, axis=1)
        self.assertGreater(float(np.min(alignment)), 0.99)
        actual_offset = geometry.plane_offset.numpy()[interface].astype(np.float64)
        fill_interface = fill[interface].astype(np.float64)
        reference_offset = np.asarray(
            [
                plic_plane_offset(value, normal)
                for value, normal in zip(fill_interface, actual_normal, strict=True)
            ]
        )
        np.testing.assert_allclose(actual_offset, reference_offset, rtol=2.0e-5, atol=2.0e-6)
        actual_area = geometry.interface_area.numpy()[interface].astype(np.float64)
        reference_area = np.asarray(
            [
                plane_interface_area(offset, normal)
                for offset, normal in zip(
                    actual_offset, actual_normal, strict=True
                )
            ]
        )
        np.testing.assert_allclose(
            actual_area, reference_area, rtol=2.0e-4, atol=2.0e-5
        )
        np.testing.assert_array_equal(geometry.valid.numpy()[interface], 1)
        curvature_required = geometry.curvature_required.numpy().astype(bool)
        curvature_skipped = interface & ~curvature_required
        actual_curvature = geometry.curvature.numpy()[curvature_required].astype(
            np.float64
        )
        reference_curvature = []
        for index in map(tuple, np.argwhere(curvature_required)):
            stencil = tuple(slice(value - 1, value + 2) for value in index)
            reference_curvature.append(
                fit_plic_curvature(
                    fill[stencil],
                    flags[stencil],
                    all_normals[index],
                    max_condition_number=geometry.max_condition_number,
                ).curvature
            )
        np.testing.assert_allclose(
            actual_curvature,
            np.asarray(reference_curvature),
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        self.assertEqual(
            diagnostics.curvature_required_cell_count,
            int(np.count_nonzero(curvature_required)),
        )
        self.assertEqual(
            diagnostics.curvature_skipped_cell_count,
            int(np.count_nonzero(curvature_skipped)),
        )
        np.testing.assert_array_equal(
            geometry.curvature_valid.numpy()[curvature_required], 1
        )
        np.testing.assert_array_equal(
            geometry.curvature_valid.numpy()[curvature_skipped], 0
        )
        np.testing.assert_array_equal(
            geometry.curvature.numpy()[curvature_skipped], 0.0
        )
        self.assertLess(abs(float(np.mean(actual_curvature)) + 1.0 / radius), 0.03)
        self.assertLess(float(np.max(actual_curvature)), 0.0)

        fsl_coverage = plic_fsl_link_coverage(
            all_normals,
            geometry.plane_offset.numpy(),
            flags,
            periodic=model.periodic,
            tolerance=1.0e-6,
        )
        self.assertGreater(fsl_coverage.total_boundary_links, 0)
        self.assertEqual(
            fsl_coverage.usable_links + fsl_coverage.no_support_links,
            fsl_coverage.total_boundary_links,
        )
        self.assertGreater(fsl_coverage.exact_links, 0)
        self.assertLess(
            fsl_coverage.extrapolated_links,
            0.05 * fsl_coverage.total_boundary_links,
        )
        self.assertLess(fsl_coverage.maximum_extrapolation, 0.02)
        self.assertGreater(fsl_coverage.destination_owned_links, 0)
        self.assertGreater(fsl_coverage.source_owned_links, 0)
        self.assertEqual(fsl_coverage.no_plane_links, 0)
        self.assertGreater(fsl_coverage.no_support_links, 0)
        usable_fraction = fsl_coverage.fraction[
            np.isin(fsl_coverage.status, (1, 2))
        ]
        self.assertGreaterEqual(float(np.min(usable_fraction)), 0.0)
        self.assertLessEqual(float(np.max(usable_fraction)), 1.0)

        stride = int(np.prod(shape))
        device_active = wp.zeros(shape, dtype=wp.int32, device=device)
        invalid_counts = wp.zeros(2, dtype=wp.int32, device=device)
        wp.launch(
            geometry_kernels.classify_fsl_hydrodynamic_nodes_kernel,
            dim=shape,
            inputs=[
                state.flags,
                geometry.normal,
                geometry.plane_offset,
                geometry.valid,
                device_active,
                invalid_counts,
                1.0e-6,
            ],
            device=device,
        )
        device_fraction = wp.zeros(27 * stride, dtype=float, device=device)
        device_extrapolation = wp.zeros(27 * stride, dtype=float, device=device)
        device_status = wp.zeros(27 * stride, dtype=wp.int32, device=device)
        device_owner = wp.zeros(27 * stride, dtype=wp.int32, device=device)
        wp.launch(
            geometry_kernels.construct_fsl_link_coverage_kernel,
            dim=27 * stride,
            inputs=[
                state.flags,
                geometry.normal,
                geometry.plane_offset,
                geometry.valid,
                device_active,
                wp.array(
                    D3Q27_DIRECTIONS.astype(np.float32),
                    dtype=wp.vec3,
                    device=device,
                ),
                device_fraction,
                device_extrapolation,
                device_status,
                device_owner,
                1.0e-6,
                int(model.periodic[0]),
                int(model.periodic[1]),
                int(model.periodic[2]),
                shape[0],
                shape[1],
                shape[2],
            ],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(invalid_counts.numpy(), 0)
        np.testing.assert_array_equal(
            device_active.numpy(), fsl_coverage.hydrodynamic_active
        )
        actual_status = device_status.numpy().reshape(shape + (27,))
        actual_owner = device_owner.numpy().reshape(shape + (27,))
        np.testing.assert_array_equal(actual_status, fsl_coverage.status)
        np.testing.assert_array_equal(actual_owner, fsl_coverage.plane_owner)
        usable = np.isin(fsl_coverage.status, (1, 2))
        np.testing.assert_allclose(
            device_fraction.numpy().reshape(shape + (27,))[usable],
            fsl_coverage.fraction[usable],
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        np.testing.assert_allclose(
            device_extrapolation.numpy().reshape(shape + (27,))[usable],
            fsl_coverage.extrapolation[usable],
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    def test_cpu_matches_closed_surface_reference(self) -> None:
        self._closed_surface_case("cpu")

    def test_cpu_closed_form_offsets_match_random_reference(self) -> None:
        self._random_offsets("cpu")

    def test_cpu_pull_link_intersections_match_reference(self) -> None:
        self._pull_link_intersections("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_matches_closed_surface_reference(self) -> None:
        self._closed_surface_case("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_closed_form_offsets_match_random_reference(self) -> None:
        self._random_offsets("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_pull_link_intersections_match_reference(self) -> None:
        self._pull_link_intersections("cuda:0")

    def test_nonperiodic_boundary_requires_explicit_wetting_geometry(self) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.5
        state.initialize_from_fill_level(fluid, fill, validate_topology=False)

        with self.assertRaisesRegex(RuntimeError, "explicitly configured contact angle"):
            HomeFreeInterfaceGeometry(model).update(state)

    def test_zero_gradient_interface_is_rejected(self) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        fluid = HomeLbmState(model)
        HomeLbmSolver(model).initialize_uniform_lattice(fluid)
        state = HomeFreeState(model)
        state.initialize_from_fill_level(fluid, np.full(shape, 0.5, dtype=np.float32))

        with self.assertRaisesRegex(RuntimeError, "undefined normals"):
            HomeFreeInterfaceGeometry(model).update(state)

    def test_underdetermined_curvature_is_rejected(self) -> None:
        shape = (5, 5, 5)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        state = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:2] = 1.0
        fill[2] = 0.5
        flags = np.zeros(shape, dtype=np.int32)
        flags[:2] = 2
        flags[2, 2, 2] = 1
        flags[2, 1, 2] = 1
        flags[2, 3, 2] = 1
        flags[2, 2, 1] = 1
        flags[2, 2, 3] = 1
        state.fill_level.assign(fill)
        state.flags.assign(flags)

        with self.assertRaisesRegex(RuntimeError, "fewer than five"):
            HomeFreeInterfaceGeometry(model).update(state)


if __name__ == "__main__":
    unittest.main()
