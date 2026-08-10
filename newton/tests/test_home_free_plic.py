# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometric invariants for the PLIC reference implementation."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_lbm import (
    fit_plic_curvature,
    PlicLinkPlaneOwner,
    PlicPullLinkStatus,
    plane_interface_area,
    plane_volume_fraction,
    plic_plane_offset,
    plic_pull_link_intersection,
    plic_pull_link_coverage,
    plic_fsl_link_coverage,
    youngs_interface_normal,
)


class TestHomeFreePlic(unittest.TestCase):
    @staticmethod
    def _quadratic_stencil(
        coefficients: tuple[float, float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b, c, h, i = coefficients
        fill = np.empty((3, 3, 3), dtype=np.float64)
        flags = np.zeros((3, 3, 3), dtype=np.int32)
        normal = np.asarray([0.0, 0.0, 1.0])
        for index in np.ndindex(fill.shape):
            x, y, z = np.asarray(index, dtype=np.float64) - 1.0
            height = a * x * x + b * y * y + c * x * y + h * x + i * y
            offset = height - z
            fill[index] = plane_volume_fraction(offset, normal)
            if 0.0 < fill[index] < 1.0:
                flags[index] = 1
            elif fill[index] == 1.0:
                flags[index] = 2
        return fill, flags

    def test_axis_aligned_cut_has_exact_volume(self) -> None:
        normal = np.asarray([2.0, 0.0, 0.0])
        self.assertAlmostEqual(plane_volume_fraction(-0.5, normal), 0.25, places=14)
        self.assertAlmostEqual(plic_plane_offset(0.25, normal), -0.5, places=14)

    def test_offset_inverts_volume_for_random_orientations(self) -> None:
        rng = np.random.default_rng(1618033)
        for _ in range(100):
            normal = rng.normal(size=3)
            fill = float(rng.uniform(1.0e-6, 1.0 - 1.0e-6))
            offset = plic_plane_offset(fill, normal)
            self.assertAlmostEqual(
                plane_volume_fraction(offset, normal), fill, delta=2.0e-13
            )

    def test_scaling_and_complement_symmetries_hold(self) -> None:
        normal = np.asarray([0.2, -0.7, 0.5])
        for fill in (0.01, 0.2, 0.5, 0.83, 0.99):
            offset = plic_plane_offset(fill, normal)
            self.assertAlmostEqual(
                plic_plane_offset(fill, 3.5 * normal), 3.5 * offset, places=13
            )
            self.assertAlmostEqual(
                plic_plane_offset(1.0 - fill, normal), -offset, places=13
            )
            self.assertAlmostEqual(
                plane_interface_area(offset, normal),
                plane_interface_area(-offset, normal),
                places=12,
            )

    def test_plane_section_area_matches_exact_axis_and_diagonal_cuts(self) -> None:
        self.assertAlmostEqual(
            plane_interface_area(0.23, np.asarray([1.0, 0.0, 0.0])),
            1.0,
            places=14,
        )
        normal = np.asarray([1.0, 1.0, 1.0])
        self.assertAlmostEqual(
            plane_interface_area(0.0, normal),
            3.0 * np.sqrt(3.0) / 4.0,
            places=14,
        )

    def test_degenerate_diagonal_endpoint_section_has_zero_area(self) -> None:
        normal = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        offset = plic_plane_offset(0.0, normal)

        self.assertEqual(plane_interface_area(offset, normal), 0.0)

    def test_plane_section_area_is_the_clipped_volume_derivative(self) -> None:
        rng = np.random.default_rng(57721)
        # The clipped-volume inclusion-exclusion polynomial loses precision
        # below this scale; 3e-6 is in its centered-difference convergence
        # regime for the seeded sections below.
        epsilon = 3.0e-6
        for _ in range(100):
            normal = rng.normal(size=3)
            fill = float(rng.uniform(0.02, 0.98))
            offset = plic_plane_offset(fill, normal)
            derivative = (
                plane_volume_fraction(offset + epsilon, normal)
                - plane_volume_fraction(offset - epsilon, normal)
            ) / (2.0 * epsilon)
            expected = np.linalg.norm(normal) * derivative
            self.assertAlmostEqual(
                plane_interface_area(offset, normal), expected, delta=1.0e-8
            )

    def test_axis_aligned_pull_link_recovers_plic_boundary_location(self) -> None:
        normal = np.asarray([1.0, 0.0, 0.0])
        offset = plic_plane_offset(0.75, normal)

        intersection = plic_pull_link_intersection(
            offset,
            normal,
            np.asarray([-1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(intersection.fraction, 0.25, places=14)
        np.testing.assert_allclose(intersection.point, (0.25, 0.0, 0.0), atol=0.0)
        self.assertAlmostEqual(intersection.distance, 0.25, places=14)

    def test_pull_link_fraction_respects_plane_and_direction_scaling(self) -> None:
        normal = np.asarray([0.6, -0.2, 0.7], dtype=np.float64)
        offset = plic_plane_offset(0.67, normal)
        direction = np.asarray([-1.0, 1.0, -1.0], dtype=np.float64)
        baseline = plic_pull_link_intersection(offset, normal, direction)

        scaled_plane = plic_pull_link_intersection(
            3.25 * offset, 3.25 * normal, direction
        )
        scaled_direction = plic_pull_link_intersection(
            offset, normal, 2.0 * direction
        )

        self.assertAlmostEqual(scaled_plane.fraction, baseline.fraction, places=14)
        np.testing.assert_allclose(scaled_plane.point, baseline.point, atol=2.0e-15)
        self.assertAlmostEqual(
            scaled_direction.fraction, 0.5 * baseline.fraction, places=14
        )
        np.testing.assert_allclose(scaled_direction.point, baseline.point, atol=2.0e-15)
        self.assertAlmostEqual(
            scaled_direction.distance, baseline.distance, places=14
        )

    def test_pull_link_allows_interface_through_lattice_node(self) -> None:
        intersection = plic_pull_link_intersection(
            0.0,
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([-1.0, 0.0, 0.0]),
        )

        self.assertEqual(intersection.fraction, 0.0)
        self.assertEqual(intersection.point, (0.0, 0.0, 0.0))
        self.assertEqual(intersection.distance, 0.0)

    def test_pull_link_rejects_nonintersecting_plic_rays(self) -> None:
        normal = np.asarray([1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "does not face"):
            plic_pull_link_intersection(
                0.2, normal, np.asarray([1.0, 0.0, 0.0])
            )
        with self.assertRaisesRegex(ValueError, "does not face"):
            plic_pull_link_intersection(
                0.2, normal, np.asarray([0.0, 1.0, 0.0])
            )
        with self.assertRaisesRegex(ValueError, "behind"):
            plic_pull_link_intersection(
                -0.2, normal, np.asarray([-1.0, 0.0, 0.0])
            )

        diagonal = np.asarray([1.0, 1.0, 0.0])
        far_offset = plic_plane_offset(0.9, diagonal)
        with self.assertRaisesRegex(ValueError, "outside"):
            plic_pull_link_intersection(
                far_offset, diagonal, np.asarray([-1.0, 0.0, 0.0])
            )

    def test_planar_om_gas_links_have_complete_plic_coverage(self) -> None:
        shape = (7, 4, 3)
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 1
        flags[1:3] = 2
        flags[3] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[0, ..., 0] = -1.0
        normals[3, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[0] = plic_plane_offset(0.75, np.asarray([-1.0, 0.0, 0.0]))
        offsets[3] = plic_plane_offset(0.75, np.asarray([1.0, 0.0, 0.0]))

        coverage = plic_pull_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(True, True, True),
        )

        self.assertEqual(coverage.total_gas_links, 2 * shape[1] * shape[2] * 9)
        self.assertEqual(coverage.valid_links, coverage.total_gas_links)
        self.assertEqual(coverage.not_facing_links, 0)
        self.assertEqual(coverage.behind_links, 0)
        self.assertEqual(coverage.outside_links, 0)
        self.assertEqual(coverage.valid_fraction, 1.0)
        np.testing.assert_allclose(
            coverage.fraction[coverage.status == int(PlicPullLinkStatus.VALID)],
            0.25,
            atol=2.0e-15,
        )

    def test_underfilled_interface_reports_behind_node_gas_links(self) -> None:
        shape = (4, 3, 2)
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 2
        flags[1] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[1, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[1] = plic_plane_offset(0.25, np.asarray([1.0, 0.0, 0.0]))

        coverage = plic_pull_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(False, True, True),
        )

        self.assertEqual(coverage.total_gas_links, shape[1] * shape[2] * 9)
        self.assertEqual(coverage.valid_links, 0)
        self.assertEqual(coverage.behind_links, coverage.total_gas_links)
        self.assertEqual(coverage.valid_fraction, 0.0)

    def test_fsl_uses_destination_plane_for_overfilled_interface(self) -> None:
        shape = (4, 3, 2)
        flags = np.zeros(shape, dtype=np.int32)
        flags[0] = 2
        flags[1] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[1, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[1] = plic_plane_offset(0.75, np.asarray([1.0, 0.0, 0.0]))

        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(False, True, True),
        )

        self.assertEqual(coverage.total_boundary_links, shape[1] * shape[2] * 9)
        self.assertEqual(coverage.usable_links, coverage.total_boundary_links)
        self.assertEqual(coverage.exact_links, coverage.total_boundary_links)
        self.assertEqual(coverage.extrapolated_links, 0)
        self.assertEqual(
            coverage.destination_owned_links, coverage.total_boundary_links
        )
        self.assertEqual(coverage.source_owned_links, 0)
        np.testing.assert_allclose(
            coverage.fraction[coverage.status == int(PlicPullLinkStatus.VALID)],
            0.25,
            atol=2.0e-15,
        )

    def test_fsl_uses_source_plane_for_underfilled_interface(self) -> None:
        shape = (5, 3, 2)
        flags = np.zeros(shape, dtype=np.int32)
        flags[:2] = 2
        flags[2] = 1
        normals = np.zeros(shape + (3,), dtype=np.float64)
        normals[2, ..., 0] = 1.0
        offsets = np.zeros(shape, dtype=np.float64)
        offsets[2] = plic_plane_offset(0.25, np.asarray([1.0, 0.0, 0.0]))

        coverage = plic_fsl_link_coverage(
            normals,
            offsets,
            flags,
            periodic=(False, True, True),
        )

        self.assertEqual(coverage.total_boundary_links, shape[1] * shape[2] * 9)
        self.assertEqual(coverage.usable_links, coverage.total_boundary_links)
        self.assertEqual(coverage.exact_links, coverage.total_boundary_links)
        self.assertEqual(coverage.extrapolated_links, 0)
        self.assertEqual(coverage.destination_owned_links, 0)
        self.assertEqual(coverage.source_owned_links, coverage.total_boundary_links)
        self.assertTrue(np.all(coverage.hydrodynamic_active[:2]))
        self.assertTrue(np.all(~coverage.hydrodynamic_active[2]))
        np.testing.assert_allclose(
            coverage.fraction[coverage.status == int(PlicPullLinkStatus.VALID)],
            0.75,
            atol=2.0e-15,
        )

    def test_youngs_normal_recovers_planar_interface_orientation(self) -> None:
        expected = np.asarray([0.31, -0.47, 0.826], dtype=np.float64)
        expected /= np.linalg.norm(expected)
        global_offset = 0.08
        fill = np.empty((3, 3, 3), dtype=np.float64)
        for index in np.ndindex(fill.shape):
            center = np.asarray(index, dtype=np.float64) - 1.0
            local_offset = global_offset - float(np.dot(expected, center))
            fill[index] = plane_volume_fraction(local_offset, expected)

        actual = youngs_interface_normal(fill)

        self.assertGreater(float(np.dot(actual, expected)), 0.995)
        self.assertAlmostEqual(float(np.linalg.norm(actual)), 1.0, places=14)

    def test_undefined_normal_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "undefined"):
            youngs_interface_normal(np.full((3, 3, 3), 0.5))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            plic_plane_offset(0.5, np.zeros(3))

    def test_monge_fit_recovers_quadratic_curvature(self) -> None:
        coefficients = (-0.08, -0.05, 0.025, 0.03, -0.02)
        fill, flags = self._quadratic_stencil(coefficients)

        fit = fit_plic_curvature(fill, flags, np.asarray([0.0, 0.0, 1.0]))

        a, b, c, h, i = coefficients
        expected = (
            a * (1.0 + i * i) + b * (1.0 + h * h) - c * h * i
        ) / (1.0 + h * h + i * i) ** 1.5
        np.testing.assert_allclose(fit.coefficients, coefficients, atol=2.0e-15)
        self.assertAlmostEqual(fit.curvature, expected, places=14)
        self.assertEqual(fit.rank, 5)
        self.assertGreaterEqual(fit.sample_count, 5)

    def test_monge_fit_rejects_rank_deficient_neighbors(self) -> None:
        fill = np.zeros((3, 3, 3), dtype=np.float64)
        flags = np.zeros((3, 3, 3), dtype=np.int32)
        fill[:, 1, 1] = 0.5
        flags[:, 1, 1] = 1

        with self.assertRaisesRegex(ValueError, "at least 5|rank deficient"):
            fit_plic_curvature(fill, flags, np.asarray([0.0, 0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
