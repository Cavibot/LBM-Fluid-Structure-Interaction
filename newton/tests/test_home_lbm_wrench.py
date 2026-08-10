# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduction and unit-conversion tests for per-body HOME fluid loads."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import FluidWrenchBuffer, LatticeScaling


class TestHomeLbmWrench(unittest.TestCase):
    def test_link_impulses_reduce_to_exact_body_wrenches(self) -> None:
        body_id = np.array([0, 0, 1], dtype=np.int32)
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        impulses = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        links = SimpleNamespace(
            device=wp.get_device("cpu"),
            link_count=3,
            body_id=wp.array(body_id, dtype=wp.int32, device="cpu"),
            intersection_lattice=wp.array(points, dtype=wp.vec3, device="cpu"),
            impulse=wp.array(impulses, dtype=wp.vec3, device="cpu"),
        )
        centers = wp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=wp.vec3, device="cpu")
        wrench = FluidWrenchBuffer(2, device="cpu")

        wrench.accumulate(links, centers)
        force, torque = wrench.lattice_numpy()

        np.testing.assert_allclose(force, [[3.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
        np.testing.assert_allclose(torque, [[0.0, 0.0, -1.0], [0.0, -4.0, 0.0]])

    def test_physical_conversion_uses_distinct_force_and_torque_units(self) -> None:
        wrench = FluidWrenchBuffer(1, device="cpu")
        wp.copy(wrench.force_lattice, wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"))
        wp.copy(wrench.torque_lattice, wp.array([[3.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"))
        scaling = LatticeScaling(cell_size=0.25, time_step=0.5, reference_density=4.0)

        force, torque = wrench.physical_numpy(scaling)

        np.testing.assert_allclose(force[0, 0], 2.0 * scaling.force_unit)
        np.testing.assert_allclose(torque[0, 0], 3.0 * scaling.torque_unit)
        self.assertAlmostEqual(scaling.momentum_unit, scaling.force_unit * scaling.time_step)
        self.assertAlmostEqual(
            scaling.angular_momentum_unit,
            scaling.torque_unit * scaling.time_step,
        )

    def test_rigid_force_writeback_adds_without_clearing_existing_loads(self) -> None:
        wrench = FluidWrenchBuffer(1, device="cpu")
        wp.copy(wrench.force_lattice, wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu"))
        wp.copy(wrench.torque_lattice, wp.array([[0.0, 3.0, 0.0]], dtype=wp.vec3, device="cpu"))
        body_f = wp.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            dtype=wp.spatial_vector,
            device="cpu",
        )
        scaling = LatticeScaling(cell_size=0.25, time_step=0.5, reference_density=4.0)

        wrench.add_to_rigid_forces(body_f, scaling)
        wp.synchronize_device("cpu")

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected[0] += 2.0 * scaling.force_unit
        expected[4] += 3.0 * scaling.torque_unit
        np.testing.assert_allclose(body_f.numpy()[0], expected)


if __name__ == "__main__":
    unittest.main()
