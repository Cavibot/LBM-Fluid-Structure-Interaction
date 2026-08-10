# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME cut-link boundary and momentum-exchange verification."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_OPPOSITE,
    D3Q27_WEIGHTS,
    CutLinkBuffer,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    linear_bouzidi_reflection,
    reconstruct_populations,
)


def _sphere_geometry(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0) + 0.5
    center = np.asarray(shape, dtype=np.float64) / 2.0
    phi = np.linalg.norm(coordinates - center, axis=-1) - 3.2
    return phi.astype(np.float32), np.where(phi < 0.0, 0, -1).astype(np.int32)


def _upload_geometry(state: HomeLbmState, phi: np.ndarray, body_id: np.ndarray) -> None:
    wp.copy(state.solid_phi, wp.array(phi, dtype=float, device=state.device))
    wp.copy(state.solid_body_id, wp.array(body_id, dtype=wp.int32, device=state.device))


def _fluid_momentum(state: HomeLbmState, fluid: np.ndarray) -> np.ndarray:
    moments = state.moments.numpy().reshape(10, -1)
    return moments[1:4, fluid.reshape(-1)].sum(axis=1, dtype=np.float64)


class TestHomeLbmBoundary(unittest.TestCase):
    def test_linear_bouzidi_reference_is_continuous_at_half_link(self) -> None:
        arguments = {
            "incoming": 0.17,
            "outgoing": 0.11,
            "weight": 1.0 / 18.0,
            "density": 1.03,
            "direction_dot_wall_velocity": 0.02,
        }
        below = linear_bouzidi_reflection(
            support_incoming=0.23, fraction=0.5 - 1.0e-12, **arguments
        )
        at_half = linear_bouzidi_reflection(
            support_incoming=None, fraction=0.5, **arguments
        )
        self.assertAlmostEqual(below, at_half, places=11)

        with self.assertRaisesRegex(ValueError, "second fluid support node"):
            linear_bouzidi_reflection(
                support_incoming=None, fraction=0.25, **arguments
            )

    def test_missing_second_fluid_node_is_a_hard_diagnostic(self) -> None:
        shape = (1, 3, 1)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        solver = HomeLbmSolver(model)
        state_in = HomeLbmState(model)
        state_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(state_in)
        phi = np.asarray([[[-0.75], [0.25], [-0.75]]], dtype=np.float32)
        body_id = np.asarray([[[0], [-1], [1]]], dtype=np.int32)
        _upload_geometry(state_in, phi, body_id)
        solver.set_cut_links(CutLinkBuffer.build_from_sdf(state_in))

        solver.step(state_in, state_out, model.time_step)
        diagnostics = solver.collect_diagnostics(state_out)
        self.assertEqual(diagnostics.unsupported_interpolation_count, 2)
        with self.assertRaisesRegex(RuntimeError, "no second fluid support node"):
            solver.validate_state(state_out)

    def _case(self) -> tuple[HomeLbmSolver, HomeLbmState, HomeLbmState, CutLinkBuffer, np.ndarray]:
        shape = (12, 12, 12)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        solver = HomeLbmSolver(model)
        state_in = HomeLbmState(model)
        state_out = HomeLbmState(model)
        solver.initialize_uniform_lattice(state_in, velocity=(0.025, -0.01, 0.005))
        phi, body_id = _sphere_geometry(shape)
        _upload_geometry(state_in, phi, body_id)
        links = CutLinkBuffer.build_from_sdf(state_in)
        solver.set_cut_links(links)
        return solver, state_in, state_out, links, phi >= 0.0

    def test_static_boundary_conserves_fluid_plus_body_momentum(self) -> None:
        solver, state_in, state_out, links, fluid = self._case()
        before = _fluid_momentum(state_in, fluid)

        solver.step(state_in, state_out, solver.model.time_step)
        wp.synchronize_device("cpu")

        after = _fluid_momentum(state_out, fluid)
        body_impulse = links.impulse.numpy().sum(axis=0, dtype=np.float64)
        np.testing.assert_allclose(after - before + body_impulse, 0.0, atol=2.0e-5)
        self.assertEqual(solver.collect_diagnostics(state_out).missing_cut_link_count, 0)

    def test_moving_boundary_transfers_opposite_impulses(self) -> None:
        solver, state_in, state_out, links, fluid = self._case()
        solver.initialize_uniform_lattice(state_in)
        wall_velocity = np.array([0.03, 0.0, 0.0], dtype=np.float64)
        links.set_uniform_wall_velocity(tuple(wall_velocity))
        before = _fluid_momentum(state_in, fluid)

        solver.step(state_in, state_out, solver.model.time_step)
        wp.synchronize_device("cpu")

        fluid_impulse = _fluid_momentum(state_out, fluid) - before
        body_impulse = links.impulse.numpy().sum(axis=0, dtype=np.float64)
        self.assertGreater(fluid_impulse[0], 0.0)
        self.assertLess(body_impulse[0], 0.0)

        fluid_moments = np.zeros(10, dtype=np.float64)
        fluid_moments[0] = 1.0
        incoming_populations = reconstruct_populations(fluid_moments)
        directions = links.direction.numpy()
        fractions = links.fraction.numpy()
        expected = np.empty((links.link_count, 3), dtype=np.float64)
        for link, (direction, fraction) in enumerate(zip(directions, fractions, strict=True)):
            opposite = D3Q27_OPPOSITE[direction]
            c_out = D3Q27_DIRECTIONS[direction]
            c_in = D3Q27_DIRECTIONS[opposite]
            outgoing = linear_bouzidi_reflection(
                incoming_populations[opposite],
                incoming_populations[direction],
                incoming_populations[opposite],
                float(fraction),
                D3Q27_WEIGHTS[direction],
                1.0,
                float(np.dot(c_out, wall_velocity)),
            )
            expected[link] = incoming_populations[opposite] * (c_in - wall_velocity)
            expected[link] -= outgoing * (c_out - wall_velocity)
        np.testing.assert_allclose(links.impulse.numpy(), expected, atol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
