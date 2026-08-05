# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 lifecycle and low-cost VOF transaction guards."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmModel, LbmSolver


def _prepared_transaction():
    shape = (6, 3, 5)
    model = LbmModel(
        fluid_grid_res=shape,
        device="cpu",
        encoding="fullf",
        collision="srt",
        interface_model="vof",
        enforce_population_positivity=False,
    )
    solver = LbmSolver(model)
    state_in = solver.create_state()
    state_out = solver.create_state()
    solver.initialize_equilibrium(state_in, 1.0, (0.0, 0.0, 0.0))
    solver.initialize_equilibrium(state_out, 1.0, (0.0, 0.0, 0.0))

    phi = np.zeros(shape, dtype=np.float32)
    phi[: shape[0] // 2] = 1.0
    phi[shape[0] // 2] = 0.5
    vof = solver._vof_solver
    assert vof is not None
    vof.initialize(state_in, phi)
    vof.initialize(state_out, phi)

    logical_f_post = wp.zeros(19 * int(np.prod(shape)), dtype=float, device="cpu")
    streamed = wp.zeros(19 * int(np.prod(shape)), dtype=float, device="cpu")
    return solver, state_in, state_out, vof, logical_f_post, streamed


class TestVofTransactionGuards(unittest.TestCase):
    def test_complete_streaming_binds_pair_without_writing_target(self) -> None:
        _, state_in, state_out, vof, logical, streamed = _prepared_transaction()
        assert state_out.vof is not None
        before = {
            name: np.asarray(getattr(state_out.vof, name).numpy()).copy()
            for name in ("mass", "phi", "pending_excess", "cell_type")
        }

        with (
            patch.object(vof.mass_transport, "compute"),
            patch.object(vof.surface_boundary, "complete_populations"),
        ):
            vof.complete_streaming(state_in, state_out, logical, streamed)

        for name, expected in before.items():
            np.testing.assert_array_equal(
                np.asarray(getattr(state_out.vof, name).numpy()),
                expected,
            )

    def test_duplicate_prepare_and_wrong_finish_target_raise(self) -> None:
        _, state_in, state_out, vof, logical, streamed = _prepared_transaction()
        with (
            patch.object(vof.mass_transport, "compute"),
            patch.object(vof.surface_boundary, "complete_populations"),
        ):
            vof.complete_streaming(state_in, state_out, logical, streamed)
            with self.assertRaisesRegex(RuntimeError, "previous VOF step"):
                vof.complete_streaming(state_in, state_out, logical, streamed)

        wrong_target = state_out.clone()
        with self.assertRaisesRegex(RuntimeError, "target does not match"):
            vof.finish_step(state_in, wrong_target)

    def test_epoch_mismatch_and_finish_failure_keep_fail_stop_marker(self) -> None:
        _, state_in, state_out, vof, logical, streamed = _prepared_transaction()
        with (
            patch.object(vof.mass_transport, "compute"),
            patch.object(vof.surface_boundary, "complete_populations"),
        ):
            vof.complete_streaming(state_in, state_out, logical, streamed)

        assert state_in.vof is not None
        state_in.vof.epoch += 1
        with self.assertRaisesRegex(RuntimeError, "prepared epoch"):
            vof.finish_step(state_in, state_out)

        # Restore the source epoch and force a commit-stage failure.  The
        # prepared marker must remain, so a new prepare cannot overwrite the
        # failed transaction's scratch.
        state_in.vof.epoch -= 1
        with patch.object(
            vof.surface_boundary,
            "restore_gas",
            return_value=None,
        ), patch.object(
            vof.topology_transition,
            "compute",
            side_effect=RuntimeError("forced finish failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced finish failure"):
                vof.finish_step(state_in, state_out)

        with (
            patch.object(vof.mass_transport, "compute"),
            patch.object(vof.surface_boundary, "complete_populations"),
        ):
            with self.assertRaisesRegex(RuntimeError, "previous VOF step"):
                vof.complete_streaming(state_in, state_out, logical, streamed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
