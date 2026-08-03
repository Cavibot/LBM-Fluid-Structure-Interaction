# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""FIX1 bounded-excess and authoritative VOF runtime-profile acceptance."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    LbmDomain,
    LbmModel,
    VofRuntimeProfile,
    collect_vof_diagnostics,
    validate_vof_diagnostics,
)
from wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak import (
    AuthoritativeVofDamBreakConfig,
    AuthoritativeVofDamBreakScene,
    build_authoritative_dam_break_phi,
)


def _model(
    profile: str,
    *,
    shape: tuple[int, int, int] = (6, 3, 5),
    device: str = "cpu",
    interval: int = 60,
) -> LbmModel:
    return LbmModel(
        fluid_grid_res=shape,
        device=device,
        encoding="fullf",
        collision="srt",
        interface_model="vof",
        vof_runtime_profile=profile,
        vof_validation_interval=interval,
        gravity_z=-2.0e-5,
        enforce_population_positivity=False,
    )


def _domain(
    profile: str,
    *,
    shape: tuple[int, int, int] = (6, 3, 5),
    device: str = "cpu",
    interval: int = 60,
) -> LbmDomain:
    domain = LbmDomain(
        _model(
            profile,
            shape=shape,
            device=device,
            interval=interval,
        )
    )
    domain.initialize_vof(build_authoritative_dam_break_phi(shape))
    return domain


class TestVofFix1RuntimeContracts(unittest.TestCase):
    def test_profile_and_sample_interval_contracts(self) -> None:
        default = _model("strict")
        self.assertEqual(
            default.vof_runtime_profile,
            VofRuntimeProfile.STRICT.value,
        )
        self.assertEqual(default.vof_validation_interval, 60)
        for profile in VofRuntimeProfile:
            self.assertEqual(
                _model(profile.value).vof_runtime_profile,
                profile.value,
            )
        with self.assertRaisesRegex(ValueError, "runtime profile"):
            _model("sometimes")
        for interval in (29, 101):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, r"\[30, 100\]"):
                    _model("sampled", interval=interval)

    def test_strict_reports_every_step_and_sampled_reports_on_cadence(
        self,
    ) -> None:
        strict = _domain("strict")
        strict.step(1.0)
        self.assertIsNotNone(strict.solver.last_vof_diagnostics)

        sampled = _domain("sampled", interval=30)
        for _ in range(29):
            sampled.step(1.0)
        self.assertIsNone(sampled.solver.last_vof_diagnostics)
        sampled.step(1.0)
        report = sampled.solver.last_vof_diagnostics
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.epoch, 30)

    def test_device_skips_full_host_stage_checks_and_reads_33_scalars(
        self,
    ) -> None:
        domain = _domain("device")
        validator = (
            "wanphys._src.fluid.fluid_grid.lbm.vof.advection."
            "validate_p2_transport_input"
        )
        with patch(validator, side_effect=AssertionError("host validator used")):
            domain.step(1.0)

        report = domain.solver.last_vof_diagnostics
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.epoch, 1)
        runtime = domain.solver._vof_device_diagnostics
        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(tuple(runtime.metrics.shape), (33,))

        host = collect_vof_diagnostics(domain.state)
        self.assertAlmostEqual(report.total_mass, host.total_mass, places=8)
        self.assertAlmostEqual(report.phi_min, host.phi_min, places=7)
        self.assertAlmostEqual(report.phi_max, host.phi_max, places=7)
        self.assertEqual(
            report.invalid_liquid_gas_adjacency_count,
            host.invalid_liquid_gas_adjacency_count,
        )

    def test_device_reports_and_accepts_zero_receiver_pending_route(self) -> None:
        domain = _domain("device")
        state = domain.state
        assert state.vof is not None
        pending = np.asarray(state.vof.pending_excess.numpy()).copy()
        cell = (4, 1, 2)
        pending[cell] = 0.125
        wp.copy(
            state.vof.pending_excess,
            wp.array(pending, dtype=float, device=state.vof.device),
        )
        state.vof.reference_mass += 0.125
        previous = np.asarray(state.vof.cell_type.numpy()).copy()
        proposed = previous.copy()
        previous[3, 1, 2] = 1
        proposed[3, 1, 2] = 2

        runtime = domain.solver._vof_device_diagnostics
        assert runtime is not None
        report = runtime.collect(
            state,
            previous_cell_type=wp.array(
                previous,
                dtype=wp.uint8,
                device=state.vof.device,
            ),
            proposed_cell_type=wp.array(
                proposed,
                dtype=wp.uint8,
                device=state.vof.device,
            ),
        )
        self.assertEqual(report.pending_zero_receiver_count, 1)
        self.assertEqual(report.pending_receiver_mismatch_count, 0)
        self.assertEqual(report.first_invalid_pending_cell, cell)
        self.assertAlmostEqual(report.first_invalid_pending_excess, 0.125)
        self.assertEqual(
            report.first_invalid_pending_stored_receiver_count,
            0,
        )
        self.assertEqual(
            report.first_invalid_pending_actual_receiver_count,
            0,
        )
        self.assertEqual(report.first_invalid_pending_final_interface_mask, 0)
        self.assertEqual(
            report.first_invalid_pending_previous_interface_mask & 1,
            1,
        )
        self.assertEqual(
            report.first_invalid_pending_proposed_liquid_mask & 1,
            1,
        )
        self.assertEqual(
            report.first_invalid_pending_neighbor_valid_mask,
            (1 << 18) - 1,
        )
        validate_vof_diagnostics(report)

    def test_device_rejects_pending_receiver_count_mismatch(self) -> None:
        domain = _domain("device")
        state = domain.state
        assert state.vof is not None
        source = (2, 1, 2)
        pending = np.asarray(state.vof.pending_excess.numpy()).copy()
        pending[source] = 0.125
        state.vof.pending_excess.assign(pending)
        state.vof.reference_mass += 0.125

        runtime = domain.solver._vof_device_diagnostics
        assert runtime is not None
        report = runtime.collect(state)
        self.assertEqual(report.pending_zero_receiver_count, 0)
        self.assertEqual(report.pending_receiver_mismatch_count, 1)
        self.assertEqual(report.first_invalid_pending_cell, source)
        with self.assertRaisesRegex(
            ValueError,
            r"receiver count does not match topology",
        ):
            validate_vof_diagnostics(report)

    def test_off_keeps_transaction_and_epoch_gates(self) -> None:
        domain = _domain("off")
        validator = (
            "wanphys._src.fluid.fluid_grid.lbm.vof.advection."
            "validate_p2_transport_input"
        )
        with patch(validator, side_effect=AssertionError("host validator used")):
            domain.step(1.0)
        self.assertIsNone(domain.solver.last_vof_diagnostics)

        assert domain.state.vof is not None
        domain.state.vof.geometry_epoch -= 1
        with self.assertRaisesRegex(ValueError, "geometry is stale"):
            domain.step(1.0)


class TestVofFix1LongRunningRegression(unittest.TestCase):
    def test_strong_gravity_crosses_original_failure_with_bounded_phi(
        self,
    ) -> None:
        scene = AuthoritativeVofDamBreakScene(
            AuthoritativeVofDamBreakConfig(
                device="cpu",
                grid_res=(12, 12, 12),
                gravity_z=-2.0e-4,
                runtime_profile="device",
            )
        )
        report = None
        for _ in range(520):
            report = scene.step()
        assert report is not None

        self.assertEqual(report.epoch, 520)
        self.assertGreaterEqual(report.phi_min, 0.0)
        self.assertLessEqual(report.phi_max, 1.0)
        self.assertLessEqual(report.relative_mass_error, 5.0e-6)
        self.assertEqual(report.invalid_pending_excess_count, 0)
        self.assertEqual(report.non_finite_count, 0)
        validate_vof_diagnostics(report)

    @unittest.skipUnless(
        wp.is_cuda_available(),
        "CUDA unavailable: FIX1 device profile remains CUDA_NOT_ACCEPTED",
    )
    def test_cuda_device_profile_compact_smoke(self) -> None:
        domain = _domain("device", device="cuda:0")
        domain.step(1.0)
        report = domain.solver.last_vof_diagnostics
        self.assertIsNotNone(report)
        assert report is not None
        validate_vof_diagnostics(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
