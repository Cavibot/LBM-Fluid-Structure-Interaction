# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeLbmModel,
    HomeLbmState,
    HomeMomentQuantizationAuditor,
    HomeMomentQuantizationSpec,
    quantize_dequantize_home_moments,
)


class TestHomeLbmQuantization(unittest.TestCase):
    def _model(self, device: str) -> HomeLbmModel:
        return HomeLbmModel(
            fluid_grid_res=(3, 2, 2),
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            max_lattice_speed=0.2,
            device=device,
        )

    def test_reference_roundtrip_respects_component_error_budget(self) -> None:
        model = self._model("cpu")
        spec = HomeMomentQuantizationSpec.from_model(model)
        rng = np.random.default_rng(481)
        centered = rng.uniform(-0.95, 0.95, size=(128, 10))
        values = centered * np.asarray(spec.bounds) + np.asarray(spec.offsets)

        reconstructed, diagnostics = quantize_dequantize_home_moments(
            values, spec
        )

        self.assertEqual(diagnostics.saturation_count, 0)
        self.assertEqual(diagnostics.invalid_count, 0)
        error = np.max(np.abs(reconstructed - values), axis=0)
        np.testing.assert_array_less(
            error,
            np.asarray(spec.maximum_roundtrip_error) + 1.0e-15,
        )

    def _run_device_audit(self, device: str) -> None:
        model = self._model(device)
        state = HomeLbmState(model)
        spec = HomeMomentQuantizationSpec.from_model(model)
        values = np.zeros((10, state.cell_count), dtype=np.float32)
        values[0] = model.reference_density
        state.moments.assign(values.reshape(-1))
        auditor = HomeMomentQuantizationAuditor(model, spec)
        diagnostics = auditor.audit(state.moments)
        self.assertEqual(diagnostics.saturation_count, 0)
        self.assertEqual(diagnostics.invalid_count, 0)

        values[1, 0] = np.float32(1.01 * spec.bounds[1])
        state.moments.assign(values.reshape(-1))
        diagnostics = auditor.audit(state.moments, raise_on_saturation=False)
        self.assertEqual(diagnostics.saturation_counts[1], 1)
        with self.assertRaisesRegex(OverflowError, "rho_u_x=1"):
            auditor.audit(state.moments)

    def test_cpu_device_audit_reports_saturation(self) -> None:
        self._run_device_audit("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_device_audit_reports_saturation(self) -> None:
        self._run_device_audit("cuda:0")


if __name__ == "__main__":
    unittest.main()
