"""Multi-step multiphase stability regressions for the unified LBM pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from scripts.diag.lbm_dambreak_stability import LEGAL_PATHS, run_case
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ
from wanphys._src.fluid.fluid_grid.lbm.encoding import enforce_population_admissibility_kernel


class TestLbmPopulationAdmissibility(unittest.TestCase):
    def test_limiter_removes_negative_populations_and_preserves_moments(self) -> None:
        populations = np.asarray(
            [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12,
            dtype=np.float32,
        )
        # Symmetric perturbation: keep density and momentum unchanged while
        # making the +/-x populations inadmissible.
        populations[0] += 0.14
        populations[1] -= 0.07
        populations[2] -= 0.07
        self.assertLess(float(populations.min()), 0.0)

        directions = np.asarray(tuple(zip(CX, CY, CZ)), dtype=np.float64)
        density_before = float(np.sum(populations, dtype=np.float64))
        momentum_before = directions.T @ populations.astype(np.float64)
        device_populations = wp.array(populations, dtype=float, device="cpu")
        wp.launch(
            enforce_population_admissibility_kernel,
            dim=(1, 1, 1),
            inputs=[device_populations, 1.0e-9, 0.4, 1, 1, 1],
            device="cpu",
        )
        actual = device_populations.numpy()
        density_after = float(np.sum(actual, dtype=np.float64))
        momentum_after = directions.T @ actual.astype(np.float64)

        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertAlmostEqual(density_after, density_before, places=6)
        np.testing.assert_allclose(momentum_after, momentum_before, atol=2.0e-7, rtol=0.0)

    def test_all_legal_paths_remain_finite_for_multistep_dambreak(self) -> None:
        failures: list[str] = []
        for encoding, collision in LEGAL_PATHS:
            result = run_case(
                encoding,
                collision,
                resolution=12,
                steps=90,
                sample_every=5,
                device="cpu",
            )
            if not result.passed:
                failures.append(
                    f"{encoding}+{collision}: step={result.first_bad_step} "
                    f"reason={result.reason} final={result.final}"
                )
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
