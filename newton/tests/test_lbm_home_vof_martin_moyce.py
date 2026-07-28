# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare HOME-FREE VOF dam-break surge front to Martin & Moyce (1952).

L0-only: bed surge tip, gate time-align, Re/Fr diagnostics. No height-eq / FSI.

    uv run --extra examples python -m unittest \\
        newton.tests.test_lbm_home_vof_martin_moyce -v
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.benchmark.home_vof_dambreak_front import (
    run_home_vof_dambreak_front,
)
from wanphys._src.fluid.fluid_grid.lbm.benchmark.martin_moyce import (
    MARTIN_MOYCE_N2_2,
    MartinMoyceScale,
    align_T_to_anchor,
    compare_front_to_martin,
    interpolate_Z_at_T,
    reference_arrays,
)


def _require_cuda() -> None:
    try:
        wp.init()
        if wp.get_cuda_device_count() <= 0:
            raise RuntimeError("no CUDA")
    except Exception as exc:  # noqa: BLE001
        raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc


class TestMartinMoyceReferenceData(unittest.TestCase):
    def test_n2_2_table_monotonic(self) -> None:
        t, z = reference_arrays(2.0)
        self.assertEqual(len(t), len(MARTIN_MOYCE_N2_2))
        self.assertTrue(np.all(np.diff(t) >= 0.0))
        self.assertTrue(np.all(np.diff(z) > 0.0))
        self.assertAlmostEqual(float(z[0]), 1.0, places=5)

    def test_scale_n2_and_re(self) -> None:
        s = MartinMoyceScale(
            a_cells=12.0, height_cells=24.0, g_lattice=3.5e-4, tau=0.51
        )
        self.assertAlmostEqual(s.n2, 2.0, places=6)
        self.assertGreater(s.T_from_steps(100), 0.0)
        self.assertAlmostEqual(s.Z_from_front_cell(12.0), 1.0, places=6)
        self.assertAlmostEqual(s.nu_lattice, (0.51 - 0.5) / 3.0, places=8)
        self.assertGreater(s.Re_column, 1.0e2)
        d = s.diagnostics()
        self.assertIn("Re_column", d)
        self.assertAlmostEqual(d["tau"], 0.51, places=6)

    def test_align_gate_shifts_to_anchor(self) -> None:
        # Synthetic front that crosses Z=1.44 at T=0.9; align to 1.19.
        t = np.array([0.0, 0.5, 0.9, 1.5, 2.0], dtype=np.float64)
        z = np.array([1.0, 1.2, 1.44, 2.0, 2.5], dtype=np.float64)
        t_a = align_T_to_anchor(t, z, z_anchor=1.44, t_anchor=1.19)
        self.assertAlmostEqual(float(t_a[2]), 1.19, places=5)
        stats = compare_front_to_martin(t, z, n2=2.0, align_gate=True)
        self.assertEqual(stats["aligned"], 1.0)


class TestHomeVofVsMartinMoyce(unittest.TestCase):
    """GPU HOME-FREE front vs Martin–Moyce n²=2 (soft quantitative band)."""

    def test_surge_front_tracks_experiment(self) -> None:
        _require_cuda()
        result = run_home_vof_dambreak_front(
            n=48,
            tau=0.51,
            sample_every=4,
            t_target=2.9,
            align_gate=True,
        )
        scale = result.scale
        stats = result.stats
        self.assertAlmostEqual(scale.n2, 2.0, places=5)
        self.assertGreater(float(stats["n_points"]), 5.0)
        self.assertGreater(float(result.z.max()), 1.8)
        self.assertLess(float(stats["mass_rel"]), 0.02)
        self.assertEqual(float(stats["aligned"]), 1.0)
        # Bed tip + gate align: mid-range typically ~10–18% at n=48 (lags late T).
        self.assertLess(
            float(stats["mean_rel"]),
            0.20,
            msg=(
                f"Martin-Moyce compare: mae={stats['mae']:.3f} "
                f"rmse={stats['rmse']:.3f} mean_rel={stats['mean_rel']:.3f} "
                f"max_abs={stats['max_abs']:.3f} Re~{scale.Re_column:.0f}"
            ),
        )
        self.assertLess(float(stats["rmse"]), 0.50)

    def test_print_comparison_table(self) -> None:
        """Human-readable T / Z_exp / Z_sim table (aligned T)."""
        _require_cuda()
        result = run_home_vof_dambreak_front(
            n=48,
            tau=0.51,
            sample_every=4,
            t_target=2.85,
            align_gate=True,
        )
        t_ref, z_ref = reference_arrays(2.0)
        z_at = interpolate_Z_at_T(result.t_aligned, result.z, t_ref)
        d = result.scale.diagnostics()
        lines = [
            f"Martin-Moyce L0 n^2={d['n2']:.2f}  a={d['a_cells']:.0f}  "
            f"|g|={d['g']:g}  tau={d['tau']:.3f}  Re~{d['Re_column']:.0f}",
            f"{'T':>6} {'Z_exp':>7} {'Z_sim':>7} {'dZ':>7}",
        ]
        for ti, ze, zs in zip(t_ref, z_ref, z_at, strict=True):
            if not np.isfinite(zs) or ti < 0.4 or ti > 2.7:
                continue
            lines.append(f"{ti:6.2f} {ze:7.2f} {zs:7.2f} {zs - ze:7.2f}")
        st = result.stats
        lines.append(
            f"mid-range: mae={st['mae']:.3f} rmse={st['rmse']:.3f} "
            f"mean_rel={st['mean_rel']:.3%}  mass_rel={st['mass_rel']:.3%}"
        )
        print("\n" + "\n".join(lines))
        self.assertTrue(np.isfinite(st["mae"]))
        self.assertGreater(float(st["n_points"]), 3.0)


if __name__ == "__main__":
    unittest.main()
