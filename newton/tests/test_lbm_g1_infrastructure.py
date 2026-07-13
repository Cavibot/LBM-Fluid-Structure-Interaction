# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LBM G1 infrastructure (lattice, stats, benchmark registry)."""

from __future__ import annotations

import unittest

from wanphys._src.fluid.fluid_grid.lbm.benchmark.metrics import (
    perf_metrics_from_step_stats,
)
from wanphys._src.fluid.fluid_grid.lbm.benchmark.registry import get_variant, list_variants
from wanphys._src.fluid.fluid_grid.lbm.core.lattice import (
    D3Q19,
    D3Q27,
    NUM_DIRS,
    get_lattice_spec,
)
from wanphys._src.fluid.fluid_grid.lbm.core.pipeline import LbmStepControl, StepStats
from wanphys._src.fluid.fluid_grid.lbm.model import LbmModel
from wanphys._src.fluid.fluid_grid.lbm.phases.shan_chen import ShanChenPhase


class TestLbmG1Infrastructure(unittest.TestCase):
    def test_d3q19_lattice_spec(self) -> None:
        spec = get_lattice_spec("D3Q19")
        self.assertEqual(spec.name, "D3Q19")
        self.assertEqual(spec.num_dirs, 19)
        self.assertEqual(NUM_DIRS, 19)
        self.assertEqual(len(spec.weights), 19)
        self.assertAlmostEqual(sum(spec.weights), 1.0, places=6)

    def test_d3q27_lattice_spec(self) -> None:
        spec = get_lattice_spec("D3Q27")
        self.assertEqual(spec.name, "D3Q27")
        self.assertEqual(spec.num_dirs, 27)
        self.assertEqual(len(spec.weights), 27)
        self.assertAlmostEqual(sum(spec.weights), 1.0, places=6)
        self.assertEqual(spec.cx[:19], D3Q19.cx)
        self.assertEqual(spec.cy[:19], D3Q19.cy)
        self.assertEqual(spec.cz[:19], D3Q19.cz)
        for i, opp in enumerate(spec.opposite):
            self.assertEqual(spec.opposite[opp], i)
            self.assertEqual(spec.cx[i] + spec.cx[opp], 0)
            self.assertEqual(spec.cy[i] + spec.cy[opp], 0)
            self.assertEqual(spec.cz[i] + spec.cz[opp], 0)
        self.assertIs(get_lattice_spec("d3q27"), D3Q27)

    def test_model_resolves_lattice(self) -> None:
        model = LbmModel(fluid_grid_res=(8, 8, 8), fluid_grid_cell_size=0.1)
        self.assertEqual(model.lattice_spec.name, D3Q19.name)
        self.assertEqual(model.num_dirs, 19)
        model27 = LbmModel(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.1,
            lattice="D3Q27",
        )
        self.assertEqual(model27.num_dirs, 27)

    def test_d3q27_rejects_shan_chen_only(self) -> None:
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(8, 8, 8),
                fluid_grid_cell_size=0.1,
                lattice="D3Q27",
                phase_mode="shan_chen",
                G=-5.0,
            )
        # Zou-He / outflow are supported on D3Q27.
        model = LbmModel(
            fluid_grid_res=(8, 8, 8),
            fluid_grid_cell_size=0.1,
            lattice="D3Q27",
            bc_types=(1, 2, 0, 0, 0, 0),
            bc_velocity=(
                (0.05, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        self.assertEqual(model.num_dirs, 27)
        self.assertEqual(model.bc_types[0], 1)

    def test_shan_chen_phase_enabled_flag(self) -> None:
        single = LbmModel(fluid_grid_res=(4, 4, 4), fluid_grid_cell_size=0.1, G=0.0)
        multi = LbmModel(fluid_grid_res=(4, 4, 4), fluid_grid_cell_size=0.1, G=-5.0)
        self.assertFalse(ShanChenPhase(single).enabled)
        self.assertTrue(ShanChenPhase(multi).enabled)

    def test_step_stats_mlups(self) -> None:
        stats = StepStats(ms_total=10.0).with_num_cells(64 * 64 * 64)
        self.assertGreater(stats.mlups, 0.0)

    def test_perf_metrics_from_step_stats(self) -> None:
        stats = StepStats(
            ms_total=2.0,
            ms_moments=0.5,
            ms_collision=1.0,
        ).with_num_cells(32 * 32 * 32)
        perf = perf_metrics_from_step_stats(
            stats,
            variant="V0",
            lattice="D3Q19",
            num_cells=32 * 32 * 32,
            bytes_per_cell_value=152.0,
        )
        self.assertEqual(perf.variant, "V0")
        self.assertEqual(perf.ms_per_lbm_step, 2.0)
        self.assertEqual(perf.bytes_per_cell, 152.0)

    def test_benchmark_registry_v0(self) -> None:
        v0 = get_variant("V0")
        self.assertEqual(v0.name, "dist_d3q19_sc")
        ids = {spec.variant_id for spec in list_variants()}
        self.assertIn("V0", ids)
        self.assertIn("V2", ids)

    def test_lbm_step_control_defaults(self) -> None:
        control = LbmStepControl()
        self.assertTrue(control.collect_stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
