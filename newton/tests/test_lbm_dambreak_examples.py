# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for WanPhys LBM dam-break examples."""

from __future__ import annotations

import importlib
import math
import types
import unittest
from typing import Any

import numpy as np


class TestLbmDamBreakExamples(unittest.TestCase):
    def test_passive_marker_dambreak_null_smoke(self) -> None:
        """The passive-marker dam-break example should construct and step."""
        module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.fluid_grid_lbm_dambreak")

        import newton.viewer

        viewer: Any = newton.viewer.ViewerNull(num_frames=1)
        example: Any = module.LbmDamBreakExample(viewer)
        example.step()

        marker_np: np.ndarray = np.asarray(example.marker.numpy(), dtype=np.float64)
        marker_sum: float = float(marker_np.sum())

        self.assertTrue(math.isfinite(marker_sum))
        self.assertGreater(marker_sum, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
