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


def _import_optional(module_name: str) -> types.ModuleType:
    """Import a historical example, or skip if the asset was removed."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(
            f"Missing historical LBM example/helper {exc.name!r} "
            f"(requested {module_name!r}). Restore the example or delete "
            "this test in a dedicated cleanup task."
        ) from exc


class TestLbmDamBreakExamples(unittest.TestCase):
    def test_passive_marker_dambreak_null_smoke(self) -> None:
        """The passive-marker dam-break example should construct and step."""
        module: types.ModuleType = _import_optional("wanphys.examples.lbm.fluid_grid_lbm_dambreak")

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
