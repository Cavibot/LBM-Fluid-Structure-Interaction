# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end numerical acceptance for the three L0 offline scenes."""

from __future__ import annotations

import unittest

import warp as wp

from wanphys.examples.lbm.home_free_offline import (
    OfflineSceneName,
    make_scene_config,
    run_scene_acceptance,
)


class TestHomeFreeOfflineAcceptance(unittest.TestCase):
    def _all_scenes_pass(self, device: str) -> None:
        for name in OfflineSceneName:
            with self.subTest(scene=name.value, device=device):
                acceptance = run_scene_acceptance(
                    make_scene_config(name, device=device)
                )
                self.assertEqual(acceptance.scene, name.value)
                self.assertGreater(acceptance.topology_change_count, 0)
                self.assertLess(acceptance.maximum_relative_mass_error, 2.0e-5)

    def test_all_l0_scenes_pass_on_cpu(self) -> None:
        self._all_scenes_pass("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_all_l0_scenes_pass_on_cuda(self) -> None:
        self._all_scenes_pass("cuda:0")


if __name__ == "__main__":
    unittest.main()
