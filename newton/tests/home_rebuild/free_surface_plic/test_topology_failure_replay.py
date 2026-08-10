# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys.examples.lbm.home_rebuild.comparisons.topology_failure_replay import (
    find_direct_links,
)


class TestTopologyFailureReplay(unittest.TestCase):
    def test_direct_links_are_unique_and_respect_periodic_depth(self) -> None:
        flags = np.ones((5, 5, 2), dtype=np.int32)
        solid = np.zeros_like(flags, dtype=bool)
        flags[2, 2, 0] = 2
        flags[2, 2, 1] = 0

        links = find_direct_links(
            flags, solid, periodic=(False, False, True)
        )

        self.assertEqual(links, [((2, 2, 0), (2, 2, 1))])

    def test_solid_gas_neighbor_is_not_a_direct_link(self) -> None:
        flags = np.ones((5, 5, 1), dtype=np.int32)
        solid = np.zeros_like(flags, dtype=bool)
        flags[2, 2, 0] = 2
        flags[3, 2, 0] = 0
        solid[3, 2, 0] = True

        self.assertEqual(
            find_direct_links(flags, solid, periodic=(False, False, False)),
            [],
        )


if __name__ == "__main__":
    unittest.main()
