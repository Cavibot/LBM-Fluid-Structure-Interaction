# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from wanphys.examples.lbm.home_rebuild.comparisons.fsl_plic_geometry_audit import (
    _plic_segment,
)


def test_plic_segment_recovers_axis_aligned_half_cell_cut() -> None:
    segment = _plic_segment(
        np.asarray((1.0, 0.0, 0.0)),
        0.0,
        (4.0, 7.0),
    )

    assert segment is not None
    np.testing.assert_allclose(sorted(segment), [(4.0, 6.5), (4.0, 7.5)])


def test_plic_segment_recovers_diagonal_cut() -> None:
    segment = _plic_segment(
        np.asarray((1.0, 1.0, 0.0)),
        0.0,
        (2.0, 3.0),
    )

    assert segment is not None
    np.testing.assert_allclose(sorted(segment), [(1.5, 3.5), (2.5, 2.5)])
