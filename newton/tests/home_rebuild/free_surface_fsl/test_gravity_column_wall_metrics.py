# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    _wall_film_metrics,
)


def test_wall_film_metrics_cover_full_depth_and_ignore_solid_shell() -> None:
    fill = np.zeros((8, 7, 6), dtype=np.float32)
    fill[2:6, 1:3, 2:-2] = 1.0
    fill[1, 1:5, 1:-1] = 1.0
    fill[-2, 1:3, 1:-1] = 0.25
    fill[2:6, 1:3, 1] = 0.75
    metrics = _wall_film_metrics(fill)

    assert metrics["left_wall_wet_cells_3d"] == 16
    assert metrics["left_wall_half_full_cells_3d"] == 16
    assert metrics["left_wall_max_half_full_height"] == 4
    assert metrics["right_wall_wet_cells_3d"] == 8
    assert metrics["right_wall_half_full_cells_3d"] == 0
    assert metrics["right_wall_max_wet_height"] == 2
    assert metrics["front_wall_wet_cells_3d"] == 14
    assert metrics["front_wall_half_full_cells_3d"] == 12
    assert metrics["front_wall_max_half_full_height"] == 4
    assert metrics["bulk_reference_height"] == 2
    assert metrics["wall_film_half_full_cells_above_bulk"] == 8
    assert metrics["wall_film_fill_volume_above_bulk"] == 8.0


def test_periodic_depth_omits_front_and_back_wall_metrics() -> None:
    fill = np.zeros((8, 7, 2), dtype=np.float32)
    fill[1, 1:5, :] = 1.0
    metrics = _wall_film_metrics(fill, periodic_depth=True)

    assert metrics["left_wall_half_full_cells_3d"] == 8
    assert "front_wall_half_full_cells_3d" not in metrics
