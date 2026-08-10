# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np

from wanphys.examples.lbm.home_rebuild.volume_artifacts import (
    extract_fill_surface,
    save_volume_snapshot,
)


def test_volume_snapshot_and_surface_preserve_explicit_axis_mapping(
    tmp_path: Path,
) -> None:
    shape = (6, 5, 4)
    fill = np.zeros(shape, dtype=np.float32)
    fill[1:4, 1:3, 1:3] = 1.0
    snapshot_path = tmp_path / "snapshot.npz"
    artifact = save_volume_snapshot(
        snapshot_path,
        fill=fill,
        mass=fill,
        flags=np.zeros(shape, dtype=np.int32),
        moments=np.zeros((*shape, 10), dtype=np.float32),
        excess_mass=np.zeros(shape, dtype=np.float32),
    )
    surface = extract_fill_surface(snapshot_path, tmp_path / "surface.obj")

    assert artifact["shape"] == [6, 5, 4]
    assert artifact["solver_axis_order"] == ["x", "height_y", "depth_z"]
    assert surface["render_axis_order"] == ["x", "depth_z", "height_y"]
    assert surface["vertices"] > 0
    assert surface["triangles"] > 0
