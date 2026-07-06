# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

# eg. python -m wanphys.geometry.examples.mesh_to_points_demo

from pathlib import Path
from wanphys._src.geometry.utils.mesh_to_points_api import obj_to_outputs

HERE = Path(__file__).resolve().parent  # 当前脚本所在目录
obj_path = (HERE / ".." / "assets" / "mesh_to_points_input.obj").resolve()

obj_to_outputs(
    str(obj_path),
    ply_path=str((HERE / "mesh_to_points_output.ply").resolve()),
    txt_path=str((HERE / "mesh_to_points_output.txt").resolve()),
    radius=0.01,
    point_count=200000,
    layers=1,
    layer_gap=0.0025,
    inward=True,
    device="cuda:0",
)