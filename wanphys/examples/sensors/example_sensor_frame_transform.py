# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SensorFrameTransform example with visualized measured frames.

Run with:
    uv run python -m wanphys.examples.sensors.example_sensor_frame_transform
"""

from __future__ import annotations

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.viewer import ViewerGL

from wanphys.rigid import RigidModelBuilder, ShapeConfig, RigidDomain
from wanphys.sensors import Sensor


def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))


def get_cfg_directory() -> str:
    return os.path.join(get_source_directory(), "cfg")


def get_cfg(filename: str) -> str:
    return os.path.join(get_cfg_directory(), filename)


@wp.kernel
def animate_frames(
    time: float,
    reference_body: int,
    target_body: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    ref_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35 * wp.sin(0.7 * time))
    body_q[reference_body] = wp.transform(wp.vec3(0.0, 0.0, 0.45), ref_q)
    body_qd[reference_body] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

    target_pos = wp.vec3(
        0.85 * wp.cos(0.9 * time),
        0.55 * wp.sin(0.9 * time),
        0.85 + 0.18 * wp.sin(1.4 * time),
    )
    yaw_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.2 * time)
    roll_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.45 * wp.sin(1.7 * time))
    body_q[target_body] = wp.transform(target_pos, yaw_q * roll_q)
    body_qd[target_body] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


@wp.kernel
def build_sensor_frame_lines(
    relative_transforms: wp.array(dtype=wp.transform),
    reference_site: int,
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    axis_length: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    axis = tid % 3
    frame = tid // 3

    ref_body = shape_body[reference_site]
    X_wr = shape_transform[reference_site]
    if ref_body >= 0:
        X_wr = wp.transform_multiply(body_q[ref_body], X_wr)

    X_wf = X_wr
    if frame > 0:
        X_wf = wp.transform_multiply(X_wr, relative_transforms[frame - 1])

    axis_vec = wp.vec3(1.0, 0.0, 0.0)
    color = wp.vec3(1.0, 0.1, 0.1)
    if axis == 1:
        axis_vec = wp.vec3(0.0, 1.0, 0.0)
        color = wp.vec3(0.1, 0.9, 0.1)
    elif axis == 2:
        axis_vec = wp.vec3(0.0, 0.0, 1.0)
        color = wp.vec3(0.15, 0.35, 1.0)

    origin = wp.transform_get_translation(X_wf)
    starts[tid] = origin
    ends[tid] = origin + wp.quat_rotate(wp.transform_get_rotation(X_wf), axis_vec * axis_length)
    colors[tid] = color


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = int(args.fps)
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_time = 0.0

        builder = RigidModelBuilder()
        builder.add_ground_plane(cfg=ShapeConfig(mu=0.6))

        self.reference_body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 0.45), wp.quat_identity()),
            label="reference_body",
        )
        self.reference_box = builder.add_shape_box(
            body=self.reference_body,
            hx=0.32,
            hy=0.20,
            hz=0.08,
            cfg=ShapeConfig(density=0.0, mu=0.4),
            label="reference_box",
        )
        self.reference_site: int = builder.add_site(
            self.reference_body,
            xform=wp.transform((0.0, 0.0, 0.18), wp.quat_identity()),
            scale=(0.06, 0.06, 0.06),
            label="reference_site",
            visible=True,
        )

        self.target_body = builder.add_body(
            xform=wp.transform((0.85, 0.0, 0.85), wp.quat_identity()),
            label="target_body",
        )
        self.target_box = builder.add_shape_box(
            body=self.target_body,
            hx=0.22,
            hy=0.14,
            hz=0.12,
            cfg=ShapeConfig(density=0.0, mu=0.4),
            label="target_box",
        )
        self.target_tip_site: int = builder.add_site(
            self.target_body,
            xform=wp.transform((0.32, 0.0, 0.0), wp.quat_identity()),
            scale=(0.05, 0.05, 0.05),
            label="target_tip_site",
            visible=True,
        )

        self.model = builder.finalize(device=str(wp.get_device()))
        self.domain = RigidDomain(self.model)
        self.state = self.domain.state
        self.frame_sensor = Sensor(self.domain, get_cfg("frame_transform_sensor.yaml"))

        measured_count = int(self.frame_sensor.sensor.transforms.shape[0])
        self.line_count = (measured_count + 1) * 3
        self.axis_starts = wp.zeros(self.line_count, dtype=wp.vec3, device=self.model.device)
        self.axis_ends = wp.zeros(self.line_count, dtype=wp.vec3, device=self.model.device)
        self.axis_colors = wp.zeros(self.line_count, dtype=wp.vec3, device=self.model.device)

        self.model.setup_viewer(self.viewer)
        self.viewer.update_shape_colors(
            {
                self.reference_site: (1.0, 0.85, 0.10),
                self.target_tip_site: (0.95, 0.20, 0.95),
                self.reference_box: (0.25, 0.25, 0.25),
                self.target_box: (0.15, 0.45, 1.0),
            }
        )

        if isinstance(self.viewer, ViewerGL):
            self.viewer.camera.pos = type(self.viewer.camera.pos)(2.6, -3.6, 2.0)
            self.viewer.camera.pitch = -22.0
            self.viewer.camera.yaw = 126.0

    def step(self):
        wp.launch(
            animate_frames,
            dim=1,
            inputs=[
                self.sim_time,
                self.reference_body,
                self.target_body,
                self.state.body_q,
                self.state.body_qd,
            ],
            device=self.model.device,
        )
        self.frame_sensor.sensor.update()
        wp.launch(
            build_sensor_frame_lines,
            dim=self.line_count,
            inputs=[
                self.frame_sensor.sensor.transforms,
                self.reference_site,
                self.model.shape_body,
                self.model.shape_transform,
                self.state.body_q,
                float(self.args.axis_length),
            ],
            outputs=[self.axis_starts, self.axis_ends, self.axis_colors],
            device=self.model.device,
        )
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state.as_newton_state())
        self.viewer.log_lines(
            "/sensor_frame_transform/axes",
            self.axis_starts,
            self.axis_ends,
            self.axis_colors,
            width=0.018,
            hidden=False,
        )
        self.viewer.end_frame()

    def test_final(self):
        transforms = self.frame_sensor.sensor.transforms.numpy()
        assert transforms.shape == (2, 7)
        assert np.isfinite(transforms).all()
        assert np.linalg.norm(transforms[0, :3]) > 0.1


def build_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--axis-length", type=float, default=0.28)
    return parser


def main():
    wp.init()
    parser = build_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
