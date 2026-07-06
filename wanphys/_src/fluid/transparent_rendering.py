# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive cube-grid visibility demo.

This demo renders an ``x * y * z`` cube grid using Newton's viewer only,
without building a physics scene. The cube color is adjustable and every
frame each cube is randomly shown or hidden.

Usage:
    python -m wanphys.examples.translucent_rendering
    python -m wanphys.examples.translucent_rendering --grid-x 10 --grid-y 6 --grid-z 4
    python -m wanphys.examples.translucent_rendering --color-r 0.2 --color-g 0.8 --color-b 0.9
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def update_cube_transforms(
    base_positions: wp.array(dtype=wp.vec3),
    hidden_z: float,
    visibility_probability: float,
    seed: int,
    out_transforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    pos = base_positions[tid]

    if wp.randf(rng) >= visibility_probability:
        pos = wp.vec3(pos[0], pos[1], hidden_z)

    out_transforms[tid] = wp.transform(pos, wp.quat_identity())


class TranslucentRenderingExample:
    """Render a cube lattice with per-frame random visibility."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.time = 0.0

        self.grid_x = int(args.grid_x)
        self.grid_y = int(args.grid_y)
        self.grid_z = int(args.grid_z)
        self.total_cubes = self.grid_x * self.grid_y * self.grid_z

        self.cube_size = float(args.cube_size)
        self.cube_gap = float(args.cube_gap)
        self.cube_half_extent = 0.5 * self.cube_size
        self.pitch = self.cube_size + self.cube_gap

        self.visibility_probability = float(args.visibility_probability)
        self.color = np.array([args.color_r, args.color_g, args.color_b], dtype=np.float32)
        self.roughness = float(args.roughness)
        self.metallic = float(args.metallic)
        self.cube_alpha = float(args.cube_alpha)
        self.seed = int(args.seed)
        self.frame_index = 0

        self.floor_margin = 0.05 * self.cube_size
        self.hidden_z = -1000.0

        self.grid_positions = self._build_grid_positions()
        self.base_positions = wp.array(self.grid_positions, dtype=wp.vec3, device=self.device)
        self.span_x, self.span_y, self.span_z = self._compute_grid_span()
        self.plane_extent = max(self.span_x, self.span_y, self.cube_size) * 1.8

        self.cube_colors = None
        self.cube_materials = None
        self.plane_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.plane_colors = wp.array([wp.vec3(0.12, 0.13, 0.16)], dtype=wp.vec3, device=self.device)
        self.plane_materials = wp.array([wp.vec4(0.7, 0.0, 1.0, 0.0)], dtype=wp.vec4, device=self.device)

        self._refresh_cube_appearance()
        self.instance_transforms = wp.empty(self.total_cubes, dtype=wp.transform, device=self.device)
        self._update_transforms()

        self._setup_camera()
        self._setup_renderer()

    def _build_grid_positions(self) -> np.ndarray:
        xs = (np.arange(self.grid_x, dtype=np.float32) - 0.5 * (self.grid_x - 1)) * self.pitch
        ys = (np.arange(self.grid_y, dtype=np.float32) - 0.5 * (self.grid_y - 1)) * self.pitch
        zs = self.cube_half_extent + self.floor_margin + np.arange(self.grid_z, dtype=np.float32) * self.pitch

        positions = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        return positions.reshape(-1, 3).astype(np.float32)

    def _compute_grid_span(self) -> tuple[float, float, float]:
        span_x = max((self.grid_x - 1) * self.pitch + self.cube_size, self.cube_size)
        span_y = max((self.grid_y - 1) * self.pitch + self.cube_size, self.cube_size)
        span_z = max((self.grid_z - 1) * self.pitch + self.cube_size + self.floor_margin, self.cube_size)
        return span_x, span_y, span_z

    def _refresh_cube_appearance(self):
        self.cube_colors = wp.array([wp.vec3(*self.color)], dtype=wp.vec3, device=self.device)
        self.cube_materials = wp.array(
            [wp.vec4(self.roughness, self.metallic, 0.0, self.cube_alpha)],
            dtype=wp.vec4,
            device=self.device,
        )

    def _update_transforms(self):
        wp.launch(
            update_cube_transforms,
            dim=self.total_cubes,
            inputs=[
                self.base_positions,
                self.hidden_z,
                self.visibility_probability,
                self.seed + self.frame_index * 7919,
            ],
            outputs=[self.instance_transforms],
            device=self.device,
            record_tape=False,
        )

    def _setup_camera(self):
        target = wp.vec3(0.0, 0.0, 0.5 * self.span_z)
        camera_pos = wp.vec3(
            self.span_x * 1.35 + 2.5 * self.cube_size,
            -self.span_y * 1.9 - 4.0 * self.cube_size,
            self.span_z * 1.65 + 4.0 * self.cube_size,
        )

        if hasattr(self.viewer, "set_camera_target"):
            self.viewer.set_camera_target(pos=camera_pos, target=target)

    def _setup_renderer(self):
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is None:
            return

        renderer.shadow_extents = max(self.span_x, self.span_y, self.span_z) * 1.4
        renderer.shadow_radius = 1.8
        renderer.diffuse_scale = 1.15
        renderer.specular_scale = 1.1

    def gui(self, ui):
        ui.text(f"Grid: {self.grid_x} x {self.grid_y} x {self.grid_z}")
        ui.text(f"Cubes: {self.total_cubes}")

        changed, value = ui.slider_float("Visible Probability", self.visibility_probability, 0.0, 1.0)
        if changed:
            self.visibility_probability = value

        changed, value = ui.slider_float("Color R", float(self.color[0]), 0.0, 1.0)
        if changed:
            self.color[0] = value
            self._refresh_cube_appearance()

        changed, value = ui.slider_float("Color G", float(self.color[1]), 0.0, 1.0)
        if changed:
            self.color[1] = value
            self._refresh_cube_appearance()

        changed, value = ui.slider_float("Color B", float(self.color[2]), 0.0, 1.0)
        if changed:
            self.color[2] = value
            self._refresh_cube_appearance()

        changed, value = ui.slider_float("Roughness", self.roughness, 0.0, 1.0)
        if changed:
            self.roughness = value
            self._refresh_cube_appearance()

        changed, value = ui.slider_float("Metallic", self.metallic, 0.0, 1.0)
        if changed:
            self.metallic = value
            self._refresh_cube_appearance()

        changed, value = ui.slider_float("Cube Alpha", self.cube_alpha, 0.05, 1.0)
        if changed:
            self.cube_alpha = value
            self._refresh_cube_appearance()

    def step(self):
        self.frame_index += 1
        self._update_transforms()
        self.time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.time)

        self.viewer.log_shapes(
            "/cube_grid",
            newton.GeoType.BOX,
            (self.cube_half_extent, self.cube_half_extent, self.cube_half_extent),
            self.instance_transforms,
            self.cube_colors,
            self.cube_materials,
        )

        self.viewer.log_shapes(
            "/ground_plane",
            newton.GeoType.PLANE,
            (self.plane_extent, self.plane_extent),
            self.plane_transform,
            self.plane_colors,
            self.plane_materials,
        )

        self.viewer.end_frame()

    def test_final(self):
        pass


def create_parser():
    parser = newton.examples.create_parser()
    default_device = "cuda:0" if wp.is_cuda_available() else "cpu"
    parser.set_defaults(device=default_device)
    parser.add_argument("--grid-x", type=int, default=8, help="Number of cubes along X.")
    parser.add_argument("--grid-y", type=int, default=8, help="Number of cubes along Y.")
    parser.add_argument("--grid-z", type=int, default=6, help="Number of cubes along Z.")
    parser.add_argument("--cube-size", type=float, default=0.2, help="Cube side length.")
    parser.add_argument("--cube-gap", type=float, default=0.0, help="Gap between adjacent cubes.")
    parser.add_argument(
        "--visibility-probability",
        type=float,
        default=0.5,
        help="Probability that a cube is visible on each frame.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Logical update rate for the random visibility animation.")
    parser.add_argument("--color-r", type=float, default=0.25, help="Cube color red channel.")
    parser.add_argument("--color-g", type=float, default=0.7, help="Cube color green channel.")
    parser.add_argument("--color-b", type=float, default=0.95, help="Cube color blue channel.")
    parser.add_argument("--roughness", type=float, default=0.35, help="Cube material roughness.")
    parser.add_argument("--metallic", type=float, default=0.05, help="Cube material metallic value.")
    parser.add_argument("--cube-alpha", type=float, default=0.45, help="Cube transparency alpha in [0, 1].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for visibility toggling.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    example = TranslucentRenderingExample(viewer, args)
    newton.examples.run(example, args)
