# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""View the final common-state HOME-Free dam-break profiles in ViewerGL."""

from __future__ import annotations

import time
from typing import Any

import newton.examples
import warp as wp

from wanphys._src.fluid.fluid_viewer import (
    FluidViewerGL,
    HomeFreeRenderField,
    ScreenSpaceFluidRenderer,
)
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    RAY_MARCH_STEPS,
    RENDER_CELL_SIZE,
    SSFR_THRESHOLD,
    _look_at_angles,
    _tank_edges,
)

from .dambreak import build_combination_dambreak, make_combination_config


class CombinationDambreakViewer:
    def __init__(
        self,
        viewer: Any,
        *,
        backend: str,
        scale: str,
        device: str,
        substeps_per_frame: int,
        print_every: int,
    ) -> None:
        self.viewer = viewer
        self.backend = backend
        self.scale = scale
        self.device = device
        self.substeps_per_frame = max(1, int(substeps_per_frame))
        self.print_every = max(0, int(print_every))
        self._build_simulation()
        self._setup_rendering()
        self.viewer._paused = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.register_key_press(self._on_key_press)
        print("Combination Viewer: Space run/pause, R reset, mouse orbit, wheel zoom")

    def _build_simulation(self) -> None:
        config = make_combination_config(
            self.backend, scale=self.scale, device=self.device
        )
        self.scene = build_combination_dambreak(config)
        self.render_field = HomeFreeRenderField(
            config.resolution,
            device=self.scene.model._device,
            source_up_axis=2,
        )
        self.sim_time = 0.0
        self.last_step_ms = 0.0

    def _setup_rendering(self) -> None:
        self.ssfr = None
        shape = self.render_field.render_shape
        self.render_extent = tuple(value * RENDER_CELL_SIZE for value in shape)
        starts, ends = _tank_edges(self.render_extent)
        self.tank_starts = wp.array(
            starts, dtype=wp.vec3, device=self.scene.model._device
        )
        self.tank_ends = wp.array(
            ends, dtype=wp.vec3, device=self.scene.model._device
        )
        if not isinstance(self.viewer, FluidViewerGL):
            return
        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=self.viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.scene.model._device,
        )
        self.viewer.register_post_render_callback(
            lambda current: self.ssfr.render(current)
        )
        ex, ey, ez = self.render_extent
        camera_position = (ex * 1.45, -ey * 2.6, ez * 1.35)
        camera_target = (ex * 0.5, ey * 0.5, ez * 0.32)
        pitch, yaw = _look_at_angles(camera_position, camera_target)
        self.viewer.set_camera(wp.vec3(*camera_position), pitch=pitch, yaw=yaw)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.R:
            self.viewer._paused = True
            self._build_simulation()
            print("Combination dam-break reset; press Space to run.")

    def step(self) -> None:
        started = time.perf_counter()
        try:
            for _ in range(self.substeps_per_frame):
                self.scene.step()
            self.scene.synchronize()
        except Exception as error:
            self.viewer._paused = True
            print(
                f"Combination Viewer paused at step {self.scene.step_index}: {error}"
            )
            return
        self.sim_time = self.scene.step_index * self.scene.config.time_step
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        if self.print_every and self.scene.step_index % self.print_every == 0:
            metrics = self.scene.measure()
            print(
                f"{self.backend} step={metrics.step} "
                f"speed={metrics.maximum_speed:.6g} "
                f"mass_error={metrics.relative_mass_error:.3g} "
                f"floor_gaps={metrics.floor_gap_columns} "
                f"detached_bulk={metrics.detached_bulk_columns} "
                f"frame_ms={self.last_step_ms:.3f}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/home-free-combination/tank",
            self.tank_starts,
            self.tank_ends,
            (0.18, 0.22, 0.28),
            width=0.008,
        )
        if self.ssfr is not None and self.ssfr.available:
            free = self.scene.fluid.free_surface_state
            density = self.render_field.update(free.fill_level, free.flags)
            self.ssfr.set_density_field(
                density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=RENDER_CELL_SIZE,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument(
        "--backend", choices=("fast", "balanced", "strict"), default="balanced"
    )
    parser.add_argument(
        "--scale", choices=("gate", "preview", "viewer-default"), default="viewer-default"
    )
    parser.add_argument("--substeps-per-frame", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=25)
    viewer, args = init_fluid_viewer(parser)
    example = CombinationDambreakViewer(
        viewer,
        backend=args.backend,
        scale=args.scale,
        device=args.device or "cuda:0",
        substeps_per_frame=args.substeps_per_frame,
        print_every=args.print_every,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
