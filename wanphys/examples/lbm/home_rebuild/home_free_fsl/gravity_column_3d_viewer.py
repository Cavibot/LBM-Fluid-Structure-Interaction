# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""View the proven link-wise HOME-Free 3D gravity column in ViewerGL."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
)
from wanphys._src.fluid.fluid_viewer import (
    FluidViewerGL,
    HomeFreeRenderField,
    ScreenSpaceFluidRenderer,
)
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)

SSFR_THRESHOLD = 0.5
RAY_MARCH_STEPS = 1600
RENDER_CELL_SIZE = 0.02

LEGACY_VIEWER_N = 128
HOME_CELL_SIZE = 0.1
HOME_PHYSICAL_VISCOSITY = 2.0e-4
HOME_PHYSICAL_GRAVITY = -5.0e-6
HIGH_LATTICE_GRAVITY = -5.0e-3
HIGH_PHYSICAL_GRAVITY = HIGH_LATTICE_GRAVITY * HOME_CELL_SIZE
HIGH_GRAVITY_RAMP_STEPS = 60


def _simulation_cell_size(config: GravityColumnConfig) -> float:
    return HOME_CELL_SIZE


def _maximum_lattice_speed(config: GravityColumnConfig) -> float:
    if config.profile == "viewer-high-gravity-home-free":
        return 0.55
    return 0.2


def _gravity_ramp_steps(config: GravityColumnConfig) -> int:
    if config.profile == "viewer-high-gravity-home-free":
        return HIGH_GRAVITY_RAMP_STEPS
    return 0


def make_viewer_config(profile: str, device: str) -> GravityColumnConfig:
    """Return a Viewer-default scene or an existing link-wise benchmark config."""
    if profile in ("viewer-default", "viewer-high-gravity"):
        # Preserve the original Viewer geometry while retaining the proven
        # HOME-Free physical parameters and unit scaling.
        return GravityColumnConfig(
            profile=(
                "viewer-high-gravity-home-free"
                if profile == "viewer-high-gravity"
                else "viewer-default-home-free"
            ),
            resolution_x=LEGACY_VIEWER_N,
            resolution_y=LEGACY_VIEWER_N,
            resolution_z=LEGACY_VIEWER_N,
            periodic_depth=False,
            column_end_x=int(LEGACY_VIEWER_N * 0.25) - 1,
            column_end_y=LEGACY_VIEWER_N - 3,
            physical_viscosity=HOME_PHYSICAL_VISCOSITY,
            physical_gravity_y=(
                HIGH_PHYSICAL_GRAVITY
                if profile == "viewer-high-gravity"
                else HOME_PHYSICAL_GRAVITY
            ),
            sample_steps=(0,),
            device=device,
        )
    if profile == "preview":
        return GravityColumnConfig(
            profile="r3.8-linkwise-true3d-preview-viewer",
            resolution_x=96,
            resolution_y=56,
            resolution_z=16,
            periodic_depth=False,
            column_end_x=24,
            column_end_y=35,
            physical_gravity_y=-5.0e-6,
            sample_steps=(0, 4000),
            device=device,
        )
    if profile == "formal":
        return GravityColumnConfig(
            profile="r3.9-linkwise-true3d-wide-formal-viewer",
            resolution_x=160,
            resolution_y=96,
            resolution_z=64,
            periodic_depth=False,
            physical_gravity_y=-5.0e-6,
            sample_steps=(0, 20000),
            device=device,
        )
    raise ValueError(f"unknown viewer profile: {profile}")


def _tank_edges(extent: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = extent
    corners = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (x, 0.0, 0.0),
            (0.0, y, 0.0),
            (x, y, 0.0),
            (0.0, 0.0, z),
            (x, 0.0, z),
            (0.0, y, z),
            (x, y, z),
        ],
        dtype=np.float32,
    )
    pairs = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return (
        np.asarray([corners[start] for start, _ in pairs], dtype=np.float32),
        np.asarray([corners[end] for _, end in pairs], dtype=np.float32),
    )


def _look_at_angles(
    position: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float]:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    dz = target[2] - position[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return pitch, yaw


class HomeFreeGravityColumnViewer:
    """Interactive render shell around the existing link-wise stepper."""

    def __init__(
        self,
        viewer: Any,
        *,
        config: GravityColumnConfig,
        substeps_per_frame: int,
        print_every: int,
        frame_delay: float = 0.0,
    ) -> None:
        self.viewer = viewer
        self.config = config
        self.substeps_per_frame = max(1, int(substeps_per_frame))
        self.print_every = max(0, int(print_every))
        self.frame_delay = max(0.0, float(frame_delay))
        self.sim_step = 0
        self.sim_time = 0.0
        self.last_step_ms = 0.0
        self.last_max_speed = 0.0
        self._build_simulation()
        self._setup_rendering()
        self.viewer._paused = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.register_key_press(self._on_key_press)
        print("HOME-Free Viewer: Space run/pause, R reset, mouse orbit, wheel zoom")

    def _build_simulation(self) -> None:
        shape = (
            self.config.resolution_x,
            self.config.resolution_y,
            self.config.resolution_z,
        )
        self.gravity_ramp_steps = _gravity_ramp_steps(self.config)
        self.target_body_acceleration = (
            0.0,
            self.config.physical_gravity_y,
            0.0,
        )
        initial_body_acceleration = self.target_body_acceleration
        if self.gravity_ramp_steps:
            initial_body_acceleration = (
                0.0,
                self.config.physical_gravity_y / self.gravity_ramp_steps,
                0.0,
            )
        self.model = HomeCoreModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=_simulation_cell_size(self.config),
            device=self.config.device,
            kinematic_viscosity=self.config.physical_viscosity,
            max_lattice_speed=_maximum_lattice_speed(self.config),
            body_acceleration=initial_body_acceleration,
        )
        self.walls = FslWallMask.closed_box(self.model)
        self.fluid_a = HomeCoreState(self.model)
        self.fluid_b = HomeCoreState(self.model)
        self.fsl_a = FslState(self.model)
        self.fsl_b = FslState(self.model)
        fill = _column_fill(self.config, self.walls)
        self.walls.initialize_hydrostatic(
            self.fluid_a,
            self.fsl_a,
            fill,
            gas_density=self.config.gas_density,
            gravity_axis=1,
            surface_coordinate=float(self.config.column_end_y + 1),
        )
        self.fluid_b.copy_from(self.fluid_a)
        self.fsl_b.copy_from(self.fsl_a)
        self.stepper = FslWallDynamicTopologyStepper(
            self.model, self.walls, gas_density=self.config.gas_density
        )
        self.render_field = HomeFreeRenderField(
            shape,
            device=self.model._device,
            gas_flag=int(FslCellFlag.GAS),
        )
        self.sim_step = 0
        self.sim_time = 0.0
        self.last_step_ms = 0.0
        self.last_max_speed = 0.0

    def _setup_rendering(self) -> None:
        self.ssfr = None
        nx, ny, nz = self.render_field.source_shape
        scale = RENDER_CELL_SIZE
        self.render_cell_size = scale
        self.render_extent = (nx * scale, nz * scale, ny * scale)
        starts, ends = _tank_edges(self.render_extent)
        self.tank_starts = wp.array(starts, dtype=wp.vec3, device=self.model._device)
        self.tank_ends = wp.array(ends, dtype=wp.vec3, device=self.model._device)
        if not isinstance(self.viewer, FluidViewerGL):
            return
        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=self.viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.model._device,
        )
        self.viewer.register_post_render_callback(lambda current: self.ssfr.render(current))

        ex, ey, ez = self.render_extent
        camera_position = (ex * 1.35, -ey * 2.3, ez * 1.25)
        camera_target = (ex * 0.48, ey * 0.5, ez * 0.35)
        camera_pos = wp.vec3(*camera_position)
        if hasattr(self.viewer, "set_camera_target"):
            self.viewer.set_camera_target(
                pos=camera_pos,
                target=wp.vec3(*camera_target),
            )
        else:
            pitch, yaw = _look_at_angles(camera_position, camera_target)
            self.viewer.set_camera(pos=camera_pos, pitch=pitch, yaw=yaw)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.R:
            self.viewer._paused = True
            self._build_simulation()
            print("HOME-Free gravity column reset; press Space to run.")

    def step(self) -> None:
        started = time.perf_counter()
        for _ in range(self.substeps_per_frame):
            if self.gravity_ramp_steps and self.sim_step < self.gravity_ramp_steps:
                ramp_fraction = float(self.sim_step + 1) / float(
                    self.gravity_ramp_steps
                )
                self.model.body_acceleration = tuple(
                    value * ramp_fraction
                    for value in self.target_body_acceleration
                )
            try:
                diagnostics = self.stepper.step(
                    self.fluid_a,
                    self.fsl_a,
                    self.fluid_b,
                    self.fsl_b,
                    self.model.time_step,
                )
            except FloatingPointError as error:
                if self.config.profile != "viewer-high-gravity-home-free":
                    raise
                self.viewer._paused = True
                print(
                    f"HOME-Free high-gravity preview paused at step "
                    f"{self.sim_step}: {error}"
                )
                return
            self.fluid_a, self.fluid_b = self.fluid_b, self.fluid_a
            self.fsl_a, self.fsl_b = self.fsl_b, self.fsl_a
            self.sim_step += 1
            self.sim_time += self.model.time_step
            self.last_max_speed = max(
                self.last_max_speed, float(diagnostics.fluid.max_speed)
            )
        wp.synchronize_device(self.model._device)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        if self.frame_delay:
            time.sleep(self.frame_delay)
        if self.print_every and self.sim_step % self.print_every < self.substeps_per_frame:
            print(
                f"HOME-Free step={self.sim_step} max_speed={self.last_max_speed:.6g} "
                f"batch_ms={self.last_step_ms:.3f}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/home-free/tank",
            self.tank_starts,
            self.tank_ends,
            (0.18, 0.22, 0.28),
            width=0.008,
        )
        if self.ssfr is not None and self.ssfr.available:
            density = self.render_field.update(
                self.fsl_a.fill_level, self.fsl_a.flags
            )
            self.ssfr.set_density_field(
                density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=self.render_cell_size,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument(
        "--profile",
        choices=(
            "viewer-default",
            "viewer-high-gravity",
            "preview",
            "formal",
        ),
        default="viewer-default",
    )
    parser.add_argument("--substeps-per-frame", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--frame-delay", type=float, default=0.0)
    viewer, args = init_fluid_viewer(parser)
    config = make_viewer_config(args.profile, args.device or "cuda:0")
    example = HomeFreeGravityColumnViewer(
        viewer,
        config=config,
        substeps_per_frame=args.substeps_per_frame,
        print_every=args.print_every,
        frame_delay=args.frame_delay,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
