# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""View the qualified projected-geometric HOME-Free dam-break in ViewerGL."""

from __future__ import annotations

import time
from typing import Any

import newton.examples
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    ProjectedGeometricFslStepper,
)
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
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.geometric_dambreak_preview import (
    GeometricDambreakConfig,
)


def make_plic_viewer_config(device: str) -> GeometricDambreakConfig:
    """Return the exact 3000-step CUDA-qualified PLIC scene."""
    return GeometricDambreakConfig(
        mode="projected-geometric",
        resolution_x=64,
        resolution_y=56,
        resolution_z=32,
        cell_size=0.1,
        time_step=1.0,
        column_end_x=24,
        column_end_y=34,
        column_offset_x=1,
        physical_viscosity=2.0e-4,
        physical_gravity_y=-5.0e-6,
        interface_roundoff_tolerance=5.0e-6,
        sample_steps=(0,),
        device=device,
    )


class ProjectedGeometricDambreakViewer:
    """Interactive shell around the qualified volume/mass-only PLIC stepper."""

    def __init__(
        self,
        viewer: Any,
        *,
        config: GeometricDambreakConfig,
        print_every: int,
    ) -> None:
        self.viewer = viewer
        self.config = config
        self.print_every = max(0, int(print_every))
        self._build_simulation()
        self._setup_rendering()
        self.viewer._paused = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.register_key_press(self._on_key_press)
        print("PLIC Viewer: Space run/pause, R reset, mouse orbit, wheel zoom")

    def _build_simulation(self) -> None:
        shape = (
            self.config.resolution_x,
            self.config.resolution_y,
            self.config.resolution_z,
        )
        self.model = HomeCoreModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=self.config.cell_size,
            time_step=self.config.time_step,
            kinematic_viscosity=self.config.physical_viscosity,
            body_acceleration=(0.0, self.config.physical_gravity_y, 0.0),
            device=self.config.device,
        )
        self.walls = FslWallMask.periodic_depth_channel(self.model)
        self.fluid_a = HomeCoreState(self.model)
        self.fluid_b = HomeCoreState(self.model)
        self.fsl_a = FslState(self.model)
        self.fsl_b = FslState(self.model)
        fill_config = GravityColumnConfig(
            profile="viewer-plic-qualified",
            resolution_x=self.config.resolution_x,
            resolution_y=self.config.resolution_y,
            resolution_z=self.config.resolution_z,
            periodic_depth=True,
            column_end_x=self.config.column_end_x,
            column_end_y=self.config.column_end_y,
            column_offset_x=self.config.column_offset_x,
            physical_viscosity=self.config.physical_viscosity,
            physical_gravity_y=self.config.physical_gravity_y,
            gas_density=self.config.gas_density,
            sample_steps=(0,),
            device=self.config.device,
        )
        fill = _column_fill(fill_config, self.walls)
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
        self.stepper = ProjectedGeometricFslStepper(
            self.model,
            self.walls,
            axes=(0, 1),
            gas_density=self.config.gas_density,
            interface_roundoff_tolerance=(
                self.config.interface_roundoff_tolerance
            ),
        )
        self.render_field = HomeFreeRenderField(shape, device=self.model._device)
        self.sim_step = 0
        self.sim_time = 0.0
        self.last_step_ms = 0.0

    def _setup_rendering(self) -> None:
        self.ssfr = None
        nx, ny, nz = self.render_field.source_shape
        self.render_extent = (
            nx * RENDER_CELL_SIZE,
            nz * RENDER_CELL_SIZE,
            ny * RENDER_CELL_SIZE,
        )
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
        camera_position = (ex * 1.45, -ey * 2.6, ez * 1.35)
        camera_target = (ex * 0.5, ey * 0.5, ez * 0.32)
        pitch, yaw = _look_at_angles(camera_position, camera_target)
        self.viewer.set_camera(wp.vec3(*camera_position), pitch=pitch, yaw=yaw)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.R:
            self.viewer._paused = True
            self._build_simulation()
            print("PLIC dam-break reset; press Space to run.")

    def step(self) -> None:
        started = time.perf_counter()
        try:
            diagnostics = self.stepper.step(
                self.fluid_a,
                self.fsl_a,
                self.fluid_b,
                self.fsl_b,
                self.model.time_step,
            )
        except Exception as error:
            self.viewer._paused = True
            print(f"PLIC Viewer paused at step {self.sim_step}: {error}")
            return
        self.fluid_a, self.fluid_b = self.fluid_b, self.fluid_a
        self.fsl_a, self.fsl_b = self.fsl_b, self.fsl_a
        self.sim_step += 1
        self.sim_time += self.model.time_step
        wp.synchronize_device(self.model._device)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        if self.print_every and self.sim_step % self.print_every == 0:
            print(
                f"PLIC step={self.sim_step} speed={diagnostics.fluid.max_speed:.6g} "
                f"projection_iter={diagnostics.projection.iteration_count} "
                f"div={diagnostics.projection.projected_maximum_divergence:.3g} "
                f"step_ms={self.last_step_ms:.3f}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/home-free-plic/tank",
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
                cell_size=RENDER_CELL_SIZE,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--print-every", type=int, default=25)
    viewer, args = init_fluid_viewer(parser)
    example = ProjectedGeometricDambreakViewer(
        viewer,
        config=make_plic_viewer_config(args.device or "cuda:0"),
        print_every=args.print_every,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
