# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""View the qualified HOME-Free Laplace droplet in Newton ViewerGL."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeFslResearchStepper,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
)
from wanphys._src.fluid.fluid_viewer import (
    FluidViewerGL,
    HomeFreeRenderField,
    ScreenSpaceFluidRenderer,
)
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    RAY_MARCH_STEPS,
    SSFR_THRESHOLD,
    _look_at_angles,
    _tank_edges,
)


@dataclass(frozen=True)
class LaplaceDropletConfig:
    resolution: int = 33
    radius: float = 10.25
    samples_per_axis: int = 16
    cell_size: float = 1.0
    time_step: float = 1.0
    kinematic_viscosity: float = 0.5
    gas_density: float = 1.0
    surface_tension: float = 0.001
    render_cell_size: float = 0.06
    device: str = "cuda:0"


def make_laplace_droplet_config(device: str) -> LaplaceDropletConfig:
    """Return a larger visual counterpart of the accepted 17-cubed test."""

    return LaplaceDropletConfig(device=device)


def sampled_sphere_vof(
    shape: tuple[int, int, int],
    radius: float,
    *,
    samples_per_axis: int,
    center: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build a sharp sphere volume fraction with topology-safe quadrature."""

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least two")
    sphere_center = (
        0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        if center is None
        else np.asarray(center, dtype=np.float64)
    )
    if sphere_center.shape != (3,) or not np.isfinite(sphere_center).all():
        raise ValueError("sphere center must contain three finite coordinates")
    if not math.isfinite(radius) or radius <= 0.5:
        raise ValueError("radius must be finite and greater than half a cell")

    coordinates = np.indices(shape).transpose(1, 2, 3, 0)
    delta = np.abs(coordinates - sphere_center)
    minimum_distance = np.linalg.norm(np.maximum(delta - 0.5, 0.0), axis=-1)
    maximum_distance = np.linalg.norm(delta + 0.5, axis=-1)
    inside = maximum_distance <= radius
    boundary = (minimum_distance < radius) & ~inside
    fill = np.zeros(shape, dtype=np.float32)
    fill[inside] = 1.0

    coordinate = (
        np.arange(samples_per_axis, dtype=np.float64) + 0.5
    ) / samples_per_axis - 0.5
    offsets = np.stack(
        np.meshgrid(coordinate, coordinate, coordinate, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    quadrature_floor = 1.0 / samples_per_axis**3
    for index in map(tuple, np.argwhere(boundary)):
        fraction = float(
            np.mean(
                np.sum(
                    (np.asarray(index) - sphere_center + offsets) ** 2,
                    axis=1,
                )
                <= radius * radius
            )
        )
        fill[index] = np.clip(
            fraction, quadrature_floor, 1.0 - quadrature_floor
        )
    return fill


class LaplaceDropletViewer:
    """Interactive visual gate for curvature and capillary pressure balance."""

    def __init__(
        self,
        viewer: Any,
        *,
        config: LaplaceDropletConfig,
        substeps_per_frame: int,
        print_every: int,
    ) -> None:
        self.viewer = viewer
        self.config = config
        self.substeps_per_frame = max(1, int(substeps_per_frame))
        self.print_every = max(0, int(print_every))
        self._build_simulation()
        self._setup_rendering()
        self.viewer._paused = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.register_key_press(self._on_key_press)
        print("Laplace droplet: Space run/pause, R reset, mouse orbit, wheel zoom")

    def _build_simulation(self) -> None:
        shape = (self.config.resolution,) * 3
        self.model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=self.config.cell_size,
            time_step=self.config.time_step,
            reference_density=self.config.gas_density,
            kinematic_viscosity=self.config.kinematic_viscosity,
            periodic=(False, False, False),
            device=self.config.device,
        )
        self.fluid_a = HomeLbmState(self.model)
        self.fluid_b = HomeLbmState(self.model)
        self.free_a = HomeFreeState(self.model)
        self.free_b = HomeFreeState(self.model)
        fill = sampled_sphere_vof(
            shape,
            self.config.radius,
            samples_per_axis=self.config.samples_per_axis,
        )
        lattice_gamma = self.model.scaling.surface_tension_to_lattice(
            self.config.surface_tension
        )
        initial_density = self.config.gas_density + 6.0 * lattice_gamma / self.config.radius
        HomeLbmSolver(self.model).initialize_uniform_lattice(
            self.fluid_a, rho=initial_density
        )
        self.free_a.initialize_from_fill_level(self.fluid_a, fill)
        self.fluid_b.copy_from(self.fluid_a)
        self.free_b.copy_from(self.free_a)
        self.stepper = HomeFreeFslResearchStepper(
            self.model,
            advection_axes=(0, 1, 2),
            gas_density=self.config.gas_density,
            surface_tension=self.config.surface_tension,
            no_support_policy="only_missing",
        )
        self.render_field = HomeFreeRenderField(
            shape,
            device=self.model._device,
            gas_flag=int(HomeFreeCellFlag.GAS),
        )
        self.initial_volume = float(np.sum(fill, dtype=np.float64))
        self.sim_step = 0
        self.sim_time = 0.0
        self.last_step_ms = 0.0

    def _setup_rendering(self) -> None:
        self.ssfr = None
        extent = self.config.resolution * self.config.render_cell_size
        self.render_extent = (extent, extent, extent)
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
        camera_position = (ex * 1.35, -ey * 2.15, ez * 1.2)
        camera_target = (ex * 0.5, ey * 0.5, ez * 0.5)
        camera_pos = wp.vec3(*camera_position)
        if hasattr(self.viewer, "set_camera_target"):
            self.viewer.set_camera_target(
                pos=camera_pos, target=wp.vec3(*camera_target)
            )
        else:
            pitch, yaw = _look_at_angles(camera_position, camera_target)
            self.viewer.set_camera(pos=camera_pos, pitch=pitch, yaw=yaw)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.R:
            self.viewer._paused = True
            self._build_simulation()
            print("Laplace droplet reset; press Space to run.")

    def step(self) -> None:
        started = time.perf_counter()
        try:
            for _ in range(self.substeps_per_frame):
                self.stepper.step(
                    self.fluid_a,
                    self.free_a,
                    self.fluid_b,
                    self.free_b,
                )
                self.fluid_a, self.fluid_b = self.fluid_b, self.fluid_a
                self.free_a, self.free_b = self.free_b, self.free_a
                self.sim_step += 1
                self.sim_time += self.model.time_step
        except Exception as error:
            self.viewer._paused = True
            print(f"Laplace droplet paused at step {self.sim_step}: {error}")
            return
        wp.synchronize_device(self.model._device)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        if self.print_every and self.sim_step % self.print_every < self.substeps_per_frame:
            volume = float(np.sum(self.free_a.fill_level.numpy(), dtype=np.float64))
            relative_drift = abs(volume - self.initial_volume) / self.initial_volume
            print(
                f"Laplace step={self.sim_step} volume_drift={relative_drift:.6g} "
                f"batch_ms={self.last_step_ms:.3f}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/home-free-laplace/domain",
            self.tank_starts,
            self.tank_ends,
            (0.18, 0.22, 0.28),
            width=0.006,
        )
        if self.ssfr is not None and self.ssfr.available:
            density = self.render_field.update(
                self.free_a.fill_level, self.free_a.flags
            )
            self.ssfr.set_density_field(
                density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=self.config.render_cell_size,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Apply a lightweight conservation gate when Newton runs in test mode."""

        if self.sim_step < 1:
            raise AssertionError("Laplace droplet did not advance")
        volume = float(np.sum(self.free_a.fill_level.numpy(), dtype=np.float64))
        relative_drift = abs(volume - self.initial_volume) / self.initial_volume
        if relative_drift >= 5.0e-4:
            raise AssertionError(
                f"Laplace droplet volume drift {relative_drift} exceeds 5e-4"
            )


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--substeps-per-frame", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=25)
    viewer, args = init_fluid_viewer(parser)
    example = LaplaceDropletViewer(
        viewer,
        config=make_laplace_droplet_config(args.device or "cuda:0"),
        substeps_per_frame=args.substeps_per_frame,
        print_every=args.print_every,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
