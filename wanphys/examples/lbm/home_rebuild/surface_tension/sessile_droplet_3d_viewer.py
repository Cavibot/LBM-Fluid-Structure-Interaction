# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare HOME-Free sessile droplets at prescribed static contact angles."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeGeometricDomain,
    HomeLbmModel,
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
class SessileDropletConfig:
    resolution: tuple[int, int, int] = (25, 25, 20)
    radius: float = 6.25
    contact_angle_degrees: float = 120.0
    surface_tension: float = 0.005
    kinematic_viscosity: float = 0.5
    maximum_steps: int = 300
    device: str = "cuda:0"


def make_sessile_droplet_config(
    device: str, *, contact_angle_degrees: float = 120.0
) -> SessileDropletConfig:
    return SessileDropletConfig(
        device=device, contact_angle_degrees=contact_angle_degrees
    )


def sessile_hemisphere_fill(config: SessileDropletConfig) -> np.ndarray:
    """Return the topology-safe hemispherical initial condition used by the gate."""

    nx, ny, _ = config.resolution
    coordinates = np.indices(config.resolution, dtype=np.float64)
    distance = np.sqrt(
        (coordinates[0] - 0.5 * (nx - 1)) ** 2
        + (coordinates[1] - 0.5 * (ny - 1)) ** 2
        + (coordinates[2] + 0.5) ** 2
    )
    return np.clip(
        0.5 + (config.radius - distance) / 2.0, 0.0, 1.0
    ).astype(np.float32)


def sessile_metrics(fill: np.ndarray) -> dict[str, float | int]:
    bottom = np.asarray(fill, dtype=np.float64)[:, :, 0]
    x, y = np.indices(bottom.shape, dtype=np.float64)
    area = float(np.sum(bottom, dtype=np.float64))
    center_x = float(np.sum(x * bottom) / area)
    center_y = float(np.sum(y * bottom) / area)
    radius_squared = float(
        np.sum(
            ((x - center_x) ** 2 + (y - center_y) ** 2) * bottom,
            dtype=np.float64,
        )
        / area
    )
    occupied_rows = np.nonzero(np.any(fill > 0.01, axis=(0, 1)))[0]
    return {
        "volume": float(np.sum(fill, dtype=np.float64)),
        "bottom_area": area,
        "bottom_rms_radius": radius_squared**0.5,
        "height": int(occupied_rows[-1] + 1),
    }


def build_sessile_domain(
    config: SessileDropletConfig,
) -> tuple[HomeLbmModel, HomeFreeGeometricDomain, dict[str, float | int]]:
    model = HomeLbmModel(
        fluid_grid_res=config.resolution,
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        reference_density=1.0,
        kinematic_viscosity=config.kinematic_viscosity,
        max_lattice_speed=0.2,
        periodic=(False, False, False),
        device=config.device,
    )
    domain = HomeFreeGeometricDomain(
        model,
        surface_tension=config.surface_tension,
        contact_angle_degrees=config.contact_angle_degrees,
        advection_axes=(0, 1, 2),
        project_courant=False,
    )
    fill = sessile_hemisphere_fill(config)
    lattice_gamma = model.scaling.surface_tension_to_lattice(
        config.surface_tension
    )
    domain.initialize_uniform_lattice(
        fill, rho=1.0 + 6.0 * lattice_gamma / config.radius
    )
    return model, domain, sessile_metrics(fill)


class SessileDropletViewer:
    """Interactive visual gate for capillary wetting at a flat wall."""

    def __init__(
        self,
        viewer: Any,
        *,
        config: SessileDropletConfig,
        print_every: int,
        steps_per_frame: int = 1,
    ) -> None:
        self.viewer = viewer
        self.config = config
        self.print_every = max(0, int(print_every))
        self.steps_per_frame = max(1, int(steps_per_frame))
        self._build_simulation(config.contact_angle_degrees)
        self._setup_rendering()
        self.viewer._paused = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.register_key_press(self._on_key_press)
        print(
            "Sessile droplet: Space run/pause, R reset, "
            "1/2/3 select 60/90/120 degrees"
        )

    def _build_simulation(self, angle: float) -> None:
        self.config = replace(self.config, contact_angle_degrees=float(angle))
        self.model, self.domain, self.initial_metrics = build_sessile_domain(
            self.config
        )
        self.render_field = HomeFreeRenderField(
            self.config.resolution,
            device=self.model._device,
            gas_flag=int(HomeFreeCellFlag.GAS),
            source_up_axis=2,
        )
        self.sim_step = 0
        self.sim_time = 0.0
        self.maximum_speed = 0.0
        self.maximum_volume_drift = 0.0
        self.last_step_ms = 0.0

    def _setup_rendering(self) -> None:
        self.ssfr = None
        extent = tuple(float(value) for value in self.config.resolution)
        starts, ends = _tank_edges(extent)
        self.tank_starts = wp.array(starts, dtype=wp.vec3, device=self.model._device)
        self.tank_ends = wp.array(ends, dtype=wp.vec3, device=self.model._device)
        if not isinstance(self.viewer, FluidViewerGL):
            return
        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=self.viewer,
            max_particles=1,
            particle_radius=0.2,
            device=self.model._device,
        )
        self.viewer.register_post_render_callback(lambda current: self.ssfr.render(current))
        camera_position = (34.0, -38.0, 20.0)
        camera_target = (12.0, 12.0, 3.5)
        camera_pos = wp.vec3(*camera_position)
        if hasattr(self.viewer, "set_camera_target"):
            self.viewer.set_camera_target(
                pos=camera_pos, target=wp.vec3(*camera_target)
            )
        else:
            pitch, yaw = _look_at_angles(camera_position, camera_target)
            self.viewer.set_camera(pos=camera_pos, pitch=pitch, yaw=yaw)

    def _reset(self, angle: float) -> None:
        self.viewer._paused = True
        self._build_simulation(angle)
        print(f"Sessile droplet reset to {angle:.0f} degrees; press Space to run.")

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.R:
            self._reset(self.config.contact_angle_degrees)
        elif symbol == pyglet.window.key._1:
            self._reset(60.0)
        elif symbol == pyglet.window.key._2:
            self._reset(90.0)
        elif symbol == pyglet.window.key._3:
            self._reset(120.0)

    def step(self) -> None:
        for _ in range(self.steps_per_frame):
            if self.sim_step >= self.config.maximum_steps:
                break
            self._step_once()

    def _step_once(self) -> None:
        if self.sim_step >= self.config.maximum_steps:
            self.viewer._paused = True
            return
        started = time.perf_counter()
        try:
            self.domain.step(self.model.time_step)
        except Exception as error:
            self.viewer._paused = True
            print(f"Sessile droplet paused at step {self.sim_step}: {error}")
            return
        self.sim_step += 1
        self.sim_time += self.model.time_step
        diagnostics = self.domain.last_geometric_diagnostics
        assert diagnostics is not None
        self.maximum_speed = max(
            self.maximum_speed, diagnostics.solver.max_speed
        )
        metrics = sessile_metrics(self.domain.free_surface_state.fill_level.numpy())
        volume_drift = abs(
            float(metrics["volume"]) - float(self.initial_metrics["volume"])
        ) / float(self.initial_metrics["volume"])
        self.maximum_volume_drift = max(self.maximum_volume_drift, volume_drift)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        if self.print_every and self.sim_step % self.print_every == 0:
            print(
                f"wetting angle={self.config.contact_angle_degrees:.0f} "
                f"step={self.sim_step} radius={metrics['bottom_rms_radius']:.6g} "
                f"speed={self.maximum_speed:.6g} drift={volume_drift:.3g} "
                f"step_ms={self.last_step_ms:.3f}"
            )
        if self.sim_step >= self.config.maximum_steps:
            self.viewer._paused = True
            print("Sessile droplet reached 300 steps; choose 1/2/3 to compare.")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/home-free-wetting/domain",
            self.tank_starts,
            self.tank_ends,
            (0.18, 0.22, 0.28),
            width=0.08,
        )
        if self.ssfr is not None and self.ssfr.available:
            density = self.render_field.update(
                self.domain.free_surface_state.fill_level,
                self.domain.free_surface_state.flags,
            )
            self.ssfr.set_density_field(
                density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=1.0,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()

    def test_final(self) -> None:
        if self.maximum_speed >= 0.02:
            raise AssertionError(
                f"sessile droplet speed {self.maximum_speed} exceeds 0.02"
            )
        if self.maximum_volume_drift >= 1.0e-2:
            raise AssertionError(
                "sessile droplet transient volume drift "
                f"{self.maximum_volume_drift} exceeds 1e-2"
            )
        final_metrics = sessile_metrics(
            self.domain.free_surface_state.fill_level.numpy()
        )
        final_volume_drift = abs(
            float(final_metrics["volume"])
            - float(self.initial_metrics["volume"])
        ) / float(self.initial_metrics["volume"])
        if final_volume_drift >= 2.0e-3:
            raise AssertionError(
                "sessile droplet final volume drift "
                f"{final_volume_drift} exceeds 2e-3"
            )


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument(
        "--contact-angle", type=float, choices=(60.0, 90.0, 120.0), default=120.0
    )
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=4,
        help="simulation steps advanced before each rendered frame",
    )
    viewer, args = init_fluid_viewer(parser)
    example = SessileDropletViewer(
        viewer,
        config=make_sessile_droplet_config(
            args.device or "cuda:0", contact_angle_degrees=args.contact_angle
        ),
        print_every=args.print_every,
        steps_per_frame=args.steps_per_frame,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
