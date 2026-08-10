# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""View the CUDA-qualified rigid sphere entry through a HOME-Free surface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeGeometricDomain,
    HomeFreeLegacyDomain,
    HomeFreeRigidCoupling,
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
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)


@dataclass(frozen=True)
class SphereEntryConfig:
    resolution: tuple[int, int, int] = (24, 24, 22)
    interface_index: int = 10
    sphere_center: tuple[float, float, float] = (12.0, 12.0, 14.8)
    sphere_radius: float = 3.0
    initial_speed_z: float = -0.15
    gravity_z: float = -2.0e-4
    kinematic_viscosity: float = 0.2
    maximum_steps: int = 200
    frame_delay: float = 0.03
    profile: str = "balanced"
    device: str = "cuda:0"


def make_sphere_entry_config(
    device: str, *, profile: str = "balanced"
) -> SphereEntryConfig:
    if profile not in ("fast", "balanced", "strict"):
        raise ValueError("profile must be fast, balanced, or strict")
    return SphereEntryConfig(device=device, profile=profile)


def _represented_mass(
    fluid: HomeFreeLegacyDomain | HomeFreeGeometricDomain,
) -> float:
    state = fluid.free_surface_state
    recipients = (
        np.zeros(state.res, dtype=np.float64)
        if getattr(fluid, "uses_independent_geometric_mass", False)
        else fluid.topology.active_neighbor_count.numpy()
    )
    return float(
        np.sum(
            state.mass.numpy() + state.excess_mass.numpy() * recipients,
            dtype=np.float64,
        )
    )


class SphereEntryViewer:
    """Interactive shell around the accepted strong-coupling entry case."""

    def __init__(
        self,
        viewer: Any,
        *,
        config: SphereEntryConfig,
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
        print(
            f"HOME-Free FSI entry ({self.config.profile}): "
            "Space run/pause, R reset, mouse orbit, wheel zoom"
        )

    def _build_simulation(self) -> None:
        self.model = HomeLbmModel(
            fluid_grid_res=self.config.resolution,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=self.config.kinematic_viscosity,
            max_lattice_speed=0.2,
            periodic=(False, False, False),
            body_acceleration=(0.0, 0.0, self.config.gravity_z),
            device=self.config.device,
        )
        if self.config.profile == "fast":
            self.fluid = HomeFreeLegacyDomain(self.model)
        else:
            self.fluid = HomeFreeGeometricDomain(
                self.model,
                contact_angle_degrees=90.0,
                project_courant=self.config.profile == "strict",
            )
        builder = RigidModelBuilder(up_axis=2, gravity=self.config.gravity_z)
        self.body = builder.add_body(
            position=self.config.sphere_center,
            label="entering_sphere",
        )
        builder.add_shape_sphere(
            self.body,
            radius=self.config.sphere_radius,
            cfg=ShapeConfig(density=2.0, has_shape_collision=False),
        )
        rigid_model = builder.finalize(device=self.config.device)
        self.rigid = RigidDomain(
            rigid_model,
            solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
        )
        self.rigid.create_state()
        self.rigid.state._body_qd = wp.array(
            [[0.0, 0.0, self.config.initial_speed_z, 0.0, 0.0, 0.0]],
            dtype=wp.spatial_vector,
            device=self.config.device,
        )
        self.coupling = HomeFreeRigidCoupling(
            self.fluid,
            self.rigid,
            rigid_substeps=2,
            strong_coupling_max_iterations=10,
            strong_coupling_tolerance=2.0e-6,
            strong_coupling_relaxation=0.8,
        )
        fill = np.zeros(self.config.resolution, dtype=np.float32)
        fill[:, :, : self.config.interface_index] = 1.0
        fill[:, :, self.config.interface_index] = 0.5
        self.fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=self.config.interface_index,
            gas_direction=1,
        )
        self.render_field = HomeFreeRenderField(
            self.config.resolution,
            device=self.model._device,
            gas_flag=int(HomeFreeCellFlag.GAS),
            source_up_axis=2,
        )
        self.initial_mass = _represented_mass(self.fluid)
        self.sim_step = 0
        self.sim_time = 0.0
        self.last_step_ms = 0.0
        self.maximum_mass_drift = 0.0

    def _setup_rendering(self) -> None:
        self.rigid.model.setup_viewer(self.viewer)
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
        camera_position = (34.0, -38.0, 27.0)
        camera_target = (12.0, 12.0, 9.5)
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
            self._reset_simulation()

    def _reset_simulation(self) -> None:
        self.viewer._paused = True
        self._build_simulation()
        print("HOME-Free FSI entry reset; press Space to run.")

    def step(self) -> None:
        if self.sim_step >= self.config.maximum_steps:
            self.viewer._paused = True
            return
        started = time.perf_counter()
        try:
            diagnostics = self.coupling.step()
        except Exception as error:
            self.viewer._paused = True
            print(f"HOME-Free FSI entry paused at step {self.sim_step}: {error}")
            return
        wp.synchronize_device(self.model._device)
        self.sim_step += 1
        self.sim_time += self.model.time_step
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        mass_drift = abs(_represented_mass(self.fluid) - self.initial_mass) / self.initial_mass
        self.maximum_mass_drift = max(self.maximum_mass_drift, mass_drift)
        if self.config.frame_delay:
            time.sleep(self.config.frame_delay)
        if self.print_every and self.sim_step % self.print_every == 0:
            position = self.rigid.state.get_body_position(self.body)
            print(
                f"FSI entry step={self.sim_step} sphere_z={float(position[2]):.6g} "
                f"wet_links={diagnostics.interface.load.wet_link_count} "
                f"fresh={diagnostics.interface.transitions.fresh_cell_count} "
                f"dead={diagnostics.interface.transitions.dead_cell_count} "
                f"mass_drift={mass_drift:.3g} step_ms={self.last_step_ms:.3f}"
            )
        if self.sim_step >= self.config.maximum_steps:
            self.viewer._paused = True
            print(
                f"HOME-Free FSI entry reached the accepted "
                f"{self.config.maximum_steps}-step endpoint; press R to replay."
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid.state.as_newton_state())
        self.viewer.log_lines(
            "/home-free-fsi/tank",
            self.tank_starts,
            self.tank_ends,
            (0.18, 0.22, 0.28),
            width=0.08,
        )
        if self.ssfr is not None and self.ssfr.available:
            density = self.render_field.update(
                self.fluid.free_surface_state.fill_level,
                self.fluid.free_surface_state.flags,
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
        position = np.asarray(
            self.rigid.state.get_body_position(self.body), dtype=np.float64
        )
        if not np.isfinite(position).all():
            raise AssertionError("sphere position is not finite")
        if self.maximum_mass_drift >= 2.0e-5:
            raise AssertionError(
                f"FSI entry mass drift {self.maximum_mass_drift} exceeds 2e-5"
            )


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("fast", "balanced", "strict"),
        default="balanced",
    )
    viewer, args = init_fluid_viewer(parser)
    example = SphereEntryViewer(
        viewer,
        config=make_sphere_entry_config(
            args.device or "cuda:0", profile=args.profile
        ),
        print_every=args.print_every,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
