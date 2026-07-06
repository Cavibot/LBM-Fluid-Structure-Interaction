# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT Shan-Chen dam-break with two dynamic rigid spheres.

A closed-domain two-phase TRT dam-break starts with water on the left and air
elsewhere. Two rigid spheres start on the dry floor ahead of the dam front:
one heavier than water and one lighter than water. Only the spheres are coupled
into the LBM solid field; invisible rigid walls keep the spheres inside the box.

Run:
    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_two_spheres \
        --viewer gl
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import newton
import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N: int = 128
DH: float = 0.02

TAU: float = 0.51
LAMBDA_TRT: float = 0.001
G_SC: float = -5.0
SC_BOUNDARY_PSI: float = -1.0
PSI_TYPE: int = 1
PSI_REF: float = 1.0

GRAVITY_LBM: float = -0.005
RIGID_GRAVITY_Z: float = -1.0
OMEGA_REG: float = 0.01

DAM_X_FRAC: float = 0.25
RHO_WATER: float = 1.8
RHO_AIR: float = 0.1

SPHERE_RADIUS: float = 0.24
HEAVY_SPHERE_DENSITY: float = 0.3
LIGHT_SPHERE_DENSITY: float = 0.1

TEXTURED_SPHERE_VISUALS_ENABLED: bool = True
SPHERE_VISUAL_TEXTURE_SIZE: int = 256
SPHERE_VISUAL_MESH_LATITUDES: int = 32
SPHERE_VISUAL_MESH_LONGITUDES: int = 48
FEEDBACK_FORCE_SCALE: float = 4.0
BUOYANCY_FORCE_SCALE: float = 0.13
WATER_HORIZONTAL_DRAG_RATE: float = 1.5
WATER_VERTICAL_DRAG_RATE: float = 15.0
WALL_THICKNESS_CELLS: float = 2.0

BUOYANCY_SAMPLE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.75, 0.0, 0.0),
    (-0.75, 0.0, 0.0),
    (0.0, 0.75, 0.0),
    (0.0, -0.75, 0.0),
    (0.0, 0.0, 0.75),
    (0.0, 0.0, -0.75),
    (0.5, 0.5, 0.0),
    (0.5, -0.5, 0.0),
    (-0.5, 0.5, 0.0),
    (-0.5, -0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.5, 0.0, -0.5),
    (-0.5, 0.0, 0.5),
    (-0.5, 0.0, -0.5),
)

SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 1600

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 5
GRAVITY_RAMP_STEPS: int = 60


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------


def _make_sphere_rotation_texture(sphere_index: int) -> np.ndarray:
    """Create a simple asymmetric UV texture so sphere rotation is visible."""
    size: int = SPHERE_VISUAL_TEXTURE_SIZE
    u: np.ndarray = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[None, :]
    v: np.ndarray = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[:, None]
    palette: tuple[tuple[int, int, int], ...] = (
        (190, 75, 60),
        (58, 132, 205),
    )
    base_color: np.ndarray = np.array(palette[sphere_index % len(palette)], dtype=np.uint8)
    stripe_color: np.ndarray = np.array((18, 27, 38), dtype=np.uint8)
    marker_color: np.ndarray = np.array((246, 238, 206), dtype=np.uint8)

    texture: np.ndarray = np.empty((size, size, 4), dtype=np.uint8)
    texture[:, :, 0:3] = base_color
    texture[:, :, 3] = 255

    longitude_stripes: np.ndarray = np.broadcast_to(
        (np.floor((u + 0.11 * float(sphere_index)) * 10.0) % 2.0) < 0.35,
        (size, size),
    )
    latitude_band: np.ndarray = np.broadcast_to(np.abs(v - 0.5) < 0.035, (size, size))
    marker_u: float = 0.24 + 0.22 * float(sphere_index)
    marker_v: float = 0.34 + 0.12 * float(sphere_index)
    marker: np.ndarray = ((u - marker_u) / 0.11) ** 2 + ((v - marker_v) / 0.08) ** 2 < 1.0

    texture[longitude_stripes | latitude_band, 0:3] = stripe_color
    texture[marker, 0:3] = marker_color
    return texture


def _add_textured_sphere_visual(
    builder: RigidModelBuilder,
    body_id: int,
    radius: float,
    sphere_index: int,
) -> None:
    visual_mesh: newton.Mesh = newton.Mesh.create_sphere(
        radius=radius,
        num_latitudes=SPHERE_VISUAL_MESH_LATITUDES,
        num_longitudes=SPHERE_VISUAL_MESH_LONGITUDES,
        compute_uvs=True,
        compute_inertia=False,
    )
    visual_mesh.texture = _make_sphere_rotation_texture(sphere_index)
    visual_mesh.color = (1.0, 1.0, 1.0)
    visual_mesh.roughness = 0.6

    visual_cfg: ShapeConfig = ShapeConfig(
        density=0.0,
        is_visible=True,
        is_solid=True,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    builder.add_shape_mesh(
        body_id,
        mesh=visual_mesh,
        cfg=visual_cfg,
        label=f"dambreak_sphere_{sphere_index}_rotation_visual",
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _init_dambreak(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    dam_x: int,
    rho_w: float,
    rho_a: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    rho = rho_w
    if i >= dam_x:
        rho = rho_a

    noise = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    noise = noise * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho + noise * 0.005 * rho, 0.01)

    density[i, j, k] = rho
    f[0 * stride + idx] = (1.0 / 3.0) * rho
    f[1 * stride + idx] = (1.0 / 18.0) * rho
    f[2 * stride + idx] = (1.0 / 18.0) * rho
    f[3 * stride + idx] = (1.0 / 18.0) * rho
    f[4 * stride + idx] = (1.0 / 18.0) * rho
    f[5 * stride + idx] = (1.0 / 18.0) * rho
    f[6 * stride + idx] = (1.0 / 18.0) * rho
    f[7 * stride + idx] = (1.0 / 36.0) * rho
    f[8 * stride + idx] = (1.0 / 36.0) * rho
    f[9 * stride + idx] = (1.0 / 36.0) * rho
    f[10 * stride + idx] = (1.0 / 36.0) * rho
    f[11 * stride + idx] = (1.0 / 36.0) * rho
    f[12 * stride + idx] = (1.0 / 36.0) * rho
    f[13 * stride + idx] = (1.0 / 36.0) * rho
    f[14 * stride + idx] = (1.0 / 36.0) * rho
    f[15 * stride + idx] = (1.0 / 36.0) * rho
    f[16 * stride + idx] = (1.0 / 36.0) * rho
    f[17 * stride + idx] = (1.0 / 36.0) * rho
    f[18 * stride + idx] = (1.0 / 36.0) * rho


@wp.kernel
def _mask_solid_density(
    density: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
) -> None:
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        density_out[i, j, k] = 0.0
    else:
        density_out[i, j, k] = density[i, j, k]


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class TrtDamBreakTwoSpheres:
    """Closed TRT dam-break with two dynamic spheres on the dry floor."""

    def __init__(
        self,
        viewer: Any,
        *,
        feedback_force_scale: float = FEEDBACK_FORCE_SCALE,
        buoyancy_force_scale: float = BUOYANCY_FORCE_SCALE,
        water_horizontal_drag_rate: float = WATER_HORIZONTAL_DRAG_RATE,
        water_vertical_drag_rate: float = WATER_VERTICAL_DRAG_RATE,
    ) -> None:
        self.viewer: Any = viewer
        if isinstance(self.viewer, FluidViewerGL):
            self.viewer._paused: bool = True

        self.model: LbmModel = LbmModel(
            fluid_grid_res=(N, N, N),
            fluid_grid_cell_size=DH,
            tau=TAU,
            G=G_SC,
            sc_boundary_psi=SC_BOUNDARY_PSI,
            psi_type=PSI_TYPE,
            psi_ref=PSI_REF,
            lambda_trt=LAMBDA_TRT,
            use_regularization=True,
            omega_reg=OMEGA_REG,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY_LBM,
        )
        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()

        self.sim_dt: float = FRAME_DT / float(SIM_SUBSTEPS)
        self.sim_time: float = 0.0
        self.frame_count: int = 0
        self._last_ms: float = 0.0
        self.buoyancy_force_scale: float = float(buoyancy_force_scale)
        self.water_horizontal_drag_rate: float = float(water_horizontal_drag_rate)
        self.water_vertical_drag_rate: float = float(water_vertical_drag_rate)
        self._sphere_volume: float = (4.0 / 3.0) * math.pi * SPHERE_RADIUS * SPHERE_RADIUS * SPHERE_RADIUS
        self._last_submerged_by_body: dict[int, float] = {}
        self._last_extra_force_by_body: dict[int, tuple[float, float, float]] = {}

        self._init_fluid()
        self._init_rigid_scene(feedback_force_scale=float(feedback_force_scale))
        self._ramp_gravity()

        self.display_density: wp.array3d = wp.zeros((N, N, N), dtype=float, device=self.model._device)
        self.ssfr: ScreenSpaceFluidRenderer | None = None
        if isinstance(self.viewer, FluidViewerGL):
            ssfr: ScreenSpaceFluidRenderer = ScreenSpaceFluidRenderer(
                viewer=self.viewer,
                max_particles=1,
                particle_radius=0.01,
                device=self.model._device,
            )
            self.ssfr = ssfr
            self.viewer.register_post_render_callback(lambda v: ssfr.render(v))

        print("Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_fluid(self) -> None:
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        dam_x: int = int(float(n) * DAM_X_FRAC)

        wp.launch(
            _init_dambreak,
            dim=(n, n, n),
            inputs=[state.f, state.density, dam_x, RHO_WATER, RHO_AIR, 42, n, n, n, stride],
            device=self.model._device,
        )
        state_out: LbmState = self.domain._state_out
        wp.copy(state_out.f, state.f)
        wp.copy(state_out.density, state.density)
        wp.copy(state_out.velocity_x, state.velocity_x)
        wp.copy(state_out.velocity_y, state.velocity_y)
        wp.copy(state_out.velocity_z, state.velocity_z)
        wp.copy(state_out.solid_phi, state.solid_phi)
        wp.copy(state_out.solid_body_id, state.solid_body_id)
        wp.synchronize_device(self.model._device)

        water_cells: int = int((state.density.numpy() > SSFR_THRESHOLD).sum())
        print(f"TRT dam-break two spheres: {n}^3, dam x<{dam_x}, water_cells={water_cells}")
        print(f"  fluid: tau={TAU}, lambda={LAMBDA_TRT}, G={G_SC}, gz_lbm={GRAVITY_LBM}")

    def _init_rigid_scene(self, *, feedback_force_scale: float) -> None:
        n: int = int(self.model.nx)
        world_x: float = float(n) * DH
        world_y: float = float(n) * DH
        world_z: float = float(n) * DH
        wall_t: float = WALL_THICKNESS_CELLS * DH
        radius: float = SPHERE_RADIUS
        z_floor: float = radius + 1.5 * DH

        builder: RigidModelBuilder = RigidModelBuilder(gravity=RIGID_GRAVITY_Z)
        wall_cfg: ShapeConfig = ShapeConfig(
            density=0.0,
            is_visible=False,
            is_solid=True,
            has_shape_collision=True,
        )

        def add_wall(
            label: str,
            center: tuple[float, float, float],
            half_extents: tuple[float, float, float],
        ) -> None:
            body: int = builder.add_body(position=center, label=label)
            builder.add_shape_box(
                body,
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=wall_cfg,
            )

        add_wall(
            "rigid_floor",
            (world_x * 0.5, world_y * 0.5, -wall_t * 0.5),
            (world_x * 0.5, world_y * 0.5, wall_t * 0.5),
        )
        add_wall(
            "rigid_ceiling",
            (world_x * 0.5, world_y * 0.5, world_z + wall_t * 0.5),
            (world_x * 0.5, world_y * 0.5, wall_t * 0.5),
        )
        add_wall(
            "rigid_xmin",
            (-wall_t * 0.5, world_y * 0.5, world_z * 0.5),
            (wall_t * 0.5, world_y * 0.5, world_z * 0.5),
        )
        add_wall(
            "rigid_xmax",
            (world_x + wall_t * 0.5, world_y * 0.5, world_z * 0.5),
            (wall_t * 0.5, world_y * 0.5, world_z * 0.5),
        )
        add_wall(
            "rigid_ymin",
            (world_x * 0.5, -wall_t * 0.5, world_z * 0.5),
            (world_x * 0.5, wall_t * 0.5, world_z * 0.5),
        )
        add_wall(
            "rigid_ymax",
            (world_x * 0.5, world_y + wall_t * 0.5, world_z * 0.5),
            (world_x * 0.5, wall_t * 0.5, world_z * 0.5),
        )

        heavy_cfg: ShapeConfig = ShapeConfig(
            density=HEAVY_SPHERE_DENSITY,
            is_visible=not TEXTURED_SPHERE_VISUALS_ENABLED,
            is_solid=True,
        )
        light_cfg: ShapeConfig = ShapeConfig(
            density=LIGHT_SPHERE_DENSITY,
            is_visible=not TEXTURED_SPHERE_VISUALS_ENABLED,
            is_solid=True,
        )
        heavy_center: tuple[float, float, float] = (world_x * 0.42, world_y * 0.38, z_floor)
        light_center: tuple[float, float, float] = (world_x * 0.42, world_y * 0.62, z_floor)

        self.heavy_body_id: int = builder.add_body(position=heavy_center, label="heavy_sphere")
        builder.add_shape_sphere(self.heavy_body_id, radius=radius, cfg=heavy_cfg)
        if TEXTURED_SPHERE_VISUALS_ENABLED:
            _add_textured_sphere_visual(builder, self.heavy_body_id, radius, 0)
        self.light_body_id: int = builder.add_body(position=light_center, label="light_sphere")
        builder.add_shape_sphere(self.light_body_id, radius=radius, cfg=light_cfg)
        if TEXTURED_SPHERE_VISUALS_ENABLED:
            _add_textured_sphere_visual(builder, self.light_body_id, radius, 1)

        self.rigid_domain: RigidDomain = RigidDomain(builder.finalize(device=self.model._device))
        self.rigid_domain.create_state()
        if self.viewer is not None:
            self.rigid_domain.model.setup_viewer(self.viewer)

        self.coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(self.domain, self.rigid_domain)
        self.coupling.add_body_sphere(body_idx=self.heavy_body_id, radius=radius)
        self.coupling.add_body_sphere(body_idx=self.light_body_id, radius=radius)
        self.coupling.set_rigid_dynamics_enabled(False)
        self.coupling.set_two_way_feedback_enabled(True, force_scale=feedback_force_scale)
        self.coupling.set_feedback_mode("momentum_exchange")

        print(
            f"  spheres: radius={radius}, heavy_density={HEAVY_SPHERE_DENSITY}, "
            f"light_density={LIGHT_SPHERE_DENSITY}, feedback_scale={feedback_force_scale}"
        )
        print(
            f"  empirical buoyancy: scale={self.buoyancy_force_scale}, "
            f"drag_xy={self.water_horizontal_drag_rate}, drag_z={self.water_vertical_drag_rate}"
        )
        print(f"  initial heavy={heavy_center}, light={light_center}, rigid_gz={RIGID_GRAVITY_Z}")

    def _ramp_gravity(self) -> None:
        target_lbm_gz: float = float(self.model.gravity_z)
        target_rigid_gz: float = RIGID_GRAVITY_Z
        self.model.gravity_z = 0.0
        self.rigid_domain.model.set_gravity((0.0, 0.0, 0.0))

        for step_index in range(GRAVITY_RAMP_STEPS):
            alpha: float = float(step_index + 1) / float(GRAVITY_RAMP_STEPS)
            self.model.gravity_z = alpha * target_lbm_gz
            self.rigid_domain.model.set_gravity((0.0, 0.0, alpha * target_rigid_gz))
            self._step_coupled()

        self.model.gravity_z = target_lbm_gz
        self.rigid_domain.model.set_gravity((0.0, 0.0, target_rigid_gz))
        wp.synchronize_device(self.model._device)
        print(f"  gravity ramp done: gz_lbm={self.model.gravity_z}, rigid_z={target_rigid_gz}")

    # ------------------------------------------------------------------
    # Step / render
    # ------------------------------------------------------------------

    def step(self) -> None:
        t0: float = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self._step_coupled()
        wp.synchronize_device(self.model._device)

        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self._print_status()

    def _step_coupled(self) -> None:
        self.coupling.step(self.sim_dt)
        self._apply_empirical_buoyancy_and_drag()
        self.rigid_domain.step(self.sim_dt)

    def _apply_empirical_buoyancy_and_drag(self) -> None:
        rho_np: np.ndarray = self.domain.state.density.numpy()
        self._apply_sphere_buoyancy_and_drag(self.heavy_body_id, HEAVY_SPHERE_DENSITY, rho_np)
        self._apply_sphere_buoyancy_and_drag(self.light_body_id, LIGHT_SPHERE_DENSITY, rho_np)

    def _apply_sphere_buoyancy_and_drag(
        self,
        body_id: int,
        sphere_density: float,
        rho_np: np.ndarray,
    ) -> None:
        position: np.ndarray = np.asarray(self.rigid_domain.state.get_body_position(body_id), dtype=np.float64)
        velocity: np.ndarray = np.asarray(self.rigid_domain.state.get_body_linear_velocity(body_id), dtype=np.float64)
        submerged_fraction: float = self._estimate_submerged_fraction(position, rho_np)
        mass: float = float(sphere_density) * self._sphere_volume

        buoyancy_z: float = (
            self.buoyancy_force_scale
            * RHO_WATER
            * self._sphere_volume
            * abs(RIGID_GRAVITY_Z)
            * submerged_fraction
        )
        drag_x: float = -self.water_horizontal_drag_rate * mass * submerged_fraction * float(velocity[0])
        drag_y: float = -self.water_horizontal_drag_rate * mass * submerged_fraction * float(velocity[1])
        drag_z: float = -self.water_vertical_drag_rate * mass * submerged_fraction * float(velocity[2])
        force: tuple[float, float, float, float, float, float] = (
            drag_x,
            drag_y,
            float(buoyancy_z + drag_z),
            0.0,
            0.0,
            0.0,
        )
        self.rigid_domain.state.add_body_force(body_id, force)
        self._last_submerged_by_body[body_id] = submerged_fraction
        self._last_extra_force_by_body[body_id] = (force[0], force[1], force[2])

    def _estimate_submerged_fraction(self, position: np.ndarray, rho_np: np.ndarray) -> float:
        water_samples: int = 0
        sample_count: int = len(BUOYANCY_SAMPLE_OFFSETS)
        for offset in BUOYANCY_SAMPLE_OFFSETS:
            sample_x: float = float(position[0] + offset[0] * SPHERE_RADIUS)
            sample_y: float = float(position[1] + offset[1] * SPHERE_RADIUS)
            sample_z: float = float(position[2] + offset[2] * SPHERE_RADIUS)
            i: int = int(np.clip(sample_x / DH, 0, N - 1))
            j: int = int(np.clip(sample_y / DH, 0, N - 1))
            k: int = int(np.clip(sample_z / DH, 0, N - 1))
            if float(rho_np[i, j, k]) > SSFR_THRESHOLD:
                water_samples += 1
        return float(water_samples) / float(sample_count)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid_domain.state.as_newton_state())

        if self.ssfr is not None and self.ssfr.available:
            state: LbmState = self.domain.state
            wp.launch(
                _mask_solid_density,
                dim=(N, N, N),
                inputs=[state.density, state.solid_phi, self.display_density],
                device=self.model._device,
            )
            self.ssfr.set_density_field(
                density=self.display_density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=DH,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )

        self.viewer.end_frame()

    def test_final(self) -> None:
        heavy_pos: np.ndarray = np.asarray(
            self.rigid_domain.state.get_body_position(self.heavy_body_id),
            dtype=np.float64,
        )
        light_pos: np.ndarray = np.asarray(
            self.rigid_domain.state.get_body_position(self.light_body_id),
            dtype=np.float64,
        )
        rho_np: np.ndarray = np.asarray(self.domain.state.density.numpy(), dtype=np.float64)
        if not np.all(np.isfinite(heavy_pos)):
            raise ValueError(f"heavy sphere position not finite: {heavy_pos}")
        if not np.all(np.isfinite(light_pos)):
            raise ValueError(f"light sphere position not finite: {light_pos}")
        if not np.all(np.isfinite(rho_np)):
            raise ValueError("density field contains non-finite values")

    def _print_status(self) -> None:
        state: LbmState = self.domain.state
        rho_np: np.ndarray = state.density.numpy()
        water: np.ndarray = rho_np > SSFR_THRESHOLD
        heavy_pos: np.ndarray = np.asarray(
            self.rigid_domain.state.get_body_position(self.heavy_body_id),
            dtype=np.float64,
        )
        light_pos: np.ndarray = np.asarray(
            self.rigid_domain.state.get_body_position(self.light_body_id),
            dtype=np.float64,
        )
        light_vel: np.ndarray = np.asarray(
            self.rigid_domain.state.get_body_linear_velocity(self.light_body_id),
            dtype=np.float64,
        )
        heavy_submerged: float = self._last_submerged_by_body.get(self.heavy_body_id, 0.0)
        light_submerged: float = self._last_submerged_by_body.get(self.light_body_id, 0.0)
        light_extra_force: tuple[float, float, float] = self._last_extra_force_by_body.get(
            self.light_body_id,
            (0.0, 0.0, 0.0),
        )
        if water.any():
            com: np.ndarray = np.argwhere(water).mean(axis=0)
            print(
                f"[t={self.sim_time:.1f}s] water={int(water.sum())} "
                f"COM=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) "
                f"heavy=({heavy_pos[0]:.2f},{heavy_pos[1]:.2f},{heavy_pos[2]:.2f}) "
                f"light=({light_pos[0]:.2f},{light_pos[1]:.2f},{light_pos[2]:.2f}) "
                f"sub=({heavy_submerged:.2f},{light_submerged:.2f}) "
                f"light_v=({light_vel[0]:+.3f},{light_vel[2]:+.3f}) "
                f"light_extra_f=({light_extra_force[0]:+.4g},{light_extra_force[2]:+.4g}) "
                f"sim={self._last_ms:.0f}ms",
                file=sys.stderr,
                flush=True,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument(
        "--feedback-force-scale",
        type=float,
        default=FEEDBACK_FORCE_SCALE,
        help="Multiplier for LBM momentum-exchange rigid feedback.",
    )
    parser.add_argument(
        "--buoyancy-force-scale",
        type=float,
        default=BUOYANCY_FORCE_SCALE,
        help="Multiplier for the empirical upward buoyancy helper.",
    )
    parser.add_argument(
        "--water-horizontal-drag-rate",
        type=float,
        default=WATER_HORIZONTAL_DRAG_RATE,
        help="Linear x/y velocity damping rate applied while a sphere is submerged.",
    )
    parser.add_argument(
        "--water-vertical-drag-rate",
        type=float,
        default=WATER_VERTICAL_DRAG_RATE,
        help="Linear z velocity damping rate applied while a sphere is submerged.",
    )
    parser.add_argument(
        "--water-drag-rate",
        type=float,
        default=None,
        help="Deprecated alias: set both horizontal and vertical submerged damping rates.",
    )
    return parser


def main() -> None:
    parser: argparse.ArgumentParser = create_parser()
    viewer: Any
    args: argparse.Namespace
    viewer, args = init_fluid_viewer(parser)
    water_horizontal_drag_rate: float = float(args.water_horizontal_drag_rate)
    water_vertical_drag_rate: float = float(args.water_vertical_drag_rate)
    if args.water_drag_rate is not None:
        water_drag_rate: float = float(args.water_drag_rate)
        water_horizontal_drag_rate = water_drag_rate
        water_vertical_drag_rate = water_drag_rate
    example: TrtDamBreakTwoSpheres = TrtDamBreakTwoSpheres(
        viewer,
        feedback_force_scale=float(args.feedback_force_scale),
        buoyancy_force_scale=float(args.buoyancy_force_scale),
        water_horizontal_drag_rate=water_horizontal_drag_rate,
        water_vertical_drag_rate=water_vertical_drag_rate,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
