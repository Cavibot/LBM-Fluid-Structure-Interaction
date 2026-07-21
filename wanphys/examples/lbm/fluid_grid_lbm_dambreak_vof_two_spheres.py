# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FREE VOF dam-break with two dynamic rigid spheres (FSI).

Uses ``lbm_backend=home_fp32`` + ``phase_mode=vof_sharp``. Rigid spheres are
rasterized into ``solid_phi`` each substep; the HOME-FREE fused kernel treats
``solid_phi < 0`` as moving walls. Fluid→rigid feedback uses the macro
approximation (distributions are not stored on the moment path).

Optional late-pool ``--height-eq`` (same IF ``φ→φ*`` regularizer as the
single-phase dam-break example). Arms after ``t=8``; body-adjacent IF keeps a
soft floor weight so sphere menisci slowly heal instead of freezing as pits.

Run:
    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \\
        --viewer gl --n 64

    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \\
        --viewer gl --n 48 --height-eq

Sphere trajectories append to ``sphere_traj.csv`` (override with
``--sphere-log PATH``, disable with ``--sphere-log ""``). Buffered writes;
use ``--sphere-log-every N`` to thin rows (default 10).

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import newton
import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.sphere_buoyancy_warp import (
    apply_sphere_buoyancy_forces_gpu,
    ensure_buoyancy_scratch,
)
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N: int = 64
DH: float = 0.02
LATTICE: str = "D3Q27"

TAU: float = 0.51
LAMBDA_TRT: float = 0.015
VOF_GAMMA: float = 1.5e-3
VOF_RHO_GAS: float = 1.0
VOF_EPSILON: float = 1.0e-3
RHO_LIQUID: float = 1.0

DAM_X_FRAC: float = 0.25
FILL_Z_FRAC: float = 0.5

RIGID_GRAVITY_Z: float = -1.0
SPHERE_RADIUS: float = 0.08
HEAVY_SPHERE_DENSITY: float = 1.35
LIGHT_SPHERE_DENSITY: float = 0.45

TEXTURED_SPHERE_VISUALS_ENABLED: bool = True
SPHERE_VISUAL_TEXTURE_SIZE: int = 256
SPHERE_VISUAL_MESH_LATITUDES: int = 32
SPHERE_VISUAL_MESH_LONGITUDES: int = 48

FEEDBACK_FORCE_SCALE: float = 6.0
BUOYANCY_FORCE_SCALE: float = 1.0
WATER_HORIZONTAL_DRAG_RATE: float = 4.0
WATER_VERTICAL_DRAG_RATE: float = 12.0
# Strong push + raw wet-fraction chatter bobbed floaters and dug meniscus pits.
FLUID_PUSH_RATE: float = 8.0
# After height-eq arms, further weaken horizontal fluid chase (pool is quiet).
LATE_POOL_PUSH_SCALE: float = 0.12
# Submerged EMA: light-sphere shell chatter was still ~0.3↔0.7 in status logs.
SUB_EMA_ALPHA: float = 0.05
SUB_DSUB_CAP: float = 0.015
WALL_THICKNESS_CELLS: float = 2.0

DEFAULT_SPHERE_LOG: str = "sphere_traj.csv"

# Sample just outside the sphere: interior cells are forced to gas by FSI mask.
BUOYANCY_SAMPLE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (1.25, 0.0, 0.0),
    (-1.25, 0.0, 0.0),
    (0.0, 1.25, 0.0),
    (0.0, -1.25, 0.0),
    (0.0, 0.0, 1.25),
    (0.0, 0.0, -1.25),
    (0.9, 0.9, 0.0),
    (0.9, -0.9, 0.0),
    (-0.9, 0.9, 0.0),
    (-0.9, -0.9, 0.0),
    (0.9, 0.0, 0.9),
    (0.9, 0.0, -0.9),
    (-0.9, 0.0, 0.9),
    (-0.9, 0.0, -0.9),
    (0.0, 0.9, 0.9),
    (0.0, 0.9, -0.9),
    (0.0, -0.9, 0.9),
    (0.0, -0.9, -0.9),
)

SSFR_THRESHOLD: float = 0.35
RAY_MARCH_STEPS: int = 800
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 12
GRAVITY_RAMP_STEPS: int = 40


# ---------------------------------------------------------------------------
# Visual helpers (shared style with Shan-Chen two-spheres example)
# ---------------------------------------------------------------------------


def _make_sphere_rotation_texture(sphere_index: int) -> np.ndarray:
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
        label=f"vof_sphere_{sphere_index}_rotation_visual",
    )


@wp.kernel
def _mask_solid_visual(
    density: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
) -> None:
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        density_out[i, j, k] = 0.0
    else:
        # SSFR threshold on liquid volume fraction × density.
        density_out[i, j, k] = density[i, j, k] * phi[i, j, k]


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class HomeVofDamBreakTwoSpheres:
    """HOME-FREE VOF dam-break with heavy + light dynamic spheres."""

    def __init__(
        self,
        viewer: Any,
        *,
        n: int = N,
        feedback_force_scale: float = FEEDBACK_FORCE_SCALE,
        buoyancy_force_scale: float = BUOYANCY_FORCE_SCALE,
        water_horizontal_drag_rate: float = WATER_HORIZONTAL_DRAG_RATE,
        water_vertical_drag_rate: float = WATER_VERTICAL_DRAG_RATE,
        sphere_log_path: str | Path | None = DEFAULT_SPHERE_LOG,
        sphere_log_every: int = 10,
        enable_height_eq: bool = False,
        enable_moment_quant: bool = False,
    ) -> None:
        self.viewer: Any = viewer
        if isinstance(self.viewer, FluidViewerGL):
            self.viewer._paused = True

        self._n = int(n)
        self._enable_height_eq = bool(enable_height_eq)
        self._enable_moment_quant = bool(enable_moment_quant)
        self._height_eq_armed = False
        self._height_eq_arm_after_t = 8.0
        self._last_height_eq: dict | None = None
        n_ref = 48
        gravity = -0.0020 * (float(n_ref) / float(self._n))
        self._substeps = max(SIM_SUBSTEPS, 12)
        self._sphere_log_every = max(1, int(sphere_log_every))
        self._sphere_log_fp: TextIO | None = None
        self._sphere_log_path: Path | None = None
        if sphere_log_path is not None and str(sphere_log_path).strip():
            self._sphere_log_path = Path(sphere_log_path).expanduser().resolve()
            self._sphere_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._sphere_log_fp = self._sphere_log_path.open("w", encoding="utf-8", newline="")
            self._sphere_log_fp.write(
                "frame,t,phase,"
                "heavy_x,heavy_y,heavy_z,heavy_vx,heavy_vy,heavy_vz,heavy_sub,"
                "light_x,light_y,light_z,light_vx,light_vy,light_vz,light_sub,"
                "heavy_fx,heavy_fy,heavy_fz,light_fx,light_fy,light_fz\n"
            )
            # No flush here — buffered I/O; flush on close / periodic.
            print(f"  sphere traj log → {self._sphere_log_path} (every {self._sphere_log_every} frame)")

        self.model = LbmModel(
            fluid_grid_res=(self._n, self._n, self._n),
            fluid_grid_cell_size=DH,
            lattice=LATTICE,
            tau=TAU,
            G=0.0,
            phase_mode="vof_sharp",
            lbm_backend="home_fp32",
            vof_rho_gas=VOF_RHO_GAS,
            vof_epsilon=VOF_EPSILON,
            vof_gamma=VOF_GAMMA,
            vof_kappa_smooth=2,
            vof_wall_wetting=0.0,
            vof_home_fill_empty=False,
            vof_home_wall_eq=True,
            vof_seal_fg=True,
            vof_quiet_fill=False,
            vof_quiet_fill_rate=0.35,
            vof_quiet_fill_u_max=0.025,
            vof_orphan_reabsorb=False,
            vof_orphan_max_cells=max(96, self._n),
            vof_orphan_height_margin=3,
            vof_bubble_pressure=False,
            vof_bubble_disjoint=False,
            vof_height_eq=False,
            vof_height_eq_rate=0.025,
            vof_height_eq_u_max=0.05,
            vof_height_eq_dh_cap=0.03,
            vof_height_eq_every=24,
            vof_home_moment_quant=self._enable_moment_quant,
            vof_home_moment_quant_dither=True,
            lambda_trt=LAMBDA_TRT,
            initial_density=RHO_LIQUID,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=gravity,
        )
        self.domain = LbmDomain(self.model)
        self.domain.create_state()
        self._late_pool = None  # no host level ON

        self.sim_dt = FRAME_DT / float(self._substeps)
        self.sim_time = 0.0
        self.frame_count = 0
        self._last_ms = 0.0
        self.buoyancy_force_scale = float(buoyancy_force_scale)
        self.water_horizontal_drag_rate = float(water_horizontal_drag_rate)
        self.water_vertical_drag_rate = float(water_vertical_drag_rate)
        self._sphere_volume = (4.0 / 3.0) * math.pi * SPHERE_RADIUS**3
        self._last_submerged_by_body: dict[int, float] = {}
        self._last_extra_force_by_body: dict[int, tuple[float, float, float]] = {}
        self._buoyancy_scratch: dict | None = None

        self._init_fluid()
        self._init_rigid_scene(feedback_force_scale=float(feedback_force_scale))
        self._log_spheres(phase="init")
        self._ramp_gravity()
        self._log_spheres(phase="post_ramp")

        self.display_density = wp.zeros(
            (self._n, self._n, self._n), dtype=float, device=self.model._device
        )
        self.ssfr: ScreenSpaceFluidRenderer | None = None
        if isinstance(self.viewer, FluidViewerGL):
            ssfr = ScreenSpaceFluidRenderer(
                viewer=self.viewer,
                max_particles=1,
                particle_radius=0.01,
                device=self.model._device,
            )
            self.ssfr = ssfr
            self.viewer.register_post_render_callback(lambda v: ssfr.render(v))

        print(
            f"HOME-VOF dam-break two spheres: {self._n}^3, tau={TAU}, "
            f"gz={gravity:.5f}, gamma={VOF_GAMMA}, substeps={self._substeps}, "
            f"height_eq={self._enable_height_eq} "
            f"(arm_t>={self._height_eq_arm_after_t}), "
            f"moment_quant={self._enable_moment_quant}"
        )
        print("Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom")

    def _init_fluid(self) -> None:
        dam_x = int(self._n * DAM_X_FRAC)
        fill_z = int(self._n * FILL_Z_FRAC)
        state = self.domain.state
        home = self.domain.solver._home_fp32
        assert home is not None
        home.seed_dam_break(state, dam_x=dam_x, fill_z=fill_z, rho_liquid=RHO_LIQUID)
        out = self.domain._state_out
        home.sync_to_state(out)
        for name in ("solid_phi", "solid_body_id", "vel_solid_u", "vel_solid_v", "vel_solid_w"):
            wp.copy(getattr(out, name), getattr(state, name))
        self.domain.solver._vof_sharp.update_visual_field(
            state, self._n, self._n, self._n
        )
        wp.synchronize_device(self.model._device)
        ctype = state.cell_type.numpy()
        print(
            f"  liquid={int((ctype == 2).sum())} interface={int((ctype == 1).sum())} "
            f"gas={int((ctype == 0).sum())} dam_x={dam_x} fill_z={fill_z}"
        )
        if self._late_pool is not None:
            self._late_pool.set_reference_volume_from_state(state)

    def _init_rigid_scene(self, *, feedback_force_scale: float) -> None:
        n = self._n
        world_x = float(n) * DH
        world_y = float(n) * DH
        world_z = float(n) * DH
        wall_t = WALL_THICKNESS_CELLS * DH
        radius = SPHERE_RADIUS
        z_floor = radius + 1.5 * DH

        builder = RigidModelBuilder(gravity=RIGID_GRAVITY_Z)
        wall_cfg = ShapeConfig(
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
            body = builder.add_body(position=center, label=label)
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

        heavy_cfg = ShapeConfig(
            density=HEAVY_SPHERE_DENSITY,
            is_visible=not TEXTURED_SPHERE_VISUALS_ENABLED,
            is_solid=True,
        )
        light_cfg = ShapeConfig(
            density=LIGHT_SPHERE_DENSITY,
            is_visible=not TEXTURED_SPHERE_VISUALS_ENABLED,
            is_solid=True,
        )
        # Place just ahead of the dam front so the bore hits quickly.
        heavy_center = (world_x * 0.32, world_y * 0.38, z_floor)
        light_center = (world_x * 0.32, world_y * 0.62, z_floor)

        self.heavy_body_id = builder.add_body(position=heavy_center, label="heavy_sphere")
        builder.add_shape_sphere(self.heavy_body_id, radius=radius, cfg=heavy_cfg)
        if TEXTURED_SPHERE_VISUALS_ENABLED:
            _add_textured_sphere_visual(builder, self.heavy_body_id, radius, 0)

        self.light_body_id = builder.add_body(position=light_center, label="light_sphere")
        builder.add_shape_sphere(self.light_body_id, radius=radius, cfg=light_cfg)
        if TEXTURED_SPHERE_VISUALS_ENABLED:
            _add_textured_sphere_visual(builder, self.light_body_id, radius, 1)

        self.rigid_domain = RigidDomain(builder.finalize(device=self.model._device))
        self.rigid_domain.create_state()
        if self.viewer is not None and hasattr(self.viewer, "set_model"):
            self.rigid_domain.model.setup_viewer(self.viewer)

        self.coupling = GridLbmRigidCoupling(self.domain, self.rigid_domain)
        self.coupling.add_body_sphere(body_idx=self.heavy_body_id, radius=radius)
        self.coupling.add_body_sphere(body_idx=self.light_body_id, radius=radius)
        # Example advances rigid after empirical buoyancy; coupling only rasters + fluid.
        self.coupling.set_rigid_dynamics_enabled(False)
        self.coupling.set_two_way_feedback_enabled(True, force_scale=feedback_force_scale)
        # home_fp32 auto-falls back to approx; set explicitly for clarity.
        self.coupling.set_feedback_mode("approx")

        print(
            f"  spheres: r={radius}, heavy_ρ={HEAVY_SPHERE_DENSITY}, "
            f"light_ρ={LIGHT_SPHERE_DENSITY}, feedback={feedback_force_scale}"
        )
        print(
            f"  buoyancy={self.buoyancy_force_scale}, "
            f"drag_xy={self.water_horizontal_drag_rate}, "
            f"drag_z={self.water_vertical_drag_rate}"
        )
        print(f"  initial heavy={heavy_center}, light={light_center}")

    def _ramp_gravity(self) -> None:
        target_lbm_gz = float(self.model.gravity_z)
        target_rigid_gz = RIGID_GRAVITY_Z
        self.model.gravity_z = 0.0
        self.rigid_domain.model.set_gravity((0.0, 0.0, 0.0))

        for step_index in range(GRAVITY_RAMP_STEPS):
            alpha = float(step_index + 1) / float(GRAVITY_RAMP_STEPS)
            self.model.gravity_z = alpha * target_lbm_gz
            self.rigid_domain.model.set_gravity((0.0, 0.0, alpha * target_rigid_gz))
            self._step_coupled()

        self.model.gravity_z = target_lbm_gz
        self.rigid_domain.model.set_gravity((0.0, 0.0, target_rigid_gz))
        wp.synchronize_device(self.model._device)
        if self._late_pool is not None:
            self._late_pool.set_reference_volume_from_state(self.domain.state)
        print(f"  gravity ramp done: gz_lbm={self.model.gravity_z}, rigid_z={target_rigid_gz}")

    def step(self) -> None:
        t0 = time.perf_counter()
        home = self.domain.solver._home_fp32
        if (
            self._enable_height_eq
            and home is not None
            and self.sim_time >= self._height_eq_arm_after_t
            and not self._height_eq_armed
        ):
            self.model.vof_height_eq = True
            self._height_eq_armed = True
            print(
                f"[height-eq ON] t={self.sim_time:.1f}s "
                f"GPU IF φ→φ* α={self.model.vof_height_eq_rate} "
                f"|Δφ|≤{self.model.vof_height_eq_dh_cap} "
                f"every={self.model.vof_height_eq_every} "
                f"(pool plane; soft fade near rigid)",
                file=sys.stderr,
                flush=True,
            )
        for _ in range(self._substeps):
            self._step_coupled()
        # Sync submerged/forces only when status/CSV needs them (not every substep).
        next_frame = self.frame_count + 1
        need_log = self._sphere_log_fp is not None and (
            next_frame % self._sphere_log_every == 0
        )
        if need_log or next_frame % 30 == 0:
            self._sync_buoyancy_submerged()
        if home is not None and self.model.vof_height_eq:
            self._last_height_eq = dict(getattr(home, "_last_height_eq_stats", {}) or {})
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1
        self._log_spheres(phase="run")
        if self.frame_count % 30 == 0:
            self._print_status()

    def _log_spheres(self, *, phase: str) -> None:
        """Append one CSV row (buffered; no per-row flush)."""
        fp = self._sphere_log_fp
        if fp is None:
            return
        if phase == "run" and (self.frame_count % self._sphere_log_every) != 0:
            return
        heavy_pos = np.asarray(
            self.rigid_domain.state.get_body_position(self.heavy_body_id),
            dtype=np.float64,
        )
        light_pos = np.asarray(
            self.rigid_domain.state.get_body_position(self.light_body_id),
            dtype=np.float64,
        )
        heavy_vel = np.asarray(
            self.rigid_domain.state.get_body_linear_velocity(self.heavy_body_id),
            dtype=np.float64,
        )
        light_vel = np.asarray(
            self.rigid_domain.state.get_body_linear_velocity(self.light_body_id),
            dtype=np.float64,
        )
        heavy_sub = float(self._last_submerged_by_body.get(self.heavy_body_id, 0.0))
        light_sub = float(self._last_submerged_by_body.get(self.light_body_id, 0.0))
        heavy_f = self._last_extra_force_by_body.get(
            self.heavy_body_id, (0.0, 0.0, 0.0)
        )
        light_f = self._last_extra_force_by_body.get(
            self.light_body_id, (0.0, 0.0, 0.0)
        )
        fp.write(
            f"{self.frame_count},{self.sim_time:.6f},{phase},"
            f"{heavy_pos[0]:.8f},{heavy_pos[1]:.8f},{heavy_pos[2]:.8f},"
            f"{heavy_vel[0]:.8f},{heavy_vel[1]:.8f},{heavy_vel[2]:.8f},{heavy_sub:.6f},"
            f"{light_pos[0]:.8f},{light_pos[1]:.8f},{light_pos[2]:.8f},"
            f"{light_vel[0]:.8f},{light_vel[1]:.8f},{light_vel[2]:.8f},{light_sub:.6f},"
            f"{heavy_f[0]:.8f},{heavy_f[1]:.8f},{heavy_f[2]:.8f},"
            f"{light_f[0]:.8f},{light_f[1]:.8f},{light_f[2]:.8f}\n"
        )
        if phase != "run" or (self.frame_count % 60) == 0:
            fp.flush()

    def close(self) -> None:
        if self._sphere_log_fp is not None:
            self._sphere_log_fp.flush()
            self._sphere_log_fp.close()
            self._sphere_log_fp = None
            if self._sphere_log_path is not None:
                print(f"  sphere traj closed: {self._sphere_log_path}")

    def _step_coupled(self) -> None:
        self.coupling.step(self.sim_dt)
        self._apply_empirical_buoyancy_and_drag()
        self.rigid_domain.step(self.sim_dt)

    def _apply_empirical_buoyancy_and_drag(self) -> None:
        state = self.domain.state
        rigid = self.rigid_domain.state
        self._buoyancy_scratch = ensure_buoyancy_scratch(
            device=self.model._device,
            offsets_xyz=BUOYANCY_SAMPLE_OFFSETS,
            body_ids=(self.heavy_body_id, self.light_body_id),
            densities=(HEAVY_SPHERE_DENSITY, LIGHT_SPHERE_DENSITY),
            scratch=self._buoyancy_scratch,
        )
        vel_scale = DH / max(self.sim_dt, 1.0e-12)
        push = FLUID_PUSH_RATE
        if self._height_eq_armed:
            push *= LATE_POOL_PUSH_SCALE
        apply_sphere_buoyancy_forces_gpu(
            phi=state.phi,
            cell=state.cell_type,
            solid=state.solid_phi,
            ux=state.velocity_x,
            uy=state.velocity_y,
            uz=state.velocity_z,
            body_q=rigid.body_q,
            body_qd=rigid.body_qd,
            body_f_apply=rigid.apply_body_forces,
            radius=SPHERE_RADIUS,
            dh=DH,
            nx=self._n,
            ny=self._n,
            nz=self._n,
            scratch=self._buoyancy_scratch,
            volume=self._sphere_volume,
            rho_liquid=RHO_LIQUID,
            gravity_abs=abs(RIGID_GRAVITY_Z),
            buoyancy_scale=self.buoyancy_force_scale,
            push_rate=push,
            drag_xy=self.water_horizontal_drag_rate,
            drag_z=self.water_vertical_drag_rate,
            vel_scale=vel_scale,
            ema_alpha=SUB_EMA_ALPHA,
            dsub_cap=SUB_DSUB_CAP,
            sync_submerged=False,
        )

    def _sync_buoyancy_submerged(self) -> None:
        scratch = self._buoyancy_scratch
        if scratch is None:
            return
        sub = scratch["submerged"].numpy()
        ids = scratch["body_ids_host"]
        for i, body_id in enumerate(ids):
            self._last_submerged_by_body[int(body_id)] = float(sub[i])
        # Forces stay on device; log zeros unless we sync (optional).
        forces = scratch["forces"].numpy()
        for i, body_id in enumerate(ids):
            f = forces[i]
            self._last_extra_force_by_body[int(body_id)] = (
                float(f[0]),
                float(f[1]),
                float(f[2]),
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid_domain.state.as_newton_state())

        if self.ssfr is not None and self.ssfr.available:
            state: LbmState = self.domain.state
            wp.launch(
                _mask_solid_visual,
                dim=(self._n, self._n, self._n),
                inputs=[state.density, state.phi, state.solid_phi, self.display_density],
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
        heavy_pos = np.asarray(
            self.rigid_domain.state.get_body_position(self.heavy_body_id),
            dtype=np.float64,
        )
        light_pos = np.asarray(
            self.rigid_domain.state.get_body_position(self.light_body_id),
            dtype=np.float64,
        )
        rho_np = np.asarray(self.domain.state.density.numpy(), dtype=np.float64)
        if not np.all(np.isfinite(heavy_pos)):
            raise ValueError(f"heavy sphere position not finite: {heavy_pos}")
        if not np.all(np.isfinite(light_pos)):
            raise ValueError(f"light sphere position not finite: {light_pos}")
        if not np.all(np.isfinite(rho_np)):
            raise ValueError("density field contains non-finite values")

    def _print_status(self) -> None:
        # No full-field D2H — keep host logs off the sim critical path.
        heavy_pos = np.asarray(
            self.rigid_domain.state.get_body_position(self.heavy_body_id),
            dtype=np.float64,
        )
        light_pos = np.asarray(
            self.rigid_domain.state.get_body_position(self.light_body_id),
            dtype=np.float64,
        )
        light_vel = np.asarray(
            self.rigid_domain.state.get_body_linear_velocity(self.light_body_id),
            dtype=np.float64,
        )
        heq = ""
        if self._last_height_eq:
            heq = (
                f" H*={self._last_height_eq.get('H_star', 0):.3f}"
                f" φ*={self._last_height_eq.get('phi_star', 0):.3f}"
                f" nIF={int(self._last_height_eq.get('n_if', 0))}"
                f" skipB={int(self._last_height_eq.get('n_body_skip', 0))}"
            )
        print(
            f"[t={self.sim_time:.1f}s] "
            f"heavy=({heavy_pos[0]:.2f},{heavy_pos[1]:.2f},{heavy_pos[2]:.2f}) "
            f"light=({light_pos[0]:.2f},{light_pos[1]:.2f},{light_pos[2]:.2f}) "
            f"sub=({self._last_submerged_by_body.get(self.heavy_body_id, 0.0):.2f},"
            f"{self._last_submerged_by_body.get(self.light_body_id, 0.0):.2f}) "
            f"light_v=({light_vel[0]:+.3f},{light_vel[2]:+.3f}) "
            f"sim={self._last_ms:.0f}ms{heq}",
            file=sys.stderr,
            flush=True,
        )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--n", type=int, default=N, help="Grid resolution (N³).")
    parser.add_argument(
        "--feedback-force-scale",
        type=float,
        default=FEEDBACK_FORCE_SCALE,
        help="Multiplier for LBM→rigid macro feedback.",
    )
    parser.add_argument(
        "--buoyancy-force-scale",
        type=float,
        default=BUOYANCY_FORCE_SCALE,
        help="Empirical upward buoyancy helper scale.",
    )
    parser.add_argument(
        "--water-horizontal-drag-rate",
        type=float,
        default=WATER_HORIZONTAL_DRAG_RATE,
    )
    parser.add_argument(
        "--water-vertical-drag-rate",
        type=float,
        default=WATER_VERTICAL_DRAG_RATE,
    )
    parser.add_argument(
        "--sphere-log",
        type=str,
        default=DEFAULT_SPHERE_LOG,
        help="CSV path for real-time sphere trajectories (empty string disables).",
    )
    parser.add_argument(
        "--sphere-log-every",
        type=int,
        default=10,
        help="Write a CSV row every N viewer frames (default 10).",
    )
    parser.add_argument(
        "--height-eq",
        action="store_true",
        help=(
            "Enable solver IF-φ leveling on the pool plane (same as dam-break "
            "--height-eq). Arms after t=8; soft-fades near rigid spheres."
        ),
    )
    parser.add_argument(
        "--moment-quant",
        action="store_true",
        help=(
            "Persistent 16-bit HOME moment SoT (5×uint32/cell). "
            "Fused loads quant → float work → re-pack; drops moment ping-pong (~25% moment bytes)."
        ),
    )
    return parser


def main() -> None:
    parser = create_parser()
    viewer, args = init_fluid_viewer(parser)
    log_path = str(args.sphere_log).strip()
    example = HomeVofDamBreakTwoSpheres(
        viewer,
        n=int(args.n),
        feedback_force_scale=float(args.feedback_force_scale),
        buoyancy_force_scale=float(args.buoyancy_force_scale),
        water_horizontal_drag_rate=float(args.water_horizontal_drag_rate),
        water_vertical_drag_rate=float(args.water_vertical_drag_rate),
        sphere_log_path=log_path if log_path else None,
        sphere_log_every=int(args.sphere_log_every),
        enable_height_eq=bool(args.height_eq),
        enable_moment_quant=bool(args.moment_quant),
    )
    try:
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
