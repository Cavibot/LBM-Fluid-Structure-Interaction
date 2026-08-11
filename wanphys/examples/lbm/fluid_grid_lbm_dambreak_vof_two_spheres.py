# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FREE VOF dam-break with two dynamic rigid spheres (FSI).

Normal-physics default (uniform liquid ρ≈1):
  SDF raster → Eq.24 walls → link ME (Eq.32: impact / torque) →
  φ-volume Archimedes Fz=s·ρ·V·|g| (correct incompressible buoyancy) → XPBD.

Do **not** use hydrostatic ``ρ(z)`` for everyday runs — that is a weakly-
compressible trick so ME can fake vertical lift. Opt-in only via ``--hydro-rho``
(turns Archimedes off). Spheres start dry ahead of the dam.

``--no-archimedes`` = no vertical buoyancy (floor skate / contact perch).
``--showcase-fsi`` / ``--me-drag`` are separate non-default plugins.

Run:
    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \\
        --viewer gl --n 64
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

from wanphys._src.fluid.fluid_grid.coupling import (
    GridLbmRigidCoupling,
    lattice_gravity_to_world,
    recommended_me_force_scale,
)
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmState
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.generic import (
    make_home_vof_model,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.phi_volume_buoyancy_warp import (
    apply_phi_volume_buoyancy_gpu,
    ensure_phi_volume_scratch,
    fibonacci_shell_offsets,
)
from wanphys.examples.lbm._home_vof_empirical_sphere_fsi import (
    EmpiricalSphereFsiConfig,
    EmpiricalSphereFsiPlugin,
    me_path_linear_drag_config,
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

# Same dam fill as the other HOME-VOF dam-break examples.
DAM_X_FRAC: float = 0.25
FILL_Z_FRAC: float = 0.5

RIGID_GRAVITY_Z: float = -1.0  # placeholder; overwritten from matched lattice g
SPHERE_RADIUS: float = 0.08
# Strong density contrast → clear sink vs float after the bore.
HEAVY_SPHERE_DENSITY: float = 1.8
LIGHT_SPHERE_DENSITY: float = 0.30

TEXTURED_SPHERE_VISUALS_ENABLED: bool = True
SPHERE_VISUAL_TEXTURE_SIZE: int = 256
SPHERE_VISUAL_MESH_LATITUDES: int = 32
SPHERE_VISUAL_MESH_LONGITUDES: int = 48

# Legacy constant kept for docs / manual overrides.
SHOWCASE_LEGACY_FORCE_SCALE: float = 6.0
# Empirical FSI (showcase plugin — not part of generic VOF / coupling core).
BUOYANCY_FORCE_SCALE: float = 1.0
# φ-volume Archimedes (default on): incompressible Fz = scale * s * ρ * V * |g|.
# scale=1 is physical Archimedes; >1 was only for exaggerated demo bobbing.
ARCHIMEDES_SCALE: float = 1.0
WATER_HORIZONTAL_DRAG_RATE: float = 4.0
WATER_VERTICAL_DRAG_RATE: float = 12.0
FLUID_PUSH_RATE: float = 8.0
LATE_POOL_PUSH_SCALE: float = 0.12
# Optional non-paper ME-path drag (off by default; enable with --me-drag).
ME_PATH_DRAG_XY: float = 1.5
ME_PATH_DRAG_Z: float = 4.0
ME_PATH_DRAG_ANG: float = 2.0
# Soft contact under matched |g|~O(15) so buoyant bodies can leave the floor.
SPHERE_CONTACT_KE: float = 120.0
SPHERE_CONTACT_KD: float = 2800.0
SPHERE_CONTACT_RESTITUTION: float = 0.0
# Lower friction reduces light-on-heavy "perch" when vertical ME is weak.
SPHERE_FRICTION_MU: float = 0.22
SUB_EMA_ALPHA: float = 0.08
SUB_DSUB_CAP: float = 0.04
WALL_THICKNESS_CELLS: float = 2.0

# Opt-in Path A only: soft-maintain ρ(z) for Eq.32 ME lift experiments.
# Off by default — not normal incompressible water.
HYDRO_RHO_RATE: float = 0.08
HYDRO_RHO_EVERY: int = 4

DEFAULT_SPHERE_LOG: str = "sphere_traj.csv"

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


def _prune_static_static_contact_pairs(model: Any) -> None:
    """Keep only pairs involving at least one dynamic body (drop wall–wall)."""
    pairs = getattr(model, "shape_contact_pairs", None)
    if pairs is None:
        return
    backend = model._newton_backend
    mass = backend.body_mass.numpy()
    shape_body = backend.shape_body.numpy()
    raw = pairs.numpy()
    kept: list[tuple[int, int]] = []
    for a, b in raw:
        ba = int(shape_body[int(a)])
        bb = int(shape_body[int(b)])
        if float(mass[ba]) > 0.0 or float(mass[bb]) > 0.0:
            kept.append((int(a), int(b)))
    if len(kept) == len(raw):
        return
    device = pairs.device
    model.shape_contact_pairs = wp.array(kept, dtype=wp.vec2i, device=device)
    # Invalidate cached collision pipelines that captured the old pair buffer.
    from wanphys._src.collision.pipeline import CollisionPipeline

    CollisionPipeline._rigid_pipeline_cache.clear()


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
        feedback_force_scale: float | None = None,
        buoyancy_force_scale: float = BUOYANCY_FORCE_SCALE,
        water_horizontal_drag_rate: float = WATER_HORIZONTAL_DRAG_RATE,
        water_vertical_drag_rate: float = WATER_VERTICAL_DRAG_RATE,
        sphere_log_path: str | Path | None = DEFAULT_SPHERE_LOG,
        sphere_log_every: int = 10,
        enable_height_eq: bool = False,
        enable_moment_quant: bool = False,
        showcase_fsi: bool = False,
        me_in_fused: bool = True,
        me_drag: bool = False,
        me_drag_xy: float = ME_PATH_DRAG_XY,
        me_drag_z: float = ME_PATH_DRAG_Z,
        me_drag_ang: float = ME_PATH_DRAG_ANG,
        archimedes: bool = True,
        archimedes_scale: float = ARCHIMEDES_SCALE,
        hydro_rho: bool = False,
        hydro_rho_rate: float = HYDRO_RHO_RATE,
        hydro_rho_every: int = HYDRO_RHO_EVERY,
    ) -> None:
        self.viewer: Any = viewer
        if isinstance(self.viewer, FluidViewerGL):
            self.viewer._paused = True

        self._n = int(n)
        self._showcase_fsi = bool(showcase_fsi)
        self._me_drag = bool(me_drag) and not self._showcase_fsi
        self._me_drag_xy = float(me_drag_xy)
        self._me_drag_z = float(me_drag_z)
        self._me_drag_ang = float(me_drag_ang)
        # Path A: maintain ρ(z) for ME buoyancy; turn off φ-volume Archimedes.
        self._hydro_rho = bool(hydro_rho) and not self._showcase_fsi
        self._hydro_rho_rate = float(hydro_rho_rate)
        self._hydro_rho_every = max(1, int(hydro_rho_every))
        # φ-volume Archimedes when not using showcase / path-A hydro maintain.
        self._archimedes = (
            bool(archimedes) and not self._showcase_fsi and not self._hydro_rho
        )
        self._archimedes_scale = float(archimedes_scale)
        self._phi_buoy_scratch: dict | None = None
        self._phi_buoy_offsets = fibonacci_shell_offsets(48, radii=(1.08, 1.18))
        self._enable_height_eq = bool(enable_height_eq)
        self._enable_moment_quant = bool(enable_moment_quant)
        self._height_eq_armed = False
        self._height_eq_arm_after_t = 8.0
        self._last_height_eq: dict | None = None
        n_ref = 48
        gravity = -0.0020 * (float(n_ref) / float(self._n))
        self._substeps = max(SIM_SUBSTEPS, 12)
        self.sim_dt = FRAME_DT / float(self._substeps)
        # Open units: same physical g for lattice hydrostatics and rigid weight.
        self._lbm_gravity_z = float(gravity)
        self._rigid_gravity_z = lattice_gravity_to_world(gravity, DH, self.sim_dt)
        self._g_matched = self._rigid_gravity_z
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
                "heavy_fx,heavy_fy,heavy_fz,light_fx,light_fy,light_fz,"
                "mass,mass0,mass_rel\n"
            )
            # No flush here — buffered I/O; flush on close / periodic.
            print(f"  sphere traj log → {self._sphere_log_path} (every {self._sphere_log_every} frame)")

        # Core: Eq.24. Showcase may prefer eq-wall for old look.
        use_wall_eq = bool(self._showcase_fsi)
        self.model = make_home_vof_model(
            fluid_grid_res=(self._n, self._n, self._n),
            fluid_grid_cell_size=DH,
            lattice=LATTICE,
            tau=TAU,
            gravity_z=gravity,
            vof_gamma=VOF_GAMMA,
            lambda_trt=LAMBDA_TRT,
            initial_density=RHO_LIQUID,
            vof_rho_gas=VOF_RHO_GAS,
            vof_epsilon=VOF_EPSILON,
            vof_home_wall_eq=use_wall_eq,
            vof_orphan_max_cells=max(96, self._n),
            vof_height_eq_rate=0.025,
            vof_height_eq_u_max=0.05,
            vof_height_eq_dh_cap=0.03,
            vof_height_eq_every=24,
            vof_hydrostatic_rho=self._hydro_rho,
            vof_hydrostatic_rho_rate=self._hydro_rho_rate,
            vof_hydrostatic_rho_every=self._hydro_rho_every,
            vof_home_moment_quant=self._enable_moment_quant,
            vof_home_moment_quant_dither=True,
        )
        self.domain = LbmDomain(self.model)
        self.domain.create_state()
        self._late_pool = None  # no host level ON

        self.sim_dt = FRAME_DT / float(self._substeps)
        self.sim_time = 0.0
        self.frame_count = 0
        self._last_ms = 0.0
        self._sphere_volume = (4.0 / 3.0) * math.pi * SPHERE_RADIUS**3
        self._last_submerged_by_body: dict[int, float] = {}
        self._last_extra_force_by_body: dict[int, tuple[float, float, float]] = {}
        self._last_liquid_mass: float = 0.0
        self._liquid_mass0: float = 0.0
        self._empirical_fsi: EmpiricalSphereFsiPlugin | None = None
        if self._showcase_fsi:
            self._empirical_cfg = EmpiricalSphereFsiConfig(
                buoyancy_scale=float(buoyancy_force_scale),
                push_rate=FLUID_PUSH_RATE,
                drag_xy=float(water_horizontal_drag_rate),
                drag_z=float(water_vertical_drag_rate),
                late_pool_push_scale=LATE_POOL_PUSH_SCALE,
                ema_alpha=SUB_EMA_ALPHA,
                dsub_cap=SUB_DSUB_CAP,
            )
        elif self._me_drag:
            self._empirical_cfg = me_path_linear_drag_config(
                drag_xy=self._me_drag_xy,
                drag_z=self._me_drag_z,
                drag_ang=self._me_drag_ang,
                ema_alpha=SUB_EMA_ALPHA,
                dsub_cap=SUB_DSUB_CAP,
            )
        else:
            self._empirical_cfg = EmpiricalSphereFsiConfig(
                buoyancy_scale=0.0,
                push_rate=0.0,
                drag_xy=0.0,
                drag_z=0.0,
            )

        if feedback_force_scale is None:
            feedback_scale = recommended_me_force_scale(
                DH,
                self.sim_dt,
                rho_fluid=RHO_LIQUID,
                match_rigid_g=True,
            )
        else:
            feedback_scale = float(feedback_force_scale)
        self._feedback_force_scale = feedback_scale

        # Core research path: stream-time ME inside fused solid pulls.
        self.model.vof_home_me_in_fused = bool(me_in_fused)

        self._init_fluid()
        self._init_rigid_scene(feedback_force_scale=feedback_scale)
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
            f"gz_lbm={gravity:.5f}, gz_rigid={self._rigid_gravity_z:.4g}, "
            f"gamma={VOF_GAMMA}, substeps={self._substeps}, "
            f"feedback=ME, me_conv={self._feedback_force_scale:.4g} (Eq.32 F=-dj Ladd, rho*dh^4/dt^2), "
            f"me_in_fused={'on' if self.model.vof_home_me_in_fused else 'off'}, "
            f"me_drag={'on' if self._me_drag else 'off'}, "
            f"archimedes={'on' if self._archimedes else 'off'}"
            f"(x{self._archimedes_scale:g}), "
            f"hydro_rho={'on' if self._hydro_rho else 'off'}"
            f"(α={self._hydro_rho_rate:g}/every={self._hydro_rho_every}), "
            f"showcase_fsi={'on' if self._showcase_fsi else 'off'}, "
            f"wall_eq={use_wall_eq}, "
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
        # Uniform ρ (normal physics). Hydrostatic ρ(z) only via --hydro-rho maintain.
        home.seed_dam_break(
            state,
            dam_x=dam_x,
            fill_z=fill_z,
            rho_liquid=RHO_LIQUID,
            hydrostatic=False,
        )
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
        self._liquid_mass0 = self._sample_liquid_mass()
        self._last_liquid_mass = self._liquid_mass0
        print(f"  mass0={self._liquid_mass0:.1f} (wet Σmass at seed)")
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

        builder = RigidModelBuilder(gravity=self._rigid_gravity_z)
        wall_cfg = ShapeConfig(
            density=0.0,
            is_visible=False,
            is_solid=True,
            has_shape_collision=True,
            mu=SPHERE_FRICTION_MU,
            ke=SPHERE_CONTACT_KE,
            kd=SPHERE_CONTACT_KD,
            restitution=SPHERE_CONTACT_RESTITUTION,
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
            mu=SPHERE_FRICTION_MU,
            ke=SPHERE_CONTACT_KE,
            kd=SPHERE_CONTACT_KD,
            restitution=SPHERE_CONTACT_RESTITUTION,
        )
        light_cfg = ShapeConfig(
            density=LIGHT_SPHERE_DENSITY,
            is_visible=not TEXTURED_SPHERE_VISUALS_ENABLED,
            is_solid=True,
            mu=SPHERE_FRICTION_MU,
            ke=SPHERE_CONTACT_KE,
            kd=SPHERE_CONTACT_KD,
            restitution=SPHERE_CONTACT_RESTITUTION,
        )
        # Dry start ahead of dam; wider y gap so light is less likely to perch on heavy
        # when vertical ME is still weak (user saw ~½R contact lift with --no-archimedes).
        dam_x_world = float(int(n * DAM_X_FRAC)) * DH
        heavy_center = (dam_x_world + 2.5 * radius, world_y * 0.30, z_floor)
        light_center = (dam_x_world + 2.5 * radius, world_y * 0.70, z_floor)

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
        # Drop wall–wall pairs from explicit broadphase (15/28 were static–static).
        _prune_static_static_contact_pairs(self.rigid_domain.model)
        if self.viewer is not None and hasattr(self.viewer, "set_model"):
            self.rigid_domain.model.setup_viewer(self.viewer)

        self.coupling = GridLbmRigidCoupling(self.domain, self.rigid_domain)
        self.coupling.add_body_sphere(body_idx=self.heavy_body_id, radius=radius)
        self.coupling.add_body_sphere(body_idx=self.light_body_id, radius=radius)
        # Example advances rigid after optional empirical; coupling only rasters + fluid + ME.
        self.coupling.set_rigid_dynamics_enabled(False)
        self.coupling.set_two_way_feedback_enabled(True, force_scale=feedback_force_scale)
        self.coupling.set_feedback_mode("momentum_exchange")
        # ME writes force F; apply J=F·dt once (no second XPBD dt on ME).
        self.coupling.set_me_integration_mode("impulse")

        print(
            f"  spheres: r={radius}, heavy_ρ={HEAVY_SPHERE_DENSITY}, "
            f"light_ρ={LIGHT_SPHERE_DENSITY}, feedback={feedback_force_scale:.4g}, "
            f"mode={self.coupling.feedback_mode}, "
            f"me_apply={self.coupling.me_integration_mode}"
        )
        print(f"  showcase_fsi={'on' if self._showcase_fsi else 'off'}")
        print(f"  me_drag={'on' if self._me_drag else 'off'}")
        print(
            f"  archimedes={'on' if self._archimedes else 'off'} "
            f"scale={self._archimedes_scale:g} "
            f"(incompressible Fz=s·ρ·V·|g|; ME keeps impact)"
        )
        print(
            f"  hydro_rho={'on' if self._hydro_rho else 'off'} "
            f"rate={self._hydro_rho_rate:g} every={self._hydro_rho_every} "
            f"(opt-in ρ(z) ME experiment; not default physics)"
        )
        if self._showcase_fsi:
            print(
                f"  empirical_fsi buoyancy={self._empirical_cfg.buoyancy_scale}, "
                f"drag_xy={self._empirical_cfg.drag_xy}, "
                f"drag_z={self._empirical_cfg.drag_z} (showcase plugin)"
            )
        elif self._me_drag:
            print(
                f"  me_path_drag drag_xy={self._empirical_cfg.drag_xy}, "
                f"drag_z={self._empirical_cfg.drag_z}, "
                f"drag_ang={self._empirical_cfg.drag_ang} (no buoyancy/push)"
            )
        print(f"  initial heavy={heavy_center}, light={light_center}")

        if self._showcase_fsi or self._me_drag:
            self._empirical_fsi = EmpiricalSphereFsiPlugin(
                device=str(self.model._device),
                body_ids=(self.heavy_body_id, self.light_body_id),
                densities=(HEAVY_SPHERE_DENSITY, LIGHT_SPHERE_DENSITY),
                radius=SPHERE_RADIUS,
                volume=self._sphere_volume,
                rho_liquid=RHO_LIQUID,
                gravity_abs=abs(self._rigid_gravity_z),
                dh=DH,
                nx=self._n,
                ny=self._n,
                nz=self._n,
                config=self._empirical_cfg,
            )

    def _ramp_gravity(self) -> None:
        target_lbm_gz = float(self.model.gravity_z)
        target_rigid_gz = float(self._rigid_gravity_z)
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
        # Re-baseline after ramp (solid mask / FS may shift inventory slightly).
        self._liquid_mass0 = self._sample_liquid_mass()
        self._last_liquid_mass = self._liquid_mass0
        print(
            f"  gravity ramp done: gz_lbm={self.model.gravity_z}, "
            f"rigid_z={target_rigid_gz:.4g} (matched a=g*dh/dt^2), "
            f"mass0={self._liquid_mass0:.1f}"
        )

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
            self._last_liquid_mass = self._sample_liquid_mass()
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
        mass = float(self._last_liquid_mass)
        mass0 = float(self._liquid_mass0) if self._liquid_mass0 > 0.0 else mass
        mass_rel = (mass / mass0) if mass0 > 0.0 else 1.0
        fp.write(
            f"{self.frame_count},{self.sim_time:.6f},{phase},"
            f"{heavy_pos[0]:.8f},{heavy_pos[1]:.8f},{heavy_pos[2]:.8f},"
            f"{heavy_vel[0]:.8f},{heavy_vel[1]:.8f},{heavy_vel[2]:.8f},{heavy_sub:.6f},"
            f"{light_pos[0]:.8f},{light_pos[1]:.8f},{light_pos[2]:.8f},"
            f"{light_vel[0]:.8f},{light_vel[1]:.8f},{light_vel[2]:.8f},{light_sub:.6f},"
            f"{heavy_f[0]:.8f},{heavy_f[1]:.8f},{heavy_f[2]:.8f},"
            f"{light_f[0]:.8f},{light_f[1]:.8f},{light_f[2]:.8f},"
            f"{mass:.6f},{mass0:.6f},{mass_rel:.8f}\n"
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
        # ME (horizontal FSI) → Archimedes lift → optional showcase/drag → XPBD.
        self.coupling.step(self.sim_dt)
        self._apply_phi_volume_archimedes()
        self._apply_empirical_buoyancy_and_drag()
        self.rigid_domain.step(self.sim_dt)

    def _apply_phi_volume_archimedes(self) -> None:
        """Upward Fz = scale * s * ρ_f * V * |g| from shell φ (not ME)."""
        if not self._archimedes:
            return
        state = self.domain.state
        rigid = self.rigid_domain.state
        self._phi_buoy_scratch = ensure_phi_volume_scratch(
            device=str(self.model._device),
            offsets_xyz=self._phi_buoy_offsets,
            body_ids=(self.heavy_body_id, self.light_body_id),
            scratch=self._phi_buoy_scratch,
        )
        subs = apply_phi_volume_buoyancy_gpu(
            phi=state.phi,
            cell=state.cell_type,
            solid=state.solid_phi,
            body_q=rigid.body_q,
            body_f_apply=rigid.apply_body_forces,
            radius=SPHERE_RADIUS,
            dh=DH,
            nx=self._n,
            ny=self._n,
            nz=self._n,
            scratch=self._phi_buoy_scratch,
            volume=self._sphere_volume,
            rho_liquid=RHO_LIQUID,
            gravity_abs=abs(self._rigid_gravity_z),
            buoyancy_scale=self._archimedes_scale,
            phi_wet=0.05,
            ema_alpha=SUB_EMA_ALPHA,
            dsub_cap=SUB_DSUB_CAP,
            sync_submerged=False,
        )
        if subs:
            self._last_submerged_by_body.update(subs)
        scratch = self._phi_buoy_scratch
        if scratch is not None:
            forces = scratch["forces"].numpy()
            ids = scratch["body_ids_host"]
            for i, body_id in enumerate(ids):
                f = forces[i]
                self._last_extra_force_by_body[int(body_id)] = (
                    float(f[0]),
                    float(f[1]),
                    float(f[2]),
                )

    def _apply_empirical_buoyancy_and_drag(self) -> None:
        if self._empirical_fsi is None:
            return
        state = self.domain.state
        rigid = self.rigid_domain.state
        vel_scale = DH / max(self.sim_dt, 1.0e-12)
        self._empirical_fsi.apply(
            phi=state.phi,
            cell=state.cell_type,
            solid=state.solid_phi,
            ux=state.velocity_x,
            uy=state.velocity_y,
            uz=state.velocity_z,
            body_q=rigid.body_q,
            body_qd=rigid.body_qd,
            body_f_apply=rigid.apply_body_forces,
            vel_scale=vel_scale,
            late_pool_armed=self._height_eq_armed,
            sync_submerged=False,
        )

    def _sync_buoyancy_submerged(self) -> None:
        plugin = self._empirical_fsi
        if plugin is not None and plugin.scratch is not None:
            scratch = plugin.scratch
            sub = scratch["submerged"].numpy()
            ids = scratch["body_ids_host"]
            for i, body_id in enumerate(ids):
                self._last_submerged_by_body[int(body_id)] = float(sub[i])
            forces = scratch["forces"].numpy()
            for i, body_id in enumerate(ids):
                f = forces[i]
                self._last_extra_force_by_body[int(body_id)] = (
                    float(f[0]),
                    float(f[1]),
                    float(f[2]),
                )
            return
        scratch = self._phi_buoy_scratch
        if scratch is not None and self._archimedes:
            sub = scratch["submerged"].numpy()
            ids = scratch["body_ids_host"]
            for i, body_id in enumerate(ids):
                self._last_submerged_by_body[int(body_id)] = float(sub[i])
            forces = scratch["forces"].numpy()
            for i, body_id in enumerate(ids):
                f = forces[i]
                self._last_extra_force_by_body[int(body_id)] = (
                    float(f[0]),
                    float(f[1]),
                    float(f[2]),
                )
            return
        self._sample_shell_wet_fraction()

    def _sample_shell_wet_fraction(self) -> None:
        """Estimate submerged fraction from φ at shell samples (status/CSV)."""
        state = self.domain.state
        phi = state.phi.numpy()
        cell = state.cell_type.numpy()
        solid = state.solid_phi.numpy()
        n = self._n
        radius = SPHERE_RADIUS
        # Sparse shell directions (same spirit as empirical plugin).
        dirs = (
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
            (0.577, 0.577, 0.577), (-0.577, 0.577, 0.577),
            (0.577, -0.577, 0.577), (-0.577, -0.577, 0.577),
            (0.577, 0.577, -0.577), (-0.577, 0.577, -0.577),
            (0.577, -0.577, -0.577), (-0.577, -0.577, -0.577),
        )
        for body_id in (self.heavy_body_id, self.light_body_id):
            pos = np.asarray(
                self.rigid_domain.state.get_body_position(body_id), dtype=np.float64
            )
            wet = 0
            valid = 0
            for ox, oy, oz in dirs:
                sx = pos[0] + ox * radius
                sy = pos[1] + oy * radius
                sz = pos[2] + oz * radius
                i = int(np.clip(sx / DH, 0, n - 1))
                j = int(np.clip(sy / DH, 0, n - 1))
                k = int(np.clip(sz / DH, 0, n - 1))
                if solid[i, j, k] < 0.0:
                    continue
                valid += 1
                if int(cell[i, j, k]) == 0:
                    continue
                if float(phi[i, j, k]) <= 0.08:
                    continue
                wet += 1
            self._last_submerged_by_body[int(body_id)] = (
                float(wet) / float(valid) if valid > 0 else 0.0
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

    def _sample_liquid_mass(self) -> float:
        """Wet inventory ``Σmass`` over liquid + interface (excludes gas/solid)."""
        home = self.domain.solver._home_fp32
        if home is None:
            return 0.0
        buf = home._ensure_gpu()
        mass = buf.mass.numpy()
        cell = buf.cell_type.numpy()
        solid = buf.solid_phi.numpy()
        wet = ((cell == 1) | (cell == 2)) & (solid >= 0.0)
        return float(mass[wet].sum())

    def _print_status(self) -> None:
        # Status cadence already syncs mass; keep host logs off the hot path.
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
        J = None
        try:
            J = self.coupling.last_me_impulse
        except Exception:
            J = None
        me = ""
        if J is not None and J.shape[0] > max(self.heavy_body_id, self.light_body_id):
            jh = J[self.heavy_body_id, 0:3]
            jl = J[self.light_body_id, 0:3]
            me = f" ME_J=({float(np.linalg.norm(jh)):.3g},{float(np.linalg.norm(jl)):.3g})"
        mass = float(self._last_liquid_mass)
        mass0 = float(self._liquid_mass0) if self._liquid_mass0 > 0.0 else mass
        dmass_pct = 100.0 * ((mass / mass0) - 1.0) if mass0 > 0.0 else 0.0
        hydro = ""
        if self._hydro_rho:
            home = self.domain.solver._home_fp32
            st = getattr(home, "_last_hydro_rho_stats", None) if home is not None else None
            if st:
                hydro = f" sρ={st.get('scale', 1):.4f}"
        print(
            f"[t={self.sim_time:.1f}s] "
            f"heavy=({heavy_pos[0]:.2f},{heavy_pos[1]:.2f},{heavy_pos[2]:.2f}) "
            f"light=({light_pos[0]:.2f},{light_pos[1]:.2f},{light_pos[2]:.2f}) "
            f"sub=({self._last_submerged_by_body.get(self.heavy_body_id, 0.0):.2f},"
            f"{self._last_submerged_by_body.get(self.light_body_id, 0.0):.2f}) "
            f"light_v=({light_vel[0]:+.3f},{light_vel[2]:+.3f}) "
            f"mass={mass:.0f} (dM={dmass_pct:+.2f}%) "
            f"sim={self._last_ms:.0f}ms{me}{hydro}{heq} "
            f"res={getattr(self.coupling, 'last_me_apply_rel', 0):.2e}",
            file=sys.stderr,
            flush=True,
        )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--n", type=int, default=N, help="Grid resolution (N³).")
    parser.add_argument(
        "--feedback-force-scale",
        type=float,
        default=None,
        help=(
            "Multiplier for LBM→rigid ME feedback. "
            "Default: recommended_me_force_scale(dh, dt)."
        ),
    )
    parser.add_argument(
        "--showcase-fsi",
        action="store_true",
        help=(
            "Enable showcase empirical buoyancy/push/drag + eq-wall. "
            "Default off: paper Sec.4.3 raster → Eq.24 → link ME → rigid."
        ),
    )
    parser.add_argument(
        "--me-in-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate ME inside fused solid pulls (default: on).",
    )
    parser.add_argument(
        "--me-drag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional non-paper submerged drag on the ME path (no buoyancy/push). "
            "Default off: paper Sec.4.3 ME only. Ignored with --showcase-fsi."
        ),
    )
    parser.add_argument(
        "--me-drag-xy",
        type=float,
        default=ME_PATH_DRAG_XY,
        help=f"ME-path horizontal drag rate when --me-drag (default {ME_PATH_DRAG_XY}).",
    )
    parser.add_argument(
        "--me-drag-z",
        type=float,
        default=ME_PATH_DRAG_Z,
        help=f"ME-path vertical drag rate when --me-drag (default {ME_PATH_DRAG_Z}).",
    )
    parser.add_argument(
        "--me-drag-ang",
        type=float,
        default=ME_PATH_DRAG_ANG,
        help=f"ME-path angular drag rate when --me-drag (default {ME_PATH_DRAG_ANG}).",
    )
    parser.add_argument(
        "--archimedes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Incompressible Archimedes Fz=s·ρ·V·|g| (default on). "
            "Normal-physics vertical buoyancy with uniform liquid ρ. "
            "Ignored when --hydro-rho."
        ),
    )
    parser.add_argument(
        "--archimedes-scale",
        type=float,
        default=ARCHIMEDES_SCALE,
        help=f"Multiplier on ρ V g s (default {ARCHIMEDES_SCALE}).",
    )
    parser.add_argument(
        "--hydro-rho",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental: maintain hydrostatic ρ(z) so Eq.32 ME can lift. "
            "Not normal incompressible water; turns off Archimedes. Default off."
        ),
    )
    parser.add_argument(
        "--hydro-rho-rate",
        type=float,
        default=HYDRO_RHO_RATE,
        help=f"Blend α toward ρ_h (default {HYDRO_RHO_RATE}).",
    )
    parser.add_argument(
        "--hydro-rho-every",
        type=int,
        default=HYDRO_RHO_EVERY,
        help=f"Apply hydro maintain every N lattice steps (default {HYDRO_RHO_EVERY}).",
    )
    parser.add_argument(
        "--buoyancy-force-scale",
        type=float,
        default=BUOYANCY_FORCE_SCALE,
        help="Empirical upward buoyancy helper scale (only with --showcase-fsi).",
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
        feedback_force_scale=(
            None
            if args.feedback_force_scale is None
            else float(args.feedback_force_scale)
        ),
        buoyancy_force_scale=float(args.buoyancy_force_scale),
        water_horizontal_drag_rate=float(args.water_horizontal_drag_rate),
        water_vertical_drag_rate=float(args.water_vertical_drag_rate),
        sphere_log_path=log_path if log_path else None,
        sphere_log_every=int(args.sphere_log_every),
        enable_height_eq=bool(args.height_eq),
        enable_moment_quant=bool(args.moment_quant),
        showcase_fsi=bool(args.showcase_fsi),
        me_in_fused=bool(args.me_in_fused),
        me_drag=bool(args.me_drag),
        me_drag_xy=float(args.me_drag_xy),
        me_drag_z=float(args.me_drag_z),
        me_drag_ang=float(args.me_drag_ang),
        archimedes=bool(args.archimedes),
        archimedes_scale=float(args.archimedes_scale),
        hydro_rho=bool(args.hydro_rho),
        hydro_rho_rate=float(args.hydro_rho_rate),
        hydro_rho_every=int(args.hydro_rho_every),
    )
    try:
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
