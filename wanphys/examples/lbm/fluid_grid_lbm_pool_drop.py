# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM two-phase near-neutral-buoyancy sphere two-way FSI visual example.

A Shan-Chen two-phase water pool (lower half, ``rho_w``) sits under an air
half (``rho_a``) with a cross-flow inlet on the -x face and outflow on +x.
A rigid sphere starts just above the water surface. It operates near neutral
buoyancy (small rigid gravity, sphere density ~ water) so the lattice-scale
momentum-exchange feedback can hold it up; the cross-flow advects it
downstream and two-way feedback (strict Ladd-1994 momentum exchange) sloshes
it back and forth, perturbing the free surface.

This is a stabilized near-neutral-buoyancy demo, NOT an Earth-gravity
sinking-ball test: at ``g=-9.81`` the momentum-exchange feedback is ~4
orders of magnitude smaller than the sphere weight, so the ball free-falls
through the pool and collapses the two-phase separation. A six-wall container
keeps both the sphere and the fluid inside the domain.

Visualization shows only the rigid sphere (+ container walls) and the fluid
density field.

Run:
    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_pool_drop --viewer gl
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import newton
import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState
from wanphys._src.fluid.fluid_grid.lbm.constants import BC_OUTFLOW
from wanphys._src.fluid.fluid_grid.lbm.constants import BC_VELOCITY_INLET
from wanphys._src.fluid.fluid_grid.lbm.constants import FACE_XMAX
from wanphys._src.fluid.fluid_grid.lbm.constants import FACE_XMIN
from wanphys._src.fluid.fluid_viewer import FluidViewerGL
from wanphys._src.fluid.fluid_viewer import ScreenSpaceFluidRenderer
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Grid
GRID_RES: tuple[int, int, int] = (48, 32, 32)
CELL_SIZE: float = 0.04

# Shan-Chen two-phase
TAU: float = 0.8
G_SC: float = -5.0
PSI_TYPE: int = 1  # PSI_EXP
PSI_REF: float = 1.0
SC_SOLID_PSI_SCALE: float = 0.9

# Two gravities (see spec §6: independent parameter systems, bridged by the coupling)
# The rigid sphere operates near neutral buoyancy: a *small* rigid gravity keeps
# the momentum-exchange feedback (lattice-scale, ~ρu²dh²) and the sphere weight
# in the same order of magnitude, so the cross-flow can advect the sphere instead
# of it free-falling through the pool. This is a near-neutral-buoyancy two-way FSI
# demo, NOT an Earth-gravity sinking-ball test (see spec §6, task 6 rationale).
GRAVITY_LBM: float = -0.0005        # lattice-unit body force on the fluid
RIGID_GRAVITY_Z: float = -0.5       # small rigid gravity: near-neutral buoyancy


# Two-phase densities
RHO_WATER: float = 1.8
RHO_AIR: float = 0.1

# Cross-flow inlet
INLET_VELOCITY: tuple[float, float, float] = (0.0004, 0.0, 0.0)

# Pool geometry: lower half is water
WATER_LEVEL_FRAC: float = 0.5

# Sphere
SPHERE_RADIUS: float = 0.12
SPHERE_DENSITIES: tuple[float, float, float] = (0.8, 1.8, 3.0)
SPHERE_DENSITY: float = SPHERE_DENSITIES[1]  # retained for single-density references
SPHERE_Y_FRACTIONS: tuple[float, float, float] = (0.3, 0.5, 0.7)
TEXTURED_SPHERE_VISUALS_ENABLED: bool = True
SPHERE_VISUAL_TEXTURE_SIZE: int = 256
SPHERE_VISUAL_MESH_LATITUDES: int = 32
SPHERE_VISUAL_MESH_LONGITUDES: int = 48

# Container (six walls) so neither the sphere nor the fluid can escape the pool.
WALL_THICKNESS_CELLS: float = 1.5

# Two-way feedback
# Force scale calibrated so the strict Ladd-1994 momentum-exchange integral at
# rest roughly balances the sphere weight (rho_sphere * V * g), giving near-neutral
# buoyancy. Empirically (headless force probe) the feedback force is ~linear in
# the scale and the balance point is fs ~= 4 for these densities/geometries; fs
# higher makes the steady upward reaction exceed weight, so the ball bounces up
# like a rubber ball instead of settling (see spec task 6 root-cause analysis).
FEEDBACK_FORCE_SCALE: float = 1.51  # empirical multiplier on the built-in dh^3 factor

# Screen-space fluid renderer
SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 350

# Time-stepping
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 8

# Pre-equilibration
PRE_EQUILIBRATE_STEPS: int = 150
GRAVITY_RAMP_STEPS: int = 80


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------


def _make_sphere_rotation_texture(sphere_index: int) -> np.ndarray:
    """Create a simple asymmetric UV texture so sphere rotation is visible."""
    size: int = SPHERE_VISUAL_TEXTURE_SIZE
    u: np.ndarray = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[None, :]
    v: np.ndarray = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[:, None]
    palette: tuple[tuple[int, int, int], ...] = (
        (45, 125, 210),
        (224, 116, 46),
        (64, 158, 104),
    )
    base_color: np.ndarray = np.array(palette[sphere_index % len(palette)], dtype=np.uint8)
    stripe_color: np.ndarray = np.array((18, 27, 38), dtype=np.uint8)
    marker_color: np.ndarray = np.array((245, 238, 210), dtype=np.uint8)

    texture: np.ndarray = np.empty((size, size, 4), dtype=np.uint8)
    texture[:, :, 0:3] = base_color
    texture[:, :, 3] = 255

    longitude_stripes: np.ndarray = np.broadcast_to(
        (np.floor((u + 0.07 * float(sphere_index)) * 10.0) % 2.0) < 0.35,
        (size, size),
    )
    latitude_band: np.ndarray = np.broadcast_to(np.abs(v - 0.5) < 0.035, (size, size))
    marker_u: float = 0.22 + 0.17 * float(sphere_index)
    marker_v: float = 0.34 + 0.09 * float(sphere_index % 2)
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
        label=f"pool_drop_sphere_{sphere_index}_rotation_visual",
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.func
def _pool_f_eq(
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    w: float,
    cx: int,
    cy: int,
    cz: int,
) -> float:
    cu = float(cx) * ux + float(cy) * uy + float(cz) * uz
    u_sq = ux * ux + uy * uy + uz * uz
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)


@wp.kernel
def _init_pool_water(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    velocity_x: wp.array3d(dtype=float),
    velocity_y: wp.array3d(dtype=float),
    velocity_z: wp.array3d(dtype=float),
    water_k_max: int,
    rho_w: float,
    rho_a: float,
    u_in_x: float,
    u_in_y: float,
    u_in_z: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Stratify water (rho_w) below k < water_k_max, air (rho_a) above.

    Water cells are seeded with the inlet velocity to reduce startup shock;
    air cells are seeded with the same horizontal velocity for a uniform
    inlet profile (accepted v1 simplification, see spec §8).
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    if k < water_k_max:
        rho = rho_w
        ux = u_in_x
        uy = u_in_y
        uz = u_in_z
    else:
        rho = rho_a
        ux = u_in_x
        uy = u_in_y
        uz = u_in_z

    noise = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    noise = noise * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho + noise * 0.005 * rho, 0.01)

    density[i, j, k] = rho
    velocity_x[i, j, k] = ux
    velocity_y[i, j, k] = uy
    velocity_z[i, j, k] = uz

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
def _init_pool_water_equilibrium(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    velocity_x: wp.array3d(dtype=float),
    velocity_y: wp.array3d(dtype=float),
    velocity_z: wp.array3d(dtype=float),
    water_k_max: int,
    rho_w: float,
    rho_a: float,
    u_in_x: float,
    u_in_y: float,
    u_in_z: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Stratify water/air and initialize f from the matching moving equilibrium."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    if k < water_k_max:
        rho = rho_w
        ux = u_in_x
        uy = u_in_y
        uz = u_in_z
    else:
        rho = rho_a
        ux = u_in_x
        uy = u_in_y
        uz = u_in_z

    noise = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    noise = noise * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho + noise * 0.005 * rho, 0.01)

    density[i, j, k] = rho
    velocity_x[i, j, k] = ux
    velocity_y[i, j, k] = uy
    velocity_z[i, j, k] = uz

    f[0 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 3.0, 0, 0, 0)
    f[1 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 18.0, 1, 0, 0)
    f[2 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 18.0, -1, 0, 0)
    f[3 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 18.0, 0, 1, 0)
    f[4 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 18.0, 0, -1, 0)
    f[5 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 18.0, 0, 0, 1)
    f[6 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 18.0, 0, 0, -1)
    f[7 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 1, 1, 0)
    f[8 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, -1, 1, 0)
    f[9 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 1, -1, 0)
    f[10 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, -1, -1, 0)
    f[11 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 1, 0, 1)
    f[12 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, -1, 0, 1)
    f[13 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 1, 0, -1)
    f[14 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, -1, 0, -1)
    f[15 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 0, 1, 1)
    f[16 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 0, -1, 1)
    f[17 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 0, 1, -1)
    f[18 * stride + idx] = _pool_f_eq(rho, ux, uy, uz, 1.0 / 36.0, 0, -1, -1)


@wp.kernel
def _mask_solid_density(
    density: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
) -> None:
    """Hide density inside rigid solids for the screen-space fluid renderer.

    Writes only to ``density_out`` (the render buffer); never modifies the
    solver's primary density. See spec §7.
    """
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        density_out[i, j, k] = 0.0
    else:
        density_out[i, j, k] = density[i, j, k]


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class LbmPoolDropExample:
    """Two-phase pool-drop demo with strict momentum-exchange two-way FSI."""

    def __init__(self, viewer: Any, *, feedback_force_scale: float = FEEDBACK_FORCE_SCALE) -> None:
        self.viewer: Any = viewer
        if isinstance(self.viewer, FluidViewerGL):
            self.viewer._paused: bool = True

        nx, ny, nz = GRID_RES
        self.model: LbmModel = LbmModel(
            fluid_grid_res=GRID_RES,
            fluid_grid_cell_size=CELL_SIZE,
            tau=TAU,
            G=G_SC,
            psi_type=PSI_TYPE,
            psi_ref=PSI_REF,
            sc_solid_psi_scale=SC_SOLID_PSI_SCALE,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY_LBM,
        )
        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()
        self.solver = self.domain.solver

        # Boundary conditions: -x inlet, +x outflow, others default bounce-back
        self.solver.set_boundary_condition(FACE_XMIN, bc_type=BC_VELOCITY_INLET, velocity=INLET_VELOCITY)
        self.solver.set_boundary_condition(FACE_XMAX, bc_type=BC_OUTFLOW)

        world_x: float = float(nx) * self.model.dh
        world_y: float = float(ny) * self.model.dh
        water_z: float = float(nz) * WATER_LEVEL_FRAC * self.model.dh

        # Spheres share x/z and are separated across y, just above the surface with a small gap.
        self.sphere_radius: float = SPHERE_RADIUS
        sphere_x: float = world_x * 0.88
        sphere_z: float = water_z + self.sphere_radius + 20.0 * self.model.dh
        self.sphere_centers: list[tuple[float, float, float]] = [
            (sphere_x, world_y * y_frac, sphere_z) for y_frac in SPHERE_Y_FRACTIONS
        ]

        print(f"LBM pool-drop: {nx}x{ny}x{nz}, dh={self.model.dh:.3f}, tau={TAU}")
        print(f"  Pool: water below z={water_z:.3f} (frac={WATER_LEVEL_FRAC})")
        print(
            f"  SC: G={G_SC}, psi={PSI_TYPE}, ref={PSI_REF}, "
            f"solid_psi_scale={SC_SOLID_PSI_SCALE}, gz_lbm={GRAVITY_LBM}"
        )
        print(f"  Inlet: u={INLET_VELOCITY}, gravity_rigid_z={RIGID_GRAVITY_Z}")
        print(
            f"  Spheres: centers={self.sphere_centers}, densities={SPHERE_DENSITIES}, radius={self.sphere_radius:.3f}, "
            f"feedback_force_scale={feedback_force_scale}"
        )

        # Substep dt is needed by ``_pre_equilibrate`` (run below) and ``step``;
        # set it early so both stages share the identical per-coupling-step dt.
        self.sim_time: float = 0.0
        self.sim_dt: float = FRAME_DT / float(SIM_SUBSTEPS)
        self.frame_count: int = 0
        self._last_sim_ms: float = 0.0

        self._init_water()
        self._init_rigid_scene(feedback_force_scale=float(feedback_force_scale))
        self._pre_equilibrate()  # implemented in Task 3

        # Render-only density buffer (never fed back into the solver; spec §7)
        self.display_density: wp.array3d = wp.zeros((nx, ny, nz), dtype=float, device=self.model._device)

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

        print("Controls: [Space] unpause  [R] reset  [mouse] orbit  [scroll] zoom")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_water(self) -> None:
        state: LbmState = self.domain.state
        nx, ny, nz = GRID_RES
        water_k_max: int = int(float(nz) * WATER_LEVEL_FRAC)
        stride: int = nx * ny * nz

        wp.launch(
            _init_pool_water_equilibrium,
            dim=(nx, ny, nz),
            inputs=[
                state.f,
                state.density,
                state.velocity_x,
                state.velocity_y,
                state.velocity_z,
                water_k_max,
                RHO_WATER,
                RHO_AIR,
                float(INLET_VELOCITY[0]),
                float(INLET_VELOCITY[1]),
                float(INLET_VELOCITY[2]),
                42,
                nx,
                ny,
                nz,
                stride,
            ],
            device=self.model._device,
        )

        # Mirror into the double-buffered out-state so streaming starts consistent.
        wp.copy(self.domain._state_out.f, state.f)
        wp.copy(self.domain._state_out.density, state.density)
        wp.copy(self.domain._state_out.velocity_x, state.velocity_x)
        wp.copy(self.domain._state_out.velocity_y, state.velocity_y)
        wp.copy(self.domain._state_out.velocity_z, state.velocity_z)
        wp.copy(self.domain._state_out.solid_phi, state.solid_phi)
        wp.copy(self.domain._state_out.solid_body_id, state.solid_body_id)

        wp.synchronize_device(self.model._device)
        rho_np: np.ndarray = state.density.numpy()
        print(f"  Init water cells: {int((rho_np > 1.0).sum())} / {nx * ny * nz}")

    def _init_rigid_scene(self, *, feedback_force_scale: float) -> None:
        builder: RigidModelBuilder = RigidModelBuilder(gravity=RIGID_GRAVITY_Z)
        nx, ny, nz = GRID_RES
        ws: float = float(nx) * self.model.dh
        wss: float = float(nz) * self.model.dh
        wsy: float = float(ny) * self.model.dh
        wt: float = WALL_THICKNESS_CELLS * self.model.dh

        # Static (zero-density) container: six walls enclosing the whole grid, so
        # the sphere cannot fall out the bottom and the fluid cannot escape. The
        # walls are kept ``is_visible=False`` (matching the dambreak reference):
        # drawing full-size boxes would fully occlude the sphere inside the box.
        # The fluid boundary is visualized by the SSFR density field, and the
        # walls still act as collision/SDF containment through the coupling.
        wall_cfg: ShapeConfig = ShapeConfig(
            density=0.0,
            is_visible=False,
            is_solid=True,
            has_shape_collision=True,
        )
        self.wall_bodies: list[tuple[int, tuple[float, float, float]]] = []

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
            self.wall_bodies.append((body, half_extents))

        # add_wall("container_xmin", (wt * 0.5, wsy * 0.5, wss * 0.5), (wt * 0.5, wsy * 0.5, wss * 0.5))
        # add_wall("container_xmax", (ws - wt * 0.5, wsy * 0.5, wss * 0.5), (wt * 0.5, wsy * 0.5, wss * 0.5))
        add_wall("container_ymin", (ws * 0.5, wt * 0.5, wss * 0.5), (ws * 0.5, wt * 0.5, wss * 0.5))
        add_wall("container_ymax", (ws * 0.5, wsy - wt * 0.5, wss * 0.5), (ws * 0.5, wt * 0.5, wss * 0.5))
        add_wall("container_floor", (ws * 0.5, wsy * 0.5, wt * 0.5), (ws * 0.5, wsy * 0.5, wt * 0.5))
        # add_wall("container_ceiling", (ws * 0.5, wsy * 0.5, wss - wt * 0.5), (ws * 0.5, wsy * 0.5, wt * 0.5))

        # Dynamic spheres with different densities (two-way momentum exchange).
        self.sphere_body_ids: list[int] = []
        for sphere_index, (sphere_center, sphere_density) in enumerate(
            zip(self.sphere_centers, SPHERE_DENSITIES, strict=True)
        ):
            sphere_cfg: ShapeConfig = ShapeConfig(
                density=sphere_density,
                is_visible=not TEXTURED_SPHERE_VISUALS_ENABLED,
                is_solid=True,
                has_shape_collision=True,
            )
            sphere_body_id: int = builder.add_body(
                position=sphere_center,
                label=f"pool_drop_sphere_{sphere_index}",
            )
            builder.add_shape_sphere(sphere_body_id, radius=self.sphere_radius, cfg=sphere_cfg)
            if TEXTURED_SPHERE_VISUALS_ENABLED:
                _add_textured_sphere_visual(
                    builder,
                    sphere_body_id,
                    self.sphere_radius,
                    sphere_index,
                )
            self.sphere_body_ids.append(sphere_body_id)
        self.rigid_domain: RigidDomain = RigidDomain(builder.finalize(device=self.model._device))
        self.rigid_domain.create_state()
        # ``setup_viewer`` registers the rigid model with the Newton viewer for
        # rendering. It requires a real viewer instance (GL/Null/USD/Rerun); skip
        # it for the headless ``viewer=None`` smoke-test path used in tests.
        if self.viewer is not None:
            self.rigid_domain.model.setup_viewer(self.viewer)

        self.coupling: GridLbmRigidCoupling = GridLbmRigidCoupling(self.domain, self.rigid_domain)
        # Walls: static, one-way (they perturb the fluid via SDF/wall velocity only).
        for body_id, half_extents in self.wall_bodies:
            self.coupling.add_body_box(body_id, half_extents)
        # Spheres: dynamic, two-way momentum exchange.
        for sphere_body_id in self.sphere_body_ids:
            self.coupling.add_body_sphere(body_idx=sphere_body_id, radius=self.sphere_radius)
        # Walls start rigid-frozen; only the spheres are advanced by dynamics.
        self.coupling.set_rigid_dynamics_enabled(True)
        self.coupling.set_two_way_feedback_enabled(True, force_scale=feedback_force_scale)
        self.coupling.set_feedback_mode("momentum_exchange")


    def _pre_equilibrate(self) -> None:
        """Two-stage pre-equilibration with dual-gravity synchronized ramp.

        Stage A: zero both gravities, freeze rigid dynamics. The sphere stays
        in place as static geometry (SDF/wall-velocity/feedback still run each
        coupling step); the two-phase pool and inlet cross-flow settle.

        Stage B: enable rigid dynamics and ramp both gravities together from
        zero to their full values over GRAVITY_RAMP_STEPS steps, so the sphere
        releases gently into the already-steady pool.

        Two-way feedback stays enabled throughout: each coupling step opens
        with ``rigid_state.clear_forces()``, so forces accumulated in Stage A
        do not leak into Stage B.

        See spec §5 for the rationale and §5.4 for the mandatory restoration.
        """
        # ---- Stage A: pool stabilization, sphere frozen as static geometry ----
        print(f"  Stage A: stabilizing pool {PRE_EQUILIBRATE_STEPS} steps (g=0, rigid frozen) ...")
        self.model.gravity_z = 0.0
        self.rigid_domain.model.set_gravity((0.0, 0.0, 0.0))
        self.coupling.set_rigid_dynamics_enabled(False)
        for _ in range(PRE_EQUILIBRATE_STEPS):
            self.coupling.step(self.sim_dt)
        wp.synchronize_device(self.model._device)

        # ---- Stage B: synchronized dual-gravity ramp + sphere release ----
        print(f"  Stage B: ramping gravity over {GRAVITY_RAMP_STEPS} steps (dual gravity) ...")
        self.coupling.set_rigid_dynamics_enabled(True)
        for step_index in range(GRAVITY_RAMP_STEPS):
            alpha: float = float(step_index + 1) / float(GRAVITY_RAMP_STEPS)
            self.model.gravity_z = alpha * GRAVITY_LBM
            self.rigid_domain.model.set_gravity((0.0, 0.0, alpha * RIGID_GRAVITY_Z))
            self.coupling.step(self.sim_dt)
        wp.synchronize_device(self.model._device)

        # ---- Mandatory restoration (spec §5.4): defend against drift / forgotten toggles ----
        self.model.gravity_z = GRAVITY_LBM
        self.rigid_domain.model.set_gravity((0.0, 0.0, RIGID_GRAVITY_Z))
        self.coupling.set_rigid_dynamics_enabled(True)
        print(f"  Pre-equilibration done: gz_lbm={self.model.gravity_z}, rigid_z={RIGID_GRAVITY_Z}, rigid_enabled=True")

    # ------------------------------------------------------------------
    # Step / render
    # ------------------------------------------------------------------

    def step(self) -> None:
        t0: float = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self.coupling.step(self.sim_dt)
        wp.synchronize_device(self.model._device)

        self._last_sim_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1

        if self.frame_count % 60 == 0:
            self._print_status()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid_domain.state.as_newton_state())

        if self.ssfr is not None and self.ssfr.available:
            nx, ny, nz = GRID_RES
            state: LbmState = self.domain.state
            # Render-only mask: never modify LbmState.density (spec §7)
            wp.launch(
                _mask_solid_density,
                dim=(nx, ny, nz),
                inputs=[state.density, state.solid_phi, self.display_density],
                device=self.model._device,
            )
            self.ssfr.set_density_field(
                density=self.display_density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=self.model.dh,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )

        self.viewer.end_frame()

    def test_final(self) -> None:
        """Headless acceptance: sphere positions finite, density field has no NaN/Inf."""
        for sphere_index, sphere_body_id in enumerate(self.sphere_body_ids):
            pos: np.ndarray = np.asarray(
                self.rigid_domain.state.get_body_position(sphere_body_id),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(pos)):
                raise ValueError(f"sphere {sphere_index} position not finite: {pos}")
        rho_np: np.ndarray = np.asarray(self.domain.state.density.numpy(), dtype=np.float64)
        if not np.all(np.isfinite(rho_np)):
            raise ValueError("density field contains non-finite values")

    def _print_status(self) -> None:
        state: LbmState = self.domain.state
        rho_np: np.ndarray = state.density.numpy()
        water: np.ndarray = rho_np > 1.0
        solid_count: int = int((state.solid_phi.numpy() < 0.0).sum())
        sphere_positions: list[np.ndarray] = [
            np.asarray(self.rigid_domain.state.get_body_position(sphere_body_id), dtype=np.float64)
            for sphere_body_id in self.sphere_body_ids
        ]
        sphere_pos_text: str = ", ".join(
            f"{sphere_index}:({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
            for sphere_index, pos in enumerate(sphere_positions)
        )
        if water.any():
            com: np.ndarray = np.argwhere(water).mean(axis=0)
            print(
                f"[t={self.sim_time:.2f}s] frame={self.frame_count} "
                f"water={int(water.sum())} solid={solid_count} "
                f"water_com=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) "
                f"sphere_pos=[{sphere_pos_text}] "
                f"sim={self._last_sim_ms:.0f}ms",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[t={self.sim_time:.2f}s] frame={self.frame_count} no water "
                f"sphere_pos=[{sphere_pos_text}]",
                file=sys.stderr,
                flush=True,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the pool-drop example CLI parser."""
    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=SPHERE_RADIUS,
        help="Rigid sphere radius in world units.",
    )
    parser.add_argument(
        "--inlet-velocity",
        nargs=3,
        type=float,
        default=list(INLET_VELOCITY),
        metavar=("UX", "UY", "UZ"),
        help="Inlet velocity vector (default: 0.04 0 0).",
    )
    parser.add_argument(
        "--feedback-force-scale",
        type=float,
        default=FEEDBACK_FORCE_SCALE,
        help="Empirical multiplier on the two-way momentum-exchange feedback force (default: 1.0).",
    )
    return parser


def main() -> None:
    """Run the LBM pool-drop two-way FSI visual demo."""
    parser: argparse.ArgumentParser = create_parser()
    viewer: Any
    args: argparse.Namespace
    viewer, args = init_fluid_viewer(parser)

    # Override module constants from CLI where provided, before construction.
    global SPHERE_RADIUS, INLET_VELOCITY, FEEDBACK_FORCE_SCALE
    SPHERE_RADIUS = float(args.sphere_radius)
    INLET_VELOCITY = (float(args.inlet_velocity[0]), float(args.inlet_velocity[1]), float(args.inlet_velocity[2]))
    FEEDBACK_FORCE_SCALE = float(args.feedback_force_scale)

    example: LbmPoolDropExample = LbmPoolDropExample(
        viewer,
        feedback_force_scale=FEEDBACK_FORCE_SCALE,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
