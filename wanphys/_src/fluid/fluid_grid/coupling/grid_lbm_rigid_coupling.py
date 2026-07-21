# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""One-way rigid-to-fluid coupling for the D3Q19 LBM solver.

The rigid body SDF is rasterised into ``solid_phi`` and the body surface
velocity is converted from world units to LBM lattice wall velocity before
being stamped into ``vel_solid_u/v/w`` each step.  The LBM solver reads
these fields to apply moving-wall bounce-back boundary conditions.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from wanphys._src.core.composite import CompositeSimulation
from wanphys.collision import CollisionPipeline
from . import coupling_kernels as ck

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState
    from wanphys._src.rigid import RigidDomain, RigidState


_SHAPE_MAP = {
    "sphere": ck._SHAPE_SPHERE,
    "box": ck._SHAPE_BOX,
    "capsule": ck._SHAPE_CAPSULE,
    "mesh": ck._SHAPE_MESH,
}


class GridLbmRigidCoupling(CompositeSimulation):
    """One-way rigid-to-LBM-fluid coupling.

    Each timestep the rigid bodies are rasterised as an SDF into the
    LBM ``solid_phi`` field and world body surface velocities are converted
    to LBM lattice wall velocities before being embedded into
    ``vel_solid_u/v/w``.  The LBM solver then applies moving-wall bounce-back
    to enforce the no-slip condition at moving boundaries.

    Parameters
    ----------
    fluid_domain:
        LBM fluid domain (owns model, solver, double-buffered state).
    rigid_domain:
        Rigid-body domain (owns Newton rigid model and state).
    """

    _SPHERE = "sphere"
    _BOX = "box"
    _CAPSULE = "capsule"
    _MESH = "mesh"

    def __init__(
        self,
        fluid_domain: LbmDomain,
        rigid_domain: RigidDomain,
    ):
        super().__init__()
        self._fluid_domain = fluid_domain
        self._rigid_domain = rigid_domain
        self._rigid_shape_query = CollisionPipeline.get_shape_query(self._rigid_domain)
        self._advance_rigid = True
        self._two_way_feedback_enabled: bool = False
        self._feedback_force_scale: float = 1.0
        self._feedback_mode: str = "approx"  # "approx" | "momentum_exchange"
        self._wall_velocity_warning_threshold: float = 0.125
        self._wall_velocity_warning_emitted: bool = False
        self._wall_velocity_warning_check_interval: int = 30
        self._wall_velocity_warning_step_count: int = 0
        self._last_lbm_feedback_wrench: np.ndarray | None = None

        self._bodies: list[dict] = []
        self._body_params_dirty = True

        # Device arrays for body shape parameters (mirrors GridLiquidRigidCoupling)
        self._body_shape_type: wp.array | None = None
        self._body_sphere_radius: wp.array | None = None
        self._body_box_half_extents: wp.array | None = None
        self._body_capsule_radius: wp.array | None = None
        self._body_capsule_half_height: wp.array | None = None
        self._body_mesh_handle: wp.array | None = None
        self._body_mesh_scale: wp.array | None = None
        self._body_mesh_max_dist: wp.array | None = None
        self._body_radius_bound: wp.array | None = None
        self._wall_speed_estimate: wp.array | None = None
        self._coupling_to_newton: wp.array | None = None
        self._previous_solid_phi: wp.array | None = None
        self._repair_uncovered_cells_enabled: bool = True
        self._repair_uncovered_density_floor: float = 1.0e-6
        self._repair_uncovered_density_ceiling: float = 10.0
        self._repair_uncovered_velocity_limit: float = 0.2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fluid_state(self) -> LbmState:
        self._ensure_states()
        return self._fluid_domain._state_in

    @property
    def rigid_state(self) -> RigidState:
        self._ensure_states()
        return self._rigid_domain._state_in

    def _ensure_states(self) -> None:
        if self._fluid_domain._state_in is None:
            self._fluid_domain.create_state()
        if self._rigid_domain._state_in is None:
            self._rigid_domain.create_state()

    def add_body_sphere(self, body_idx: int, radius: float) -> None:
        self._bodies.append(
            {"body_idx": body_idx, "shape": self._SPHERE, "radius": float(radius)}
        )
        self._body_params_dirty = True

    def add_body_box(self, body_idx: int, half_extents: tuple[float, float, float]) -> None:
        self._bodies.append(
            {
                "body_idx": body_idx,
                "shape": self._BOX,
                "half_extents": tuple(float(v) for v in half_extents),
            }
        )
        self._body_params_dirty = True

    def add_body_capsule(self, body_idx: int, radius: float, half_height: float) -> None:
        self._bodies.append(
            {
                "body_idx": body_idx,
                "shape": self._CAPSULE,
                "radius": float(radius),
                "half_height": float(half_height),
            }
        )
        self._body_params_dirty = True

    def add_body_mesh(self, body_idx: int, mesh_id: int, scale: float = 1.0) -> None:
        self._bodies.append(
            {
                "body_idx": body_idx,
                "shape": self._MESH,
                "mesh_id": int(mesh_id),
                "scale": float(scale),
            }
        )
        self._body_params_dirty = True

    def set_rigid_dynamics_enabled(self, enabled: bool) -> None:
        self._advance_rigid = bool(enabled)

    def set_wall_velocity_warning_threshold(self, threshold: float) -> None:
        """Set the warning threshold for LBM wall velocity; use <=0 to disable."""
        self._wall_velocity_warning_threshold: float = float(threshold)
        self._wall_velocity_warning_emitted: bool = False

    def set_two_way_feedback_enabled(self, enabled: bool, force_scale: float = 1.0) -> None:
        """Enable approximate LBM fluid-to-rigid feedback.

        The first two coupling phases are one-way by default.  When enabled,
        a post-step boundary scan accumulates macro-velocity momentum feedback
        into ``rigid_state.body_f`` before the optional rigid-body step.
        """
        self._two_way_feedback_enabled: bool = bool(enabled)
        self._feedback_force_scale: float = float(force_scale)

    def set_feedback_mode(self, mode: str) -> None:
        """Set the feedback force computation mode.

        Args:
            mode: ``"approx"`` for macro-velocity approximation (legacy),
                  ``"momentum_exchange"`` for strict distribution-based
                  momentum-exchange (Ladd 1994).
        """
        if mode not in ("approx", "momentum_exchange"):
            raise ValueError(f"Unsupported feedback_mode: {mode!r}. "
                             f"Expected 'approx' or 'momentum_exchange'.")
        self._feedback_mode: str = mode

    def set_uncovered_cell_repair_enabled(
        self,
        enabled: bool,
        *,
        density_floor: float | None = None,
        density_ceiling: float | None = None,
        velocity_limit: float | None = None,
    ) -> None:
        """Enable repair for LBM cells newly uncovered by moving rigid bodies.

        Newly uncovered cells are reset from stable face-neighbour fluid cells
        and their D3Q19 distributions are rebuilt at equilibrium. This avoids
        exposing stale density that was hidden inside a moving solid. The
        repair is only applied on one-way coupling steps; two-way feedback
        keeps the original distribution history for force diagnostics.
        """
        self._repair_uncovered_cells_enabled = bool(enabled)
        if density_floor is not None:
            self._repair_uncovered_density_floor = float(density_floor)
        if density_ceiling is not None:
            self._repair_uncovered_density_ceiling = float(density_ceiling)
        if velocity_limit is not None:
            self._repair_uncovered_velocity_limit = float(velocity_limit)

    def get_last_lbm_feedback_wrench(self, body_idx: int) -> np.ndarray:
        """Return the last LBM feedback wrench retained for diagnostics."""
        if self._last_lbm_feedback_wrench is None:
            rigid_state: RigidState = self.rigid_state
            if rigid_state.body_f is None:
                return np.zeros(6, dtype=np.float64)
            body_count: int = int(rigid_state.body_f.shape[0])
            self._last_lbm_feedback_wrench = np.zeros((body_count, 6), dtype=np.float64)

        body_index: int = int(body_idx)
        if body_index < 0 or body_index >= int(self._last_lbm_feedback_wrench.shape[0]):
            raise IndexError(f"body_idx {body_idx} out of range")
        return self._last_lbm_feedback_wrench[body_index].copy()

    def _shape_radius_bound(self, entry: dict) -> float:
        """Return a conservative host-side radius bound for warning estimates."""
        shape: str = str(entry["shape"])
        if shape == self._SPHERE:
            return abs(float(entry.get("radius", 0.0)))
        if shape == self._BOX:
            half_extents: tuple[float, float, float] = entry["half_extents"]
            hx: float = float(half_extents[0])
            hy: float = float(half_extents[1])
            hz: float = float(half_extents[2])
            return math.sqrt(hx * hx + hy * hy + hz * hz)
        if shape == self._CAPSULE:
            radius: float = abs(float(entry.get("radius", 0.0)))
            half_height: float = abs(float(entry.get("half_height", 0.0)))
            return radius + half_height
        if shape == self._MESH:
            return abs(float(entry.get("scale", 1.0)))
        return 0.0

    def _warn_if_lbm_wall_velocity_is_large(self, body_qd: wp.array, velocity_scale: float) -> None:
        """Warn when estimated rigid wall speed is high in LBM lattice units."""
        next_warning_step_count: int = self._wall_velocity_warning_step_count + 1
        self._wall_velocity_warning_step_count: int = next_warning_step_count
        if self._wall_velocity_warning_threshold <= 0.0 or self._wall_velocity_warning_emitted:
            return
        if (
            self._wall_velocity_warning_step_count > 1
            and self._wall_velocity_warning_step_count % self._wall_velocity_warning_check_interval != 0
        ):
            return

        if (
            len(self._bodies) == 0
            or self._coupling_to_newton is None
            or self._body_radius_bound is None
            or self._wall_speed_estimate is None
        ):
            return
        wp.launch(
            ck.estimate_lbm_wall_speed_all_bodies,
            dim=len(self._bodies),
            inputs=[
                body_qd,
                self._coupling_to_newton,
                self._body_radius_bound,
                velocity_scale,
                self._wall_speed_estimate,
            ],
        )
        wall_speed_estimate_np: np.ndarray = np.asarray(self._wall_speed_estimate.numpy(), dtype=np.float64)
        if wall_speed_estimate_np.size == 0:
            return
        estimated_lbm_wall_speed: float = float(np.max(wall_speed_estimate_np))
        if estimated_lbm_wall_speed <= self._wall_velocity_warning_threshold:
            return

        warnings.warn(
            f"Estimated LBM wall velocity {estimated_lbm_wall_speed:.6g} exceeds "
            f"{self._wall_velocity_warning_threshold:.6g} lattice units; "
            "reduce rigid world speed, reduce dt, or increase dh. "
            "This warning does not clamp the prescribed rigid motion.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._wall_velocity_warning_emitted: bool = True

    # ------------------------------------------------------------------
    # Body-parameter upload (mirrors GridLiquidRigidCoupling)
    # ------------------------------------------------------------------

    def _upload_body_params(self, device: wp.context.Device) -> None:
        if not self._body_params_dirty:
            return
        n: int = len(self._bodies)
        if n == 0:
            self._body_params_dirty = False
            return

        shape_type: list[int] = []
        sphere_radius: list[float] = []
        box_half_extents: list[wp.vec3] = []
        capsule_radius: list[float] = []
        capsule_half_height: list[float] = []
        mesh_handle: list[wp.uint64] = []
        mesh_scale: list[float] = []
        radius_bound: list[float] = []
        coupling_to_newton: list[int] = []

        for entry in self._bodies:
            shape_type.append(_SHAPE_MAP[entry["shape"]])
            coupling_to_newton.append(entry["body_idx"])
            radius_bound.append(self._shape_radius_bound(entry))
            s: str = entry["shape"]
            sphere_radius.append(entry.get("radius", 0.0) if s == self._SPHERE else 0.0)
            if s == self._BOX:
                h: tuple[float, float, float] = entry["half_extents"]
                box_half_extents.append(wp.vec3(h[0], h[1], h[2]))
            else:
                box_half_extents.append(wp.vec3(0.0, 0.0, 0.0))
            capsule_radius.append(entry.get("radius", 0.0) if s == self._CAPSULE else 0.0)
            capsule_half_height.append(entry.get("half_height", 0.0) if s == self._CAPSULE else 0.0)
            mesh_handle.append(wp.uint64(entry.get("mesh_id", 0)) if s == self._MESH else wp.uint64(0))
            mesh_scale.append(entry.get("scale", 1.0) if s == self._MESH else 1.0)

        self._body_shape_type = wp.array(np.array(shape_type, dtype=np.int32), dtype=wp.int32, device=device)
        self._body_sphere_radius = wp.array(np.array(sphere_radius, dtype=np.float32), dtype=float, device=device)
        self._body_box_half_extents = wp.array(
            box_half_extents, dtype=wp.vec3, device=device,
        )
        self._body_capsule_radius = wp.array(np.array(capsule_radius, dtype=np.float32), dtype=float, device=device)
        self._body_capsule_half_height = wp.array(np.array(capsule_half_height, dtype=np.float32), dtype=float, device=device)
        self._body_mesh_handle = wp.array(np.array(mesh_handle, dtype=np.uint64), dtype=wp.uint64, device=device)
        self._body_mesh_scale = wp.array(np.array(mesh_scale, dtype=np.float32), dtype=float, device=device)
        self._body_radius_bound: wp.array = wp.array(np.array(radius_bound, dtype=np.float32), dtype=float, device=device)
        self._wall_speed_estimate: wp.array = wp.zeros(n, dtype=float, device=device)
        self._coupling_to_newton = wp.array(np.array(coupling_to_newton, dtype=np.int32), dtype=wp.int32, device=device)
        self._body_mesh_max_dist = wp.full(n, 1.0e6, dtype=float, device=device)

        self._body_params_dirty = False

    def _ensure_uncovered_repair_buffer(self, device: wp.context.Device) -> None:
        model: LbmModel = self._fluid_domain.model
        expected_shape: tuple[int, int, int] = (int(model.nx), int(model.ny), int(model.nz))
        if self._previous_solid_phi is not None and tuple(self._previous_solid_phi.shape) == expected_shape:
            return
        self._previous_solid_phi = wp.empty(expected_shape, dtype=float, device=device)

    # ------------------------------------------------------------------
    # CompositeSimulation protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset simulation to initial state."""
        super().reset()
        self._fluid_domain.create_state()
        self._rigid_domain.create_state()

    # ------------------------------------------------------------------
    # Main timestep
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        self._ensure_states()

        fluid_state: LbmState = self._fluid_domain._state_in
        rigid_state: RigidState = self._rigid_domain._state_in
        model: LbmModel = self._fluid_domain.model
        device: wp.context.Device = model._device
        dh: float = model.dh
        dh_float: float = float(dh)
        if not math.isfinite(dh_float) or dh_float <= 0.0:
            raise ValueError(f"LBM cell size must be finite and positive, got dh={dh}")
        velocity_scale: float = float(dt) / dh_float
        if not math.isfinite(velocity_scale):
            raise ValueError(f"Invalid LBM rigid velocity scale: dt={dt}, dh={dh}")
        nx: int = int(model.nx)
        ny: int = int(model.ny)
        nz: int = int(model.nz)

        self._upload_body_params(device)

        repair_uncovered_cells: bool = self._repair_uncovered_cells_enabled and not self._two_way_feedback_enabled
        previous_solid_phi: wp.array | None = None
        if repair_uncovered_cells:
            self._ensure_uncovered_repair_buffer(device)
            previous_solid_phi = self._previous_solid_phi
            if previous_solid_phi is not None:
                wp.copy(previous_solid_phi, fluid_state.solid_phi)

        # 1. Reset solid fields on the fluid state
        fluid_state.solid_phi.fill_(1000.0)
        fluid_state.solid_body_id.fill_(-1)
        fluid_state.vel_solid_u.zero_()
        fluid_state.vel_solid_v.zero_()
        fluid_state.vel_solid_w.zero_()

        # 2. Rasterise rigid-body SDF into solid_phi / solid_body_id
        if self._body_shape_type is not None and len(self._bodies) > 0:
            wp.launch(
                ck.rasterize_all_body_sdf_warp,
                dim=(nx, ny, nz),
                inputs=[
                    fluid_state.solid_phi,
                    fluid_state.solid_body_id,
                    dh,
                    rigid_state.body_q,
                    len(self._bodies),
                    self._coupling_to_newton,
                    self._body_shape_type,
                    self._body_sphere_radius,
                    self._body_box_half_extents,
                    self._body_capsule_radius,
                    self._body_capsule_half_height,
                    self._body_mesh_handle,
                    self._body_mesh_scale,
                    self._body_mesh_max_dist,
                ],
            )

            if repair_uncovered_cells and previous_solid_phi is not None:
                stride: int = nx * ny * nz
                wp.launch(
                    ck.repair_lbm_uncovered_solid_cells,
                    dim=(nx, ny, nz),
                    inputs=[
                        previous_solid_phi,
                        fluid_state.solid_phi,
                        fluid_state.f,
                        fluid_state.density,
                        fluid_state.velocity_x,
                        fluid_state.velocity_y,
                        fluid_state.velocity_z,
                        nx,
                        ny,
                        nz,
                        stride,
                        float(model.initial_density),
                        float(self._repair_uncovered_density_floor),
                        float(self._repair_uncovered_density_ceiling),
                        float(self._repair_uncovered_velocity_limit),
                    ],
                )

            # 3. Convert world surface velocity to LBM wall velocity in vel_solid_u/v/w
            rigid_backend: object = self._rigid_domain.model._newton_backend
            body_qd: wp.array = rigid_state.body_qd
            self._warn_if_lbm_wall_velocity_is_large(body_qd, velocity_scale)
            velocity_launches: list[tuple[object, tuple[int, int, int], object, int]] = [
                (ck.embed_all_solid_velocity_u, (nx + 1, ny, nz), fluid_state.vel_solid_u, nx),
                (ck.embed_all_solid_velocity_v, (nx, ny + 1, nz), fluid_state.vel_solid_v, ny),
                (ck.embed_all_solid_velocity_w, (nx, ny, nz + 1), fluid_state.vel_solid_w, nz),
            ]
            for kernel, dim, face_arr, axis_extent in velocity_launches:
                wp.launch(
                    kernel,
                    dim=dim,
                    inputs=[
                        fluid_state.solid_phi,
                        fluid_state.solid_body_id,
                        face_arr,
                        dh,
                        axis_extent,
                        rigid_state.body_q,
                        body_qd,
                        rigid_backend.body_com,
                        velocity_scale,
                    ],
                )

        # 4. Advance fluid
        self._fluid_domain.step(dt)
        fluid_state_after: LbmState = self._fluid_domain._state_in

        # 5. Optionally accumulate LBM fluid-to-rigid feedback.
        if (
            self._two_way_feedback_enabled
            and self._body_shape_type is not None
            and len(self._bodies) > 0
            and rigid_state.body_f is not None
        ):
            rigid_backend: object = self._rigid_domain.model._newton_backend
            rigid_state.clear_forces()
            # Skip diagnostic D2H of body_f unless something reads the wrench.
            keep_wrench = self._last_lbm_feedback_wrench is not None
            pre_feedback_wrench: np.ndarray | None = None
            if keep_wrench:
                pre_feedback_wrench = np.asarray(
                    rigid_state.body_f.numpy(), dtype=np.float64
                ).copy()

            if self._feedback_mode == "momentum_exchange":
                # HOME-FREE moment path zeros ``f``; fall back to macro feedback.
                use_me = str(getattr(model, "lbm_backend", "dist")).lower() != "home_fp32"
                if use_me:
                    # Strict momentum-exchange: use f_post (after step) and f_pre (before step)
                    f_post_stream: wp.array = fluid_state_after.f  # _state_in.f after swap
                    f_pre_stream: wp.array = self._fluid_domain._state_out.f  # _state_out.f after swap
                    stride: int = nx * ny * nz
                    wp.launch(
                        ck.accumulate_lbm_momentum_exchange_all_bodies,
                        dim=(nx, ny, nz),
                        inputs=[
                            f_post_stream,
                            f_pre_stream,
                            fluid_state_after.solid_phi,
                            fluid_state_after.solid_body_id,
                            dh,
                            nx,
                            ny,
                            nz,
                            stride,
                            rigid_state.body_q,
                            rigid_backend.body_com,
                            rigid_state.body_f,
                            self._feedback_force_scale,
                        ],
                    )
                else:
                    if not getattr(self, "_home_fp32_feedback_fallback_warned", False):
                        warnings.warn(
                            "home_fp32 has no distribution ``f``; "
                            "using approx macro-velocity LBM→rigid feedback "
                            "instead of momentum_exchange.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._home_fp32_feedback_fallback_warned = True
                    wp.launch(
                        ck.accumulate_lbm_boundary_feedback_all_bodies,
                        dim=(nx, ny, nz),
                        inputs=[
                            fluid_state_after.density,
                            fluid_state_after.velocity_x,
                            fluid_state_after.velocity_y,
                            fluid_state_after.velocity_z,
                            fluid_state_after.solid_phi,
                            fluid_state_after.solid_body_id,
                            fluid_state_after.vel_solid_u,
                            fluid_state_after.vel_solid_v,
                            fluid_state_after.vel_solid_w,
                            dh,
                            nx,
                            ny,
                            nz,
                            rigid_state.body_q,
                            rigid_backend.body_com,
                            rigid_state.body_f,
                            self._feedback_force_scale,
                        ],
                    )
            else:
                # Legacy macro-velocity approximation
                wp.launch(
                    ck.accumulate_lbm_boundary_feedback_all_bodies,
                    dim=(nx, ny, nz),
                    inputs=[
                        fluid_state_after.density,
                        fluid_state_after.velocity_x,
                        fluid_state_after.velocity_y,
                        fluid_state_after.velocity_z,
                        fluid_state_after.solid_phi,
                        fluid_state_after.solid_body_id,
                        fluid_state_after.vel_solid_u,
                        fluid_state_after.vel_solid_v,
                        fluid_state_after.vel_solid_w,
                        dh,
                        nx,
                        ny,
                        nz,
                        rigid_state.body_q,
                        rigid_backend.body_com,
                        rigid_state.body_f,
                        self._feedback_force_scale,
                    ],
                )

            if keep_wrench and pre_feedback_wrench is not None:
                post_feedback_wrench: np.ndarray = np.asarray(
                    rigid_state.body_f.numpy(), dtype=np.float64
                )
                self._last_lbm_feedback_wrench = (
                    post_feedback_wrench - pre_feedback_wrench
                ).copy()
        elif self._last_lbm_feedback_wrench is not None:
            self._last_lbm_feedback_wrench.fill(0.0)

        # 6. Optionally advance rigid
        if self._advance_rigid:
            self._rigid_domain.step(dt)

        self._time += dt
