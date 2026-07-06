"""Two-way coupling between the FLIP MAC-grid liquid solver and rigid bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from wanphys._src.core.composite import CompositeSimulation
from wanphys.collision import CollisionPipeline
from . import coupling_kernels as ck

if TYPE_CHECKING:
    from wanphys._src.fluid import FluidGridLiquidDomain, FluidGridLiquidState
    from wanphys._src.rigid import RigidDomain, RigidState


_SHAPE_MAP = {
    "sphere": ck._SHAPE_SPHERE,
    "box": ck._SHAPE_BOX,
    "capsule": ck._SHAPE_CAPSULE,
    "mesh": ck._SHAPE_MESH,
}


class GridLiquidRigidCoupling(CompositeSimulation):
    """Owns and advances a coupled grid-liquid + rigid-body simulation."""

    _SPHERE = "sphere"
    _BOX = "box"
    _CAPSULE = "capsule"
    _MESH = "mesh"

    def __init__(
        self,
        fluid_domain: FluidGridLiquidDomain,
        rigid_domain: RigidDomain,
    ):
        super().__init__()
        self._fluid_domain = fluid_domain
        self._rigid_domain = rigid_domain
        self._density = float(fluid_domain.model.density)
        self._rigid_shape_query = CollisionPipeline.get_shape_query(self._rigid_domain)
        self._advance_rigid = True

        self._bodies: list[dict] = []
        self._body_params_dirty = True

        self._body_shape_type: wp.array | None = None
        self._body_sphere_radius: wp.array | None = None
        self._body_box_half_extents: wp.array | None = None
        self._body_capsule_radius: wp.array | None = None
        self._body_capsule_half_height: wp.array | None = None
        self._body_mesh_handle: wp.array | None = None
        self._body_mesh_scale: wp.array | None = None
        self._body_mesh_max_dist: wp.array | None = None
        self._coupling_to_newton: wp.array | None = None

    @property
    def fluid_state(self) -> FluidGridLiquidState:
        self._ensure_states()
        return self._fluid_domain._state_in

    @property
    def fluid_state_out(self) -> FluidGridLiquidState:
        self._ensure_states()
        return self._fluid_domain._state_out

    @property
    def rigid_state(self) -> RigidState:
        self._ensure_states()
        return self._rigid_domain._state_in

    def _ensure_states(self) -> None:
        if self._fluid_domain._state_in is None or self._fluid_domain._state_out is None:
            self._fluid_domain.create_state()
        if self._rigid_domain._state_in is None or self._rigid_domain._state_out is None:
            self._rigid_domain.create_state()

    def add_body_sphere(self, body_idx: int, radius: float) -> None:
        self._bodies.append(
            {
                "body_idx": body_idx,
                "shape": self._SPHERE,
                "radius": float(radius),
            }
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

    def _upload_body_params(self, device: wp.context) -> None:
        if not self._body_params_dirty:
            return
        n = len(self._bodies)
        if n == 0:
            self._body_params_dirty = False
            return

        shape_type = []
        sphere_radius = []
        box_half_extents = []
        capsule_radius = []
        capsule_half_height = []
        mesh_handle = []
        mesh_scale = []
        coupling_to_newton = []

        default_sphere_r = 0.0
        default_box_hx = wp.vec3(0.0, 0.0, 0.0)
        default_capsule_r = 0.0
        default_capsule_hh = 0.0
        default_mesh_h = wp.uint64(0)
        default_mesh_s = 1.0

        for entry in self._bodies:
            shape_type.append(_SHAPE_MAP[entry["shape"]])
            coupling_to_newton.append(entry["body_idx"])

            is_sphere = entry["shape"] == self._SPHERE
            is_box = entry["shape"] == self._BOX
            is_capsule = entry["shape"] == self._CAPSULE
            is_mesh = entry["shape"] == self._MESH

            sphere_radius.append(entry.get("radius", default_sphere_r) if is_sphere else default_sphere_r)
            if is_box:
                hx, hy, hz = entry["half_extents"]
                box_half_extents.append(wp.vec3(hx, hy, hz))
            else:
                box_half_extents.append(default_box_hx)

            capsule_radius.append(entry.get("radius", default_capsule_r) if is_capsule else default_capsule_r)
            capsule_half_height.append(entry.get("half_height", default_capsule_hh) if is_capsule else default_capsule_hh)

            if is_mesh:
                mesh_handle.append(wp.uint64(entry["mesh_id"]))
                mesh_scale.append(entry.get("scale", 1.0))
            else:
                mesh_handle.append(default_mesh_h)
                mesh_scale.append(default_mesh_s)

        self._body_shape_type = wp.array(shape_type, dtype=wp.int32, device=device)
        self._body_sphere_radius = wp.array(sphere_radius, dtype=float, device=device)
        self._body_box_half_extents = wp.array(box_half_extents, dtype=wp.vec3, device=device)
        self._body_capsule_radius = wp.array(capsule_radius, dtype=float, device=device)
        self._body_capsule_half_height = wp.array(capsule_half_height, dtype=float, device=device)
        self._body_mesh_handle = wp.array(mesh_handle, dtype=wp.uint64, device=device)
        self._body_mesh_scale = wp.array(mesh_scale, dtype=float, device=device)
        self._body_mesh_max_dist = wp.full(n, 1.0e6, dtype=float, device=device)
        self._coupling_to_newton = wp.array(coupling_to_newton, dtype=wp.int32, device=device)

        self._body_params_dirty = False

    def step(self, dt: float) -> None:
        self._ensure_states()

        fluid_state = self._fluid_domain._state_in
        rigid_state = self._rigid_domain._state_in

        model = self._fluid_domain.model
        dh = model.dh
        nx, ny, nz = model.nx, model.ny, model.nz
        grid_dim = (nx, ny, nz)
        rigid_backend = self._rigid_domain.model._newton_backend

        device = model._device
        self._upload_body_params(device)

        # Pressure coupling contributes a transient external wrench each step.
        # Reset the accumulator before stamping this frame's force/torque so
        # stale values do not persist after the body leaves the fluid.
        rigid_state.clear_forces()

        self._prepare_fluid_boundary_conditions(
            fluid_state=fluid_state,
            rigid_state=rigid_state,
            rigid_backend=rigid_backend,
            grid_dim=grid_dim,
            dh=dh,
        )

        self._accumulate_pressure_force(
            fluid_state=fluid_state,
            rigid_state=rigid_state,
            rigid_backend=rigid_backend,
            grid_dim=grid_dim,
            dh=dh,
            nx=nx,
            ny=ny,
            nz=nz,
        )

        self._fluid_domain.step(dt)
        if self._advance_rigid:
            self._rigid_domain.step(dt)
        self._time += dt

    def _prepare_fluid_boundary_conditions(
        self,
        fluid_state: FluidGridLiquidState,
        rigid_state: RigidState,
        rigid_backend,
        grid_dim: tuple[int, int, int],
        dh: float,
    ) -> None:
        fluid_state.solid_phi.fill_(1000.0)
        fluid_state.solid_body_id.fill_(-1)
        fluid_state.vel_solid_u.zero_()
        fluid_state.vel_solid_v.zero_()
        fluid_state.vel_solid_w.zero_()

        if not self._bodies:
            return

        wp.launch(
            ck.rasterize_all_body_sdf_warp,
            dim=grid_dim,
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

        # wp.launch(
        #     ck.rasterize_all_body_sdf,
        #     dim=grid_dim,
        #     inputs=[
        #         fluid_state.solid_phi,
        #         fluid_state.solid_body_id,
        #         dh,
        #         rigid_state.body_q,
        #         len(self._bodies),
        #         self._rigid_shape_query.data,
        #         self._coupling_to_newton,
        #     ],
        # )

        wp.launch(
            ck.embed_all_solid_velocity_u,
            dim=(grid_dim[0] + 1, grid_dim[1], grid_dim[2]),
            inputs=[
                fluid_state.solid_phi,
                fluid_state.solid_body_id,
                fluid_state.vel_solid_u,
                dh,
                grid_dim[0],
                rigid_state.body_q,
                rigid_state.body_qd,
                rigid_backend.body_com,
                1.0,
            ],
        )
        wp.launch(
            ck.embed_all_solid_velocity_v,
            dim=(grid_dim[0], grid_dim[1] + 1, grid_dim[2]),
            inputs=[
                fluid_state.solid_phi,
                fluid_state.solid_body_id,
                fluid_state.vel_solid_v,
                dh,
                grid_dim[1],
                rigid_state.body_q,
                rigid_state.body_qd,
                rigid_backend.body_com,
                1.0,
            ],
        )
        wp.launch(
            ck.embed_all_solid_velocity_w,
            dim=(grid_dim[0], grid_dim[1], grid_dim[2] + 1),
            inputs=[
                fluid_state.solid_phi,
                fluid_state.solid_body_id,
                fluid_state.vel_solid_w,
                dh,
                grid_dim[2],
                rigid_state.body_q,
                rigid_state.body_qd,
                rigid_backend.body_com,
                1.0,
            ],
        )

    def _accumulate_pressure_force(
        self,
        fluid_state: FluidGridLiquidState,
        rigid_state: RigidState,
        rigid_backend,
        grid_dim: tuple[int, int, int],
        dh: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        if rigid_state.body_count <= 0 or not self._bodies:
            return

        pressure_scale = self._density

        wp.launch(
            ck.accumulate_pressure_force_all_bodies,
            dim=grid_dim,
            inputs=[
                fluid_state.pressure,
                fluid_state.cell_type,
                fluid_state.solid_phi,
                fluid_state.solid_body_id,
                dh,
                nx,
                ny,
                nz,
                rigid_state.body_q,
                rigid_backend.body_com,
                rigid_state.body_f,
                pressure_scale,
            ],
        )

    def reset(self) -> None:
        super().reset()
        self._fluid_domain.create_state()
        self._rigid_domain.create_state()
