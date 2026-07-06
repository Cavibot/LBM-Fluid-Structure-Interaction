import numpy as np
import warp as wp
from pxr import Usd
from collections.abc import Callable
from typing import Any

import newton
import newton.examples
import newton.viewer
from newton.usd import get_mesh, get_scale, get_transform
from wanphys._src.core import CompositeSimulation
from wanphys._src.fluid.fluid_particle.domain import ParticleFluidDomain
from wanphys._src.fluid.fluid_particle.pbf.model import PBFModel
from wanphys._src.rigid import RigidDomain, RigidModelBuilder, ShapeConfig
from wanphys.collision import CollisionPipeline
from wanphys.examples import get_asset

try:
    from newton.geometry import ParticleFlags
except ImportError:
    from newton._src.geometry import ParticleFlags


@wp.kernel
def apply_twoway_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=int),
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    margin: float,
    max_push_frac: float,
    vel_damp: float,
    friction: float,
    particle_mass: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    sim_dt: float,
):
    tid = wp.tid()
    count = soft_contact_count[0]
    if tid >= count:
        return

    p = soft_contact_particle[tid]
    s = soft_contact_shape[tid]
    if p < 0 or s < 0:
        return
    if (particle_flags[p] & ParticleFlags.ACTIVE) == 0:
        return

    n = soft_contact_normal[tid]
    n_len = wp.length(n)
    if n_len < 1.0e-6:
        return
    n = n / n_len

    body_pos_local = soft_contact_body_pos[tid]
    body_vel_local = soft_contact_body_vel[tid]

    rigid = shape_body[s]
    X_wb = wp.transform_identity()
    if rigid >= 0:
        X_wb = body_q[rigid]

    surf_w = wp.transform_point(X_wb, body_pos_local)

    com_w = wp.vec3(0.0, 0.0, 0.0)
    lever = wp.vec3(0.0, 0.0, 0.0)
    v_body_linear = wp.vec3(0.0, 0.0, 0.0)
    v_body_angular = wp.vec3(0.0, 0.0, 0.0)
    if rigid >= 0:
        com_w = wp.transform_point(X_wb, body_com[rigid])
        lever = surf_w - com_w
        body_sv = body_qd[rigid]
        v_body_linear = wp.spatial_top(body_sv)
        v_body_angular = wp.spatial_bottom(body_sv)

    body_vel_w = v_body_linear + wp.transform_vector(X_wb, body_vel_local) + wp.cross(v_body_angular, lever)

    x = particle_q[p]
    v = particle_qd[p]
    r = particle_radius[p]
    v_old = v

    d = wp.dot(x - surf_w, n)
    pen = (margin + r) - d
    if pen <= 0.0:
        return

    max_push = wp.max(0.0, max_push_frac) * r
    push = wp.min(pen, max_push)
    x = x + n * push

    rel_v = v - body_vel_w

    vn = wp.dot(rel_v, n)
    if vn < 0.0:
        rel_v = rel_v - vn * n

    vt = rel_v - n * wp.dot(rel_v, n)
    vt_len = wp.length(vt)
    if vt_len > 1.0e-6 and vn < 0.0:
        max_reduce = friction * (-vn)
        reduce = wp.min(vt_len, max_reduce)
        vt = vt - vt * (reduce / vt_len)
        rel_v = vt

    rel_v = rel_v * (1.0 - vel_damp)
    v = rel_v + body_vel_w

    particle_q[p] = x
    particle_qd[p] = v

    if rigid >= 0:
        J = (v_old - v) * particle_mass[p]
        F = J / sim_dt
        torque = wp.cross(lever, F)
        wp.atomic_add(body_f, rigid, wp.spatial_vector(F, torque))


class PBFRigidTwoWayCoupling(CompositeSimulation):
    def __init__(
        self,
        rigid: RigidDomain,
        pbf: ParticleFluidDomain,
        contact_iters: int = 3,
        contact_margin: float = 0.002,
        contact_max_push_frac: float = 0.30,
        contact_vel_damp: float = 0.015,
        contact_friction: float = 0.20,
        external_forces: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self.rigid: RigidDomain = rigid
        self.pbf: ParticleFluidDomain = pbf
        self.contact_iters: int = int(contact_iters)
        self.contact_margin: float = float(contact_margin)
        self.contact_max_push_frac: float = float(contact_max_push_frac)
        self.contact_vel_damp: float = float(contact_vel_damp)
        self.contact_friction: float = float(contact_friction)
        self.external_forces: Callable[[float], None] | None = external_forces

        self.rigid.create_state()
        self.pbf.create_state()

    def step(self, dt: float) -> None:
        rigid_state = self.rigid.state

        rigid_state.clear_forces()
        if self.external_forces is not None:
            self.external_forces(dt)

        self.pbf.step(dt)
        fluid_state = self.pbf.state
        fluid_model = self.pbf.model

        for _ in range(max(self.contact_iters, 0)):
            contacts = CollisionPipeline.collide_rigid_fluid(self.rigid, self.pbf)
            if contacts is None:
                break
            wp.launch(
                apply_twoway_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    fluid_state.particle_q,
                    fluid_state.particle_qd,
                    fluid_model.particle_radius,
                    fluid_model.particle_flags,
                    rigid_state.body_q,
                    rigid_state.body_qd,
                    self.rigid.model.shape_body,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.contact_margin,
                    self.contact_max_push_frac,
                    self.contact_vel_damp,
                    self.contact_friction,
                    fluid_model.particle_mass,
                    rigid_state.body_f,
                    self.rigid.model.body_com,
                    dt,
                ],
                device=self.rigid.model.device,
            )


        rigid_contacts = CollisionPipeline.collide_rigid(self.rigid)
        self.rigid.step(dt, contacts=rigid_contacts)
        rigid_state.clear_forces()

        self._time += dt

    def reset(self) -> None:
        super().reset()
        self.rigid.create_state()
        self.pbf.create_state()


PBFDomain: type[ParticleFluidDomain] = ParticleFluidDomain
