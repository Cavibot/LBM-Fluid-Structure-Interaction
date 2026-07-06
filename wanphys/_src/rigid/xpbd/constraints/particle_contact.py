# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Particle contact constraint groups: particle–shape and particle–particle."""

from __future__ import annotations

import warp as wp

from .._types import ParticleFlags
from ..constraint_base import ConstraintGroup
from ..context import ConstraintPhase, XPBDContext

# ────────────────────────────────────────────────────────────────────
# Warp kernels
# ────────────────────────────────────────────────────────────────────


@wp.kernel
def _apply_particle_shape_restitution(
    particle_x_new: wp.array(dtype=wp.vec3),
    particle_v_new: wp.array(dtype=wp.vec3),
    particle_x_old: wp.array(dtype=wp.vec3),
    particle_v_old: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    particle_ka: float,
    restitution: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    # x_new = particle_x_new[particle_index]
    v_new = particle_v_new[particle_index]
    px = particle_x_old[particle_index]
    v_old = particle_v_old[particle_index]

    X_wb = wp.transform_identity()
    # X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        # X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    # r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    rel_vel_old = wp.dot(n, v_old)
    rel_vel_new = wp.dot(n, v_new)

    if rel_vel_old < 0.0:
        # dv = -n * wp.max(-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0), 0.0)
        dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

        # compute inverse masses
        # w1 = particle_invmass[particle_index]
        # w2 = 0.0
        # if body_index >= 0:
        #     angular = wp.cross(r, n)
        #     q = wp.transform_get_rotation(X_wb)
        #     rot_angular = wp.quat_rotate_inv(q, angular)
        #     I_inv = body_I_inv[body_index]
        #     w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
        # denom = w1 + w2
        # if denom == 0.0:
        #     return

        wp.atomic_add(particle_v_out, tid, dv)


@wp.kernel
def _solve_particle_shape_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    # body velocity
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # normal
    lambda_n = c
    delta_n = n * lambda_n

    # friction
    vn = wp.dot(n, v)
    vt = v - n * vn

    # compute inverse masses
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    delta_total = (delta_f - delta_n) / denom * relaxation

    wp.atomic_add(delta, particle_index, w1 * delta_total)

    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_total, delta_t))


@wp.kernel
def _solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) != 0 and index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = v - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


# ────────────────────────────────────────────────────────────────────
# Constraint group implementations
# ────────────────────────────────────────────────────────────────────


class ParticleShapeContact(ConstraintGroup):
    """Particle–rigid-shape contact constraint with Coulomb friction.

    Resolves penetrations between particles and rigid body shapes.  Also
    writes reaction forces into ``ctx.body_deltas`` so that the rigid bodies
    are affected by the contact as well.

    The optional :meth:`apply_restitution` pass corrects particle velocities
    after all XPBD iterations have finished.
    """

    phase = ConstraintPhase.PARTICLE

    def __init__(self, relaxation: float = 0.9, enable_restitution: bool = False) -> None:
        self.relaxation = relaxation
        self.enable_restitution = enable_restitution

    def is_active(self, model, contacts) -> bool:
        return model.particle_count > 0 and model.shape_count > 0 and contacts is not None

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        contacts = ctx.contacts
        wp.launch(
            kernel=_solve_particle_shape_contacts,
            dim=contacts.soft_contact_max,
            inputs=[
                ctx.particle_q,
                ctx.particle_qd,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_flags,
                ctx.body_q,
                ctx.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.shape_body,
                model.shape_material_mu,
                model.soft_contact_mu,
                model.particle_adhesion,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_max,
                ctx.dt,
                self.relaxation,
            ],
            outputs=[ctx.particle_deltas, ctx.body_deltas],
            device=model.device,
        )

    def apply_restitution(self, ctx: XPBDContext) -> None:
        if not self.enable_restitution:
            return
        model = ctx.model
        contacts = ctx.contacts
        if model.particle_count == 0 or contacts is None:
            return
        wp.launch(
            kernel=_apply_particle_shape_restitution,
            dim=model.particle_count,
            inputs=[
                ctx.particle_q,
                ctx.particle_qd,
                ctx.particle_q_init,
                ctx.particle_qd_init,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_flags,
                ctx.body_q,
                ctx.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.shape_body,
                model.particle_adhesion,
                model.soft_contact_restitution,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_max,
                ctx.dt,
                self.relaxation,
            ],
            outputs=[ctx.state_out.particle_qd],
            device=model.device,
        )


class ParticleParticleContact(ConstraintGroup):
    """Particle–particle contact constraint using spatial hashing.

    Detects and resolves contacts between particles via a
    :class:`wp.HashGrid`.  Supports friction and cohesion.
    """

    phase = ConstraintPhase.PARTICLE

    def __init__(self, relaxation: float = 0.9) -> None:
        self.relaxation = relaxation

    def is_active(self, model, contacts) -> bool:
        return model.particle_count > 1 and model.particle_max_radius > 0.0

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        wp.launch(
            kernel=_solve_particle_particle_contacts,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                ctx.particle_q,
                ctx.particle_qd,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_flags,
                model.particle_mu,
                model.particle_cohesion,
                model.particle_max_radius,
                ctx.dt,
                self.relaxation,
            ],
            outputs=[ctx.particle_deltas],
            device=model.device,
        )
