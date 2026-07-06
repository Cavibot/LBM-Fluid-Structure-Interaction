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

"""Rigid-body contact constraint group: normal + friction + restitution."""

from __future__ import annotations

import warp as wp

from ..constraint_base import ConstraintGroup
from ..math_utils import compute_contact_constraint_delta, velocity_at_point
from ..context import ConstraintPhase, XPBDContext

# ────────────────────────────────────────────────────────────────────
# Warp helper functions (@wp.func)
# ────────────────────────────────────────────────────────────────────



# compute_contact_constraint_delta moved to math_utils.py — imported above.


# ────────────────────────────────────────────────────────────────────
# Warp kernels
# ────────────────────────────────────────────────────────────────────


@wp.kernel
def _solve_body_contact_positions(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    shape_material_mu_torsional: wp.array(dtype=float),
    shape_material_mu_rolling: wp.array(dtype=float),
    relaxation: float,
    dt: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
    contact_inv_weight: wp.array(dtype=float),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return

    # find body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    # compute body position in world space
    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])

    thickness = contact_thickness0[tid] + contact_thickness1[tid]
    n = -contact_normal[tid]
    d = wp.dot(n, bx_b - bx_a) - thickness

    if d >= 0.0:
        return

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    # angular velocities
    omega_a = wp.vec3(0.0)
    omega_b = wp.vec3(0.0)
    # contact offset in body frame
    offset_a = contact_offset0[tid]
    offset_b = contact_offset1[tid]

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        com_a = body_com[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        omega_a = wp.spatial_bottom(body_qd[body_a])

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        com_b = body_com[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        omega_b = wp.spatial_bottom(body_qd[body_b])

    # use average contact material properties
    mat_nonzero = 0
    mu = 0.0
    torsional_friction = 0.0
    rolling_friction = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        mu += shape_material_mu[shape_a]
        torsional_friction += shape_material_mu_torsional[shape_a]
        rolling_friction += shape_material_mu_rolling[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        mu += shape_material_mu[shape_b]
        torsional_friction += shape_material_mu_torsional[shape_b]
        rolling_friction += shape_material_mu_rolling[shape_b]
    if mat_nonzero > 0:
        mu /= float(mat_nonzero)
        torsional_friction /= float(mat_nonzero)
        rolling_friction /= float(mat_nonzero)

    r_a = bx_a - wp.transform_point(X_wb_a, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    angular_a = -wp.cross(r_a, n)
    angular_b = wp.cross(r_b, n)

    if contact_inv_weight:
        if body_a >= 0:
            wp.atomic_add(contact_inv_weight, body_a, 1.0)
        if body_b >= 0:
            wp.atomic_add(contact_inv_weight, body_b, 1.0)

    lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -n, n, angular_a, angular_b, relaxation, dt
    )

    lin_delta_a = -n * lambda_n
    lin_delta_b = n * lambda_n
    ang_delta_a = angular_a * lambda_n
    ang_delta_b = angular_b * lambda_n

    # linear friction
    if mu > 0.0:
        # add on displacement from surface offsets, this ensures we include any rotational effects due to thickness from feature
        # need to use the current rotation to account for friction due to angular effects (e.g.: slipping contact)
        bx_a += wp.transform_vector(X_wb_a, offset_a)
        bx_b += wp.transform_vector(X_wb_b, offset_b)

        # update delta
        delta = bx_b - bx_a
        friction_delta = delta - wp.dot(n, delta) * n

        perp = wp.normalize(friction_delta)

        r_a = bx_a - wp.transform_point(X_wb_a, com_a)
        r_b = bx_b - wp.transform_point(X_wb_b, com_b)

        angular_a = -wp.cross(r_a, perp)
        angular_b = wp.cross(r_b, perp)

        err = wp.length(friction_delta)

        if err > 0.0:
            lambda_fr = compute_contact_constraint_delta(
                err,
                X_wb_a,
                X_wb_b,
                m_inv_a,
                m_inv_b,
                I_inv_a,
                I_inv_b,
                -perp,
                perp,
                angular_a,
                angular_b,
                relaxation,
                dt,
            )

            # limit friction based on incremental normal force, good approximation to limiting on total force
            lambda_fr = wp.max(lambda_fr, -lambda_n * mu)

            lin_delta_a -= perp * lambda_fr
            lin_delta_b += perp * lambda_fr

            ang_delta_a += angular_a * lambda_fr
            ang_delta_b += angular_b * lambda_fr

    delta_omega = omega_b - omega_a

    if torsional_friction > 0.0:
        err = wp.dot(delta_omega, n) * dt

        if wp.abs(err) > 0.0:
            lin = wp.vec3(0.0)
            lambda_torsion = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -n, n, relaxation, dt
            )

            lambda_torsion = wp.clamp(lambda_torsion, -lambda_n * torsional_friction, lambda_n * torsional_friction)

            ang_delta_a -= n * lambda_torsion
            ang_delta_b += n * lambda_torsion

    if rolling_friction > 0.0:
        delta_omega -= wp.dot(n, delta_omega) * n
        err = wp.length(delta_omega) * dt
        if err > 0.0:
            lin = wp.vec3(0.0)
            roll_n = wp.normalize(delta_omega)
            lambda_roll = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -roll_n, roll_n, relaxation, dt
            )

            lambda_roll = wp.max(lambda_roll, -lambda_n * rolling_friction)

            ang_delta_a -= roll_n * lambda_roll
            ang_delta_b += roll_n * lambda_roll

    if body_a >= 0:
        wp.atomic_add(deltas, body_a, wp.spatial_vector(lin_delta_a, ang_delta_a))
    if body_b >= 0:
        wp.atomic_add(deltas, body_b, wp.spatial_vector(lin_delta_b, ang_delta_b))


@wp.kernel
def _update_body_velocities(
    poses: wp.array(dtype=wp.transform),
    poses_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    pose = poses[tid]
    pose_prev = poses_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)

    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Update body velocities according to Alg. 2
    # XXX we consider the body COM as the origin of the body frame
    x_com = x + wp.quat_rotate(q, body_com[tid])
    x_com_prev = x_prev + wp.quat_rotate(q_prev, body_com[tid])

    # XXX consider the velocity of the COM
    v = (x_com - x_com_prev) / dt
    dq = q * wp.quat_inverse(q_prev)

    omega = 2.0 / dt * wp.vec3(dq[0], dq[1], dq[2])
    if dq[3] < 0.0:
        omega = -omega

    qd_out[tid] = wp.spatial_vector(v, omega)


@wp.kernel
def _apply_rigid_restitution(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_material_restitution: wp.array(dtype=float),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    contact_inv_weight: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1

    # use average contact material properties
    mat_nonzero = 0
    restitution = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        restitution /= float(mat_nonzero)
    if body_a == body_b:
        return

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # body to world transform
    X_wb_a_prev = wp.transform_identity()
    X_wb_b_prev = wp.transform_identity()
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # previous velocity at contact points
    v_a = wp.vec3(0.0)
    v_b = wp.vec3(0.0)
    # new velocity at contact points
    v_a_new = wp.vec3(0.0)
    v_b_new = wp.vec3(0.0)
    # inverse mass used to compute the impulse
    inv_mass = 0.0

    if body_a >= 0:
        X_wb_a_prev = body_q_prev[body_a]
        # X_wb_a = body_q[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        com_a = body_com[body_a]

    if body_b >= 0:
        X_wb_b_prev = body_q_prev[body_b]
        # X_wb_b = body_q[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        com_b = body_com[body_b]

    # compute body position in world space
    bx_a = wp.transform_point(X_wb_a_prev, contact_point0[tid] + contact_offset0[tid])
    bx_b = wp.transform_point(X_wb_b_prev, contact_point1[tid] + contact_offset1[tid])

    thickness = contact_thickness0[tid] + contact_thickness1[tid]
    n = contact_normal[tid]
    d = -wp.dot(n, bx_b - bx_a) - thickness
    if d >= 0.0:
        return

    r_a = bx_a - wp.transform_point(X_wb_a_prev, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b_prev, com_b)

    rxn_a = wp.vec3(0.0)
    rxn_b = wp.vec3(0.0)
    if body_a >= 0:
        v_a = velocity_at_point(body_qd_prev[body_a], r_a) + gravity[0] * dt
        v_a_new = velocity_at_point(body_qd[body_a], r_a)
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        rxn_a = wp.quat_rotate_inv(q_a, wp.cross(r_a, n))
        # Eq. 2
        inv_mass_a = m_inv_a + wp.dot(rxn_a, I_inv_a * rxn_a)
        # if contact_inv_weight:
        #     if contact_inv_weight[body_a] > 0.0:
        #         inv_mass_a *= contact_inv_weight[body_a]
        inv_mass += inv_mass_a
    if body_b >= 0:
        v_b = velocity_at_point(body_qd_prev[body_b], r_b) + gravity[0] * dt
        v_b_new = velocity_at_point(body_qd[body_b], r_b)
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        rxn_b = wp.quat_rotate_inv(q_b, wp.cross(r_b, n))
        # Eq. 3
        inv_mass_b = m_inv_b + wp.dot(rxn_b, I_inv_b * rxn_b)
        # if contact_inv_weight:
        #     if contact_inv_weight[body_b] > 0.0:
        #         inv_mass_b *= contact_inv_weight[body_b]
        inv_mass += inv_mass_b

    if inv_mass == 0.0:
        return

    # Eq. 29
    rel_vel_old = wp.dot(n, v_a - v_b)
    rel_vel_new = wp.dot(n, v_a_new - v_b_new)

    if rel_vel_old >= 0.0:
        return

    # Eq. 34
    dv = (-rel_vel_new - restitution * rel_vel_old) / inv_mass

    # Eq. 33
    if body_a >= 0:
        dv_a = dv
        # if contact_inv_weight:
        #     if contact_inv_weight[body_a] > 0.0:
        #         dv_a *= contact_inv_weight[body_a]
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        dq = wp.quat_rotate(q_a, I_inv_a * rxn_a * dv_a)
        wp.atomic_add(deltas, body_a, wp.spatial_vector(n * m_inv_a * dv_a, dq))

    if body_b >= 0:
        dv_b = -dv
        # if contact_inv_weight:
        #     if contact_inv_weight[body_b] > 0.0:
        #         dv_b *= contact_inv_weight[body_b]
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        dq = wp.quat_rotate(q_b, I_inv_b * rxn_b * dv_b)
        wp.atomic_add(deltas, body_b, wp.spatial_vector(n * m_inv_b * dv_b, dq))


@wp.kernel
def _apply_body_delta_velocities(
    deltas: wp.array(dtype=wp.spatial_vector),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    wp.atomic_add(qd_out, tid, deltas[tid])


# ────────────────────────────────────────────────────────────────────
# Constraint group implementation
# ────────────────────────────────────────────────────────────────────


class RigidContactConstraint(ConstraintGroup):
    """Rigid-body contact constraint (normal, friction, torsional/rolling friction).

    Also owns the optional post-iteration restitution pass and the
    position-based velocity update for rigid bodies.

    Execution semantics (per iteration):
      * ``body_deltas`` is **independently zeroed** before :meth:`project`.
      * The solver applies deltas with the optional ``contact_inv_weight``.

    Attributes:
        relaxation: Over-relaxation factor for the contact solver.
        con_weighting: Whether to use per-body contact inverse weighting.
        enable_restitution: Whether to run the restitution pass after iterations.
        compute_velocity_from_position_delta: Whether to recompute ``body_qd``
            from the position change after all iterations.
    """

    phase = ConstraintPhase.RIGID_CONTACT

    def __init__(
        self,
        relaxation: float = 0.8,
        con_weighting: bool = True,
        enable_restitution: bool = False,
        compute_velocity_from_position_delta: bool = False,
    ) -> None:
        self.relaxation = relaxation
        self.con_weighting = con_weighting
        self.enable_restitution = enable_restitution
        self.compute_velocity_from_position_delta = compute_velocity_from_position_delta

    def is_active(self, model, contacts) -> bool:
        return model.body_count > 0 and contacts is not None

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        contacts = ctx.contacts
        assert contacts is not None  # guaranteed by is_active

        # Zero inv_weight for this iteration if weighting is enabled
        if self.con_weighting and ctx.rigid_contact_inv_weight is not None:
            ctx.rigid_contact_inv_weight.zero_()

        wp.launch(
            kernel=_solve_body_contact_positions,
            dim=contacts.rigid_contact_max,
            inputs=[
                ctx.body_q,
                ctx.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                model.shape_material_mu,
                model.shape_material_mu_torsional,
                model.shape_material_mu_rolling,
                self.relaxation,
                ctx.dt,
            ],
            outputs=[
                ctx.body_deltas,
                ctx.rigid_contact_inv_weight,
            ],
            device=model.device,
        )

        # Snapshot inv_weight from the first iteration (for restitution)
        if self.enable_restitution and iteration == 0:
            if self.con_weighting and ctx.rigid_contact_inv_weight is not None:
                ctx.rigid_contact_inv_weight_init = wp.clone(ctx.rigid_contact_inv_weight)
            else:
                ctx.rigid_contact_inv_weight_init = None

    def apply_restitution(self, ctx: XPBDContext) -> None:
        """Apply rigid-body restitution + optional position-based velocity update."""
        model = ctx.model

        # --- Position-based velocity update (optional) ---
        if self.compute_velocity_from_position_delta and not ctx.requires_grad:
            out_body_qd = ctx.state_out.body_qd
            wp.launch(
                kernel=_update_body_velocities,
                dim=model.body_count,
                inputs=[ctx.state_out.body_q, ctx.body_q_init, model.body_com, ctx.dt],
                outputs=[out_body_qd],
                device=model.device,
            )

        # --- Rigid restitution ---
        if not self.enable_restitution:
            return

        contacts = ctx.contacts
        if contacts is None:
            return

        ctx.body_deltas.zero_()

        wp.launch(
            kernel=_apply_rigid_restitution,
            dim=contacts.rigid_contact_max,
            inputs=[
                ctx.state_out.body_q,
                ctx.state_out.body_qd,
                ctx.body_q_init,
                ctx.body_qd_init,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                model.shape_material_restitution,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                ctx.rigid_contact_inv_weight_init,
                model.gravity,
                ctx.dt,
            ],
            outputs=[
                ctx.body_deltas,
            ],
            device=model.device,
        )

        wp.launch(
            kernel=_apply_body_delta_velocities,
            dim=model.body_count,
            inputs=[
                ctx.body_deltas,
            ],
            outputs=[ctx.state_out.body_qd],
            device=model.device,
        )
