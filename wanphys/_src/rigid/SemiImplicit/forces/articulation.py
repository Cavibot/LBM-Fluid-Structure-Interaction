# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation (joint / constraint) force kernels.

Evaluates penalty-based constraint forces for all supported articulation
types.  Two evaluation strategies are available:

1. **Unified kernel** – a single GPU kernel that handles every joint type
   with branching.  Works for all types including D6.
2. **Dispatched kernels** – joints are pre-sorted by type on CPU; a
   specialised kernel is launched per type, eliminating branch divergence.
   D6 joints are *not* supported on this path.

The :class:`ArticulationDispatcher` handles the pre-sorting and per-type
kernel launch logic.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import warp as wp

from ..common import ArticulationType, quat_to_euler_xyz, quat_twist_component


# =====================================================================
# Single-DOF spring response
# =====================================================================


@wp.func
def dof_spring_response(
    q: float,
    qd: float,
    target_pos: float,
    target_vel: float,
    target_ke: float,
    target_kd: float,
    lim_lo: float,
    lim_hi: float,
    lim_ke: float,
    lim_kd: float,
) -> float:
    """PD controller + joint-limit forces for a single DOF."""

    lim_f = 0.0
    damp_f = 0.0
    ctrl_f = 0.0

    ctrl_f = target_ke * (target_pos - q) + target_kd * (target_vel - qd)

    if q < lim_lo:
        lim_f = lim_ke * (lim_lo - q)
        damp_f = -lim_kd * qd
        ctrl_f = 0.0
    elif q > lim_hi:
        lim_f = lim_ke * (lim_hi - q)
        damp_f = -lim_kd * qd
        ctrl_f = 0.0

    return lim_f + damp_f + ctrl_f


# =====================================================================
# Unified articulation kernel (all joint types, with branching)
# =====================================================================


@wp.kernel
def articulated_constraint_kernel(
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    jnt_dof_start: wp.array(dtype=int),
    jnt_type: wp.array(dtype=int),
    jnt_enabled: wp.array(dtype=bool),
    jnt_child: wp.array(dtype=int),
    jnt_parent: wp.array(dtype=int),
    jnt_tf_p: wp.array(dtype=wp.transform),
    jnt_tf_c: wp.array(dtype=wp.transform),
    jnt_axis: wp.array(dtype=wp.vec3),
    jnt_dof_dim: wp.array(dtype=int, ndim=2),
    jnt_act_f: wp.array(dtype=float),
    jnt_target_pos: wp.array(dtype=float),
    jnt_target_vel: wp.array(dtype=float),
    jnt_target_ke: wp.array(dtype=float),
    jnt_target_kd: wp.array(dtype=float),
    jnt_lim_lo: wp.array(dtype=float),
    jnt_lim_hi: wp.array(dtype=float),
    jnt_lim_ke: wp.array(dtype=float),
    jnt_lim_kd: wp.array(dtype=float),
    attach_ke: float,
    attach_kd: float,
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Evaluate constraint forces for one articulation of any type."""
    idx = wp.tid()
    jtype = jnt_type[idx]

    body_c = jnt_child[idx]
    body_p = jnt_parent[idx]

    if not jnt_enabled[idx]:
        return

    dof0 = jnt_dof_start[idx]
    if jtype == ArticulationType.FREE or jtype == ArticulationType.DISTANCE:
        wrench = wp.spatial_vector(
            jnt_act_f[dof0 + 0],
            jnt_act_f[dof0 + 1],
            jnt_act_f[dof0 + 2],
            jnt_act_f[dof0 + 3],
            jnt_act_f[dof0 + 4],
            jnt_act_f[dof0 + 5],
        )
        wp.atomic_add(body_wrench, body_c, wrench)
        return

    tf_pj = jnt_tf_p[idx]
    tf_cj = jnt_tf_c[idx]

    tf_world_p = tf_pj
    arm_p = wp.vec3()
    omega_p = wp.vec3()
    linvel_p = wp.vec3()

    if body_p >= 0:
        tf_world_p = body_tf[body_p] * tf_world_p
        arm_p = wp.transform_get_translation(tf_world_p) - wp.transform_point(body_tf[body_p], body_com[body_p])

        twist_p = body_vel[body_p]
        omega_p = wp.spatial_bottom(twist_p)
        linvel_p = wp.spatial_top(twist_p) + wp.cross(omega_p, arm_p)

    tf_world_c = body_tf[body_c] * tf_cj
    arm_c = wp.transform_get_translation(tf_world_c) - wp.transform_point(body_tf[body_c], body_com[body_c])

    twist_c = body_vel[body_c]
    omega_c = wp.spatial_bottom(twist_c)
    linvel_c = wp.spatial_top(twist_c) + wp.cross(omega_c, arm_c)

    lin_dof = jnt_dof_dim[idx, 0]
    ang_dof = jnt_dof_dim[idx, 1]

    pos_p = wp.transform_get_translation(tf_world_p)
    pos_c = wp.transform_get_translation(tf_world_c)

    rot_p = wp.transform_get_rotation(tf_world_p)
    rot_c = wp.transform_get_rotation(tf_world_c)

    dx = pos_c - pos_p
    dq = wp.quat_inverse(rot_p) * rot_c
    dv = linvel_c - linvel_p
    dw = omega_c - omega_p

    net_f = wp.vec3()
    net_tau = wp.vec3()

    ang_damp_scale = 0.01

    if jtype == ArticulationType.FIXED:
        ang_err = wp.normalize(wp.vec3(dq[0], dq[1], dq[2])) * wp.acos(dq[3]) * 2.0

        net_f += dx * attach_ke + dv * attach_kd
        net_tau += (
            wp.transform_vector(tf_world_p, ang_err) * attach_ke + dw * attach_kd * ang_damp_scale
        )

    if jtype == ArticulationType.PRISMATIC:
        axis = jnt_axis[dof0]
        axis_w = wp.transform_vector(tf_world_p, axis)

        q_lin = wp.dot(dx, axis_w)
        qd_lin = wp.dot(dv, axis_w)

        net_f = axis_w * (
            -jnt_act_f[dof0]
            - dof_spring_response(
                q_lin, qd_lin,
                jnt_target_pos[dof0], jnt_target_vel[dof0],
                jnt_target_ke[dof0], jnt_target_kd[dof0],
                jnt_lim_lo[dof0], jnt_lim_hi[dof0],
                jnt_lim_ke[dof0], jnt_lim_kd[dof0],
            )
        )

        ang_err = wp.normalize(wp.vec3(dq[0], dq[1], dq[2])) * wp.acos(dq[3]) * 2.0

        net_f += (dx - q_lin * axis_w) * attach_ke + (dv - qd_lin * axis_w) * attach_kd
        net_tau += (
            wp.transform_vector(tf_world_p, ang_err) * attach_ke + dw * attach_kd * ang_damp_scale
        )

    if jtype == ArticulationType.REVOLUTE:
        axis = jnt_axis[dof0]

        axis_w_p = wp.transform_vector(tf_world_p, axis)
        axis_w_c = wp.transform_vector(tf_world_c, axis)

        twist = quat_twist_component(axis, dq)

        q_ang = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
        qd_ang = wp.dot(dw, axis_w_p)

        net_tau = axis_w_p * (
            -jnt_act_f[dof0]
            - dof_spring_response(
                q_ang, qd_ang,
                jnt_target_pos[dof0], jnt_target_vel[dof0],
                jnt_target_ke[dof0], jnt_target_kd[dof0],
                jnt_lim_lo[dof0], jnt_lim_hi[dof0],
                jnt_lim_ke[dof0], jnt_lim_kd[dof0],
            )
        )

        swing_err = wp.cross(axis_w_p, axis_w_c)

        net_f += dx * attach_ke + dv * attach_kd
        net_tau += swing_err * attach_ke + (dw - qd_ang * axis_w_p) * attach_kd * ang_damp_scale

    if jtype == ArticulationType.BALL:
        ang_err = wp.normalize(wp.vec3(dq[0], dq[1], dq[2])) * wp.acos(dq[3]) * 2.0

        net_f += dx * attach_ke + dv * attach_kd
        net_tau += wp.vec3(-jnt_act_f[dof0], -jnt_act_f[dof0 + 1], -jnt_act_f[dof0 + 2])

    if jtype == ArticulationType.D6:
        proj_pos = wp.vec3(0.0)
        proj_vel = wp.vec3(0.0)
        if lin_dof >= 1:
            a0 = wp.transform_vector(tf_world_p, jnt_axis[dof0 + 0])
            q0 = wp.dot(dx, a0)
            qd0 = wp.dot(dv, a0)

            net_f += a0 * (
                -jnt_act_f[dof0]
                - dof_spring_response(
                    q0, qd0,
                    jnt_target_pos[dof0 + 0], jnt_target_vel[dof0 + 0],
                    jnt_target_ke[dof0 + 0], jnt_target_kd[dof0 + 0],
                    jnt_lim_lo[dof0 + 0], jnt_lim_hi[dof0 + 0],
                    jnt_lim_ke[dof0 + 0], jnt_lim_kd[dof0 + 0],
                )
            )
            proj_pos += q0 * a0
            proj_vel += qd0 * a0

        if lin_dof >= 2:
            a1 = wp.transform_vector(tf_world_p, jnt_axis[dof0 + 1])
            q1 = wp.dot(dx, a1)
            qd1 = wp.dot(dv, a1)

            net_f += a1 * (
                -jnt_act_f[dof0 + 1]
                - dof_spring_response(
                    q1, qd1,
                    jnt_target_pos[dof0 + 1], jnt_target_vel[dof0 + 1],
                    jnt_target_ke[dof0 + 1], jnt_target_kd[dof0 + 1],
                    jnt_lim_lo[dof0 + 1], jnt_lim_hi[dof0 + 1],
                    jnt_lim_ke[dof0 + 1], jnt_lim_kd[dof0 + 1],
                )
            )
            proj_pos += q1 * a1
            proj_vel += qd1 * a1

        if lin_dof == 3:
            a2 = wp.transform_vector(tf_world_p, jnt_axis[dof0 + 2])
            q2 = wp.dot(dx, a2)
            qd2 = wp.dot(dv, a2)

            net_f += a2 * (
                -jnt_act_f[dof0 + 2]
                - dof_spring_response(
                    q2, qd2,
                    jnt_target_pos[dof0 + 2], jnt_target_vel[dof0 + 2],
                    jnt_target_ke[dof0 + 2], jnt_target_kd[dof0 + 2],
                    jnt_lim_lo[dof0 + 2], jnt_lim_hi[dof0 + 2],
                    jnt_lim_ke[dof0 + 2], jnt_lim_kd[dof0 + 2],
                )
            )
            proj_pos += q2 * a2
            proj_vel += qd2 * a2

        net_f += (dx - proj_pos) * attach_ke + (dv - proj_vel) * attach_kd

        if ang_dof == 0:
            ang_err = wp.normalize(wp.vec3(dq[0], dq[1], dq[2])) * wp.acos(dq[3]) * 2.0
            net_tau += (
                wp.transform_vector(tf_world_p, ang_err) * attach_ke + dw * attach_kd * ang_damp_scale
            )

        i_0 = lin_dof + dof0 + 0
        i_1 = lin_dof + dof0 + 1
        i_2 = lin_dof + dof0 + 2
        dof_ang0 = dof0 + lin_dof

        if ang_dof == 1:
            axis = jnt_axis[i_0]
            axis_w_p = wp.transform_vector(tf_world_p, axis)
            axis_w_c = wp.transform_vector(tf_world_c, axis)

            twist = quat_twist_component(axis, dq)

            q_ang = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
            qd_ang = wp.dot(dw, axis_w_p)

            net_tau = axis_w_p * (
                -jnt_act_f[dof_ang0]
                - dof_spring_response(
                    q_ang, qd_ang,
                    jnt_target_pos[i_0], jnt_target_vel[i_0],
                    jnt_target_ke[i_0], jnt_target_kd[i_0],
                    jnt_lim_lo[i_0], jnt_lim_hi[i_0],
                    jnt_lim_ke[i_0], jnt_lim_kd[i_0],
                )
            )

            swing_err = wp.cross(axis_w_p, axis_w_c)
            net_tau += swing_err * attach_ke + (dw - qd_ang * axis_w_p) * attach_kd * ang_damp_scale

        if ang_dof == 2:
            q_pc = wp.quat_inverse(rot_p) * rot_c
            angles = quat_to_euler_xyz(q_pc)

            orig_a0 = jnt_axis[i_0]
            orig_a1 = jnt_axis[i_1]
            orig_a2 = wp.cross(orig_a0, orig_a1)

            a0 = orig_a0
            q_0 = wp.quat_from_axis_angle(a0, angles[0])

            a1 = wp.quat_rotate(q_0, orig_a1)
            q_1 = wp.quat_from_axis_angle(a1, angles[1])

            a2 = wp.quat_rotate(q_1 * q_0, orig_a2)

            a0 = wp.transform_vector(tf_world_p, a0)
            a1 = wp.transform_vector(tf_world_p, a1)
            a2 = wp.transform_vector(tf_world_p, a2)

            net_tau += a0 * (
                -jnt_act_f[dof_ang0]
                - dof_spring_response(
                    angles[0], wp.dot(a0, dw),
                    jnt_target_pos[i_0], jnt_target_vel[i_0],
                    jnt_target_ke[i_0], jnt_target_kd[i_0],
                    jnt_lim_lo[i_0], jnt_lim_hi[i_0],
                    jnt_lim_ke[i_0], jnt_lim_kd[i_0],
                )
            )
            net_tau += a1 * (
                -jnt_act_f[dof_ang0 + 1]
                - dof_spring_response(
                    angles[1], wp.dot(a1, dw),
                    jnt_target_pos[i_1], jnt_target_vel[i_1],
                    jnt_target_ke[i_1], jnt_target_kd[i_1],
                    jnt_lim_lo[i_1], jnt_lim_hi[i_1],
                    jnt_lim_ke[i_1], jnt_lim_kd[i_1],
                )
            )
            net_tau += a2 * -dof_spring_response(
                angles[2], wp.dot(a2, dw),
                0.0, 0.0,
                attach_ke, attach_kd * ang_damp_scale,
                0.0, 0.0, 0.0, 0.0,
            )

        if ang_dof == 3:
            q_pc = wp.quat_inverse(rot_p) * rot_c
            angles = quat_to_euler_xyz(q_pc)

            orig_a0 = jnt_axis[i_0]
            orig_a1 = jnt_axis[i_1]
            orig_a2 = jnt_axis[i_2]

            a0 = orig_a0
            q_0 = wp.quat_from_axis_angle(a0, angles[0])

            a1 = wp.quat_rotate(q_0, orig_a1)
            q_1 = wp.quat_from_axis_angle(a1, angles[1])

            a2 = wp.quat_rotate(q_1 * q_0, orig_a2)

            a0 = wp.transform_vector(tf_world_p, a0)
            a1 = wp.transform_vector(tf_world_p, a1)
            a2 = wp.transform_vector(tf_world_p, a2)

            net_tau += a0 * (
                -jnt_act_f[dof_ang0]
                - dof_spring_response(
                    angles[0], wp.dot(a0, dw),
                    jnt_target_pos[i_0], jnt_target_vel[i_0],
                    jnt_target_ke[i_0], jnt_target_kd[i_0],
                    jnt_lim_lo[i_0], jnt_lim_hi[i_0],
                    jnt_lim_ke[i_0], jnt_lim_kd[i_0],
                )
            )
            net_tau += a1 * (
                -jnt_act_f[dof_ang0 + 1]
                - dof_spring_response(
                    angles[1], wp.dot(a1, dw),
                    jnt_target_pos[i_1], jnt_target_vel[i_1],
                    jnt_target_ke[i_1], jnt_target_kd[i_1],
                    jnt_lim_lo[i_1], jnt_lim_hi[i_1],
                    jnt_lim_ke[i_1], jnt_lim_kd[i_1],
                )
            )
            net_tau += a2 * (
                -jnt_act_f[dof_ang0 + 2]
                - dof_spring_response(
                    angles[2], wp.dot(a2, dw),
                    jnt_target_pos[i_2], jnt_target_vel[i_2],
                    jnt_target_ke[i_2], jnt_target_kd[i_2],
                    jnt_lim_lo[i_2], jnt_lim_hi[i_2],
                    jnt_lim_ke[i_2], jnt_lim_kd[i_2],
                )
            )

    # write forces to parent and child bodies
    if body_p >= 0:
        wp.atomic_add(body_wrench, body_p, wp.spatial_vector(net_f, net_tau + wp.cross(arm_p, net_f)))

    wp.atomic_sub(body_wrench, body_c, wp.spatial_vector(net_f, net_tau + wp.cross(arm_c, net_f)))


# =====================================================================
# Per-type specialised kernels (reduced branch divergence)
# =====================================================================


@wp.kernel
def _free_constraint_kernel(
    jnt_indices: wp.array(dtype=int),
    jnt_dof_start: wp.array(dtype=int),
    jnt_child: wp.array(dtype=int),
    jnt_act_f: wp.array(dtype=float),
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """FREE / DISTANCE joints – direct wrench application."""
    tid = wp.tid()
    ji = jnt_indices[tid]

    body_c = jnt_child[ji]
    dof0 = jnt_dof_start[ji]

    wrench = wp.spatial_vector(
        jnt_act_f[dof0 + 0], jnt_act_f[dof0 + 1], jnt_act_f[dof0 + 2],
        jnt_act_f[dof0 + 3], jnt_act_f[dof0 + 4], jnt_act_f[dof0 + 5],
    )
    wp.atomic_add(body_wrench, body_c, wrench)


@wp.kernel
def _fixed_constraint_kernel(
    jnt_indices: wp.array(dtype=int),
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    jnt_child: wp.array(dtype=int),
    jnt_parent: wp.array(dtype=int),
    jnt_tf_p: wp.array(dtype=wp.transform),
    jnt_tf_c: wp.array(dtype=wp.transform),
    attach_ke: float,
    attach_kd: float,
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """FIXED joints – rigid attachment constraint."""
    tid = wp.tid()
    ji = jnt_indices[tid]

    body_c = jnt_child[ji]
    body_p = jnt_parent[ji]
    tf_pj = jnt_tf_p[ji]
    tf_cj = jnt_tf_c[ji]

    ang_damp_scale = 0.01

    tf_world_p = tf_pj
    arm_p = wp.vec3()
    omega_p = wp.vec3()
    linvel_p = wp.vec3()

    if body_p >= 0:
        tf_world_p = body_tf[body_p] * tf_world_p
        arm_p = wp.transform_get_translation(tf_world_p) - wp.transform_point(body_tf[body_p], body_com[body_p])
        tw_p = body_vel[body_p]
        omega_p = wp.spatial_bottom(tw_p)
        linvel_p = wp.spatial_top(tw_p) + wp.cross(omega_p, arm_p)

    tf_world_c = body_tf[body_c] * tf_cj
    arm_c = wp.transform_get_translation(tf_world_c) - wp.transform_point(body_tf[body_c], body_com[body_c])
    tw_c = body_vel[body_c]
    omega_c = wp.spatial_bottom(tw_c)
    linvel_c = wp.spatial_top(tw_c) + wp.cross(omega_c, arm_c)

    rot_p = wp.transform_get_rotation(tf_world_p)
    rot_c = wp.transform_get_rotation(tf_world_c)

    dx = wp.transform_get_translation(tf_world_c) - wp.transform_get_translation(tf_world_p)
    dq = wp.quat_inverse(rot_p) * rot_c
    dv = linvel_c - linvel_p
    dw = omega_c - omega_p

    ang_err = wp.normalize(wp.vec3(dq[0], dq[1], dq[2])) * wp.acos(dq[3]) * 2.0

    net_f = dx * attach_ke + dv * attach_kd
    net_tau = wp.transform_vector(tf_world_p, ang_err) * attach_ke + dw * attach_kd * ang_damp_scale

    if body_p >= 0:
        wp.atomic_add(body_wrench, body_p, wp.spatial_vector(net_f, net_tau + wp.cross(arm_p, net_f)))
    wp.atomic_sub(body_wrench, body_c, wp.spatial_vector(net_f, net_tau + wp.cross(arm_c, net_f)))


@wp.kernel
def _prismatic_constraint_kernel(
    jnt_indices: wp.array(dtype=int),
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    jnt_dof_start: wp.array(dtype=int),
    jnt_child: wp.array(dtype=int),
    jnt_parent: wp.array(dtype=int),
    jnt_tf_p: wp.array(dtype=wp.transform),
    jnt_tf_c: wp.array(dtype=wp.transform),
    jnt_axis: wp.array(dtype=wp.vec3),
    jnt_act_f: wp.array(dtype=float),
    jnt_target_pos: wp.array(dtype=float),
    jnt_target_vel: wp.array(dtype=float),
    jnt_target_ke: wp.array(dtype=float),
    jnt_target_kd: wp.array(dtype=float),
    jnt_lim_lo: wp.array(dtype=float),
    jnt_lim_hi: wp.array(dtype=float),
    jnt_lim_ke: wp.array(dtype=float),
    jnt_lim_kd: wp.array(dtype=float),
    attach_ke: float,
    attach_kd: float,
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """PRISMATIC joints – sliding along a single axis."""
    tid = wp.tid()
    ji = jnt_indices[tid]

    body_c = jnt_child[ji]
    body_p = jnt_parent[ji]
    dof0 = jnt_dof_start[ji]
    tf_pj = jnt_tf_p[ji]
    tf_cj = jnt_tf_c[ji]

    ang_damp_scale = 0.01

    tf_world_p = tf_pj
    arm_p = wp.vec3()
    omega_p = wp.vec3()
    linvel_p = wp.vec3()

    if body_p >= 0:
        tf_world_p = body_tf[body_p] * tf_world_p
        arm_p = wp.transform_get_translation(tf_world_p) - wp.transform_point(body_tf[body_p], body_com[body_p])
        tw_p = body_vel[body_p]
        omega_p = wp.spatial_bottom(tw_p)
        linvel_p = wp.spatial_top(tw_p) + wp.cross(omega_p, arm_p)

    tf_world_c = body_tf[body_c] * tf_cj
    arm_c = wp.transform_get_translation(tf_world_c) - wp.transform_point(body_tf[body_c], body_com[body_c])
    tw_c = body_vel[body_c]
    omega_c = wp.spatial_bottom(tw_c)
    linvel_c = wp.spatial_top(tw_c) + wp.cross(omega_c, arm_c)

    dx = wp.transform_get_translation(tf_world_c) - wp.transform_get_translation(tf_world_p)
    dq = wp.quat_inverse(wp.transform_get_rotation(tf_world_p)) * wp.transform_get_rotation(tf_world_c)
    dv = linvel_c - linvel_p
    dw = omega_c - omega_p

    axis = jnt_axis[dof0]
    axis_w = wp.transform_vector(tf_world_p, axis)

    q_lin = wp.dot(dx, axis_w)
    qd_lin = wp.dot(dv, axis_w)

    net_f = axis_w * (
        -jnt_act_f[dof0]
        - dof_spring_response(
            q_lin, qd_lin,
            jnt_target_pos[dof0], jnt_target_vel[dof0],
            jnt_target_ke[dof0], jnt_target_kd[dof0],
            jnt_lim_lo[dof0], jnt_lim_hi[dof0],
            jnt_lim_ke[dof0], jnt_lim_kd[dof0],
        )
    )

    ang_err = wp.normalize(wp.vec3(dq[0], dq[1], dq[2])) * wp.acos(dq[3]) * 2.0
    net_f += (dx - q_lin * axis_w) * attach_ke + (dv - qd_lin * axis_w) * attach_kd
    net_tau = wp.transform_vector(tf_world_p, ang_err) * attach_ke + dw * attach_kd * ang_damp_scale

    if body_p >= 0:
        wp.atomic_add(body_wrench, body_p, wp.spatial_vector(net_f, net_tau + wp.cross(arm_p, net_f)))
    wp.atomic_sub(body_wrench, body_c, wp.spatial_vector(net_f, net_tau + wp.cross(arm_c, net_f)))


@wp.kernel
def _revolute_constraint_kernel(
    jnt_indices: wp.array(dtype=int),
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    jnt_dof_start: wp.array(dtype=int),
    jnt_child: wp.array(dtype=int),
    jnt_parent: wp.array(dtype=int),
    jnt_tf_p: wp.array(dtype=wp.transform),
    jnt_tf_c: wp.array(dtype=wp.transform),
    jnt_axis: wp.array(dtype=wp.vec3),
    jnt_act_f: wp.array(dtype=float),
    jnt_target_pos: wp.array(dtype=float),
    jnt_target_vel: wp.array(dtype=float),
    jnt_target_ke: wp.array(dtype=float),
    jnt_target_kd: wp.array(dtype=float),
    jnt_lim_lo: wp.array(dtype=float),
    jnt_lim_hi: wp.array(dtype=float),
    jnt_lim_ke: wp.array(dtype=float),
    jnt_lim_kd: wp.array(dtype=float),
    attach_ke: float,
    attach_kd: float,
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """REVOLUTE joints – rotation about a single axis."""
    tid = wp.tid()
    ji = jnt_indices[tid]

    body_c = jnt_child[ji]
    body_p = jnt_parent[ji]
    dof0 = jnt_dof_start[ji]
    tf_pj = jnt_tf_p[ji]
    tf_cj = jnt_tf_c[ji]

    ang_damp_scale = 0.01

    tf_world_p = tf_pj
    arm_p = wp.vec3()
    omega_p = wp.vec3()
    linvel_p = wp.vec3()

    if body_p >= 0:
        tf_world_p = body_tf[body_p] * tf_world_p
        arm_p = wp.transform_get_translation(tf_world_p) - wp.transform_point(body_tf[body_p], body_com[body_p])
        tw_p = body_vel[body_p]
        omega_p = wp.spatial_bottom(tw_p)
        linvel_p = wp.spatial_top(tw_p) + wp.cross(omega_p, arm_p)

    tf_world_c = body_tf[body_c] * tf_cj
    arm_c = wp.transform_get_translation(tf_world_c) - wp.transform_point(body_tf[body_c], body_com[body_c])
    tw_c = body_vel[body_c]
    omega_c = wp.spatial_bottom(tw_c)
    linvel_c = wp.spatial_top(tw_c) + wp.cross(omega_c, arm_c)

    dx = wp.transform_get_translation(tf_world_c) - wp.transform_get_translation(tf_world_p)
    dq = wp.quat_inverse(wp.transform_get_rotation(tf_world_p)) * wp.transform_get_rotation(tf_world_c)
    dv = linvel_c - linvel_p
    dw = omega_c - omega_p

    axis = jnt_axis[dof0]
    axis_w_p = wp.transform_vector(tf_world_p, axis)
    axis_w_c = wp.transform_vector(tf_world_c, axis)

    twist = quat_twist_component(axis, dq)
    q_ang = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
    qd_ang = wp.dot(dw, axis_w_p)

    net_tau = axis_w_p * (
        -jnt_act_f[dof0]
        - dof_spring_response(
            q_ang, qd_ang,
            jnt_target_pos[dof0], jnt_target_vel[dof0],
            jnt_target_ke[dof0], jnt_target_kd[dof0],
            jnt_lim_lo[dof0], jnt_lim_hi[dof0],
            jnt_lim_ke[dof0], jnt_lim_kd[dof0],
        )
    )

    swing_err = wp.cross(axis_w_p, axis_w_c)
    net_f = dx * attach_ke + dv * attach_kd
    net_tau += swing_err * attach_ke + (dw - qd_ang * axis_w_p) * attach_kd * ang_damp_scale

    if body_p >= 0:
        wp.atomic_add(body_wrench, body_p, wp.spatial_vector(net_f, net_tau + wp.cross(arm_p, net_f)))
    wp.atomic_sub(body_wrench, body_c, wp.spatial_vector(net_f, net_tau + wp.cross(arm_c, net_f)))


@wp.kernel
def _ball_constraint_kernel(
    jnt_indices: wp.array(dtype=int),
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    jnt_dof_start: wp.array(dtype=int),
    jnt_child: wp.array(dtype=int),
    jnt_parent: wp.array(dtype=int),
    jnt_tf_p: wp.array(dtype=wp.transform),
    jnt_tf_c: wp.array(dtype=wp.transform),
    jnt_act_f: wp.array(dtype=float),
    attach_ke: float,
    attach_kd: float,
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """BALL joints – spherical constraint."""
    tid = wp.tid()
    ji = jnt_indices[tid]

    body_c = jnt_child[ji]
    body_p = jnt_parent[ji]
    dof0 = jnt_dof_start[ji]
    tf_pj = jnt_tf_p[ji]
    tf_cj = jnt_tf_c[ji]

    tf_world_p = tf_pj
    arm_p = wp.vec3()
    omega_p = wp.vec3()
    linvel_p = wp.vec3()

    if body_p >= 0:
        tf_world_p = body_tf[body_p] * tf_world_p
        arm_p = wp.transform_get_translation(tf_world_p) - wp.transform_point(body_tf[body_p], body_com[body_p])
        tw_p = body_vel[body_p]
        omega_p = wp.spatial_bottom(tw_p)
        linvel_p = wp.spatial_top(tw_p) + wp.cross(omega_p, arm_p)

    tf_world_c = body_tf[body_c] * tf_cj
    arm_c = wp.transform_get_translation(tf_world_c) - wp.transform_point(body_tf[body_c], body_com[body_c])
    tw_c = body_vel[body_c]
    omega_c = wp.spatial_bottom(tw_c)
    linvel_c = wp.spatial_top(tw_c) + wp.cross(omega_c, arm_c)

    dx = wp.transform_get_translation(tf_world_c) - wp.transform_get_translation(tf_world_p)
    dv = linvel_c - linvel_p

    net_f = dx * attach_ke + dv * attach_kd
    net_tau = wp.vec3(-jnt_act_f[dof0], -jnt_act_f[dof0 + 1], -jnt_act_f[dof0 + 2])

    if body_p >= 0:
        wp.atomic_add(body_wrench, body_p, wp.spatial_vector(net_f, net_tau + wp.cross(arm_p, net_f)))
    wp.atomic_sub(body_wrench, body_c, wp.spatial_vector(net_f, net_tau + wp.cross(arm_c, net_f)))


# =====================================================================
# Articulation dispatcher (pre-sort + per-type launch)
# =====================================================================


class ArticulationDispatcher:
    """Manages joint pre-sorting and dispatches specialised kernels per type.

    During initialisation, joints are classified by type on the CPU and
    stored as per-type index arrays on the GPU.  At evaluation time, one
    kernel is launched per type that has at least one member, eliminating
    within-warp branch divergence.

    D6 joints are *not* supported on the dispatched path.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or model.device
        self._build_index_tables()

    def _build_index_tables(self):
        if self.model.joint_count == 0:
            self.type_indices = {}
            return

        types_np = self.model.joint_type.numpy()
        enabled_np = self.model.joint_enabled.numpy()

        self.type_indices = {}

        _type_map = {
            ArticulationType.FREE: "free",
            ArticulationType.DISTANCE: "distance",
            ArticulationType.FIXED: "fixed",
            ArticulationType.PRISMATIC: "prismatic",
            ArticulationType.REVOLUTE: "revolute",
            ArticulationType.BALL: "ball",
            ArticulationType.D6: "d6",
        }

        for atype, label in _type_map.items():
            mask = (types_np == atype) & (enabled_np == 1)
            indices = np.where(mask)[0].astype(np.int32)
            if len(indices) > 0:
                self.type_indices[label] = wp.array(indices, dtype=int, device=self.device)

    def get_type_count(self, label: str) -> int:
        arr = self.type_indices.get(label)
        return len(arr) if arr is not None else 0

    def get_statistics(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.type_indices.items()}


# =====================================================================
# Python-side launchers
# =====================================================================


def apply_articulation_forces(
    model, state, control, body_wrench: wp.array,
    attach_ke: float, attach_kd: float,
):
    """Evaluate joint constraint forces using the unified kernel."""
    if model.joint_count:
        wp.launch(
            kernel=articulated_constraint_kernel,
            dim=model.joint_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_qd_start,
                model.joint_type,
                model.joint_enabled,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                control.joint_f,
                control.joint_target_pos,
                control.joint_target_vel,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                attach_ke,
                attach_kd,
            ],
            outputs=[body_wrench],
            device=model.device,
        )


def apply_articulation_forces_dispatched(
    model, state, control, body_wrench: wp.array,
    attach_ke: float, attach_kd: float,
    dispatcher: Optional[ArticulationDispatcher] = None,
):
    """Evaluate joint constraint forces using per-type dispatched kernels.

    D6 joints are not supported; the caller should detect them and fall
    back to :func:`apply_articulation_forces`.
    """
    if model.joint_count == 0:
        return

    if dispatcher is None:
        dispatcher = ArticulationDispatcher(model)

    # FREE / DISTANCE
    for label in ["free", "distance"]:
        if label in dispatcher.type_indices:
            wp.launch(
                kernel=_free_constraint_kernel,
                dim=len(dispatcher.type_indices[label]),
                inputs=[
                    dispatcher.type_indices[label],
                    model.joint_qd_start,
                    model.joint_child,
                    control.joint_f,
                ],
                outputs=[body_wrench],
                device=model.device,
            )

    # FIXED
    if "fixed" in dispatcher.type_indices:
        wp.launch(
            kernel=_fixed_constraint_kernel,
            dim=len(dispatcher.type_indices["fixed"]),
            inputs=[
                dispatcher.type_indices["fixed"],
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                attach_ke,
                attach_kd,
            ],
            outputs=[body_wrench],
            device=model.device,
        )

    # PRISMATIC
    if "prismatic" in dispatcher.type_indices:
        wp.launch(
            kernel=_prismatic_constraint_kernel,
            dim=len(dispatcher.type_indices["prismatic"]),
            inputs=[
                dispatcher.type_indices["prismatic"],
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_qd_start,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                control.joint_f,
                control.joint_target_pos,
                control.joint_target_vel,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                attach_ke,
                attach_kd,
            ],
            outputs=[body_wrench],
            device=model.device,
        )

    # REVOLUTE
    if "revolute" in dispatcher.type_indices:
        wp.launch(
            kernel=_revolute_constraint_kernel,
            dim=len(dispatcher.type_indices["revolute"]),
            inputs=[
                dispatcher.type_indices["revolute"],
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_qd_start,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                control.joint_f,
                control.joint_target_pos,
                control.joint_target_vel,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                attach_ke,
                attach_kd,
            ],
            outputs=[body_wrench],
            device=model.device,
        )

    # BALL
    if "ball" in dispatcher.type_indices:
        wp.launch(
            kernel=_ball_constraint_kernel,
            dim=len(dispatcher.type_indices["ball"]),
            inputs=[
                dispatcher.type_indices["ball"],
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_qd_start,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                control.joint_f,
                attach_ke,
                attach_kd,
            ],
            outputs=[body_wrench],
            device=model.device,
        )
