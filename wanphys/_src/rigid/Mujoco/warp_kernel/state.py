# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""State conversion and FK kernels for the WanPhys MuJoCo adapter."""

from __future__ import annotations

import warp as wp

from newton._src.sim import JointType

from .common import mj_quat_to_warp, warp_quat_to_mj


@wp.kernel
def convert_mj_coords_to_warp_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    joints_per_world: int,
    up_axis: int,
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    worldid, jntid = wp.tid()
    joint_kind = joint_type[jntid]
    q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]
    if joint_kind == JointType.FREE:
        for i in range(3):
            joint_q[wq_i + i] = qpos[worldid, q_i + i]
        rot = mj_quat_to_warp(wp.quat(qpos[worldid, q_i + 3], qpos[worldid, q_i + 4], qpos[worldid, q_i + 5], qpos[worldid, q_i + 6]))
        joint_q[wq_i + 3] = rot[0]
        joint_q[wq_i + 4] = rot[1]
        joint_q[wq_i + 5] = rot[2]
        joint_q[wq_i + 6] = rot[3]
        joint_qd[wqd_i + 0] = qvel[worldid, qd_i + 0]
        joint_qd[wqd_i + 1] = qvel[worldid, qd_i + 1]
        joint_qd[wqd_i + 2] = qvel[worldid, qd_i + 2]
        w = wp.vec3(qvel[worldid, qd_i + 3], qvel[worldid, qd_i + 4], qvel[worldid, qd_i + 5])
        w = wp.quat_rotate(rot, w)
        joint_qd[wqd_i + 3] = w[0]
        joint_qd[wqd_i + 4] = w[1]
        joint_qd[wqd_i + 5] = w[2]
    elif joint_kind == JointType.BALL:
        rot = mj_quat_to_warp(wp.quat(qpos[worldid, q_i], qpos[worldid, q_i + 1], qpos[worldid, q_i + 2], qpos[worldid, q_i + 3]))
        joint_q[wq_i] = rot[0]
        joint_q[wq_i + 1] = rot[1]
        joint_q[wq_i + 2] = rot[2]
        joint_q[wq_i + 3] = rot[3]
        for i in range(3):
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            joint_q[wq_i + i] = qpos[worldid, q_i + i]
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]


@wp.kernel
def convert_warp_coords_to_mj_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joints_per_world: int,
    up_axis: int,
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
):
    worldid, jntid = wp.tid()
    joint_kind = joint_type[jntid]
    q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]
    if joint_kind == JointType.FREE:
        for i in range(3):
            qpos[worldid, q_i + i] = joint_q[wq_i + i]
        rot = wp.quat(joint_q[wq_i + 3], joint_q[wq_i + 4], joint_q[wq_i + 5], joint_q[wq_i + 6])
        rot_mj = warp_quat_to_mj(rot)
        qpos[worldid, q_i + 3] = rot_mj[0]
        qpos[worldid, q_i + 4] = rot_mj[1]
        qpos[worldid, q_i + 5] = rot_mj[2]
        qpos[worldid, q_i + 6] = rot_mj[3]
        qvel[worldid, qd_i + 0] = joint_qd[wqd_i + 0]
        qvel[worldid, qd_i + 1] = joint_qd[wqd_i + 1]
        qvel[worldid, qd_i + 2] = joint_qd[wqd_i + 2]
        w = wp.vec3(joint_qd[wqd_i + 3], joint_qd[wqd_i + 4], joint_qd[wqd_i + 5])
        w = wp.quat_rotate_inv(rot, w)
        qvel[worldid, qd_i + 3] = w[0]
        qvel[worldid, qd_i + 4] = w[1]
        qvel[worldid, qd_i + 5] = w[2]
    elif joint_kind == JointType.BALL:
        rot_mj = warp_quat_to_mj(wp.quat(joint_q[wq_i + 0], joint_q[wq_i + 1], joint_q[wq_i + 2], joint_q[wq_i + 3]))
        qpos[worldid, q_i + 0] = rot_mj[0]
        qpos[worldid, q_i + 1] = rot_mj[1]
        qpos[worldid, q_i + 2] = rot_mj[2]
        qpos[worldid, q_i + 3] = rot_mj[3]
        for i in range(3):
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            qpos[worldid, q_i + i] = joint_q[wq_i + i]
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]


@wp.func
def eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    for i in range(joint_start, joint_end):
        parent = joint_parent[i]
        child = joint_child[i]
        joint_kind = joint_type[i]
        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]
        X_wpj = X_pj
        v_wpj = wp.spatial_vector()
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj
            r_p = wp.transform_get_translation(X_wpj) - wp.transform_point(X_wp, body_com[parent])
            v_wp = body_qd[parent]
            w_p = wp.spatial_bottom(v_wp)
            v_p = wp.spatial_top(v_wp) + wp.cross(w_p, r_p)
            v_wpj = wp.spatial_vector(v_p, w_p)
        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]
        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())
        if joint_kind == JointType.PRISMATIC:
            axis = joint_axis[qd_start]
            X_j = wp.transform(axis * joint_q[q_start], wp.quat_identity())
            v_j = wp.spatial_vector(axis * joint_qd[qd_start], wp.vec3())
        if joint_kind == JointType.REVOLUTE:
            axis = joint_axis[qd_start]
            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, joint_q[q_start]))
            v_j = wp.spatial_vector(wp.vec3(), axis * joint_qd[qd_start])
        if joint_kind == JointType.BALL:
            X_j = wp.transform(wp.vec3(), wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3]))
            v_j = wp.spatial_vector(wp.vec3(), wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]))
        if joint_kind == JointType.FREE or joint_kind == JointType.DISTANCE:
            X_j = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )
            v_j = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )
        if joint_kind == JointType.D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)
            for j in range(lin_axis_count):
                axis = joint_axis[qd_start + j]
                pos += axis * joint_q[q_start + j]
                vel_v += axis * joint_qd[qd_start + j]
            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            for j in range(ang_axis_count):
                axis = joint_axis[iqd + j]
                rot = rot * wp.quat_from_axis_angle(axis, joint_q[iq + j])
                vel_w += joint_qd[iqd + j] * axis
            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_v, vel_w)
        X_wcj = X_wpj * X_j
        X_wc = X_wcj * wp.transform_inverse(X_cj)
        linear_vel = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_vel = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))
        v_wc = v_wpj + wp.spatial_vector(linear_vel, angular_vel)
        body_q[child] = X_wc
        body_qd[child] = v_wc


@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    eval_single_articulation_fk(
        articulation_start[tid],
        articulation_start[tid + 1],
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        body_q,
        body_qd,
    )


@wp.kernel
def convert_body_xforms_to_warp_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    body_q: wp.array(dtype=wp.transform),
):
    world, mjc_body = wp.tid()
    source_body = mjc_body_to_newton[world, mjc_body]
    if source_body >= 0:
        body_q[source_body] = wp.transform(xpos[world, mjc_body], mj_quat_to_warp(xquat[world, mjc_body]))


__all__ = [
    "convert_mj_coords_to_warp_kernel",
    "convert_warp_coords_to_mj_kernel",
    "eval_articulation_fk",
    "convert_body_xforms_to_warp_kernel",
]
