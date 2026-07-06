# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Control and force-injection kernels for the WanPhys MuJoCo adapter."""

from __future__ import annotations

import warp as wp

from newton._src.sim import JointType

from .common import decode_actuator_axis


@wp.kernel
def apply_mjc_control_kernel(
    mjc_actuator_to_newton_axis: wp.array2d(dtype=wp.int32),
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    mj_ctrl: wp.array2d(dtype=wp.float32),
):
    world, mjc_actuator = wp.tid()
    source_axis, is_velocity, is_mapped = decode_actuator_axis(mjc_actuator_to_newton_axis[world, mjc_actuator])
    if not is_mapped:
        return
    if not is_velocity:
        mj_ctrl[world, mjc_actuator] = joint_target_pos[source_axis]
    else:
        mj_ctrl[world, mjc_actuator] = joint_target_vel[source_axis]


@wp.kernel
def apply_mjc_body_f_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    body_f: wp.array(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    world, mjc_body = wp.tid()
    source_body = mjc_body_to_newton[world, mjc_body]
    if source_body >= 0:
        f = body_f[source_body]
        xfrc_applied[world, mjc_body] = wp.spatial_vector(wp.vec3(f[0], f[1], f[2]), wp.vec3(f[3], f[4], f[5]))


@wp.kernel
def apply_mjc_qfrc_kernel(
    body_q: wp.array(dtype=wp.transform),
    joint_f: wp.array(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    joint_child: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array2d(dtype=wp.int32),
    joints_per_world: int,
    bodies_per_world: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    worldid, jntid = wp.tid()
    child = joint_child[jntid]
    qd_i = joint_qd_start[jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]
    jtype = joint_type[jntid]
    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        rot = wp.transform_get_rotation(body_q[worldid * bodies_per_world + child])
        v = wp.vec3(joint_f[wqd_i + 0], joint_f[wqd_i + 1], joint_f[wqd_i + 2])
        w = wp.vec3(joint_f[wqd_i + 3], joint_f[wqd_i + 4], joint_f[wqd_i + 5])
        w = wp.quat_rotate_inv(rot, w)
        qfrc_applied[worldid, qd_i + 0] = v[0]
        qfrc_applied[worldid, qd_i + 1] = v[1]
        qfrc_applied[worldid, qd_i + 2] = v[2]
        qfrc_applied[worldid, qd_i + 3] = w[0]
        qfrc_applied[worldid, qd_i + 4] = w[1]
        qfrc_applied[worldid, qd_i + 5] = w[2]
    elif jtype == JointType.BALL:
        qfrc_applied[worldid, qd_i + 0] = joint_f[wqd_i + 0]
        qfrc_applied[worldid, qd_i + 1] = joint_f[wqd_i + 1]
        qfrc_applied[worldid, qd_i + 2] = joint_f[wqd_i + 2]
    else:
        for i in range(joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]):
            qfrc_applied[worldid, qd_i + i] = joint_f[wqd_i + i]


__all__ = [
    "apply_mjc_control_kernel",
    "apply_mjc_body_f_kernel",
    "apply_mjc_qfrc_kernel",
]
