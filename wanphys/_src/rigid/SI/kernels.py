# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
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

"""Sequential Impulse GPU kernels for WanPhys rigid body simulation.

Covers velocity integration, position update, joint constraint solving,
contact impulse solving, and restitution.
"""

import warp as wp

from ._types import JointType


# ==================== velocity integration ====================

@wp.func
def integrate_rigid_body_velocity(
    q: wp.transform,            # body-local -> world
    qd: wp.spatial_vector,      # (v, w), world
    f: wp.spatial_vector,       # (force, torque), world
    com: wp.vec3,               # body-local
    inertia: wp.mat33,          # I, body-local
    inv_mass: float,            # m^{-1}
    inv_inertia: wp.mat33,      # I^{-1}, body-local
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
):
    # R: body-local -> world
    r0 = wp.transform_get_rotation(q)  # wp.quat

    w0 = wp.spatial_bottom(qd)  # wp.vec3, w (angular), world
    v0 = wp.spatial_top(qd)     # wp.vec3, v (linear), world

    t0 = wp.spatial_bottom(f)   # wp.vec3, torque, world
    f0 = wp.spatial_top(f)      # wp.vec3, force, world

    # v1 = v0 + (f/m + g) * dt  (wp.vec3, world)
    v1 = v0 + (f0 * inv_mass + gravity[0] * wp.nonzero(inv_mass)) * dt

    # angular in body-local:
    # w_b = R^T * w0  (wp.vec3, body-local)
    wb = wp.quat_rotate_inv(r0, w0)
    # tau_b = R^T * t0 - w_b x (I * w_b)   (Euler equation, coriolis term)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)

    # w1_b = w_b + I^{-1} * tau_b * dt
    # w1 = R * w1_b  (wp.vec3, world)
    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)

    # angular damping: w1 *= (1 - damping * dt)
    w1 *= 1.0 - angular_damping * dt

    qd_new = wp.spatial_vector(v1, w1)

    return qd_new


@wp.kernel
def si_integrate_body_velocities(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
    # outputs
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    inv_mass = body_inv_mass[tid]
    inertia = body_inertia[tid]
    inv_inertia = body_inv_inertia[tid]
    com = body_com[tid]

    qd_new = integrate_rigid_body_velocity(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        gravity,
        angular_damping,
        dt,
    )

    body_qd_new[tid] = qd_new


# ==================== position update ====================

@wp.kernel
def si_update_body_positions(
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    dt: float,
    # outputs
    body_q_out: wp.array(dtype=wp.transform),
):
    tid = wp.tid()

    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        body_q_out[tid] = body_q_prev[tid]
        return

    tf_prev = body_q_prev[tid]
    qd = body_qd[tid]

    p0 = wp.transform_get_translation(tf_prev)  # wp.vec3, world
    q0 = wp.transform_get_rotation(tf_prev)     # wp.quat

    v = wp.spatial_top(qd)     # wp.vec3, linear vel, world
    w = wp.spatial_bottom(qd)  # wp.vec3, angular vel, world

    com = body_com[tid]  # wp.vec3, body-local

    # x_com = p0 + R * com  (wp.vec3, world)
    x_com = p0 + wp.quat_rotate(q0, com)

    # x_com_new = x_com + v * dt
    x_com_new = x_com + v * dt

    # q1 = q0 + 0.5 * quat(w*dt, 0) * q0,  then normalize
    q1 = q0 + 0.5 * wp.quat(w * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # p1 = x_com_new - R1 * com
    p1 = x_com_new - wp.quat_rotate(q1, com)

    body_q_out[tid] = wp.transform(p1, q1)


@wp.kernel
def solve_body_joints_si(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),

    body_com: wp.array(dtype=wp.vec3),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),

    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=wp.bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),

    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),

    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),

    beta: float,                      # Baumgarte coefficient for linear constraints
    angular_beta: float,              # Baumgarte coefficient for angular constraints
    joint_impulse_relaxation: float,  # Scaling factor for computed joint impulses
    dt: float,
    # outputs -- Jacobi buffer
    body_joint_deltas: wp.array(dtype=wp.spatial_vector),
    body_joint_count: wp.array(dtype=float),
):
    tid = wp.tid()

    if joint_enabled[tid] == 0:
        return

    jtype = joint_type[tid]
    if jtype == JointType.FREE:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]

    # --- parent joint anchor -> world ---
    X_wp = joint_X_p[tid]       # wp.transform, joint-local -> parent body-local
    pose_p = X_wp
    inv_m_p = 0.0               # float, ground: m^{-1}=0
    inv_I_p = wp.mat33()        # wp.mat33, ground: I^{-1}=0
    com_p = wp.vec3()           # wp.vec3, body-local

    if id_p >= 0:
        pose_p = body_q[id_p]   # wp.transform, body-local -> world
        X_wp = pose_p * X_wp    # wp.transform, joint-local -> world
        inv_m_p = body_inv_m[id_p]  # float
        inv_I_p = body_inv_I[id_p]  # wp.mat33, body-local
        com_p = body_com[id_p]      # wp.vec3, body-local

    # --- child joint anchor -> world ---
    pose_c = body_q[id_c]                  # wp.transform, body-local -> world
    X_wc = pose_c * joint_X_c[tid]         # wp.transform, joint-local -> world
    inv_m_c = body_inv_m[id_c]             # float
    inv_I_c = body_inv_I[id_c]             # wp.mat33, body-local
    com_c = body_com[id_c]                 # wp.vec3, body-local

    if inv_m_p == 0.0 and inv_m_c == 0.0:
        return

    # count active joints per body (for Jacobi averaging)
    if id_p >= 0:
        wp.atomic_add(body_joint_count, id_p, 1.0)
    wp.atomic_add(body_joint_count, id_c, 1.0)

    # velocities (world)
    # spatial_vector: top=v(linear), bottom=w(angular)
    v_p = wp.vec3()
    w_p = wp.vec3()
    if id_p >= 0:
        v_p = wp.spatial_top(body_qd[id_p])     # wp.vec3, world
        w_p = wp.spatial_bottom(body_qd[id_p])   # wp.vec3, world

    v_c = wp.spatial_top(body_qd[id_c])          # wp.vec3, world
    w_c = wp.spatial_bottom(body_qd[id_c])        # wp.vec3, world

    # body rotations, for I^{-1} frame conversion: body-local <-> world
    q_body_p = wp.transform_get_rotation(pose_p)  # wp.quat
    q_body_c = wp.transform_get_rotation(pose_c)  # wp.quat

    # world COM: c_w = R * c_local + t
    world_com_p = wp.transform_point(pose_p, com_p)  # wp.vec3, world
    world_com_c = wp.transform_point(pose_c, com_c)  # wp.vec3, world

    # joint anchor positions (world)
    x_p = wp.transform_get_translation(X_wp)  # wp.vec3, world
    x_c = wp.transform_get_translation(X_wc)  # wp.vec3, world

    # lever arm: r = x_joint - x_com  (wp.vec3, world)
    r_p = x_p - world_com_p
    r_c = x_c - world_com_c

    # joint frame rotation matrix, columns = joint axes (world)
    frame_p = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))  # wp.mat33

    # LINEAR CONSTRAINTS (FIXED, BALL, REVOLUTE)
    # C_i = n_i . (x_c - x_p) = 0,  i=0,1,2
    if jtype == JointType.FIXED or jtype == JointType.BALL or jtype == JointType.REVOLUTE:
        for i in range(3):
            # n = frame_p[:,i]  (wp.vec3, world, |n|=1)
            n = wp.vec3(frame_p[0, i], frame_p[1, i], frame_p[2, i])

            # C = n . (x_c - x_p)  (float)
            C = wp.dot(x_c - x_p, n)

            # dC/dt = n.(v_c-v_p) + (r_c x n).w_c - (r_p x n).w_p  (float)
            # identity: n.(w x r) = (r x n).w
            jv = (
                wp.dot(n, v_c - v_p)
                + wp.dot(wp.cross(r_c, n), w_c)
                - wp.dot(wp.cross(r_p, n), w_p)
            )

            # bias = clamp(beta * C / dt)  (float)
            max_bias = float(5.0)
            bias = wp.clamp(beta * C / dt, -max_bias, max_bias)

            # J_w = r x n  (wp.vec3, world)
            ang_p = wp.cross(r_p, n)
            ang_c = wp.cross(r_c, n)
            # world -> body-local: a_local = R^T * a_world  (wp.vec3, body-local)
            ang_p_local = wp.quat_rotate_inv(q_body_p, ang_p)
            ang_c_local = wp.quat_rotate_inv(q_body_c, ang_c)

            # K = m^{-1}_p + m^{-1}_c + a_p^T * I^{-1}_p * a_p + a_c^T * I^{-1}_c * a_c  (float)
            K = inv_m_p + inv_m_c
            K += wp.dot(ang_p_local, inv_I_p * ang_p_local)
            K += wp.dot(ang_c_local, inv_I_c * ang_c_local)

            if K > 0.0:
                relaxation = joint_impulse_relaxation
                # lambda = -(dC/dt + bias) / K * relaxation  (float)
                dl = -(jv + bias) / K * relaxation

                if id_p >= 0:
                    # dw_p = R_p * (I^{-1}_p * a_p_local)  (wp.vec3, body-local -> world)
                    dw_p = wp.quat_rotate(q_body_p, inv_I_p * ang_p_local)
                    # dv_p = -m^{-1}_p * lambda * n,  dw_p = -lambda * dw_p
                    wp.atomic_add(
                        body_joint_deltas, id_p,
                        wp.spatial_vector(
                            -n * (inv_m_p * dl),
                            -dw_p * dl,
                        ),
                    )

                dw_c = wp.quat_rotate(q_body_c, inv_I_c * ang_c_local)
                # dv_c = +m^{-1}_c * lambda * n,  dw_c = +lambda * dw_c
                wp.atomic_add(
                    body_joint_deltas, id_c,
                    wp.spatial_vector(
                        n * (inv_m_c * dl),
                        dw_c * dl,
                    ),
                )

    # ANGULAR CONSTRAINTS (FIXED, REVOLUTE)
    # Swing-Twist decomposition for large angles
    if jtype == JointType.FIXED or jtype == JointType.REVOLUTE:

        q_p = wp.transform_get_rotation(X_wp)  # wp.quat, joint-parent -> world
        q_c = wp.transform_get_rotation(X_wc)  # wp.quat, joint-child  -> world

        # q_rel = q_p^{-1} * q_c  (wp.quat, joint-local)
        q_rel = wp.quat_inverse(q_p) * q_c

        # same hemisphere (q and -q = same rotation)
        if q_rel[3] < 0.0:
            q_rel = wp.quat(-q_rel[0], -q_rel[1], -q_rel[2], -q_rel[3])

        # joint axes (wp.vec3, world)
        axis_x = wp.vec3(frame_p[0, 0], frame_p[1, 0], frame_p[2, 0])
        axis_y = wp.vec3(frame_p[0, 1], frame_p[1, 1], frame_p[2, 1])
        axis_z = wp.vec3(frame_p[0, 2], frame_p[1, 2], frame_p[2, 2])

        # hinge axis (wp.vec3, joint-local)
        axis_start = joint_qd_start[tid]
        hinge_local = joint_axis[axis_start]

        # argmax_i |hinge_local . e_i| -> hinge / t1 / t2
        hinge_dot_x = wp.abs(wp.dot(hinge_local, wp.vec3(1.0, 0.0, 0.0)))
        hinge_dot_y = wp.abs(wp.dot(hinge_local, wp.vec3(0.0, 1.0, 0.0)))
        hinge_dot_z = wp.abs(wp.dot(hinge_local, wp.vec3(0.0, 0.0, 1.0)))

        if hinge_dot_x >= hinge_dot_y and hinge_dot_x >= hinge_dot_z:
            hinge = axis_x;  t1 = axis_y;  t2 = axis_z
            hinge_local_std = wp.vec3(1.0, 0.0, 0.0)
        elif hinge_dot_y >= hinge_dot_z:
            hinge = axis_y;  t1 = axis_x;  t2 = axis_z
            hinge_local_std = wp.vec3(0.0, 1.0, 0.0)
        else:
            hinge = axis_z;  t1 = axis_x;  t2 = axis_y
            hinge_local_std = wp.vec3(0.0, 0.0, 1.0)

        # --- Swing-Twist: q_rel = swing * twist ---
        # twist = rotation around hinge (REVOLUTE free DOF)
        # swing = rotation perpendicular to hinge (locked)

        q_rel_xyz = wp.vec3(q_rel[0], q_rel[1], q_rel[2])  # wp.vec3

        # twist_xyz = (v . h) * h  (projection onto hinge)
        d = wp.dot(q_rel_xyz, hinge_local_std)   # float
        twist_xyz = hinge_local_std * d           # wp.vec3
        twist_w = q_rel[3]                        # float

        # twist = normalize(twist_xyz, twist_w)  (wp.quat)
        twist_len = wp.sqrt(d * d + twist_w * twist_w)  # float
        if twist_len > 1e-8:
            twist = wp.quat(twist_xyz[0] / twist_len, twist_xyz[1] / twist_len,
                          twist_xyz[2] / twist_len, twist_w / twist_len)
        else:
            twist = wp.quat(0.0, 0.0, 0.0, 1.0)

        # swing = q_rel * twist^{-1}  (wp.quat)
        swing = q_rel * wp.quat_inverse(twist)

        # angle error ~= 2*(qx,qy,qz), small angle ~= theta * axis_hat
        # joint-local -> world
        swing_error_local = wp.vec3(swing[0], swing[1], swing[2]) * 2.0  # wp.vec3, joint-local
        swing_error_world = wp.quat_rotate(q_p, swing_error_local)       # wp.vec3, world

        twist_error_local = wp.vec3(twist[0], twist[1], twist[2]) * 2.0  # wp.vec3, joint-local
        twist_error_world = wp.quat_rotate(q_p, twist_error_local)       # wp.vec3, world

        # FIXED: lock 3 axes,  REVOLUTE: lock 2 axes (t1, t2)
        axis_count = 3 if jtype == JointType.FIXED else 2

        for i in range(axis_count):
            if jtype == JointType.FIXED:
                if i == 0:
                    n = hinge
                    angle_err = wp.dot(twist_error_world, n)   # hinge dir -> twist error
                elif i == 1:
                    n = t1
                    angle_err = wp.dot(swing_error_world, n)
                else:
                    n = t2
                    angle_err = wp.dot(swing_error_world, n)
            else:  # REVOLUTE: only lock swing
                if i == 0:
                    n = t1
                else:
                    n = t2
                angle_err = wp.dot(swing_error_world, n)

            # dC/dt = n . (w_c - w_p)  (float)
            jv = wp.dot(n, w_c - w_p)

            # Angular Baumgarte stabilisation.
            max_bias = float(5.0)
            bias = wp.clamp(angular_beta * angle_err / dt, -max_bias, max_bias)

            # n_local = R^T * n  (wp.vec3, body-local)
            n_local_p = wp.quat_rotate_inv(q_body_p, n)
            n_local_c = wp.quat_rotate_inv(q_body_c, n)

            # K = n_p^T * I^{-1}_p * n_p + n_c^T * I^{-1}_c * n_c  (float, pure angular)
            K = wp.dot(n_local_p, inv_I_p * n_local_p) + wp.dot(n_local_c, inv_I_c * n_local_c)
            if K > 0.0:
                relaxation = joint_impulse_relaxation
                # lambda = -(dC/dt + bias) / K * relaxation  (float)
                dl = -(jv + bias) / K * relaxation

                if id_p >= 0:
                    # R_p * (I^{-1}_p * n_local_p)  (wp.vec3, body-local -> world)
                    dw_p = wp.quat_rotate(q_body_p, inv_I_p * n_local_p)
                    # dw_p = -lambda * dw_p
                    wp.atomic_add(
                        body_joint_deltas, id_p,
                        wp.spatial_vector(wp.vec3(), -dw_p * dl),
                    )

                dw_c = wp.quat_rotate(q_body_c, inv_I_c * n_local_c)
                # dw_c = +lambda * dw_c
                wp.atomic_add(
                    body_joint_deltas, id_c,
                    wp.spatial_vector(wp.vec3(), dw_c * dl),
                )

    # PRISMATIC JOINT
    # 1 DOF along slide axis, lock 2 linear + 3 angular
    if jtype == JointType.PRISMATIC:

        # joint axes (wp.vec3, world)
        pri_axis_x = wp.vec3(frame_p[0, 0], frame_p[1, 0], frame_p[2, 0])
        pri_axis_y = wp.vec3(frame_p[0, 1], frame_p[1, 1], frame_p[2, 1])
        pri_axis_z = wp.vec3(frame_p[0, 2], frame_p[1, 2], frame_p[2, 2])

        # slide axis (wp.vec3, joint-local)
        axis_start = joint_qd_start[tid]
        slide_axis_local = joint_axis[axis_start]

        # argmax_i |slide . e_i| -> slide_axis(free) / t1,t2(locked)
        slide_dot_x = wp.abs(wp.dot(slide_axis_local, wp.vec3(1.0, 0.0, 0.0)))
        slide_dot_y = wp.abs(wp.dot(slide_axis_local, wp.vec3(0.0, 1.0, 0.0)))
        slide_dot_z = wp.abs(wp.dot(slide_axis_local, wp.vec3(0.0, 0.0, 1.0)))

        if slide_dot_x >= slide_dot_y and slide_dot_x >= slide_dot_z:
            slide_axis = pri_axis_x;  t1 = pri_axis_y;  t2 = pri_axis_z
        elif slide_dot_y >= slide_dot_z:
            slide_axis = pri_axis_y;  t1 = pri_axis_x;  t2 = pri_axis_z
        else:
            slide_axis = pri_axis_z;  t1 = pri_axis_x;  t2 = pri_axis_y

        # --- linear: lock t1 ---
        n = t1
        C = wp.dot(x_c - x_p, n)                       # float
        jv = (
            wp.dot(n, v_c - v_p)
            + wp.dot(wp.cross(r_c, n), w_c)
            - wp.dot(wp.cross(r_p, n), w_p)
        )

        max_bias = float(5.0)
        bias = wp.clamp(beta * C / dt, -max_bias, max_bias)

        ang_p = wp.cross(r_p, n)                            # wp.vec3, world
        ang_c = wp.cross(r_c, n)
        ang_p_local = wp.quat_rotate_inv(q_body_p, ang_p)   # wp.vec3, body-local
        ang_c_local = wp.quat_rotate_inv(q_body_c, ang_c)

        K = inv_m_p + inv_m_c
        K += wp.dot(ang_p_local, inv_I_p * ang_p_local)
        K += wp.dot(ang_c_local, inv_I_c * ang_c_local)

        if K > 0.0:
            relaxation = joint_impulse_relaxation
            dl = -(jv + bias) / K * relaxation

            if id_p >= 0:
                dw_p = wp.quat_rotate(q_body_p, inv_I_p * ang_p_local)  # body-local -> world
                wp.atomic_add(
                    body_joint_deltas, id_p,
                    wp.spatial_vector(
                        -n * (inv_m_p * dl),
                        -dw_p * dl,
                    ),
                )

            dw_c = wp.quat_rotate(q_body_c, inv_I_c * ang_c_local)
            wp.atomic_add(
                body_joint_deltas, id_c,
                wp.spatial_vector(
                    n * (inv_m_c * dl),
                    dw_c * dl,
                ),
            )

        # --- linear: lock t2 ---
        n = t2
        C = wp.dot(x_c - x_p, n)
        jv = (
            wp.dot(n, v_c - v_p)
            + wp.dot(wp.cross(r_c, n), w_c)
            - wp.dot(wp.cross(r_p, n), w_p)
        )

        max_bias = float(5.0)
        bias = wp.clamp(beta * C / dt, -max_bias, max_bias)

        ang_p = wp.cross(r_p, n)
        ang_c = wp.cross(r_c, n)
        ang_p_local = wp.quat_rotate_inv(q_body_p, ang_p)
        ang_c_local = wp.quat_rotate_inv(q_body_c, ang_c)

        K = inv_m_p + inv_m_c
        K += wp.dot(ang_p_local, inv_I_p * ang_p_local)
        K += wp.dot(ang_c_local, inv_I_c * ang_c_local)

        if K > 0.0:
            relaxation = joint_impulse_relaxation
            dl = -(jv + bias) / K * relaxation

            if id_p >= 0:
                dw_p = wp.quat_rotate(q_body_p, inv_I_p * ang_p_local)
                wp.atomic_add(
                    body_joint_deltas, id_p,
                    wp.spatial_vector(
                        -n * (inv_m_p * dl),
                        -dw_p * dl,
                    ),
                )

            dw_c = wp.quat_rotate(q_body_c, inv_I_c * ang_c_local)
            wp.atomic_add(
                body_joint_deltas, id_c,
                wp.spatial_vector(
                    n * (inv_m_c * dl),
                    dw_c * dl,
                ),
            )

        # --- angular: lock all 3 axes ---
        # axis-angle: theta = 2*arcsin(|v|), axis_hat = v/|v|, e = theta * axis_hat

        q_p = wp.transform_get_rotation(X_wp)   # wp.quat
        q_c = wp.transform_get_rotation(X_wc)
        # q_rel = q_p^{-1} * q_c  (wp.quat, joint-local)
        q_rel = wp.quat_inverse(q_p) * q_c
        if q_rel[3] < 0.0:
            q_rel = wp.quat(-q_rel[0], -q_rel[1], -q_rel[2], -q_rel[3])

        q_rel_xyz = wp.vec3(q_rel[0], q_rel[1], q_rel[2])  # wp.vec3
        sin_half_angle = wp.length(q_rel_xyz)                # float, = sin(theta/2)

        if sin_half_angle > 1e-8:
            # theta = 2 * arcsin(|v|),  e_local = theta * v/|v|  (wp.vec3, joint-local)
            half_angle = wp.asin(wp.clamp(sin_half_angle, -1.0, 1.0))
            angle = half_angle * 2.0
            axis = q_rel_xyz / sin_half_angle
            angle_error_local = axis * angle
        else:
            angle_error_local = wp.vec3(0.0, 0.0, 0.0)

        # joint-local -> world
        angle_error_world = wp.quat_rotate(q_p, angle_error_local)  # wp.vec3, world

        for i in range(3):
            n = wp.vec3(frame_p[0, i], frame_p[1, i], frame_p[2, i])  # wp.vec3, world

            jv = wp.dot(n, w_c - w_p)                # float
            angle_err = wp.dot(angle_error_world, n)  # float

            # Angular Baumgarte stabilisation.
            max_bias = float(5.0)
            bias = wp.clamp(angular_beta * angle_err / dt, -max_bias, max_bias)

            n_local_p = wp.quat_rotate_inv(q_body_p, n)  # wp.vec3, body-local
            n_local_c = wp.quat_rotate_inv(q_body_c, n)

            # K = n_p^T * I^{-1}_p * n_p + n_c^T * I^{-1}_c * n_c  (float)
            K = wp.dot(n_local_p, inv_I_p * n_local_p) + wp.dot(n_local_c, inv_I_c * n_local_c)

            if K > 0.0:
                relaxation = joint_impulse_relaxation
                dl = -(jv + bias) / K * relaxation

                if id_p >= 0:
                    dw_p = wp.quat_rotate(q_body_p, inv_I_p * n_local_p)  # body-local -> world
                    wp.atomic_add(
                        body_joint_deltas, id_p,
                        wp.spatial_vector(wp.vec3(), -dw_p * dl),
                    )

                dw_c = wp.quat_rotate(q_body_c, inv_I_c * n_local_c)
                wp.atomic_add(
                    body_joint_deltas, id_c,
                    wp.spatial_vector(wp.vec3(), dw_c * dl),
                )


@wp.kernel
def si_apply_body_joint_deltas(
    body_joint_deltas: wp.array(dtype=wp.spatial_vector),
    body_joint_count: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    # outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Jacobi averaging for joint deltas: delta_applied = delta_accumulated / N_joints"""
    tid = wp.tid()

    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        return

    count = body_joint_count[tid]  # float
    if count < 1.0:
        return

    delta = body_joint_deltas[tid]  # wp.spatial_vector
    weight = 1.0 / count            # float

    qd = body_qd[tid]
    # v_new = v + dv * weight,  w_new = w + dw * weight
    v = wp.spatial_top(qd) + wp.spatial_top(delta) * weight
    w = wp.spatial_bottom(qd) + wp.spatial_bottom(delta) * weight

    body_qd[tid] = wp.spatial_vector(v, w)


@wp.kernel
def solve_body_contact_velocities_si(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),

    body_com: wp.array(dtype=wp.vec3),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),

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
    shape_material_torsional_friction: wp.array(dtype=float),
    shape_material_rolling_friction: wp.array(dtype=float),

    beta: float,
    dt: float,
    # outputs -- Jacobi buffer
    body_deltas: wp.array(dtype=wp.spatial_vector),
    body_contact_count: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return

    # --- transforms ---
    X_wa = wp.transform_identity()   # wp.transform, body-local -> world
    X_wb = wp.transform_identity()
    if body_a >= 0:
        X_wa = body_q[body_a]
    if body_b >= 0:
        X_wb = body_q[body_b]

    # body rotations for I^{-1} frame conversion
    q_a = wp.transform_get_rotation(X_wa)  # wp.quat
    q_b = wp.transform_get_rotation(X_wb)  # wp.quat

    # world contact points: p = R * p_local + t  (wp.vec3, world)
    pa = wp.transform_point(X_wa, contact_point0[tid])
    pb = wp.transform_point(X_wb, contact_point1[tid])

    # contact normal (wp.vec3, world), penetration depth (float)
    n = -contact_normal[tid]
    thickness = contact_thickness0[tid] + contact_thickness1[tid]
    # d = n . (pb - pa) - thickness  (float, d<0 means penetration)
    d = wp.dot(n, pb - pa) - thickness

    if d >= 0.0:
        return

    # count active contacts per body (for Jacobi averaging)
    if body_a >= 0:
        wp.atomic_add(body_contact_count, body_a, 1.0)
    if body_b >= 0:
        wp.atomic_add(body_contact_count, body_b, 1.0)

    # --- body properties ---
    inv_m_a = 0.0        # float
    inv_m_b = 0.0
    inv_I_a = wp.mat33() # wp.mat33, body-local
    inv_I_b = wp.mat33()
    com_a = wp.vec3()    # wp.vec3, body-local
    com_b = wp.vec3()

    v_a = wp.vec3()      # wp.vec3, linear vel, world
    w_a = wp.vec3()      # wp.vec3, angular vel, world
    v_b = wp.vec3()
    w_b = wp.vec3()

    if body_a >= 0:
        inv_m_a = body_inv_m[body_a]
        inv_I_a = body_inv_I[body_a]
        com_a = body_com[body_a]
        v_a = wp.spatial_top(body_qd[body_a])
        w_a = wp.spatial_bottom(body_qd[body_a])

    if body_b >= 0:
        inv_m_b = body_inv_m[body_b]
        inv_I_b = body_inv_I[body_b]
        com_b = body_com[body_b]
        v_b = wp.spatial_top(body_qd[body_b])
        w_b = wp.spatial_bottom(body_qd[body_b])

    # lever arms: r = p_contact - p_com  (wp.vec3, world)
    ra = pa - wp.transform_point(X_wa, com_a)
    rb = pb - wp.transform_point(X_wb, com_b)

    # 1) NORMAL IMPULSE

    # contact point velocity: v_contact = v + w x r  (wp.vec3, world)
    va_c = v_a + wp.cross(w_a, ra)
    vb_c = v_b + wp.cross(w_b, rb)
    # relative velocity: v_rel = vb - va  (wp.vec3, world)
    rel_v = vb_c - va_c

    # v_n = v_rel . n  (float)
    vn = wp.dot(rel_v, n)
    # bias = beta * d / dt  (float, Baumgarte)
    bias = beta * d / dt

    # effective mass K:
    ang_a = wp.cross(ra, n)                          # wp.vec3, world
    ang_b = wp.cross(rb, n)
    ang_a_local = wp.quat_rotate_inv(q_a, ang_a)     # wp.vec3, body-local
    ang_b_local = wp.quat_rotate_inv(q_b, ang_b)

    # K = m^{-1}_a + m^{-1}_b + a_a^T * I^{-1}_a * a_a + a_b^T * I^{-1}_b * a_b  (float)
    K = inv_m_a + inv_m_b
    K += wp.dot(ang_a_local, inv_I_a * ang_a_local)
    K += wp.dot(ang_b_local, inv_I_b * ang_b_local)

    lambda_n = 0.0  # float
    if K > 0.0:
        # lambda_n = -(v_n + bias) / K,  clamped >= 0 (non-penetration)
        lambda_n = -(vn + bias) / K
        lambda_n = wp.max(lambda_n, 0.0)

        if body_a >= 0:
            dw_a = wp.quat_rotate(q_a, inv_I_a * ang_a_local)
            wp.atomic_add(
                body_deltas, body_a,
                wp.spatial_vector(
                    -n * (inv_m_a * lambda_n),
                    -dw_a * lambda_n,
                ),
            )

        if body_b >= 0:
            dw_b = wp.quat_rotate(q_b, inv_I_b * ang_b_local)
            wp.atomic_add(
                body_deltas, body_b,
                wp.spatial_vector(
                    n * (inv_m_b * lambda_n),
                    dw_b * lambda_n,
                ),
            )

    # 2) TANGENTIAL FRICTION (Coulomb)
    mu = 0.0  # float, average friction coefficient
    count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        count += 1
    if count > 0:
        mu /= float(count)

    # v_t = v_rel - v_n * n  (wp.vec3, tangential velocity, world)
    vt = rel_v - vn * n
    vt_len = wp.length(vt)  # float

    if mu > 0.0 and vt_len > 1e-6:
        # t = v_t / |v_t|  (wp.vec3, tangent direction, world)
        t = vt / vt_len

        ang_ta = wp.cross(ra, t)                          # wp.vec3, world
        ang_tb = wp.cross(rb, t)
        ang_ta_local = wp.quat_rotate_inv(q_a, ang_ta)    # wp.vec3, body-local
        ang_tb_local = wp.quat_rotate_inv(q_b, ang_tb)

        Kt = inv_m_a + inv_m_b
        Kt += wp.dot(ang_ta_local, inv_I_a * ang_ta_local)
        Kt += wp.dot(ang_tb_local, inv_I_b * ang_tb_local)

        if Kt > 0.0:
            # lambda_t = -|v_t| / K_t,  clamped to [-mu*lambda_n, mu*lambda_n]  (Coulomb cone)
            lambda_t = -vt_len / Kt
            lambda_t = wp.clamp(lambda_t, -mu * lambda_n, mu * lambda_n)

            if body_a >= 0:
                dw_ta = wp.quat_rotate(q_a, inv_I_a * ang_ta_local)
                wp.atomic_add(
                    body_deltas, body_a,
                    wp.spatial_vector(
                        -t * (inv_m_a * lambda_t),
                        -dw_ta * lambda_t,
                    ),
                )

            if body_b >= 0:
                dw_tb = wp.quat_rotate(q_b, inv_I_b * ang_tb_local)
                wp.atomic_add(
                    body_deltas, body_b,
                    wp.spatial_vector(
                        t * (inv_m_b * lambda_t),
                        dw_tb * lambda_t,
                    ),
                )

    # 3) TORSIONAL FRICTION
    # dw = w_b - w_a  (wp.vec3, relative angular vel, world)
    delta_w = w_b - w_a
    # w_n = dw . n  (float, normal component of relative angular vel)
    wn = wp.dot(delta_w, n)

    torsion_mu = 0.0  # float
    if shape_a >= 0:
        torsion_mu += shape_material_torsional_friction[shape_a]
    if shape_b >= 0:
        torsion_mu += shape_material_torsional_friction[shape_b]
    torsion_mu *= 0.5

    if torsion_mu > 0.0 and wp.abs(wn) > 1e-6:
        n_local_a = wp.quat_rotate_inv(q_a, n)
        n_local_b = wp.quat_rotate_inv(q_b, n)
        Ktw = wp.dot(n_local_a, inv_I_a * n_local_a) + wp.dot(n_local_b, inv_I_b * n_local_b)
        if Ktw > 0.0:
            lambda_tw = -wn / Ktw
            lambda_tw = wp.clamp(lambda_tw, -torsion_mu * lambda_n, torsion_mu * lambda_n)

            if body_a >= 0:
                dw_tw_a = wp.quat_rotate(q_a, inv_I_a * n_local_a)
                wp.atomic_add(body_deltas, body_a, wp.spatial_vector(wp.vec3(), -dw_tw_a * lambda_tw))
            if body_b >= 0:
                dw_tw_b = wp.quat_rotate(q_b, inv_I_b * n_local_b)
                wp.atomic_add(body_deltas, body_b, wp.spatial_vector(wp.vec3(), dw_tw_b * lambda_tw))

    # 4) ROLLING FRICTION
    # w_t = dw - w_n * n  (wp.vec3, tangential angular vel, world)
    w_t = delta_w - wn * n
    w_t_len = wp.length(w_t)  # float

    roll_mu = 0.0  # float
    if shape_a >= 0:
        roll_mu += shape_material_rolling_friction[shape_a]
    if shape_b >= 0:
        roll_mu += shape_material_rolling_friction[shape_b]
    roll_mu *= 0.5

    if roll_mu > 0.0 and w_t_len > 1e-6:
        t_dir = w_t / w_t_len
        t_local_a = wp.quat_rotate_inv(q_a, t_dir)
        t_local_b = wp.quat_rotate_inv(q_b, t_dir)
        Kroll = wp.dot(t_local_a, inv_I_a * t_local_a) + wp.dot(t_local_b, inv_I_b * t_local_b)

        if Kroll > 0.0:
            lambda_roll = -w_t_len / Kroll
            lambda_roll = wp.clamp(lambda_roll, -roll_mu * lambda_n, roll_mu * lambda_n)

            if body_a >= 0:
                dw_roll_a = wp.quat_rotate(q_a, inv_I_a * t_local_a)
                wp.atomic_add(body_deltas, body_a, wp.spatial_vector(wp.vec3(), -dw_roll_a * lambda_roll))
            if body_b >= 0:
                dw_roll_b = wp.quat_rotate(q_b, inv_I_b * t_local_b)
                wp.atomic_add(body_deltas, body_b, wp.spatial_vector(wp.vec3(), dw_roll_b * lambda_roll))


@wp.kernel
def si_apply_body_contact_deltas(
    body_deltas: wp.array(dtype=wp.spatial_vector),
    body_contact_count: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    # outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Jacobi averaging: delta_applied = delta_accumulated / N_contacts"""
    tid = wp.tid()

    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        return

    count = body_contact_count[tid]  # float
    if count < 1.0:
        return

    delta = body_deltas[tid]         # wp.spatial_vector
    weight = 1.0 / count             # float

    qd = body_qd[tid]
    v = wp.spatial_top(qd) + wp.spatial_top(delta) * weight
    w = wp.spatial_bottom(qd) + wp.spatial_bottom(delta) * weight

    body_qd[tid] = wp.spatial_vector(v, w)


@wp.func
def velocity_at_point_si(qd: wp.spatial_vector, r: wp.vec3) -> wp.vec3:
    """Compute velocity at a point offset from COM."""
    v = wp.spatial_top(qd)    # linear velocity
    w = wp.spatial_bottom(qd)  # angular velocity
    return v + wp.cross(w, r)


@wp.kernel
def si_apply_rigid_restitution(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),  # velocity before constraint iterations
    body_com: wp.array(dtype=wp.vec3),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_material_restitution: wp.array(dtype=float),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    # outputs
    body_deltas: wp.array(dtype=wp.spatial_vector),
):
    """Apply restitution impulse after constraint iterations.

    Based on XPBD restitution (Detailed Rigid Body Simulation with Extended Position Based Dynamics, Eq. 34):
    dv = (-vn_new - e * vn_old) / K

    where:
    - vn_new: relative normal velocity after constraint iterations
    - vn_old: relative normal velocity before constraint iterations (+ gravity*dt)
    - e: coefficient of restitution
    """
    tid = wp.tid()

    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1

    # Get average restitution from materials
    mat_count = 0
    restitution = 0.0
    if shape_a >= 0:
        mat_count += 1
        restitution += shape_material_restitution[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_count += 1
        restitution += shape_material_restitution[shape_b]
        body_b = shape_body[shape_b]

    if mat_count > 0:
        restitution /= float(mat_count)

    if body_a == body_b:
        return

    # Skip if restitution is zero
    if restitution <= 0.0:
        return

    # Get transforms
    X_wa = wp.transform_identity()
    X_wb = wp.transform_identity()
    if body_a >= 0:
        X_wa = body_q[body_a]
    if body_b >= 0:
        X_wb = body_q[body_b]

    # Contact points in world space
    pa = wp.transform_point(X_wa, contact_point0[tid])
    pb = wp.transform_point(X_wb, contact_point1[tid])

    # Check penetration
    n = -contact_normal[tid]
    thickness = contact_thickness0[tid] + contact_thickness1[tid]
    d = wp.dot(n, pb - pa) - thickness

    if d >= 0.0:
        return  # Not in contact

    # Body properties
    inv_m_a = 0.0
    inv_m_b = 0.0
    inv_I_a = wp.mat33()
    inv_I_b = wp.mat33()
    com_a = wp.vec3()
    com_b = wp.vec3()

    if body_a >= 0:
        inv_m_a = body_inv_m[body_a]
        inv_I_a = body_inv_I[body_a]
        com_a = body_com[body_a]
    if body_b >= 0:
        inv_m_b = body_inv_m[body_b]
        inv_I_b = body_inv_I[body_b]
        com_b = body_com[body_b]

    # Lever arms from COM to contact point
    ra = pa - wp.transform_point(X_wa, com_a)
    rb = pb - wp.transform_point(X_wb, com_b)

    # Rotations for transforming angular terms
    q_a = wp.transform_get_rotation(X_wa)
    q_b = wp.transform_get_rotation(X_wb)

    # Compute velocities at contact points
    # Old velocity (before iterations) + gravity contribution
    v_a_old = wp.vec3()
    v_b_old = wp.vec3()
    v_a_new = wp.vec3()
    v_b_new = wp.vec3()

    if body_a >= 0:
        v_a_old = velocity_at_point_si(body_qd_prev[body_a], ra) + gravity[0] * dt
        v_a_new = velocity_at_point_si(body_qd[body_a], ra)
    if body_b >= 0:
        v_b_old = velocity_at_point_si(body_qd_prev[body_b], rb) + gravity[0] * dt
        v_b_new = velocity_at_point_si(body_qd[body_b], rb)

    # Relative normal velocities
    vn_old = wp.dot(n, v_a_old - v_b_old)
    vn_new = wp.dot(n, v_a_new - v_b_new)

    # Only apply restitution if bodies were approaching before
    if vn_old >= 0.0:
        return

    # Compute effective mass
    ang_a = wp.cross(ra, n)
    ang_b = wp.cross(rb, n)
    ang_a_local = wp.quat_rotate_inv(q_a, ang_a)
    ang_b_local = wp.quat_rotate_inv(q_b, ang_b)

    K = inv_m_a + inv_m_b
    K += wp.dot(ang_a_local, inv_I_a * ang_a_local)
    K += wp.dot(ang_b_local, inv_I_b * ang_b_local)

    if K <= 0.0:
        return

    # Restitution impulse: Eq. 34 from XPBD paper
    # dv = (-vn_new - e * vn_old) / K
    dv = (-vn_new - restitution * vn_old) / K

    # Apply impulse to both bodies
    if body_a >= 0:
        dw_a = wp.quat_rotate(q_a, inv_I_a * ang_a_local)
        wp.atomic_add(body_deltas, body_a, wp.spatial_vector(n * inv_m_a * dv, dw_a * dv))

    if body_b >= 0:
        dw_b = wp.quat_rotate(q_b, inv_I_b * ang_b_local)
        wp.atomic_add(body_deltas, body_b, wp.spatial_vector(-n * inv_m_b * dv, -dw_b * dv))


# @wp.kernel
# def si_apply_restitution_deltas(
#     body_deltas: wp.array(dtype=wp.spatial_vector),
#     body_inv_mass: wp.array(dtype=float),
#     # outputs
#     body_qd: wp.array(dtype=wp.spatial_vector),
# ):
#     """Apply restitution deltas directly to body velocities."""
#     tid = wp.tid()
#
#     inv_m = body_inv_mass[tid]
#     if inv_m == 0.0:
#         return
#
#     body_qd[tid] = wp.spatial_vector(v, w)

@wp.kernel
def si_apply_restitution_deltas(
    body_deltas: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    # outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Apply restitution deltas directly to body velocities."""
    tid = wp.tid()
    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        return
    delta = body_deltas[tid]
    v = wp.spatial_top(delta)      # linear delta
    w = wp.spatial_bottom(delta)   # angular delta
    cur = body_qd[tid]
    body_qd[tid] = wp.spatial_vector(
        wp.spatial_top(cur) + v,
        wp.spatial_bottom(cur) + w,
    )


@wp.kernel
def apply_joint_actuation(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_f: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Accumulate control forces/torques from joint actuators into body_f."""
    tid = wp.tid()
    jtype = joint_type[tid]
    if jtype == JointType.FIXED:
        return

    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    pose_c = body_q[id_c]
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(pose_c) - wp.transform_point(pose_c, com_c)

    qd_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    t_total = wp.vec3()
    f_total = wp.vec3()

    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        f_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
        t_total = wp.vec3(joint_f[qd_start + 3], joint_f[qd_start + 4], joint_f[qd_start + 5])
    elif jtype == JointType.BALL:
        t_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
    elif jtype == JointType.REVOLUTE or jtype == JointType.PRISMATIC or jtype == JointType.D6:
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            f = joint_f[qd_start + 0]
            f_total += f * wp.transform_vector(X_wp, axis)
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            f = joint_f[qd_start + 1]
            f_total += f * wp.transform_vector(X_wp, axis)
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            f = joint_f[qd_start + 2]
            f_total += f * wp.transform_vector(X_wp, axis)
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            f = joint_f[qd_start + lin_axis_count + 0]
            t_total += f * wp.transform_vector(X_wp, axis)
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            f = joint_f[qd_start + lin_axis_count + 1]
            t_total += f * wp.transform_vector(X_wp, axis)
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            f = joint_f[qd_start + lin_axis_count + 2]
            t_total += f * wp.transform_vector(X_wp, axis)

    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))
    wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total)))
