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

"""Body joint constraint group: positional + angular joint limits, targets, and forces."""

from __future__ import annotations

import warp as wp

from .._types import JointType
from ..math_utils import vec_abs, vec_max, vec_min
from ..constraint_base import ConstraintGroup
from ..math_utils import compute_angular_correction, compute_positional_correction
from ..context import ConstraintPhase, XPBDContext

# ────────────────────────────────────────────────────────────────────
# Warp helper functions (@wp.func)
# ────────────────────────────────────────────────────────────────────


@wp.func
def _update_joint_axis_limits(
    axis: wp.vec3, limit_lower: float, limit_upper: float, input_limits: wp.spatial_vector
):
    # update the 3D linear/angular limits (spatial_vector [lower, upper]) given the axis vector and limits
    lo_temp = axis * limit_lower
    up_temp = axis * limit_upper
    lo = vec_min(lo_temp, up_temp)
    up = vec_max(lo_temp, up_temp)
    input_lower = wp.spatial_top(input_limits)
    input_upper = wp.spatial_bottom(input_limits)
    lower = vec_min(input_lower, lo)
    upper = vec_max(input_upper, up)
    return wp.spatial_vector(lower, upper)


@wp.func
def _update_joint_axis_weighted_target(
    axis: wp.vec3, target: float, weight: float, input_target_weight: wp.spatial_vector
):
    axis_targets = wp.spatial_top(input_target_weight)
    axis_weights = wp.spatial_bottom(input_target_weight)

    weighted_axis = axis * weight
    axis_targets += weighted_axis * target  # weighted target (to be normalized later by sum of weights)
    axis_weights += vec_abs(weighted_axis)

    return wp.spatial_vector(axis_targets, axis_weights)



# compute_positional_correction / compute_angular_correction moved to
# math_utils.py and are imported above as public names.


# --------------------------------------------------------------------
# Warp kernels
# --------------------------------------------------------------------


@wp.kernel
def _apply_joint_forces(
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
    tid = wp.tid()
    type = joint_type[tid]
    if type == JointType.FIXED:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    # X_cj = joint_X_c[tid]

    X_wp = X_pj
    pose_p = X_pj  # world = local * inv(pose_p)
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    # # local joint rotations
    # q_p = wp.transform_get_rotation(X_wp)
    # q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    qd_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    if type == JointType.FREE or type == JointType.DISTANCE:
        f_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
        t_total = wp.vec3(joint_f[qd_start + 3], joint_f[qd_start + 4], joint_f[qd_start + 5])
    elif type == JointType.BALL:
        t_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])

    elif type == JointType.REVOLUTE or type == JointType.PRISMATIC or type == JointType.D6:
        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a dynamic for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            f = joint_f[qd_start + 0]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            f = joint_f[qd_start + 1]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            f = joint_f[qd_start + 2]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p

        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            f = joint_f[qd_start + lin_axis_count + 0]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            f = joint_f[qd_start + lin_axis_count + 1]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            f = joint_f[qd_start + lin_axis_count + 2]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p

    else:
        print("joint type not handled in apply_joint_forces")

    # write forces
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))
    wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total)))


@wp.kernel
def _solve_body_joints(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_linear_compliance: float,
    joint_angular_compliance: float,
    angular_relaxation: float,
    linear_relaxation: float,
    dt: float,
    joint_linear_lambdas: wp.array2d(dtype=float),
    joint_angular_lambdas: wp.array2d(dtype=float),
    deltas: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    type = joint_type[tid]

    if joint_enabled[tid] == 0:
        return
    if type == JointType.FREE:
        return
    # if type == JointType.FIXED:
    #     return
    # if type == JointType.REVOLUTE:
    #     return
    # if type == JointType.PRISMATIC:
    #     return
    # if type == JointType.BALL:
    #     return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    vel_p = wp.vec3(0.0)
    omega_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
        vel_p = wp.spatial_top(body_qd[id_p])
        omega_p = wp.spatial_bottom(body_qd[id_p])

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    vel_c = wp.spatial_top(body_qd[id_c])
    omega_c = wp.spatial_bottom(body_qd[id_c])

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    rel_pose = wp.transform_inverse(X_wp) * X_wc
    rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    # x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    linear_compliance = joint_linear_compliance
    angular_compliance = joint_angular_compliance

    axis_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    world_com_p = wp.transform_point(pose_p, com_p)
    world_com_c = wp.transform_point(pose_c, com_c)

    # handle positional constraints
    if type == JointType.DISTANCE:
        r_p = wp.transform_get_translation(X_wp) - world_com_p
        r_c = wp.transform_get_translation(X_wc) - world_com_c
        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]
        if lower < 0.0 and upper < 0.0:
            # no limits
            return
        d = wp.length(rel_p)
        err = 0.0
        if lower >= 0.0 and d < lower:
            err = d - lower
            # use a more descriptive direction vector for the constraint
            # in case the joint parent and child anchors are very close
            rel_p = err * wp.normalize(world_com_c - world_com_p)
        elif upper >= 0.0 and d > upper:
            err = d - upper

        if wp.abs(err) > 1e-9:
            # compute gradients
            linear_c = rel_p
            linear_p = -linear_c
            r_c = x_c - world_com_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )
            lambda_in = 0.0
            compliance = linear_compliance
            ke = joint_target_ke[axis_start]
            if ke > 0.0:
                compliance = 1.0 / ke
            damping = joint_target_kd[axis_start]
            lambda_in = joint_linear_lambdas[tid, 0]

            d_lambda = compute_positional_correction(
                err,
                derr,
                pose_p,
                pose_c,
                m_inv_p,
                m_inv_c,
                I_inv_p,
                I_inv_c,
                linear_p,
                linear_c,
                angular_p,
                angular_c,
                lambda_in,
                compliance,
                damping,
                dt,
            )
            joint_linear_lambdas[tid, 0] = lambda_in + d_lambda

            lin_delta_p += linear_p * (d_lambda * linear_relaxation)
            ang_delta_p += angular_p * (d_lambda * angular_relaxation)
            lin_delta_c += linear_c * (d_lambda * linear_relaxation)
            ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    else:
        # compute joint target, stiffness, damping
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        axis_target_pos_ke = wp.spatial_vector()
        axis_target_vel_kd = wp.spatial_vector()
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if lin_axis_count > 0:
            axis = joint_axis[axis_start]
            lo_temp = axis * joint_limit_lower[axis_start]
            up_temp = axis * joint_limit_upper[axis_start]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            ke = joint_target_ke[axis_start]
            kd = joint_target_kd[axis_start]
            target_pos = joint_target_pos[axis_start]
            target_vel = joint_target_vel[axis_start]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = _update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = _update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if lin_axis_count > 1:
            axis_idx = axis_start + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = _update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_pos[axis_idx]
            target_vel = joint_target_vel[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = _update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = _update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if lin_axis_count > 2:
            axis_idx = axis_start + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = _update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_pos[axis_idx]
            target_vel = joint_target_vel[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = _update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = _update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)

        axis_target_pos = wp.spatial_top(axis_target_pos_ke)
        axis_stiffness = wp.spatial_bottom(axis_target_pos_ke)
        axis_target_vel = wp.spatial_top(axis_target_vel_kd)
        axis_damping = wp.spatial_bottom(axis_target_vel_kd)
        for i in range(3):
            if axis_stiffness[i] > 0.0:
                axis_target_pos[i] /= axis_stiffness[i]
        for i in range(3):
            if axis_damping[i] > 0.0:
                axis_target_vel[i] /= axis_damping[i]
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        frame_p = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))
        # note that x_c appearing in both is correct
        r_p = x_c - world_com_p
        r_c = x_c - wp.transform_point(pose_c, com_c)

        # for loop will be unrolled, so we can modify local variables
        for dim in range(3):
            e = rel_p[dim]

            # compute gradients
            linear_c = wp.vec3(frame_p[0, dim], frame_p[1, dim], frame_p[2, dim])
            linear_p = -linear_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )

            err = 0.0
            compliance = linear_compliance
            damping = 0.0

            target_vel = axis_target_vel[dim]
            derr_rel = derr - target_vel

            # consider joint limits irrespective of axis mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
            elif e > upper:
                err = e - upper
            else:
                target_pos = axis_target_pos[dim]
                target_pos = wp.clamp(target_pos, lower, upper)

                if axis_stiffness[dim] > 0.0:
                    err = e - target_pos
                    compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif axis_damping[dim] > 0.0:
                    compliance = 1.0 / axis_damping[dim]
                    damping = axis_damping[dim]

            if wp.abs(err) > 1e-9 or wp.abs(derr_rel) > 1e-9:
                # lambda_in = 0.0
                lambda_in = joint_linear_lambdas[tid, dim]
                d_lambda = compute_positional_correction(
                    err,
                    derr_rel,
                    pose_p,
                    pose_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    linear_p,
                    linear_c,
                    angular_p,
                    angular_c,
                    lambda_in,
                    compliance,
                    damping,
                    dt,
                )
                joint_linear_lambdas[tid, 0] = lambda_in + d_lambda
                lin_delta_p += linear_p * (d_lambda * linear_relaxation)
                ang_delta_p += angular_p * (d_lambda * angular_relaxation)
                lin_delta_c += linear_c * (d_lambda * linear_relaxation)
                ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    if type == JointType.FIXED or type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        # handle angular constraints

        # local joint rotations
        q_p = wp.transform_get_rotation(X_wp)
        q_c = wp.transform_get_rotation(X_wc)

        # make quats lie in same hemisphere
        if wp.dot(q_p, q_c) < 0.0:
            q_c *= -1.0

        rel_q = wp.quat_inverse(q_p) * q_c

        qtwist = wp.normalize(wp.quat(rel_q[0], 0.0, 0.0, rel_q[3]))
        qswing = rel_q * wp.quat_inverse(qtwist)

        # decompose to a compound rotation each axis
        s = wp.sqrt(rel_q[0] * rel_q[0] + rel_q[3] * rel_q[3])
        invs = 1.0 / s
        invscube = invs * invs * invs

        # handle axis-angle joints

        # rescale twist from quaternion space to angular
        err_0 = 2.0 * wp.asin(wp.clamp(qtwist[0], -1.0, 1.0))
        err_1 = qswing[1]
        err_2 = qswing[2]
        # analytic gradients of swing-twist decomposition
        grad_0 = wp.quat(invs - rel_q[0] * rel_q[0] * invscube, 0.0, 0.0, -(rel_q[3] * rel_q[0]) * invscube)
        grad_1 = wp.quat(
            -rel_q[3] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
            rel_q[3] * invs,
            -rel_q[0] * invs,
            rel_q[0] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
        )
        grad_2 = wp.quat(
            rel_q[3] * (rel_q[3] * rel_q[1] - rel_q[0] * rel_q[2]) * invscube,
            rel_q[0] * invs,
            rel_q[3] * invs,
            rel_q[0] * (rel_q[2] * rel_q[0] - rel_q[3] * rel_q[1]) * invscube,
        )
        grad_0 *= 2.0 / wp.abs(qtwist[3])
        # grad_0 *= 2.0 / wp.sqrt(1.0-qtwist[0]*qtwist[0])	# derivative of asin(x) = 1/sqrt(1-x^2)

        # rescale swing
        swing_sq = qswing[3] * qswing[3]
        # if swing axis magnitude close to zero vector, just treat in quaternion space
        angularEps = 1.0e-4
        if swing_sq + angularEps < 1.0:
            d = wp.sqrt(1.0 - qswing[3] * qswing[3])
            theta = 2.0 * wp.acos(wp.clamp(qswing[3], -1.0, 1.0))
            scale = theta / d

            err_1 *= scale
            err_2 *= scale

            grad_1 *= scale
            grad_2 *= scale

        errs = wp.vec3(err_0, err_1, err_2)
        grad_x = wp.vec3(grad_0[0], grad_1[0], grad_2[0])
        grad_y = wp.vec3(grad_0[1], grad_1[1], grad_2[1])
        grad_z = wp.vec3(grad_0[2], grad_1[2], grad_2[2])
        grad_w = wp.vec3(grad_0[3], grad_1[3], grad_2[3])

        # compute joint target, stiffness, damping
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        axis_target_pos_ke = wp.spatial_vector()  # [weighted_target_pos, ke_weights]
        axis_target_vel_kd = wp.spatial_vector()  # [weighted_target_vel, kd_weights]
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if ang_axis_count > 0:
            axis_idx = axis_start + lin_axis_count
            axis = joint_axis[axis_idx]
            lo_temp = axis * joint_limit_lower[axis_idx]
            up_temp = axis * joint_limit_upper[axis_idx]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_pos[axis_idx]
            target_vel = joint_target_vel[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = _update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = _update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if ang_axis_count > 1:
            axis_idx = axis_start + lin_axis_count + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = _update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_pos[axis_idx]
            target_vel = joint_target_vel[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = _update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = _update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if ang_axis_count > 2:
            axis_idx = axis_start + lin_axis_count + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = _update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_pos[axis_idx]
            target_vel = joint_target_vel[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = _update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = _update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)

        axis_target_pos = wp.spatial_top(axis_target_pos_ke)
        axis_stiffness = wp.spatial_bottom(axis_target_pos_ke)
        axis_target_vel = wp.spatial_top(axis_target_vel_kd)
        axis_damping = wp.spatial_bottom(axis_target_vel_kd)
        for i in range(3):
            if axis_stiffness[i] > 0.0:
                axis_target_pos[i] /= axis_stiffness[i]
        for i in range(3):
            if axis_damping[i] > 0.0:
                axis_target_vel[i] /= axis_damping[i]
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        for dim in range(3):
            e = errs[dim]

            # analytic gradients of swing-twist decomposition
            grad = wp.quat(grad_x[dim], grad_y[dim], grad_z[dim], grad_w[dim])

            quat_c = 0.5 * q_p * grad * wp.quat_inverse(q_c)
            angular_c = wp.vec3(quat_c[0], quat_c[1], quat_c[2])
            angular_p = -angular_c
            # time derivative of the constraint
            derr = wp.dot(angular_p, omega_p) + wp.dot(angular_c, omega_c)

            err = 0.0
            compliance = angular_compliance
            damping = 0.0

            target_vel = axis_target_vel[dim]
            angular_c_len = wp.length(angular_c)
            derr_rel = derr - target_vel * angular_c_len

            # consider joint limits irrespective of mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
            elif e > upper:
                err = e - upper
            else:
                target_pos = axis_target_pos[dim]
                target_pos = wp.clamp(target_pos, lower, upper)

                if axis_stiffness[dim] > 0.0:
                    err = e - target_pos
                    compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif axis_damping[dim] > 0.0:
                    damping = axis_damping[dim]
                    compliance = 1.0 / axis_damping[dim]

            lambda_in = joint_angular_lambdas[tid, dim]
            raw_d_lambda = (
                compute_angular_correction(
                    err, derr_rel, pose_p, pose_c, I_inv_p, I_inv_c, angular_p, angular_c, lambda_in, compliance, damping, dt
                )
                * angular_relaxation
            )
            joint_angular_lambdas[tid, dim] = lambda_in + raw_d_lambda
            d_lambda = raw_d_lambda * angular_relaxation
            # update deltas
            ang_delta_p += angular_p * d_lambda
            ang_delta_c += angular_c * d_lambda

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(lin_delta_p, ang_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(lin_delta_c, ang_delta_c))

@wp.kernel
def apply_body_delta_velocities(
    deltas: wp.array(dtype=wp.spatial_vector),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    wp.atomic_add(qd_out, tid, deltas[tid])

@wp.kernel
def apply_substep_damping(
    body_q:          wp.array(dtype=wp.transform),
    body_qd:         wp.array(dtype=wp.spatial_vector),
    body_inv_m:      wp.array(dtype=float),
    body_inv_I:      wp.array(dtype=wp.mat33),
    body_com:        wp.array(dtype=wp.vec3),
    joint_type:      wp.array(dtype=int),
    joint_enabled:   wp.array(dtype=bool),
    joint_parent:    wp.array(dtype=int),
    joint_child:     wp.array(dtype=int),
    joint_X_p:       wp.array(dtype=wp.transform),
    joint_X_c:       wp.array(dtype=wp.transform),
    joint_target_kd: wp.array(dtype=float),
    joint_qd_start:  wp.array(dtype=int),
    joint_dof_dim:   wp.array(dtype=int, ndim=2),
    dt:              float,
    # output
    deltas:          wp.array(dtype=wp.spatial_vector),
):
    """Velocity-level Rayleigh joint damping.

    This follows the 2020 XPBD velocity solve idea: damp the full relative
    linear and angular velocity vectors, using the maximum joint target
    damping value across each axis group. Fixed joints have no DOF axes, so
    they use the first damping slot as a fallback.
    """
    tid = wp.tid()
    if not joint_enabled[tid]:
        return

    type = joint_type[tid]
    # Free joints do not define constraint axes.
    if type == JointType.FREE:
        return

    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    # The child body must be dynamic.
    if body_inv_m[id_c] == 0.0:
        return

    # Joint anchors in world space.
    X_wp = joint_X_p[tid]
    pose_p = X_wp
    com_p  = wp.vec3(0.0)
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if id_p >= 0:
        pose_p  = body_q[id_p]
        X_wp    = pose_p * X_wp
        com_p   = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]

    pose_c  = body_q[id_c]
    com_c   = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]

    # Body velocities.
    v_p     = wp.vec3(0.0)
    omega_p = wp.vec3(0.0)
    if id_p >= 0:
        vel_p   = body_qd[id_p]
        v_p     = wp.spatial_top(vel_p)
        omega_p = wp.spatial_bottom(vel_p)

    vel_c   = body_qd[id_c]
    v_c     = wp.spatial_top(vel_c)
    omega_c = wp.spatial_bottom(vel_c)

    # Use the maximum damping value in each axis group.
    axis_start     = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]
    total_dof      = lin_axis_count + ang_axis_count

    mu_lin = float(0.0)
    mu_ang = float(0.0)

    if total_dof == 0:
        # Fixed joints share the first damping slot for linear and angular damping.
        kd_fixed = joint_target_kd[axis_start]
        mu_lin = kd_fixed
        mu_ang = kd_fixed
    else:
        # Maximum linear-axis damping.
        for dim in range(lin_axis_count):
            kd = joint_target_kd[axis_start + dim]
            if kd > mu_lin:
                mu_lin = kd
        # Maximum angular-axis damping.
        for dim in range(ang_axis_count):
            kd = joint_target_kd[axis_start + lin_axis_count + dim]
            if kd > mu_ang:
                mu_ang = kd

    # Zero damping means this pass has nothing to do.
    if mu_lin == 0.0 and mu_ang == 0.0:
        return

    # Moment arms from each center of mass to the joint anchor.
    x_anchor_p = wp.transform_get_translation(X_wp)
    x_anchor_c = wp.transform_get_translation(pose_c * joint_X_c[tid])

    world_com_p = wp.transform_point(pose_p, com_p)
    world_com_c = wp.transform_point(pose_c, com_c)

    r_p = x_anchor_p - world_com_p
    r_c = x_anchor_c - world_com_c

    q_p = wp.transform_get_rotation(pose_p)
    q_c = wp.transform_get_rotation(pose_c)

    lin_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # Linear Rayleigh damping.
    if mu_lin > 0.0:
        v_contact_c = v_c + wp.cross(omega_c, r_c)
        v_contact_p = v_p + wp.cross(omega_p, r_p)
        v_rel = v_contact_c - v_contact_p

        v_rel_mag = wp.length(v_rel)
        if v_rel_mag > 1.0e-9:
            scale = wp.min(mu_lin * dt, 1.0)
            dv_damp = -v_rel * scale

            n_lin = v_rel / v_rel_mag

            r_p_x_n = wp.quat_rotate_inv(q_p, wp.cross(r_p, n_lin))
            r_c_x_n = wp.quat_rotate_inv(q_c, wp.cross(r_c, n_lin))
            w_lin = (m_inv_p + wp.dot(r_p_x_n, I_inv_p * r_p_x_n)
                   + m_inv_c + wp.dot(r_c_x_n, I_inv_c * r_c_x_n))

            if w_lin > 0.0:
                impulse_mag = wp.dot(dv_damp, n_lin) / w_lin
                p_lin = n_lin * impulse_mag

                lin_delta_p -= p_lin * m_inv_p
                lin_delta_c += p_lin * m_inv_c
                ang_delta_p -= wp.quat_rotate(q_p,
                                I_inv_p * r_p_x_n * impulse_mag)
                ang_delta_c += wp.quat_rotate(q_c,
                                I_inv_c * r_c_x_n * impulse_mag)

    # Angular Rayleigh damping.
    if mu_ang > 0.0:
        omega_rel = omega_c - omega_p

        omega_rel_mag = wp.length(omega_rel)
        if omega_rel_mag > 1.0e-9:
            scale = wp.min(mu_ang * dt, 1.0)
            dw_damp = -omega_rel * scale

            n_ang = omega_rel / omega_rel_mag

            n_p = wp.quat_rotate_inv(q_p, n_ang)
            n_c = wp.quat_rotate_inv(q_c, n_ang)
            w_ang = (wp.dot(n_p, I_inv_p * n_p)
                   + wp.dot(n_c, I_inv_c * n_c))

            if w_ang > 0.0:
                impulse_mag = wp.dot(dw_damp, n_ang) / w_ang
                ang_delta_p -= wp.quat_rotate(q_p,
                                I_inv_p * n_p * impulse_mag)
                ang_delta_c += wp.quat_rotate(q_c,
                                I_inv_c * n_c * impulse_mag)

    if id_p >= 0:
        wp.atomic_add(deltas, id_p,
                      wp.spatial_vector(lin_delta_p, ang_delta_p))
    wp.atomic_add(deltas, id_c,
                  wp.spatial_vector(lin_delta_c, ang_delta_c))




# --------------------------------------------------------------------
# Constraint group implementation
# --------------------------------------------------------------------


class BodyJointConstraint(ConstraintGroup):
    """Rigid-body joint constraint (positional + angular).

    Supports all joint types: FIXED, REVOLUTE, PRISMATIC, BALL, D6, DISTANCE.
    Handles joint limits, position/velocity targets, and compliance.

    The :meth:`initialize` step applies joint generalised forces to
    ``state_in.body_f`` (pre-integration), while :meth:`project` solves the
    positional / angular constraint via ``_solve_body_joints``.
    """

    phase = ConstraintPhase.BODY_JOINT

    def __init__(
        self,
        linear_relaxation: float = 0.7,
        angular_relaxation: float = 0.4,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        enable_substep_damping: bool = True,
        warm_start: bool = True,
    ) -> None:
        self.linear_relaxation = linear_relaxation
        self.angular_relaxation = angular_relaxation
        self.linear_compliance = linear_compliance
        self.angular_compliance = angular_compliance
        self.enable_substep_damping = enable_substep_damping
        self.joint_linear_lambdas: wp.array | None = None
        self.joint_angular_lambdas: wp.array | None = None
        self.warm_start = warm_start

    def is_active(self, model, contacts) -> bool:
        return model.joint_count > 0

    def initialize(self, ctx: XPBDContext) -> None:
        """Apply joint generalised forces to body_f (before integration)."""
        model = ctx.model
        if self.joint_linear_lambdas is None or self.joint_linear_lambdas.shape[0] != model.joint_count:
            self.joint_linear_lambdas = wp.zeros((model.joint_count, 3), dtype=float, device=model.device)

        if self.joint_angular_lambdas is None or self.joint_angular_lambdas.shape[0] != model.joint_count:
            self.joint_angular_lambdas = wp.zeros((model.joint_count, 3), dtype=float, device=model.device)

        wp.launch(
            kernel=_apply_joint_forces,
            dim=model.joint_count,
            inputs=[
                ctx.state_in.body_q,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_axis,
                ctx.control.joint_f,
            ],
            outputs=[ctx.state_in.body_f],
            device=model.device,
        )

    def reset_iteration(self, ctx: XPBDContext, iteration: int) -> None:
        if iteration == 0:
            if self.joint_linear_lambdas is not None:
                self.joint_linear_lambdas.zero_()
            if self.joint_angular_lambdas is not None:
                self.joint_angular_lambdas.zero_()
        elif self.warm_start == False:
            self.joint_linear_lambdas.zero_()
            self.joint_angular_lambdas.zero_()

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        if self.enable_substep_damping and model.joint_count and model.body_count:
            ctx.body_deltas.zero_()
            wp.launch(
                kernel=apply_substep_damping,
                dim=model.joint_count,
                inputs=[
                    ctx.state_out.body_q,
                    ctx.state_out.body_qd,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.body_com,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_target_kd,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    ctx.dt
                ],
                outputs=[ctx.body_deltas],
                device=model.device,
            )

            # Damping deltas update velocity only.
            wp.launch(
                kernel=apply_body_delta_velocities,
                dim=model.body_count,
                inputs=[ctx.body_deltas],
                outputs=[ctx.state_out.body_qd],
                device=model.device,
            )
        wp.launch(
            kernel=_solve_body_joints,
            dim=model.joint_count,
            inputs=[
                ctx.body_q,
                ctx.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_axis,
                ctx.control.joint_target_pos,
                ctx.control.joint_target_vel,
                model.joint_target_ke,
                model.joint_target_kd,
                self.linear_compliance,
                self.angular_compliance,
                self.angular_relaxation,
                self.linear_relaxation,
                ctx.dt,
                self.joint_linear_lambdas,
                self.joint_angular_lambdas,
            ],
            outputs=[ctx.body_deltas],
            device=model.device,
        )
