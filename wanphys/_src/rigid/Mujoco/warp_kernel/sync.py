# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Runtime synchronization kernels for the WanPhys MuJoCo adapter.

These kernels update MuJoCo runtime buffers after the source model changes.
They are grouped by the same categories used by ``sync.py``:
- body inertial properties
- actuator / DOF / joint properties
- mocap and joint transforms
- geom, model, and equality-constraint properties
"""

from __future__ import annotations

import warp as wp

from .common import decode_actuator_axis, vec5, vec10, warp_quat_to_mj


@wp.kernel
def update_body_mass_ipos_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3f),
    body_mass: wp.array(dtype=float),
    body_gravcomp: wp.array(dtype=float),
    up_axis: int,
    body_ipos: wp.array2d(dtype=wp.vec3f),
    body_mass_out: wp.array2d(dtype=float),
    body_gravcomp_out: wp.array2d(dtype=float),
):
    """Refresh body COM, mass, and gravcomp in MuJoCo model storage."""
    world, mjc_body = wp.tid()
    source_body = mjc_body_to_newton[world, mjc_body]
    if source_body < 0:
        return

    # MuJoCo and the source model may use different "up" conventions, so COM
    # coordinates are rotated when the runtime is configured for Y-up.
    if up_axis == 1:
        body_ipos[world, mjc_body] = wp.vec3f(body_com[source_body][0], -body_com[source_body][2], body_com[source_body][1])
    else:
        body_ipos[world, mjc_body] = body_com[source_body]
    body_mass_out[world, mjc_body] = body_mass[source_body]
    if body_gravcomp:
        body_gravcomp_out[world, mjc_body] = body_gravcomp[source_body]


@wp.kernel
def update_body_inertia_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    body_inertia: wp.array(dtype=wp.mat33f),
    body_inertia_out: wp.array2d(dtype=wp.vec3f),
    body_iquat_out: wp.array2d(dtype=wp.quatf),
):
    """Diagonalize and upload body inertia into MuJoCo's principal-axis format."""
    world, mjc_body = wp.tid()
    source_body = mjc_body_to_newton[world, mjc_body]
    if source_body < 0:
        return
    I = body_inertia[source_body]
    eigenvectors, eigenvalues = wp.eig3(I)
    vecs_transposed = wp.transpose(eigenvectors)

    # Keep the same descending ordering MuJoCo expects for principal inertia.
    for i in range(2):
        for j in range(2 - i):
            if eigenvalues[j] < eigenvalues[j + 1]:
                temp_val = eigenvalues[j]
                eigenvalues[j] = eigenvalues[j + 1]
                eigenvalues[j + 1] = temp_val
                temp_vec = vecs_transposed[j]
                vecs_transposed[j] = vecs_transposed[j + 1]
                vecs_transposed[j + 1] = temp_vec
    q = wp.normalize(wp.quat_from_matrix(wp.transpose(vecs_transposed)))
    body_inertia_out[world, mjc_body] = eigenvalues
    body_iquat_out[world, mjc_body] = warp_quat_to_mj(q)


@wp.kernel
def update_axis_properties_kernel(
    mjc_actuator_to_newton_axis: wp.array2d(dtype=wp.int32),
    joint_target_kp: wp.array(dtype=float),
    joint_target_kv: wp.array(dtype=float),
    actuator_bias: wp.array2d(dtype=vec10),
    actuator_gain: wp.array2d(dtype=vec10),
):
    """Refresh actuator gain/bias parameters from source target gains."""
    world, mjc_actuator = wp.tid()
    source_axis, is_velocity, is_mapped = decode_actuator_axis(mjc_actuator_to_newton_axis[world, mjc_actuator])
    if not is_mapped:
        return
    if not is_velocity:
        kp = joint_target_kp[source_axis]
        actuator_bias[world, mjc_actuator][1] = -kp
        actuator_gain[world, mjc_actuator][0] = kp
    else:
        kv = joint_target_kv[source_axis]
        actuator_bias[world, mjc_actuator][2] = -kv
        actuator_gain[world, mjc_actuator][0] = kv


@wp.kernel
def update_dof_properties_kernel(
    mjc_dof_to_newton_dof: wp.array2d(dtype=wp.int32),
    joint_armature: wp.array(dtype=float),
    joint_friction: wp.array(dtype=float),
    joint_damping: wp.array(dtype=float),
    dof_solimp: wp.array(dtype=vec5),
    dof_solref: wp.array(dtype=wp.vec2),
    dof_armature: wp.array2d(dtype=float),
    dof_frictionloss: wp.array2d(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    dof_solimp_out: wp.array2d(dtype=vec5),
    dof_solref_out: wp.array2d(dtype=wp.vec2),
):
    """Refresh per-DOF armature, friction, damping, and optional solver params."""
    world, mjc_dof = wp.tid()
    source_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if source_dof < 0:
        return
    dof_armature[world, mjc_dof] = joint_armature[source_dof]
    dof_frictionloss[world, mjc_dof] = joint_friction[source_dof]
    if joint_damping:
        dof_damping[world, mjc_dof] = joint_damping[source_dof]
    if dof_solimp:
        dof_solimp_out[world, mjc_dof] = dof_solimp[source_dof]
    if dof_solref:
        dof_solref_out[world, mjc_dof] = dof_solref[source_dof]


@wp.kernel
def update_jnt_properties_kernel(
    mjc_jnt_to_newton_dof: wp.array2d(dtype=wp.int32),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_effort_limit: wp.array(dtype=float),
    solimplimit: wp.array(dtype=vec5),
    joint_stiffness: wp.array(dtype=float),
    limit_margin: wp.array(dtype=float),
    jnt_solimp: wp.array2d(dtype=vec5),
    jnt_solref: wp.array2d(dtype=wp.vec2),
    jnt_stiffness: wp.array2d(dtype=float),
    jnt_margin: wp.array2d(dtype=float),
    jnt_range: wp.array2d(dtype=wp.vec2),
    jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
):
    """Refresh joint-level limit metadata and actuator force ranges."""
    world, mjc_jnt = wp.tid()
    source_dof = mjc_jnt_to_newton_dof[world, mjc_jnt]
    if source_dof < 0:
        return
    if joint_limit_ke[source_dof] > 0.0:
        jnt_solref[world, mjc_jnt] = wp.vec2(-joint_limit_ke[source_dof], -joint_limit_kd[source_dof])
    if solimplimit:
        jnt_solimp[world, mjc_jnt] = solimplimit[source_dof]
    if joint_stiffness:
        jnt_stiffness[world, mjc_jnt] = joint_stiffness[source_dof]
    if limit_margin:
        jnt_margin[world, mjc_jnt] = limit_margin[source_dof]
    jnt_range[world, mjc_jnt] = wp.vec2(joint_limit_lower[source_dof], joint_limit_upper[source_dof])
    effort_limit = joint_effort_limit[source_dof]
    jnt_actfrcrange[world, mjc_jnt] = wp.vec2(-effort_limit, effort_limit)


@wp.kernel
def update_mocap_transforms_kernel(
    mjc_mocap_to_newton_jnt: wp.array2d(dtype=wp.int32),
    newton_joint_X_p: wp.array(dtype=wp.transform),
    newton_joint_X_c: wp.array(dtype=wp.transform),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
):
    """Refresh mocap target transforms for fixed-base or mocap-driven bodies."""
    world, mocap_idx = wp.tid()
    source_joint = mjc_mocap_to_newton_jnt[world, mocap_idx]
    if source_joint < 0:
        return
    tf = newton_joint_X_p[source_joint] * wp.transform_inverse(newton_joint_X_c[source_joint])
    mocap_pos[world, mocap_idx] = tf.p
    mocap_quat[world, mocap_idx] = warp_quat_to_mj(tf.q)


@wp.kernel
def update_joint_transforms_kernel(
    mjc_jnt_to_newton_jnt: wp.array2d(dtype=wp.int32),
    mjc_jnt_to_newton_dof: wp.array2d(dtype=wp.int32),
    mjc_jnt_bodyid: wp.array(dtype=wp.int32),
    mjc_jnt_type: wp.array(dtype=wp.int32),
    newton_joint_X_p: wp.array(dtype=wp.transform),
    newton_joint_X_c: wp.array(dtype=wp.transform),
    newton_joint_axis: wp.array(dtype=wp.vec3),
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
):
    """Refresh joint-local frames and attached body transforms in MuJoCo storage."""
    world, mjc_jnt = wp.tid()
    source_joint = mjc_jnt_to_newton_jnt[world, mjc_jnt]
    if source_joint < 0:
        return
    source_dof = mjc_jnt_to_newton_dof[world, mjc_jnt]
    if mjc_jnt_type[mjc_jnt] == 0:
        return
    child_xform = newton_joint_X_c[source_joint]
    parent_xform = newton_joint_X_p[source_joint]
    tf = parent_xform * wp.transform_inverse(child_xform)
    mjc_body = mjc_jnt_bodyid[mjc_jnt]
    body_pos[world, mjc_body] = tf.p
    body_quat[world, mjc_body] = warp_quat_to_mj(tf.q)
    if source_dof >= 0:
        jnt_axis[world, mjc_jnt] = wp.quat_rotate(child_xform.q, newton_joint_axis[source_dof])
    jnt_pos[world, mjc_jnt] = child_xform.p


@wp.kernel(enable_backward=False)
def update_shape_mappings_kernel(
    geom_to_shape_idx: wp.array(dtype=wp.int32),
    geom_is_static: wp.array(dtype=bool),
    shape_range_len: int,
    first_env_shape_base: int,
    mjc_geom_to_newton_shape: wp.array(dtype=wp.int32, ndim=2),
):
    """Expand template-world geom mappings into world-indexed runtime mappings."""
    world, geom_idx = wp.tid()
    template_or_static_idx = geom_to_shape_idx[geom_idx]
    if template_or_static_idx < 0:
        return
    if geom_is_static[geom_idx]:
        source_shape_idx = template_or_static_idx
    else:
        source_shape_idx = first_env_shape_base + template_or_static_idx + world * shape_range_len
    mjc_geom_to_newton_shape[world, geom_idx] = source_shape_idx


@wp.kernel
def update_model_properties_kernel(
    gravity_src: wp.array(dtype=wp.vec3),
    gravity_dst: wp.array(dtype=wp.vec3f),
):
    """Broadcast world-independent gravity into each MuJoCo runtime world."""
    world_idx = wp.tid()
    gravity_dst[world_idx] = gravity_src[0]


@wp.kernel
def update_geom_properties_kernel(
    shape_collision_radius: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_ke: wp.array(dtype=float),
    shape_kd: wp.array(dtype=float),
    shape_size: wp.array(dtype=wp.vec3f),
    shape_transform: wp.array(dtype=wp.transform),
    mjc_geom_to_newton_shape: wp.array2d(dtype=wp.int32),
    geom_type: wp.array(dtype=int),
    GEOM_TYPE_MESH: int,
    geom_dataid: wp.array(dtype=int),
    mesh_pos: wp.array(dtype=wp.vec3),
    mesh_quat: wp.array(dtype=wp.quat),
    shape_torsional_friction: wp.array(dtype=float),
    shape_rolling_friction: wp.array(dtype=float),
    shape_geom_solimp: wp.array(dtype=vec5),
    shape_geom_solmix: wp.array(dtype=float),
    shape_geom_gap: wp.array(dtype=float),
    geom_rbound: wp.array2d(dtype=float),
    geom_friction: wp.array2d(dtype=wp.vec3f),
    geom_solref: wp.array2d(dtype=wp.vec2f),
    geom_size: wp.array2d(dtype=wp.vec3f),
    geom_pos: wp.array2d(dtype=wp.vec3f),
    geom_quat: wp.array2d(dtype=wp.quatf),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_solmix: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
):
    """Refresh MuJoCo geom material, pose, and solver buffers from source shapes."""
    world, geom_idx = wp.tid()
    shape_idx = mjc_geom_to_newton_shape[world, geom_idx]
    if shape_idx < 0:
        return

    geom_rbound[world, geom_idx] = shape_collision_radius[shape_idx]
    geom_friction[world, geom_idx] = wp.vec3f(shape_mu[shape_idx], shape_torsional_friction[shape_idx], shape_rolling_friction[shape_idx])
    ke, kd = shape_ke[shape_idx], shape_kd[shape_idx]
    if ke > 0.0 and kd > 0.0:
        # Convert source stiffness/damping into MuJoCo's timeconst/dampratio form.
        timeconst = 2.0 / kd
        dampratio = wp.sqrt(1.0 / (timeconst * timeconst * ke))
        geom_solref[world, geom_idx] = wp.vec2f(timeconst, dampratio)
    else:
        geom_solref[world, geom_idx] = wp.vec2f(0.02, 1.0)
    if shape_geom_solimp:
        geom_solimp[world, geom_idx] = shape_geom_solimp[shape_idx]
    if shape_geom_solmix:
        geom_solmix[world, geom_idx] = shape_geom_solmix[shape_idx]
    if shape_geom_gap:
        geom_gap[world, geom_idx] = shape_geom_gap[shape_idx]
    geom_size[world, geom_idx] = shape_size[shape_idx]
    tf = shape_transform[shape_idx]
    if geom_type[geom_idx] == GEOM_TYPE_MESH:
        # Mesh geoms carry an extra baked-in mesh transform in MuJoCo data.
        mesh_id = geom_dataid[geom_idx]
        mesh_tf = wp.transform(mesh_pos[mesh_id], wp.quat(mesh_quat[mesh_id].y, mesh_quat[mesh_id].z, mesh_quat[mesh_id].w, mesh_quat[mesh_id].x))
        tf = tf * mesh_tf
    geom_pos[world, geom_idx] = tf.p
    geom_quat[world, geom_idx] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)


@wp.kernel
def update_eq_properties_kernel(
    mjc_eq_to_newton_eq: wp.array2d(dtype=wp.int32),
    eq_solref: wp.array(dtype=wp.vec2),
    eq_solref_out: wp.array2d(dtype=wp.vec2),
):
    """Refresh equality-constraint solver parameters from source constraint data."""
    world, mjc_eq = wp.tid()
    source_eq = mjc_eq_to_newton_eq[world, mjc_eq]
    if source_eq < 0:
        return
    if eq_solref:
        eq_solref_out[world, mjc_eq] = eq_solref[source_eq]


__all__ = [
    "update_body_mass_ipos_kernel",
    "update_body_inertia_kernel",
    "update_axis_properties_kernel",
    "update_dof_properties_kernel",
    "update_jnt_properties_kernel",
    "update_mocap_transforms_kernel",
    "update_joint_transforms_kernel",
    "update_shape_mappings_kernel",
    "update_model_properties_kernel",
    "update_geom_properties_kernel",
    "update_eq_properties_kernel",
]
