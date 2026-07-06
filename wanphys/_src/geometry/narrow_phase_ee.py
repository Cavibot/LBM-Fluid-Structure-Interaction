# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from .utils.geometry_utils import (
    build_orthonormal_basis,
    evaluate_self_contact_force_norm,
    compute_friction,
    damp_collision,
    mat32,
)

########################################################################################################################
#################################################    Style3D Kernel   ##################################################
########################################################################################################################


@wp.kernel
def handle_edge_edge_contacts_geometry_kernel(
    thickness: float,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=int, ndim=2),
    broad_phase_ee: wp.array(dtype=int, ndim=2),
    max_contacts: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_eid: wp.array(dtype=int, ndim=2),
    contact_s: wp.array(dtype=float, ndim=2),
    contact_t: wp.array(dtype=float, ndim=2),
    contact_dir: wp.array(dtype=wp.vec3, ndim=2),
    contact_dist: wp.array(dtype=float, ndim=2),
    contact_limit: wp.array(dtype=float, ndim=2),
    contact_point: wp.array(dtype=wp.vec3, ndim=2),
    contact_penetration: wp.array(dtype=float, ndim=2),
):
    eid = wp.tid()
    edge0 = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    x0 = pos[edge0[0]]
    x1 = pos[edge0[1]]
    len0 = wp.length(x0 - x1)

    out_count = wp.int32(0)

    count = wp.min(broad_phase_ee[0, eid], max_contacts)
    for i in range(count):
        idx = broad_phase_ee[i + 1, eid]
        edge1 = wp.vec4i(edge_indices[idx, 2], edge_indices[idx, 3], edge_indices[idx, 0], edge_indices[idx, 1])
        x2, x3 = pos[edge1[0]], pos[edge1[1]]
        edge_edge_parallel_epsilon = wp.float32(1e-5)

        st = wp.closest_point_edge_edge(x0, x1, x2, x3, edge_edge_parallel_epsilon)
        s, t = st[0], st[1]

        if (s <= 0) or (s >= 1) or (t <= 0) or (t >= 1):
            continue

        c1 = wp.lerp(x0, x1, s)
        c2 = wp.lerp(x2, x3, t)
        dir = c1 - c2
        dist = wp.length(dir)
        limited_thickness = thickness

        len1 = wp.length(x2 - x3)
        avg_len = (len0 + len1) * 0.5
        if edge0[2] == edge1[0] or edge0[3] == edge1[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge0[2] == edge1[1] or edge0[3] == edge1[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        if edge1[2] == edge0[0] or edge1[3] == edge0[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge1[2] == edge0[1] or edge1[3] == edge0[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)

        if 1e-6 < dist < limited_thickness:
            contact_eid[out_count, eid] = idx
            contact_s[out_count, eid] = s
            contact_t[out_count, eid] = t
            contact_dir[out_count, eid] = wp.normalize(dir)
            contact_dist[out_count, eid] = dist
            contact_limit[out_count, eid] = limited_thickness
            contact_point[out_count, eid] = (c1 + c2) * 0.5
            contact_penetration[out_count, eid] = limited_thickness - dist
            out_count += 1

    contact_count[eid] = out_count


@wp.kernel
def handle_edge_edge_contacts_force_kernel(
    stiff_factor: float,
    edge_indices: wp.array(dtype=int, ndim=2),
    static_diags: wp.array(dtype=float),
    max_contacts: int,
    contact_count: wp.array(dtype=int),
    contact_eid: wp.array(dtype=int, ndim=2),
    contact_s: wp.array(dtype=float, ndim=2),
    contact_t: wp.array(dtype=float, ndim=2),
    contact_dir: wp.array(dtype=wp.vec3, ndim=2),
    contact_dist: wp.array(dtype=float, ndim=2),
    contact_limit: wp.array(dtype=float, ndim=2),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessian_diags: wp.array(dtype=wp.mat33),
):
    eid = wp.tid()
    edge0 = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge0[0]] + static_diags[edge0[1]]) / 2.0
    is_collided = wp.int32(0)

    count = wp.min(contact_count[eid], max_contacts)
    for i in range(count):
        idx = contact_eid[i, eid]
        edge1 = wp.vec4i(edge_indices[idx, 2], edge_indices[idx, 3], edge_indices[idx, 0], edge_indices[idx, 1])
        s = contact_s[i, eid]
        t = contact_t[i, eid]
        dir = contact_dir[i, eid]
        dist = contact_dist[i, eid]
        limited_thickness = contact_limit[i, eid]

        stiff_1 = (static_diags[edge1[0]] + static_diags[edge1[1]]) / 2.0
        stiff = stiff_factor * (stiff_0 * stiff_1) / (stiff_0 + stiff_1)

        force = stiff * dir * (limited_thickness - dist)
        hess = stiff * wp.outer(dir, dir)

        force0 += force * (1.0 - s)
        force1 += force * s
        wp.atomic_add(forces, edge1[0], -force * (1.0 - t))
        wp.atomic_add(forces, edge1[1], -force * t)

        hess0 += hess * (1.0 - s) * (1.0 - s)
        hess1 += hess * s * s
        wp.atomic_add(hessian_diags, edge1[0], hess * (1.0 - t) * (1.0 - t))
        wp.atomic_add(hessian_diags, edge1[1], hess * t * t)
        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, edge0[0], force0)
        wp.atomic_add(forces, edge0[1], force1)
        wp.atomic_add(hessian_diags, edge0[0], hess0)
        wp.atomic_add(hessian_diags, edge0[1], hess1)


########################################################################################################################
###################################################    VBD Kernel   ####################################################
########################################################################################################################

@wp.kernel
def build_edge_edge_contact_geometry_kernel(
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_colliding_edges: wp.array(dtype=wp.int32),
    edge_edge_parallel_epsilon: float,
    # outputs
    contact_s: wp.array(dtype=float),
    contact_t: wp.array(dtype=float),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_dist: wp.array(dtype=float),
):
    tid = wp.tid()
    e1 = edge_colliding_edges[2 * tid]
    e2 = edge_colliding_edges[2 * tid + 1]

    if e1 < 0 or e2 < 0:
        contact_s[tid] = 0.0
        contact_t[tid] = 0.0
        contact_normal[tid] = wp.vec3(0.0)
        contact_dist[tid] = 0.0
        return

    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]
    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]
    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    dist = st[2]

    c1 = e1_v1_pos + (e1_v2_pos - e1_v1_pos) * s
    c2 = e2_v1_pos + (e2_v2_pos - e2_v1_pos) * t
    diff = c1 - c2

    contact_s[tid] = s
    contact_t[tid] = t
    contact_dist[tid] = dist
    contact_normal[tid] = diff / dist if dist > 0.0 else wp.vec3(0.0)


@wp.func
def evaluate_edge_edge_contact_2_vertices_cached(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    contact_s: float,
    contact_t: float,
    contact_normal: wp.vec3,
    contact_dist: float,
):
    if contact_dist <= 0.0 or contact_dist >= collision_radius:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return False, collision_force, collision_force, collision_hessian, collision_hessian

    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]
    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]
    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    s = contact_s
    t = contact_t
    collision_normal = contact_normal
    dis = contact_dist

    bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

    dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

    collision_force = -dEdD * collision_normal
    collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

    c1 = e1_v1_pos + (e1_v2_pos - e1_v1_pos) * s
    c2 = e2_v1_pos + (e2_v2_pos - e2_v1_pos) * t

    # friction
    c1_prev = pos_anchor[e1_v1] + (pos_anchor[e1_v2] - pos_anchor[e1_v1]) * s
    c2_prev = pos_anchor[e2_v1] + (pos_anchor[e2_v2] - pos_anchor[e2_v1]) * t

    dx = (c1 - c1_prev) - (c2 - c2_prev)
    axis_1, axis_2 = build_orthonormal_basis(collision_normal)

    T = mat32(
        axis_1[0],
        axis_2[0],
        axis_1[1],
        axis_2[1],
        axis_1[2],
        axis_2[2],
    )

    u = wp.transpose(T) * dx
    eps_U = friction_epsilon * dt

    friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

    displacement_0 = pos_anchor[e1_v1] - e1_v1_pos
    displacement_1 = pos_anchor[e1_v2] - e1_v2_pos

    collision_force_0 = collision_force * bs[0]
    collision_force_1 = collision_force * bs[1]

    collision_hessian_0 = collision_hessian * bs[0] * bs[0]
    collision_hessian_1 = collision_hessian * bs[1] * bs[1]

    collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
    damping_force, damping_hessian = damp_collision(
        displacement_0,
        collision_normal * collision_normal_sign[0],
        collision_hessian_0,
        collision_damping,
        dt,
    )

    collision_force_0 += damping_force + bs[0] * friction_force
    collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

    damping_force, damping_hessian = damp_collision(
        displacement_1,
        collision_normal * collision_normal_sign[1],
        collision_hessian_1,
        collision_damping,
        dt,
    )
    collision_force_1 += damping_force + bs[1] * friction_force
    collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

    return True, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1
