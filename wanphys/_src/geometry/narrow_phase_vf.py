# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from .utils.geometry_utils import (
    triangle_normal, 
    triangle_barycentric,
    build_orthonormal_basis,
    evaluate_self_contact_force_norm,
    compute_friction,
    damp_collision,
    mat32,
)

from newton._src.geometry.kernels import (
    triangle_closest_point,
)


########################################################################################################################
#################################################    Style3D Kernel   ##################################################
########################################################################################################################

@wp.kernel
def handle_vertex_triangle_contacts_geometry_kernel(
    thickness: float,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int, ndim=2),
    broad_phase_vf: wp.array(dtype=int, ndim=2),
    max_contacts: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_fid: wp.array(dtype=int, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_dist: wp.array(dtype=float, ndim=2),
    contact_bary: wp.array(dtype=wp.vec3, ndim=2),
    contact_point: wp.array(dtype=wp.vec3, ndim=2),
    contact_penetration: wp.array(dtype=float, ndim=2),
):
    vid = wp.tid()

    x0 = pos[vid]  # vertex position
    out_count = wp.int32(0)

    count = wp.min(broad_phase_vf[0, vid], max_contacts)
    for i in range(count):
        fid = broad_phase_vf[i + 1, vid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])
        x1 = pos[face[0]]
        x2 = pos[face[1]]
        x3 = pos[face[2]]
        tri_normal = triangle_normal(x1, x2, x3)
        dist = wp.dot(x0 - x1, tri_normal)
        p = x0 - tri_normal * dist
        bary_coord = triangle_barycentric(x1, x2, x3, p)

        if wp.abs(dist) > thickness:
            continue
        if bary_coord[0] < 0.0 or bary_coord[1] < 0.0 or bary_coord[2] < 0.0:
            continue  # is outside triangle

        contact_fid[out_count, vid] = fid
        contact_normal[out_count, vid] = tri_normal
        contact_dist[out_count, vid] = dist
        contact_bary[out_count, vid] = bary_coord
        contact_point[out_count, vid] = p
        contact_penetration[out_count, vid] = thickness - wp.abs(dist)
        out_count += 1

    contact_count[vid] = out_count

@wp.kernel
def handle_vertex_triangle_contacts_force_kernel(
    thickness: float,
    stiff_factor: float,
    tri_indices: wp.array(dtype=int, ndim=2),
    static_diags: wp.array(dtype=float),
    max_contacts: int,
    contact_count: wp.array(dtype=int),
    contact_fid: wp.array(dtype=int, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_dist: wp.array(dtype=float, ndim=2),
    contact_bary: wp.array(dtype=wp.vec3, ndim=2),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessian_diags: wp.array(dtype=wp.mat33),
):
    vid = wp.tid()

    force0 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    vert_stiff = static_diags[vid]
    is_collided = wp.int32(0)

    count = wp.min(contact_count[vid], max_contacts)
    for i in range(count):
        fid = contact_fid[i, vid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])
        tri_normal = contact_normal[i, vid]
        dist = contact_dist[i, vid]
        bary_coord = contact_bary[i, vid]

        face_stiff = (static_diags[face[0]] + static_diags[face[1]] + static_diags[face[2]]) / 3.0
        stiff = stiff_factor * (vert_stiff * face_stiff) / (vert_stiff + face_stiff)

        force = stiff * tri_normal * (thickness - wp.abs(dist)) * wp.sign(dist)
        hess = stiff * wp.outer(tri_normal, tri_normal)

        force0 += force
        wp.atomic_add(forces, face[0], -force * bary_coord[0])
        wp.atomic_add(forces, face[1], -force * bary_coord[1])
        wp.atomic_add(forces, face[2], -force * bary_coord[2])

        hess0 += hess
        wp.atomic_add(hessian_diags, face[0], hess * bary_coord[0] * bary_coord[0])
        wp.atomic_add(hessian_diags, face[1], hess * bary_coord[1] * bary_coord[1])
        wp.atomic_add(hessian_diags, face[2], hess * bary_coord[2] * bary_coord[2])
        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, vid, force0)
        wp.atomic_add(hessian_diags, vid, hess0)

########################################################################################################################
###################################################    VBD Kernel   ####################################################
########################################################################################################################

@wp.kernel
def build_vertex_triangle_contact_geometry_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    # outputs
    contact_bary: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_dist: wp.array(dtype=float),
):
    tid = wp.tid()
    v = vertex_colliding_triangles[2 * tid]
    tri = vertex_colliding_triangles[2 * tid + 1]

    if v < 0 or tri < 0:
        contact_bary[tid] = wp.vec3(0.0)
        contact_normal[tid] = wp.vec3(0.0)
        contact_dist[tid] = 0.0
        return

    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]
    p = pos[v]

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)
    diff = p - closest_p
    dist = wp.length(diff)

    contact_bary[tid] = bary
    contact_dist[tid] = dist
    contact_normal[tid] = diff / dist if dist > 0.0 else wp.vec3(0.0)


@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices_cached(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    contact_bary: wp.vec3,
    contact_normal: wp.vec3,
    contact_dist: float,
):
    if contact_dist <= 0.0 or contact_dist >= collision_radius:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
        )

    bary = contact_bary
    collision_normal = contact_normal
    dis = contact_dist

    bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

    dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

    collision_force = -dEdD * collision_normal
    collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

    # friction force
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]
    p = pos[v]

    closest_p = bary[0] * a + bary[1] * b + bary[2] * c
    dx_v = p - pos_anchor[v]

    closest_p_prev = (
        bary[0] * pos_anchor[tri_indices[tri, 0]]
        + bary[1] * pos_anchor[tri_indices[tri, 1]]
        + bary[2] * pos_anchor[tri_indices[tri, 2]]
    )

    dx = dx_v - (closest_p - closest_p_prev)

    e0, e1 = build_orthonormal_basis(collision_normal)

    T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

    u = wp.transpose(T) * dx
    eps_U = friction_epsilon * dt

    friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

    displacement_0 = pos_anchor[tri_indices[tri, 0]] - a
    displacement_1 = pos_anchor[tri_indices[tri, 1]] - b
    displacement_2 = pos_anchor[tri_indices[tri, 2]] - c
    displacement_3 = pos_anchor[v] - p

    collision_force_0 = collision_force * bs[0]
    collision_force_1 = collision_force * bs[1]
    collision_force_2 = collision_force * bs[2]
    collision_force_3 = collision_force * bs[3]

    collision_hessian_0 = collision_hessian * bs[0] * bs[0]
    collision_hessian_1 = collision_hessian * bs[1] * bs[1]
    collision_hessian_2 = collision_hessian * bs[2] * bs[2]
    collision_hessian_3 = collision_hessian * bs[3] * bs[3]

    collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
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

    damping_force, damping_hessian = damp_collision(
        displacement_2,
        collision_normal * collision_normal_sign[2],
        collision_hessian_2,
        collision_damping,
        dt,
    )
    collision_force_2 += damping_force + bs[2] * friction_force
    collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

    damping_force, damping_hessian = damp_collision(
        displacement_3,
        collision_normal * collision_normal_sign[3],
        collision_hessian_3,
        collision_damping,
        dt,
    )
    collision_force_3 += damping_force + bs[3] * friction_force
    collision_hessian_3 += damping_hessian + bs[3] * bs[3] * friction_hessian

    return (
        True,
        collision_force_0,
        collision_force_1,
        collision_force_2,
        collision_force_3,
        collision_hessian_0,
        collision_hessian_1,
        collision_hessian_2,
        collision_hessian_3,
    )