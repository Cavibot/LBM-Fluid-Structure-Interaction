# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry.kernels import (
    triangle_closest_point,
    vertex_adjacent_to_triangle,
)

from warp.types import float32, matrix

NUM_THREADS_PER_COLLISION_PRIMITIVE = 4

class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass

@wp.func
def triangle_normal(A: wp.vec3, B: wp.vec3, C: wp.vec3):
    n = wp.cross(B - A, C - A)
    ln = wp.length(n)
    return wp.vec3(0.0) if ln < 1.0e-12 else (n / ln)


@wp.func
def triangle_barycentric(A: wp.vec3, B: wp.vec3, C: wp.vec3, P: wp.vec3):
    v0 = A - C
    v1 = B - C
    v2 = P - C
    dot00 = wp.dot(v0, v0)
    dot01 = wp.dot(v0, v1)
    dot02 = wp.dot(v0, v2)
    dot11 = wp.dot(v1, v1)
    dot12 = wp.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    invDenom = 0.0 if wp.abs(denom) < 1.0e-12 else 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return wp.vec3(u, v, 1.0 - u - v)


@wp.func
def compute_projected_isotropic_friction(
    friction_mu: float,
    normal_load: float,
    n_hat: wp.vec3,
    slip_u: wp.vec3,
    eps_u: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Isotropic Coulomb friction in world frame using projector P = I - n n^T.

    Regularization: if ||u_t|| <= eps_u, uses a linear ramp; otherwise 1/||u_t||.

    Args:
        friction_mu: Coulomb friction coefficient (>= 0).
        normal_load: Normal load magnitude (>= 0).
        n_hat: Unit contact normal (world frame).
        slip_u: Tangential slip displacement over dt (world frame).
        eps_u: Smoothing distance (same units as slip_u, > 0).

    Returns:
        tuple[wp.vec3, wp.mat33]: (force, Hessian) in world frame.
    """
    # Tangential slip in the contact tangent plane without forming P: u_t = u - n * (n dot u)
    dot_nu = wp.dot(n_hat, slip_u)
    u_t = slip_u - n_hat * dot_nu
    u_norm = wp.length(u_t)

    if u_norm > 0.0:
        # IPC-style regularization
        if u_norm > eps_u:
            f1_SF_over_x = 1.0 / u_norm
        else:
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        # Factor common scalar; force aligned with u_t, Hessian proportional to projector
        scale = friction_mu * normal_load * f1_SF_over_x
        f = -(scale * u_t)
        K = scale * (wp.identity(3, float) - wp.outer(n_hat, n_hat))
    else:
        f = wp.vec3(0.0)
        K = wp.mat33(0.0)

    return f, K


@wp.func
def intersection_gradient_vector(R: wp.vec3, E: wp.vec3, N: wp.vec3):
    """
    Reference: Resolving Surface Collisions through Intersection Contour Minimization, Pascal Volino & Magnenat-Thalmann, 2006.

    Args:
        R: The direction of the intersection segment
        E: Direction vector of the edge
        N: The normals of the polygons
    """
    dot_EN = wp.dot(E, N)
    if wp.abs(dot_EN) > 1e-6:
        return R - 2.0 * N * wp.dot(E, R) / dot_EN
    else:
        return R
    
# style3d BVH uses
@wp.func
def line_intersects_aabb(v0: wp.vec3, v1: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    # Slab method
    dir = v1 - v0
    tmin = 0.0
    tmax = 1.0

    for i in range(3):
        if wp.abs(dir[i]) < 1.0e-8:
            # Segment is parallel to slab. Reject if origin not within slab
            if v0[i] < lower[i] or v0[i] > upper[i]:
                return False
        else:
            invD = 1.0 / dir[i]
            t1 = (lower[i] - v0[i]) * invD
            t2 = (upper[i] - v0[i]) * invD

            tmin = wp.max(tmin, wp.min(t1, t2))
            tmax = wp.min(tmax, wp.max(t1, t2))
            if tmax < tmin:
                return False

    return True


@wp.kernel
def compute_tri_aabbs_kernel(
    enlarge: float,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    v1 = pos[tri_indices[t_id, 0]]
    v2 = pos[tri_indices[t_id, 1]]
    v3 = pos[tri_indices[t_id, 2]]

    lower = wp.min(wp.min(v1, v2), v3)
    upper = wp.max(wp.max(v1, v2), v3)

    lower_bounds[t_id] = lower - wp.vec3(enlarge)
    upper_bounds[t_id] = upper + wp.vec3(enlarge)


@wp.kernel
def compute_edge_aabbs_kernel(
    enlarge: float,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    e_id = wp.tid()

    v1 = pos[edge_indices[e_id, 2]]
    v2 = pos[edge_indices[e_id, 3]]

    lower_bounds[e_id] = wp.min(v1, v2) - wp.vec3(enlarge)
    upper_bounds[e_id] = wp.max(v1, v2) + wp.vec3(enlarge)


@wp.kernel
def aabb_vs_aabb_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    query_radius: float,
    ignore_self_hits: bool,
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    tid = wp.int32(wp.tid())
    lower = lower_bounds[tid] - wp.vec3(query_radius)
    upper = upper_bounds[tid] + wp.vec3(query_radius)

    query_count = wp.int32(0)
    query_index = wp.int32(-1)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, query_index):
        if not (ignore_self_hits and query_index <= tid):
            query_results[query_count + 1, tid] = query_index
            query_count += 1

    query_results[0, tid] = query_count


@wp.kernel
def aabb_vs_line_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    ignore_self_hits: bool,
    vertices: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    eid = wp.int32(wp.tid())
    v1 = vertices[edge_indices[eid, 2]]
    v2 = vertices[edge_indices[eid, 3]]

    query_count = wp.int32(0)
    query_index = wp.int32(-1)
    query = wp.bvh_query_ray(bvh_id, v1, v2 - v1)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, query_index):
        if not (ignore_self_hits and query_index <= eid):
            if line_intersects_aabb(v1, v2, lower_bounds[query_index], upper_bounds[query_index]):
                query_results[query_count + 1, eid] = query_index
                query_count += 1

    query_results[0, eid] = query_count


@wp.kernel
def triangle_vs_point_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    query_radius: float,
    max_dist: float,
    ignore_self_hits: bool,
    pos: wp.array(dtype=wp.vec3),
    tri_pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int, ndim=2),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    vid = wp.tid()

    x0 = pos[vid]
    lower = x0 - wp.vec3(query_radius)
    upper = x0 + wp.vec3(query_radius)

    tri_index = wp.int32(-1)
    query_count = wp.int32(0)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if ignore_self_hits and vertex_adjacent_to_triangle(vid, t1, t2, t3):
            continue

        closest_p, _bary, _feature_type = triangle_closest_point(tri_pos[t1], tri_pos[t2], tri_pos[t3], x0)

        dist = wp.length(closest_p - x0)

        if dist < max_dist:
            query_results[query_count + 1, vid] = tri_index
            query_count += 1

    query_results[0, vid] = query_count


@wp.kernel
def edge_vs_edge_kernel(
    bvh_id: wp.uint64,
    query_list_rows: int,
    query_radius: float,
    max_dist: float,
    ignore_self_hits: bool,
    test_pos: wp.array(dtype=wp.vec3),
    test_edge_indices: wp.array(dtype=int, ndim=2),
    edge_pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=int, ndim=2),
    # outputs
    query_results: wp.array(dtype=int, ndim=2),
):
    eid = wp.int32(wp.tid())

    v0 = test_edge_indices[eid, 2]
    v1 = test_edge_indices[eid, 3]

    x0 = test_pos[v0]
    x1 = test_pos[v1]

    lower = wp.min(x0, x1) - wp.vec3(query_radius)
    upper = wp.max(x0, x1) + wp.vec3(query_radius)

    edge_index = wp.int32(-1)
    query_count = wp.int32(0)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    while (query_count < query_list_rows - 1) and wp.bvh_query_next(query, edge_index):
        if ignore_self_hits and edge_index <= eid:
            continue
        v2 = edge_indices[edge_index, 2]
        v3 = edge_indices[edge_index, 3]
        if ignore_self_hits and (v0 == v2 or v0 == v3 or v1 == v2 or v1 == v3):
            continue

        x2, x3 = edge_pos[v2], edge_pos[v3]
        edge_edge_parallel_epsilon = wp.float32(1e-5)
        st = wp.closest_point_edge_edge(x0, x1, x2, x3, edge_edge_parallel_epsilon)
        s = st[0]
        t = st[1]
        c1 = wp.lerp(x0, x1, s)
        c2 = wp.lerp(x2, x3, t)
        dist = wp.length(c1 - c2)

        if dist < max_dist:
            query_results[query_count + 1, eid] = edge_index
            query_count += 1

    query_results[0, eid] = query_count


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    body_particle_contact_ke: float,
    body_particle_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    """
    Evaluate particle-rigid body contact force and Hessian (on particle side).

    Computes contact forces and Hessians for a particle interacting with a rigid body shape.
    The function is agnostic to whether the rigid body is static, kinematic, or dynamic.

    Contact model:
    - Normal: Linear spring-damper (stiffness: body_particle_contact_ke, damping: body_particle_contact_kd)
    - Friction: 3D projector-based Coulomb friction with IPC regularization
    - Normal direction: Points from rigid surface towards particle (into particle)

    Args:
        particle_index: Index of the particle
        particle_pos: Current particle position (world frame)
        particle_prev_pos: Previous particle position (world frame) used as the
            "previous" position for finite-difference contact-relative velocity.
        contact_index: Index in the body-particle contact arrays
        body_particle_contact_ke: Contact stiffness (model-level or AVBD adaptive)
        body_particle_contact_kd: Contact damping (model-level or AVBD averaged)
        friction_mu: Friction coefficient (model-level or AVBD averaged)
        friction_epsilon: Friction regularization distance
        particle_radius: Array of particle radii
        shape_material_mu: Array of shape friction coefficients
        shape_body: Array mapping shape index to body index
        body_q: Current body transforms
        body_q_prev: Previous body transforms (for finite-difference body
            velocity when available)
        body_qd: Body spatial velocities (fallback when no previous pose is provided)
        body_com: Body centers of mass (local frame)
        contact_shape: Array of shape indices for each soft contact
        contact_body_pos: Array of contact points (local to shape)
        contact_body_vel: Array of contact velocities (local frame)
        contact_normal: Array of contact normals (world frame, from rigid to particle)
        dt: Time window [s] used for finite-difference damping/friction.

    Returns:
        tuple[wp.vec3, wp.mat33]: (force, Hessian) on the particle (world frame)
    """
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0.0:
        body_contact_force_norm = penetration_depth * body_particle_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = body_particle_contact_ke * wp.outer(n, n)

        # Use the larger of body-particle friction and shape material friction
        mu = wp.max(friction_mu, shape_material_mu[shape_index])

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0.0:
            # Damping coefficient is scaled by contact stiffness (consistent with rigid-rigid)
            damping_coeff = body_particle_contact_kd * body_particle_contact_ke
            damping_hessian = (damping_coeff / dt) * wp.outer(n, n)
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_prev:
            # if body_q_prev is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        else:
            # otherwise use the instantaneous velocity
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]

            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)

            # compute the body velocity at the particle position
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # Friction using 3D projector approach (consistent with rigid-rigid contacts)
        eps_u = friction_epsilon * dt
        friction_force, friction_hessian = compute_projected_isotropic_friction(
            mu, body_contact_force_norm, n, relative_translation, eps_u
        )
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian


@wp.kernel
def eval_body_contact_kernel(
    # inputs
    dt: float,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    forces: wp.array(dtype=wp.vec3),
    hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = wp.min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]
        body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
            particle_idx,
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
            shape_material_mu,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
        )
        wp.atomic_add(forces, particle_idx, body_contact_force)
        wp.atomic_add(hessians, particle_idx, body_contact_hessian)


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > 1e-5:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that is perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the 1D friction force and hessian.
    Args:
        mu: Friction coefficient.
        normal_contact_force: normal contact force.
        T: Transformation matrix (3x2 matrix).
        u: 2D displacement vector.
    """
    # Friction
    u_norm = wp.length(u)

    if u_norm > 0.0:
        # IPC friction
        if u_norm > eps_u:
            # constant stage
            f1_SF_over_x = 1.0 / u_norm
        else:
            # smooth transition
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)

        # Different from IPC, we treat the contact normal as constant
        # this significantly improves the stability
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * wp.identity(2, float)) * wp.transpose(T)
    else:
        force = wp.vec3(0.0, 0.0, 0.0)
        hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return force, hessian


@wp.func
def damp_collision(
    displacement: wp.vec3,
    collision_normal: wp.vec3,
    collision_hessian: wp.mat33,
    collision_damping: float,
    dt: float,
):
    if wp.dot(displacement, collision_normal) > 0:
        damping_hessian = (collision_damping / dt) * collision_hessian
        damping_force = damping_hessian * displacement
        return damping_force, damping_hessian
    else:
        return wp.vec3(0.0), wp.mat33(0.0)
    

@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    body_particle_contact_ke: float,
    body_particle_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    """
    Evaluate particle-rigid body contact force and Hessian (on particle side).

    Computes contact forces and Hessians for a particle interacting with a rigid body shape.
    The function is agnostic to whether the rigid body is static, kinematic, or dynamic.

    Contact model:
    - Normal: Linear spring-damper (stiffness: body_particle_contact_ke, damping: body_particle_contact_kd)
    - Friction: 3D projector-based Coulomb friction with IPC regularization
    - Normal direction: Points from rigid surface towards particle (into particle)

    Args:
        particle_index: Index of the particle
        particle_pos: Current particle position (world frame)
        particle_prev_pos: Previous particle position (world frame) used as the
            "previous" position for finite-difference contact-relative velocity.
        contact_index: Index in the body-particle contact arrays
        body_particle_contact_ke: Contact stiffness (model-level or AVBD adaptive)
        body_particle_contact_kd: Contact damping (model-level or AVBD averaged)
        friction_mu: Friction coefficient (model-level or AVBD averaged)
        friction_epsilon: Friction regularization distance
        particle_radius: Array of particle radii
        shape_material_mu: Array of shape friction coefficients
        shape_body: Array mapping shape index to body index
        body_q: Current body transforms
        body_q_prev: Previous body transforms (for finite-difference body
            velocity when available)
        body_qd: Body spatial velocities (fallback when no previous pose is provided)
        body_com: Body centers of mass (local frame)
        contact_shape: Array of shape indices for each soft contact
        contact_body_pos: Array of contact points (local to shape)
        contact_body_vel: Array of contact velocities (local frame)
        contact_normal: Array of contact normals (world frame, from rigid to particle)
        dt: Time window [s] used for finite-difference damping/friction.

    Returns:
        tuple[wp.vec3, wp.mat33]: (force, Hessian) on the particle (world frame)
    """
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0.0:
        body_contact_force_norm = penetration_depth * body_particle_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = body_particle_contact_ke * wp.outer(n, n)

        # Use the larger of body-particle friction and shape material friction
        mu = wp.max(friction_mu, shape_material_mu[shape_index])

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0.0:
            # Damping coefficient is scaled by contact stiffness (consistent with rigid-rigid)
            damping_coeff = body_particle_contact_kd * body_particle_contact_ke
            damping_hessian = (damping_coeff / dt) * wp.outer(n, n)
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_prev:
            # if body_q_prev is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        else:
            # otherwise use the instantaneous velocity
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]

            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)

            # compute the body velocity at the particle position
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # Friction using 3D projector approach (consistent with rigid-rigid contacts)
        eps_u = friction_epsilon * dt
        friction_force, friction_hessian = compute_projected_isotropic_friction(
            mu, body_contact_force_norm, n, relative_translation, eps_u
        )
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian
