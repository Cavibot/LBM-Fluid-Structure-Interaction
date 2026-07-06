# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact / collision response kernels.

Five contact interaction types are handled:

* **Particle–particle proximity** – hash-grid spatial query with penalty
  normal force + Coulomb tangential friction.
* **Mesh–particle** – triangle-to-particle penetration with barycentric
  force distribution.
* **Particle–shape** – soft contacts between particles and rigid shapes.
* **Rigid–rigid** – penalty-based body contact with per-contact material
  overrides and Huber-norm smoothed friction.
* **Mesh–body** – triangle mesh vs. rigid body (utility; not in default pipeline).

Physics
-------
All contact responses use a penalty (regularised) model:

    F_normal = k_n * gap + k_d * min(v_n, 0)   (gap < 0 = penetration)
    F_friction = min(k_f * |v_t|, mu * |F_n|) * v_t_hat   (Coulomb cone)

Rigid–rigid friction uses the Huber norm for smooth differentiability near
zero sliding velocity.
"""

from __future__ import annotations

import warp as wp

from ..common import ParticleFlag, triangle_closest_barycentric


# =====================================================================
# Shared contact response helper
# =====================================================================


@wp.func
def penalty_contact_response(
    normal: wp.vec3,
    rel_vel: wp.vec3,
    gap: float,
    k_n: float,
    k_d: float,
    k_f: float,
    k_mu: float,
) -> wp.vec3:
    """Penalty normal force + Coulomb friction for one contact pair.

    normal  – contact normal pointing from b toward a (unit vector)
    rel_vel – relative velocity of a w.r.t. b at the contact point
    gap     – signed separation (negative = penetrating)

    Normal force:   F_n = k_n * gap + k_d * min(v_n, 0)
    Friction force: F_f = min(k_f * |v_t|, mu * |F_n|) * v_t_hat
    """
    v_n = wp.dot(normal, rel_vel)                  # normal component of relative velocity
    F_n = gap * k_n + wp.min(v_n, 0.0) * k_d      # penalty + damping

    v_t = rel_vel - normal * v_n                   # tangential relative velocity
    v_t_speed = wp.length(v_t)

    v_t_hat = wp.vec3(0.0)
    if v_t_speed > 0.0:
        v_t_hat = v_t / v_t_speed

    F_f = wp.min(v_t_speed * k_f, k_mu * wp.abs(F_n))

    return -normal * F_n - v_t_hat * F_f


# =====================================================================
# 1. Particle–particle proximity
# =====================================================================


@wp.kernel
def particle_proximity_kernel(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    flags: wp.array(dtype=wp.int32),
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    # output
    pforce: wp.array(dtype=wp.vec3),
):
    """Particle–particle contact via spatial hash grid.

    Each thread handles one particle; neighbours within (r_i + r_max + cohesion)
    are queried from the hash grid.  Contact is active when the gap is within
    the cohesion distance (gap <= k_cohesion, where gap < 0 = penetrating).
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (flags[i] & ParticleFlag.ACTIVE) == 0:
        return

    xi = pos[i]
    vi = vel[i]
    ri = radius[i]

    f = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, ri + max_radius + k_cohesion)
    j = int(0)

    while wp.hash_grid_query_next(query, j):
        if (flags[j] & ParticleFlag.ACTIVE) != 0 and j != i:
            sep  = xi - pos[j]
            dist = wp.length(sep)
            gap  = dist - ri - radius[j]   # positive = separated, negative = overlapping

            if gap <= k_cohesion:
                normal  = sep / dist
                rel_vel = vi - vel[j]
                f += penalty_contact_response(normal, rel_vel, gap,
                                              k_contact, k_damp, k_friction, k_mu)

    pforce[i] = f


# =====================================================================
# 2. Mesh–particle contact
# =====================================================================


@wp.kernel
def mesh_particle_contact_kernel(
    num_particles: int,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    tri_elem: wp.array2d(dtype=int),
    tri_mat: wp.array2d(dtype=float),
    radius: wp.array(dtype=float),
    contact_stiffness: float,
    frc: wp.array(dtype=wp.vec3),
):
    """Triangle-to-particle penetration contact.

    One thread per (face, particle) pair.  The closest point on the triangle
    is found via barycentric projection; if the particle centre is within its
    collision radius, a normal penalty force is applied and distributed to the
    three triangle nodes by barycentric weights.
    """
    tid = wp.tid()
    face = tid // num_particles
    part = tid  % num_particles

    n0 = tri_elem[face, 0]
    n1 = tri_elem[face, 1]
    n2 = tri_elem[face, 2]

    # skip if particle is a triangle vertex (self-collision)
    if n0 == part or n1 == part or n2 == part:
        return

    pt = pos[part]
    pa = pos[n0];  pb = pos[n1];  pc = pos[n2]

    bary    = triangle_closest_barycentric(pa, pb, pc, pt)
    closest = pa * bary[0] + pb * bary[1] + pc * bary[2]

    diff = pt - closest
    dist = wp.length(diff)

    r_col = radius[part]
    if dist >= r_col or dist < 1.0e-6:
        return

    normal      = diff / dist
    penetration = r_col - dist                     # positive = overlapping

    # normal penalty force on particle; reaction distributed to triangle nodes
    fn = normal * (contact_stiffness * penetration)

    wp.atomic_add(frc, part, fn)
    wp.atomic_add(frc, n0, -fn * bary[0])
    wp.atomic_add(frc, n1, -fn * bary[1])
    wp.atomic_add(frc, n2, -fn * bary[2])


# =====================================================================
# 3. Particle–shape (soft) contact
# =====================================================================


@wp.kernel
def particle_shape_contact_kernel(
    part_pos: wp.array(dtype=wp.vec3),
    part_vel: wp.array(dtype=wp.vec3),
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    part_radius: wp.array(dtype=float),
    part_flags: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    shape_ke: wp.array(dtype=float),
    shape_kd: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_ka: wp.array(dtype=float),
    soft_ke: float,
    soft_kd: float,
    soft_kf: float,
    soft_mu: float,
    soft_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    wrench_in_world: bool,
    # outputs
    pforce: wp.array(dtype=wp.vec3),
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Particle-to-rigid-shape soft contact.

    Material parameters are averaged between the particle's global soft
    contact settings and the per-shape material.  The contact is active
    when the signed gap (particle surface to shape surface) is within the
    adhesion distance ``soft_ka``.

    Moment arm convention:
      wrench_in_world=True  → torque = r_world × f  (world-frame arm)
      wrench_in_world=False → torque = r_com × f    (CoM-relative arm)
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    si = contact_shape[tid]
    bi = shape_body[si]
    pi = contact_particle[tid]

    if (part_flags[pi] & ParticleFlag.ACTIVE) == 0:
        return

    px = part_pos[pi]
    pv = part_vel[pi]

    # body transform and velocity (identity if no body)
    X_wb  = wp.transform_identity()
    X_com = wp.vec3()
    bv_s  = wp.spatial_vector()

    if bi >= 0:
        X_wb  = body_tf[bi]
        X_com = body_com[bi]
        bv_s  = body_vel[bi]

    # contact point in world space and moment arm from CoM
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r  = bx - wp.transform_point(X_wb, X_com)

    n   = contact_normal[tid]
    gap = wp.dot(n, px - bx) - part_radius[pi]

    if gap > soft_ka:
        return

    # average material properties
    ke = 0.5 * (soft_ke + shape_ke[si])
    kd = 0.5 * (soft_kd + shape_kd[si])
    kf = 0.5 * (soft_kf + shape_kf[si])
    mu = 0.5 * (soft_mu + shape_mu[si])

    # body velocity at contact point
    omega_b = wp.spatial_bottom(bv_s)
    linv_b  = wp.spatial_top(bv_s)

    bv_pt = linv_b + wp.transform_vector(X_wb, contact_body_vel[tid])
    if wrench_in_world:
        bv_pt += wp.cross(omega_b, bx)
    else:
        bv_pt += wp.cross(omega_b, r)

    rel_v = pv - bv_pt
    v_n   = wp.dot(n, rel_v)
    v_t   = rel_v - n * v_n

    # normal penalty + damping + friction
    fn      = n * gap * ke
    fd      = n * wp.min(v_n, 0.0) * kd
    ft      = wp.normalize(v_t) * wp.min(kf * wp.length(v_t), wp.abs(mu * gap * ke))
    f_total = fn + fd + ft

    wp.atomic_sub(pforce, pi, f_total)

    if bi >= 0:
        if wrench_in_world:
            wp.atomic_sub(body_wrench, bi, wp.spatial_vector(f_total, wp.cross(bx, f_total)))
        else:
            wp.atomic_add(body_wrench, bi, wp.spatial_vector(f_total, wp.cross(r,  f_total)))


# =====================================================================
# 4. Mesh–body contact (utility, not in default pipeline)
# =====================================================================


@wp.kernel
def mesh_body_contact_kernel(
    num_particles: int,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int),
    body_x: wp.array(dtype=wp.vec3),
    body_r: wp.array(dtype=wp.quat),
    body_v: wp.array(dtype=wp.vec3),
    body_w: wp.array(dtype=wp.vec3),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_dist: wp.array(dtype=float),
    contact_mat: wp.array(dtype=int),
    materials: wp.array(dtype=float),
    tri_frc: wp.array(dtype=wp.vec3),
):
    """Triangle mesh vs rigid body contact (not used in default pipeline).

    One thread per (face, body-contact-point) pair.  The rigid body's
    contact point is projected onto the triangle; a penalty force is
    applied to the triangle nodes weighted by barycentric coordinates.
    """
    tid = wp.tid()

    face = tid // num_particles
    part = tid  % num_particles

    cb = contact_body[part]
    cp = contact_point[part]
    cd = contact_dist[part]
    cm = contact_mat[part]

    ke = materials[cm * 4 + 0]
    kd = materials[cm * 4 + 1]
    kf = materials[cm * 4 + 2]
    mu = materials[cm * 4 + 3]

    # world-space contact point on the rigid body (with offset along arm)
    bx  = body_x[cb]
    br  = body_r[cb]
    arm = wp.quat_rotate(br, cp)
    world_pt = bx + arm + wp.normalize(arm) * cd

    # velocity of the rigid body at the contact point
    body_vel_at_pt = body_v[cb] + wp.cross(body_w[cb], arm)

    n0 = tri_indices[face * 3 + 0]
    n1 = tri_indices[face * 3 + 1]
    n2 = tri_indices[face * 3 + 2]

    pa = pos[n0];  pb = pos[n1];  pc = pos[n2]
    va = vel[n0];  vb = vel[n1];  vc = vel[n2]

    bary    = triangle_closest_barycentric(pa, pb, pc, world_pt)
    closest = pa * bary[0] + pb * bary[1] + pc * bary[2]

    diff   = world_pt - closest
    dist2  = wp.dot(diff, diff)
    normal = wp.normalize(diff)

    # gap: negative of squared distance offset (legacy formulation)
    gap = wp.min(dist2 - 0.05, 0.0)

    fn = gap * ke

    # relative velocity of triangle surface w.r.t. body contact point
    v_tri = va * bary[0] + vb * bary[1] + vc * bary[2]
    rel_v = v_tri - body_vel_at_pt

    v_n = wp.dot(normal, rel_v)
    v_t = rel_v - normal * v_n

    fd = -wp.max(v_n, 0.0) * kd * wp.step(gap)

    # Coulomb friction via axis decomposition
    lower = mu * (fn + fd)
    upper = -lower

    nx = wp.cross(normal, wp.vec3(0.0, 0.0, 1.0))
    nz = wp.cross(normal, wp.vec3(1.0, 0.0, 0.0))

    vx = wp.clamp(wp.dot(nx * kf, v_t), lower, upper)
    vz = wp.clamp(wp.dot(nz * kf, v_t), lower, upper)

    ft = (nx * vx + nz * vz) * (-wp.step(gap))

    f_total = normal * (fn + fd) + ft

    wp.atomic_add(tri_frc, n0, f_total * bary[0])
    wp.atomic_add(tri_frc, n1, f_total * bary[1])
    wp.atomic_add(tri_frc, n2, f_total * bary[2])


# =====================================================================
# 5. Rigid–rigid penalty contact
# =====================================================================


@wp.kernel
def rigid_penalty_contact_kernel(
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_ke: wp.array(dtype=float),
    shape_kd: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_ka: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_pt_a: wp.array(dtype=wp.vec3),
    contact_pt_b: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape_a: wp.array(dtype=int),
    contact_shape_b: wp.array(dtype=int),
    contact_thick_a: wp.array(dtype=float),
    contact_thick_b: wp.array(dtype=float),
    rigid_stiffness: wp.array(dtype=float),
    rigid_damping: wp.array(dtype=float),
    rigid_friction_scale: wp.array(dtype=float),
    wrench_in_world: bool,
    friction_smoothing: float,
    # output
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Rigid body–body penalty contact with Huber-norm friction.

    Material parameters are averaged across the two shapes.  Per-contact
    overrides (rigid_stiffness / rigid_damping / rigid_friction_scale) take
    precedence when positive.

    Friction uses the Huber norm  ||v_t||_huber  for smooth gradients near
    zero sliding velocity, controlled by ``friction_smoothing``.

    Gap convention: gap = n · (pt_a - pt_b), positive = separated.
    Contact is active when gap < adhesion distance ``ka``.
    """
    tid = wp.tid()

    if tid >= contact_count[0]:
        return

    sa = contact_shape_a[tid]
    sb = contact_shape_b[tid]
    if sa == sb:
        return

    # --- average material properties across the two shapes ---
    ke = 0.0;  kd = 0.0;  kf = 0.0;  ka = 0.0;  mu = 0.0
    n_mat = 0
    ba = -1;  bb = -1

    if sa >= 0:
        ke += shape_ke[sa];  kd += shape_kd[sa]
        kf += shape_kf[sa];  ka += shape_ka[sa];  mu += shape_mu[sa]
        ba = shape_body[sa];  n_mat += 1

    if sb >= 0:
        ke += shape_ke[sb];  kd += shape_kd[sb]
        kf += shape_kf[sb];  ka += shape_ka[sb];  mu += shape_mu[sb]
        bb = shape_body[sb];  n_mat += 1

    if n_mat > 0:
        inv_n = 1.0 / float(n_mat)
        ke *= inv_n;  kd *= inv_n;  kf *= inv_n;  ka *= inv_n;  mu *= inv_n

    # --- per-contact overrides ---
    if rigid_stiffness:
        ck = rigid_stiffness[tid]
        if ck > 0.0:
            ke = ck
        cd = rigid_damping[tid]
        if cd > 0.0:
            kd = cd
        cf = rigid_friction_scale[tid]
        if cf > 0.0:
            mu = mu * cf

    n = contact_normal[tid]
    th_a = contact_thick_a[tid]
    th_b = contact_thick_b[tid]

    # transform contact points to world space and apply thickness offsets
    pt_a = contact_pt_a[tid]
    pt_b = contact_pt_b[tid]
    r_a  = wp.vec3(0.0)
    r_b  = wp.vec3(0.0)

    if ba >= 0:
        tf_a = body_tf[ba]
        pt_a = wp.transform_point(tf_a, pt_a) - th_a * n
        r_a  = pt_a - wp.transform_point(tf_a, body_com[ba])

    if bb >= 0:
        tf_b = body_tf[bb]
        pt_b = wp.transform_point(tf_b, pt_b) + th_b * n
        r_b  = pt_b - wp.transform_point(tf_b, body_com[bb])

    gap = wp.dot(n, pt_a - pt_b)   # positive = separated
    if gap >= ka:
        return

    # --- velocities at contact points ---
    va = wp.vec3(0.0)
    vb = wp.vec3(0.0)

    if ba >= 0:
        sv_a    = body_vel[ba]
        omega_a = wp.spatial_bottom(sv_a)
        linv_a  = wp.spatial_top(sv_a)
        va = linv_a + wp.cross(omega_a, pt_a if wrench_in_world else r_a)

    if bb >= 0:
        sv_b    = body_vel[bb]
        omega_b = wp.spatial_bottom(sv_b)
        linv_b  = wp.spatial_top(sv_b)
        vb = linv_b + wp.cross(omega_b, pt_b if wrench_in_world else r_b)

    rel_v = va - vb
    v_n   = wp.dot(n, rel_v)
    v_t   = rel_v - n * v_n

    # normal penalty + damping (only damp approach velocity)
    fn = gap * ke
    # fd = wp.min(v_n, 0.0) * kd * wp.step(gap)
    fd = wp.min(v_n, 0.0) * kd

    # Huber-norm friction (smooth near zero sliding)
    ft = wp.vec3(0.0)
    if gap < 0.0:
        vs = wp.norm_huber(v_t, delta=friction_smoothing)
        if vs > 0.0:
            ft = (v_t / vs) * wp.min(kf * vs, -mu * (fn + fd))

    f_total = n * (fn + fd) + ft

    # scatter wrenches to bodies
    if ba >= 0:
        arm_a = pt_a if wrench_in_world else r_a
        if wrench_in_world:
            wp.atomic_add(body_wrench, ba, wp.spatial_vector(f_total, wp.cross(arm_a, f_total)))
        else:
            wp.atomic_sub(body_wrench, ba, wp.spatial_vector(f_total, wp.cross(arm_a, f_total)))

    if bb >= 0:
        arm_b = pt_b if wrench_in_world else r_b
        if wrench_in_world:
            wp.atomic_sub(body_wrench, bb, wp.spatial_vector(f_total, wp.cross(arm_b, f_total)))
        else:
            wp.atomic_add(body_wrench, bb, wp.spatial_vector(f_total, wp.cross(arm_b, f_total)))


# =====================================================================
# Python-side launchers
# =====================================================================


def apply_particle_proximity(model, state, pforce: wp.array):
    """Launch particle–particle proximity contact kernel."""
    if model.particle_count > 1 and model.particle_grid is not None:
        wp.launch(
            kernel=particle_proximity_kernel,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_flags,
                model.particle_ke,
                model.particle_kd,
                model.particle_kf,
                model.particle_mu,
                model.particle_cohesion,
                model.particle_max_radius,
            ],
            outputs=[pforce],
            device=model.device,
        )


def apply_mesh_particle_contact(model, state, pforce: wp.array):
    """Launch mesh–particle triangle contact kernel."""
    if model.tri_count and model.particle_count:
        wp.launch(
            kernel=mesh_particle_contact_kernel,
            dim=model.tri_count * model.particle_count,
            inputs=[
                model.particle_count,
                state.particle_q,
                state.particle_qd,
                model.tri_indices,
                model.tri_materials,
                model.particle_radius,
                model.soft_contact_ke,
            ],
            outputs=[pforce],
            device=model.device,
        )


def apply_rigid_contacts(
    model,
    state,
    contacts,
    friction_smoothing: float = 1.0,
    wrench_in_world: bool = False,
    body_wrench_out: wp.array | None = None,
):
    """Launch rigid body–body penalty contact kernel."""
    if contacts is not None and contacts.rigid_contact_max:
        if body_wrench_out is None:
            body_wrench_out = state.body_f
        wp.launch(
            kernel=rigid_penalty_contact_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_kf,
                model.shape_material_ka,
                model.shape_material_mu,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                wrench_in_world,
                friction_smoothing,
            ],
            outputs=[body_wrench_out],
            device=model.device,
        )


def apply_particle_shape_contact(
    model,
    state,
    contacts,
    pforce: wp.array,
    body_wrench: wp.array,
    wrench_in_world: bool = False,
):
    """Launch particle–shape soft contact kernel."""
    if contacts is not None and contacts.soft_contact_max:
        wp.launch(
            kernel=particle_shape_contact_kernel,
            dim=contacts.soft_contact_max,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.body_q,
                state.body_qd,
                model.particle_radius,
                model.particle_flags,
                model.body_com,
                model.shape_body,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_kf,
                model.shape_material_mu,
                model.shape_material_ka,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_kf,
                model.soft_contact_mu,
                model.particle_adhesion,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_max,
                wrench_in_world,
            ],
            outputs=[pforce, body_wrench],
            device=model.device,
        )
