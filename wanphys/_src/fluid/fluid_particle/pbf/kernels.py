# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
# pyright: reportInvalidTypeForm=false

"""Warp GPU kernels for Position Based Fluids (PBF).

This module contains all Warp kernels used by the PBF solver:
- SPH kernel functions (poly6, spiky gradient)
- Density and lambda calculation
- Position update and correction
- Boundary contact resolution
- Vorticity confinement and XSPH viscosity

Reference:
    Macklin, M., & Müller, M. (2013). Position based fluids.
    ACM Transactions on Graphics (TOG), 32(4), 1-12.
"""

from __future__ import annotations

import warp as wp

# ParticleFlags.ACTIVE == 1 in Newton; reuse local builder constant to avoid Newton import.
from ..builder import _PARTICLE_FLAG_ACTIVE


class _ParticleFlags:
    ACTIVE = _PARTICLE_FLAG_ACTIVE


ParticleFlags = _ParticleFlags
# =============================================================================
# SPH Kernel Functions
# =============================================================================


@wp.func
def poly6(r_sq: float, h_sq: float, poly6_coef: float) -> float:
    """Poly6 smoothing kernel (scalar form).

    W_poly6(r, h) = 315 / (64*pi*h^9) * (h^2 - r^2)^3, for r < h

    Args:
        r_sq: Squared distance between particles.
        h_sq: Squared support radius.
        poly6_coef: Coefficient for the poly6 kernel.

    Returns:
        Kernel value.
    """
    value = float(0.0)
    if r_sq < h_sq:
        diff = h_sq - r_sq
        rhs = diff * diff * diff
        value = poly6_coef * rhs
    return value


@wp.func
def spiky_gradient(r: wp.vec3, r_len: float, h: float, spiky_grad_coef: float) -> wp.vec3:
    """Spiky kernel gradient.

    gradW_spiky = -45/(pi*h^6) * (h-r)^2 * (r_ij / |r_ij|), for 0 < r < h

    Args:
        r: Vector from neighbor to particle (x_i - x_j).
        r_len: Length of r vector.
        h: Support radius.
        spiky_grad_coef: Coefficient for the spiky gradient kernel.

    Returns:
        Gradient vector.
    """
    value = wp.vec3(0.0)
    if 1e-6 < r_len < h:
        diff = h - r_len
        rhs = diff * diff
        scalar = spiky_grad_coef * rhs / r_len
        value = scalar * r
    return value


@wp.kernel
def integrate_gravity(
    particle_q_in: wp.array(dtype=wp.vec3),
    particle_qd_in: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    v_max: float,
    # outputs
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Apply gravity to particles (semi-implicit Euler) with velocity clamping."""
    i = wp.tid()

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        particle_q_out[i] = particle_q_in[i]
        particle_qd_out[i] = particle_qd_in[i]
        return

    # Update velocity
    v = particle_qd_in[i] + gravity * dt

    # Enforce velocity limit to prevent instability
    v_mag = wp.length(v)
    if v_mag > v_max:
        v = v * (v_max / v_mag)

    particle_qd_out[i] = v

    # Update position
    particle_q_out[i] = particle_q_in[i] + v * dt

# =============================================================================
# Density and Lambda Calculation
# =============================================================================


@wp.kernel
def calculate_lambdas(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_rest_density: float,
    support_radius: float,
    support_radius_sq: float,
    poly6_coef: float,
    spiky_grad_coef: float,
    relaxation_parameter: float,
    # output
    particle_lambdas: wp.array(dtype=float),
):
    """Calculate constraint multipliers (lambda) for each particle.

    Implements Equation 9 and 11 from PBF paper:
    λ_i = -C_i / (Σ_k |∇_pk C_i|² + ε)

    where C_i = (rho_i / rho_0) - 1 is the density constraint.
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        particle_lambdas[i] = 0.0
        return

    # Pre-compute constants
    h = support_radius
    h_sq = support_radius_sq

    density_i = float(0.0)
    constraint_gradient_i = wp.vec3(0.0)
    constraint_gradient_value_sq = float(0.0)

    x_i = particle_x[i]
    rest_density_i = particle_rest_density

    # Neighbor query
    query = wp.hash_grid_query(grid, x_i, support_radius)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        r = x_i - particle_x[index]
        r_sq = wp.length_sq(r)
        if r_sq < h_sq:
            mass_j = particle_mass[index]
            rest_density_j = particle_rest_density

            # Density contribution
            w = poly6(r_sq, h_sq, poly6_coef)
            density_i += mass_j * w

            # Constraint gradient
            r_len = wp.sqrt(r_sq)
            w_grad = spiky_gradient(r, r_len, h, spiky_grad_coef)

            constraint_gradient_j = mass_j * w_grad / rest_density_j
            constraint_gradient_value_sq += wp.length_sq(constraint_gradient_j)
            constraint_gradient_i += constraint_gradient_j

    # Compute lambda (Eq. 11)
    constraint_i = (density_i / rest_density_i) - 1.0
    constraint_i = wp.max(constraint_i, 0.0)  # Only compress, don't expand
    denominator = constraint_gradient_value_sq + wp.length_sq(constraint_gradient_i) + relaxation_parameter
    particle_lambdas[i] = -constraint_i / denominator


# =============================================================================
# Position Update
# =============================================================================


@wp.kernel
def calculate_position_update(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_lambdas: wp.array(dtype=float),
    particle_rest_density: float,
    support_radius: float,
    support_radius_sq: float,
    poly6_coef: float,
    spiky_grad_coef: float,
    k: float,
    n: float,
    d_q: float,
    # output
    particle_position_deltas: wp.array(dtype=wp.vec3),
):
    """Calculate position corrections for each particle.

    Implements Equation 12 and 13 from PBF paper:
    delta_p_i = (1/rho_0) * sum_j (lambda_i + lambda_j + s_corr) * gradW(p_i - p_j)

    where s_corr is the artificial pressure term for anti-clustering.
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        particle_position_deltas[i] = wp.vec3(0.0)
        return

    # Pre-compute constants
    h = support_radius
    h_sq = support_radius_sq

    # W(d_q) for artificial pressure
    w_dq = float(1.0)
    if d_q < h:
        dq_sq = d_q * d_q
        w_dq = poly6(dq_sq, h_sq, poly6_coef)

    x_i = particle_x[i]
    lambda_i = particle_lambdas[i]

    query = wp.hash_grid_query(grid, x_i, support_radius)
    index = int(0)

    particle_position_delta = wp.vec3(0.0)
    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        r = x_i - particle_x[index]
        r_sq = wp.length_sq(r)

        if 1e-12 < r_sq < h_sq:
            mass_j = particle_mass[index]
            r_len = wp.sqrt(r_sq)

            # Artificial pressure (Eq. 13)
            w = poly6(r_sq, h_sq, poly6_coef)
            ratio = w / w_dq
            s_corr = -k * wp.pow(ratio, n)

            # Position update (Eq. 12)
            lambda_j = particle_lambdas[index]
            w_grad = spiky_gradient(r, r_len, h, spiky_grad_coef)
            factor = lambda_i + lambda_j + s_corr

            particle_position_delta += factor * w_grad * mass_j

    particle_position_deltas[i] = particle_position_delta / particle_rest_density


@wp.kernel
def apply_position_deltas(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_position_deltas: wp.array(dtype=wp.vec3),
):
    """Apply position corrections to particles."""
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    particle_x[i] += particle_position_deltas[i]


# =============================================================================
# Boundary Contact Resolution
# =============================================================================
@wp.kernel
def solve_boundary_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_radius: float,
    particle_mass: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    friction: float,
    dt: float,
    # output
    particle_deltas: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    ):
    """Resolve particle-boundary contacts and apply two-way coupling.

    Particle response uses normal separation plus Coulomb-like tangential
    correction. For dynamic rigid bodies, the opposite reaction is accumulated
    into ``body_f`` as force and torque in world space.
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    p_idx = contact_particle[tid]
    shape_idx = contact_shape[tid]
    if p_idx < 0 or shape_idx < 0:
        return
    if (particle_flags[p_idx] & ParticleFlags.ACTIVE) == 0:
        return

    body_idx = shape_body[shape_idx]

    px = particle_x[p_idx]
    pv = particle_v[p_idx]
    mass = particle_mass[p_idx]

    # Build world transform for the contacted rigid body (identity for static/world).
    X_wb = wp.transform_identity()
    if body_idx >= 0:
        X_wb = body_q[body_idx]

    # Contact point and normal in world space.
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    n = contact_normal[tid]
    n_len = wp.length(n)
    if n_len < 1e-6:
        return
    n = n / n_len

    # Signed separation constraint C = n·(x - x_contact) - radius.
    c = wp.dot(n, px - bx) - particle_radius

    if c > 0.0:
        return

    # Compute full body contact-point velocity in world frame.
    body_vel_w = wp.vec3(0.0)
    com_w = wp.vec3(0.0)
    lever = wp.vec3(0.0)
    if body_idx >= 0:
        com_w = wp.transform_point(X_wb, body_com[body_idx])
        lever = bx - com_w
        sv = body_qd[body_idx]
        body_vel_w = (
            wp.spatial_top(sv)
            + wp.transform_vector(X_wb, contact_body_vel[tid])
            + wp.cross(wp.spatial_bottom(sv), lever)
        )

    v_rel = pv - body_vel_w

    # Split correction into normal and tangential components.
    # Use a softened, clamped pushout instead of full penetration removal
    # to reduce velocity spikes after v = (x_new - x_old) / dt reconstruction.
    penetration = -c
    max_pushout = 0.2 * particle_radius
    normal_stiffness = 0.6
    # delta_n = n * (wp.min(penetration, max_pushout) * normal_stiffness)
    delta_n = n * penetration * normal_stiffness

    vn = wp.dot(n, v_rel)
    vt = v_rel - n * vn

    # Coulomb friction: do not over-correct beyond available tangential motion.
    vt_len = wp.length(vt)
    if vt_len > 1e-6:
        max_friction_dist = friction * wp.abs(c)
        friction_dist = wp.min(max_friction_dist, vt_len * dt)

        # Friction correction is opposite to tangential relative velocity.
        delta_f = -wp.normalize(vt) * friction_dist
    else:
        delta_f = wp.vec3(0.0, 0.0, 0.0)

    # Total correction with infinite-mass boundary assumption.
    total_delta = delta_n + delta_f

    # Apply correction to particle
    wp.atomic_add(particle_deltas, p_idx, total_delta)

    # Convert correction impulse to force/torque for rigid body feedback.
    if body_idx >= 0 and dt > 0.0:
        j = -(mass / dt) * total_delta
        f = j / dt
        torque = wp.cross(lever, f)
        wp.atomic_add(body_f, body_idx, wp.spatial_vector(f, torque))


# =============================================================================
# Velocity Update
# =============================================================================


@wp.kernel
def update_velocity(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_x_init: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    # output
    particle_v: wp.array(dtype=wp.vec3),
):
    """Reconstruct velocities from corrected positions.

    Uses the standard finite-difference estimate:
    ``v = (x_new - x_init) / dt``.
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    particle_v[i] = (particle_x[i] - particle_x_init[i]) / dt


# =============================================================================
# AABB Boundary Enforcement
# =============================================================================


@wp.kernel
def clamp_aabb_bounds(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    particle_radius: float,
):
    """Clamp particle positions to an AABB (center bounds shrunk by radius)."""
    i = wp.tid()

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_q[i]

    lo = bounds_min + wp.vec3(particle_radius, particle_radius, particle_radius)
    hi = bounds_max - wp.vec3(particle_radius, particle_radius, particle_radius)

    x0 = wp.clamp(x[0], lo[0], hi[0])
    x1 = wp.clamp(x[1], lo[1], hi[1])
    x2 = wp.clamp(x[2], lo[2], hi[2])
    particle_q[i] = wp.vec3(x0, x1, x2)


@wp.kernel
def enforce_aabb_bounds(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    particle_radius: float,
    restitution: float,
    damping: float,
):
    """Enforce AABB boundary constraints with restitution and damping."""
    i = wp.tid()

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    lo = bounds_min + wp.vec3(particle_radius, particle_radius, particle_radius)
    hi = bounds_max - wp.vec3(particle_radius, particle_radius, particle_radius)

    x = particle_q[i]
    v = particle_qd[i]
    hit = 0

    if x[0] < lo[0]:
        hit = 1
        x = wp.vec3(lo[0], x[1], x[2])
        v = wp.vec3(-v[0] * restitution, v[1], v[2])
    elif x[0] > hi[0]:
        hit = 1
        x = wp.vec3(hi[0], x[1], x[2])
        v = wp.vec3(-v[0] * restitution, v[1], v[2])

    if x[1] < lo[1]:
        hit = 1
        x = wp.vec3(x[0], lo[1], x[2])
        v = wp.vec3(v[0], -v[1] * restitution, v[2])
    elif x[1] > hi[1]:
        hit = 1
        x = wp.vec3(x[0], hi[1], x[2])
        v = wp.vec3(v[0], -v[1] * restitution, v[2])

    if x[2] < lo[2]:
        hit = 1
        x = wp.vec3(x[0], x[1], lo[2])
        v = wp.vec3(v[0], v[1], -v[2] * restitution)
    elif x[2] > hi[2]:
        hit = 1
        x = wp.vec3(x[0], x[1], hi[2])
        v = wp.vec3(v[0], v[1], -v[2] * restitution)

    if hit == 1:
        v *= wp.max(0.0, 1.0 - damping)

    particle_q[i] = x
    particle_qd[i] = v


# =============================================================================
# Vorticity Confinement and XSPH Viscosity
# =============================================================================


@wp.kernel
def calculate_vorticity(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_rest_density: float,
    support_radius: float,
    support_radius_sq: float,
    spiky_grad_coef: float,
    # output
    particle_vorticities: wp.array(dtype=wp.vec3),
):
    """Calculate vorticity (curl of velocity) for each particle.

    omega_i = sum_j cross(v_j - v_i, gradW(x_i - x_j))
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        particle_vorticities[i] = wp.vec3(0.0)
        return

    # Pre-compute constants
    h = support_radius
    h_sq = support_radius_sq

    x_i = particle_x[i]
    v_i = particle_v[i]

    query = wp.hash_grid_query(grid, x_i, support_radius)
    index = int(0)
    particle_vorticity = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        r = x_i - particle_x[index]
        r_sq = wp.length_sq(r)

        if 1e-12 < r_sq < h_sq:
            r_len = wp.sqrt(r_sq)
            v_ij = particle_v[index] - v_i
            w_grad = spiky_gradient(r, r_len, h, spiky_grad_coef)
            mass_j = particle_mass[index]
            rest_density_j = particle_rest_density

            particle_vorticity += wp.cross(v_ij, w_grad) * mass_j / rest_density_j

    particle_vorticities[i] = particle_vorticity


@wp.kernel
def apply_vorticity_confinement_and_XSPH_viscosity(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_rest_density: float,
    particle_vorticities: wp.array(dtype=wp.vec3),
    support_radius: float,
    support_radius_sq: float,
    poly6_coef: float,
    spiky_grad_coef: float,
    dt: float,
    vorticity_coefficient: float,
    c: float,
):
    """Apply vorticity confinement force and XSPH viscosity.

    Vorticity confinement (Eq. 16):
    f_vorticity = eps * cross(N, omega_i)
    where N = grad(|omega|) / |grad(|omega|)|

    XSPH viscosity (Eq. 17):
    v_i += c * sum_j (v_j - v_i) * W(x_i - x_j)
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    # Pre-compute constants
    h = support_radius
    h_sq = support_radius_sq

    x_i = particle_x[i]
    v_i = particle_v[i]
    vorticity_i = particle_vorticities[i]
    vorticity_i_value = wp.length(vorticity_i)

    query = wp.hash_grid_query(grid, x_i, support_radius)
    index = int(0)

    vorticity_value_gradient = wp.vec3(0.0)
    particle_velocity_delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        r = x_i - particle_x[index]
        r_sq = wp.length_sq(r)

        if 1e-12 < r_sq < h_sq:
            r_len = wp.sqrt(r_sq)

            mass_ratio = particle_mass[index] / particle_rest_density

            # Gradient of vorticity magnitude
            vorticity_j_value = wp.length(particle_vorticities[index])
            w_grad = spiky_gradient(r, r_len, h, spiky_grad_coef)

            vorticity_value_gradient += (vorticity_j_value - vorticity_i_value) * w_grad * mass_ratio

            # XSPH viscosity
            v_ij = particle_v[index] - v_i
            w = poly6(r_sq, h_sq, poly6_coef)
            particle_velocity_delta += v_ij * w * mass_ratio

    # XSPH contribution
    particle_velocity_delta = c * particle_velocity_delta

    # Vorticity confinement force
    vorticity_value_gradient_value = wp.length(vorticity_value_gradient)
    if vorticity_value_gradient_value > 1e-6:
        N = vorticity_value_gradient / vorticity_value_gradient_value
        vorticity_force = vorticity_coefficient * wp.cross(N, vorticity_i)
        particle_velocity_delta += vorticity_force * dt * (1.0 / particle_mass[i])

    particle_v[i] += particle_velocity_delta
