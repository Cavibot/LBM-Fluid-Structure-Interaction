# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp GPU kernels for Weakly Compressible SPH (WCSPH).

This module contains all Warp kernels used by the WCSPH solver:
- SPH kernel functions (poly6, spiky gradient, viscosity Laplacian)
- Density and pressure calculation (Tait EOS)
- SPH force computation (pressure + viscosity)
- Semi-implicit time integration
- Boundary Enforcement
- XSPH velocity smoothing

Reference:
    Becker, M., & Teschner, M. (2007). Weakly compressible SPH for free surface flows.
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
def poly6_kernel(r: float, h: float, poly6_coef: float) -> float:
    """Poly6 smoothing kernel.

    W_poly6(r, h) = 315 / (64*pi*h^9) * (h^2 - r^2)^3, for 0 <= r <= h

    Args:
        r: Distance between particles.
        h: Support radius (smoothing length).
        poly6_coef: Pre-computed coefficient (315 / (64*pi*h^9)).

    Returns:
        Kernel value.
    """
    if r >= h:
        return 0.0
    x = h * h - r * r
    return poly6_coef * x * x * x


@wp.func
def spiky_grad(r_ij: wp.vec3, r: float, h: float, spiky_grad_coef: float) -> wp.vec3:
    """Spiky kernel gradient.

    ∇W_spiky = -45/(pi*h^6) * (h-r)^2 * (r_ij / r), for 0 < r <= h

    Args:
        r_ij: Vector from particle j to particle i.
        r: Distance between particles.
        h: Support radius.
        spiky_grad_coef: Pre-computed coefficient (-45 / (pi*h^6)).

    Returns:
        Gradient vector.
    """
    if r <= 1.0e-12 or r >= h:
        return wp.vec3(0.0, 0.0, 0.0)
    x = h - r
    return r_ij * (spiky_grad_coef * x * x / r)


@wp.func
def visc_laplacian(r: float, h: float, visc_lap_coef: float) -> float:
    """Viscosity kernel Laplacian.

    ∇²W_visc = 45/(pi*h^6) * (h-r), for 0 <= r <= h

    Args:
        r: Distance between particles.
        h: Support radius.
        visc_lap_coef: Pre-computed coefficient (45 / (pi*h^6)).

    Returns:
        Laplacian value.
    """
    if r >= h:
        return 0.0
    return visc_lap_coef * (h - r)


# =============================================================================
# Density and Pressure Calculation
# =============================================================================


@wp.kernel
def compute_density_pressure(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    rest_density: float,
    poly6_coef: float,
    c0: float,
    gamma: float,
    # outputs
    rho: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
):
    """Compute particle density and pressure using Tait EOS.

    Density: ρ_i = Σ_j m_j W(|x_i - x_j|, h)
    Pressure: p_i = k * ((ρ_i/ρ_0)^γ - 1), where k = ρ_0 * c_0² / γ
    """
    tid = wp.tid()

    # Order threads by cell for cache-friendly access
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        rho[i] = rest_density
        pressure[i] = 0.0
        return

    xi = particle_q[i]
    rho_i = float(0.0)

    # Neighbor query
    query = wp.hash_grid_query(grid, xi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue
        r = wp.length(xi - particle_q[j])
        rho_i += particle_mass[j] * poly6_kernel(r, h, poly6_coef)

    # Clamp to avoid division-by-zero
    rho_i = wp.max(rho_i, 1.0e-6)
    rho[i] = rho_i

    # Tait EOS
    k = rest_density * c0 * c0 / gamma
    ratio = rho_i / rest_density
    p = k * (wp.pow(ratio, gamma) - 1.0)
    pressure[i] = wp.max(p, 0.0)


@wp.kernel
def compute_density_only(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    rest_density: float,
    poly6_coef: float,
    # outputs
    rho: wp.array(dtype=float),
):
    """Compute particle density only (for XSPH smoothing)."""
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        rho[i] = rest_density
        return

    xi = particle_q[i]
    rho_i = float(0.0)

    query = wp.hash_grid_query(grid, xi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue
        r = wp.length(xi - particle_q[j])
        rho_i += particle_mass[j] * poly6_kernel(r, h, poly6_coef)

    rho_i = wp.max(rho_i, 1.0e-6)
    rho[i] = rho_i


# =============================================================================
# SPH Force Computation
# =============================================================================


@wp.kernel
def compute_sph_forces(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    viscosity: float,
    spiky_grad_coef: float,
    visc_lap_coef: float,
    rho: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
    # outputs
    sph_f: wp.array(dtype=wp.vec3),
):
    """Compute SPH forces (pressure + viscosity).

    F_pressure = -m * Σ_j m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W_spiky
    F_viscosity = μ * Σ_j m_j * (v_j - v_i) / ρ_j * ∇²W_visc
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        sph_f[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    rhoi = wp.max(rho[i], 1.0e-6)
    pi = pressure[i]

    z = float(0.0)
    f_press = wp.vec3(z, z, z)
    f_visc = wp.vec3(z, z, z)

    query = wp.hash_grid_query(grid, xi, h)
    j = int(0)

    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue

        xj = particle_q[j]
        r_ij = xi - xj
        r = wp.length(r_ij)

        if r <= 1.0e-6 or r >= h:
            continue

        rhoj = wp.max(rho[j], 1.0e-6)
        pj = pressure[j]
        mj = particle_mass[j]

        # Pressure force (symmetric form)
        grad = spiky_grad(r_ij, r, h, spiky_grad_coef)
        f_press += grad * (-mj * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)))

        # Viscosity force
        lap = visc_laplacian(r, h, visc_lap_coef)
        f_visc += (particle_qd[j] - vi) * (viscosity * mj * lap / (rhoi * rhoj))

    # Convert to force (F = m * a)
    sph_f[i] = (f_press + f_visc) * particle_mass[i]


# =============================================================================
# Time Integration
# =============================================================================


@wp.kernel
def integrate_semi_implicit(
    x_in: wp.array(dtype=wp.vec3),
    v_in: wp.array(dtype=wp.vec3),
    f_in: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    """Semi-implicit (symplectic Euler) time integration with velocity clamping.

    v_new = v + (F/m + g) * dt  (kick)
    x_new = x + v_new * dt      (drift)
    """
    i = wp.tid()

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        x_out[i] = x_in[i]
        v_out[i] = v_in[i]
        return

    m = wp.max(particle_mass[i], 1.0e-8)
    a = f_in[i] / m + gravity

    # Kick (update velocity)
    v = v_in[i] + a * dt

    # Enforce velocity limit to prevent instability
    v_mag = wp.length(v)
    if v_mag > v_max:
        v = v * (v_max / v_mag)

    # Drift (update position)
    x = x_in[i] + v * dt

    v_out[i] = v
    x_out[i] = x


# =============================================================================
# Boundary Enforcement 推粒子解决穿透、改速度实现非弹性碰撞 + 阻尼
# =============================================================================
@wp.kernel
def resolve_soft_contacts_inelastic(
    # Particle state (modified in-place)
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    # Rigid body transforms
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    # Soft contacts data
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    # Parameters
    margin: float,
    max_push_frac: float,
    vel_damp: float,
):
    """Conservative soft contact resolution for particle-mesh collision."""
    tid = wp.tid()
    count = soft_contact_count[0]
    if tid >= count:
        return

    #拿到 contact 对应的粒子和 shape
    p = soft_contact_particle[tid]
    s = soft_contact_shape[tid]
    if p < 0 or s < 0:
        return
    if (particle_flags[p] & ParticleFlags.ACTIVE) == 0:
        return

    #处理法线
    n = soft_contact_normal[tid]
    n_len = wp.length(n)
    if n_len < 1.0e-6:
        return
    n = n / n_len

    # Transform contact point to world space
    body_pos_local = soft_contact_body_pos[tid]
    body_vel_local = soft_contact_body_vel[tid]

    rigid = shape_body[s]
    X_wb = wp.transform_identity()
    if rigid >= 0:
        X_wb = body_q[rigid]
    #把接触点和表面速度变到世界坐标
    surf_w = wp.transform_point(X_wb, body_pos_local)
    body_vel_w = wp.transform_vector(X_wb, body_vel_local)

    x = particle_q[p]
    v = particle_qd[p]
    r = particle_radius[p]

    # Penetration depth
    d = wp.dot(x - surf_w, n)
    pen = (margin + r) - d
    if pen <= 0.0:
        return

    # Limited position correction
    max_push = wp.max(0.0, max_push_frac) * r
    push = wp.min(pen, max_push)
    x = x + n * push

    # Relative velocity
    rel_v = v - body_vel_w

    # Non-elastic response: remove inward normal velocity
    vn = wp.dot(rel_v, n)
    if vn < 0.0:
        rel_v = rel_v - vn * n

    # Damping
    rel_v = rel_v * (1.0 - vel_damp)

    v = rel_v + body_vel_w

    particle_q[p] = x
    particle_qd[p] = v



# =============================================================================
# XSPH Velocity Smoothing
# =============================================================================


@wp.kernel
def compute_xsph_delta_v(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    h: float,
    c: float,
    poly6_coef: float,
    rho: wp.array(dtype=float),
    # outputs
    delta_v: wp.array(dtype=wp.vec3),
):
    """Compute XSPH velocity correction for smoothing.

    Δv_i = c * Σ_j (m_j / ρ_j) * (v_j - v_i) * W(|x_i - x_j|, h)
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        delta_v[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    xi = particle_q[i]
    vi = particle_qd[i]

    dv = wp.vec3(0.0, 0.0, 0.0)

    query = wp.hash_grid_query(grid, xi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue

        xj = particle_q[j]
        r = wp.length(xi - xj)
        if r >= h:
            continue

        rhoj = wp.max(rho[j], 1.0e-6)
        w = poly6_kernel(r, h, poly6_coef)
        dv += (particle_mass[j] / rhoj) * (particle_qd[j] - vi) * w

    delta_v[i] = dv * c


# =============================================================================
# Utility Kernels
# =============================================================================


@wp.kernel
def add_vec3(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    """Add two vec3 arrays element-wise."""
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def clear_vec3(a: wp.array(dtype=wp.vec3)):
    """Clear vec3 array to zero."""
    i = wp.tid()
    a[i] = wp.vec3(0.0, 0.0, 0.0)
