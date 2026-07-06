# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp GPU kernels for Divergence-Free SPH (DFSPH).

This module contains:
- Cubic Spline kernel and gradient
- Density and alpha factor computation
- Divergence solver (velocity correction)
- Pressure solver (constant density)
- Non-pressure forces (explicit gravity + viscosity)
- Position and velocity integration
"""

from __future__ import annotations

import warp as wp

from ..builder import _PARTICLE_FLAG_ACTIVE


class _ParticleFlags:
    ACTIVE = _PARTICLE_FLAG_ACTIVE


ParticleFlags = _ParticleFlags

# SPH kernel constants (Cubic Spline)
PI = 3.14159265359
M_K = 8.0 / PI
M_L = 48.0 / PI
EPS = 1e-6

# =============================================================================
# Smoothing kernel functions
# =============================================================================
@wp.func
def c_Cubic_W_P(q: float):
    """Polynomial part of the cubic spline kernel."""
    res = 0.0
    if q <= 1.0:
        if q <= 0.5:
            qq = q * q
            qqq = qq * q
            res = 6.0 * qqq - 6.0 * qq + 1.0
        else:
            factor = 1.0 - q
            res = 2.0 * factor * factor * factor
    return res

@wp.func
def c_Cubic_W(r_vec: wp.vec3, h: float):
    """Cubic spline kernel W(r, h)."""
    r = wp.length(r_vec)
    q = r / h
    h3 = 1.0 / (h * h * h)
    return c_Cubic_W_P(q) * M_K * h3

@wp.func
def c_CubicGradW(r: wp.vec3, h: float):
    """Gradient of cubic spline kernel ∇W(r, h)."""
    res = wp.vec3(0.0)
    rl = wp.length(r)
    q = rl / h
    h3 = 1.0 / (h * h * h)
    if (rl > 1.0e-5) and (q <= 1.0):
        gradq = r / (rl * h)
        if q <= 0.5:
            res = M_L * h3 * q * (3.0 * q - 2.0) * gradq
        else:
            factor = 1.0 - q
            res = -M_L * h3 * (factor * factor) * gradq
    return res

# =============================================================================
# Density and DFSPH alpha
# =============================================================================
@wp.kernel
def compute_density(
    n: int,
    rho: wp.array(dtype=float),
    pos: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    mass: float,
    h: float,
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    has_boundary: int,
):
    """Compute density: ρ_i = Σ m_j W_ij + Σ ψ_b W_ib."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        rho[tid] = 0.0
        return

    density = mass * c_Cubic_W(wp.vec3(0.0), h)

    x_i = pos[tid]
    query = wp.hash_grid_query(grid, x_i, h)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index != tid and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            r = x_i - pos[index]
            density += mass * c_Cubic_W(r, h)

    if has_boundary != 0:
        query_b = wp.hash_grid_query(b_grid, x_i, h)
        index_b = int(0)
        while wp.hash_grid_query_next(query_b, index_b):
            r = x_i - b_pos[index_b]
            density += b_psi[index_b] * c_Cubic_W(r, h)

    rho[tid] = density

@wp.kernel
def compute_dfsph_alpha(
    n: int,
    alpha: wp.array(dtype=float),
    pos: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    volume: float,
    h: float,
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    rho0: float,
    has_boundary: int,
):
    """Compute DFSPH factor alpha = -1 / (Σ |∇W|^2)."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        alpha[tid] = 0.0
        return

    sum_grad_sq = float(0.0)
    sum_grad = wp.vec3(0.0)

    x_i = pos[tid]
    query = wp.hash_grid_query(grid, x_i, h)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if index != tid and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            r = x_i - pos[index]
            grad = c_CubicGradW(r, h) * volume
            sum_grad_sq += wp.length_sq(grad)
            sum_grad += grad

    if has_boundary != 0:
        query_b = wp.hash_grid_query(b_grid, x_i, h)
        index_b = int(0)
        while wp.hash_grid_query_next(query_b, index_b):
            V_b = b_psi[index_b] / rho0
            sum_grad += c_CubicGradW(x_i - b_pos[index_b], h) * V_b

    sum_grad_sq += wp.length_sq(sum_grad)

    if sum_grad_sq > EPS:
        alpha[tid] = -1.0 / sum_grad_sq
    else:
        alpha[tid] = 0.0

# =============================================================================
# Divergence solver (velocity correction)
# =============================================================================
@wp.func
def update_drho_divergence(
    i: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    adv_rho: wp.array(dtype=float),
    volume: float,
    h: float,
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    rho0: float,
    has_boundary: int,
):
    """Compute predicted density change rate Dρ/Dt."""
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        adv_rho[i] = 0.0
        return

    d_rho = float(0.0)
    x_i = pos[i]
    v_i = vel[i]

    query = wp.hash_grid_query(grid, x_i, h)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index != i and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            r = x_i - pos[index]
            grad = c_CubicGradW(r, h)
            d_rho += wp.dot(v_i - vel[index], grad)

    if has_boundary != 0:
        query_b = wp.hash_grid_query(b_grid, x_i, h)
        index_b = int(0)
        while wp.hash_grid_query_next(query_b, index_b):
            V_b = b_psi[index_b] / rho0
            d_rho += wp.dot(v_i, c_CubicGradW(x_i - b_pos[index_b], h)) * (V_b / volume)

    adv_rho[i] = wp.max(d_rho * volume, 0.0)

@wp.kernel
def begin_divergence_iter(
    n: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    adv_rho: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    kappa_v: wp.array(dtype=float),
    dt_arr: wp.array(dtype=float),
    volume: float,
    h: float,
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    rho0: float,
    has_boundary: int,
):
    """Initialize divergence solver iteration."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        adv_rho[tid] = 0.0
        kappa_v[tid] = 0.0
        return

    dt = dt_arr[0]
    update_drho_divergence(
        tid, grid, pos, vel, particle_flags, adv_rho, volume, h,
        b_pos, b_psi, b_grid, rho0, has_boundary
    )

    if dt > EPS:
        alpha[tid] = alpha[tid] / dt
    kappa_v[tid] = 0.0

@wp.kernel
def divergence_step(
    n: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    adv_rho: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    kappa_v: wp.array(dtype=float),
    avg_density_err: wp.array(dtype=float),
    dt_arr: wp.array(dtype=float),
    volume: float,
    h: float,
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    rho0: float,
    has_boundary: int,
):
    """Perform one divergence solver iteration."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    dt = dt_arr[0]
    b_i = adv_rho[tid]
    k_i = b_i * alpha[tid]
    kappa_v[tid] += k_i

    query = wp.hash_grid_query(grid, pos[tid], h)
    index = int(0)
    delta_v = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if index != tid and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            r = pos[tid] - pos[index]
            grad = c_CubicGradW(r, h)
            k_sum = k_i + alpha[index] * adv_rho[index]
            if wp.abs(k_sum) > EPS:
                delta_v += grad * k_sum

    if has_boundary != 0:
        query_b = wp.hash_grid_query(b_grid, pos[tid], h)
        index_b = int(0)
        while wp.hash_grid_query_next(query_b, index_b):
            V_b = b_psi[index_b] / rho0
            delta_v += c_CubicGradW(pos[tid] - b_pos[index_b], h) * 2.0 * k_i * (V_b / volume)

    vel[tid] += delta_v * dt * volume

    update_drho_divergence(
        tid, grid, pos, vel, particle_flags, adv_rho, volume, h,
        b_pos, b_psi, b_grid, rho0, has_boundary
    )

@wp.kernel
def end_divergence_iter(
    n: int,
    particle_flags: wp.array(dtype=wp.int32),
    kappa_v: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    dt_arr: wp.array(dtype=float),
):
    """Finalize divergence solver iteration."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    dt = dt_arr[0]
    kappa_v[tid] *= dt
    if dt > EPS:
        alpha[tid] *= dt

# =============================================================================
# Non-pressure forces (explicit)
# =============================================================================
@wp.kernel
def compute_forces_and_update_vel(
    n: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    rho: wp.array(dtype=float),
    mass: float,
    gravity: wp.vec3,
    h: float,
    viscosity: float,
    dt_arr: wp.array(dtype=float),
):
    """Compute non-pressure forces and update velocity in-place."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    acc = gravity
    x_i = pos[tid]
    v_i = vel[tid]

    query = wp.hash_grid_query(grid, x_i, h)
    index = int(0)

    visc_force = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if index != tid and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            x_j = pos[index]
            v_j = vel[index]
            rho_j = rho[index]

            r_vec = x_i - x_j
            r_len = wp.length(r_vec)

            if r_len > 1e-6 and rho_j > EPS:
                gradW = c_CubicGradW(r_vec, h)
                laplacian = 2.0 * wp.dot(r_vec, gradW) / (r_len * r_len + 0.01 * h * h)
                visc_force += (v_i - v_j) * (mass / rho_j * laplacian)

    acc += visc_force * viscosity
    vel[tid] += acc * dt_arr[0]

# =============================================================================
# Pressure solver (constant density)
# =============================================================================
@wp.func
def update_drho_pressure(
    i: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    adv_rho: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    rho0: float,
    volume: float,
    h: float,
    dt: float,
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    has_boundary: int,
):
    """Update normalized predicted density for pressure solve."""
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        adv_rho[i] = 1.0
        return

    d_rho = float(0.0)
    query = wp.hash_grid_query(grid, pos[i], h)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index != i and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            r = pos[i] - pos[index]
            d_rho += wp.dot(vel[i] - vel[index], c_CubicGradW(r, h))

    if has_boundary != 0:
        query_b = wp.hash_grid_query(b_grid, pos[i], h)
        index_b = int(0)
        while wp.hash_grid_query_next(query_b, index_b):
            V_b = b_psi[index_b] / rho0
            d_rho += wp.dot(vel[i], c_CubicGradW(pos[i] - b_pos[index_b], h)) * (V_b / volume)

    adv_rho[i] = rho[i] / rho0 + dt * d_rho * volume
    adv_rho[i] = wp.max(1.0, adv_rho[i])

@wp.kernel
def begin_pressure_iter(
    n: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    adv_rho: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    rho0: float,
    volume: float,
    h: float,
    dt_arr: wp.array(dtype=float),
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    has_boundary: int,
):
    """Initialize pressure solver iteration."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        adv_rho[tid] = 1.0
        kappa[tid] = 0.0
        return

    dt = dt_arr[0]
    update_drho_pressure(
        tid, grid, pos, vel, particle_flags, adv_rho, rho, rho0, volume, h, dt,
        b_pos, b_psi, b_grid, has_boundary
    )

    if dt > EPS:
        alpha[tid] = alpha[tid] / (dt * dt)
    kappa[tid] = 0.0

@wp.kernel
def pressure_iter(
    n: int,
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    adv_rho: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    kappa: wp.array(dtype=float),
    prev_err: wp.array(dtype=float),
    curr_err: wp.array(dtype=float),
    tolerance: float,
    rho0: float,
    volume: float,
    h: float,
    dt_arr: wp.array(dtype=float),
    b_pos: wp.array(dtype=wp.vec3),
    b_psi: wp.array(dtype=float),
    b_grid: wp.uint64,
    has_boundary: int,
):
    """Perform one pressure solver iteration with GPU-side early exit."""
    if prev_err[0] <= tolerance:
        return

    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    dt = dt_arr[0]

    b_i = adv_rho[tid] - 1.0
    k_i = b_i * alpha[tid]
    kappa[tid] += k_i

    delta_v = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, pos[tid], h)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index != tid and (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            b_j = adv_rho[index] - 1.0
            k_j = b_j * alpha[index]
            k_sum = k_i + k_j
            if wp.abs(k_sum) > EPS:
                r = pos[tid] - pos[index]
                grad = c_CubicGradW(r, h)
                delta_v += grad * k_sum

    if has_boundary != 0:
        query_b = wp.hash_grid_query(b_grid, pos[tid], h)
        index_b = int(0)
        while wp.hash_grid_query_next(query_b, index_b):
            V_b = b_psi[index_b] / rho0
            delta_v += c_CubicGradW(pos[tid] - b_pos[index_b], h) * 2.0 * k_i * (V_b / volume)

    vel[tid] += delta_v * dt * volume

    update_drho_pressure(
        tid, grid, pos, vel, particle_flags, adv_rho, rho, rho0, volume, h, dt,
        b_pos, b_psi, b_grid, has_boundary
    )

    err_i = wp.abs(adv_rho[tid] - 1.0)
    wp.atomic_max(curr_err, 0, err_i)

@wp.kernel
def end_pressure_iter(
    n: int,
    particle_flags: wp.array(dtype=wp.int32),
    kappa: wp.array(dtype=float),
    dt_arr: wp.array(dtype=float),
):
    """Finalize pressure solver iteration."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    dt = dt_arr[0]
    kappa[tid] *= (dt * dt)

# =============================================================================
# Integration
# =============================================================================
@wp.kernel
def update_pos(
    n: int,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt_arr: wp.array(dtype=float),
):
    """Update position: x = x + v * dt."""
    tid = wp.tid()
    if tid >= n:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    pos[tid] += vel[tid] * dt_arr[0]
