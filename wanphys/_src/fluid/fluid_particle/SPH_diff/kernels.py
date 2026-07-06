from __future__ import annotations

from enum import IntEnum

import warp as wp

from ..builder import _PARTICLE_FLAG_ACTIVE
from ..dfsph.kernels import c_CubicGradW as cubic_kernel_derivative
from ..dfsph.kernels import c_Cubic_W as cubic_kernel


class _ParticleFlags:
    ACTIVE = _PARTICLE_FLAG_ACTIVE


ParticleFlags = _ParticleFlags


@wp.func
def diff_viscous_kernel_cubic(r: wp.vec3,  v: wp.vec3, neighbor_v: wp.vec3 , neighbor_rho: float, smoothing_length: float):
    # calculate distance
    distance = wp.sqrt(wp.dot(r, r))
    dim = 3
    v_xy = wp.dot(v - neighbor_v, r)
    # calculate terms of kernel
    res = float(2 * (dim + 2)) * v_xy / (
        distance**2. + 0.01 * smoothing_length**2.) / neighbor_rho * cubic_kernel_derivative(r, smoothing_length)
    return res


@wp.func
def diff_pressure_kernel_cubic(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, rho: float , neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        # term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_2 = neighbor_pressure / (neighbor_rho * neighbor_rho) + pressure / (rho * rho)
        term_3 = cubic_kernel_derivative(xyz, smoothing_length)  # gradient of SPH kernel (grad W); TODO: use another kernel
        return term_2 * term_3
    else:
        return wp.vec3()


# Used for fluid-solid distinction


class MaterialType(IntEnum):
    SOLID = 0
    FLUID = 1

@wp.struct
class MaterialMarks():
    # store material id per particle (int) and dynamic flag (int)
    material: wp.array(dtype=int)
    is_dynamic: wp.array(dtype=int)

@wp.struct
class RigidBodies():
    rigid_rest_cm: wp.array(dtype=wp.vec3)
    rigid_x: wp.array(dtype=wp.vec3)
    rigid_v0: wp.array(dtype=wp.vec3)
    rigid_v: wp.array(dtype=wp.vec3)
    rigid_quaternion: wp.array(dtype=wp.quat)
    rigid_omega: wp.array(dtype=wp.vec3)
    rigid_omega0: wp.array(dtype=wp.vec3)
    rigid_force: wp.array(dtype=wp.vec3)
    rigid_torque: wp.array(dtype=wp.vec3)
    rigid_mass: wp.array(dtype=wp.float32)
    rigid_inertia: wp.array(dtype=wp.mat33)
    rigid_inertia0: wp.array(dtype=wp.mat33)
    rigid_inv_inertia: wp.array(dtype=wp.mat33)
    rigid_inv_mass: wp.array(dtype=wp.float32)


@wp.func
def is_dynamic_rigid_body(mtr: MaterialMarks, idx: int):
    return mtr.material[idx] == MaterialType.SOLID and mtr.is_dynamic[idx] != 0


@wp.func
def is_active_particle(particle_flags: wp.array(dtype=wp.int32), idx: int):
    return (particle_flags[idx] & ParticleFlags.ACTIVE) != 0



@wp.kernel
def compute_dfsph_factor_kernel(
    n: int,
    alpha: wp.array(dtype=float),
    particle_x: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    volume: float,
    smoothing_length: float,
    boundary_q: wp.array(dtype=wp.vec3),
    boundary_psi: wp.array(dtype=float),
    boundary_grid: wp.uint64,
    rho0: float,
    has_boundary: int,
): 
    tid = wp.tid()
    if tid >= n:
        return

    if not is_active_particle(particle_flags, tid):
        alpha[tid] = 0.0
        return

    x_i = particle_x[tid]

    sum_grad_p_k = float(0.0)
    grad_p_i = wp.vec3(0.0, 0.0, 0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    for index in neighbors:
        if index == tid:
            continue
            
        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        
        if d < smoothing_length:
            if is_active_particle(particle_flags, index):
                grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -volume * grad_W
                sum_grad_p_k += wp.length_sq(grad_p_j)
                grad_p_i -= grad_p_j

    if has_boundary != 0:
        neighbors_b = wp.hash_grid_query(boundary_grid, x_i, smoothing_length)
        index_b = int(0)
        while wp.hash_grid_query_next(neighbors_b, index_b):
            r_vec = x_i - boundary_q[index_b]
            grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
            grad_p_i += grad_W * (boundary_psi[index_b] / rho0)

    sum_grad_p_k += wp.length_sq(grad_p_i)

    if sum_grad_p_k > 1e-6:
        alpha[tid] = -1.0 / sum_grad_p_k 
    else:
        alpha[tid] = 0.0

@wp.kernel
def compute_density_adv_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    base_density: float,
    density_adv_out: wp.array(dtype=float)
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        density_adv_out[i] = 0.0
        return
        
    # get local particle variables
    x_i = particle_x[i]
    v_i = particle_v[i]
    
    delta = float(0.0)

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    for index in neighbors:
        if index == i:
            continue
            
        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        
        if d < smoothing_length:
            v_j = particle_v[index]
            
            if mtr.material[index] == MaterialType.FLUID or mtr.material[index] == MaterialType.SOLID:
                grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                
                v_ij = v_i - v_j
                delta += m_V[index] * wp.dot(v_ij, grad_W)
    
    density_ratio = particle_rho[i] / base_density
    adv_val = density_ratio + dt * delta
    density_adv_out[i] = wp.max(adv_val, 1.0)


@wp.kernel
def compute_density_change_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dim: int,
    density_change_out: wp.array(dtype=float)
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        density_change_out[i] = 0.0
        return

    x_i = particle_x[i]
    v_i = particle_v[i]

    density_adv = float(0.0)
    num_neighbors = wp.int32(0)

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)

        if d < smoothing_length:
            if mtr.material[index] == MaterialType.FLUID or mtr.material[index] == MaterialType.SOLID:
                v_j = particle_v[index]
                grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                density_adv += m_V[index] * wp.dot(v_i - v_j, grad_w)
                num_neighbors += 1

    density_adv = wp.max(density_adv, 0.0)
    
    # Do not perform divergence solve when particle deficiency happens
    if dim == 3:
        if num_neighbors < 20:
            density_adv = 0.0
    else:
        if num_neighbors < 7:
            density_adv = 0.0

    density_change_out[i] = density_adv

@wp.kernel
def pressure_solve_iteration_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    density_adv: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    base_density: float,
    particle_v_out: wp.array(dtype=wp.vec3),
    object_id: wp.array(dtype=wp.int32),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    rigid_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        # For non-fluid particles, we might not update velocity here, but we should copy it
        particle_v_out[i] = particle_v[i]
        return
        
    # get local particle variables
    x_i = particle_x[i]
    v_i = particle_v[i]
    
    # Evaluate rhs
    # b_i = self.ps.density_adv[p_i] - 1.0
    # k_i = b_i * self.ps.dfsph_factor[p_i]
    # NOTE: dfpsh_factor needs to be scaled by 1/dt^2
    inv_dt2 = 1.0 / (dt * dt)
    
    b_i = density_adv[i] - 1.0
    k_i = b_i * dfsph_factor[i] * inv_dt2
    
    m_eps = 1e-5
    
    # particle contact
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue
            
        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        
        if d < smoothing_length:
            
            if mtr.material[index] == MaterialType.FLUID:

                b_j = density_adv[index] - 1.0
                k_j = b_j * dfsph_factor[index] * inv_dt2
                
                k_sum = k_i + k_j # assuming density_0 ratio is 1
                
                if wp.abs(k_sum) > m_eps:
                    grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = -m_V[index] * grad_W
                    
                    force = -dt * k_sum * grad_p_j
                    vel_change_sum += force

            elif mtr.material[index] == MaterialType.SOLID:
                 if wp.abs(k_i) > m_eps:
                    grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = -m_V[index] * grad_W
                    
                    vel_change = -dt * k_i * grad_p_j
                    
                    vel_change_sum += vel_change 
                    
                    if mtr.is_dynamic[index] != 0:
                        r_id = object_id[index]
                        
                        rho_i = density_adv[i] * base_density # Approximate current density
                        force_rigid = -vel_change * (1.0/dt) * rho_i * m_V[i]
                        
                        wp.atomic_add(rigid_force, r_id, force_rigid)
                        wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)

@wp.kernel
def divergence_solve_iteration_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_change: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    object_id: wp.array(dtype=wp.int32),
    rigid_x: wp.array(dtype=wp.vec3),
    particle_v_out: wp.array(dtype=wp.vec3),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        particle_v_out[i] = particle_v[i]
        return

    x_i = particle_x[i]

    inv_dt = 1.0 / dt
    m_eps = 1e-5

    b_i = density_change[i]
    k_i = b_i * dfsph_factor[i] * inv_dt

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)

        if d < smoothing_length:
            if mtr.material[index] == MaterialType.FLUID:
                b_j = density_change[index]
                k_j = b_j * dfsph_factor[index] * inv_dt
                k_sum = k_i + k_j

                if wp.abs(k_sum) > m_eps:
                    grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = -m_V[index] * grad_w
                    vel_change_sum += -dt * k_sum * grad_p_j

            elif mtr.material[index] == MaterialType.SOLID:
                if wp.abs(k_i) > m_eps:
                    grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = -m_V[index] * grad_w
                    vel_change = -dt * k_i * grad_p_j
                    vel_change_sum += vel_change

                    if mtr.is_dynamic[index] != 0:
                        r_id = object_id[index]
                        force_rigid = -vel_change * inv_dt * particle_rho[i] * m_V[i]
                        wp.atomic_add(rigid_force, r_id, force_rigid)
                        wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def pressure_solve_iteration_kernel_fluid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    density_adv: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt2 = 1.0 / (dt * dt)
    m_eps = 1e-5
    b_i = density_adv[i] - 1.0
    k_i = b_i * dfsph_factor[i] * inv_dt2

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.FLUID:
            b_j = density_adv[index] - 1.0
            k_j = b_j * dfsph_factor[index] * inv_dt2
            k_sum = k_i + k_j

            if wp.abs(k_sum) > m_eps:
                grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -m_V[index] * grad_w
                vel_change_sum += -dt * k_sum * grad_p_j

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def pressure_solve_iteration_kernel_solid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    density_adv: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    base_density: float,
    particle_v_out: wp.array(dtype=wp.vec3),
    object_id: wp.array(dtype=wp.int32),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    rigid_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt = 1.0 / dt
    inv_dt2 = 1.0 / (dt * dt)
    m_eps = 1e-5
    b_i = density_adv[i] - 1.0
    k_i = b_i * dfsph_factor[i] * inv_dt2

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.SOLID:
            if wp.abs(k_i) > m_eps:
                grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -m_V[index] * grad_w
                vel_change = -dt * k_i * grad_p_j
                vel_change_sum += vel_change

                if mtr.is_dynamic[index] != 0:
                    r_id = object_id[index]
                    rho_i = density_adv[i] * base_density
                    force_rigid = -vel_change * inv_dt * rho_i * m_V[i]
                    wp.atomic_add(rigid_force, r_id, force_rigid)
                    wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def divergence_solve_iteration_kernel_fluid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    density_change: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt = 1.0 / dt
    m_eps = 1e-5
    b_i = density_change[i]
    k_i = b_i * dfsph_factor[i] * inv_dt

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.FLUID:
            b_j = density_change[index]
            k_j = b_j * dfsph_factor[index] * inv_dt
            k_sum = k_i + k_j

            if wp.abs(k_sum) > m_eps:
                grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -m_V[index] * grad_w
                vel_change_sum += -dt * k_sum * grad_p_j

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def divergence_solve_iteration_kernel_solid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_change: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    object_id: wp.array(dtype=wp.int32),
    rigid_x: wp.array(dtype=wp.vec3),
    particle_v_out: wp.array(dtype=wp.vec3),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt = 1.0 / dt
    m_eps = 1e-5
    b_i = density_change[i]
    k_i = b_i * dfsph_factor[i] * inv_dt

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.SOLID:
            if wp.abs(k_i) > m_eps:
                grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -m_V[index] * grad_w
                vel_change = -dt * k_i * grad_p_j
                vel_change_sum += vel_change

                if mtr.is_dynamic[index] != 0:
                    r_id = object_id[index]
                    force_rigid = -vel_change * inv_dt * particle_rho[i] * m_V[i]
                    wp.atomic_add(rigid_force, r_id, force_rigid)
                    wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def compute_density_error_kernel(
    density_adv: wp.array(dtype=float),
    mtr: MaterialMarks,
    base_density: float,
    offset: float,
    error_sum: wp.array(dtype=float)
):
    tid = wp.tid()
    if mtr.material[tid] == MaterialType.FLUID:
        err = base_density * density_adv[tid] - offset

        wp.atomic_add(error_sum, 0, err)


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    base_density: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = particle_x[i]
    rho = float(0.0)
    # rho = m_V[i] * cubic_kernel(wp.vec3(0.0, 0.0, 0.0), smoothing_length)
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    count = float(0.0)
    if mtr.material[i] == MaterialType.FLUID:
        for index in neighbors:
            if index == i or wp.length(x - particle_x[index]) > smoothing_length:
                continue
            distance = x - particle_x[index]
            rho += m_V[index] * cubic_kernel(distance, smoothing_length)
            count += 1.
        particle_rho[i] = density_normalization * base_density * rho

    particle_rho[i] = wp.max(particle_rho[i], base_density)

@wp.kernel
def compute_pressure(
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    mtr: MaterialMarks,
    stiffness: float,
    exponent: float,
    base_density: float,
):
    tid = wp.tid()
    if mtr.material[tid] == MaterialType.FLUID:
        rho = particle_rho[tid]
        rho = wp.max(rho, base_density)
        particle_rho[tid] = rho
        pressure = stiffness * (wp.pow(rho / base_density, exponent) - 1.0)
        particle_p[tid] = pressure


@wp.kernel
def compute_non_pressure_forces(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    viscous_normalization: float,
    smoothing_length: float,
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    base_density: float,
    particle_viscous_force: wp.array(dtype=wp.vec3),
    surface_tension: float,
    gravity: float,
    a_non_p: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        particle_viscous_force[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    x = particle_x[i]
    v = particle_v[i]

    viscous_force = wp.vec3(0.0, 0.0, 0.0)
    surface_tension_acc = wp.vec3(0.0, 0.0, 0.0)
    particle_diameter = 0.5 * smoothing_length
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    for index in neighbors:
        if index != i:
            d = wp.length(x - particle_x[index])
            if d < smoothing_length:
                neighbor_v = particle_v[index]
                neighbor_rho = particle_rho[index]
                relative_position = particle_x[index] - x

                if mtr.material[index] == MaterialType.FLUID:
                    viscous_force += base_density * m_V[index] * diff_viscous_kernel_cubic(
                        relative_position,
                        v,
                        neighbor_v,
                        neighbor_rho,
                        smoothing_length,
                    )
                    r = x - particle_x[index]
                    r2 = wp.dot(r, r)
                    m_i = wp.max(m_V[i] * base_density, 1.0e-8)
                    m_j = m_V[index] * base_density
                    if r2 > particle_diameter * particle_diameter:
                        surface_tension_acc += -surface_tension / m_i * m_j * r * cubic_kernel(r, smoothing_length)
                    else:
                        r_ref = wp.vec3(particle_diameter, 0.0, 0.0)
                        surface_tension_acc += -surface_tension / m_i * m_j * r * cubic_kernel(r_ref, smoothing_length)

    particle_viscous_force[i] = viscous_normalization * viscous_force
    a_non_p[i] = particle_viscous_force[i] # + wp.vec3(0.0, 0.0, gravity)


@wp.kernel
def compute_pressure_a(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    base_density: float,
    pressure_normalization_no_mass: float,
    smoothing_length: float,
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    particle_pressure_force: wp.array(dtype=wp.vec3),
    neibor_nums: wp.array(dtype=wp.int32),
    object_id: wp.array(dtype=wp.int32),
    rbs: RigidBodies,
    particle_a: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x = particle_x[i]
    rho = particle_rho[i]
    neibor_nums[i] = 0
    pressure = particle_p[i]
    pressure_force = wp.vec3()
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        for index in neighbors:
            if index != i and wp.length(x - particle_x[index]) < smoothing_length:
                neighbor_rho = particle_rho[index]
                neighbor_pressure = particle_p[index]
                relative_position = particle_x[index] - x
                if mtr.material[index] == MaterialType.FLUID:
                    pressure_force += base_density * m_V[index] * diff_pressure_kernel_cubic(
                        relative_position,
                        pressure,
                        neighbor_pressure,
                        rho,
                        neighbor_rho,
                        smoothing_length,
                    )
                elif mtr.material[index] == MaterialType.SOLID:
                    fp = base_density * m_V[index] * diff_pressure_kernel_cubic(
                        relative_position,
                        pressure,
                        pressure,
                        rho,
                        base_density,
                        smoothing_length,
                    )
                    d = wp.length(relative_position)
                    if d < smoothing_length * (1.0 / 2.0):
                        neibor_nums[i] = 1
                    pressure_force += fp
                    if is_dynamic_rigid_body(mtr, index):
                        r_id = object_id[index]
                        force = -fp * rho * m_V[i]
                        rbs.rigid_force[r_id] += force
                        rbs.rigid_torque[r_id] += wp.cross(x - rbs.rigid_x[r_id], force)

        pressure_force = pressure_force * pressure_normalization_no_mass
        particle_pressure_force[i] = pressure_force
        particle_a[i] += pressure_force


@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    damping_coef: float,
    width: float,
    height: float,
    length: float,
    mtr: MaterialMarks,
):
    tid = wp.tid()
    x = particle_x[tid]
    v = particle_v[tid]

    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[0] > width:
        x = wp.vec3(width, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)
    if x[2] > length:
        x = wp.vec3(x[0], x[1], length)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    particle_x[tid] = x
    particle_v[tid] = v

@wp.kernel
def kick(particle_a: wp.array(dtype=wp.vec3), dt: float, particle_v: wp.array(dtype=wp.vec3), 
         particle_v_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v_out[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float,
          particle_x_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x_out[tid] = x + particle_v[tid] * dt



@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3),
    smoothing_length: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()
    nr_x = wp.int32(width / 4.0 / smoothing_length)
    nr_y = wp.int32(height / smoothing_length)
    nr_z = wp.int32(length / 4.0 / smoothing_length)

    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))
    particle_x[tid] = pos


@wp.func
def simulate_collisions_warp(particle_v: wp.array(dtype=wp.vec3), idx: int, n: wp.vec3):
    c_f = 0.5
    v = particle_v[idx]
    particle_v[idx] = v - (1.0 + c_f) * wp.dot(v, n) * n


@wp.kernel
def enforce_boundary_3D_warp(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    mtr: MaterialMarks,
    lower_bound: wp.vec3,
    upper_bound: wp.vec3,
    padding: float,
):
    tid = wp.tid()

    if mtr.material[tid] != MaterialType.FLUID or not mtr.is_dynamic[tid]:
        return

    pos = particle_x[tid]
    collision_normal = wp.vec3(0.0, 0.0, 0.0)

    if pos[0] > upper_bound[0] - padding:
        collision_normal = collision_normal + wp.vec3(-1.0, 0.0, 0.0)
        pos = wp.vec3(upper_bound[0] - padding, pos[1], pos[2])
    if pos[0] < lower_bound[0] + padding:
        collision_normal = collision_normal + wp.vec3(1.0, 0.0, 0.0)
        pos = wp.vec3(lower_bound[0] + padding, pos[1], pos[2])

    if pos[1] > upper_bound[1] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, -1.0, 0.0)
        pos = wp.vec3(pos[0], upper_bound[1] - padding, pos[2])
    if pos[1] < lower_bound[1] + padding:
        collision_normal = collision_normal + wp.vec3(0.0, 1.0, 0.0)
        pos = wp.vec3(pos[0], lower_bound[1] + padding, pos[2])

    if pos[2] > upper_bound[2] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, -1.0)
        pos = wp.vec3(pos[0], pos[1], upper_bound[2] - padding)
    if pos[2] < lower_bound[2] + padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, 1.0)
        pos = wp.vec3(pos[0], pos[1], lower_bound[2] + padding)

    particle_x[tid] = pos

    cn_len = wp.length(collision_normal)
    if cn_len > 1e-6:
        n = collision_normal / cn_len
        simulate_collisions_warp(particle_v, tid, n)

@wp.kernel
def compute_moving_boundary_volume(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    m_V : wp.array(dtype=wp.float32),
    density_normalization_no_mass: float, # constant term in poly6 kernel multi mass of particle
    smoothing_length: float,
    mtr : MaterialMarks
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid) # order threads by cell
    if is_dynamic_rigid_body(mtr, i):
        x = particle_x[i]
        rho = cubic_kernel(wp.vec3(), smoothing_length)  # self-contribution
        # loop through neighbors to compute density
        neighbors = wp.hash_grid_query(grid, x, smoothing_length)
        for index in neighbors:
            if mtr.material[index] == MaterialType.SOLID:
                # compute distance
                distance = x - particle_x[index]
                # compute kernel derivative, the cube term in poly6 kernel
                rho += cubic_kernel(distance, smoothing_length)

        m_V[i] = 1.0 / rho * 3.0  

@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    stiffness: float,
    exponent : float,
    base_density: float,
    gravity: float,
    pressure_normalization_no_mass: float,
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    particle_pressure_force: wp.array(dtype=wp.vec3),
    particle_viscous_force: wp.array(dtype=wp.vec3),
    debug_val: wp.array(dtype=wp.float32),
    object_id: wp.array(dtype=wp.int32),
    particle_a_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = particle_p[i]

    # store forces
    pressure_force = wp.vec3()
    # viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        # loop through neighbors to compute acceleration
        for index in neighbors:
            dis = wp.length(x - particle_x[index])
            debug_val[i] = max(debug_val[i], wp.float32(dis / smoothing_length))
            if index != i and dis  < smoothing_length:
                # get neighbor velocity
                # neighbor_v = particle_v[index]
                # get neighbor density and pressures
                neighbor_rho = particle_rho[index]
                # neighbor_pressure = stiffness * (wp.pow(neighbor_rho / base_density, exponent) - 1.0)  # TODO: consider storing pressure to save computation
                neighbor_pressure = particle_p[index]
                # neighbor_pressure = isotropic_exp * (neighbor_rho - base_density) 

                # compute relative position
                relative_position = particle_x[index] - x
                if mtr.material[index] == MaterialType.FLUID:

                    # distance check for support radius
                    d = wp.length(relative_position)
                    if d < smoothing_length:
                        # term_2: pressure contributions
                        term_2 = neighbor_pressure / (neighbor_rho * neighbor_rho) + pressure / (rho * rho)
                        # term_3: gradient of cubic kernel
                        term_3 = cubic_kernel_derivative(relative_position, smoothing_length)
                        # accumulate pressure force contribution
                        pressure_force += base_density * m_V[index] * term_2 * term_3

        particle_pressure_force[i] = pressure_force * pressure_normalization_no_mass
        # add external potential
        particle_a_out[i] = particle_pressure_force[i] + particle_viscous_force[i] + wp.vec3(0.0, 0.0, gravity)


@wp.kernel
def compute_rigid_force_torque(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    base_density: float,
    pressure_normalization_no_mass: float,
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    object_id: wp.array(dtype=wp.int32),
    debug_val: wp.array(dtype=wp.float32),
    rigid_x: wp.array(dtype=wp.vec3),
    use_custom_grad: bool,
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    particle_a_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    # 采用新的EOS公式计算压强 
    pressure = particle_p[i]
    # pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    # viscous_force = wp.vec3()
    fixed_h = smoothing_length * 10.
    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        count = wp.int32(0)
        # loop through neighbors to compute acceleration
        for index in neighbors:
            if mtr.material[index] != MaterialType.SOLID:
                continue
            relative_position = particle_x[index] - x
            if index != i :
                count += 1
                if mtr.material[index] == MaterialType.SOLID:
                    d = wp.length(relative_position)
                    debug_val[i] = max(debug_val[i], wp.float32(d / smoothing_length))                    
                    if d < smoothing_length * (1.0):
                        debug_val[i] = wp.float32(d / smoothing_length)
                    term_2 = pressure / (base_density * base_density) + pressure / (rho * rho)
                    term_3 = cubic_kernel_derivative(relative_position, smoothing_length)
                    fp = base_density * m_V[index] * term_2 * term_3
                    pressure_force += fp
                    if  is_dynamic_rigid_body(mtr, index):
                        r_id = object_id[index]
                        # convert contribution to a force compatible with DFSPH's convention
                        force = - fp * rho * m_V[i]
                        wp.atomic_add(rigid_force, r_id, force)
                        wp.atomic_add(rigid_torque, r_id, wp.cross(x - rigid_x[r_id], force))

        particle_a_out[i] += pressure_force  * pressure_normalization_no_mass


@wp.kernel
def compute_moving_boundary_volume_object_id(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    m_V: wp.array(dtype=wp.float32),
    density_normalization_no_mass: float,
    smoothing_length: float,
    object_id: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if object_id[i] <= 0:
        return

    x = particle_x[i]
    rho = cubic_kernel(wp.vec3(), smoothing_length)
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    for index in neighbors:
        if object_id[index] > 0:
            distance = x - particle_x[index]
            rho += cubic_kernel(distance, smoothing_length)

    m_V[i] = 1.0 / rho * 3.0


@wp.kernel
def compute_density_object_id(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
    m_V: wp.array(dtype=float),
    base_density: float,
    particle_flags: wp.array(dtype=wp.int32),
    boundary_q: wp.array(dtype=wp.vec3),
    boundary_psi: wp.array(dtype=float),
    boundary_grid: wp.uint64,
    has_boundary: int,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if not is_active_particle(particle_flags, i):
        particle_rho[i] = 0.0
        return

    x = particle_x[i]
    rho = float(0.0)
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    for index in neighbors:
        if index == i or not is_active_particle(particle_flags, index) or wp.length(x - particle_x[index]) > smoothing_length:
            continue
        distance = x - particle_x[index]
        rho += base_density * m_V[index] * cubic_kernel(distance, smoothing_length)

    if has_boundary != 0:
        neighbors_b = wp.hash_grid_query(boundary_grid, x, smoothing_length)
        index_b = int(0)
        while wp.hash_grid_query_next(neighbors_b, index_b):
            distance = x - boundary_q[index_b]
            rho += boundary_psi[index_b] * cubic_kernel(distance, smoothing_length)

    particle_rho[i] = wp.max(density_normalization * rho, base_density)


@wp.kernel
def compute_pressure_object_id(
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    stiffness: float,
    exponent: float,
    base_density: float,
):
    tid = wp.tid()
    if not is_active_particle(particle_flags, tid):
        particle_p[tid] = 0.0
        particle_rho[tid] = base_density
        return

    rho = wp.max(particle_rho[tid], base_density)
    particle_rho[tid] = rho
    particle_p[tid] = stiffness * (wp.pow(rho / base_density, exponent) - 1.0)


@wp.kernel
def compute_non_pressure_forces_object_id(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    viscous_normalization: float,
    smoothing_length: float,
    particle_flags: wp.array(dtype=wp.int32),
    m_V: wp.array(dtype=float),
    base_density: float,
    particle_viscous_force: wp.array(dtype=wp.vec3),
    surface_tension: float,
    gravity: float,
    a_non_p: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if not is_active_particle(particle_flags, i):
        particle_viscous_force[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    x = particle_x[i]
    v = particle_v[i]
    viscous_force = wp.vec3(0.0, 0.0, 0.0)
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    for index in neighbors:
        if index == i or not is_active_particle(particle_flags, index):
            continue
        d = wp.length(x - particle_x[index])
        if d < smoothing_length:
            neighbor_v = particle_v[index]
            neighbor_rho = particle_rho[index]
            relative_position = particle_x[index] - x
            viscous_force += base_density * m_V[index] * diff_viscous_kernel_cubic(
                relative_position,
                v,
                neighbor_v,
                neighbor_rho,
                smoothing_length,
            )

    particle_viscous_force[i] = viscous_normalization * viscous_force
    a_non_p[i] = particle_viscous_force[i]


@wp.kernel
def get_acceleration_object_id(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    stiffness: float,
    exponent: float,
    base_density: float,
    gravity: float,
    pressure_normalization_no_mass: float,
    smoothing_length: float,
    particle_flags: wp.array(dtype=wp.int32),
    m_V: wp.array(dtype=float),
    particle_pressure_force: wp.array(dtype=wp.vec3),
    particle_viscous_force: wp.array(dtype=wp.vec3),
    debug_val: wp.array(dtype=wp.float32),
    boundary_q: wp.array(dtype=wp.vec3),
    boundary_psi: wp.array(dtype=float),
    boundary_grid: wp.uint64,
    boundary_object_id: wp.array(dtype=wp.int32),
    has_boundary: int,
    particle_a_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if not is_active_particle(particle_flags, i):
        return

    x = particle_x[i]
    rho = particle_rho[i]
    pressure = particle_p[i]
    pressure_force = wp.vec3()
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    for index in neighbors:
        dis = wp.length(x - particle_x[index])
        debug_val[i] = max(debug_val[i], wp.float32(dis / smoothing_length))
        if index == i or dis >= smoothing_length or not is_active_particle(particle_flags, index):
            continue
        neighbor_rho = particle_rho[index]
        neighbor_pressure = particle_p[index]
        relative_position = particle_x[index] - x
        term_2 = neighbor_pressure / (neighbor_rho * neighbor_rho) + pressure / (rho * rho)
        term_3 = cubic_kernel_derivative(relative_position, smoothing_length)
        pressure_force += base_density * m_V[index] * term_2 * term_3

    if has_boundary != 0:
        neighbors_b = wp.hash_grid_query(boundary_grid, x, smoothing_length)
        index_b = int(0)
        while wp.hash_grid_query_next(neighbors_b, index_b):
            relative_position = boundary_q[index_b] - x
            d = wp.length(relative_position)
            debug_val[i] = max(debug_val[i], wp.float32(d / smoothing_length))
            if d < smoothing_length:
                term_2 = pressure / (rho * rho)
                term_3 = cubic_kernel_derivative(relative_position, smoothing_length)
                pressure_force += boundary_psi[index_b] * term_2 * term_3

    particle_pressure_force[i] = pressure_force * pressure_normalization_no_mass
    particle_a_out[i] = particle_pressure_force[i] + particle_viscous_force[i] + wp.vec3(0.0, 0.0, gravity)


@wp.kernel
def compute_rigid_force_torque_object_id(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    base_density: float,
    pressure_normalization_no_mass: float,
    smoothing_length: float,
    m_V: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    debug_val: wp.array(dtype=wp.float32),
    rigid_x: wp.array(dtype=wp.vec3),
    use_custom_grad: bool,
    boundary_q: wp.array(dtype=wp.vec3),
    boundary_psi: wp.array(dtype=float),
    boundary_grid: wp.uint64,
    boundary_object_id: wp.array(dtype=wp.int32),
    has_boundary: int,
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    particle_a_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if not is_active_particle(particle_flags, i):
        return

    x = particle_x[i]
    rho = particle_rho[i]
    pressure = particle_p[i]
    pressure_force = wp.vec3()

    if has_boundary != 0:
        neighbors_b = wp.hash_grid_query(boundary_grid, x, smoothing_length)
        index_b = int(0)
        while wp.hash_grid_query_next(neighbors_b, index_b):
            relative_position = boundary_q[index_b] - x
            d = wp.length(relative_position)
            debug_val[i] = max(debug_val[i], wp.float32(d / smoothing_length))
            if d < smoothing_length:
                term_2 = pressure / (rho * rho)
                term_3 = cubic_kernel_derivative(relative_position, smoothing_length)
                fp = boundary_psi[index_b] * term_2 * term_3
                r_id = boundary_object_id[index_b]
                if r_id > 0:
                    force = -fp * rho * m_V[i]
                    wp.atomic_add(rigid_force, r_id, force)
                    wp.atomic_add(rigid_torque, r_id, wp.cross(x - rigid_x[r_id], force))

    particle_a_out[i] += pressure_force * pressure_normalization_no_mass


@wp.kernel
def sync_boundary_particles_from_object_id(
    boundary_q: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    boundary_indices: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= boundary_q.shape[0]:
        return
    boundary_q[tid] = particle_q[boundary_indices[tid]]


@wp.kernel
def compute_boundary_psi_from_boundary_grid(
    boundary_grid: wp.uint64,
    boundary_q: wp.array(dtype=wp.vec3),
    boundary_psi: wp.array(dtype=wp.float32),
    smoothing_length: float,
    rest_density: float,
):
    tid = wp.tid()
    if tid >= boundary_q.shape[0]:
        return

    x = boundary_q[tid]
    rho = cubic_kernel(wp.vec3(), smoothing_length)

    neighbors_b = wp.hash_grid_query(boundary_grid, x, smoothing_length)
    index_b = int(0)
    while wp.hash_grid_query_next(neighbors_b, index_b):
        if index_b == tid:
            continue
        distance = x - boundary_q[index_b]
        rho += cubic_kernel(distance, smoothing_length)

    # conservative factor (same as previous moving-volume heuristic)
    m_v = 1.0 / rho * 3.0 # 3.8
    # boundary_psi stores the boundary contribution in the same units as before
    boundary_psi[tid] = m_v * rest_density


@wp.kernel
def enforce_boundary_3D_warp_object_id(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    lower_bound: wp.vec3,
    upper_bound: wp.vec3,
    padding: float,
):
    tid = wp.tid()
    if not is_active_particle(particle_flags, tid):
        return

    pos = particle_x[tid]
    collision_normal = wp.vec3(0.0, 0.0, 0.0)

    if pos[0] > upper_bound[0] - padding:
        collision_normal = collision_normal + wp.vec3(-1.0, 0.0, 0.0)
        pos = wp.vec3(upper_bound[0] - padding, pos[1], pos[2])
    if pos[0] < lower_bound[0] + padding:
        collision_normal = collision_normal + wp.vec3(1.0, 0.0, 0.0)
        pos = wp.vec3(lower_bound[0] + padding, pos[1], pos[2])

    if pos[1] > upper_bound[1] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, -1.0, 0.0)
        pos = wp.vec3(pos[0], upper_bound[1] - padding, pos[2])
    if pos[1] < lower_bound[1] + padding:
        collision_normal = collision_normal + wp.vec3(0.0, 1.0, 0.0)
        pos = wp.vec3(pos[0], lower_bound[1] + padding, pos[2])

    if pos[2] > upper_bound[2] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, -1.0)
        pos = wp.vec3(pos[0], pos[1], upper_bound[2] - padding)
    if pos[2] < lower_bound[2] + padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, 1.0)
        pos = wp.vec3(pos[0], pos[1], lower_bound[2] + padding)

    particle_x[tid] = pos

    cn_len = wp.length(collision_normal)
    if cn_len > 1e-6:
        n = collision_normal / cn_len
        simulate_collisions_warp(particle_v, tid, n)
