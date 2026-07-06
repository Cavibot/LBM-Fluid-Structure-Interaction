import warp as wp

from ..base_kernels import sample_velocity_mac as sample_mac_velocity
from ..liquid.kernels import (
    advect_particles_in_grid_rk2,
    advect_u,
    advect_v,
    advect_w,
    apply_gravity_u,
    apply_gravity_v,
    apply_gravity_w,
    bake_solid_box_kernel,
    bake_solid_mesh_kernel,
    build_u_face_weights,
    build_v_face_weights,
    build_w_face_weights,
    compute_divergence_cell_type_mac,
    enforce_solid_u,
    enforce_solid_v,
    enforce_solid_w,
    extrapolate_u_from_valid,
    extrapolate_v_from_valid,
    extrapolate_w_from_valid,
    fill_grid_face,
    initialize_liquid_cell_state,
    mark_liquid_cells_from_particles,
    normalize_face_velocity_with_valid,
    pressure_apply_operator_cell_type_mac,
    pressure_build_inv_diag_cell_type_mac,
    pressure_jacobi_cell_type_mac,
    project_u_cell_type,
    project_v_cell_type,
    project_w_cell_type,
    resolve_particle_solid_collision_with_velocity,
)


@wp.kernel
def p2g_velocity_u_apic(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_c: wp.array(dtype=wp.mat33),
    grid_u: wp.array3d(dtype=float),
    weight_u: wp.array3d(dtype=float),
    solid_weight_u: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_v[tid]
    c = particle_c[tid]

    gx = p[0] / dh
    gy = p[1] / dh - 0.5
    gz = p[2] / dh - 0.5

    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))

    fx = gx - float(i0)
    fy = gy - float(j0)
    fz = gz - float(k0)

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                if 0 <= i <= nx and 0 <= j < ny and 0 <= k < nz:
                    face_frac = solid_weight_u[i, j, k]
                    if face_frac > 0.0:
                        wx = (1.0 - fx) if di == 0 else fx
                        wy = (1.0 - fy) if dj == 0 else fy
                        wz = (1.0 - fz) if dk == 0 else fz
                        w = wx * wy * wz * face_frac

                        x_face = wp.vec3(float(i) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
                        dpos = x_face - p
                        v_affine = v + c * dpos

                        wp.atomic_add(grid_u, i, j, k, v_affine[0] * w)
                        wp.atomic_add(weight_u, i, j, k, w)


@wp.kernel
def p2g_velocity_v_apic(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_c: wp.array(dtype=wp.mat33),
    grid_v: wp.array3d(dtype=float),
    weight_v: wp.array3d(dtype=float),
    solid_weight_v: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_v[tid]
    c = particle_c[tid]

    gx = p[0] / dh - 0.5
    gy = p[1] / dh
    gz = p[2] / dh - 0.5

    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))

    fx = gx - float(i0)
    fy = gy - float(j0)
    fz = gz - float(k0)

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                if 0 <= i < nx and 0 <= j <= ny and 0 <= k < nz:
                    face_frac = solid_weight_v[i, j, k]
                    if face_frac > 0.0:
                        wx = (1.0 - fx) if di == 0 else fx
                        wy = (1.0 - fy) if dj == 0 else fy
                        wz = (1.0 - fz) if dk == 0 else fz
                        w = wx * wy * wz * face_frac

                        x_face = wp.vec3((float(i) + 0.5) * dh, float(j) * dh, (float(k) + 0.5) * dh)
                        dpos = x_face - p
                        v_affine = v + c * dpos

                        wp.atomic_add(grid_v, i, j, k, v_affine[1] * w)
                        wp.atomic_add(weight_v, i, j, k, w)


@wp.kernel
def p2g_velocity_w_apic(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_c: wp.array(dtype=wp.mat33),
    grid_w: wp.array3d(dtype=float),
    weight_w: wp.array3d(dtype=float),
    solid_weight_w: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_v[tid]
    c = particle_c[tid]

    gx = p[0] / dh - 0.5
    gy = p[1] / dh - 0.5
    gz = p[2] / dh

    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))

    fx = gx - float(i0)
    fy = gy - float(j0)
    fz = gz - float(k0)

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                if 0 <= i < nx and 0 <= j < ny and 0 <= k <= nz:
                    face_frac = solid_weight_w[i, j, k]
                    if face_frac > 0.0:
                        wx = (1.0 - fx) if di == 0 else fx
                        wy = (1.0 - fy) if dj == 0 else fy
                        wz = (1.0 - fz) if dk == 0 else fz
                        w = wx * wy * wz * face_frac

                        x_face = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, float(k) * dh)
                        dpos = x_face - p
                        v_affine = v + c * dpos

                        wp.atomic_add(grid_w, i, j, k, v_affine[2] * w)
                        wp.atomic_add(weight_w, i, j, k, w)


@wp.kernel
def update_particle_velocity_apic(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_c: wp.array(dtype=wp.mat33),
    vel_u: wp.array3d(dtype=float),
    vel_v: wp.array3d(dtype=float),
    vel_w: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]

    v_center = sample_mac_velocity(vel_u, vel_v, vel_w, p, dh, nx, ny, nz)

    dx = wp.vec3(dh, 0.0, 0.0)
    dy = wp.vec3(0.0, dh, 0.0)
    dz = wp.vec3(0.0, 0.0, dh)
    inv_2h = 0.5 / dh

    dvel_dx = (
        sample_mac_velocity(vel_u, vel_v, vel_w, p + dx, dh, nx, ny, nz)
        - sample_mac_velocity(vel_u, vel_v, vel_w, p - dx, dh, nx, ny, nz)
    ) * inv_2h
    dvel_dy = (
        sample_mac_velocity(vel_u, vel_v, vel_w, p + dy, dh, nx, ny, nz)
        - sample_mac_velocity(vel_u, vel_v, vel_w, p - dy, dh, nx, ny, nz)
    ) * inv_2h
    dvel_dz = (
        sample_mac_velocity(vel_u, vel_v, vel_w, p + dz, dh, nx, ny, nz)
        - sample_mac_velocity(vel_u, vel_v, vel_w, p - dz, dh, nx, ny, nz)
    ) * inv_2h

    particle_v[tid] = v_center
    particle_c[tid] = wp.mat33(
        dvel_dx[0],
        dvel_dy[0],
        dvel_dz[0],
        dvel_dx[1],
        dvel_dy[1],
        dvel_dz[1],
        dvel_dx[2],
        dvel_dy[2],
        dvel_dz[2],
    )

