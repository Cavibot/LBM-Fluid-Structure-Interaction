import warp as wp

from ..base_kernels import (
    advect_scalar_mac,
    advect_u_mac,
    advect_v_mac,
    advect_w_mac,
    apply_force_mac,
    bake_solid_sphere_kernel,
    compute_divergence_solid_mac,
    dissipate_scalar,
    enforce_solid_u_mac,
    enforce_solid_v_mac,
    enforce_solid_w_mac,
    pressure_jacobi_solid_mac,
    pressure_apply_operator_solid_mac,
    pressure_build_inv_diag_solid_mac,
    project_u_solid_mac,
    project_v_solid_mac,
    project_w_solid_mac,
)

from .sparse_hash_grid import pack_key, hash_func, HASH_MASK, BLOCK_SIZE, BLOCK_VOL


# Keep legacy kernel names for solver compatibility
compute_divergence_mac = compute_divergence_solid_mac
pressure_jacobi_mac = pressure_jacobi_solid_mac
pressure_apply_operator_mac = pressure_apply_operator_solid_mac
pressure_build_inv_diag_mac = pressure_build_inv_diag_solid_mac
project_u_mac = project_u_solid_mac
project_v_mac = project_v_solid_mac
project_w_mac = project_w_solid_mac


@wp.func
def is_fluid_cell(ix: int, iy: int, iz: int, nx: int, ny: int, nz: int, solid_phi: wp.array3d(dtype=float)) -> float:
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 0.0
    if solid_phi[ix, iy, iz] < 0.0:
        return 0.0
    return 1.0


@wp.func
def sample_cell_velocity_mac(
        ix: int,
        iy: int,
        iz: int,
        u: wp.array3d(dtype=float),
        v: wp.array3d(dtype=float),
        w: wp.array3d(dtype=float),
        solid_phi: wp.array3d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
) -> wp.vec3:
    if is_fluid_cell(ix, iy, iz, nx, ny, nz, solid_phi) < 0.5:
        return wp.vec3(0.0, 0.0, 0.0)

    return wp.vec3(
        0.5 * (u[ix, iy, iz] + u[ix + 1, iy, iz]),
        0.5 * (v[ix, iy, iz] + v[ix, iy + 1, iz]),
        0.5 * (w[ix, iy, iz] + w[ix, iy, iz + 1]),
    )


@wp.func
def sample_curl_mac(
        ix: int,
        iy: int,
        iz: int,
        curl: wp.array3d(dtype=wp.vec3),
        solid_phi: wp.array3d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
) -> wp.vec3:
    if is_fluid_cell(ix, iy, iz, nx, ny, nz, solid_phi) < 0.5:
        return wp.vec3(0.0, 0.0, 0.0)
    return curl[ix, iy, iz]


@wp.func
def compute_vorticity_force_mac(
        ix: int,
        iy: int,
        iz: int,
        curl: wp.array3d(dtype=wp.vec3),
        solid_phi: wp.array3d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
        dh: float,
        vorticity_scale: float,
) -> wp.vec3:
    if is_fluid_cell(ix, iy, iz, nx, ny, nz, solid_phi) < 0.5:
        return wp.vec3(0.0, 0.0, 0.0)

    curl_center = curl[ix, iy, iz]
    curl_mag_xp = wp.length(sample_curl_mac(ix + 1, iy, iz, curl, solid_phi, nx, ny, nz))
    curl_mag_xm = wp.length(sample_curl_mac(ix - 1, iy, iz, curl, solid_phi, nx, ny, nz))
    curl_mag_yp = wp.length(sample_curl_mac(ix, iy + 1, iz, curl, solid_phi, nx, ny, nz))
    curl_mag_ym = wp.length(sample_curl_mac(ix, iy - 1, iz, curl, solid_phi, nx, ny, nz))
    curl_mag_zp = wp.length(sample_curl_mac(ix, iy, iz + 1, curl, solid_phi, nx, ny, nz))
    curl_mag_zm = wp.length(sample_curl_mac(ix, iy, iz - 1, curl, solid_phi, nx, ny, nz))

    grad_mag = wp.vec3(
        (curl_mag_xp - curl_mag_xm) / (2.0 * dh),
        (curl_mag_yp - curl_mag_ym) / (2.0 * dh),
        (curl_mag_zp - curl_mag_zm) / (2.0 * dh),
    )

    grad_len = wp.length(grad_mag)
    if grad_len < 1.0e-6:
        return wp.vec3(0.0, 0.0, 0.0)

    normal = grad_mag / grad_len
    return vorticity_scale * dh * wp.cross(normal, curl_center)


@wp.kernel
def compute_curl_mac(
        u: wp.array3d(dtype=float),
        v: wp.array3d(dtype=float),
        w: wp.array3d(dtype=float),
        solid_phi: wp.array3d(dtype=float),
        curl_out: wp.array3d(dtype=wp.vec3),
        nx: int,
        ny: int,
        nz: int,
        dh: float,
):
    i, j, k = wp.tid()

    if solid_phi[i, j, k] < 0.0:
        curl_out[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel_xp = sample_cell_velocity_mac(i + 1, j, k, u, v, w, solid_phi, nx, ny, nz)
    vel_xm = sample_cell_velocity_mac(i - 1, j, k, u, v, w, solid_phi, nx, ny, nz)
    vel_yp = sample_cell_velocity_mac(i, j + 1, k, u, v, w, solid_phi, nx, ny, nz)
    vel_ym = sample_cell_velocity_mac(i, j - 1, k, u, v, w, solid_phi, nx, ny, nz)
    vel_zp = sample_cell_velocity_mac(i, j, k + 1, u, v, w, solid_phi, nx, ny, nz)
    vel_zm = sample_cell_velocity_mac(i, j, k - 1, u, v, w, solid_phi, nx, ny, nz)

    du_dy = (vel_yp[0] - vel_ym[0]) / (2.0 * dh)
    du_dz = (vel_zp[0] - vel_zm[0]) / (2.0 * dh)
    dv_dx = (vel_xp[1] - vel_xm[1]) / (2.0 * dh)
    dv_dz = (vel_zp[1] - vel_zm[1]) / (2.0 * dh)
    dw_dx = (vel_xp[2] - vel_xm[2]) / (2.0 * dh)
    dw_dy = (vel_yp[2] - vel_ym[2]) / (2.0 * dh)

    curl_out[i, j, k] = wp.vec3(
        dw_dy - dv_dz,
        du_dz - dw_dx,
        dv_dx - du_dy,
    )


@wp.kernel
def apply_vorticity_u_mac(
        u: wp.array3d(dtype=float),
        curl: wp.array3d(dtype=wp.vec3),
        solid_phi: wp.array3d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
        dh: float,
        dt: float,
        vorticity_scale: float,
):
    i, j, k = wp.tid()
    if i == 0 or i == nx:
        return

    force_sum = 0.0
    weight_sum = 0.0

    left_weight = is_fluid_cell(i - 1, j, k, nx, ny, nz, solid_phi)
    if left_weight > 0.5:
        force_sum += compute_vorticity_force_mac(i - 1, j, k, curl, solid_phi, nx, ny, nz, dh, vorticity_scale)[0]
        weight_sum += 1.0

    right_weight = is_fluid_cell(i, j, k, nx, ny, nz, solid_phi)
    if right_weight > 0.5:
        force_sum += compute_vorticity_force_mac(i, j, k, curl, solid_phi, nx, ny, nz, dh, vorticity_scale)[0]
        weight_sum += 1.0

    if weight_sum > 0.0:
        u[i, j, k] += dt * force_sum / weight_sum


@wp.kernel
def apply_vorticity_v_mac(
        v: wp.array3d(dtype=float),
        curl: wp.array3d(dtype=wp.vec3),
        solid_phi: wp.array3d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
        dh: float,
        dt: float,
        vorticity_scale: float,
):
    i, j, k = wp.tid()
    if j == 0 or j == ny:
        return

    force_sum = 0.0
    weight_sum = 0.0

    down_weight = is_fluid_cell(i, j - 1, k, nx, ny, nz, solid_phi)
    if down_weight > 0.5:
        force_sum += compute_vorticity_force_mac(i, j - 1, k, curl, solid_phi, nx, ny, nz, dh, vorticity_scale)[1]
        weight_sum += 1.0

    up_weight = is_fluid_cell(i, j, k, nx, ny, nz, solid_phi)
    if up_weight > 0.5:
        force_sum += compute_vorticity_force_mac(i, j, k, curl, solid_phi, nx, ny, nz, dh, vorticity_scale)[1]
        weight_sum += 1.0

    if weight_sum > 0.0:
        v[i, j, k] += dt * force_sum / weight_sum


@wp.kernel
def apply_vorticity_w_mac(
        w: wp.array3d(dtype=float),
        curl: wp.array3d(dtype=wp.vec3),
        solid_phi: wp.array3d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
        dh: float,
        dt: float,
        vorticity_scale: float,
):
    i, j, k = wp.tid()
    if k == 0 or k == nz:
        return

    force_sum = 0.0
    weight_sum = 0.0

    back_weight = is_fluid_cell(i, j, k - 1, nx, ny, nz, solid_phi)
    if back_weight > 0.5:
        force_sum += compute_vorticity_force_mac(i, j, k - 1, curl, solid_phi, nx, ny, nz, dh, vorticity_scale)[2]
        weight_sum += 1.0

    front_weight = is_fluid_cell(i, j, k, nx, ny, nz, solid_phi)
    if front_weight > 0.5:
        force_sum += compute_vorticity_force_mac(i, j, k, curl, solid_phi, nx, ny, nz, dh, vorticity_scale)[2]
        weight_sum += 1.0

    if weight_sum > 0.0:
        w[i, j, k] += dt * force_sum / weight_sum

@wp.func
def evaluate_source(gx: int, gy: int, gz: int, nx: int, ny: int) -> float:
    # 完美复刻稠密版的整数索引算距逻辑，还原那 0.5 格的“右偏”
    cx = wp.float32(nx) / 2.0
    cy = wp.float32(ny) / 2.0
    x = wp.float32(gx)
    y = wp.float32(gy)
    z = wp.float32(gz)

    source_radius = wp.float32(nx) * 0.15
    source_height = wp.float32(nx) * 0.08

    dx = x - cx
    dy = y - cy
    r = wp.sqrt(dx * dx + dy * dy)

    if r < source_radius and z < source_height:
        dist_norm = r / source_radius
        return 1.0 - wp.smoothstep(0.0, 1.0, dist_norm)
    return 0.0


@wp.kernel
def inject_velocity_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        vel_w: wp.array(dtype=float),
        n_grid: int,
        dh: float
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    vy = (voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)) // BLOCK_SIZE
    vz = (voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)) % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = (wp.float32(gx) + 0.5) * dh
    py = (wp.float32(gy) + 0.5) * dh

    center_x = wp.float32(n_grid) * 0.5 * dh
    center_y = wp.float32(n_grid) * 0.5 * dh
    source_radius = wp.float32(n_grid) * 0.15 * dh
    source_height = wp.float32(n_grid) * 0.08 * dh

    dx = px - center_x
    dy = py - center_y
    dist = wp.sqrt(dx * dx + dy * dy)

    if dist < source_radius:
        t_xy = 1.0 - wp.smoothstep(0.0, 1.0, dist / source_radius)

        pz_curr = wp.float32(gz) * dh
        in_source_curr = (pz_curr < source_height)
        pz_prev = wp.float32(gz - 1) * dh
        in_source_prev = (gz > 0) and (pz_prev < source_height)

        # 严格对齐稠密的 0.5，且不用 dt
        base_speed = 0.5
        if in_source_curr:
            vel_w[wp.tid()] += base_speed * t_xy
        if in_source_prev:
            vel_w[wp.tid()] += base_speed * t_xy


@wp.kernel
def inject_density_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        density: wp.array(dtype=float),
        n_grid: int,
        dh: float,
        source_strength: float
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    vy = (voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)) // BLOCK_SIZE
    vz = (voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)) % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = (wp.float32(gx) + 0.5) * dh
    py = (wp.float32(gy) + 0.5) * dh
    pz = (wp.float32(gz) + 0.5) * dh

    center_x = wp.float32(n_grid) * 0.5 * dh
    center_y = wp.float32(n_grid) * 0.5 * dh
    source_radius = wp.float32(n_grid) * 0.15 * dh
    source_height = wp.float32(n_grid) * 0.08 * dh

    dx = px - center_x
    dy = py - center_y
    dist = wp.sqrt(dx * dx + dy * dy)

    if dist < source_radius and pz < source_height:
        t = 1.0 - wp.smoothstep(0.0, 1.0, dist / source_radius)
        density[wp.tid()] += (1.0 - density[wp.tid()]) * source_strength * t

@wp.func
def get_sparse_idx(ix: int, iy: int, iz: int, hash_keys: wp.array(dtype=int), hash_vals: wp.array(dtype=int)) -> int:
    bx = ix // BLOCK_SIZE
    by = iy // BLOCK_SIZE
    bz = iz // BLOCK_SIZE
    lx = ix % BLOCK_SIZE
    ly = iy % BLOCK_SIZE
    lz = iz % BLOCK_SIZE

    packed = pack_key(bx, by, bz)
    h = hash_func(bx, by, bz)

    for i in range(100):
        idx = (h + i) & HASH_MASK
        k = hash_keys[idx]
        if k == packed:
            block_id = hash_vals[idx]
            if block_id >= 0:
                return block_id * BLOCK_VOL + lx * BLOCK_SIZE * BLOCK_SIZE + ly * BLOCK_SIZE + lz
            return -1
        if k == -1:
            return -1
    return -1

#0403 modify boundary sample
@wp.func
def clamp_center_pos(x: float, y: float, z: float, dh: float, nx: int, ny: int, nz: int) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x, 0.5 * dh, (wp.float32(nx) - 0.5) * dh),
        wp.clamp(y, 0.5 * dh, (wp.float32(ny) - 0.5) * dh),
        wp.clamp(z, 0.5 * dh, (wp.float32(nz) - 0.5) * dh),
    )


@wp.func
def clamp_u_pos(x: float, y: float, z: float, dh: float, nx: int, ny: int, nz: int) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x, 0.5 * dh, (wp.float32(nx) + 0.5) * dh),
        wp.clamp(y, 0.5 * dh, (wp.float32(ny) - 0.5) * dh),
        wp.clamp(z, 0.5 * dh, (wp.float32(nz) - 0.5) * dh),
    )


@wp.func
def clamp_v_pos(x: float, y: float, z: float, dh: float, nx: int, ny: int, nz: int) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x, 0.5 * dh, (wp.float32(nx) - 0.5) * dh),
        wp.clamp(y, 0.5 * dh, (wp.float32(ny) + 0.5) * dh),
        wp.clamp(z, 0.5 * dh, (wp.float32(nz) - 0.5) * dh),
    )


@wp.func
def clamp_w_pos(x: float, y: float, z: float, dh: float, nx: int, ny: int, nz: int) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x, 0.5 * dh, (wp.float32(nx) - 0.5) * dh),
        wp.clamp(y, 0.5 * dh, (wp.float32(ny) - 0.5) * dh),
        wp.clamp(z, 0.5 * dh, (wp.float32(nz) + 0.5) * dh),
    )


@wp.func
def sample_sparse_scalar(x: float, y: float, z: float, dh: float, hash_keys: wp.array(dtype=int),
                         hash_vals: wp.array(dtype=int), data: wp.array(dtype=float)) -> float:
    gx = x / dh - 0.5
    gy = y / dh - 0.5
    gz = z / dh - 0.5

    ix = wp.int32(wp.floor(gx))
    iy = wp.int32(wp.floor(gy))
    iz = wp.int32(wp.floor(gz))

    s1 = gx - wp.float32(ix)
    t1 = gy - wp.float32(iy)
    u1 = gz - wp.float32(iz)
    s0 = 1.0 - s1
    t0 = 1.0 - t1
    u0 = 1.0 - u1

    id000 = get_sparse_idx(ix, iy, iz, hash_keys, hash_vals)
    id100 = get_sparse_idx(ix + 1, iy, iz, hash_keys, hash_vals)
    id010 = get_sparse_idx(ix, iy + 1, iz, hash_keys, hash_vals)
    id110 = get_sparse_idx(ix + 1, iy + 1, iz, hash_keys, hash_vals)
    id001 = get_sparse_idx(ix, iy, iz + 1, hash_keys, hash_vals)
    id101 = get_sparse_idx(ix + 1, iy, iz + 1, hash_keys, hash_vals)
    id011 = get_sparse_idx(ix, iy + 1, iz + 1, hash_keys, hash_vals)
    id111 = get_sparse_idx(ix + 1, iy + 1, iz + 1, hash_keys, hash_vals)

    c000 = data[id000] if id000 >= 0 else 0.0
    c100 = data[id100] if id100 >= 0 else 0.0
    c010 = data[id010] if id010 >= 0 else 0.0
    c110 = data[id110] if id110 >= 0 else 0.0
    c001 = data[id001] if id001 >= 0 else 0.0
    c101 = data[id101] if id101 >= 0 else 0.0
    c011 = data[id011] if id011 >= 0 else 0.0
    c111 = data[id111] if id111 >= 0 else 0.0

    c0 = s0 * (t0 * c000 + t1 * c010) + s1 * (t0 * c100 + t1 * c110)
    c1 = s0 * (t0 * c001 + t1 * c011) + s1 * (t0 * c101 + t1 * c111)
    return u0 * c0 + u1 * c1

# 330 add new hashtable
# 0403 add nx ny nz params
@wp.kernel
def advect_scalar_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        density_in: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        density_out: wp.array(dtype=float),
        # hash_keys: wp.array(dtype=int),
        # hash_vals: wp.array(dtype=int),
        # modify
        new_hash_keys: wp.array(dtype=int),  # <--- 新增：新字典 (查新速度场用)
        new_hash_vals: wp.array(dtype=int),  # <--- 新增：新字典
        old_hash_keys: wp.array(dtype=int),  # <--- 老字典 (查旧密度用)
        old_hash_vals: wp.array(dtype=int),  # <--- 老字典
        dt: float,
        dh: float,
        nx: int,
        ny: int,
        nz: int
        ):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL

    coord = block_coords[block_id]

    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE

    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = (wp.float32(gx) + 0.5) * dh
    py = (wp.float32(gy) + 0.5) * dh
    pz = (wp.float32(gz) + 0.5) * dh

    # 330 add offset
    # u = sample_sparse_scalar(px + 0.5 * dh, py, pz, dh, new_hash_keys, new_hash_vals, vel_u)
    # v = sample_sparse_scalar(px, py + 0.5 * dh, pz, dh, new_hash_keys, new_hash_vals, vel_v)
    # w = sample_sparse_scalar(px, py, pz + 0.5 * dh, dh, new_hash_keys, new_hash_vals, vel_w)


    # px_prev = px - u * dt
    # py_prev = py - v * dt
    # pz_prev = pz - w * dt
    # 0403 test
    #px_prev = wp.max(px_prev, 0.5 * dh)

    #0403
    u_pos = clamp_u_pos(px + 0.5 * dh, py, pz, dh, nx, ny, nz)
    v_pos = clamp_v_pos(px, py + 0.5 * dh, pz, dh, nx, ny, nz)
    w_pos = clamp_w_pos(px, py, pz + 0.5 * dh, dh, nx, ny, nz)

    u = sample_sparse_scalar(u_pos[0], u_pos[1], u_pos[2], dh, new_hash_keys, new_hash_vals, vel_u)
    v = sample_sparse_scalar(v_pos[0], v_pos[1], v_pos[2], dh, new_hash_keys, new_hash_vals, vel_v)
    w = sample_sparse_scalar(w_pos[0], w_pos[1], w_pos[2], dh, new_hash_keys, new_hash_vals, vel_w)

    prev_pos = clamp_center_pos(px - u * dt, py - v * dt, pz - w * dt, dh, nx, ny, nz)

    density_out[wp.tid()] = sample_sparse_scalar(
        prev_pos[0], prev_pos[1], prev_pos[2],
        dh,
        old_hash_keys,
        old_hash_vals,
        density_in,
    )

    #density_out[wp.tid()] = sample_sparse_scalar(px_prev, py_prev, pz_prev, dh, old_hash_keys, old_hash_vals, density_in)


@wp.func
def get_sparse_val(ix: int, iy: int, iz: int, hash_keys: wp.array(dtype=int), hash_vals: wp.array(dtype=int), data: wp.array(dtype=float)) -> float:
    idx = get_sparse_idx(ix, iy, iz, hash_keys, hash_vals)
    if idx >= 0:
        return data[idx]
    return 0.0


@wp.func
def get_sparse_vec3(ix: int, iy: int, iz: int, hash_keys: wp.array(dtype=int), hash_vals: wp.array(dtype=int),
                    data: wp.array(dtype=wp.vec3)) -> wp.vec3:
    idx = get_sparse_idx(ix, iy, iz, hash_keys, hash_vals)
    if idx >= 0:
        return data[idx]
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def sparse_cell_weight(ix: int, iy: int, iz: int, hash_keys: wp.array(dtype=int), hash_vals: wp.array(dtype=int)) -> float:
    idx = get_sparse_idx(ix, iy, iz, hash_keys, hash_vals)
    if idx >= 0:
        return 1.0
    return 0.0


@wp.func
def sample_cell_velocity_sparse(
        ix: int,
        iy: int,
        iz: int,
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
) -> wp.vec3:
    return wp.vec3(
        0.5 * (get_sparse_val(ix, iy, iz, hash_keys, hash_vals, vel_u) +
               get_sparse_val(ix + 1, iy, iz, hash_keys, hash_vals, vel_u)),
        0.5 * (get_sparse_val(ix, iy, iz, hash_keys, hash_vals, vel_v) +
               get_sparse_val(ix, iy + 1, iz, hash_keys, hash_vals, vel_v)),
        0.5 * (get_sparse_val(ix, iy, iz, hash_keys, hash_vals, vel_w) +
               get_sparse_val(ix, iy, iz + 1, hash_keys, hash_vals, vel_w)),
    )


@wp.func
def sample_curl_sparse(
        ix: int,
        iy: int,
        iz: int,
        curl: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
) -> wp.vec3:
    return get_sparse_vec3(ix, iy, iz, hash_keys, hash_vals, curl)


@wp.func
def compute_vorticity_force_sparse(
        ix: int, iy: int, iz: int,
        curl: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float, vorticity_scale: float
) -> wp.vec3:
    idx = get_sparse_idx(ix, iy, iz, hash_keys, hash_vals)
    if idx < 0:
        return wp.vec3(0.0, 0.0, 0.0)

    curl_center = curl[idx]

    idx_xp = get_sparse_idx(ix + 1, iy, iz, hash_keys, hash_vals)
    curl_mag_xp = float(0.0)
    if idx_xp >= 0:
        curl_mag_xp = wp.length(curl[idx_xp])

    idx_xm = get_sparse_idx(ix - 1, iy, iz, hash_keys, hash_vals)
    curl_mag_xm = float(0.0)
    if idx_xm >= 0:
        curl_mag_xm = wp.length(curl[idx_xm])

    idx_yp = get_sparse_idx(ix, iy + 1, iz, hash_keys, hash_vals)
    curl_mag_yp = float(0.0)
    if idx_yp >= 0:
        curl_mag_yp = wp.length(curl[idx_yp])

    idx_ym = get_sparse_idx(ix, iy - 1, iz, hash_keys, hash_vals)
    curl_mag_ym = float(0.0)
    if idx_ym >= 0:
        curl_mag_ym = wp.length(curl[idx_ym])

    idx_zp = get_sparse_idx(ix, iy, iz + 1, hash_keys, hash_vals)
    curl_mag_zp = float(0.0)
    if idx_zp >= 0:
        curl_mag_zp = wp.length(curl[idx_zp])

    idx_zm = get_sparse_idx(ix, iy, iz - 1, hash_keys, hash_vals)
    curl_mag_zm = float(0.0)
    if idx_zm >= 0:
        curl_mag_zm = wp.length(curl[idx_zm])

    grad_mag = wp.vec3(
        (curl_mag_xp - curl_mag_xm) / (2.0 * dh),
        (curl_mag_yp - curl_mag_ym) / (2.0 * dh),
        (curl_mag_zp - curl_mag_zm) / (2.0 * dh),
    )

    grad_len = wp.length(grad_mag)
    if grad_len < 1.0e-6:
        return wp.vec3(0.0, 0.0, 0.0)

    normal = grad_mag / grad_len
    return vorticity_scale * dh * wp.cross(normal, curl_center)

@wp.kernel
def compute_curl_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        curl_out: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float,
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]

    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE

    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    vel_xp = sample_cell_velocity_sparse(gx + 1, gy, gz, vel_u, vel_v, vel_w, hash_keys, hash_vals)
    vel_xm = sample_cell_velocity_sparse(gx - 1, gy, gz, vel_u, vel_v, vel_w, hash_keys, hash_vals)
    vel_yp = sample_cell_velocity_sparse(gx, gy + 1, gz, vel_u, vel_v, vel_w, hash_keys, hash_vals)
    vel_ym = sample_cell_velocity_sparse(gx, gy - 1, gz, vel_u, vel_v, vel_w, hash_keys, hash_vals)
    vel_zp = sample_cell_velocity_sparse(gx, gy, gz + 1, vel_u, vel_v, vel_w, hash_keys, hash_vals)
    vel_zm = sample_cell_velocity_sparse(gx, gy, gz - 1, vel_u, vel_v, vel_w, hash_keys, hash_vals)

    du_dy = (vel_yp[0] - vel_ym[0]) / (2.0 * dh)
    du_dz = (vel_zp[0] - vel_zm[0]) / (2.0 * dh)
    dv_dx = (vel_xp[1] - vel_xm[1]) / (2.0 * dh)
    dv_dz = (vel_zp[1] - vel_zm[1]) / (2.0 * dh)
    dw_dx = (vel_xp[2] - vel_xm[2]) / (2.0 * dh)
    dw_dy = (vel_yp[2] - vel_ym[2]) / (2.0 * dh)

    curl_out[wp.tid()] = wp.vec3(
        dw_dy - dv_dz,
        du_dz - dw_dx,
        dv_dx - du_dy,
    )


@wp.kernel
def apply_vorticity_u_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        vel_u: wp.array(dtype=float),
        curl: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float,
        dt: float,
        vorticity_scale: float,
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]

    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE

    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    force_sum = 0.0
    weight_sum = 0.0

    left_weight = sparse_cell_weight(gx - 1, gy, gz, hash_keys, hash_vals)
    if left_weight > 0.0:
        force_sum += compute_vorticity_force_sparse(gx - 1, gy, gz, curl, hash_keys, hash_vals, dh, vorticity_scale)[0]
        weight_sum += left_weight

    right_weight = sparse_cell_weight(gx, gy, gz, hash_keys, hash_vals)
    if right_weight > 0.0:
        force_sum += compute_vorticity_force_sparse(gx, gy, gz, curl, hash_keys, hash_vals, dh, vorticity_scale)[0]
        weight_sum += right_weight

    if weight_sum > 0.0:
        vel_u[wp.tid()] += dt * force_sum / weight_sum


@wp.kernel
def apply_vorticity_v_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        vel_v: wp.array(dtype=float),
        curl: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float,
        dt: float,
        vorticity_scale: float,
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]

    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE

    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    force_sum = 0.0
    weight_sum = 0.0

    down_weight = sparse_cell_weight(gx, gy - 1, gz, hash_keys, hash_vals)
    if down_weight > 0.0:
        force_sum += compute_vorticity_force_sparse(gx, gy - 1, gz, curl, hash_keys, hash_vals, dh, vorticity_scale)[1]
        weight_sum += down_weight

    up_weight = sparse_cell_weight(gx, gy, gz, hash_keys, hash_vals)
    if up_weight > 0.0:
        force_sum += compute_vorticity_force_sparse(gx, gy, gz, curl, hash_keys, hash_vals, dh, vorticity_scale)[1]
        weight_sum += up_weight

    if weight_sum > 0.0:
        vel_v[wp.tid()] += dt * force_sum / weight_sum


@wp.kernel
def apply_vorticity_w_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        vel_w: wp.array(dtype=float),
        curl: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float,
        dt: float,
        vorticity_scale: float,
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]

    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE

    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    force_sum = 0.0
    weight_sum = 0.0

    back_weight = sparse_cell_weight(gx, gy, gz - 1, hash_keys, hash_vals)
    if back_weight > 0.0:
        force_sum += compute_vorticity_force_sparse(gx, gy, gz - 1, curl, hash_keys, hash_vals, dh, vorticity_scale)[2]
        weight_sum += back_weight

    front_weight = sparse_cell_weight(gx, gy, gz, hash_keys, hash_vals)
    if front_weight > 0.0:
        force_sum += compute_vorticity_force_sparse(gx, gy, gz, curl, hash_keys, hash_vals, dh, vorticity_scale)[2]
        weight_sum += front_weight

    if weight_sum > 0.0:
        vel_w[wp.tid()] += dt * force_sum / weight_sum


@wp.kernel
def compute_divergence_sparse(
    block_coords: wp.array(dtype=wp.vec3i),
    vel_u: wp.array(dtype=float),
    vel_v: wp.array(dtype=float),
    vel_w: wp.array(dtype=float),
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
    dh: float,
    div_out: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    u_left = get_sparse_val(gx, gy, gz, hash_keys, hash_vals, vel_u)
    if gx == 0: u_left = 0.0

    u_right = get_sparse_val(gx + 1, gy, gz, hash_keys, hash_vals, vel_u)
    if gx == nx - 1: u_right = 0.0

    v_down = get_sparse_val(gx, gy, gz, hash_keys, hash_vals, vel_v)
    if gy == 0: v_down = 0.0

    v_up = get_sparse_val(gx, gy + 1, gz, hash_keys, hash_vals, vel_v)
    if gy == ny - 1: v_up = 0.0

    w_back = get_sparse_val(gx, gy, gz, hash_keys, hash_vals, vel_w)
    if gz == 0: w_back = 0.0

    w_front = get_sparse_val(gx, gy, gz + 1, hash_keys, hash_vals, vel_w)
    if gz == nz - 1: w_front = 0.0

    div_out[wp.tid()] = (u_right - u_left + v_up - v_down + w_front - w_back) / dh


@wp.kernel
def pressure_jacobi_sparse(
    block_coords: wp.array(dtype=wp.vec3i),
    p_in: wp.array(dtype=float),
    p_out: wp.array(dtype=float),
    div_in: wp.array(dtype=float),
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
    dh: float,
    nx: int,
    ny: int,
    nz: int
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    p_center = p_in[wp.tid()]

    idx_left = get_sparse_idx(gx - 1, gy, gz, hash_keys, hash_vals)
    p_left = p_in[idx_left] if idx_left >= 0 else 0.0

    idx_right = get_sparse_idx(gx + 1, gy, gz, hash_keys, hash_vals)
    p_right = p_in[idx_right] if idx_right >= 0 else 0.0

    idx_down = get_sparse_idx(gx, gy - 1, gz, hash_keys, hash_vals)
    p_down = p_in[idx_down] if idx_down >= 0 else 0.0

    idx_up = get_sparse_idx(gx, gy + 1, gz, hash_keys, hash_vals)
    p_up = p_in[idx_up] if idx_up >= 0 else 0.0

    idx_back = get_sparse_idx(gx, gy, gz - 1, hash_keys, hash_vals)
    p_back = p_in[idx_back] if idx_back >= 0 else 0.0

    idx_front = get_sparse_idx(gx, gy, gz + 1, hash_keys, hash_vals)
    p_front = p_in[idx_front] if idx_front >= 0 else 0.0

    # 只有绝对的模拟框边界，才是不可穿越的墙壁 (Neumann 边界)
    if gx == 0: p_left = p_center
    if gx == nx - 1: p_right = p_center
    if gy == 0: p_down = p_center
    if gy == ny - 1: p_up = p_center
    if gz == 0: p_back = p_center
    if gz == nz - 1: p_front = p_center

    p_out[wp.tid()] = (p_left + p_right + p_down + p_up + p_front + p_back - dh * dh * div_in[wp.tid()]) / 6.0

# 0403 fix boundary & add nx ny nz params
@wp.kernel
def project_velocity_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        pressure: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float,
        nx: int,
        ny: int,
        nz: int
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    p_center = get_sparse_val(gx, gy, gz, hash_keys, hash_vals, pressure)

    # 允许流体向0压强的未分配区域喷射
    idx_left = get_sparse_idx(gx - 1, gy, gz, hash_keys, hash_vals)
    p_left = pressure[idx_left] if idx_left >= 0 else 0.0

    idx_down = get_sparse_idx(gx, gy - 1, gz, hash_keys, hash_vals)
    p_down = pressure[idx_down] if idx_down >= 0 else 0.0

    idx_back = get_sparse_idx(gx, gy, gz - 1, hash_keys, hash_vals)
    p_back = pressure[idx_back] if idx_back >= 0 else 0.0

    vel_u[wp.tid()] -= (p_center - p_left) / dh
    vel_v[wp.tid()] -= (p_center - p_down) / dh
    vel_w[wp.tid()] -= (p_center - p_back) / dh

    if gx == 0: vel_u[wp.tid()] = 0.0
    if gy == 0: vel_v[wp.tid()] = 0.0
    if gz == 0: vel_w[wp.tid()] = 0.0

    # 天花板
    if gz == nz - 1 and vel_w[wp.tid()] > 0.0:
        vel_w[wp.tid()] = 0.0


# 330 add offset
# 0403 fix boundary & add nx ny nz params
@wp.kernel
def advect_u_sparse(
    block_coords: wp.array(dtype=wp.vec3i),
    vel_u_in: wp.array(dtype=float),
    vel_v_in: wp.array(dtype=float),
    vel_w_in: wp.array(dtype=float),
    vel_u_out: wp.array(dtype=float),
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
    dt: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = wp.float32(gx) * dh
    py = (wp.float32(gy) + 0.5) * dh
    pz = (wp.float32(gz) + 0.5) * dh

    # 采样速度
    # u = sample_sparse_scalar(px + 0.5 * dh, py, pz, dh, hash_keys, hash_vals, vel_u_in)
    # v = sample_sparse_scalar(px, py + 0.5 * dh, pz, dh, hash_keys, hash_vals, vel_v_in)
    # w = sample_sparse_scalar(px, py, pz + 0.5 * dh, dh, hash_keys, hash_vals, vel_w_in)

    # px_prev = px - u * dt
    # py_prev = py - v * dt
    # pz_prev = pz - w * dt

    # vel_u_out[wp.tid()] = sample_sparse_scalar(px_prev + 0.5 * dh, py_prev, pz_prev, dh, hash_keys, hash_vals, vel_u_in)

    #0403
    u_pos = clamp_u_pos(px + 0.5 * dh, py, pz, dh, nx, ny, nz)
    v_pos = clamp_v_pos(px, py + 0.5 * dh, pz, dh, nx, ny, nz)
    w_pos = clamp_w_pos(px, py, pz + 0.5 * dh, dh, nx, ny, nz)

    u = sample_sparse_scalar(u_pos[0], u_pos[1], u_pos[2], dh, hash_keys, hash_vals, vel_u_in)
    v = sample_sparse_scalar(v_pos[0], v_pos[1], v_pos[2], dh, hash_keys, hash_vals, vel_v_in)
    w = sample_sparse_scalar(w_pos[0], w_pos[1], w_pos[2], dh, hash_keys, hash_vals, vel_w_in)

    prev_u_pos = clamp_u_pos(px - u * dt + 0.5 * dh, py - v * dt, pz - w * dt, dh, nx, ny, nz)

    vel_u_out[wp.tid()] = sample_sparse_scalar(
        prev_u_pos[0], prev_u_pos[1], prev_u_pos[2],
        dh,
        hash_keys,
        hash_vals,
        vel_u_in,
    )



@wp.kernel
def advect_v_sparse(
    block_coords: wp.array(dtype=wp.vec3i),
    vel_u_in: wp.array(dtype=float),
    vel_v_in: wp.array(dtype=float),
    vel_w_in: wp.array(dtype=float),
    vel_v_out: wp.array(dtype=float),
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
    dt: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = (wp.float32(gx) + 0.5) * dh
    py = wp.float32(gy) * dh
    pz = (wp.float32(gz) + 0.5) * dh

    # u = sample_sparse_scalar(px + 0.5 * dh, py, pz, dh, hash_keys, hash_vals, vel_u_in)
    # v = sample_sparse_scalar(px, py + 0.5 * dh, pz, dh, hash_keys, hash_vals, vel_v_in)
    # w = sample_sparse_scalar(px, py, pz + 0.5 * dh, dh, hash_keys, hash_vals, vel_w_in)

    # px_prev = px - u * dt
    # py_prev = py - v * dt
    # pz_prev = pz - w * dt

    # vel_v_out[wp.tid()] = sample_sparse_scalar(px_prev, py_prev + 0.5 * dh, pz_prev, dh, hash_keys, hash_vals, vel_v_in)

    #0403
    u_pos = clamp_u_pos(px + 0.5 * dh, py, pz, dh, nx, ny, nz)
    v_pos = clamp_v_pos(px, py + 0.5 * dh, pz, dh, nx, ny, nz)
    w_pos = clamp_w_pos(px, py, pz + 0.5 * dh, dh, nx, ny, nz)

    u = sample_sparse_scalar(u_pos[0], u_pos[1], u_pos[2], dh, hash_keys, hash_vals, vel_u_in)
    v = sample_sparse_scalar(v_pos[0], v_pos[1], v_pos[2], dh, hash_keys, hash_vals, vel_v_in)
    w = sample_sparse_scalar(w_pos[0], w_pos[1], w_pos[2], dh, hash_keys, hash_vals, vel_w_in)

    prev_v_pos = clamp_v_pos(px - u * dt, py - v * dt + 0.5 * dh, pz - w * dt, dh, nx, ny, nz)

    vel_v_out[wp.tid()] = sample_sparse_scalar(
        prev_v_pos[0], prev_v_pos[1], prev_v_pos[2],
        dh,
        hash_keys,
        hash_vals,
        vel_v_in,
    )

@wp.kernel
def advect_w_sparse(
    block_coords: wp.array(dtype=wp.vec3i),
    vel_u_in: wp.array(dtype=float),
    vel_v_in: wp.array(dtype=float),
    vel_w_in: wp.array(dtype=float),
    vel_w_out: wp.array(dtype=float),
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
    dt: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]
    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE
    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = (wp.float32(gx) + 0.5) * dh
    py = (wp.float32(gy) + 0.5) * dh
    pz = wp.float32(gz) * dh

    # u = sample_sparse_scalar(px + 0.5 * dh, py, pz, dh, hash_keys, hash_vals, vel_u_in)
    # v = sample_sparse_scalar(px, py + 0.5 * dh, pz, dh, hash_keys, hash_vals, vel_v_in)
    # w = sample_sparse_scalar(px, py, pz + 0.5 * dh, dh, hash_keys, hash_vals, vel_w_in)


    # px_prev = px - u * dt
    # py_prev = py - v * dt
    # pz_prev = pz - w * dt

    # vel_w_out[wp.tid()] = sample_sparse_scalar(px_prev, py_prev, pz_prev + 0.5 * dh, dh, hash_keys, hash_vals, vel_w_in)

    #0403
    u_pos = clamp_u_pos(px + 0.5 * dh, py, pz, dh, nx, ny, nz)
    v_pos = clamp_v_pos(px, py + 0.5 * dh, pz, dh, nx, ny, nz)
    w_pos = clamp_w_pos(px, py, pz + 0.5 * dh, dh, nx, ny, nz)

    u = sample_sparse_scalar(u_pos[0], u_pos[1], u_pos[2], dh, hash_keys, hash_vals, vel_u_in)
    v = sample_sparse_scalar(v_pos[0], v_pos[1], v_pos[2], dh, hash_keys, hash_vals, vel_v_in)
    w = sample_sparse_scalar(w_pos[0], w_pos[1], w_pos[2], dh, hash_keys, hash_vals, vel_w_in)

    prev_w_pos = clamp_w_pos(px - u * dt, py - v * dt, pz - w * dt + 0.5 * dh, dh, nx, ny, nz)

    vel_w_out[wp.tid()] = sample_sparse_scalar(
        prev_w_pos[0], prev_w_pos[1], prev_w_pos[2],
        dh,
        hash_keys,
        hash_vals,
        vel_w_in,
    )



@wp.kernel
def apply_force_sparse(
    vel: wp.array(dtype=float),
    force: float,
    dt: float
):
    vel[wp.tid()] += force * dt

@wp.kernel
def enforce_solid_velocity_sparse(
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        solid_phi: wp.array(dtype=float)
):
    tid = wp.tid()
    if solid_phi[tid] < 0.0:
        vel_u[tid] = 0.0
        vel_v[tid] = 0.0
        vel_w[tid] = 0.0

@wp.kernel
def dissipate_scalar_sparse(
    field: wp.array(dtype=float),
    rate: float,
    dt: float
):
    tid = wp.tid()
    decay = wp.max(0.0, 1.0 - rate * dt)
    field[tid] = field[tid] * decay


# MODIFIED: prune now keeps blocks primarily based on actual dense occupancy,
# while velocity only acts as a secondary support signal for nearby transport.
@wp.kernel
def prune_blocks_sparse(
        old_coords: wp.array(dtype=wp.vec3i),
        density: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        new_coords: wp.array(dtype=wp.vec3i),
        new_count: wp.array(dtype=int)
):
    block_id = wp.tid()
    dense_count = int(0)
    support_speed_count = int(0)
    max_density = float(0.0)
    max_speed = float(0.0)
    for i in range(512):
        idx = block_id * 512 + i
        d = density[idx]
        u = wp.abs(vel_u[idx])
        v = wp.abs(vel_v[idx])
        w = wp.abs(vel_w[idx])
        speed = wp.max(wp.max(u, v), w)

        max_density = wp.max(max_density, d)
        max_speed = wp.max(max_speed, speed)

        if d > 8.0e-3:
            dense_count += 1
        if d > 2.0e-3 and speed > 1.2e-1:
            support_speed_count += 1

    if max_density > 1.0e-4 or max_speed > 1.0e-2:
        idx = wp.atomic_add(new_count, 0, 1)
        new_coords[idx] = old_coords[block_id]

@wp.kernel
def update_particle_colors_sparse(
        positions: wp.array(dtype=wp.vec3),
        density: wp.array(dtype=float),
        colors: wp.array(dtype=wp.vec3),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float
):
    tid = wp.tid()
    p = positions[tid]

    d = sample_sparse_scalar(p[0], p[1], p[2], dh, hash_keys, hash_vals, density)
    val = wp.min(d * 2.0, 1.0)
    colors[tid] = wp.vec3(val * 0.8, val * 0.9, val * 1.0)


@wp.kernel
def advect_particles_sparse(
    positions: wp.array(dtype=wp.vec3),
    vel_u: wp.array(dtype=float),
    vel_v: wp.array(dtype=float),
    vel_w: wp.array(dtype=float),
    density: wp.array(dtype=float),  # 读取密度场
    hash_keys: wp.array(dtype=int),
    hash_vals: wp.array(dtype=int),
    dh: float,
    dt: float,
    offset: float,
    sim_time: float,
    grid_size_z: float,
    grid_size_x: float
):
    tid = wp.tid()
    p = positions[tid]

    sp_x = p[0] + offset
    sp_y = p[1] + offset
    sp_z = p[2]

    u = sample_sparse_scalar(sp_x + 0.5 * dh, sp_y, sp_z, dh, hash_keys, hash_vals, vel_u)
    v = sample_sparse_scalar(sp_x, sp_y + 0.5 * dh, sp_z, dh, hash_keys, hash_vals, vel_v)
    w = sample_sparse_scalar(sp_x, sp_y, sp_z + 0.5 * dh, dh, hash_keys, hash_vals, vel_w)

    new_x = p[0] + u * dt
    new_y = p[1] + v * dt
    new_z = p[2] + w * dt

    # 取消所有传送，严格还原稠密版本的纯物理卡边界
    new_x = wp.clamp(new_x, -offset + 0.1, grid_size_x - offset - 0.1)
    new_y = wp.clamp(new_y, -offset + 0.1, grid_size_x - offset - 0.1)
    new_z = wp.clamp(new_z, 0.1, grid_size_z - 0.1)

    positions[tid] = wp.vec3(new_x, new_y, new_z)


@wp.kernel
def update_particle_visuals_sparse(
        positions: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=float),
        density: wp.array(dtype=float),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        dh: float,
        offset: float,
        base_radius: float,
        threshold: float
):
    tid = wp.tid()
    p = positions[tid]

    sp_x = p[0] + offset
    sp_y = p[1] + offset
    sp_z = p[2]

    d = sample_sparse_scalar(sp_x, sp_y, sp_z, dh, hash_keys, hash_vals, density)

    min_d = threshold
    max_d = threshold * 4.0
    t = wp.clamp((d - min_d) / (max_d - min_d), 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)

    if alpha < 0.01:
        radii[tid] = 0.0
        colors[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        radii[tid] = base_radius * alpha
        val = 0.2 + 0.8 * alpha
        colors[tid] = wp.vec3(val, val, val)
#
#
# @wp.kernel
# def update_particle_visuals_sparse(
#         positions: wp.array(dtype=wp.vec3),
#         colors: wp.array(dtype=wp.vec3),
#         radii: wp.array(dtype=float),
#         density: wp.array(dtype=float),
#         hash_keys: wp.array(dtype=int),
#         hash_vals: wp.array(dtype=int),
#         dh: float,
#         offset: float,
#         base_radius: float,
#         threshold: float  # <--- 新增：原版代码用来控制烟雾可见度的阈值
# ):
#     tid = wp.tid()
#     p = positions[tid]
#
#     sp_x = p[0] + offset
#     sp_y = p[1] + offset
#     sp_z = p[2]
#
#     # 查表获取密度
#     d = sample_sparse_scalar(sp_x, sp_y, sp_z, dh, hash_keys, hash_vals, density)
#
#     # =========================================================
#     # 完美复刻你原来的均匀网格 Smoothstep 渲染逻辑！
#     # =========================================================
#     min_d = threshold
#     max_d = threshold * 4.0
#     t = wp.clamp((d - min_d) / (max_d - min_d), 0.0, 1.0)
#     alpha = t * t * (3.0 - 2.0 * t)
#
#     if alpha < 0.01:
#         radii[tid] = 0.0
#         colors[tid] = wp.vec3(0.0, 0.0, 0.0)
#     else:
#         radii[tid] = base_radius * alpha
#         val = 0.2 + 0.8 * alpha
#         colors[tid] = wp.vec3(val, val, val)  # 还原成灰白色的烟雾！

#330 add
@wp.kernel
def generate_block_wireframes_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        starts: wp.array(dtype=wp.vec3),
        ends: wp.array(dtype=wp.vec3),
        block_phys_size: float,
        offset: float
):
    tid = wp.tid()
    coord = block_coords[tid]

    # 计算 Block 左下角的物理世界坐标 (对齐渲染坐标系)
    bx = wp.float32(coord[0]) * block_phys_size - offset
    by = wp.float32(coord[1]) * block_phys_size - offset
    bz = wp.float32(coord[2]) * block_phys_size

    # 计算 8 个顶点的坐标
    p000 = wp.vec3(bx, by, bz)
    p100 = wp.vec3(bx + block_phys_size, by, bz)
    p010 = wp.vec3(bx, by + block_phys_size, bz)
    p110 = wp.vec3(bx + block_phys_size, by + block_phys_size, bz)
    p001 = wp.vec3(bx, by, bz + block_phys_size)
    p101 = wp.vec3(bx + block_phys_size, by, bz + block_phys_size)
    p011 = wp.vec3(bx, by + block_phys_size, bz + block_phys_size)
    p111 = wp.vec3(bx + block_phys_size, by + block_phys_size, bz + block_phys_size)

    # 每个 Block 需要画 12 条线
    line_idx = tid * 12

    # 底面 4 条边
    starts[line_idx + 0] = p000; ends[line_idx + 0] = p100
    starts[line_idx + 1] = p100; ends[line_idx + 1] = p110
    starts[line_idx + 2] = p110; ends[line_idx + 2] = p010
    starts[line_idx + 3] = p010; ends[line_idx + 3] = p000

    # 顶面 4 条边
    starts[line_idx + 4] = p001; ends[line_idx + 4] = p101
    starts[line_idx + 5] = p101; ends[line_idx + 5] = p111
    starts[line_idx + 6] = p111; ends[line_idx + 6] = p011
    starts[line_idx + 7] = p011; ends[line_idx + 7] = p001

    # 垂直的 4 条柱子
    starts[line_idx + 8] = p000; ends[line_idx + 8] = p001
    starts[line_idx + 9] = p100; ends[line_idx + 9] = p101
    starts[line_idx + 10] = p110; ends[line_idx + 10] = p111
    starts[line_idx + 11] = p010; ends[line_idx + 11] = p011

@wp.kernel
def apply_buoyancy_sparse(
    vel_w: wp.array(dtype=float),
    density: wp.array(dtype=float),
    buoyancy: float,
    dt: float
):
    tid = wp.tid()
    vel_w[tid] += density[tid] * buoyancy * dt

@wp.kernel
def damp_velocity_sparse(
    vel_u: wp.array(dtype=float),
    vel_v: wp.array(dtype=float),
    vel_w: wp.array(dtype=float),
    damping: float,
    dt: float
):
    tid = wp.tid()
    decay = wp.max(0.0, 1.0 - damping * dt)
    vel_u[tid] = vel_u[tid] * decay
    vel_v[tid] = vel_v[tid] * decay
    vel_w[tid] = vel_w[tid] * decay

@wp.func
def evaluate_source_dense(gx: int, gy: int, gz: int, n_grid: int) -> float:
    cx = wp.float32(n_grid) / 2.0
    cy = wp.float32(n_grid) / 2.0
    x = wp.float32(gx)
    y = wp.float32(gy)
    z = wp.float32(gz)
    source_radius = wp.float32(n_grid) * 0.15
    source_height = wp.float32(n_grid) * 0.08

    dx = x - cx
    dy = y - cy
    r = wp.sqrt(dx * dx + dy * dy)

    if r < source_radius and z < source_height:
        dist_norm = r / source_radius
        return 1.0 - wp.smoothstep(0.0, 1.0, dist_norm)
    return 0.0