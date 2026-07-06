import warp as wp

from ..base_kernels import (
    advect_u_mac as advect_u,
    advect_v_mac as advect_v,
    advect_w_mac as advect_w,
    bake_solid_box_kernel,
    bake_solid_mesh_kernel,
    divergence_cell_with_solid_bc,
    enforce_solid_u_mac as enforce_solid_u,
    enforce_solid_v_mac as enforce_solid_v,
    enforce_solid_w_mac as enforce_solid_w,
    sample_scalar_trilinear as sample_float_trilinear,
    sample_velocity_mac as sample_mac_velocity,
)


CELL_AIR = wp.constant(0)
CELL_FLUID = wp.constant(1)
CELL_SOLID = wp.constant(2)


@wp.kernel
def fill_grid(grid: wp.array3d(dtype=float), val: float):
    i, j, k = wp.tid()
    grid[i, j, k] = val


@wp.kernel
def fill_grid_face(grid: wp.array3d(dtype=float), val: float):
    i, j, k = wp.tid()
    grid[i, j, k] = val

@wp.func
def voxel_open_face(phi0: float, phi1: float) -> float:
    # Treat solids as binary voxels throughout the solve: if either adjacent
    # cell is solid, the MAC face is closed.
    if phi0 < 0.0 or phi1 < 0.0:
        return 0.0
    return 1.0


@wp.func
def sample_centered_scalar_world(
    grid: wp.array3d(dtype=float),
    p_world: wp.vec3,
    dh: float,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    return sample_float_trilinear(
        grid,
        wp.vec3(p_world[0] / dh - 0.5, p_world[1] / dh - 0.5, p_world[2] / dh - 0.5),
        nx,
        ny,
        nz,
    )

@wp.kernel
def build_u_face_weights(
    solid_phi: wp.array3d(dtype=float),
    face_weight: wp.array3d(dtype=float),
    nx: int,
):
    i, j, k = wp.tid()
    if i == 0 or i == nx:
        face_weight[i, j, k] = 0.0
        return
    face_weight[i, j, k] = voxel_open_face(solid_phi[i - 1, j, k], solid_phi[i, j, k])


@wp.kernel
def build_v_face_weights(
    solid_phi: wp.array3d(dtype=float),
    face_weight: wp.array3d(dtype=float),
    ny: int,
):
    i, j, k = wp.tid()
    if j == 0 or j == ny:
        face_weight[i, j, k] = 0.0
        return
    face_weight[i, j, k] = voxel_open_face(solid_phi[i, j - 1, k], solid_phi[i, j, k])


@wp.kernel
def build_w_face_weights(
    solid_phi: wp.array3d(dtype=float),
    face_weight: wp.array3d(dtype=float),
    nz: int,
):
    i, j, k = wp.tid()
    if k == 0 or k == nz:
        face_weight[i, j, k] = 0.0
        return
    face_weight[i, j, k] = voxel_open_face(solid_phi[i, j, k - 1], solid_phi[i, j, k])


@wp.kernel
def initialize_liquid_cell_state(
    solid_phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    density: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    density[i, j, k] = 0.0

    if solid_phi[i, j, k] < 0.0:
        cell_type[i, j, k] = wp.int32(CELL_SOLID)
    else:
        cell_type[i, j, k] = wp.int32(CELL_AIR)


@wp.kernel
def mark_liquid_cells_from_particles(
    particles: wp.array(dtype=wp.vec3),
    solid_phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    density: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    pid = wp.tid()
    p = particles[pid]

    i = int(wp.floor(p[0] / dh))
    j = int(wp.floor(p[1] / dh))
    k = int(wp.floor(p[2] / dh))

    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return
    if solid_phi[i, j, k] < 0.0:
        return

    cell_type[i, j, k] = wp.int32(CELL_FLUID)
    wp.atomic_add(density, i, j, k, 1.0)

@wp.kernel
def p2g_velocity_u_solid_phi(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    grid_u: wp.array3d(dtype=float),
    weight_u: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_v[tid]

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
                    if i == 0 or i == nx:
                        continue
                    if solid_phi[i - 1, j, k] < 0.0 or solid_phi[i, j, k] < 0.0:
                        continue
                    wx = (1.0 - fx) if di == 0 else fx
                    wy = (1.0 - fy) if dj == 0 else fy
                    wz = (1.0 - fz) if dk == 0 else fz
                    w = wx * wy * wz
                    wp.atomic_add(grid_u, i, j, k, v[0] * w)
                    wp.atomic_add(weight_u, i, j, k, w)

@wp.kernel
def p2g_velocity_v_solid_phi(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    grid_v: wp.array3d(dtype=float),
    weight_v: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_v[tid]

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
                    if j == 0 or j == ny:
                        continue
                    if solid_phi[i, j - 1, k] < 0.0 or solid_phi[i, j, k] < 0.0:
                        continue
                    wx = (1.0 - fx) if di == 0 else fx
                    wy = (1.0 - fy) if dj == 0 else fy
                    wz = (1.0 - fz) if dk == 0 else fz
                    w = wx * wy * wz
                    wp.atomic_add(grid_v, i, j, k, v[1] * w)
                    wp.atomic_add(weight_v, i, j, k, w)

@wp.kernel
def p2g_velocity_w_solid_phi(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    grid_w: wp.array3d(dtype=float),
    weight_w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_v[tid]

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
                    if k == 0 or k == nz:
                        continue
                    if solid_phi[i, j, k - 1] < 0.0 or solid_phi[i, j, k] < 0.0:
                        continue
                    wx = (1.0 - fx) if di == 0 else fx
                    wy = (1.0 - fy) if dj == 0 else fy
                    wz = (1.0 - fz) if dk == 0 else fz
                    w = wx * wy * wz
                    wp.atomic_add(grid_w, i, j, k, v[2] * w)
                    wp.atomic_add(weight_w, i, j, k, w)

@wp.kernel
def normalize_face_velocity_with_valid(
    vel: wp.array3d(dtype=float),
    weight: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.int32),
    solid_weight: wp.array3d(dtype=float),
    eps: float,
):
    i, j, k = wp.tid()
    face_frac = solid_weight[i, j, k]
    w = weight[i, j, k]
    if face_frac > 0.0 and w > eps:
        vel[i, j, k] = vel[i, j, k] / w
        valid[i, j, k] = wp.int32(1)
    else:
        vel[i, j, k] = 0.0
        valid[i, j, k] = wp.int32(0)

@wp.kernel
def normalize_u_face_velocity_with_valid_solid_phi(
    vel: wp.array3d(dtype=float),
    weight: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.uint8),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    eps: float,
):
    i, j, k = wp.tid()
    w = weight[i, j, k]
    if i != 0 and i != nx and solid_phi[i - 1, j, k] >= 0.0 and solid_phi[i, j, k] >= 0.0 and w > eps:
        vel[i, j, k] = vel[i, j, k] / w
        valid[i, j, k] = wp.uint8(1)
    else:
        vel[i, j, k] = 0.0
        valid[i, j, k] = wp.uint8(0)

@wp.kernel
def normalize_v_face_velocity_with_valid_solid_phi(
    vel: wp.array3d(dtype=float),
    weight: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.uint8),
    solid_phi: wp.array3d(dtype=float),
    ny: int,
    eps: float,
):
    i, j, k = wp.tid()
    w = weight[i, j, k]
    if j != 0 and j != ny and solid_phi[i, j - 1, k] >= 0.0 and solid_phi[i, j, k] >= 0.0 and w > eps:
        vel[i, j, k] = vel[i, j, k] / w
        valid[i, j, k] = wp.uint8(1)
    else:
        vel[i, j, k] = 0.0
        valid[i, j, k] = wp.uint8(0)

@wp.kernel
def normalize_w_face_velocity_with_valid_solid_phi(
    vel: wp.array3d(dtype=float),
    weight: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.uint8),
    solid_phi: wp.array3d(dtype=float),
    nz: int,
    eps: float,
):
    i, j, k = wp.tid()
    w = weight[i, j, k]
    if k != 0 and k != nz and solid_phi[i, j, k - 1] >= 0.0 and solid_phi[i, j, k] >= 0.0 and w > eps:
        vel[i, j, k] = vel[i, j, k] / w
        valid[i, j, k] = wp.uint8(1)
    else:
        vel[i, j, k] = 0.0
        valid[i, j, k] = wp.uint8(0)

@wp.kernel
def extrapolate_u_from_valid(
    u_in: wp.array3d(dtype=float),
    valid_in: wp.array3d(dtype=wp.int32),
    u_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.int32),
    vel_solid_u: wp.array3d(dtype=float),
    solid_weight_u: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if solid_weight_u[i, j, k] <= 0.0:
        # Preserve embedded rigid boundary velocity inside solid-covered faces
        # so velocity extrapolation can continue from moving boundaries.
        u_out[i, j, k] = vel_solid_u[i, j, k]
        valid_out[i, j, k] = wp.int32(1)
        return

    if valid_in[i, j, k] == 1:
        u_out[i, j, k] = u_in[i, j, k]
        valid_out[i, j, k] = wp.int32(1)
        return

    sum_v = 0.0
    count = wp.int32(0)

    if i > 0 and valid_in[i - 1, j, k] == 1:
        sum_v += u_in[i - 1, j, k]
        count += 1
    if i < nx and valid_in[i + 1, j, k] == 1:
        sum_v += u_in[i + 1, j, k]
        count += 1
    if j > 0 and valid_in[i, j - 1, k] == 1:
        sum_v += u_in[i, j - 1, k]
        count += 1
    if j < ny - 1 and valid_in[i, j + 1, k] == 1:
        sum_v += u_in[i, j + 1, k]
        count += 1
    if k > 0 and valid_in[i, j, k - 1] == 1:
        sum_v += u_in[i, j, k - 1]
        count += 1
    if k < nz - 1 and valid_in[i, j, k + 1] == 1:
        sum_v += u_in[i, j, k + 1]
        count += 1

    if count > 0:
        u_out[i, j, k] = sum_v / float(count)
        valid_out[i, j, k] = wp.int32(1)
    else:
        u_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.int32(0)


@wp.kernel
def extrapolate_u_from_valid_solid_phi(
    u_in: wp.array3d(dtype=float),
    valid_in: wp.array3d(dtype=wp.uint8),
    u_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.uint8),
    vel_solid_u: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if i == 0 or i == nx:
        u_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.uint8(1)
        return

    if solid_phi[i - 1, j, k] * solid_phi[i, j, k] < 0.0:
        u_out[i, j, k] = vel_solid_u[i, j, k]
        valid_out[i, j, k] = wp.uint8(1)
        return
    
    if solid_phi[i - 1, j, k] < 0.0 and solid_phi[i, j, k] < 0.0:
        return

    if valid_in[i, j, k] > wp.uint8(0):
        u_out[i, j, k] = u_in[i, j, k]
        valid_out[i, j, k] = wp.uint8(1)
        return

    sum_v = 0.0
    count = wp.int32(0)

    if i > 0 and valid_in[i - 1, j, k] > wp.uint8(0):
        sum_v += u_in[i - 1, j, k]
        count += 1
    if i < nx and valid_in[i + 1, j, k] > wp.uint8(0):
        sum_v += u_in[i + 1, j, k]
        count += 1
    if j > 0 and valid_in[i, j - 1, k] > wp.uint8(0):
        sum_v += u_in[i, j - 1, k]
        count += 1
    if j < ny - 1 and valid_in[i, j + 1, k] > wp.uint8(0):
        sum_v += u_in[i, j + 1, k]
        count += 1
    if k > 0 and valid_in[i, j, k - 1] > wp.uint8(0):
        sum_v += u_in[i, j, k - 1]
        count += 1
    if k < nz - 1 and valid_in[i, j, k + 1] > wp.uint8(0):
        sum_v += u_in[i, j, k + 1]
        count += 1

    if count > 0:
        u_out[i, j, k] = sum_v / float(count)
        valid_out[i, j, k] = wp.uint8(1)
    else:
        u_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.uint8(0)


@wp.kernel
def extrapolate_v_from_valid(
    v_in: wp.array3d(dtype=float),
    valid_in: wp.array3d(dtype=wp.int32),
    v_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.int32),
    vel_solid_v: wp.array3d(dtype=float),
    solid_weight_v: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if solid_weight_v[i, j, k] <= 0.0:
        v_out[i, j, k] = vel_solid_v[i, j, k]
        valid_out[i, j, k] = wp.int32(1)
        return

    if valid_in[i, j, k] == 1:
        v_out[i, j, k] = v_in[i, j, k]
        valid_out[i, j, k] = wp.int32(1)
        return

    sum_v = 0.0
    count = wp.int32(0)

    if i > 0 and valid_in[i - 1, j, k] == 1:
        sum_v += v_in[i - 1, j, k]
        count += 1
    if i < nx - 1 and valid_in[i + 1, j, k] == 1:
        sum_v += v_in[i + 1, j, k]
        count += 1
    if j > 0 and valid_in[i, j - 1, k] == 1:
        sum_v += v_in[i, j - 1, k]
        count += 1
    if j < ny and valid_in[i, j + 1, k] == 1:
        sum_v += v_in[i, j + 1, k]
        count += 1
    if k > 0 and valid_in[i, j, k - 1] == 1:
        sum_v += v_in[i, j, k - 1]
        count += 1
    if k < nz - 1 and valid_in[i, j, k + 1] == 1:
        sum_v += v_in[i, j, k + 1]
        count += 1

    if count > 0:
        v_out[i, j, k] = sum_v / float(count)
        valid_out[i, j, k] = wp.int32(1)
    else:
        v_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.int32(0)


@wp.kernel
def extrapolate_v_from_valid_solid_phi(
    v_in: wp.array3d(dtype=float),
    valid_in: wp.array3d(dtype=wp.uint8),
    v_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.uint8),
    vel_solid_v: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if j == 0 or j == ny:
        v_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.uint8(1)
        return

    if solid_phi[i, j - 1, k] * solid_phi[i, j, k] < 0.0:
        v_out[i, j, k] = vel_solid_v[i, j, k]
        valid_out[i, j, k] = wp.uint8(1)
        return
    
    if solid_phi[i, j - 1, k] < 0.0 and solid_phi[i, j, k] < 0.0:
        return

    if valid_in[i, j, k] > wp.uint8(0):
        v_out[i, j, k] = v_in[i, j, k]
        valid_out[i, j, k] = wp.uint8(1)
        return

    sum_v = 0.0
    count = wp.int32(0)

    if i > 0 and valid_in[i - 1, j, k] > wp.uint8(0):
        sum_v += v_in[i - 1, j, k]
        count += 1
    if i < nx - 1 and valid_in[i + 1, j, k] > wp.uint8(0):
        sum_v += v_in[i + 1, j, k]
        count += 1
    if j > 0 and valid_in[i, j - 1, k] > wp.uint8(0):
        sum_v += v_in[i, j - 1, k]
        count += 1
    if j < ny and valid_in[i, j + 1, k] > wp.uint8(0):
        sum_v += v_in[i, j + 1, k]
        count += 1
    if k > 0 and valid_in[i, j, k - 1] > wp.uint8(0):
        sum_v += v_in[i, j, k - 1]
        count += 1
    if k < nz - 1 and valid_in[i, j, k + 1] > wp.uint8(0):
        sum_v += v_in[i, j, k + 1]
        count += 1

    if count > 0:
        v_out[i, j, k] = sum_v / float(count)
        valid_out[i, j, k] = wp.uint8(1)
    else:
        v_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.uint8(0)


@wp.kernel
def extrapolate_w_from_valid(
    w_in: wp.array3d(dtype=float),
    valid_in: wp.array3d(dtype=wp.int32),
    w_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.int32),
    vel_solid_w: wp.array3d(dtype=float),
    solid_weight_w: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if solid_weight_w[i, j, k] <= 0.0:
        w_out[i, j, k] = vel_solid_w[i, j, k]
        valid_out[i, j, k] = wp.int32(1)
        return

    if valid_in[i, j, k] == 1:
        w_out[i, j, k] = w_in[i, j, k]
        valid_out[i, j, k] = wp.int32(1)
        return

    sum_v = 0.0
    count = wp.int32(0)

    if i > 0 and valid_in[i - 1, j, k] == 1:
        sum_v += w_in[i - 1, j, k]
        count += 1
    if i < nx - 1 and valid_in[i + 1, j, k] == 1:
        sum_v += w_in[i + 1, j, k]
        count += 1
    if j > 0 and valid_in[i, j - 1, k] == 1:
        sum_v += w_in[i, j - 1, k]
        count += 1
    if j < ny - 1 and valid_in[i, j + 1, k] == 1:
        sum_v += w_in[i, j + 1, k]
        count += 1
    if k > 0 and valid_in[i, j, k - 1] == 1:
        sum_v += w_in[i, j, k - 1]
        count += 1
    if k < nz and valid_in[i, j, k + 1] == 1:
        sum_v += w_in[i, j, k + 1]
        count += 1

    if count > 0:
        w_out[i, j, k] = sum_v / float(count)
        valid_out[i, j, k] = wp.int32(1)
    else:
        w_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.int32(0)


@wp.kernel
def extrapolate_w_from_valid_solid_phi(
    w_in: wp.array3d(dtype=float),
    valid_in: wp.array3d(dtype=wp.uint8),
    w_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.uint8),
    vel_solid_w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if k == 0 or k == nz:
        w_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.uint8(1)
        return

    if solid_phi[i, j, k - 1] * solid_phi[i, j, k] < 0.0:
        w_out[i, j, k] = vel_solid_w[i, j, k]
        valid_out[i, j, k] = wp.uint8(1)
        return
    
    if solid_phi[i, j, k - 1] < 0.0 and solid_phi[i, j, k] < 0.0:
        return

    if valid_in[i, j, k] > wp.uint8(0):
        w_out[i, j, k] = w_in[i, j, k]
        valid_out[i, j, k] = wp.uint8(1)
        return

    sum_v = 0.0
    count = wp.int32(0)

    if i > 0 and valid_in[i - 1, j, k] > wp.uint8(0):
        sum_v += w_in[i - 1, j, k]
        count += 1
    if i < nx - 1 and valid_in[i + 1, j, k] > wp.uint8(0):
        sum_v += w_in[i + 1, j, k]
        count += 1
    if j > 0 and valid_in[i, j - 1, k] > wp.uint8(0):
        sum_v += w_in[i, j - 1, k]
        count += 1
    if j < ny - 1 and valid_in[i, j + 1, k] > wp.uint8(0):
        sum_v += w_in[i, j + 1, k]
        count += 1
    if k > 0 and valid_in[i, j, k - 1] > wp.uint8(0):
        sum_v += w_in[i, j, k - 1]
        count += 1
    if k < nz and valid_in[i, j, k + 1] > wp.uint8(0):
        sum_v += w_in[i, j, k + 1]
        count += 1

    if count > 0:
        w_out[i, j, k] = sum_v / float(count)
        valid_out[i, j, k] = wp.uint8(1)
    else:
        w_out[i, j, k] = 0.0
        valid_out[i, j, k] = wp.uint8(0)


@wp.kernel
def update_particle_velocity_flip_pic(
    particle_q: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    vel_u_new: wp.array3d(dtype=float),
    vel_v_new: wp.array3d(dtype=float),
    vel_w_new: wp.array3d(dtype=float),
    vel_u_old: wp.array3d(dtype=float),
    vel_v_old: wp.array3d(dtype=float),
    vel_w_old: wp.array3d(dtype=float),
    flip_pic_blend: float,
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]

    grid_v_new = sample_mac_velocity(vel_u_new, vel_v_new, vel_w_new, p, dh, nx, ny, nz)
    grid_v_old = sample_mac_velocity(vel_u_old, vel_v_old, vel_w_old, p, dh, nx, ny, nz)

    flip_v = particle_v[tid] + (grid_v_new - grid_v_old)
    particle_v[tid] = (1.0 - flip_pic_blend) * flip_v + flip_pic_blend * grid_v_new


@wp.kernel
def advect_particles_with_velocity(
    particles: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    dt: float,
    boundary_min: wp.vec3,
    boundary_max: wp.vec3,
):
    tid = wp.tid()
    p = particles[tid]
    v = particle_v[tid]

    p_next = p + v * dt

    if p_next[0] <= boundary_min[0]:
        p_next[0] = boundary_min[0]
        if v[0] < 0.0:
            v[0] = 0.0
    if p_next[0] >= boundary_max[0]:
        p_next[0] = boundary_max[0]
        if v[0] > 0.0:
            v[0] = 0.0

    if p_next[1] <= boundary_min[1]:
        p_next[1] = boundary_min[1]
        if v[1] < 0.0:
            v[1] = 0.0
    if p_next[1] >= boundary_max[1]:
        p_next[1] = boundary_max[1]
        if v[1] > 0.0:
            v[1] = 0.0

    if p_next[2] <= boundary_min[2]:
        p_next[2] = boundary_min[2]
        if v[2] < 0.0:
            v[2] = 0.0
    if p_next[2] >= boundary_max[2]:
        p_next[2] = boundary_max[2]
        if v[2] > 0.0:
            v[2] = 0.0

    particles[tid] = p_next
    particle_v[tid] = v

@wp.kernel
def advect_particles_in_grid_rk2(
    particles: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    vel_u: wp.array3d(dtype=float),
    vel_v: wp.array3d(dtype=float),
    vel_w: wp.array3d(dtype=float),
    dt: float,
    nx: int,
    ny: int,
    nz: int,
    dh: float,
    boundary_min: wp.vec3,
    boundary_max: wp.vec3,
    boundary_pad: float,
):
    tid = wp.tid()
    p = particles[tid]

    v1 = sample_mac_velocity(vel_u, vel_v, vel_w, p, dh, nx, ny, nz)
    p_mid = p + 0.5 * dt * v1
    v2 = sample_mac_velocity(vel_u, vel_v, vel_w, p_mid, dh, nx, ny, nz)
    p_next = p + dt * v2

    pv = particle_v[tid]

    min_x = boundary_min[0] + boundary_pad
    min_y = boundary_min[1] + boundary_pad
    min_z = boundary_min[2] + boundary_pad
    max_x = boundary_max[0] - boundary_pad
    max_y = boundary_max[1] - boundary_pad
    max_z = boundary_max[2] - boundary_pad


    if p_next[0] <= min_x:
        p_next[0] = min_x
        if pv[0] < 0.0:
            pv[0] = 0.0
    if p_next[0] >= max_x:
        p_next[0] = max_x
        if pv[0] > 0.0:
            pv[0] = 0.0

    if p_next[1] <= min_y:
        p_next[1] = min_y
        if pv[1] < 0.0:
            pv[1] = 0.0
    if p_next[1] >= max_y:
        p_next[1] = max_y
        if pv[1] > 0.0:
            pv[1] = 0.0

    if p_next[2] <= min_z:
        p_next[2] = min_z
        if pv[2] < 0.0:
            pv[2] = 0.0
    if p_next[2] >= max_z:
        p_next[2] = max_z
        if pv[2] > 0.0:
            pv[2] = 0.0

    particles[tid] = p_next
    particle_v[tid] = pv
    
@wp.kernel
def resolve_particle_solid_collision_with_velocity(
    particles: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particles[tid]

    sdf = sample_centered_scalar_world(solid_phi, p, dh, nx, ny, nz)
    # Keep particle centers at least half a cell away from binary solid voxels
    # to avoid repeated wall re-penetration on coarse grids.
    margin = 0.5 * dh

    if sdf < margin:
        eps = 0.5 * dh
        sdf_x1 = sample_centered_scalar_world(solid_phi, p + wp.vec3(eps, 0.0, 0.0), dh, nx, ny, nz)
        sdf_x0 = sample_centered_scalar_world(solid_phi, p - wp.vec3(eps, 0.0, 0.0), dh, nx, ny, nz)
        sdf_y1 = sample_centered_scalar_world(solid_phi, p + wp.vec3(0.0, eps, 0.0), dh, nx, ny, nz)
        sdf_y0 = sample_centered_scalar_world(solid_phi, p - wp.vec3(0.0, eps, 0.0), dh, nx, ny, nz)
        sdf_z1 = sample_centered_scalar_world(solid_phi, p + wp.vec3(0.0, 0.0, eps), dh, nx, ny, nz)
        sdf_z0 = sample_centered_scalar_world(solid_phi, p - wp.vec3(0.0, 0.0, eps), dh, nx, ny, nz)

        grad = wp.vec3(sdf_x1 - sdf_x0, sdf_y1 - sdf_y0, sdf_z1 - sdf_z0)
        len_sq = wp.dot(grad, grad)
        if len_sq > 1e-8:
            normal = wp.normalize(grad)
            p_corr = p + normal * (margin - sdf)
            particles[tid] = p_corr

            v = particle_v[tid]
            # Resolve collision in the moving-boundary frame.
            v_solid = sample_mac_velocity(vel_solid_u, vel_solid_v, vel_solid_w, p_corr, dh, nx, ny, nz)
            v_rel = v - v_solid
            vn_rel = wp.dot(v_rel, normal)
            if vn_rel < 0.0:
                v_rel = v_rel - vn_rel * normal
            particle_v[tid] = v_rel + v_solid

@wp.kernel
def apply_gravity_u(v: wp.array3d(dtype=float), g: float, dt: float):
    i, j, k = wp.tid()
    v[i, j, k] += g * dt


@wp.kernel
def apply_gravity_v(v: wp.array3d(dtype=float), g: float, dt: float):
    i, j, k = wp.tid()
    v[i, j, k] += g * dt


@wp.kernel
def apply_gravity_w(v: wp.array3d(dtype=float), g: float, dt: float):
    i, j, k = wp.tid()
    v[i, j, k] += g * dt


@wp.kernel
def compute_divergence_cell_type_mac(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    div_out: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    i, j, k = wp.tid()

    if cell_type[i, j, k] != wp.int32(CELL_FLUID) or solid_phi[i, j, k] < 0.0:
        div_out[i, j, k] = 0.0
        return

    div_out[i, j, k] = divergence_cell_with_solid_bc(u, v, w, solid_phi, i, j, k, nx, ny, nz, dh)


@wp.func
def get_pressure_neighbor_contribution_cell_type(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
    pressure: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
) -> wp.vec2:
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return wp.vec2(0.0, 0.0)

    ctype = cell_type[ix, iy, iz]
    if ctype == wp.int32(CELL_SOLID):
        return wp.vec2(0.0, 0.0)
    if ctype == wp.int32(CELL_FLUID):
        return wp.vec2(pressure[ix, iy, iz], 1.0)

    return wp.vec2(0.0, 1.0)


@wp.kernel
def pressure_jacobi_cell_type_mac(
    pressure_in: wp.array3d(dtype=float),
    pressure_out: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()

    if cell_type[i, j, k] != wp.int32(CELL_FLUID):
        pressure_out[i, j, k] = 0.0
        return

    res_l = get_pressure_neighbor_contribution_cell_type(i - 1, j, k, nx, ny, nz, pressure_in, cell_type)
    res_r = get_pressure_neighbor_contribution_cell_type(i + 1, j, k, nx, ny, nz, pressure_in, cell_type)
    res_d = get_pressure_neighbor_contribution_cell_type(i, j - 1, k, nx, ny, nz, pressure_in, cell_type)
    res_u = get_pressure_neighbor_contribution_cell_type(i, j + 1, k, nx, ny, nz, pressure_in, cell_type)
    res_b = get_pressure_neighbor_contribution_cell_type(i, j, k - 1, nx, ny, nz, pressure_in, cell_type)
    res_f = get_pressure_neighbor_contribution_cell_type(i, j, k + 1, nx, ny, nz, pressure_in, cell_type)

    weight_sum = res_l[1] + res_r[1] + res_d[1] + res_u[1] + res_b[1] + res_f[1]

    if weight_sum < 0.1:
        pressure_out[i, j, k] = 0.0
    else:
        sum_p = res_l[0] + res_r[0] + res_d[0] + res_u[0] + res_b[0] + res_f[0]
        pressure_out[i, j, k] = (sum_p - div[i, j, k] * dh * dh / dt) / weight_sum


@wp.kernel
def project_u_cell_type(
    u: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    nx: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if i == 0 or i == nx:
        return

    type_l = cell_type[i - 1, j, k]
    type_r = cell_type[i, j, k]
    if type_l == wp.int32(CELL_SOLID) or type_r == wp.int32(CELL_SOLID):
        return
    if type_l != wp.int32(CELL_FLUID) and type_r != wp.int32(CELL_FLUID):
        return

    p_l = 0.0
    if type_l == wp.int32(CELL_FLUID):
        p_l = p[i - 1, j, k]
    p_r = 0.0
    if type_r == wp.int32(CELL_FLUID):
        p_r = p[i, j, k]

    u[i, j, k] -= dt * (p_r - p_l) / dh


@wp.kernel
def project_v_cell_type(
    v: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    ny: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if j == 0 or j == ny:
        return

    type_d = cell_type[i, j - 1, k]
    type_u = cell_type[i, j, k]
    if type_d == wp.int32(CELL_SOLID) or type_u == wp.int32(CELL_SOLID):
        return
    if type_d != wp.int32(CELL_FLUID) and type_u != wp.int32(CELL_FLUID):
        return

    p_d = 0.0
    if type_d == wp.int32(CELL_FLUID):
        p_d = p[i, j - 1, k]
    p_u = 0.0
    if type_u == wp.int32(CELL_FLUID):
        p_u = p[i, j, k]

    v[i, j, k] -= dt * (p_u - p_d) / dh


@wp.kernel
def project_w_cell_type(
    w: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    nz: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if k == 0 or k == nz:
        return

    type_b = cell_type[i, j, k - 1]
    type_f = cell_type[i, j, k]
    if type_b == wp.int32(CELL_SOLID) or type_f == wp.int32(CELL_SOLID):
        return
    if type_b != wp.int32(CELL_FLUID) and type_f != wp.int32(CELL_FLUID):
        return

    p_b = 0.0
    if type_b == wp.int32(CELL_FLUID):
        p_b = p[i, j, k - 1]
    p_f = 0.0
    if type_f == wp.int32(CELL_FLUID):
        p_f = p[i, j, k]

    w[i, j, k] -= dt * (p_f - p_b) / dh


@wp.kernel
def pressure_apply_operator_cell_type_mac(
    x: wp.array3d(dtype=float),
    ax: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if cell_type[i, j, k] != wp.int32(CELL_FLUID):
        ax[i, j, k] = 0.0
        return

    res_l = get_pressure_neighbor_contribution_cell_type(i - 1, j, k, nx, ny, nz, x, cell_type)
    res_r = get_pressure_neighbor_contribution_cell_type(i + 1, j, k, nx, ny, nz, x, cell_type)
    res_d = get_pressure_neighbor_contribution_cell_type(i, j - 1, k, nx, ny, nz, x, cell_type)
    res_u = get_pressure_neighbor_contribution_cell_type(i, j + 1, k, nx, ny, nz, x, cell_type)
    res_b = get_pressure_neighbor_contribution_cell_type(i, j, k - 1, nx, ny, nz, x, cell_type)
    res_f = get_pressure_neighbor_contribution_cell_type(i, j, k + 1, nx, ny, nz, x, cell_type)

    weight_sum = res_l[1] + res_r[1] + res_d[1] + res_u[1] + res_b[1] + res_f[1]
    if weight_sum < 0.1:
        ax[i, j, k] = 0.0
        return

    sum_neighbors = res_l[0] + res_r[0] + res_d[0] + res_u[0] + res_b[0] + res_f[0]
    ax[i, j, k] = weight_sum * x[i, j, k] - sum_neighbors


@wp.kernel
def pressure_build_inv_diag_cell_type_mac(
    inv_diag: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if cell_type[i, j, k] != wp.int32(CELL_FLUID):
        inv_diag[i, j, k] = 0.0
        return

    w_l = get_pressure_neighbor_contribution_cell_type(i - 1, j, k, nx, ny, nz, inv_diag, cell_type)[1]
    w_r = get_pressure_neighbor_contribution_cell_type(i + 1, j, k, nx, ny, nz, inv_diag, cell_type)[1]
    w_d = get_pressure_neighbor_contribution_cell_type(i, j - 1, k, nx, ny, nz, inv_diag, cell_type)[1]
    w_u = get_pressure_neighbor_contribution_cell_type(i, j + 1, k, nx, ny, nz, inv_diag, cell_type)[1]
    w_b = get_pressure_neighbor_contribution_cell_type(i, j, k - 1, nx, ny, nz, inv_diag, cell_type)[1]
    w_f = get_pressure_neighbor_contribution_cell_type(i, j, k + 1, nx, ny, nz, inv_diag, cell_type)[1]

    weight_sum = w_l + w_r + w_d + w_u + w_b + w_f
    if weight_sum < 0.1:
        inv_diag[i, j, k] = 0.0
    else:
        inv_diag[i, j, k] = 1.0 / weight_sum
