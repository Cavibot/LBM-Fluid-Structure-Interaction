import warp as wp


@wp.func
def clamp_index(idx: int, n: int) -> int:
    # Clamp index to [0, n-1] for periodic boundary conditions
    return wp.clamp(idx, 0, n - 1)

@wp.kernel
def dissipate_scalar(field: wp.array3d(dtype=float), rate: float, dt: float):
    i, j, k = wp.tid()
    decay = wp.max(0.0, 1.0 - rate * dt)
    field[i, j, k] = field[i, j, k] * decay

@wp.func
def sample_scalar_trilinear(
    grid: wp.array3d(dtype=float), 
    p: wp.vec3, 
    dim_x: int, dim_y: int, dim_z: int
) -> float:
    i = int(wp.floor(p[0]))
    j = int(wp.floor(p[1]))
    k = int(wp.floor(p[2]))

    fx = p[0] - float(i)
    fy = p[1] - float(j)
    fz = p[2] - float(k)

    i0 = wp.clamp(i, 0, dim_x - 1); i1 = wp.clamp(i + 1, 0, dim_x - 1)
    j0 = wp.clamp(j, 0, dim_y - 1); j1 = wp.clamp(j + 1, 0, dim_y - 1)
    k0 = wp.clamp(k, 0, dim_z - 1); k1 = wp.clamp(k + 1, 0, dim_z - 1)

    c00 = grid[i0, j0, k0] * (1.0 - fx) + grid[i1, j0, k0] * fx
    c10 = grid[i0, j1, k0] * (1.0 - fx) + grid[i1, j1, k0] * fx
    c01 = grid[i0, j0, k1] * (1.0 - fx) + grid[i1, j0, k1] * fx
    c11 = grid[i0, j1, k1] * (1.0 - fx) + grid[i1, j1, k1] * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz

@wp.func
def sample_velocity_mac(
    u: wp.array3d(dtype=float), 
    v: wp.array3d(dtype=float), 
    w: wp.array3d(dtype=float), 
    p_world: wp.vec3, dh: float, 
    nx: int, ny: int, nz: int
) -> wp.vec3:
    # Sample velocity in 3 dimensions
    px = p_world[0] / dh
    py = p_world[1] / dh
    pz = p_world[2] / dh

    vx = sample_scalar_trilinear(u, wp.vec3(px, py - 0.5, pz - 0.5), nx + 1, ny, nz)
    vy = sample_scalar_trilinear(v, wp.vec3(px - 0.5, py, pz - 0.5), nx, ny + 1, nz)
    vz = sample_scalar_trilinear(w, wp.vec3(px - 0.5, py - 0.5, pz), nx, ny, nz + 1)

    return wp.vec3(vx, vy, vz)

@wp.kernel
def advect_u_mac(
    u_in: wp.array3d(dtype=float), 
    v_in: wp.array3d(dtype=float), 
    w_in: wp.array3d(dtype=float),
    u_out: wp.array3d(dtype=float), 
    dt: float, dh: float, 
    nx: int, ny: int, nz: int
):
    # Advect the u-component of velocity using semi-Lagrangian advection with periodic boundaries.
    i, j, k = wp.tid()

    pos = wp.vec3(float(i) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    vel = sample_velocity_mac(u_in, v_in, w_in, pos, dh, nx, ny, nz)
    pos_old = pos - vel * dt
    p_idx = wp.vec3(pos_old[0] / dh, pos_old[1] / dh - 0.5, pos_old[2] / dh - 0.5)
    u_out[i, j, k] = sample_scalar_trilinear(u_in, p_idx, nx + 1, ny, nz)

@wp.kernel
def advect_v_mac(
    u_in: wp.array3d(dtype=float), 
    v_in: wp.array3d(dtype=float), 
    w_in: wp.array3d(dtype=float),
    v_out: wp.array3d(dtype=float), 
    dt: float, dh: float, 
    nx: int, ny: int, nz: int
):
    # Advect the v-component of velocity using semi-Lagrangian advection with periodic boundaries.
    i, j, k = wp.tid()

    pos = wp.vec3((float(i) + 0.5) * dh, float(j) * dh, (float(k) + 0.5) * dh)
    vel = sample_velocity_mac(u_in, v_in, w_in, pos, dh, nx, ny, nz)
    pos_old = pos - vel * dt
    p_idx = wp.vec3(pos_old[0] / dh - 0.5, pos_old[1] / dh, pos_old[2] / dh - 0.5)
    v_out[i, j, k] = sample_scalar_trilinear(v_in, p_idx, nx, ny + 1, nz)

@wp.kernel
def advect_w_mac(
    u_in: wp.array3d(dtype=float), 
    v_in: wp.array3d(dtype=float),
    w_in: wp.array3d(dtype=float),
    w_out: wp.array3d(dtype=float), 
    dt: float, dh: float, 
    nx: int, ny: int, nz: int
):
    # Advect the w-component of velocity using semi-Lagrangian advection with periodic boundaries.
    i, j, k = wp.tid()

    pos = wp.vec3((float(i)+0.5) * dh, (float(j) + 0.5) * dh, float(k) * dh)
    vel = sample_velocity_mac(u_in, v_in, w_in, pos, dh, nx, ny, nz)
    pos_old = pos - vel * dt
    p_idx = wp.vec3(pos_old[0] / dh - 0.5, pos_old[1] / dh - 0.5, pos_old[2] / dh)
    w_out[i, j, k] = sample_scalar_trilinear(w_in, p_idx, nx, ny, nz + 1)

@wp.kernel
def advect_scalar_mac(
    scalar_in: wp.array3d(dtype=float), 
    u_in: wp.array3d(dtype=float), 
    v_in: wp.array3d(dtype=float), 
    w_in: wp.array3d(dtype=float),
    scalar_out: wp.array3d(dtype=float), 
    dt: float, dh: float, 
    nx: int, ny: int, nz: int
):
    # Advect a scalar field using semi-Lagrangian advection with periodic boundaries.
    i, j, k = wp.tid()

    pos = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    vel = sample_velocity_mac(u_in, v_in, w_in, pos, dh, nx, ny, nz)
    pos_old = pos - vel * dt
    p_idx = wp.vec3(pos_old[0] / dh - 0.5, pos_old[1] / dh - 0.5, pos_old[2] / dh - 0.5)
    scalar_out[i, j, k] = sample_scalar_trilinear(scalar_in, p_idx, nx, ny, nz)

@wp.kernel
def apply_force_mac(v: wp.array3d(dtype=float), force: float, dt: float):
    i, j, k = wp.tid()
    v[i, j, k] = v[i, j, k] + force * dt

@wp.kernel
def enforce_solid_u_mac(u: wp.array3d(dtype=float), solid_phi: wp.array3d(dtype=float), nx: int):
    i, j, k = wp.tid()
    if i == 0 or i == nx:
        u[i, j, k] = 0.0
        return

    s_l = solid_phi[i-1, j, k]
    s_r = solid_phi[i, j, k]

    if s_l * s_r <= 0.0:
        u[i, j, k] = 0.0

@wp.kernel
def enforce_solid_v_mac(v: wp.array3d(dtype=float), solid_phi: wp.array3d(dtype=float), ny: int):
    i, j, k = wp.tid()
    if j == 0 or j == ny:
        v[i, j, k] = 0.0
        return
        
    s_d = solid_phi[i, j-1, k]
    s_u = solid_phi[i, j, k]
    if s_d * s_u <= 0.0:
        v[i, j, k] = 0.0

@wp.kernel
def enforce_solid_w_mac(w: wp.array3d(dtype=float), solid_phi: wp.array3d(dtype=float), nz: int):
    i, j, k = wp.tid()
    if k == 0 or k == nz:
        w[i, j, k] = 0.0
        return
        
    s_b = solid_phi[i, j, k-1]
    s_f = solid_phi[i, j, k]
    if s_b * s_f <= 0.0:
        w[i, j, k] = 0.0

@wp.kernel
def bake_solid_box_kernel(
    solid_phi: wp.array3d(dtype=float),
    dh: float,
    center: wp.vec3,
    half_extents: wp.vec3
):
    i, j, k = wp.tid()
    p = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)

    # Grid cell center to box surface
    d = wp.vec3(
        wp.abs(p[0] - center[0]) - half_extents[0],
        wp.abs(p[1] - center[1]) - half_extents[1],
        wp.abs(p[2] - center[2]) - half_extents[2]
    )

    outside_dist = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0)))
    inside_dist = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    box_sdf = outside_dist + inside_dist
    
    # Merge multiple solid sdf into one
    solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], box_sdf)

@wp.kernel
def bake_solid_sphere_kernel(
    solid_phi: wp.array3d(dtype=float),
    dh: float,
    center: wp.vec3,
    radius: float
):
    i, j, k = wp.tid()
    p = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)

    # Grid cell center to sphere surface
    dist = wp.length(p - center) - radius

    # Merge multiple solid sdf into one
    solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], dist)

@wp.kernel
def bake_solid_mesh_kernel(
    solid_phi: wp.array3d(dtype=float),
    mesh: wp.uint64,
    dh: float,
    pos: wp.vec3,
    rot: wp.quat,
    scale: float,
    max_dist: float
):
    i, j, k = wp.tid()
    p_world = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    
    # World space into local space(mesh)
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p_world - pos) / scale
    
    sign = float(0.0)
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    
    # Query distance
    if wp.mesh_query_point(mesh, p_local, max_dist, sign, face, u, v):
        cp_local = wp.mesh_eval_position(mesh, face, u, v)
        dist_local = wp.length(p_local - cp_local)
        sdf = dist_local * scale * sign

        solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], sdf)

@wp.func
def divergence_cell_with_solid_bc(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    dh: float,
) -> float:
    s_l = solid_phi[wp.max(i - 1, 0), j, k]
    if i == 0:
        s_l = -1.0

    s_r = solid_phi[wp.min(i + 1, nx - 1), j, k]
    if i == nx - 1:
        s_r = -1.0

    s_d = solid_phi[i, wp.max(j - 1, 0), k]
    if j == 0:
        s_d = -1.0

    s_u = solid_phi[i, wp.min(j + 1, ny - 1), k]
    if j == ny - 1:
        s_u = -1.0

    s_b = solid_phi[i, j, wp.max(k - 1, 0)]
    if k == 0:
        s_b = -1.0

    s_f = solid_phi[i, j, wp.min(k + 1, nz - 1)]
    if k == nz - 1:
        s_f = -1.0

    u_l = u[i, j, k]
    if s_l < 0.0:
        u_l = 0.0

    u_r = u[i + 1, j, k]
    if s_r < 0.0:
        u_r = 0.0

    v_d = v[i, j, k]
    if s_d < 0.0:
        v_d = 0.0

    v_u = v[i, j + 1, k]
    if s_u < 0.0:
        v_u = 0.0

    w_b = w[i, j, k]
    if s_b < 0.0:
        w_b = 0.0

    w_f = w[i, j, k + 1]
    if s_f < 0.0:
        w_f = 0.0

    return (u_r - u_l + v_u - v_d + w_f - w_b) / dh


@wp.func
def pressure_neighbor_weight_solid(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
    pressure: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
) -> wp.vec2:
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return wp.vec2(0.0, 0.0)

    if solid_phi[ix, iy, iz] < 0.0:
        return wp.vec2(0.0, 0.0)

    return wp.vec2(pressure[ix, iy, iz], 1.0)


@wp.kernel
def compute_divergence_solid_mac(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    div_out: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        div_out[i, j, k] = 0.0
        return

    div_out[i, j, k] = divergence_cell_with_solid_bc(u, v, w, solid_phi, i, j, k, nx, ny, nz, dh)


@wp.kernel
def pressure_jacobi_solid_mac(
    pressure_in: wp.array3d(dtype=float),
    pressure_out: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        pressure_out[i, j, k] = 0.0
        return

    res_l = pressure_neighbor_weight_solid(i - 1, j, k, nx, ny, nz, pressure_in, solid_phi)
    res_r = pressure_neighbor_weight_solid(i + 1, j, k, nx, ny, nz, pressure_in, solid_phi)
    res_d = pressure_neighbor_weight_solid(i, j - 1, k, nx, ny, nz, pressure_in, solid_phi)
    res_u = pressure_neighbor_weight_solid(i, j + 1, k, nx, ny, nz, pressure_in, solid_phi)
    res_b = pressure_neighbor_weight_solid(i, j, k - 1, nx, ny, nz, pressure_in, solid_phi)
    res_f = pressure_neighbor_weight_solid(i, j, k + 1, nx, ny, nz, pressure_in, solid_phi)

    weight_sum = res_l[1] + res_r[1] + res_d[1] + res_u[1] + res_b[1] + res_f[1]
    if weight_sum < 0.1:
        pressure_out[i, j, k] = 0.0
        return

    sum_p = res_l[0] + res_r[0] + res_d[0] + res_u[0] + res_b[0] + res_f[0]
    pressure_out[i, j, k] = (sum_p - div[i, j, k] * dh * dh / dt) / weight_sum


@wp.kernel
def project_u_solid_mac(
    u: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if i == 0 or i == nx:
        return
    if solid_phi[i - 1, j, k] < 0.0 or solid_phi[i, j, k] < 0.0:
        return
    u[i, j, k] -= dt * (p[i, j, k] - p[i - 1, j, k]) / dh


@wp.kernel
def project_v_solid_mac(
    v: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    ny: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if j == 0 or j == ny:
        return
    if solid_phi[i, j - 1, k] < 0.0 or solid_phi[i, j, k] < 0.0:
        return
    v[i, j, k] -= dt * (p[i, j, k] - p[i, j - 1, k]) / dh


@wp.kernel
def project_w_solid_mac(
    w: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nz: int,
    dh: float,
    dt: float,
):
    i, j, k = wp.tid()
    if k == 0 or k == nz:
        return
    if solid_phi[i, j, k - 1] < 0.0 or solid_phi[i, j, k] < 0.0:
        return
    w[i, j, k] -= dt * (p[i, j, k] - p[i, j, k - 1]) / dh

@wp.kernel
def pressure_apply_operator_solid_mac(
    x: wp.array3d(dtype=float),
    ax: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if solid_phi[i, j, k] < 0.0:
        ax[i, j, k] = 0.0
        return

    res_l = pressure_neighbor_weight_solid(i - 1, j, k, nx, ny, nz, x, solid_phi)
    res_r = pressure_neighbor_weight_solid(i + 1, j, k, nx, ny, nz, x, solid_phi)
    res_d = pressure_neighbor_weight_solid(i, j - 1, k, nx, ny, nz, x, solid_phi)
    res_u = pressure_neighbor_weight_solid(i, j + 1, k, nx, ny, nz, x, solid_phi)
    res_b = pressure_neighbor_weight_solid(i, j, k - 1, nx, ny, nz, x, solid_phi)
    res_f = pressure_neighbor_weight_solid(i, j, k + 1, nx, ny, nz, x, solid_phi)

    weight_sum = res_l[1] + res_r[1] + res_d[1] + res_u[1] + res_b[1] + res_f[1]
    if weight_sum < 0.1:
        ax[i, j, k] = 0.0
        return

    sum_neighbors = res_l[0] + res_r[0] + res_d[0] + res_u[0] + res_b[0] + res_f[0]
    ax[i, j, k] = weight_sum * x[i, j, k] - sum_neighbors


@wp.kernel
def pressure_build_inv_diag_solid_mac(
    inv_diag: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    if solid_phi[i, j, k] < 0.0:
        inv_diag[i, j, k] = 0.0
        return

    w_l = pressure_neighbor_weight_solid(i - 1, j, k, nx, ny, nz, inv_diag, solid_phi)[1]
    w_r = pressure_neighbor_weight_solid(i + 1, j, k, nx, ny, nz, inv_diag, solid_phi)[1]
    w_d = pressure_neighbor_weight_solid(i, j - 1, k, nx, ny, nz, inv_diag, solid_phi)[1]
    w_u = pressure_neighbor_weight_solid(i, j + 1, k, nx, ny, nz, inv_diag, solid_phi)[1]
    w_b = pressure_neighbor_weight_solid(i, j, k - 1, nx, ny, nz, inv_diag, solid_phi)[1]
    w_f = pressure_neighbor_weight_solid(i, j, k + 1, nx, ny, nz, inv_diag, solid_phi)[1]

    weight_sum = w_l + w_r + w_d + w_u + w_b + w_f
    if weight_sum < 0.1:
        inv_diag[i, j, k] = 0.0
    else:
        inv_diag[i, j, k] = 1.0 / weight_sum

