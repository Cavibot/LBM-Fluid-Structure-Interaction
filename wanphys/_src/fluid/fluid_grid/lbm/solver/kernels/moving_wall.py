"""Moving-wall transport kernels and their local sampling helpers."""

from __future__ import annotations

import warp as wp

from .common import direction_weight, direction_x, direction_y, direction_z

_INV_CS2 = 3.0

@wp.func
def _clamp_int(value: int, lower: int, upper: int) -> int:
    """Clamp an integer index for MAC-face sampling."""
    result = value
    if result < lower:
        result = lower
    if result > upper:
        result = upper
    return result


@wp.func
def _sample_solid_u(
    vel_solid_u: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    ii = _clamp_int(i, 0, nx)
    jj = _clamp_int(j, 0, ny - 1)
    kk = _clamp_int(k, 0, nz - 1)
    return vel_solid_u[ii, jj, kk]


@wp.func
def _sample_solid_v(
    vel_solid_v: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    ii = _clamp_int(i, 0, nx - 1)
    jj = _clamp_int(j, 0, ny)
    kk = _clamp_int(k, 0, nz - 1)
    return vel_solid_v[ii, jj, kk]


@wp.func
def _sample_solid_w(
    vel_solid_w: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    ii = _clamp_int(i, 0, nx - 1)
    jj = _clamp_int(j, 0, ny - 1)
    kk = _clamp_int(k, 0, nz)
    return vel_solid_w[ii, jj, kk]


@wp.func
def _solid_wall_velocity_dot(
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    cx: int,
    cy: int,
    cz: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """Return c_i dot u_wall at the half-link wall.

    Axis links sample the crossed MAC face.  Diagonal D3Q19 links sample each
    active component on its crossed MAC face and average the two adjacent faces
    that straddle the diagonal link midpoint.
    """
    wall_dot = 0.0

    if cx != 0:
        ui = i
        if cx < 0:
            ui = i + 1
        u_wall = _sample_solid_u(vel_solid_u, ui, j, k, nx, ny, nz)
        if cy != 0:
            u_wall = 0.5 * (
                u_wall + _sample_solid_u(vel_solid_u, ui, j - cy, k, nx, ny, nz)
            )
        if cz != 0:
            u_wall = 0.5 * (
                u_wall + _sample_solid_u(vel_solid_u, ui, j, k - cz, nx, ny, nz)
            )
        wall_dot += float(cx) * u_wall

    if cy != 0:
        vj = j
        if cy < 0:
            vj = j + 1
        v_wall = _sample_solid_v(vel_solid_v, i, vj, k, nx, ny, nz)
        if cx != 0:
            v_wall = 0.5 * (
                v_wall + _sample_solid_v(vel_solid_v, i - cx, vj, k, nx, ny, nz)
            )
        if cz != 0:
            v_wall = 0.5 * (
                v_wall + _sample_solid_v(vel_solid_v, i, vj, k - cz, nx, ny, nz)
            )
        wall_dot += float(cy) * v_wall

    if cz != 0:
        wk = k
        if cz < 0:
            wk = k + 1
        w_wall = _sample_solid_w(vel_solid_w, i, j, wk, nx, ny, nz)
        if cx != 0:
            w_wall = 0.5 * (
                w_wall + _sample_solid_w(vel_solid_w, i - cx, j, wk, nx, ny, nz)
            )
        if cy != 0:
            w_wall = 0.5 * (
                w_wall + _sample_solid_w(vel_solid_w, i, j - cy, wk, nx, ny, nz)
            )
        wall_dot += float(cz) * w_wall

    return wall_dot


@wp.func
def _moving_wall_correction(
    has_moving_walls: int,
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    cx: int,
    cy: int,
    cz: int,
    w: float,
    rho_w: float,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """Moving-wall bounce-back correction (2 w_i rho c_i·u_wall / c_s^2).

    When *has_moving_walls* is 0, all walls are static and the
    correction is identically zero — short-circuit to avoid the
    MAC-face velocity lookups.
    """
    if has_moving_walls == 0:
        return 0.0
    wall_dot = _solid_wall_velocity_dot(
        vel_solid_u,
        vel_solid_v,
        vel_solid_w,
        i,
        j,
        k,
        cx,
        cy,
        cz,
        nx,
        ny,
        nz,
    )
    return 2.0 * w * rho_w * _INV_CS2 * wall_dot


@wp.kernel
def apply_moving_wall_transport_kernel(
    f_star: wp.array(dtype=float),
    rho_hint: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    bc_types: wp.array(dtype=wp.int32),
    use_cut_link: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Add moving-wall correction to links already bounced by streaming."""

    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    face_count = 0
    touches_open = False
    if i == 0:
        face_count += 1
        touches_open = touches_open or bc_types[0] == 1 or bc_types[0] == 2 or bc_types[0] == 4
    if i == nx - 1:
        face_count += 1
        touches_open = touches_open or bc_types[1] == 1 or bc_types[1] == 2 or bc_types[1] == 4
    if j == 0:
        face_count += 1
        touches_open = touches_open or bc_types[2] == 1 or bc_types[2] == 2 or bc_types[2] == 4
    if j == ny - 1:
        face_count += 1
        touches_open = touches_open or bc_types[3] == 1 or bc_types[3] == 2 or bc_types[3] == 4
    if k == 0:
        face_count += 1
        touches_open = touches_open or bc_types[4] == 1 or bc_types[4] == 2 or bc_types[4] == 4
    if k == nz - 1:
        face_count += 1
        touches_open = touches_open or bc_types[5] == 1 or bc_types[5] == 2 or bc_types[5] == 4
    if face_count > 1 and touches_open:
        return
    # Accumulate unconditionally and select afterwards.  In Warp 1.12 a
    # loop nested in this data-dependent branch produces malformed CPU C++.
    # The unconditional sum is cheap (D3Q19) and is equivalent here.
    r_hint = rho_hint[i, j, k]
    r_sum = float(0.0)
    for q_sum in range(19):
        r_sum += f_star[q_sum * stride + idx]
    r = wp.where(r_hint <= 1.0e-12, wp.max(r_sum, 1.0e-12), r_hint)

    for q in range(1, 19):
        cx, cy, cz = direction_x(q), direction_y(q), direction_z(q)
        si, sj, sk = i - cx, j - cy, k - cz
        outside = False
        if si < 0 or si >= nx:
            if px != 0:
                if si < 0:
                    si += nx
                else:
                    si -= nx
            else:
                outside = True
        if sj < 0 or sj >= ny:
            if py != 0:
                if sj < 0:
                    sj += ny
                else:
                    sj -= ny
            else:
                outside = True
        if sk < 0 or sk >= nz:
            if pz != 0:
                if sk < 0:
                    sk += nz
                else:
                    sk -= nz
            else:
                outside = True

        bounced = outside
        if not outside:
            bounced = solid_phi[si, sj, sk] < 0.0
        if bounced:
            correction_scale = 1.0
            if use_cut_link != 0 and not outside:
                phi_fluid = solid_phi[i, j, k]
                phi_solid = solid_phi[si, sj, sk]
                denominator = phi_fluid - phi_solid
                if denominator > 1.0e-12:
                    link_fraction = wp.clamp(phi_fluid / denominator, 1.0e-6, 1.0)
                    if link_fraction >= 0.5:
                        correction_scale = 1.0 / (2.0 * link_fraction)
            f_star[q * stride + idx] += correction_scale * _moving_wall_correction(
                1,
                vel_solid_u,
                vel_solid_v,
                vel_solid_w,
                i,
                j,
                k,
                cx,
                cy,
                cz,
                direction_weight(q),
                r,
                nx,
                ny,
                nz,
            )
