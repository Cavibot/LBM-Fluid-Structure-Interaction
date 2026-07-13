# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp GPU kernels for the D3Q19 Lattice Boltzmann Method.

Each kernel is launched over ``(nx, ny, nz)`` threads (or per-face shapes
for MAC interpolation).  Distribution functions are stored in a single flat
``wp.array(dtype=float)`` of length ``19 * nx * ny * nz``.  Macroscopic
fields (*rho*, *ux*, *uy*, *uz*, *solid_phi*) use standard
:class:`wp.array3d`.

Kernel summary
--------------
* :func:`compute_moments_kernel` -- rho, u from the 19 distributions.
* :func:`collide_stream_bounceback_kernel` -- fused BGK collide, pull-stream,
  and halfway bounce-back in a single launch.
* :func:`apply_guo_force_kernel` -- Guo body-force correction (uniform, optional).
* :func:`_psi` -- Shan-Chen pseudopotential (effective mass).
* :func:`compute_shan_chen_force_kernel` -- Shan-Chen interaction force from
  pseudopotential gradients.
* :func:`apply_velocity_shift_kernel` -- velocity-shift forcing for the
  Shan-Chen interaction (standard SC approach).
* :func:`moments_to_mac_u_kernel` / ``_v_`` / ``_w_`` -- cell-centred ->
  MAC-face velocity interpolation.
* :func:`initialize_equilibrium_kernel` -- set *f* to equilibrium for given
  (rho0, u0).
* :func:`compute_pressure_kernel` -- *p = c_s^2 . rho*.
"""

from __future__ import annotations

import warp as wp

from .core.lattice import NUM_DIRS

# Lattice speed of sound squared = 1/3, so inv = 3.
_INV_CS2 = 3.0

# ---------------------------------------------------------------------------
# Helper -- Maxwell-Boltzmann equilibrium (inlined by Warp)
# ---------------------------------------------------------------------------


@wp.func
def _f_eq(
    w: float,
    rho: float,
    ux: float,
    uy: float,
    uz: float,
    cx: int,
    cy: int,
    cz: int,
) -> float:
    """D3Q19 equilibrium distribution for a single direction.

    ``f_i^eq = w_i rho [1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5 u^2]``
    """
    cu = float(cx) * ux + float(cy) * uy + float(cz) * uz
    u_sq = ux * ux + uy * uy + uz * uz
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)


# ---------------------------------------------------------------------------
# Kernel 1 -- macroscopic moment computation
# ---------------------------------------------------------------------------


@wp.kernel
def compute_moments_kernel(
    f: wp.array(dtype=float),
    stride: int,
    nx: int,
    ny: int,
    nz: int,
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
) -> None:
    """Compute rho and u from the 19 distribution functions.

    Launched over ``(nx, ny, nz)``.  Each thread sums all 19 directions
    at its cell.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    # Accumulate
    r = 0.0
    mx = 0.0
    my = 0.0
    mz = 0.0

    # d=0: rest (0,0,0)  w=1/3
    f0 = f[0 * stride + idx]
    r += f0
    # no momentum contribution

    # d=1: +x (1,0,0)  w=1/18
    f1 = f[1 * stride + idx]
    r += f1
    mx += f1

    # d=2: -x (-1,0,0)  w=1/18
    f2 = f[2 * stride + idx]
    r += f2
    mx -= f2

    # d=3: +y (0,1,0)  w=1/18
    f3 = f[3 * stride + idx]
    r += f3
    my += f3

    # d=4: -y (0,-1,0)  w=1/18
    f4 = f[4 * stride + idx]
    r += f4
    my -= f4

    # d=5: +z (0,0,1)  w=1/18
    f5 = f[5 * stride + idx]
    r += f5
    mz += f5

    # d=6: -z (0,0,-1)  w=1/18
    f6 = f[6 * stride + idx]
    r += f6
    mz -= f6

    # d=7: +x+y (1,1,0)  w=1/36
    f7 = f[7 * stride + idx]
    r += f7
    mx += f7
    my += f7

    # d=8: -x+y (-1,1,0)  w=1/36
    f8 = f[8 * stride + idx]
    r += f8
    mx -= f8
    my += f8

    # d=9: +x-y (1,-1,0)  w=1/36
    f9 = f[9 * stride + idx]
    r += f9
    mx += f9
    my -= f9

    # d=10: -x-y (-1,-1,0)  w=1/36
    f10 = f[10 * stride + idx]
    r += f10
    mx -= f10
    my -= f10

    # d=11: +x+z (1,0,1)  w=1/36
    f11 = f[11 * stride + idx]
    r += f11
    mx += f11
    mz += f11

    # d=12: -x+z (-1,0,1)  w=1/36
    f12 = f[12 * stride + idx]
    r += f12
    mx -= f12
    mz += f12

    # d=13: +x-z (1,0,-1)  w=1/36
    f13 = f[13 * stride + idx]
    r += f13
    mx += f13
    mz -= f13

    # d=14: -x-z (-1,0,-1)  w=1/36
    f14 = f[14 * stride + idx]
    r += f14
    mx -= f14
    mz -= f14

    # d=15: +y+z (0,1,1)  w=1/36
    f15 = f[15 * stride + idx]
    r += f15
    my += f15
    mz += f15

    # d=16: -y+z (0,-1,1)  w=1/36
    f16 = f[16 * stride + idx]
    r += f16
    my -= f16
    mz += f16

    # d=17: +y-z (0,1,-1)  w=1/36
    f17 = f[17 * stride + idx]
    r += f17
    my += f17
    mz -= f17

    # d=18: -y-z (0,-1,-1)  w=1/36
    f18 = f[18 * stride + idx]
    r += f18
    my -= f18
    mz -= f18

    # Safety clamp: floor 0.005 prevents F/ρ divergence, ceiling 10.0
    # catches runaway mass creation from non-conservative SC forces
    # without restricting normal interface density fluctuations.
    r = wp.clamp(r, 0.005, 10.0)

    rho[i, j, k] = r
    ux[i, j, k] = mx / r
    uy[i, j, k] = my / r
    uz[i, j, k] = mz / r


# ---------------------------------------------------------------------------
# Kernel 2 -- fused collide + pull-stream + halfway bounce-back
# ---------------------------------------------------------------------------


@wp.func
def _trt_collide(
    f_src: float,
    f_opp_src: float,
    rho_src: float,
    ux_src: float,
    uy_src: float,
    uz_src: float,
    w: float,
    cx: int,
    cy: int,
    cz: int,
    omega_plus: float,
    omega_minus: float,
) -> float:
    """TRT (Two-Relaxation-Time) collision for a single direction.

    Decomposes the distribution into even (+) and odd (-) symmetric
    parts and relaxes each with its own frequency.  When
    ``omega_plus == omega_minus`` this reduces exactly to standard BGK.
    """
    cu = float(cx) * ux_src + float(cy) * uy_src + float(cz) * uz_src
    u_sq = ux_src * ux_src + uy_src * uy_src + uz_src * uz_src

    # Even / odd decomposition
    f_plus = 0.5 * (f_src + f_opp_src)
    f_minus = 0.5 * (f_src - f_opp_src)

    # Equilibrium even / odd parts
    # feq = w * rho * (1 + 3cu + 4.5cu^2 - 1.5u^2)
    # feq^+ = w * rho * (1 + 4.5cu^2 - 1.5u^2)
    # feq^- = w * rho * 3cu
    feq_plus = w * rho_src * (1.0 + 4.5 * cu * cu - 1.5 * u_sq)
    feq_minus = w * rho_src * 3.0 * cu

    return (
        f_src
        - omega_plus * (f_plus - feq_plus)
        - omega_minus * (f_minus - feq_minus)
    )


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
def collide_stream_bounceback_kernel(
    f_in: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    f_out: wp.array(dtype=float),
    omega_plus: float,
    omega_minus: float,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Fused TRT/BGK collide, pull-stream, and halfway bounce-back.

    When ``omega_plus == omega_minus`` this reduces to standard BGK.
    The TRT "magic" parameter is set via ``LbmModel.lambda_trt``.
    Moving-wall bounce-back adds ``2*w_i*rho*(c_i.u_wall)/c_s^2`` only on
    solid/out-of-domain links.  Axis links sample the crossed MAC face.
    Diagonal links average the two adjacent MAC faces per active component
    that straddle the diagonal half-link midpoint.

    *px*, *py*, *pz* are periodic-boundary flags (0 or 1).  When an axis
    is periodic, out-of-bounds neighbour lookups wrap to the opposite face
    instead of falling through to bounce-back.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    # ---- current-cell macros (used by bounce-back branches) ---------------
    rho_w = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]

    # 碰撞逻辑处理
    # =======================================================================
    # d = 0  --  rest  (0,0,0)  w = 1/3
    # =======================================================================
    f_src = f_in[0 * stride + idx]
    f_out[0 * stride + idx] = f_src - omega_plus * (
        f_src
        - _f_eq(1.0 / 3.0, rho_w, vx, vy, vz, 0, 0, 0)
    )

    # =======================================================================
    # d = 1  --  +x  (1,0,0)  w = 1/18   opposite = d=2 (-x)
    # =======================================================================
    si = i - 1
    sj = j
    sk = k
    # periodic xmin -> wrap to xmax
    if si < 0 and px:
        si = nx - 1
    if si >= 0 and si < nx and solid_phi[si, sj, sk] >= 0.0:  # 流体邻居
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[1 * stride + src_idx]
        f_opp_src = f_in[2 * stride + src_idx]
        f_out[1 * stride + idx] = _trt_collide(  # TRT碰撞
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 18.0, 1, 0, 0, omega_plus, omega_minus
        )
    else:  #固体邻居
        # bounce-back: opposite d=2 (-x) at current cell
        f_opp = f_in[2 * stride + idx]
        f_out[1 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, -1, 0, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 1, 0, 0, 1.0 / 18.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 2  --  -x  (-1,0,0)  w = 1/18   opposite = d=1 (+x)
    # =======================================================================
    si = i + 1
    sj = j
    sk = k
    # periodic xmax -> wrap to xmin
    if si >= nx and px:
        si = 0
    if si >= 0 and si < nx and solid_phi[si, sj, sk] >= 0.0:
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[2 * stride + src_idx]
        f_opp_src = f_in[1 * stride + src_idx]
        f_out[2 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 18.0, -1, 0, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[1 * stride + idx]
        f_out[2 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 1, 0, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, -1, 0, 0, 1.0 / 18.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 3  --  +y  (0,1,0)  w = 1/18   opposite = d=4 (-y)
    # =======================================================================
    si = i
    sj = j - 1
    sk = k
    # periodic ymin -> wrap to ymax
    if sj < 0 and py:
        sj = ny - 1
    if sj >= 0 and sj < ny and solid_phi[si, sj, sk] >= 0.0:
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[3 * stride + src_idx]
        f_opp_src = f_in[4 * stride + src_idx]
        f_out[3 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 18.0, 0, 1, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[4 * stride + idx]
        f_out[3 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, -1, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, 1, 0, 1.0 / 18.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 4  --  -y  (0,-1,0)  w = 1/18   opposite = d=3 (+y)
    # =======================================================================
    si = i
    sj = j + 1
    sk = k
    # periodic ymax -> wrap to ymin
    if sj >= ny and py:
        sj = 0
    if sj >= 0 and sj < ny and solid_phi[si, sj, sk] >= 0.0:
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[4 * stride + src_idx]
        f_opp_src = f_in[3 * stride + src_idx]
        f_out[4 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 18.0, 0, -1, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[3 * stride + idx]
        f_out[4 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 1, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, -1, 0, 1.0 / 18.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 5  --  +z  (0,0,1)  w = 1/18   opposite = d=6 (-z)
    # =======================================================================
    si = i
    sj = j
    sk = k - 1
    # periodic zmin -> wrap to zmax
    if sk < 0 and pz:
        sk = nz - 1
    if sk >= 0 and sk < nz and solid_phi[si, sj, sk] >= 0.0:
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[5 * stride + src_idx]
        f_opp_src = f_in[6 * stride + src_idx]
        f_out[5 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 18.0, 0, 0, 1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[6 * stride + idx]
        f_out[5 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, -1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, 0, 1, 1.0 / 18.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 6  --  -z  (0,0,-1)  w = 1/18   opposite = d=5 (+z)
    # =======================================================================
    si = i
    sj = j
    sk = k + 1
    # periodic zmax -> wrap to zmin
    if sk >= nz and pz:
        sk = 0
    if sk >= 0 and sk < nz and solid_phi[si, sj, sk] >= 0.0:
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[6 * stride + src_idx]
        f_opp_src = f_in[5 * stride + src_idx]
        f_out[6 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 18.0, 0, 0, -1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[5 * stride + idx]
        f_out[6 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, 1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, 0, -1, 1.0 / 18.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 7  --  +x+y  (1,1,0)  w = 1/36   opposite = d=10 (-x,-y)
    # =======================================================================
    si = i - 1
    sj = j - 1
    sk = k
    # periodic xmin -> xmax, ymin -> ymax
    if si < 0 and px:
        si = nx - 1
    if sj < 0 and py:
        sj = ny - 1
    if (
        si >= 0 and si < nx
        and sj >= 0 and sj < ny
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[7 * stride + src_idx]
        f_opp_src = f_in[10 * stride + src_idx]
        f_out[7 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 1, 1, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[10 * stride + idx]
        f_out[7 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 1, 1, 0, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 8  --  -x+y  (-1,1,0)  w = 1/36   opposite = d=9 (+x,-y)
    # =======================================================================
    si = i + 1
    sj = j - 1
    sk = k
    # periodic xmax -> xmin, ymin -> ymax
    if si >= nx and px:
        si = 0
    if sj < 0 and py:
        sj = ny - 1
    if (
        si >= 0 and si < nx
        and sj >= 0 and sj < ny
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[8 * stride + src_idx]
        f_opp_src = f_in[9 * stride + src_idx]
        f_out[8 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, -1, 1, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[9 * stride + idx]
        f_out[8 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, -1, 1, 0, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 9  --  +x-y  (1,-1,0)  w = 1/36   opposite = d=8 (-x,+y)
    # =======================================================================
    si = i - 1
    sj = j + 1
    sk = k
    # periodic xmin -> xmax, ymax -> ymin
    if si < 0 and px:
        si = nx - 1
    if sj >= ny and py:
        sj = 0
    if (
        si >= 0 and si < nx
        and sj >= 0 and sj < ny
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[9 * stride + src_idx]
        f_opp_src = f_in[8 * stride + src_idx]
        f_out[9 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 1, -1, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[8 * stride + idx]
        f_out[9 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 1, -1, 0, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 10 --  -x-y  (-1,-1,0)  w = 1/36   opposite = d=7 (+x,+y)
    # =======================================================================
    si = i + 1
    sj = j + 1
    sk = k
    # periodic xmax -> xmin, ymax -> ymin
    if si >= nx and px:
        si = 0
    if sj >= ny and py:
        sj = 0
    if (
        si >= 0 and si < nx
        and sj >= 0 and sj < ny
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[10 * stride + src_idx]
        f_opp_src = f_in[7 * stride + src_idx]
        f_out[10 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, -1, -1, 0, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[7 * stride + idx]
        f_out[10 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, -1, -1, 0, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 11 --  +x+z  (1,0,1)  w = 1/36   opposite = d=14 (-x,-z)
    # =======================================================================
    si = i - 1
    sj = j
    sk = k - 1
    # periodic xmin -> xmax, zmin -> zmax
    if si < 0 and px:
        si = nx - 1
    if sk < 0 and pz:
        sk = nz - 1
    if (
        si >= 0 and si < nx
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[11 * stride + src_idx]
        f_opp_src = f_in[14 * stride + src_idx]
        f_out[11 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 1, 0, 1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[14 * stride + idx]
        f_out[11 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 1, 0, 1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 12 --  -x+z  (-1,0,1)  w = 1/36   opposite = d=13 (+x,-z)
    # =======================================================================
    si = i + 1
    sj = j
    sk = k - 1
    # periodic xmax -> xmin, zmin -> zmax
    if si >= nx and px:
        si = 0
    if sk < 0 and pz:
        sk = nz - 1
    if (
        si >= 0 and si < nx
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[12 * stride + src_idx]
        f_opp_src = f_in[13 * stride + src_idx]
        f_out[12 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, -1, 0, 1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[13 * stride + idx]
        f_out[12 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, -1, 0, 1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 13 --  +x-z  (1,0,-1)  w = 1/36   opposite = d=12 (-x,+z)
    # =======================================================================
    si = i - 1
    sj = j
    sk = k + 1
    # periodic xmin -> xmax, zmax -> zmin
    if si < 0 and px:
        si = nx - 1
    if sk >= nz and pz:
        sk = 0
    if (
        si >= 0 and si < nx
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[13 * stride + src_idx]
        f_opp_src = f_in[12 * stride + src_idx]
        f_out[13 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 1, 0, -1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[12 * stride + idx]
        f_out[13 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 1, 0, -1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 14 --  -x-z  (-1,0,-1)  w = 1/36   opposite = d=11 (+x,+z)
    # =======================================================================
    si = i + 1
    sj = j
    sk = k + 1
    # periodic xmax -> xmin, zmax -> zmin
    if si >= nx and px:
        si = 0
    if sk >= nz and pz:
        sk = 0
    if (
        si >= 0 and si < nx
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[14 * stride + src_idx]
        f_opp_src = f_in[11 * stride + src_idx]
        f_out[14 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, -1, 0, -1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[11 * stride + idx]
        f_out[14 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, -1, 0, -1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 15 --  +y+z  (0,1,1)  w = 1/36   opposite = d=18 (-y,-z)
    # =======================================================================
    si = i
    sj = j - 1
    sk = k - 1
    # periodic ymin -> ymax, zmin -> zmax
    if sj < 0 and py:
        sj = ny - 1
    if sk < 0 and pz:
        sk = nz - 1
    if (
        sj >= 0 and sj < ny
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[15 * stride + src_idx]
        f_opp_src = f_in[18 * stride + src_idx]
        f_out[15 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 0, 1, 1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[18 * stride + idx]
        f_out[15 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, 1, 1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 16 --  -y+z  (0,-1,1)  w = 1/36   opposite = d=17 (+y,-z)
    # =======================================================================
    si = i
    sj = j + 1
    sk = k - 1
    # periodic ymax -> ymin, zmin -> zmax
    if sj >= ny and py:
        sj = 0
    if sk < 0 and pz:
        sk = nz - 1
    if (
        sj >= 0 and sj < ny
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[16 * stride + src_idx]
        f_opp_src = f_in[17 * stride + src_idx]
        f_out[16 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 0, -1, 1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[17 * stride + idx]
        f_out[16 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, -1, 1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 17 --  +y-z  (0,1,-1)  w = 1/36   opposite = d=16 (-y,+z)
    # =======================================================================
    si = i
    sj = j - 1
    sk = k + 1
    # periodic ymin -> ymax, zmax -> zmin
    if sj < 0 and py:
        sj = ny - 1
    if sk >= nz and pz:
        sk = 0
    if (
        sj >= 0 and sj < ny
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[17 * stride + src_idx]
        f_opp_src = f_in[16 * stride + src_idx]
        f_out[17 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 0, 1, -1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[16 * stride + idx]
        f_out[17 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, 1, -1, 1.0 / 36.0, rho_w, nx, ny, nz
        )

    # =======================================================================
    # d = 18 --  -y-z  (0,-1,-1)  w = 1/36   opposite = d=15 (+y,+z)
    # =======================================================================
    si = i
    sj = j + 1
    sk = k + 1
    # periodic ymax -> ymin, zmax -> zmin
    if sj >= ny and py:
        sj = 0
    if sk >= nz and pz:
        sk = 0
    if (
        sj >= 0 and sj < ny
        and sk >= 0 and sk < nz
        and solid_phi[si, sj, sk] >= 0.0
    ):
        src_idx = si * ny * nz + sj * nz + sk
        f_src = f_in[18 * stride + src_idx]
        f_opp_src = f_in[15 * stride + src_idx]
        f_out[18 * stride + idx] = _trt_collide(
            f_src, f_opp_src, rho[si, sj, sk], ux[si, sj, sk], uy[si, sj, sk], uz[si, sj, sk],
            1.0 / 36.0, 0, -1, -1, omega_plus, omega_minus
        )
    else:
        f_opp = f_in[15 * stride + idx]
        f_out[18 * stride + idx] = f_opp - omega_minus * (
            f_opp
            - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1)
        ) + _moving_wall_correction(
            vel_solid_u, vel_solid_v, vel_solid_w,
            i, j, k, 0, -1, -1, 1.0 / 36.0, rho_w, nx, ny, nz
        )


# ---------------------------------------------------------------------------
# Kernel 3 -- Guo body-force correction (optional, called only when F != 0)
# ---------------------------------------------------------------------------


@wp.kernel
def apply_guo_force_kernel(
    f: wp.array(dtype=float),
    fx: float,
    fy: float,
    fz: float,
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Add Guo forcing term to the pre-collision distributions (in-place).

    ``Deltaf_d = (1 - omega/2) . w_d . 3 . (c_d . F)``   (simplified for body forces).

    Only directions with non-zero *c_d.F* are modified.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    factor = (1.0 - omega * 0.5) * 3.0  # (1 - omega/2) x 3

    # --- face directions (w = 1/18) ---
    # d=1: +x
    f[1 * stride + idx] += factor * (1.0 / 18.0) * fx
    # d=2: -x
    f[2 * stride + idx] += factor * (1.0 / 18.0) * (-fx)
    # d=3: +y
    f[3 * stride + idx] += factor * (1.0 / 18.0) * fy
    # d=4: -y
    f[4 * stride + idx] += factor * (1.0 / 18.0) * (-fy)
    # d=5: +z
    f[5 * stride + idx] += factor * (1.0 / 18.0) * fz
    # d=6: -z
    f[6 * stride + idx] += factor * (1.0 / 18.0) * (-fz)

    # --- edge directions (w = 1/36) ---
    # d=7:  +x+y   (+fx + fy)
    f[7 * stride + idx] += factor * (1.0 / 36.0) * (fx + fy)
    # d=8:  -x+y   (-fx + fy)
    f[8 * stride + idx] += factor * (1.0 / 36.0) * (-fx + fy)
    # d=9:  +x-y   (+fx - fy)
    f[9 * stride + idx] += factor * (1.0 / 36.0) * (fx - fy)
    # d=10: -x-y   (-fx - fy)
    f[10 * stride + idx] += factor * (1.0 / 36.0) * (-fx - fy)
    # d=11: +x+z   (+fx + fz)
    f[11 * stride + idx] += factor * (1.0 / 36.0) * (fx + fz)
    # d=12: -x+z   (-fx + fz)
    f[12 * stride + idx] += factor * (1.0 / 36.0) * (-fx + fz)
    # d=13: +x-z   (+fx - fz)
    f[13 * stride + idx] += factor * (1.0 / 36.0) * (fx - fz)
    # d=14: -x-z   (-fx - fz)
    f[14 * stride + idx] += factor * (1.0 / 36.0) * (-fx - fz)
    # d=15: +y+z   (+fy + fz)
    f[15 * stride + idx] += factor * (1.0 / 36.0) * (fy + fz)
    # d=16: -y+z   (-fy + fz)
    f[16 * stride + idx] += factor * (1.0 / 36.0) * (-fy + fz)
    # d=17: +y-z   (+fy - fz)
    f[17 * stride + idx] += factor * (1.0 / 36.0) * (fy - fz)
    # d=18: -y-z   (-fy - fz)
    f[18 * stride + idx] += factor * (1.0 / 36.0) * (-fy - fz)


# ---------------------------------------------------------------------------
# Kernel 3b -- Shan-Chen interaction force computation
# ---------------------------------------------------------------------------


@wp.func
def _psi(rho: float, psi_type: int, ref_rho: float) -> float:
    """Shan-Chen pseudopotential (effective mass).

    ``PSI_RHO`` (0): ψ = ρ.
    ``PSI_EXP`` (1): ψ = 1 - exp(-ρ / ρ_ref).
    """
    if psi_type == 1:
        return 1.0 - wp.exp(-rho / ref_rho)
    return rho


@wp.func
def _resolve_boundary_psi(
    ni: int,
    nj: int,
    nk: int,
    rho: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    psi_c: float,
    psi_type: int,
    psi_ref: float,
    solid_psi_scale: float,
    boundary_psi: float,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """Return the pseudopotential ψ of a D3Q19 neighbour at (ni, nj, nk).

    Virtual-density method: fluid neighbours use their actual ψ(ρ); solid
    bodies and out-of-bounds domain faces each contribute a constant virtual
    ψ so that the unified SC force ``F = -G ψ(x) Σ w_i ψ_n e_i`` captures
    both fluid–fluid interaction and fluid–wall adsorption in a single sum.

    When a periodic axis flag (*px*, *py*, *pz*) is set, out-of-bounds
    neighbours on that axis wrap to the opposite face and contribute their
    actual fluid ψ instead of a virtual wall ψ.
    """
    # ---- periodic wrap -------------------------------------------------------
    if ni < 0 and px:
        ni = nx - 1
    if ni >= nx and px:
        ni = 0
    if nj < 0 and py:
        nj = ny - 1
    if nj >= ny and py:
        nj = 0
    if nk < 0 and pz:
        nk = nz - 1
    if nk >= nz and pz:
        nk = 0

    # ---- classify wrapped neighbour ------------------------------------------
    in_bounds = (
        (ni >= 0) and (ni < nx)
        and (nj >= 0) and (nj < ny)
        and (nk >= 0) and (nk < nz)
    )
    if in_bounds and solid_phi[ni, nj, nk] >= 0.0:
        return _psi(rho[ni, nj, nk], psi_type, psi_ref)   # fluid

    if in_bounds:
        return psi_c * solid_psi_scale                     # solid body

    # out-of-bounds (domain wall on non-periodic axis)
    if boundary_psi >= 0.0:
        return boundary_psi                                 # fixed wall ψ
    return psi_c * solid_psi_scale                          # legacy mirror


@wp.kernel
def compute_shan_chen_force_kernel(
    rho: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    G: float,
    psi_type: int,
    psi_ref: float,
    solid_psi_scale: float,
    boundary_psi: float,
    homogeneous_early_out: int,
    homogeneous_rel_tol: float,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Unified Shan-Chen force via the virtual-density method.

    ``F(x) = -G ψ(x) Σ_i w_i ψ_n e_i``

    Every D3Q19 neighbour contributes to the same sum:
    - **fluid**: actual ψ(ρ) from the density field.
    - **solid body** (``solid_phi < 0``): virtual ψ = ``ψ_c × solid_psi_scale``.
    - **domain wall** (out-of-bounds on non-periodic axis): virtual ψ
      = *boundary_psi* when ≥ 0, else the legacy mirror closure
      ``ψ_c × solid_psi_scale``.
    - **periodic neighbour**: wraps to the opposite face via *px*/*py*/*pz*.

    Fluid–fluid interaction and fluid–wall adsorption are both controlled
    by the single interaction strength *G*; wetting is tuned via the
    virtual ψ of the wall / solid.
    """
    i, j, k = wp.tid()

    if solid_phi[i, j, k] < 0.0:
        fx[i, j, k] = 0.0
        fy[i, j, k] = 0.0
        fz[i, j, k] = 0.0
        return

    # ---- current-cell pseudopotential --------------------------------------
    rho_c = rho[i, j, k]
    psi_c = _psi(rho_c, psi_type, psi_ref)

    # ---- homogeneous-region early-out ---------------------------------------
    # In bulk fluid (all neighbours have similar ρ, none is solid/wall),
    # the symmetric sum Σ w_i ψ_n e_i ≈ 0.  We check 6 face neighbours;
    # if any has deviant ρ, is solid, or is out-of-bounds on a non-periodic
    # axis, we fall through to the full 19-dir computation.
    # Periodic OOB neighbours wrap to the opposite face for the density check.
    lo = rho_c * (1.0 - homogeneous_rel_tol)
    hi = rho_c * (1.0 + homogeneous_rel_tol)
    homogeneous = 1
    if homogeneous_early_out == 0:
        homogeneous = 0
    # +x
    if homogeneous == 1:
        ni = i + 1
        if ni < nx:
            pass  # in-bounds
        elif px:
            ni = 0  # wrap xmax -> xmin
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[ni, j, k] < 0.0 or rho[ni, j, k] < lo or rho[ni, j, k] > hi:
                homogeneous = 0
    # -x
    if homogeneous == 1:
        ni = i - 1
        if ni >= 0:
            pass  # in-bounds
        elif px:
            ni = nx - 1  # wrap xmin -> xmax
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[ni, j, k] < 0.0 or rho[ni, j, k] < lo or rho[ni, j, k] > hi:
                homogeneous = 0
    # +y
    if homogeneous == 1:
        nj = j + 1
        if nj < ny:
            pass  # in-bounds
        elif py:
            nj = 0  # wrap ymax -> ymin
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, nj, k] < 0.0 or rho[i, nj, k] < lo or rho[i, nj, k] > hi:
                homogeneous = 0
    # -y
    if homogeneous == 1:
        nj = j - 1
        if nj >= 0:
            pass  # in-bounds
        elif py:
            nj = ny - 1  # wrap ymin -> ymax
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, nj, k] < 0.0 or rho[i, nj, k] < lo or rho[i, nj, k] > hi:
                homogeneous = 0
    # +z
    if homogeneous == 1:
        nk = k + 1
        if nk < nz:
            pass  # in-bounds
        elif pz:
            nk = 0  # wrap zmax -> zmin
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, j, nk] < 0.0 or rho[i, j, nk] < lo or rho[i, j, nk] > hi:
                homogeneous = 0
    # -z
    if homogeneous == 1:
        nk = k - 1
        if nk >= 0:
            pass  # in-bounds
        elif pz:
            nk = nz - 1  # wrap zmin -> zmax
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, j, nk] < 0.0 or rho[i, j, nk] < lo or rho[i, j, nk] > hi:
                homogeneous = 0
    if homogeneous == 1:
        fx[i, j, k] = 0.0
        fy[i, j, k] = 0.0
        fz[i, j, k] = 0.0
        return

    # ---- accumulate Σ w_i ψ(x+e_i) e_i ------------------------------------
    sx = 0.0
    sy = 0.0
    sz = 0.0

    # ========================================================================
    # Face directions (w = 1/18)
    # ========================================================================

    # d=1: +x  (1,0,0)
    ni = i + 1; nj = j; nk = k
    sx += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz) * 1.0

    # d=2: -x  (-1,0,0)
    ni = i - 1; nj = j; nk = k
    sx += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz) * (-1.0)

    # d=3: +y  (0,1,0)
    ni = i; nj = j + 1; nk = k
    sy += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz) * 1.0

    # d=4: -y  (0,-1,0)
    ni = i; nj = j - 1; nk = k
    sy += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz) * (-1.0)

    # d=5: +z  (0,0,1)
    ni = i; nj = j; nk = k + 1
    sz += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz) * 1.0

    # d=6: -z  (0,0,-1)
    ni = i; nj = j; nk = k - 1
    sz += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz) * (-1.0)

    # ========================================================================
    # Edge directions (w = 1/36)
    # ========================================================================

    # d=7: +x+y  (1,1,0)
    ni = i + 1; nj = j + 1; nk = k
    psi_d7 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d7 * 1.0
    sy += (1.0 / 36.0) * psi_d7 * 1.0

    # d=8: -x+y  (-1,1,0)
    ni = i - 1; nj = j + 1; nk = k
    psi_d8 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d8 * (-1.0)
    sy += (1.0 / 36.0) * psi_d8 * 1.0

    # d=9: +x-y  (1,-1,0)
    ni = i + 1; nj = j - 1; nk = k
    psi_d9 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d9 * 1.0
    sy += (1.0 / 36.0) * psi_d9 * (-1.0)

    # d=10: -x-y  (-1,-1,0)
    ni = i - 1; nj = j - 1; nk = k
    psi_d10 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d10 * (-1.0)
    sy += (1.0 / 36.0) * psi_d10 * (-1.0)

    # d=11: +x+z  (1,0,1)
    ni = i + 1; nj = j; nk = k + 1
    psi_d11 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d11 * 1.0
    sz += (1.0 / 36.0) * psi_d11 * 1.0

    # d=12: -x+z  (-1,0,1)
    ni = i - 1; nj = j; nk = k + 1
    psi_d12 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d12 * (-1.0)
    sz += (1.0 / 36.0) * psi_d12 * 1.0

    # d=13: +x-z  (1,0,-1)
    ni = i + 1; nj = j; nk = k - 1
    psi_d13 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d13 * 1.0
    sz += (1.0 / 36.0) * psi_d13 * (-1.0)

    # d=14: -x-z  (-1,0,-1)
    ni = i - 1; nj = j; nk = k - 1
    psi_d14 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d14 * (-1.0)
    sz += (1.0 / 36.0) * psi_d14 * (-1.0)

    # d=15: +y+z  (0,1,1)
    ni = i; nj = j + 1; nk = k + 1
    psi_d15 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d15 * 1.0
    sz += (1.0 / 36.0) * psi_d15 * 1.0

    # d=16: -y+z  (0,-1,1)
    ni = i; nj = j - 1; nk = k + 1
    psi_d16 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d16 * (-1.0)
    sz += (1.0 / 36.0) * psi_d16 * 1.0

    # d=17: +y-z  (0,1,-1)
    ni = i; nj = j + 1; nk = k - 1
    psi_d17 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d17 * 1.0
    sz += (1.0 / 36.0) * psi_d17 * (-1.0)

    # d=18: -y-z  (0,-1,-1)
    ni = i; nj = j - 1; nk = k - 1
    psi_d18 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d18 * (-1.0)
    sz += (1.0 / 36.0) * psi_d18 * (-1.0)

    # ---- unified force: F = -G ψ_c Σ w_i ψ_n e_i -------------------------
    coeff = -G * psi_c
    fx[i, j, k] = coeff * sx
    fy[i, j, k] = coeff * sy
    fz[i, j, k] = coeff * sz


# ---------------------------------------------------------------------------
# Kernel 3c -- velocity shift for body forces (Shan-Chen interaction)
# ---------------------------------------------------------------------------


@wp.kernel
def apply_velocity_shift_kernel(
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    gx: float,
    gy: float,
    gz: float,
    tau_shift: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Shift the equilibrium velocity by all body forces.

    ``u_eq = u + τ_shift · (F_int / ρ + g)``

    where *F_int* is the per-cell Shan-Chen interaction force and *g* is
    a uniform body force (e.g. gravity).

    .. important::

       The current solver passes ``τ₋`` (``model.tau``) for Shan-Chen flows.
       This keeps the shifted equilibrium velocity bounded when TRT uses a
       larger even-mode relaxation time for ghost-mode damping.

    This is the standard velocity-shift scheme (Shan & Chen 1993): all
    body forces go through a single shift, and the shifted velocity
    replaces *u* in the collision equilibrium.
    """
    i, j, k = wp.tid()
    r = rho[i, j, k]
    inv_rho = 1.0 / wp.max(r, 1.0e-12)

    # Compute shifted equilibrium velocity
    ux_s = ux[i, j, k] + tau_shift * (fx[i, j, k] * inv_rho + gx)
    uy_s = uy[i, j, k] + tau_shift * (fy[i, j, k] * inv_rho + gy)
    uz_s = uz[i, j, k] + tau_shift * (fz[i, j, k] * inv_rho + gz)

    # Soft clamp at Mach ~0.7 (|u_eq| ≤ 0.4 lu).
    # This is the theoretical limit where D3Q19 f_eq stays positive in
    # the worst-case direction (1 − 3|u| + 3|u|² > 0 for |u| < ~0.42).
    # Only the few extreme interface cells (≪ 1%) ever hit this ceiling.
    u_sq = ux_s * ux_s + uy_s * uy_s + uz_s * uz_s
    if u_sq > 0.16:  # ≈ Mach 0.69
        sc = wp.sqrt(0.16 / u_sq)
        ux_s = ux_s * sc
        uy_s = uy_s * sc
        uz_s = uz_s * sc

    ux[i, j, k] = ux_s
    uy[i, j, k] = uy_s
    uz[i, j, k] = uz_s


# ---------------------------------------------------------------------------
# Kernel 3d -- restore physical velocity after collision (reverse shift)
# ---------------------------------------------------------------------------


@wp.kernel
def restore_physical_velocity_kernel(
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    gx: float,
    gy: float,
    gz: float,
    tau_shift: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Reverse the velocity shift to recover physical velocity for output.

    ``u_phys = u_eq - τ_shift · (F_int / ρ + g)``

    This kernel MUST be launched after the collide-stream and boundary-
    condition kernels have consumed the shifted equilibrium velocity,
    but BEFORE the macroscopic fields are copied to ``state_out`` and
    MAC-face interpolation is performed.

    Together with :func:`apply_velocity_shift_kernel`, this restores the
    correct separation between the equilibrium velocity (used inside
    collision) and the physical velocity (output to state / visualization).

    .. important::

       *tau_shift* must be the **same** value passed to
       :func:`apply_velocity_shift_kernel` — currently ``τ₋``
       (``model.tau``) in the solver's Shan-Chen path.
       See that kernel's documentation for the rationale.
    """
    i, j, k = wp.tid()
    r = rho[i, j, k]
    inv_rho = 1.0 / wp.max(r, 1.0e-12)
    ux[i, j, k] = ux[i, j, k] - tau_shift * (fx[i, j, k] * inv_rho + gx)
    uy[i, j, k] = uy[i, j, k] - tau_shift * (fy[i, j, k] * inv_rho + gy)
    uz[i, j, k] = uz[i, j, k] - tau_shift * (fz[i, j, k] * inv_rho + gz)


# ---------------------------------------------------------------------------
# Kernel 4 -- MAC-face velocity interpolation (3 kernels, one per component)
# ---------------------------------------------------------------------------


@wp.kernel
def moments_to_mac_u_kernel(
    ux: wp.array3d(dtype=float),
    vel_u: wp.array3d(dtype=float),
    nx: int,
) -> None:
    """Interpolate cell-centred *ux* -> MAC x-face velocity ``vel_u``.

    Launched over ``(nx+1, ny, nz)``.
    """
    i, j, k = wp.tid()
    if i == 0:
        vel_u[i, j, k] = ux[0, j, k]
    elif i == nx:
        vel_u[i, j, k] = ux[nx - 1, j, k]
    else:
        vel_u[i, j, k] = 0.5 * (ux[i - 1, j, k] + ux[i, j, k])


@wp.kernel
def moments_to_mac_v_kernel(
    uy: wp.array3d(dtype=float),
    vel_v: wp.array3d(dtype=float),
    ny: int,
) -> None:
    """Interpolate cell-centred *uy* -> MAC y-face velocity ``vel_v``.

    Launched over ``(nx, ny+1, nz)``.
    """
    i, j, k = wp.tid()
    if j == 0:
        vel_v[i, j, k] = uy[i, 0, k]
    elif j == ny:
        vel_v[i, j, k] = uy[i, ny - 1, k]
    else:
        vel_v[i, j, k] = 0.5 * (uy[i, j - 1, k] + uy[i, j, k])


@wp.kernel
def moments_to_mac_w_kernel(
    uz: wp.array3d(dtype=float),
    vel_w: wp.array3d(dtype=float),
    nz: int,
) -> None:
    """Interpolate cell-centred *uz* -> MAC z-face velocity ``vel_w``.

    Launched over ``(nx, ny, nz+1)``.
    """
    i, j, k = wp.tid()
    if k == 0:
        vel_w[i, j, k] = uz[i, j, 0]
    elif k == nz:
        vel_w[i, j, k] = uz[i, j, nz - 1]
    else:
        vel_w[i, j, k] = 0.5 * (uz[i, j, k - 1] + uz[i, j, k])


# ---------------------------------------------------------------------------
# Kernel 5 -- equilibrium initialisation
# ---------------------------------------------------------------------------


@wp.kernel
def initialize_equilibrium_kernel(
    f: wp.array(dtype=float),
    rho0: float,
    u0x: float,
    u0y: float,
    u0z: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Set *f* to the Maxwell-Boltzmann equilibrium for uniform (rho0, u0)."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    f[0 * stride + idx] = _f_eq(1.0 / 3.0, rho0, u0x, u0y, u0z, 0, 0, 0)
    f[1 * stride + idx] = _f_eq(1.0 / 18.0, rho0, u0x, u0y, u0z, 1, 0, 0)
    f[2 * stride + idx] = _f_eq(1.0 / 18.0, rho0, u0x, u0y, u0z, -1, 0, 0)
    f[3 * stride + idx] = _f_eq(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, 1, 0)
    f[4 * stride + idx] = _f_eq(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, -1, 0)
    f[5 * stride + idx] = _f_eq(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, 0, 1)
    f[6 * stride + idx] = _f_eq(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, 0, -1)
    f[7 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, 1, 0)
    f[8 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, 1, 0)
    f[9 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, -1, 0)
    f[10 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, -1, 0)
    f[11 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, 0, 1)
    f[12 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, 0, 1)
    f[13 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, 0, -1)
    f[14 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, 0, -1)
    f[15 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, 1, 1)
    f[16 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, -1, 1)
    f[17 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, 1, -1)
    f[18 * stride + idx] = _f_eq(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, -1, -1)


# ---------------------------------------------------------------------------
# Kernel 6 -- pressure from density (equation of state)
# ---------------------------------------------------------------------------


@wp.kernel
def compute_pressure_kernel(
    rho: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
) -> None:
    """Compute pressure via the ideal-gas equation of state: ``p = c_s^2 rho``."""
    i, j, k = wp.tid()
    pressure[i, j, k] = (1.0 / 3.0) * rho[i, j, k]


# ---------------------------------------------------------------------------
# Kernel 7 – boundary conditions (Zou-He inlet, convective outflow)
# ---------------------------------------------------------------------------


@wp.kernel
def apply_boundary_conditions_kernel(
    f: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    bc_types: wp.array(dtype=wp.int32),
    bc_vel_x: wp.array(dtype=float),
    bc_vel_y: wp.array(dtype=float),
    bc_vel_z: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Apply Zou-He velocity inlet or convective outflow on boundary faces.

    Launched over the entire grid.  Interior cells are early-return no-ops.
    Bounce-back faces (bc_type == 0) are also no-ops — already handled
    by the collision-stream kernel.
    """
    i, j, k = wp.tid()

    # Only operate on boundary cells
    on_xmin = (i == 0)
    on_xmax = (i == nx - 1)
    on_ymin = (j == 0)
    on_ymax = (j == ny - 1)
    on_zmin = (k == 0)
    on_zmax = (k == nz - 1)

    if not (on_xmin or on_xmax or on_ymin or on_ymax or on_zmin or on_zmax):
        return

    # Corner/edge cells: skip (default bounce-back from collision-stream).
    face_count = 0
    if on_xmin:
        face_count += 1
    if on_xmax:
        face_count += 1
    if on_ymin:
        face_count += 1
    if on_ymax:
        face_count += 1
    if on_zmin:
        face_count += 1
    if on_zmax:
        face_count += 1
    if face_count != 1:
        return

    idx = i * ny * nz + j * nz + k

    # ===================================================================
    # Face 0: x-min  (i == 0)
    # ===================================================================
    if on_xmin:
        bc = bc_types[0]
        # --- Zou-He velocity inlet --------------------------------
        if bc == 1:
            vx = bc_vel_x[0]
            vy = bc_vel_y[0]
            vz = bc_vel_z[0]
            denom = 1.0 - vx
            if denom <= 0.0:
                return
            # Known distribution sum for rho calculation
            known = (
                f[0 * stride + idx] + f[2 * stride + idx]
                + 2.0 * (
                    f[3 * stride + idx] + f[4 * stride + idx]
                    + f[5 * stride + idx] + f[6 * stride + idx]
                    + f[15 * stride + idx] + f[16 * stride + idx]
                    + f[17 * stride + idx] + f[18 * stride + idx]
                )
                + f[8 * stride + idx] + f[10 * stride + idx]
                + f[12 * stride + idx] + f[14 * stride + idx]
            )
            rho_w = known / denom

            # Incoming: f1 (+x), f7 (+x+y), f9 (+x-y), f11 (+x+z), f13 (+x-z)
            # Bounce-back of non-equilibrium from opposite directions
            f[1 * stride + idx] = (
                _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 1, 0, 0)
                + (f[2 * stride + idx] - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, -1, 0, 0))
            )
            f[7 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0)
                + (f[10 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0))
            )
            f[9 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0)
                + (f[8 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0))
            )
            f[11 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1)
                + (f[14 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1))
            )
            f[13 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1)
                + (f[12 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1))
            )
        # --- Convective outflow ---------------------------------
        elif bc == 2:
            src_idx = 1 * ny * nz + j * nz + k  # i=1
            for d in range(NUM_DIRS):
                f[d * stride + idx] = f[d * stride + src_idx]

    # ===================================================================
    # Face 1: x-max  (i == nx - 1)
    # ===================================================================
    if on_xmax:
        bc = bc_types[1]
        if bc == 1:
            vx = bc_vel_x[1]
            vy = bc_vel_y[1]
            vz = bc_vel_z[1]
            denom = 1.0 + vx
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx] + f[1 * stride + idx]
                + 2.0 * (
                    f[3 * stride + idx] + f[4 * stride + idx]
                    + f[5 * stride + idx] + f[6 * stride + idx]
                    + f[15 * stride + idx] + f[16 * stride + idx]
                    + f[17 * stride + idx] + f[18 * stride + idx]
                )
                + f[7 * stride + idx] + f[9 * stride + idx]
                + f[11 * stride + idx] + f[13 * stride + idx]
            )
            rho_w = known / denom
            # Incoming: f2 (-x), f8 (-x+y), f10 (-x-y), f12 (-x+z), f14 (-x-z)
            f[2 * stride + idx] = (
                _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, -1, 0, 0)
                + (f[1 * stride + idx] - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 1, 0, 0))
            )
            f[8 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0)
                + (f[9 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0))
            )
            f[10 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0)
                + (f[7 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0))
            )
            f[12 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1)
                + (f[13 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1))
            )
            f[14 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1)
                + (f[11 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1))
            )
        elif bc == 2:
            src_idx = (nx - 2) * ny * nz + j * nz + k
            for d in range(NUM_DIRS):
                f[d * stride + idx] = f[d * stride + src_idx]

    # ===================================================================
    # Face 2: y-min  (j == 0)
    # ===================================================================
    if on_ymin:
        bc = bc_types[2]
        if bc == 1:
            vx = bc_vel_x[2]
            vy = bc_vel_y[2]
            vz = bc_vel_z[2]
            denom = 1.0 - vy
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx] + f[4 * stride + idx]
                + 2.0 * (
                    f[1 * stride + idx] + f[2 * stride + idx]
                    + f[5 * stride + idx] + f[6 * stride + idx]
                    + f[11 * stride + idx] + f[12 * stride + idx]
                    + f[13 * stride + idx] + f[14 * stride + idx]
                )
                + f[9 * stride + idx] + f[10 * stride + idx]
                + f[16 * stride + idx] + f[18 * stride + idx]
            )
            rho_w = known / denom
            # Incoming: f3 (+y), f7 (+x+y), f8 (-x+y), f15 (+y+z), f17 (+y-z)
            f[3 * stride + idx] = (
                _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 1, 0)
                + (f[4 * stride + idx] - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, -1, 0))
            )
            f[7 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0)
                + (f[10 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0))
            )
            f[8 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0)
                + (f[9 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0))
            )
            f[15 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1)
                + (f[18 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1))
            )
            f[17 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1)
                + (f[16 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + 1 * nz + k
            for d in range(NUM_DIRS):
                f[d * stride + idx] = f[d * stride + src_idx]

    # ===================================================================
    # Face 3: y-max  (j == ny - 1)
    # ===================================================================
    if on_ymax:
        bc = bc_types[3]
        if bc == 1:
            vx = bc_vel_x[3]
            vy = bc_vel_y[3]
            vz = bc_vel_z[3]
            denom = 1.0 + vy
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx] + f[3 * stride + idx]
                + 2.0 * (
                    f[1 * stride + idx] + f[2 * stride + idx]
                    + f[5 * stride + idx] + f[6 * stride + idx]
                    + f[11 * stride + idx] + f[12 * stride + idx]
                    + f[13 * stride + idx] + f[14 * stride + idx]
                )
                + f[7 * stride + idx] + f[8 * stride + idx]
                + f[15 * stride + idx] + f[17 * stride + idx]
            )
            rho_w = known / denom
            # Incoming: f4 (-y), f9 (+x-y), f10 (-x-y), f16 (-y+z), f18 (-y-z)
            f[4 * stride + idx] = (
                _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, -1, 0)
                + (f[3 * stride + idx] - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 1, 0))
            )
            f[9 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0)
                + (f[8 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0))
            )
            f[10 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0)
                + (f[7 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0))
            )
            f[16 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1)
                + (f[17 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1))
            )
            f[18 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1)
                + (f[15 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + (ny - 2) * nz + k
            for d in range(NUM_DIRS):
                f[d * stride + idx] = f[d * stride + src_idx]

    # ===================================================================
    # Face 4: z-min  (k == 0)
    # ===================================================================
    if on_zmin:
        bc = bc_types[4]
        if bc == 1:
            vx = bc_vel_x[4]
            vy = bc_vel_y[4]
            vz = bc_vel_z[4]
            denom = 1.0 - vz
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx] + f[6 * stride + idx]
                + 2.0 * (
                    f[1 * stride + idx] + f[2 * stride + idx]
                    + f[3 * stride + idx] + f[4 * stride + idx]
                    + f[7 * stride + idx] + f[8 * stride + idx]
                    + f[9 * stride + idx] + f[10 * stride + idx]
                )
                + f[13 * stride + idx] + f[14 * stride + idx]
                + f[17 * stride + idx] + f[18 * stride + idx]
            )
            rho_w = known / denom
            # Incoming: f5 (+z), f11 (+x+z), f12 (-x+z), f15 (+y+z), f16 (-y+z)
            f[5 * stride + idx] = (
                _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, 1)
                + (f[6 * stride + idx] - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, -1))
            )
            f[11 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1)
                + (f[14 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1))
            )
            f[12 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1)
                + (f[13 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1))
            )
            f[15 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1)
                + (f[18 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1))
            )
            f[16 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1)
                + (f[17 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + j * nz + 1
            for d in range(NUM_DIRS):
                f[d * stride + idx] = f[d * stride + src_idx]

    # ===================================================================
    # Face 5: z-max  (k == nz - 1)
    # ===================================================================
    if on_zmax:
        bc = bc_types[5]
        if bc == 1:
            vx = bc_vel_x[5]
            vy = bc_vel_y[5]
            vz = bc_vel_z[5]
            denom = 1.0 + vz
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx] + f[5 * stride + idx]
                + 2.0 * (
                    f[1 * stride + idx] + f[2 * stride + idx]
                    + f[3 * stride + idx] + f[4 * stride + idx]
                    + f[7 * stride + idx] + f[8 * stride + idx]
                    + f[9 * stride + idx] + f[10 * stride + idx]
                )
                + f[11 * stride + idx] + f[12 * stride + idx]
                + f[15 * stride + idx] + f[16 * stride + idx]
            )
            rho_w = known / denom
            # Incoming: f6 (-z), f13 (+x-z), f14 (-x-z), f17 (+y-z), f18 (-y-z)
            f[6 * stride + idx] = (
                _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, -1)
                + (f[5 * stride + idx] - _f_eq(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, 1))
            )
            f[13 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1)
                + (f[12 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1))
            )
            f[14 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1)
                + (f[11 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1))
            )
            f[17 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1)
                + (f[16 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1))
            )
            f[18 * stride + idx] = (
                _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1)
                + (f[15 * stride + idx] - _f_eq(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + j * nz + (nz - 2)
            for d in range(NUM_DIRS):
                f[d * stride + idx] = f[d * stride + src_idx]


# ---------------------------------------------------------------------------
# Kernel 8 – regularized BGK (damps ghost modes for multiphase stability)
# ---------------------------------------------------------------------------


@wp.kernel
def reg_trt_kernel(
    f: wp.array(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    omega_reg: float,     # blend: 1.0 = full reg, 0.0 = no-op (keep original f)
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Regularized non-equilibrium filter (pre-collision step).

    Projects the full non-equilibrium part of *f* onto the second-order
    Hermite basis (stress tensor) and reconstructs a regularized
    distribution.  The 2nd-order Hermite projection is purely even,
    which eliminates both even ghost modes (non-hydrodynamic stress) and
    odd ghost modes (spurious currents) simultaneously.

    The regularized distribution is blended with the original *f* using
    *omega_reg*:  ``f_out = omega_reg * f_reg + (1-omega_reg) * f_in``.

    No relaxation rates are applied here — relaxation is done by the
    subsequent TRT/BGK collide-stream kernel.

    .. note::

       Full regularization eliminates the odd non-equilibrium, so the
       TRT collision on the regularized fraction reduces to BGK-like
       relaxation with ``omega_plus``.  The TRT magic parameter still
       applies to the unregularized fraction when ``omega_reg < 1``.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    # ---- skip regularization near non-periodic domain-boundary cells (2 layers)
    # Bounce-back at walls relies on odd (momentum-carrying) non-equilibrium
    # in the distributions that stream to the wall cell from the neighbour.
    # Regularizing those neighbour cells damps the odd modes before they
    # reach the wall, weakening no-slip and causing systematic mass drift
    # when body forces (gravity / SC) are present.
    #
    # Periodic faces have no bounce-back wall to protect, so cells near
    # periodic boundaries can safely participate in regularization.
    if (
        (not px and i <= 1) or (not px and i >= nx - 2)
        or (not py and j <= 1) or (not py and j >= ny - 2)
        or (not pz and k <= 1) or (not pz and k >= nz - 2)
    ):
        return

    r = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]

    # ---- accumulate non-equilibrium stress tensor Pi_ab ---------------------
    # Pi_ab = Σ_d (f_d - f_d^eq) * c_da * c_db
    pixx = 0.0
    piyy = 0.0
    pizz = 0.0
    pixy = 0.0
    piyz = 0.0
    pixz = 0.0

    feq = 0.0
    fne = 0.0

    # d=0: rest (0,0,0) — contributes nothing to Pi
    # (c_a = 0 for all a, so all products are 0)

    # d=1: +x (1,0,0) w=1/18
    feq = _f_eq(1.0 / 18.0, r, vx, vy, vz, 1, 0, 0)
    fne = f[1 * stride + idx] - feq
    pixx += fne * 1.0  # cx^2 = 1

    # d=2: -x (-1,0,0) w=1/18
    feq = _f_eq(1.0 / 18.0, r, vx, vy, vz, -1, 0, 0)
    fne = f[2 * stride + idx] - feq
    pixx += fne * 1.0  # cx^2 = 1

    # d=3: +y (0,1,0) w=1/18
    feq = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, 1, 0)
    fne = f[3 * stride + idx] - feq
    piyy += fne * 1.0  # cy^2 = 1

    # d=4: -y (0,-1,0) w=1/18
    feq = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, -1, 0)
    fne = f[4 * stride + idx] - feq
    piyy += fne * 1.0  # cy^2 = 1

    # d=5: +z (0,0,1) w=1/18
    feq = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, 0, 1)
    fne = f[5 * stride + idx] - feq
    pizz += fne * 1.0  # cz^2 = 1

    # d=6: -z (0,0,-1) w=1/18
    feq = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, 0, -1)
    fne = f[6 * stride + idx] - feq
    pizz += fne * 1.0  # cz^2 = 1

    # Edge directions (w=1/36, each contributes to multiple Pi components)
    # d=7: +x+y (1,1,0)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, 1, 0)
    fne = f[7 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * 1.0

    # d=8: -x+y (-1,1,0)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, 1, 0)
    fne = f[8 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * (-1.0)

    # d=9: +x-y (1,-1,0)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, -1, 0)
    fne = f[9 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * (-1.0)

    # d=10: -x-y (-1,-1,0)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, -1, 0)
    fne = f[10 * stride + idx] - feq
    pixx += fne * 1.0
    piyy += fne * 1.0
    pixy += fne * 1.0

    # d=11: +x+z (1,0,1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, 0, 1)
    fne = f[11 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * 1.0

    # d=12: -x+z (-1,0,1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, 0, 1)
    fne = f[12 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * (-1.0)

    # d=13: +x-z (1,0,-1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, 0, -1)
    fne = f[13 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * (-1.0)

    # d=14: -x-z (-1,0,-1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, 0, -1)
    fne = f[14 * stride + idx] - feq
    pixx += fne * 1.0
    pizz += fne * 1.0
    pixz += fne * 1.0

    # d=15: +y+z (0,1,1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, 1, 1)
    fne = f[15 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * 1.0

    # d=16: -y+z (0,-1,1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, -1, 1)
    fne = f[16 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * (-1.0)

    # d=17: +y-z (0,1,-1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, 1, -1)
    fne = f[17 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * (-1.0)

    # d=18: -y-z (0,-1,-1)
    feq = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, -1, -1)
    fne = f[18 * stride + idx] - feq
    piyy += fne * 1.0
    pizz += fne * 1.0
    piyz += fne * 1.0

    # ---- reconstruct regularized even part from stress tensor ---------------
    # f_d^(neq,reg) = w_d * (9/2) * Q_dab * Pi_ab  (Hermite projection)
    fac_face = (1.0 / 18.0) * 4.5
    fac_edge = (1.0 / 36.0) * 4.5

    # Odd-preserving even-part blend.
    #
    # The 2nd-order Hermite projection is purely even (nr_d == nr_opp for
    # every direction pair).  Blending the full distribution:
    #     f_new = omega_reg * (feq + nr) + (1-omega_reg) * f
    # damps the odd (momentum-carrying) non-equilibrium by omega_reg.
    # At wall-adjacent cells this weakens bounce-back no-slip, causing
    # systematic mass drift in multiphase simulations with body forces.
    #
    # Instead we blend ONLY the even part while fully preserving the odd
    # part — both the equilibrium momentum and the shear non-equilibrium:
    #     inc = omega_reg * (nr + 0.5*(feq_d + feq_opp - f_d - f_opp))
    #     f_new_d = f_d + inc        (inc is the same for d and opp_d)
    #     f_new_opp = f_opp + inc
    # This preserves odd non-equilibrium (f_d - f_opp unchanged) while
    # still damping even ghost modes via the Hermite projection.
    # Boundary cells (2 layers from each face) skip regularization entirely
    # to avoid any contamination of bounce-back distributions.
    #
    # Mass conservation requires d=0 to be regularized:
    #   f_0^(neq,reg) = -1/2 * (pixx + piyy + pizz)

    # d=0: rest — no opposite direction, odd part is identically zero.
    feq0 = _f_eq(1.0 / 3.0, r, vx, vy, vz, 0, 0, 0)
    nr0 = -0.5 * (pixx + piyy + pizz)
    f[0 * stride + idx] = omega_reg * (feq0 + nr0) + (1.0 - omega_reg) * f[0 * stride + idx]

    # d=1 (+x), d=2 (-x): Q_xx=2/3, Q_yy=-1/3, Q_zz=-1/3
    feq1 = _f_eq(1.0 / 18.0, r, vx, vy, vz, 1, 0, 0)
    feq2 = _f_eq(1.0 / 18.0, r, vx, vy, vz, -1, 0, 0)
    nr1 = fac_face * ((2.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz)
    _inc = omega_reg * (nr1 + 0.5 * (feq1 + feq2 - f[1 * stride + idx] - f[2 * stride + idx]))
    f[1 * stride + idx] = f[1 * stride + idx] + _inc
    f[2 * stride + idx] = f[2 * stride + idx] + _inc

    # d=3 (+y), d=4 (-y): Q_xx=-1/3, Q_yy=2/3, Q_zz=-1/3
    feq3 = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, 1, 0)
    feq4 = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, -1, 0)
    nr34 = fac_face * ((-1.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz)
    _inc = omega_reg * (nr34 + 0.5 * (feq3 + feq4 - f[3 * stride + idx] - f[4 * stride + idx]))
    f[3 * stride + idx] = f[3 * stride + idx] + _inc
    f[4 * stride + idx] = f[4 * stride + idx] + _inc

    # d=5 (+z), d=6 (-z): Q_xx=-1/3, Q_yy=-1/3, Q_zz=2/3
    feq5 = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, 0, 1)
    feq6 = _f_eq(1.0 / 18.0, r, vx, vy, vz, 0, 0, -1)
    nr56 = fac_face * ((-1.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (2.0 / 3.0) * pizz)
    _inc = omega_reg * (nr56 + 0.5 * (feq5 + feq6 - f[5 * stride + idx] - f[6 * stride + idx]))
    f[5 * stride + idx] = f[5 * stride + idx] + _inc
    f[6 * stride + idx] = f[6 * stride + idx] + _inc

    # d=7 (+x+y), d=10 (-x-y): Q_xx=2/3, Q_yy=2/3, Q_zz=-1/3, Q_xy=1
    feq7 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, 1, 0)
    feq10 = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, -1, 0)
    nr7 = fac_edge * ((2.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz + 2.0 * pixy)
    _inc = omega_reg * (nr7 + 0.5 * (feq7 + feq10 - f[7 * stride + idx] - f[10 * stride + idx]))
    f[7 * stride + idx] = f[7 * stride + idx] + _inc
    f[10 * stride + idx] = f[10 * stride + idx] + _inc

    # d=8 (-x+y), d=9 (+x-y): Q_xx=2/3, Q_yy=2/3, Q_zz=-1/3, Q_xy=-1
    feq8 = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, 1, 0)
    feq9 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, -1, 0)
    nr8 = fac_edge * ((2.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (-1.0 / 3.0) * pizz + 2.0 * (-pixy))
    _inc = omega_reg * (nr8 + 0.5 * (feq8 + feq9 - f[8 * stride + idx] - f[9 * stride + idx]))
    f[8 * stride + idx] = f[8 * stride + idx] + _inc
    f[9 * stride + idx] = f[9 * stride + idx] + _inc

    # d=11 (+x+z), d=14 (-x-z): Q_xx=2/3, Q_yy=-1/3, Q_zz=2/3, Q_xz=1
    feq11 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, 0, 1)
    feq14 = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, 0, -1)
    nr11 = fac_edge * ((2.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * pixz)
    _inc = omega_reg * (nr11 + 0.5 * (feq11 + feq14 - f[11 * stride + idx] - f[14 * stride + idx]))
    f[11 * stride + idx] = f[11 * stride + idx] + _inc
    f[14 * stride + idx] = f[14 * stride + idx] + _inc

    # d=12 (-x+z), d=13 (+x-z): Q_xx=2/3, Q_yy=-1/3, Q_zz=2/3, Q_xz=-1
    feq12 = _f_eq(1.0 / 36.0, r, vx, vy, vz, -1, 0, 1)
    feq13 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 1, 0, -1)
    nr12 = fac_edge * ((2.0 / 3.0) * pixx + (-1.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * (-pixz))
    _inc = omega_reg * (nr12 + 0.5 * (feq12 + feq13 - f[12 * stride + idx] - f[13 * stride + idx]))
    f[12 * stride + idx] = f[12 * stride + idx] + _inc
    f[13 * stride + idx] = f[13 * stride + idx] + _inc

    # d=15 (+y+z), d=18 (-y-z): Q_xx=-1/3, Q_yy=2/3, Q_zz=2/3, Q_yz=1
    feq15 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, 1, 1)
    feq18 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, -1, -1)
    nr15 = fac_edge * ((-1.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * piyz)
    _inc = omega_reg * (nr15 + 0.5 * (feq15 + feq18 - f[15 * stride + idx] - f[18 * stride + idx]))
    f[15 * stride + idx] = f[15 * stride + idx] + _inc
    f[18 * stride + idx] = f[18 * stride + idx] + _inc

    # d=16 (-y+z), d=17 (+y-z): Q_xx=-1/3, Q_yy=2/3, Q_zz=2/3, Q_yz=-1
    feq16 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, -1, 1)
    feq17 = _f_eq(1.0 / 36.0, r, vx, vy, vz, 0, 1, -1)
    nr16 = fac_edge * ((-1.0 / 3.0) * pixx + (2.0 / 3.0) * piyy + (2.0 / 3.0) * pizz + 2.0 * (-piyz))
    _inc = omega_reg * (nr16 + 0.5 * (feq16 + feq17 - f[16 * stride + idx] - f[17 * stride + idx]))
    f[16 * stride + idx] = f[16 * stride + idx] + _inc
    f[17 * stride + idx] = f[17 * stride + idx] + _inc

