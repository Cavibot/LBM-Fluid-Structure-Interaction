# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Branch-minimal periodic D3Q27 HOME kernels."""

from __future__ import annotations

import warp as wp


@wp.func
def _reconstruct_population(
    moments: wp.array(dtype=float), c: wp.vec3, weight: float, cell: int, stride: int
) -> float:
    rho = moments[cell]
    if rho <= 0.0:
        return 0.0
    jx = moments[stride + cell]
    jy = moments[2 * stride + cell]
    jz = moments[3 * stride + cell]
    sxx = moments[4 * stride + cell]
    syy = moments[5 * stride + cell]
    szz = moments[6 * stride + cell]
    sxy = moments[7 * stride + cell]
    sxz = moments[8 * stride + cell]
    syz = moments[9 * stride + cell]
    ux = jx / rho
    uy = jy / rho
    uz = jz / rho
    cx = c[0]
    cy = c[1]
    cz = c[2]
    cu = cx * ux + cy * uy + cz * uz
    cj = cx * jx + cy * jy + cz * jz
    trace_s = sxx + syy + szz
    csc = sxx * cx * cx + syy * cy * cy + szz * cz * cz
    csc += 2.0 * (sxy * cx * cy + sxz * cx * cz + syz * cy * cz)
    h2a2 = csc - trace_s / 3.0
    su_x = sxx * ux + sxy * uy + sxz * uz
    su_y = sxy * ux + syy * uy + syz * uz
    su_z = sxz * ux + syz * uy + szz * uz
    u2 = ux * ux + uy * uy + uz * uz
    trace_a3_x = 2.0 * su_x + trace_s * ux - 2.0 * rho * u2 * ux
    trace_a3_y = 2.0 * su_y + trace_s * uy - 2.0 * rho * u2 * uy
    trace_a3_z = 2.0 * su_z + trace_s * uz - 2.0 * rho * u2 * uz
    a3_ccc = 3.0 * csc * cu - 2.0 * rho * cu * cu * cu
    h3a3 = a3_ccc - (cx * trace_a3_x + cy * trace_a3_y + cz * trace_a3_z)
    return weight * (rho + 3.0 * cj + 4.5 * h2a2 + 4.5 * h3a3)


@wp.func
def _record_diagnostics(
    rho: float,
    jx: float,
    jy: float,
    jz: float,
    mxx: float,
    myy: float,
    mzz: float,
    mxy: float,
    mxz: float,
    myz: float,
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
):
    valid = wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(jx) and wp.isfinite(jy) and wp.isfinite(jz)
    valid = valid and wp.isfinite(mxx) and wp.isfinite(myy) and wp.isfinite(mzz)
    valid = valid and wp.isfinite(mxy) and wp.isfinite(mxz) and wp.isfinite(myz)
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        return
    ux = jx / rho
    uy = jy / rho
    uz = jz / rho
    nxx = mxx / rho - ux * ux
    nyy = myy / rho - uy * uy
    nzz = mzz / rho - uz * uz
    nxy = mxy / rho - ux * uy
    nxz = mxz / rho - ux * uz
    nyz = myz / rho - uy * uz
    speed_squared = ux * ux + uy * uy + uz * uz
    stress_squared = nxx * nxx + nyy * nyy + nzz * nzz
    stress_squared += 2.0 * (nxy * nxy + nxz * nxz + nyz * nyz)
    wp.atomic_min(min_density, 0, rho)
    wp.atomic_max(max_density, 0, rho)
    wp.atomic_max(max_speed_squared, 0, speed_squared)
    wp.atomic_max(max_stress_squared, 0, stress_squared)


@wp.func
def _collide_stress(
    rho: float,
    jx: float,
    jy: float,
    jz: float,
    mxx: float,
    myy: float,
    mzz: float,
    mxy: float,
    mxz: float,
    myz: float,
    shear_omega: float,
    acceleration: wp.vec3,
):
    fx = rho * acceleration[0]
    fy = rho * acceleration[1]
    fz = rho * acceleration[2]
    ux = (jx + 0.5 * fx) / rho
    uy = (jy + 0.5 * fy) / rho
    uz = (jz + 0.5 * fz) / rho
    sxx = mxx / rho
    syy = myy / rho
    szz = mzz / rho
    sxy = mxy / rho
    sxz = mxz / rho
    syz = myz / rho
    trace_s = sxx + syy + szz
    u2 = ux * ux + uy * uy + uz * uz
    keep = 1.0 - shear_omega
    post_sxx = ux * ux + keep * (sxx - trace_s / 3.0 - ux * ux + u2 / 3.0)
    post_syy = uy * uy + keep * (syy - trace_s / 3.0 - uy * uy + u2 / 3.0)
    post_szz = uz * uz + keep * (szz - trace_s / 3.0 - uz * uz + u2 / 3.0)
    post_sxy = ux * uy + keep * (sxy - ux * uy)
    post_sxz = ux * uz + keep * (sxz - ux * uz)
    post_syz = uy * uz + keep * (syz - uy * uz)
    force_power = fx * ux + fy * uy + fz * uz
    post_sxx += (fx * ux + keep * (3.0 * fx * ux - force_power) / 3.0) / rho
    post_syy += (fy * uy + keep * (3.0 * fy * uy - force_power) / 3.0) / rho
    post_szz += (fz * uz + keep * (3.0 * fz * uz - force_power) / 3.0) / rho
    force_shear = (1.0 - 0.5 * shear_omega) / rho
    post_sxy += force_shear * (fx * uy + fy * ux)
    post_sxz += force_shear * (fx * uz + fz * ux)
    post_syz += force_shear * (fy * uz + fz * uy)
    return wp.mat33(
        rho * post_sxx,
        rho * post_sxy,
        rho * post_sxz,
        rho * post_sxy,
        rho * post_syy,
        rho * post_syz,
        rho * post_sxz,
        rho * post_syz,
        rho * post_szz,
    )


@wp.kernel
def initialize_uniform_kernel(
    moments: wp.array(dtype=float), rho: float, velocity: wp.vec3, stride: int
):
    cell = wp.tid()
    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]
    moments[cell] = rho
    moments[stride + cell] = rho * ux
    moments[2 * stride + cell] = rho * uy
    moments[3 * stride + cell] = rho * uz
    moments[4 * stride + cell] = rho * ux * ux
    moments[5 * stride + cell] = rho * uy * uy
    moments[6 * stride + cell] = rho * uz * uz
    moments[7 * stride + cell] = rho * ux * uy
    moments[8 * stride + cell] = rho * ux * uz
    moments[9 * stride + cell] = rho * uy * uz


@wp.kernel
def stream_collide_periodic_kernel(
    moments_in: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    moments_out: wp.array(dtype=float),
    shear_omega: float,
    acceleration: wp.vec3,
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    rho = float(0.0)
    jx = float(0.0)
    jy = float(0.0)
    jz = float(0.0)
    qxx = float(0.0)
    qyy = float(0.0)
    qzz = float(0.0)
    qxy = float(0.0)
    qxz = float(0.0)
    qyz = float(0.0)
    for direction in range(27):
        c = directions[direction]
        source_i = (i - int(c[0]) + nx) % nx
        source_j = (j - int(c[1]) + ny) % ny
        source_k = (k - int(c[2]) + nz) % nz
        source = source_i * ny * nz + source_j * nz + source_k
        value = _reconstruct_population(moments_in, c, weights[direction], source, stride)
        cx = c[0]
        cy = c[1]
        cz = c[2]
        rho += value
        jx += value * cx
        jy += value * cy
        jz += value * cz
        qxx += value * cx * cx
        qyy += value * cy * cy
        qzz += value * cz * cz
        qxy += value * cx * cy
        qxz += value * cx * cz
        qyz += value * cy * cz

    if rho <= 0.0:
        for component in range(10):
            moments_out[component * stride + cell] = 0.0
        wp.atomic_add(invalid_count, 0, 1)
        return

    stress = _collide_stress(
        rho,
        jx,
        jy,
        jz,
        qxx - rho / 3.0,
        qyy - rho / 3.0,
        qzz - rho / 3.0,
        qxy,
        qxz,
        qyz,
        shear_omega,
        acceleration,
    )
    out_jx = jx + rho * acceleration[0]
    out_jy = jy + rho * acceleration[1]
    out_jz = jz + rho * acceleration[2]
    out_xx = stress[0, 0]
    out_yy = stress[1, 1]
    out_zz = stress[2, 2]
    out_xy = stress[0, 1]
    out_xz = stress[0, 2]
    out_yz = stress[1, 2]
    moments_out[cell] = rho
    moments_out[stride + cell] = out_jx
    moments_out[2 * stride + cell] = out_jy
    moments_out[3 * stride + cell] = out_jz
    moments_out[4 * stride + cell] = out_xx
    moments_out[5 * stride + cell] = out_yy
    moments_out[6 * stride + cell] = out_zz
    moments_out[7 * stride + cell] = out_xy
    moments_out[8 * stride + cell] = out_xz
    moments_out[9 * stride + cell] = out_yz
    _record_diagnostics(
        rho, out_jx, out_jy, out_jz, out_xx, out_yy, out_zz, out_xy, out_xz, out_yz,
        invalid_count, min_density, max_density, max_speed_squared, max_stress_squared,
    )


@wp.kernel
def collide_kernel(
    moments_in: wp.array(dtype=float),
    moments_out: wp.array(dtype=float),
    shear_omega: float,
    acceleration: wp.vec3,
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
    stride: int,
):
    cell = wp.tid()
    rho = moments_in[cell]
    jx = moments_in[stride + cell]
    jy = moments_in[2 * stride + cell]
    jz = moments_in[3 * stride + cell]
    mxx = moments_in[4 * stride + cell]
    myy = moments_in[5 * stride + cell]
    mzz = moments_in[6 * stride + cell]
    mxy = moments_in[7 * stride + cell]
    mxz = moments_in[8 * stride + cell]
    myz = moments_in[9 * stride + cell]
    valid = wp.isfinite(rho) and rho > 0.0
    valid = valid and wp.isfinite(jx) and wp.isfinite(jy) and wp.isfinite(jz)
    valid = valid and wp.isfinite(mxx) and wp.isfinite(myy) and wp.isfinite(mzz)
    valid = valid and wp.isfinite(mxy) and wp.isfinite(mxz) and wp.isfinite(myz)
    if not valid:
        wp.atomic_add(invalid_count, 0, 1)
        for component in range(10):
            moments_out[component * stride + cell] = moments_in[component * stride + cell]
        return

    stress = _collide_stress(
        rho,
        jx,
        jy,
        jz,
        mxx,
        myy,
        mzz,
        mxy,
        mxz,
        myz,
        shear_omega,
        acceleration,
    )
    out_jx = jx + rho * acceleration[0]
    out_jy = jy + rho * acceleration[1]
    out_jz = jz + rho * acceleration[2]
    moments_out[cell] = rho
    moments_out[stride + cell] = out_jx
    moments_out[2 * stride + cell] = out_jy
    moments_out[3 * stride + cell] = out_jz
    moments_out[4 * stride + cell] = stress[0, 0]
    moments_out[5 * stride + cell] = stress[1, 1]
    moments_out[6 * stride + cell] = stress[2, 2]
    moments_out[7 * stride + cell] = stress[0, 1]
    moments_out[8 * stride + cell] = stress[0, 2]
    moments_out[9 * stride + cell] = stress[1, 2]
    _record_diagnostics(
        rho,
        out_jx,
        out_jy,
        out_jz,
        stress[0, 0],
        stress[1, 1],
        stress[2, 2],
        stress[0, 1],
        stress[0, 2],
        stress[1, 2],
        invalid_count,
        min_density,
        max_density,
        max_speed_squared,
        max_stress_squared,
    )


@wp.kernel
def diagnose_kernel(
    moments: wp.array(dtype=float),
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
    stride: int,
):
    cell = wp.tid()
    _record_diagnostics(
        moments[cell],
        moments[stride + cell],
        moments[2 * stride + cell],
        moments[3 * stride + cell],
        moments[4 * stride + cell],
        moments[5 * stride + cell],
        moments[6 * stride + cell],
        moments[7 * stride + cell],
        moments[8 * stride + cell],
        moments[9 * stride + cell],
        invalid_count,
        min_density,
        max_density,
        max_speed_squared,
        max_stress_squared,
    )


@wp.kernel
def pack_diagnostics_kernel(
    invalid_count: wp.array(dtype=wp.int32),
    min_density: wp.array(dtype=float),
    max_density: wp.array(dtype=float),
    max_speed_squared: wp.array(dtype=float),
    max_stress_squared: wp.array(dtype=float),
    packed_count: wp.array(dtype=wp.int32),
    packed_values: wp.array(dtype=float),
):
    packed_count[0] = invalid_count[0]
    packed_values[0] = min_density[0]
    packed_values[1] = max_density[0]
    packed_values[2] = max_speed_squared[0]
    packed_values[3] = max_stress_squared[0]
