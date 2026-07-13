# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""PLIC curvature for FSLBM surface tension (Lehmann 2019 / FluidX3D).

Implements Parker–Youngs normals, analytic plane–cube PLIC offsets, and
least-squares Monge-patch mean curvature. Used in Eq. (12):

    ρ_g = (p_g - 2 γ κ) / c_s² = ρ_atm - 6 γ κ   (c_s² = 1/3)
"""

from __future__ import annotations

import warp as wp

from .vof_kernels import CELL_INTERFACE, _wrap_index

vec5 = wp.types.vector(length=5, dtype=float)
vec25 = wp.types.vector(length=25, dtype=float)


@wp.func
def _sq(x: float) -> float:
    return x * x


@wp.func
def _cb(x: float) -> float:
    return x * x * x


@wp.func
def _sample_phi(
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    di: int,
    dj: int,
    dk: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    phi_c: float,
) -> float:
    """Sample φ; solids / OOB use center value (Neumann)."""
    si = _wrap_index(i + di, nx, px)
    sj = _wrap_index(j + dj, ny, py)
    sk = _wrap_index(k + dk, nz, pz)
    if si < 0 or sj < 0 or sk < 0:
        return phi_c
    if solid_phi[si, sj, sk] < 0.0:
        return phi_c
    return phi[si, sj, sk]


@wp.func
def _parker_youngs_normal(
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
    phi_c: float,
) -> wp.vec3:
    """Parker–Youngs normal on the 3×3×3 stencil (FluidX3D / Pohl)."""
    nx_s = float(0.0)
    ny_s = float(0.0)
    nz_s = float(0.0)
    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                zeros = int(0)
                if di == 0:
                    zeros = zeros + 1
                if dj == 0:
                    zeros = zeros + 1
                if dk == 0:
                    zeros = zeros + 1
                w = float(1.0)
                if zeros == 2:
                    w = 4.0
                elif zeros == 1:
                    w = 2.0
                p = _sample_phi(
                    phi, solid_phi, i, j, k, di, dj, dk, px, py, pz, nx, ny, nz, phi_c
                )
                # FluidX3D: n ~ sum w (φ_{-c} - φ_{+c}) ≡ -∇φ (outward from liquid).
                nx_s = nx_s + w * float(-di) * p
                ny_s = ny_s + w * float(-dj) * p
                nz_s = nz_s + w * float(-dk) * p
    nlen = wp.sqrt(nx_s * nx_s + ny_s * ny_s + nz_s * nz_s)
    if nlen < 1.0e-8:
        return wp.vec3(0.0, 0.0, 1.0)
    inv = 1.0 / nlen
    return wp.vec3(nx_s * inv, ny_s * inv, nz_s * inv)


@wp.func
def _plic_cube_reduced(V: float, n1: float, n2: float, n3: float) -> float:
    """Analytic plane–cube offset (Scardovelli–Zaleski / Kawano / Lehmann)."""
    n12 = n1 + n2
    n3V = n3 * V
    if n12 <= 2.0 * n3V:
        return n3V + 0.5 * n12
    sqn1 = _sq(n1)
    n26 = 6.0 * n2
    v1 = sqn1 / n26
    if v1 <= n3V and n3V < v1 + 0.5 * (n2 - n1):
        return 0.5 * (n1 + wp.sqrt(sqn1 + 8.0 * n2 * (n3V - v1)))
    V6 = n1 * n26 * n3V
    if n3V < v1:
        return wp.pow(wp.max(V6, 0.0), 1.0 / 3.0)
    if n3 < n12:
        v3 = (
            _sq(n3) * (3.0 * n12 - n3)
            + sqn1 * (n1 - 3.0 * n3)
            + _sq(n2) * (n2 - 3.0 * n3)
        ) / (n1 * n26)
    else:
        v3 = 0.5 * n12
    sqn12 = sqn1 + _sq(n2)
    V6cbn12 = V6 - _cb(n1) - _cb(n2)
    case34 = n3V < v3
    if case34:
        a = V6cbn12
        b = sqn12
        c = n12
    else:
        a = 0.5 * (V6cbn12 - _cb(n3))
        b = 0.5 * (sqn12 + _sq(n3))
        c = 0.5
    t2 = _sq(c) - b
    if t2 <= 1.0e-12:
        return c
    t = wp.sqrt(t2)
    arg = (_cb(c) - 0.5 * a - 1.5 * b * c) / _cb(t)
    arg = wp.clamp(arg, -1.0, 1.0)
    return c - 2.0 * t * wp.sin(0.33333334 * wp.asin(arg))


@wp.func
def _plic_cube(V0: float, n: wp.vec3) -> float:
    """Unit-cube PLIC plane offset for volume fraction V0 and normal n."""
    ax = wp.abs(n[0])
    ay = wp.abs(n[1])
    az = wp.abs(n[2])
    V = 0.5 - wp.abs(V0 - 0.5)
    l = ax + ay + az
    if l < 1.0e-12:
        return 0.0
    n1 = wp.min(wp.min(ax, ay), az) / l
    n3 = wp.max(wp.max(ax, ay), az) / l
    n2 = wp.max(1.0 - n1 - n3, 0.0)
    d = _plic_cube_reduced(V, n1, n2, n3)
    return l * wp.sign(V0 - 0.5) * wp.abs(0.5 - d)


@wp.func
def _lu_solve5(M_in: vec25, b: vec5) -> vec5:
    """In-place LU solve for a 5×5 row-major system (FluidX3D style)."""
    M = vec25()
    for i in range(25):
        M[i] = M_in[i]
    x = vec5()
    for i in range(5):
        diag = M[5 * i + i]
        if wp.abs(diag) < 1.0e-12:
            return vec5()
        for j in range(i + 1, 5):
            M[5 * j + i] = M[5 * j + i] / diag
            lij = M[5 * j + i]
            for k in range(i + 1, 5):
                M[5 * j + k] = M[5 * j + k] - lij * M[5 * i + k]
    for i in range(5):
        x[i] = b[i]
        for k in range(i):
            x[i] = x[i] - M[5 * i + k] * x[k]
    for ii in range(5):
        i = 4 - ii
        for k in range(i + 1, 5):
            x[i] = x[i] - M[5 * i + k] * x[k]
        x[i] = x[i] / M[5 * i + i]
    return x


@wp.func
def _plic_curvature(
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """Mean curvature via PLIC offsets + Monge least-squares fit."""
    phi_c = phi[i, j, k]
    bz = _parker_youngs_normal(
        phi, solid_phi, i, j, k, px, py, pz, nx, ny, nz, phi_c
    )
    rn = wp.vec3(0.56270900, 0.32704452, 0.75921047)
    by = wp.normalize(wp.cross(bz, rn))
    bx = wp.cross(by, bz)

    center_offset = _plic_cube(phi_c, bz)
    M = vec25()
    b = vec5()
    for t in range(25):
        M[t] = 0.0
    for t in range(5):
        b[t] = 0.0
    number = int(0)

    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                p = _sample_phi(
                    phi, solid_phi, i, j, k, di, dj, dk, px, py, pz, nx, ny, nz, phi_c
                )
                if p <= 0.0 or p >= 1.0:
                    continue
                ei = wp.vec3(float(di), float(dj), float(dk))
                offset = _plic_cube(p, bz) - center_offset
                x = wp.dot(ei, bx)
                y = wp.dot(ei, by)
                z = wp.dot(ei, bz) + offset
                x2 = x * x
                y2 = y * y
                x3 = x2 * x
                y3 = y2 * y
                M[0] = M[0] + x2 * x2
                M[1] = M[1] + x2 * y2
                M[2] = M[2] + x3 * y
                M[3] = M[3] + x3
                M[4] = M[4] + x2 * y
                b[0] = b[0] + x2 * z
                M[6] = M[6] + y2 * y2
                M[7] = M[7] + x * y3
                M[8] = M[8] + x * y2
                M[9] = M[9] + y3
                b[1] = b[1] + y2 * z
                M[12] = M[12] + x2 * y2
                M[13] = M[13] + x2 * y
                M[14] = M[14] + x * y2
                b[2] = b[2] + x * y * z
                M[18] = M[18] + x2
                M[19] = M[19] + x * y
                b[3] = b[3] + x * z
                M[24] = M[24] + y2
                b[4] = b[4] + y * z
                number = number + 1

    if number < 5:
        return 0.0

    # Symmetrize M.
    for r in range(1, 5):
        for col in range(r):
            M[r * 5 + col] = M[col * 5 + r]

    coeff = _lu_solve5(M, b)
    A = coeff[0]
    B = coeff[1]
    C = coeff[2]
    H = coeff[3]
    I = coeff[4]
    denom = H * H + I * I + 1.0
    K = (A * (I * I + 1.0) + B * (H * H + 1.0) - C * H * I) * wp.pow(
        1.0 / denom, 1.5
    )
    return wp.clamp(K, -1.0, 1.0)


@wp.kernel
def vof_compute_kappa_kernel(
    phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    kappa: wp.array3d(dtype=float),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Write PLIC mean curvature on interface cells; zero elsewhere."""
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        kappa[i, j, k] = 0.0
        return
    if int(cell_type[i, j, k]) != CELL_INTERFACE:
        kappa[i, j, k] = 0.0
        return
    kappa[i, j, k] = _plic_curvature(
        phi, solid_phi, i, j, k, px, py, pz, nx, ny, nz
    )


@wp.kernel
def vof_smooth_kappa_kernel(
    kappa_in: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    kappa_out: wp.array3d(dtype=float),
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """One Jacobi pass: average κ with face-neighbor interface cells.

    Reduces lattice-noise capillary forces that pin frozen surface bumps.
    """
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0 or int(cell_type[i, j, k]) != CELL_INTERFACE:
        kappa_out[i, j, k] = 0.0
        return

    acc = kappa_in[i, j, k]
    wsum = float(1.0)
    # ±x, ±y, ±z face neighbors only.
    for axis in range(3):
        for sgn in range(2):
            di = 0
            dj = 0
            dk = 0
            step = 1 if sgn == 0 else -1
            if axis == 0:
                di = step
            elif axis == 1:
                dj = step
            else:
                dk = step
            si = _wrap_index(i + di, nx, px)
            sj = _wrap_index(j + dj, ny, py)
            sk = _wrap_index(k + dk, nz, pz)
            if si < 0 or sj < 0 or sk < 0:
                continue
            if solid_phi[si, sj, sk] < 0.0:
                continue
            if int(cell_type[si, sj, sk]) != CELL_INTERFACE:
                continue
            acc = acc + kappa_in[si, sj, sk]
            wsum = wsum + 1.0
    kappa_out[i, j, k] = acc / wsum
