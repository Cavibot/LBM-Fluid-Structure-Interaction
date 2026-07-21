# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Stability-guided 16-bit fixed-point moment quantization (HOME).

Follows the VN-guided ranges from the quant HOME-LBM paper (§5.4):
store ``ρ, u, S^neq`` as packed uint16 pairs (5×uint32 / cell).

HOME-FREE stores Hermite cell-centered ``(ρ, u, S)`` (not raw ``Π`` /
``ρu`` / ``ρS``). The Hermite basis already folds ``c_s²`` into the projector
``(c_α c_β − c_s² δ_αβ)``, so equilibrium is ``S^eq ≈ u ⊗ u`` and neq is::

    S_αα^neq = S_αα − u_α²
    S_αβ^neq = S_αβ − u_α u_β

Do **not** subtract ``c_s²`` again (that was the ρ-collapse bug: liquid
``S≈0`` became ``S^neq≈−1/3``, clamped to ``−0.1``, then rebuilt as ``≈0.23``).

VOF widens ρ to ``[0, 2]`` so gas / empty cells stay representable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
        HomeVofGpuBuffers,
    )

# Paper defaults (weakly compressible single-phase); ρ widened for VOF.
RHO_MIN: float = 0.0
RHO_MAX: float = 2.0
U_MIN: float = -0.4
U_MAX: float = 0.4
SNEQ_MIN: float = -0.1
SNEQ_MAX: float = 0.1
QMAX: float = 65535.0  # 2^16 - 1


@wp.func
def _quant_u16(m: float, m_min: float, m_max: float, dither: float) -> wp.uint32:
    span = m_max - m_min
    m_clamped = m
    if m_clamped < m_min:
        m_clamped = m_min
    if m_clamped > m_max:
        m_clamped = m_max
    m_n = (m_clamped - m_min) / span
    # Spatial dither in [-0.5, 0.5] quant steps.
    m_n = m_n + dither / QMAX
    if m_n < 0.0:
        m_n = 0.0
    if m_n > 1.0:
        m_n = 1.0
    q = wp.uint32(m_n * QMAX + 0.5)
    if q > wp.uint32(65535):
        q = wp.uint32(65535)
    return q


@wp.func
def _dequant_u16(q: wp.uint32, m_min: float, m_max: float) -> float:
    span = m_max - m_min
    return m_min + float(q) * (span / QMAX)


@wp.func
def _pack2(lo: wp.uint32, hi: wp.uint32) -> wp.uint32:
    # Avoid bit-shift quirks: lo in low 16 bits, hi in high 16 bits.
    return lo + hi * wp.uint32(65536)


@wp.func
def _unpack_lo(w: wp.uint32) -> wp.uint32:
    return w - (w / wp.uint32(65536)) * wp.uint32(65536)


@wp.func
def _unpack_hi(w: wp.uint32) -> wp.uint32:
    return w / wp.uint32(65536)


@wp.func
def _dither01(i: int, j: int, k: int, salt: int) -> float:
    """Deterministic hash → roughly U[-0.5, 0.5]."""
    n = i * 73856093 ^ j * 19349663 ^ k * 83492791 ^ salt * 2654435761
    # Map low 16 bits to [0,1) then center.
    u = float(n & 65535) / 65536.0
    return u - 0.5


@wp.kernel
def pack_home_moments_u16_kernel(
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
    moment_q: wp.array4d(dtype=wp.uint32),
    rho_min: float,
    rho_max: float,
    u_min: float,
    u_max: float,
    s_min: float,
    s_max: float,
    use_dither: int,
) -> None:
    i, j, k = wp.tid()
    r = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]
    # Store Hermite S^neq (eq = u⊗u; cs² already in Hermite basis).
    sn_xx = sxx[i, j, k] - vx * vx
    sn_yy = syy[i, j, k] - vy * vy
    sn_zz = szz[i, j, k] - vz * vz
    sn_xy = sxy[i, j, k] - vx * vy
    sn_xz = sxz[i, j, k] - vx * vz
    sn_yz = syz[i, j, k] - vy * vz

    d0 = float(0.0)
    d1 = float(0.0)
    d2 = float(0.0)
    d3 = float(0.0)
    d4 = float(0.0)
    d5 = float(0.0)
    d6 = float(0.0)
    d7 = float(0.0)
    d8 = float(0.0)
    d9 = float(0.0)
    if use_dither != 0:
        d0 = _dither01(i, j, k, 1)
        d1 = _dither01(i, j, k, 2)
        d2 = _dither01(i, j, k, 3)
        d3 = _dither01(i, j, k, 4)
        d4 = _dither01(i, j, k, 5)
        d5 = _dither01(i, j, k, 6)
        d6 = _dither01(i, j, k, 7)
        d7 = _dither01(i, j, k, 8)
        d8 = _dither01(i, j, k, 9)
        d9 = _dither01(i, j, k, 10)

    q_r = _quant_u16(r, rho_min, rho_max, d0)
    q_ux = _quant_u16(vx, u_min, u_max, d1)
    q_uy = _quant_u16(vy, u_min, u_max, d2)
    q_uz = _quant_u16(vz, u_min, u_max, d3)
    q_xx = _quant_u16(sn_xx, s_min, s_max, d4)
    q_yy = _quant_u16(sn_yy, s_min, s_max, d5)
    q_zz = _quant_u16(sn_zz, s_min, s_max, d6)
    q_xy = _quant_u16(sn_xy, s_min, s_max, d7)
    q_xz = _quant_u16(sn_xz, s_min, s_max, d8)
    q_yz = _quant_u16(sn_yz, s_min, s_max, d9)

    moment_q[i, j, k, 0] = _pack2(q_r, q_ux)
    moment_q[i, j, k, 1] = _pack2(q_uy, q_uz)
    moment_q[i, j, k, 2] = _pack2(q_xx, q_yy)
    moment_q[i, j, k, 3] = _pack2(q_zz, q_xy)
    moment_q[i, j, k, 4] = _pack2(q_xz, q_yz)


@wp.kernel
def unpack_home_moments_u16_kernel(
    moment_q: wp.array4d(dtype=wp.uint32),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    sxx: wp.array3d(dtype=float),
    syy: wp.array3d(dtype=float),
    szz: wp.array3d(dtype=float),
    sxy: wp.array3d(dtype=float),
    sxz: wp.array3d(dtype=float),
    syz: wp.array3d(dtype=float),
    rho_min: float,
    rho_max: float,
    u_min: float,
    u_max: float,
    s_min: float,
    s_max: float,
) -> None:
    i, j, k = wp.tid()
    w0 = moment_q[i, j, k, 0]
    w1 = moment_q[i, j, k, 1]
    w2 = moment_q[i, j, k, 2]
    w3 = moment_q[i, j, k, 3]
    w4 = moment_q[i, j, k, 4]

    r = _dequant_u16(_unpack_lo(w0), rho_min, rho_max)
    vx = _dequant_u16(_unpack_hi(w0), u_min, u_max)
    vy = _dequant_u16(_unpack_lo(w1), u_min, u_max)
    vz = _dequant_u16(_unpack_hi(w1), u_min, u_max)
    sn_xx = _dequant_u16(_unpack_lo(w2), s_min, s_max)
    sn_yy = _dequant_u16(_unpack_hi(w2), s_min, s_max)
    sn_zz = _dequant_u16(_unpack_lo(w3), s_min, s_max)
    sn_xy = _dequant_u16(_unpack_hi(w3), s_min, s_max)
    sn_xz = _dequant_u16(_unpack_lo(w4), s_min, s_max)
    sn_yz = _dequant_u16(_unpack_hi(w4), s_min, s_max)

    rho[i, j, k] = r
    ux[i, j, k] = vx
    uy[i, j, k] = vy
    uz[i, j, k] = vz
    # Reconstruct Hermite S from S^neq + u⊗u.
    sxx[i, j, k] = sn_xx + vx * vx
    syy[i, j, k] = sn_yy + vy * vy
    szz[i, j, k] = sn_zz + vz * vz
    sxy[i, j, k] = sn_xy + vx * vy
    sxz[i, j, k] = sn_xz + vx * vz
    syz[i, j, k] = sn_yz + vy * vz


@wp.func
def load_home_moments_u16(
    moment_q: wp.array4d(dtype=wp.uint32),
    i: int,
    j: int,
    k: int,
    rho_min: float,
    rho_max: float,
    u_min: float,
    u_max: float,
    s_min: float,
    s_max: float,
):
    """Dequant one cell from packed SoT → (ρ,u,S) Hermite moments."""
    w0 = moment_q[i, j, k, 0]
    w1 = moment_q[i, j, k, 1]
    w2 = moment_q[i, j, k, 2]
    w3 = moment_q[i, j, k, 3]
    w4 = moment_q[i, j, k, 4]
    r = _dequant_u16(_unpack_lo(w0), rho_min, rho_max)
    vx = _dequant_u16(_unpack_hi(w0), u_min, u_max)
    vy = _dequant_u16(_unpack_lo(w1), u_min, u_max)
    vz = _dequant_u16(_unpack_hi(w1), u_min, u_max)
    sn_xx = _dequant_u16(_unpack_lo(w2), s_min, s_max)
    sn_yy = _dequant_u16(_unpack_hi(w2), s_min, s_max)
    sn_zz = _dequant_u16(_unpack_lo(w3), s_min, s_max)
    sn_xy = _dequant_u16(_unpack_hi(w3), s_min, s_max)
    sn_xz = _dequant_u16(_unpack_lo(w4), s_min, s_max)
    sn_yz = _dequant_u16(_unpack_hi(w4), s_min, s_max)
    return (
        r,
        vx,
        vy,
        vz,
        sn_xx + vx * vx,
        sn_yy + vy * vy,
        sn_zz + vz * vz,
        sn_xy + vx * vy,
        sn_xz + vx * vz,
        sn_yz + vy * vz,
    )


def ensure_moment_quant_buffers(buf: HomeVofGpuBuffers) -> None:
    """Allocate packed moment SoT on ``buf`` if missing."""
    nx, ny, nz = buf.shape
    q = getattr(buf, "moment_q", None)
    if q is not None:
        shape = q.shape
        if int(shape[0]) == nx and int(shape[1]) == ny and int(shape[2]) == nz:
            return
    device = buf.device
    buf.moment_q = wp.zeros((nx, ny, nz, 5), dtype=wp.uint32, device=device)
    buf.moment_quant_ready = False


def pack_moments_from_float(
    buf: HomeVofGpuBuffers,
    *,
    dither: bool = True,
) -> None:
    """Pack primary float moments → persistent ``buf.moment_q``."""
    ensure_moment_quant_buffers(buf)
    nx, ny, nz = buf.shape
    wp.launch(
        pack_home_moments_u16_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.rho,
            buf.ux,
            buf.uy,
            buf.uz,
            buf.sxx,
            buf.syy,
            buf.szz,
            buf.sxy,
            buf.sxz,
            buf.syz,
            buf.moment_q,
            float(RHO_MIN),
            float(RHO_MAX),
            float(U_MIN),
            float(U_MAX),
            float(SNEQ_MIN),
            float(SNEQ_MAX),
            1 if dither else 0,
        ],
        device=buf.device,
    )
    buf.moment_quant_ready = True


def unpack_moments_to_float(buf: HomeVofGpuBuffers) -> None:
    """Unpack ``buf.moment_q`` → primary float moments (debug / sync helper).

    Persistent quant path loads from ``moment_q`` inside the fused kernel and
    does not need a full-field unpack each step.
    """
    if not getattr(buf, "moment_quant_ready", False):
        return
    nx, ny, nz = buf.shape
    wp.launch(
        unpack_home_moments_u16_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.moment_q,
            buf.rho,
            buf.ux,
            buf.uy,
            buf.uz,
            buf.sxx,
            buf.syy,
            buf.szz,
            buf.sxy,
            buf.sxz,
            buf.syz,
            float(RHO_MIN),
            float(RHO_MAX),
            float(U_MIN),
            float(U_MAX),
            float(SNEQ_MIN),
            float(SNEQ_MAX),
        ],
        device=buf.device,
    )


def moment_quant_bytes_per_cell() -> int:
    """Packed moment SoT footprint (5×uint32)."""
    return 5 * 4


def moment_float_bytes_per_cell() -> int:
    """Working float moment footprint (10×float32, single set)."""
    return 10 * 4


def moment_persistent_bytes_per_cell() -> int:
    """Paper FSI-hybrid: one float working set + one quant SoT."""
    return moment_float_bytes_per_cell() + moment_quant_bytes_per_cell()


def moment_fp32_double_bytes_per_cell() -> int:
    """Baseline fp32 ping-pong (10×float × 2)."""
    return 2 * moment_float_bytes_per_cell()
