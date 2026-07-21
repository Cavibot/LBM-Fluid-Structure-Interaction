# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU (Warp) late-pool IF φ→φ* leveling — in-place on HOME-FREE buffers.

Keeps work on device; at most one small ``accum.numpy()`` sync for log stats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
        HomeVofGpuBuffers,
    )

CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2

_PHI_LO: float = 0.18
_PHI_HI: float = 0.82


def _ensure_scratch(buf: HomeVofGpuBuffers) -> None:
    nx, ny, nz = buf.shape
    if getattr(buf, "_heq_pool_k", None) is not None:
        shape = buf._heq_pool_k.shape
        if int(shape[0]) == nx and int(shape[1]) == ny:
            return
    device = buf.device
    buf._heq_pool_k = wp.zeros((nx, ny), dtype=wp.int32, device=device)
    buf._heq_body_w = wp.zeros((nx, ny), dtype=float, device=device)
    buf._heq_solid2d = wp.zeros((nx, ny), dtype=wp.int32, device=device)
    buf._heq_solid2d_tmp = wp.zeros((nx, ny), dtype=wp.int32, device=device)
    buf._heq_dist2d = wp.zeros((nx, ny), dtype=wp.int32, device=device)
    buf._heq_hist = wp.zeros(max(nz, 1), dtype=wp.int32, device=device)
    # 0 n_if, 1 unused, 2 sum_pr, 3 sum_r, 4 m0, 5 m1, 6 sum_spd, 7 n_plane,
    # 8 n_touch, 9 n_body0, 10 skipped, 11 phi_star, 12 do_apply
    buf._heq_accum = wp.zeros(16, dtype=float, device=device)
    buf._heq_mode = wp.zeros(1, dtype=wp.int32, device=device)


@wp.kernel
def heq_mark_pool_k_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=float),
    pool_k: wp.array2d(dtype=wp.int32),
    phi_dust: float,
    nz: int,
) -> None:
    i, j = wp.tid()
    pool_k[i, j] = -1
    for k in range(nz):
        if solid[i, j, k] < 0.0:
            continue
        if int(cell[i, j, k]) != CELL_INTERFACE:
            continue
        if phi[i, j, k] < phi_dust:
            continue
        below_ok = int(0)
        if k == 0:
            below_ok = 1
        elif solid[i, j, k - 1] < 0.0:
            below_ok = 1
        elif int(cell[i, j, k - 1]) == CELL_LIQUID:
            below_ok = 1
        if below_ok == 0:
            continue
        if k + 1 < nz and solid[i, j, k + 1] >= 0.0:
            if int(cell[i, j, k + 1]) == CELL_LIQUID:
                continue
        pool_k[i, j] = k
        break


@wp.kernel
def heq_hist_kernel(
    pool_k: wp.array2d(dtype=wp.int32),
    hist: wp.array(dtype=wp.int32),
    accum: wp.array(dtype=float),
) -> None:
    i, j = wp.tid()
    k = int(pool_k[i, j])
    if k < 0:
        return
    wp.atomic_add(hist, k, 1)
    wp.atomic_add(accum, 0, 1.0)


@wp.kernel
def heq_argmax_hist_kernel(
    hist: wp.array(dtype=wp.int32),
    mode_out: wp.array(dtype=wp.int32),
    nz: int,
) -> None:
    best_k = int(0)
    best_v = int(hist[0])
    for k in range(1, nz):
        v = int(hist[k])
        if v > best_v:
            best_v = v
            best_k = k
    mode_out[0] = best_k


@wp.kernel
def heq_clear_plane_accum_kernel(accum: wp.array(dtype=float)) -> None:
    for t in range(2, 16):
        accum[t] = 0.0


@wp.kernel
def heq_mark_solid2d_kernel(
    solid: wp.array3d(dtype=float),
    pool_k: wp.array2d(dtype=wp.int32),
    solid2d: wp.array2d(dtype=wp.int32),
    nz: int,
) -> None:
    i, j = wp.tid()
    k = int(pool_k[i, j])
    solid2d[i, j] = 0
    if k < 0:
        for kk in range(nz):
            if solid[i, j, kk] < 0.0:
                solid2d[i, j] = 1
                return
        return
    k0 = k - 1
    if k0 < 0:
        k0 = 0
    k1 = k + 1
    if k1 >= nz:
        k1 = nz - 1
    for kk in range(k0, k1 + 1):
        if solid[i, j, kk] < 0.0:
            solid2d[i, j] = 1
            return


@wp.kernel
def heq_dilate_solid2d_kernel(
    src: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.int32),
    nx: int,
    ny: int,
) -> None:
    i, j = wp.tid()
    if int(src[i, j]) != 0:
        dst[i, j] = 1
        return
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni = i + di
            nj = j + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                continue
            if int(src[ni, nj]) != 0:
                dst[i, j] = 1
                return
    dst[i, j] = 0


@wp.kernel
def heq_init_dist_kernel(
    solid2d: wp.array2d(dtype=wp.int32),
    dist: wp.array2d(dtype=wp.int32),
    max_r: int,
) -> None:
    i, j = wp.tid()
    if int(solid2d[i, j]) != 0:
        dist[i, j] = 0
    else:
        dist[i, j] = max_r + 1


@wp.kernel
def heq_dist_step_kernel(
    solid_r: wp.array2d(dtype=wp.int32),
    dist: wp.array2d(dtype=wp.int32),
    r: int,
) -> None:
    i, j = wp.tid()
    if int(dist[i, j]) <= r:
        return
    if int(solid_r[i, j]) != 0:
        dist[i, j] = r


@wp.kernel
def heq_dist_to_weight_kernel(
    dist: wp.array2d(dtype=wp.int32),
    pool_k: wp.array2d(dtype=wp.int32),
    body_w: wp.array2d(dtype=float),
    max_r: int,
    w_min: float,
) -> None:
    """Soft fade near rigid: never fully zero (except empty pool columns).

    A hard ``w=0`` ring left permanent meniscus pits / ripple rings around
    spheres while the open pool leveled to ``φ*``. Floor ``w_min`` still
    softens contact-line fighting but slowly heals the crater.
    """
    i, j = wp.tid()
    if int(pool_k[i, j]) < 0:
        body_w[i, j] = 0.0
        return
    d = int(dist[i, j])
    wm = w_min
    if wm < 0.0:
        wm = 0.0
    if wm > 1.0:
        wm = 1.0
    if d <= 0:
        # On solid footprint — no IF leveling.
        body_w[i, j] = 0.0
    elif d >= max_r:
        body_w[i, j] = 1.0
    elif d <= 1:
        body_w[i, j] = wm
    else:
        # Lerp wm → 1 over (1, max_r].
        t = float(d - 1) / float(max_r - 1)
        body_w[i, j] = wm + (1.0 - wm) * t


@wp.kernel
def heq_accum_plane_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    pool_k: wp.array2d(dtype=wp.int32),
    body_w: wp.array2d(dtype=float),
    mode_arr: wp.array(dtype=wp.int32),
    accum: wp.array(dtype=float),
) -> None:
    i, j = wp.tid()
    mode_k = int(mode_arr[0])
    k = int(pool_k[i, j])
    if k != mode_k:
        return
    if int(cell[i, j, k]) != CELL_INTERFACE:
        return
    w = body_w[i, j]
    r = rho[i, j, k]
    if r < 0.05:
        r = 1.0
    p = phi[i, j, k]
    wp.atomic_add(accum, 4, mass[i, j, k])
    wp.atomic_add(accum, 7, 1.0)
    if w < 1.0e-12:
        wp.atomic_add(accum, 9, 1.0)
    if w >= 0.5:
        wp.atomic_add(accum, 2, p * r * w)
        wp.atomic_add(accum, 3, r * w)
    spd = wp.sqrt(
        ux[i, j, k] * ux[i, j, k]
        + uy[i, j, k] * uy[i, j, k]
        + uz[i, j, k] * uz[i, j, k]
    )
    wp.atomic_add(accum, 6, spd)


@wp.kernel
def heq_decide_kernel(
    accum: wp.array(dtype=float),
    u_max: float,
    phi_lo: float,
    phi_hi: float,
) -> None:
    n_if = accum[0]
    n_plane = accum[7]
    u_mean = accum[6] / wp.max(n_plane, 1.0)
    do_apply = float(1.0)
    skipped = float(0.0)
    if n_if < 4.0 or n_plane < 4.0 or u_mean > u_max:
        do_apply = 0.0
        skipped = 1.0
    accum[10] = skipped
    sum_r = accum[3]
    phi_star = float(0.5)
    if sum_r > 1.0e-6:
        phi_star = accum[2] / sum_r
    lo = phi_lo + 0.02
    hi = phi_hi - 0.02
    if phi_star < lo:
        phi_star = lo
    if phi_star > hi:
        phi_star = hi
    accum[11] = phi_star
    accum[12] = do_apply


@wp.kernel
def heq_apply_phi_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    pool_k: wp.array2d(dtype=wp.int32),
    body_w: wp.array2d(dtype=float),
    mode_arr: wp.array(dtype=wp.int32),
    accum: wp.array(dtype=float),
    alpha: float,
    dh_cap: float,
    u_damp: float,
    phi_lo: float,
    phi_hi: float,
) -> None:
    i, j = wp.tid()
    mode_k = int(mode_arr[0])
    k = int(pool_k[i, j])
    if k != mode_k:
        return
    if int(cell[i, j, k]) != CELL_INTERFACE:
        return
    if u_damp > 0.0:
        s = 1.0 - u_damp
        ux[i, j, k] = ux[i, j, k] * s
        uy[i, j, k] = uy[i, j, k] * s
        uz[i, j, k] = uz[i, j, k] * s
    if accum[12] < 0.5:
        wp.atomic_add(accum, 5, mass[i, j, k])
        return
    w = body_w[i, j]
    a = alpha * w
    if a <= 0.0:
        wp.atomic_add(accum, 5, mass[i, j, k])
        return
    r = rho[i, j, k]
    if r < 0.05:
        r = 1.0
        rho[i, j, k] = r
    p = phi[i, j, k]
    phi_star = accum[11]
    dp = a * (phi_star - p)
    if dp > dh_cap:
        dp = dh_cap
    if dp < -dh_cap:
        dp = -dh_cap
    if wp.abs(dp) > 1.0e-12:
        wp.atomic_add(accum, 8, 1.0)
    p_new = p + dp
    if p_new < phi_lo:
        p_new = phi_lo
    if p_new > phi_hi:
        p_new = phi_hi
    phi[i, j, k] = p_new
    mass[i, j, k] = p_new * r
    wp.atomic_add(accum, 5, mass[i, j, k])


@wp.kernel
def heq_renorm_plane_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    pool_k: wp.array2d(dtype=wp.int32),
    mode_arr: wp.array(dtype=wp.int32),
    accum: wp.array(dtype=float),
) -> None:
    i, j = wp.tid()
    if accum[12] < 0.5:
        return
    m0 = accum[4]
    m1 = accum[5]
    if m1 <= 1.0e-6 or wp.abs(m1 - m0) <= 1.0e-6:
        return
    scale = m0 / m1
    mode_k = int(mode_arr[0])
    k = int(pool_k[i, j])
    if k != mode_k:
        return
    if int(cell[i, j, k]) != CELL_INTERFACE:
        return
    m = mass[i, j, k] * scale
    mass[i, j, k] = m
    r = rho[i, j, k]
    if r < 0.05:
        r = 1.0
        rho[i, j, k] = r
    phi[i, j, k] = m / r


def apply_vof_height_equation_gpu(
    buf: HomeVofGpuBuffers,
    *,
    rate: float = 0.05,
    u_max: float = 0.05,
    phi_dust: float = 0.05,
    dh_cap: float = 0.05,
    u_damp: float = 0.04,
    body_max_r: int = 4,
    body_w_min: float = 0.35,
    sync_stats: bool = True,
) -> dict[str, float]:
    """In-place GPU plane ``φ→φ*``."""
    nx, ny, nz = buf.shape
    device = buf.device
    dim2 = (nx, ny)
    _ensure_scratch(buf)

    pool_k = buf._heq_pool_k
    body_w = buf._heq_body_w
    solid2d = buf._heq_solid2d
    solid_tmp = buf._heq_solid2d_tmp
    dist2d = buf._heq_dist2d
    hist = buf._heq_hist
    accum = buf._heq_accum
    mode_arr = buf._heq_mode

    hist.zero_()
    accum.zero_()

    wp.launch(
        heq_mark_pool_k_kernel,
        dim=dim2,
        inputs=[buf.cell_type, buf.phi, buf.solid_phi, pool_k, float(phi_dust), nz],
        device=device,
    )
    wp.launch(
        heq_hist_kernel,
        dim=dim2,
        inputs=[pool_k, hist, accum],
        device=device,
    )
    wp.launch(
        heq_argmax_hist_kernel,
        dim=1,
        inputs=[hist, mode_arr, nz],
        device=device,
    )

    wp.launch(
        heq_mark_solid2d_kernel,
        dim=dim2,
        inputs=[buf.solid_phi, pool_k, solid2d, nz],
        device=device,
    )
    wp.launch(
        heq_init_dist_kernel,
        dim=dim2,
        inputs=[solid2d, dist2d, int(body_max_r)],
        device=device,
    )
    cur = solid2d
    nxt = solid_tmp
    for r in range(1, int(body_max_r) + 1):
        wp.launch(
            heq_dilate_solid2d_kernel,
            dim=dim2,
            inputs=[cur, nxt, nx, ny],
            device=device,
        )
        wp.launch(
            heq_dist_step_kernel,
            dim=dim2,
            inputs=[nxt, dist2d, int(r)],
            device=device,
        )
        cur, nxt = nxt, cur

    wp.launch(
        heq_dist_to_weight_kernel,
        dim=dim2,
        inputs=[dist2d, pool_k, body_w, int(body_max_r), float(body_w_min)],
        device=device,
    )

    wp.launch(heq_clear_plane_accum_kernel, dim=1, inputs=[accum], device=device)
    wp.launch(
        heq_accum_plane_kernel,
        dim=dim2,
        inputs=[
            buf.cell_type,
            buf.phi,
            buf.mass,
            buf.rho,
            buf.ux,
            buf.uy,
            buf.uz,
            pool_k,
            body_w,
            mode_arr,
            accum,
        ],
        device=device,
    )
    wp.launch(
        heq_decide_kernel,
        dim=1,
        inputs=[accum, float(u_max), float(_PHI_LO), float(_PHI_HI)],
        device=device,
    )
    wp.launch(
        heq_apply_phi_kernel,
        dim=dim2,
        inputs=[
            buf.cell_type,
            buf.phi,
            buf.mass,
            buf.rho,
            buf.ux,
            buf.uy,
            buf.uz,
            pool_k,
            body_w,
            mode_arr,
            accum,
            float(np.clip(rate, 0.0, 1.0)),
            float(max(dh_cap, 1.0e-4)),
            float(np.clip(u_damp, 0.0, 0.5)),
            float(_PHI_LO),
            float(_PHI_HI),
        ],
        device=device,
    )
    wp.launch(
        heq_renorm_plane_kernel,
        dim=dim2,
        inputs=[
            buf.cell_type,
            buf.phi,
            buf.mass,
            buf.rho,
            pool_k,
            mode_arr,
            accum,
        ],
        device=device,
    )

    if not sync_stats:
        return {"gpu": 1.0, "mass_delta": 0.0, "dmass": 0.0}

    a = accum.numpy()
    mode_k = int(mode_arr.numpy()[0])
    n_plane = float(a[7])
    u_mean = float(a[6] / max(n_plane, 1.0))
    return {
        "n_wet": float(a[0]),
        "n_if": float(a[0]),
        "n_plane": n_plane,
        "n_touch": float(a[8]),
        "n_drop": 0.0,
        "n_heal": 0.0,
        "n_boost": 0.0,
        "n_body_skip": float(a[9]),
        "mode_k": float(mode_k),
        "phi_star": float(a[11]),
        "H_star": float(mode_k) + float(a[11]),
        "phi_std": 0.0,
        "u_mean": u_mean,
        "alpha": float(np.clip(rate, 0.0, 1.0)),
        "n_shift": 0.0,
        "mass_before": float(a[4]),
        "mass_after": float(a[4]),
        "mass_delta": 0.0,
        "dmass": 0.0,
        "skipped": float(a[10]),
        "gpu": 1.0,
    }
