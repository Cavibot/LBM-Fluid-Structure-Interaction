# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Soft-maintain hydrostatic ``ρ(z)`` so link ME (Eq.32) can see Archimedes.

Under Guo body force with nearly uniform ``ρ≈1``, ``f*+f−2w≈0`` and vertical
ME lift vanishes. Seeding ``ρ(z)`` alone is not enough after a dam break —
this pass gently blends liquid/IF density toward

    ρ_h = ρ0 · (1 + |g| · depth / c_s²)

using the **modal** free-surface height, then **rescales** wet ``ρ``/``mass``
so ``Σmass`` is unchanged (no invented water).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
        HomeVofGpuBuffers,
    )

CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2


def _ensure_scratch(buf: HomeVofGpuBuffers, nz: int) -> None:
    nx, ny, _ = buf.shape
    pk = getattr(buf, "_hydro_fs_k", None)
    hist = getattr(buf, "_hydro_hist", None)
    mode = getattr(buf, "_hydro_mode", None)
    mass_acc = getattr(buf, "_hydro_mass_acc", None)
    ok = (
        pk is not None
        and hist is not None
        and mode is not None
        and mass_acc is not None
        and int(pk.shape[0]) == nx
        and int(pk.shape[1]) == ny
        and int(hist.shape[0]) == max(nz, 1)
    )
    if ok:
        return
    device = buf.device
    buf._hydro_fs_k = wp.zeros((nx, ny), dtype=wp.int32, device=device)
    buf._hydro_hist = wp.zeros(max(nz, 1), dtype=wp.int32, device=device)
    buf._hydro_mode = wp.zeros(1, dtype=wp.int32, device=device)
    # [0]=Σmass before, [1]=Σmass after blend
    buf._hydro_mass_acc = wp.zeros(2, dtype=float, device=device)


@wp.kernel
def hydro_mark_fs_k_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=float),
    fs_k: wp.array2d(dtype=wp.int32),
    phi_wet: float,
    nz: int,
) -> None:
    """Topmost liquid / wet IF in each column (cell-center free-surface index)."""
    i, j = wp.tid()
    fs_k[i, j] = -1
    for t in range(nz):
        k = nz - 1 - t
        if solid[i, j, k] < 0.0:
            continue
        ct = int(cell[i, j, k])
        if ct == CELL_LIQUID:
            fs_k[i, j] = k
            return
        if ct == CELL_INTERFACE and phi[i, j, k] > phi_wet:
            fs_k[i, j] = k
            return


@wp.kernel
def hydro_clear_hist_kernel(hist: wp.array(dtype=wp.int32)) -> None:
    i = wp.tid()
    hist[i] = 0


@wp.kernel
def hydro_hist_kernel(
    fs_k: wp.array2d(dtype=wp.int32),
    hist: wp.array(dtype=wp.int32),
) -> None:
    i, j = wp.tid()
    k = int(fs_k[i, j])
    if k < 0:
        return
    wp.atomic_add(hist, k, 1)


@wp.kernel
def hydro_argmax_hist_kernel(
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
def hydro_sum_mass_kernel(
    cell: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    acc_idx: int,
    acc: wp.array(dtype=float),
) -> None:
    i, j, k = wp.tid()
    if solid[i, j, k] < 0.0:
        return
    ct = int(cell[i, j, k])
    if ct != CELL_LIQUID and ct != CELL_INTERFACE:
        return
    wp.atomic_add(acc, acc_idx, mass[i, j, k])


@wp.kernel
def hydro_blend_rho_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=float),
    mode_k: wp.array(dtype=wp.int32),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    rho0: float,
    g_abs: float,
    cs2: float,
    alpha: float,
    rho_min: float,
    rho_max: float,
) -> None:
    i, j, k = wp.tid()
    if solid[i, j, k] < 0.0:
        return
    k_fs = int(mode_k[0])
    if k_fs < 0:
        return
    if k > k_fs:
        return
    ct = int(cell[i, j, k])
    if ct != CELL_LIQUID and ct != CELL_INTERFACE:
        return

    depth = float(k_fs) - (float(k) + 0.5)
    if depth < 0.0:
        depth = 0.0
    rho_h = rho0 * (1.0 + g_abs * depth / cs2)
    if rho_h < rho_min:
        rho_h = rho_min
    if rho_h > rho_max:
        rho_h = rho_max

    r = rho[i, j, k]
    r_new = (1.0 - alpha) * r + alpha * rho_h
    if r_new < rho_min:
        r_new = rho_min
    if r_new > rho_max:
        r_new = rho_max
    rho[i, j, k] = r_new

    if ct == CELL_LIQUID:
        mass[i, j, k] = r_new
    else:
        p = phi[i, j, k]
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        mass[i, j, k] = p * r_new


@wp.kernel
def hydro_rescale_mass_kernel(
    cell: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    acc: wp.array(dtype=float),
) -> None:
    """Scale wet ρ/mass so Σmass matches pre-blend inventory (no clamp)."""
    i, j, k = wp.tid()
    if solid[i, j, k] < 0.0:
        return
    ct = int(cell[i, j, k])
    if ct != CELL_LIQUID and ct != CELL_INTERFACE:
        return
    m0 = acc[0]
    m1 = acc[1]
    if m0 <= 0.0 or m1 <= 0.0:
        return
    s = m0 / m1
    rho[i, j, k] = rho[i, j, k] * s
    mass[i, j, k] = mass[i, j, k] * s


def apply_hydrostatic_rho_gpu(
    buf: HomeVofGpuBuffers,
    *,
    rho0: float,
    g_latt_z: float,
    alpha: float,
    cs2: float = 1.0 / 3.0,
    phi_wet: float = 0.25,
    rho_min_frac: float = 0.90,
    rho_max_frac: float = 1.20,
) -> dict[str, float]:
    """Blend toward modal-plane hydrostatic ``ρ(z)``, then restore ``Σmass``.

    Returns ``{mass_before, mass_after_blend, mass_after_scale, scale}``.
    """
    g_abs = abs(float(g_latt_z))
    a = float(alpha)
    if g_abs <= 0.0 or a <= 0.0:
        return {
            "mass_before": 0.0,
            "mass_after_blend": 0.0,
            "mass_after_scale": 0.0,
            "scale": 1.0,
        }
    nx, ny, nz = buf.shape
    _ensure_scratch(buf, nz)
    r0 = float(rho0)
    rmin = r0 * float(rho_min_frac)
    rmax = r0 * float(rho_max_frac)
    buf._hydro_mass_acc.zero_()

    wp.launch(
        hydro_clear_hist_kernel,
        dim=max(nz, 1),
        inputs=[buf._hydro_hist],
        device=buf.device,
    )
    wp.launch(
        hydro_mark_fs_k_kernel,
        dim=(nx, ny),
        inputs=[
            buf.cell_type,
            buf.phi,
            buf.solid_phi,
            buf._hydro_fs_k,
            float(phi_wet),
            int(nz),
        ],
        device=buf.device,
    )
    wp.launch(
        hydro_hist_kernel,
        dim=(nx, ny),
        inputs=[buf._hydro_fs_k, buf._hydro_hist],
        device=buf.device,
    )
    wp.launch(
        hydro_argmax_hist_kernel,
        dim=1,
        inputs=[buf._hydro_hist, buf._hydro_mode, int(nz)],
        device=buf.device,
    )
    wp.launch(
        hydro_sum_mass_kernel,
        dim=(nx, ny, nz),
        inputs=[buf.cell_type, buf.solid_phi, buf.mass, 0, buf._hydro_mass_acc],
        device=buf.device,
    )
    wp.launch(
        hydro_blend_rho_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.cell_type,
            buf.phi,
            buf.solid_phi,
            buf._hydro_mode,
            buf.rho,
            buf.mass,
            r0,
            g_abs,
            float(cs2),
            a,
            rmin,
            rmax,
        ],
        device=buf.device,
    )
    wp.launch(
        hydro_sum_mass_kernel,
        dim=(nx, ny, nz),
        inputs=[buf.cell_type, buf.solid_phi, buf.mass, 1, buf._hydro_mass_acc],
        device=buf.device,
    )
    wp.launch(
        hydro_rescale_mass_kernel,
        dim=(nx, ny, nz),
        inputs=[
            buf.cell_type,
            buf.solid_phi,
            buf.rho,
            buf.mass,
            buf._hydro_mass_acc,
        ],
        device=buf.device,
    )
    acc = buf._hydro_mass_acc.numpy()
    m0 = float(acc[0])
    m1 = float(acc[1])
    scale = (m0 / m1) if (m0 > 0.0 and m1 > 0.0) else 1.0
    return {
        "mass_before": m0,
        "mass_after_blend": m1,
        "mass_after_scale": m0,
        "scale": scale,
    }
