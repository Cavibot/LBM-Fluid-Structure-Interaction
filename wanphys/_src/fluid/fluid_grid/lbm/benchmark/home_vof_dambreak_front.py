# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""L0 HOME-FREE dam-break surge-front runner (no rigid / no surface cosmetics)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
    HomeVofGpuBuffers,
    alloc_home_vof_gpu,
    seed_home_vof_gpu,
    set_face_bc_gpu,
    step_home_vof_gpu,
)
from wanphys._src.fluid.fluid_grid.lbm.benchmark.martin_moyce import (
    MartinMoyceScale,
    align_T_to_anchor,
    compare_front_to_martin,
)


@dataclass(frozen=True)
class DamBreakFrontResult:
    t: np.ndarray
    z: np.ndarray
    t_aligned: np.ndarray
    scale: MartinMoyceScale
    stats: dict[str, float]
    water_mass0: float
    water_mass1: float


def bed_surge_front_cells(
    buf: HomeVofGpuBuffers,
    *,
    phi_wet: float = 0.05,
    bed_band: int = 3,
) -> float:
    """Surge tip on the dry bed: max ``i`` wet in the lowest ``bed_band`` layers.

    Whole-volume max-i can latch onto splash/airborne blobs and lag the bed tip.
    """
    wp.synchronize_device(buf.device)
    phi = buf.phi.numpy()
    cell = buf.cell_type.numpy()
    mass = buf.mass.numpy()
    nz = int(phi.shape[2])
    k_max = min(max(int(bed_band), 1), nz)
    wet = (
        ((cell[:, :, :k_max] != 0) & (phi[:, :, :k_max] > float(phi_wet)))
        | (mass[:, :, :k_max] > float(phi_wet))
    )
    if not np.any(wet):
        return 0.0
    i_max = int(np.max(np.argwhere(wet)[:, 0]))
    return float(i_max + 1)


def water_mass_inventory(buf: HomeVofGpuBuffers) -> float:
    wp.synchronize_device(buf.device)
    mass = buf.mass.numpy().astype(np.float64)
    cell = buf.cell_type.numpy()
    solid = buf.solid_phi.numpy()
    mask = (cell != 0) & (solid >= 0.0)
    return float(mass[mask].sum())


def recommended_g(n: int, *, n_ref: int = 48, g_ref: float = 3.5e-4) -> float:
    """Keep ``g·H`` roughly constant vs resolution (``H∝n``)."""
    return float(g_ref) * (float(n_ref) / max(float(n), 1.0))


def run_home_vof_dambreak_front(
    *,
    n: int = 48,
    fz: float | None = None,
    tau: float = 0.51,
    sample_every: int = 4,
    t_target: float = 2.9,
    bed_band: int = 3,
    align_gate: bool = True,
    device: str | None = None,
) -> DamBreakFrontResult:
    """Pure HOME-FREE column collapse; return Martin-scaled front history."""
    if device is None:
        wp.init()
        if wp.get_cuda_device_count() <= 0:
            raise RuntimeError("CUDA required for HOME-FREE GPU dam-break front run")
        device = "cuda:0"

    dam_x = int(n) // 4
    fill_z = int(n) // 2
    if fill_z != 2 * dam_x:
        raise ValueError(f"n^2=2 geometry requires fill_z==2*dam_x, got {dam_x=}, {fill_z=}")

    g = abs(float(fz)) if fz is not None else recommended_g(int(n))
    fz_s = -g

    bc = HomeDomainBC.all_walls()
    buf = alloc_home_vof_gpu((n, n, n), "D3Q27", device, domain_bc=bc)
    set_face_bc_gpu(buf, bc)
    seed_home_vof_gpu(buf, dam_x=dam_x, fill_z=fill_z, rho_liquid=1.0)

    scale = MartinMoyceScale(
        a_cells=float(dam_x),
        height_cells=float(fill_z),
        g_lattice=g,
        tau=float(tau),
    )
    a = float(dam_x)
    steps_needed = int(math.ceil(t_target / (scale.n * math.sqrt(g / a))))
    steps_needed = max(steps_needed, 50)

    m0 = water_mass_inventory(buf)
    t_hist = [0.0]
    z_hist = [scale.Z_from_front_cell(bed_surge_front_cells(buf, bed_band=bed_band))]

    for s in range(1, steps_needed + 1):
        step_home_vof_gpu(
            buf,
            tau=float(tau),
            fx=0.0,
            fy=0.0,
            fz=float(fz_s),
            rho_g0=1.0,
            gamma=0.0,
            home_fill_empty=False,
            home_wall_eq=True,
            seal_fg=True,
            wall_film_drain=False,
        )
        if s % int(sample_every) == 0 or s == steps_needed:
            t_hist.append(scale.T_from_steps(s))
            z_hist.append(
                scale.Z_from_front_cell(bed_surge_front_cells(buf, bed_band=bed_band))
            )

    t_arr = np.asarray(t_hist, dtype=np.float64)
    z_arr = np.asarray(z_hist, dtype=np.float64)
    t_aligned = align_T_to_anchor(t_arr, z_arr) if align_gate else t_arr.copy()
    stats = compare_front_to_martin(
        t_arr, z_arr, n2=2.0, t_min=0.8, t_max=2.8, align_gate=align_gate
    )
    m1 = water_mass_inventory(buf)
    stats["mass_rel"] = abs(m1 - m0) / max(abs(m0), 1.0e-12)
    stats.update({f"scale_{k}": v for k, v in scale.diagnostics().items()})
    return DamBreakFrontResult(
        t=t_arr,
        z=z_arr,
        t_aligned=t_aligned,
        scale=scale,
        stats=stats,
        water_mass0=m0,
        water_mass1=m1,
    )
