# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
"""Headless probe: late-pool |u|, neighbor |Δφ|, |Δmass| on INTERFACE.

Theory: near rest, Δφ between neighbors can remain, but Körner δm ~ θ u → 0.

  uv run python wanphys/examples/lbm/_debug_korner_flux.py
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
    CELL_INTERFACE,
)


def _iface_neighbor_dphi(phi: np.ndarray, iface: np.ndarray) -> tuple[float, float, int]:
    """Mean/max |Δφ| on horizontal face-neighbor INTERFACE pairs."""
    dphis = []
    # +x
    m = iface[:-1, :, :] & iface[1:, :, :]
    if m.any():
        dphis.append(np.abs(phi[:-1, :, :][m] - phi[1:, :, :][m]))
    # +y
    m = iface[:, :-1, :] & iface[:, 1:, :]
    if m.any():
        dphis.append(np.abs(phi[:, :-1, :][m] - phi[:, 1:, :][m]))
    if not dphis:
        return 0.0, 0.0, 0
    arr = np.concatenate(dphis)
    return float(arr.mean()), float(arr.max()), int(arr.size)


def _korner_proxy_from_u(phi: np.ndarray, iface: np.ndarray, ux, uy, uz) -> tuple[float, float]:
    """Cheap proxy |δm| ~ θ |u·n| on I–I face pairs (equilibrium leading term).

    Near equilibrium, f_in - f_opp ∝ c·u, so |δm| ∝ θ |u_n|.
    """
    vals = []
    # +x faces
    m = iface[:-1, :, :] & iface[1:, :, :]
    if m.any():
        th = 0.5 * (phi[:-1, :, :][m] + phi[1:, :, :][m])
        # characteristic speed along link: average |ux|
        un = 0.5 * (np.abs(ux[:-1, :, :][m]) + np.abs(ux[1:, :, :][m]))
        vals.append(th * un)
    # +y faces
    m = iface[:, :-1, :] & iface[:, 1:, :]
    if m.any():
        th = 0.5 * (phi[:, :-1, :][m] + phi[:, 1:, :][m])
        un = 0.5 * (np.abs(uy[:, :-1, :][m]) + np.abs(uy[:, 1:, :][m]))
        vals.append(th * un)
    if not vals:
        return 0.0, 0.0
    arr = np.concatenate(vals)
    return float(arr.mean()), float(arr.max())


def main() -> int:
    wp.init()
    n = 48
    dh = 1.28 / n
    model = LbmModel(
        fluid_grid_res=(n, n, n),
        fluid_grid_cell_size=dh,
        lattice="D3Q27",
        tau=0.51,
        G=0.0,
        phase_mode="vof_sharp",
        lbm_backend="home_fp32",
        vof_rho_gas=1.0,
        vof_epsilon=1e-3,
        vof_gamma=1.5e-3,
        vof_kappa_smooth=2,
        vof_wall_wetting=0.0,
        vof_seal_fg=True,
        vof_quiet_fill=False,
        initial_density=1.0,
        gravity_z=-0.0020,
    )
    dom = LbmDomain(model)
    dom.create_state()
    home = dom.solver._home_fp32
    assert home is not None
    home.seed_dam_break(dom.state, dam_x=n // 4, fill_z=n // 2)

    # ~t=10 s: example uses ~12 substeps/frame @ 60 fps → 720 steps/s.
    warm = 720 * 10
    print(f"warming {warm} lattice steps (~t=10s)...", flush=True)
    for i in range(warm):
        dom.step(1.0)
        if (i + 1) % 2400 == 0:
            print(f"  ... {i+1}/{warm}", flush=True)
    wp.synchronize_device(model._device)

    buf = home._ensure_gpu()
    probe_steps = 60
    print(
        f"\nprobe {probe_steps} steps at late time\n"
        f"{'step':>5} {'|u|_I':>10} {'|Δφ|':>10} {'|Δφ|max':>10} "
        f"{'|Δm|_I':>12} {'|Δm|max':>12} {'θ|u_n|':>10}",
        flush=True,
    )

    dmass_means: list[float] = []
    u_means: list[float] = []
    dphi_means: list[float] = []
    proxy_means: list[float] = []

    for s in range(probe_steps):
        phi0 = buf.phi.numpy()
        ctype0 = buf.cell_type.numpy()
        mass0 = buf.mass.numpy()
        ux = buf.ux.numpy()
        uy = buf.uy.numpy()
        uz = buf.uz.numpy()
        iface = ctype0 == CELL_INTERFACE
        n_i = int(iface.sum())
        if n_i == 0:
            print("no interface cells", flush=True)
            break
        speed = np.sqrt(ux[iface] ** 2 + uy[iface] ** 2 + uz[iface] ** 2)
        u_mean = float(speed.mean())
        dphi_mean, dphi_max, _ = _iface_neighbor_dphi(phi0, iface)
        proxy_mean, _ = _korner_proxy_from_u(phi0, iface, ux, uy, uz)

        dom.step(1.0)
        wp.synchronize_device(model._device)
        mass1 = buf.mass.numpy()
        dabs = np.abs(mass1[iface] - mass0[iface])
        dm_mean = float(dabs.mean())
        dm_max = float(dabs.max())

        dmass_means.append(dm_mean)
        u_means.append(u_mean)
        dphi_means.append(dphi_mean)
        proxy_means.append(proxy_mean)

        if s % 2 == 0 or s < 5:
            print(
                f"{s:5d} {u_mean:10.4e} {dphi_mean:10.4e} {dphi_max:10.4e} "
                f"{dm_mean:12.4e} {dm_max:12.4e} {proxy_mean:10.4e}",
                flush=True,
            )

    print("\n=== late-window summary ===", flush=True)
    print(
        f"n_I (last): {n_i}",
        flush=True,
    )
    print(
        f"|u|_I:      mean={np.mean(u_means):.4e}  std={np.std(u_means):.4e}",
        flush=True,
    )
    print(
        f"|Δφ|_pair:  mean={np.mean(dphi_means):.4e}  std={np.std(dphi_means):.4e}",
        flush=True,
    )
    print(
        f"|Δmass|/step on I: mean={np.mean(dmass_means):.4e}  "
        f"std={np.std(dmass_means):.4e}  "
        f"CV={np.std(dmass_means)/max(np.mean(dmass_means),1e-30):.3f}",
        flush=True,
    )
    print(
        f"θ|u_n| proxy: mean={np.mean(proxy_means):.4e}  std={np.std(proxy_means):.4e}",
        flush=True,
    )

    # Trend: first third vs last third of |Δmass|
    a = np.mean(dmass_means[: probe_steps // 3])
    b = np.mean(dmass_means[-probe_steps // 3 :])
    print(f"|Δmass| first-third mean={a:.4e}  last-third mean={b:.4e}", flush=True)

    dphi = float(np.mean(dphi_means))
    dm = float(np.mean(dmass_means))
    cv = float(np.std(dmass_means) / max(dm, 1e-30))
    if dphi > 0.02 and dm < 5e-3:
        print(
            "\nCONCLUSION: residual neighbor |Δφ| remains, while per-step |Δmass| "
            "on interface is small → exchange largely stalled (theory-compatible).",
            flush=True,
        )
    elif cv > 0.4 and dm > 5e-3:
        print(
            "\nCONCLUSION: |Δmass| still sizable with high CV → ongoing "
            "exchange / oscillation, not frozen.",
            flush=True,
        )
    else:
        print(
            "\nCONCLUSION: intermediate — weak residual exchange; see numbers.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
