# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless moment-encoded HOME-FREE VOF dam-break smoke (H4).

Uses ``home_fp32_ref.vof_step`` (10 moments/cell + φ). Distribution VOF /
quantized HOME are separate paths.

Run:
    uv run python -m wanphys.examples.lbm.fluid_grid_lbm_home_vof_dambreak
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    CELL_INTERFACE,
    CELL_LIQUID,
    HomeDomainBC,
    seed_dam_break_column,
    step_home_vof_numpy,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HOME-FREE VOF dam-break (moments)")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--g", type=float, default=-0.0003)
    args = p.parse_args()

    n = args.n
    st = seed_dam_break_column(
        (n, max(n // 2, 8), n), dam_x=n // 4, fill_z=n // 2, rho_liquid=1.0,
    )
    phi0 = float(st.phi.sum())
    bc = HomeDomainBC.all_walls()
    t0 = time.perf_counter()
    out = st
    for step in range(args.steps):
        out = step_home_vof_numpy(
            out,
            lattice="D3Q27",
            tau=args.tau,
            fz=args.g,
            domain_bc=bc,
        )
        if step % 10 == 0 or step == args.steps - 1:
            phi = float(out.phi.sum())
            n_l = int((out.cell_type == CELL_LIQUID).sum())
            n_i = int((out.cell_type == CELL_INTERFACE).sum())
            umax = float(np.max(np.abs(out.moments.uz)))
            print(
                f"step={step:3d}  φ={phi:.2f} (Δ={(phi - phi0) / phi0 * 100:+.2f}%)  "
                f"L={n_l} I={n_i}  |uz|_max={umax:.4f}"
            )
    print(f"done in {time.perf_counter() - t0:.2f}s (numpy HOME-FREE VOF)")


if __name__ == "__main__":
    main()
