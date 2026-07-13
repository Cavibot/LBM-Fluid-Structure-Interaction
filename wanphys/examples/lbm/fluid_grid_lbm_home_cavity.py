# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless HOME-LBM fp32 lid-driven cavity smoke (H2).

Uses moment-encoded reconstruct-stream + collide with HOME Eq. 24 moving
top wall. Not wired into ``LbmSolver`` / viewer yet.

Run:
    uv run python -m wanphys.examples.lbm.fluid_grid_lbm_home_cavity
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    HomeDomainBC,
    make_uniform_equilibrium,
    step_domain_numpy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="HOME fp32 lid-driven cavity")
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--ulid", type=float, default=0.08)
    parser.add_argument("--lattice", type=str, default="D3Q27")
    args = parser.parse_args()

    field = make_uniform_equilibrium((args.nx, args.ny, args.nz), rho0=1.0)
    bc = HomeDomainBC.all_walls(lid_face="ymax", lid_ux=args.ulid)
    t0 = time.perf_counter()
    out = field
    for step in range(args.steps):
        out = step_domain_numpy(
            out, lattice=args.lattice, tau=args.tau, domain_bc=bc,
        )
        if step % 20 == 0 or step == args.steps - 1:
            ke = float(np.mean(out.ux**2 + out.uy**2 + out.uz**2))
            umax = float(np.max(np.abs(out.ux)))
            print(
                f"step={step:4d}  ke={ke:.6e}  |ux|_max={umax:.5f}  "
                f"rho∈[{out.rho.min():.4f},{out.rho.max():.4f}]"
            )
    dt = time.perf_counter() - t0
    print(f"done in {dt:.2f}s ({args.steps / dt:.1f} steps/s, numpy)")


if __name__ == "__main__":
    main()
