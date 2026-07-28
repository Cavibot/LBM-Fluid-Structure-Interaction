# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""L0: HOME-FREE dam-break vs Martin & Moyce (1952) surge front.

No rigid bodies, no height-eq / empirical FSI. Prints a table and writes CSV.

    uv run --extra examples python -m wanphys.examples.lbm.run_martin_moyce_compare --n 48
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.benchmark.home_vof_dambreak_front import (
    run_home_vof_dambreak_front,
)
from wanphys._src.fluid.fluid_grid.lbm.benchmark.martin_moyce import (
    interpolate_Z_at_T,
    reference_arrays,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Martin-Moyce L0 dam-break compare")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--tau", type=float, default=0.51)
    p.add_argument("--fz", type=float, default=None, help="lattice g_z (default: scale with n)")
    p.add_argument("--t-target", type=float, default=2.9)
    p.add_argument("--no-align-gate", action="store_true")
    p.add_argument(
        "--csv",
        type=str,
        default="martin_moyce_compare.csv",
        help="output CSV path",
    )
    args = p.parse_args()

    result = run_home_vof_dambreak_front(
        n=int(args.n),
        fz=args.fz,
        tau=float(args.tau),
        t_target=float(args.t_target),
        align_gate=not bool(args.no_align_gate),
    )
    t_ref, z_ref = reference_arrays(2.0)
    t_plot = result.t_aligned
    z_at = interpolate_Z_at_T(t_plot, result.z, t_ref)

    d = result.scale.diagnostics()
    print(
        f"L0 HOME-FREE Martin-Moyce  n={args.n}  n^2={d['n2']:.2f}  "
        f"tau={d['tau']:.3f}  nu={d['nu']:.4g}  Re~{d['Re_column']:.0f}"
    )
    print(
        f"mass {result.water_mass0:.4g} -> {result.water_mass1:.4g}  "
        f"rel={result.stats['mass_rel']:.3%}"
    )
    print(f"{'T':>6} {'Z_exp':>7} {'Z_sim':>7} {'dZ':>7}")
    rows: list[dict[str, float]] = []
    for ti, ze, zs in zip(t_ref, z_ref, z_at, strict=True):
        if not np.isfinite(zs):
            continue
        if ti < 0.35 or ti > 3.0:
            continue
        dz = float(zs - ze)
        print(f"{ti:6.2f} {ze:7.2f} {zs:7.2f} {dz:7.2f}")
        rows.append({"T": float(ti), "Z_exp": float(ze), "Z_sim": float(zs), "dZ": dz})

    st = result.stats
    print(
        f"mid-range: mae={st['mae']:.3f} rmse={st['rmse']:.3f} "
        f"mean_rel={st['mean_rel']:.3%}  align_gate={bool(st.get('aligned', 0))}"
    )

    out = Path(args.csv)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["T", "Z_exp", "Z_sim", "dZ"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
