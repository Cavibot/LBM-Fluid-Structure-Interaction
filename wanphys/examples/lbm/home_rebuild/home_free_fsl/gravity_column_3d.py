# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a closed-depth, genuinely three-dimensional link-wise gravity column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    run_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("preview", "formal"), default="preview")
    args = parser.parse_args()

    if args.profile == "preview":
        config = GravityColumnConfig(
            profile="r3.8-linkwise-true3d-preview",
            resolution_x=96,
            resolution_y=56,
            resolution_z=16,
            periodic_depth=False,
            column_end_x=24,
            column_end_y=35,
            physical_gravity_y=-5.0e-6,
            sample_steps=(0, 500, 1000, 1500, 2000, 3000, 4000),
            save_volume_snapshots=True,
            device=args.device,
        )
    else:
        config = GravityColumnConfig(
            profile="r3.9-linkwise-true3d-wide-formal",
            resolution_x=160,
            resolution_y=96,
            resolution_z=64,
            periodic_depth=False,
            physical_gravity_y=-5.0e-6,
            sample_steps=(0, 2000, 4000, 6000, 8000, 10000, 12500, 15000, 20000),
            save_volume_snapshots=True,
            device=args.device,
        )
    print(json.dumps(run_scene(config, args.output), indent=2))


if __name__ == "__main__":
    main()
