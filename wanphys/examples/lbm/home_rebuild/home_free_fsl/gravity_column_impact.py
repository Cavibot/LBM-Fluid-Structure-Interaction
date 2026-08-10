# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the R3.7 gravity-column impact and reflected-wave audit."""

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
    args = parser.parse_args()
    config = GravityColumnConfig(
        profile="r3.7-periodic-depth-wall-impact",
        resolution_z=1,
        periodic_depth=True,
        physical_gravity_y=-5.0e-6,
        sample_steps=(0, 2000, 4000, 6000, 8000, 10000, 12500, 15000, 20000),
        device=args.device,
    )
    print(json.dumps(run_scene(config, args.output), indent=2))


if __name__ == "__main__":
    main()
