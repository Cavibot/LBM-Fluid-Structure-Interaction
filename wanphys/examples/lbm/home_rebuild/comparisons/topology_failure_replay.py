# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay the projected geometric Dam-break until its first topology rollback."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    ProjectedGeometricFslStepper,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
    _snapshot,
)


@dataclass(frozen=True)
class TopologyFailureReplayConfig:
    resolution_x: int = 96
    resolution_y: int = 56
    resolution_z: int = 1
    column_end_x: int = 24
    column_end_y: int = 34
    physical_viscosity: float = 2.0e-4
    physical_gravity_y: float = -5.0e-6
    gas_density: float = 1.0
    maximum_steps: int = 200
    endpoint_tolerance: float = 2.0e-6
    device: str = "cuda:0"


def find_direct_links(
    flags: np.ndarray,
    solid: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Return unique face/edge/corner LIQUID-GAS pairs."""

    liquid = int(FslCellFlag.LIQUID)
    gas = int(FslCellFlag.GAS)
    shape = flags.shape
    pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for cell in np.argwhere((flags == liquid) & ~solid):
        left = tuple(int(value) for value in cell)
        for offset in np.ndindex(3, 3, 3):
            delta = tuple(value - 1 for value in offset)
            if delta == (0, 0, 0):
                continue
            neighbor = list(left)
            valid = True
            for axis in range(3):
                neighbor[axis] += delta[axis]
                if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                    if not periodic[axis]:
                        valid = False
                        break
                    neighbor[axis] %= shape[axis]
            if not valid:
                continue
            right = tuple(neighbor)
            if not solid[right] and flags[right] == gas:
                pairs.add((left, right))
    return sorted(pairs)


def _endpoint_counts(
    fill: np.ndarray, solid: np.ndarray, tolerance: float
) -> dict[str, int]:
    active = ~solid
    return {
        "exact_empty": int(np.count_nonzero(active & (fill == 0.0))),
        "empty_sliver": int(
            np.count_nonzero(active & (fill > 0.0) & (fill <= tolerance))
        ),
        "interior_interface": int(
            np.count_nonzero(
                active & (fill > tolerance) & (fill < 1.0 - tolerance)
            )
        ),
        "full_sliver": int(
            np.count_nonzero(
                active & (fill >= 1.0 - tolerance) & (fill < 1.0)
            )
        ),
        "exact_full": int(np.count_nonzero(active & (fill == 1.0))),
    }


def run_replay(
    config: TopologyFailureReplayConfig, output_dir: Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (
        config.resolution_x,
        config.resolution_y,
        config.resolution_z,
    )
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=config.device,
        kinematic_viscosity=config.physical_viscosity,
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
    )
    walls = FslWallMask.periodic_depth_channel(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    scene = GravityColumnConfig(
        resolution_x=config.resolution_x,
        resolution_y=config.resolution_y,
        resolution_z=config.resolution_z,
        periodic_depth=True,
        column_end_x=config.column_end_x,
        column_end_y=config.column_end_y,
        physical_viscosity=config.physical_viscosity,
        physical_gravity_y=config.physical_gravity_y,
        gas_density=config.gas_density,
        sample_steps=(0, config.maximum_steps),
        device=config.device,
    )
    walls.initialize_hydrostatic(
        fluid_a,
        fsl_a,
        _column_fill(scene, walls),
        gas_density=config.gas_density,
        gravity_axis=1,
        surface_coordinate=float(config.column_end_y + 1),
    )
    stepper = ProjectedGeometricFslStepper(
        model,
        walls,
        gas_density=config.gas_density,
        interface_roundoff_tolerance=config.endpoint_tolerance,
    )

    failure: dict[str, object] | None = None
    source_before_failure: dict[str, np.ndarray] | None = None
    for step in range(1, config.maximum_steps + 1):
        source = _snapshot(fluid_a, fsl_a)
        try:
            stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
        except Exception as error:
            source_before_failure = source
            failure = {
                "step": step,
                "type": type(error).__name__,
                "message": str(error),
            }
            break
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    if failure is None or source_before_failure is None:
        raise RuntimeError("topology replay did not reproduce a failure")
    if stepper.last_transported_fill is None or stepper.last_transported_mass is None:
        raise RuntimeError("failed topology step did not retain transported candidates")

    transported_fill = stepper.last_transported_fill.numpy().astype(np.float64)
    transported_mass = stepper.last_transported_mass.numpy().astype(np.float64)
    target_flags = stepper.topology.target_flags.numpy().copy()
    direct_links = find_direct_links(
        target_flags,
        walls.host,
        periodic=tuple(not closed for closed in walls.closed_axes),
    )
    link_records = [
        {
            "liquid": liquid,
            "gas": gas,
            "liquid_fill": float(transported_fill[liquid]),
            "gas_fill": float(transported_fill[gas]),
            "liquid_mass": float(transported_mass[liquid]),
            "gas_mass": float(transported_mass[gas]),
            "source_liquid_flag": int(source_before_failure["flags"][liquid]),
            "source_gas_flag": int(source_before_failure["flags"][gas]),
        }
        for liquid, gas in direct_links
    ]
    np.savez_compressed(
        output_dir / "failure-state.npz",
        source_fill=source_before_failure["fill"],
        source_mass=source_before_failure["mass"],
        source_flags=source_before_failure["flags"],
        transported_fill=transported_fill,
        transported_mass=transported_mass,
        target_flags=target_flags,
        solid=walls.host,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    center = shape[2] // 2
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    axes[0].imshow(
        source_before_failure["fill"][:, :, center].T,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
    )
    axes[0].set_title(f"source fill, step {int(failure['step']) - 1}")
    axes[1].imshow(
        transported_fill[:, :, center].T,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
    )
    axes[1].set_title("transported fill")
    endpoint_distance = np.minimum(transported_fill, 1.0 - transported_fill)
    axes[2].imshow(
        np.log10(np.maximum(endpoint_distance[:, :, center].T, 1.0e-12)),
        origin="lower",
        vmin=-12.0,
        vmax=-1.0,
        cmap="magma",
        interpolation="nearest",
    )
    axes[2].set_title("log10 distance to endpoint")
    axes[3].imshow(
        target_flags[:, :, center].T,
        origin="lower",
        vmin=0,
        vmax=2,
        cmap=ListedColormap(("#f4f4f4", "#50a7d9", "#123c78")),
        interpolation="nearest",
    )
    for liquid, gas in direct_links:
        if liquid[2] == center:
            axes[3].plot(liquid[0], liquid[1], "rx", markersize=8)
        if gas[2] == center:
            axes[3].plot(gas[0], gas[1], "r+", markersize=8)
    axes[3].set_title("target flags and direct links")
    for axis in axes:
        axis.set_xlim(0, shape[0] - 1)
        axis.set_ylim(0, shape[1] - 1)
    overview = "topology-failure-overview.png"
    figure.savefig(output_dir / overview, dpi=180)
    plt.close(figure)

    manifest: dict[str, object] = {
        "config": asdict(config),
        "failure": failure,
        "endpoint_counts": _endpoint_counts(
            transported_fill, walls.host, config.endpoint_tolerance
        ),
        "source_flag_counts": {
            str(flag): int(np.count_nonzero(source_before_failure["flags"] == flag))
            for flag in range(3)
        },
        "target_flag_counts": {
            str(flag): int(np.count_nonzero(target_flags == flag))
            for flag in range(3)
        },
        "direct_link_count": len(direct_links),
        "direct_links": link_records,
        "transported_volume": float(np.sum(transported_fill, dtype=np.float64)),
        "transported_mass": float(np.sum(transported_mass, dtype=np.float64)),
        "overview": overview,
        "state": "failure-state.npz",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TopologyFailureReplayConfig(device=args.device)
    print(json.dumps(run_replay(config, args.output), indent=2))


if __name__ == "__main__":
    main()
