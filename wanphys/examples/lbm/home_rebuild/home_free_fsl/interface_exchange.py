# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualize the R3.1 static and translating planar-interface mass oracle."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslMassAdvector,
    FslState,
    initialize_fsl_fields,
    link_mass_exchange,
)


@dataclass(frozen=True)
class InterfaceExchangeConfig:
    resolution_x: int = 96
    resolution_y: int = 32
    slab_start: int = 24
    slab_end: int = 72
    interface_fill: float = 0.5
    translation_speed: float = 0.05

    def validate(self) -> None:
        if self.resolution_x < 16 or self.resolution_y < 4:
            raise ValueError("interface audit requires at least a 16x4 grid")
        if not 2 <= self.slab_start < self.slab_end <= self.resolution_x - 2:
            raise ValueError("slab must leave at least two gas cells on each periodic side")
        if not 0.0 < self.interface_fill < 1.0:
            raise ValueError("interface_fill must be strictly fractional")
        if not 0.0 < self.translation_speed < 0.2:
            raise ValueError("translation_speed must be in (0, 0.2)")


def equilibrium_moments(shape: tuple[int, int, int], velocity_x: float) -> np.ndarray:
    moments = np.zeros(shape + (10,), dtype=np.float64)
    moments[..., 0] = 1.0
    moments[..., 1] = velocity_x
    moments[..., 4] = velocity_x * velocity_x
    return moments


def initial_fill(config: InterfaceExchangeConfig) -> np.ndarray:
    config.validate()
    shape = (config.resolution_x, config.resolution_y, 1)
    fill = np.zeros(shape, dtype=np.float64)
    fill[config.slab_start : config.slab_end, :, :] = 1.0
    fill[config.slab_start - 1, :, :] = config.interface_fill
    fill[config.slab_end, :, :] = config.interface_fill
    return fill


def render_field(
    *,
    output_path: Path,
    title: str,
    field: np.ndarray,
    profile_initial: np.ndarray,
    profile_final: np.ndarray,
    color_limits: tuple[float, float],
    color_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (field_axis, profile_axis) = plt.subplots(
        2, 1, figsize=(9.0, 6.8), dpi=150, gridspec_kw={"height_ratios": (2.2, 1.0)}
    )
    image = field_axis.imshow(
        field[:, :, 0].T,
        origin="lower",
        cmap="RdBu_r" if color_limits[0] < 0.0 else "Blues",
        vmin=color_limits[0],
        vmax=color_limits[1],
        interpolation="nearest",
        aspect="auto",
    )
    field_axis.set_title(title)
    field_axis.set_xlabel("x lattice cell")
    field_axis.set_ylabel("y lattice cell")
    fig.colorbar(image, ax=field_axis, label=color_label)
    x = np.arange(profile_initial.size)
    profile_axis.plot(x, profile_initial, color="#555555", linewidth=1.8, label="before")
    profile_axis.plot(x, profile_final, color="#087f8c", linewidth=1.8, label="after")
    profile_axis.set_xlim(0, profile_initial.size - 1)
    profile_axis.set_ylim(-0.08, 1.08)
    profile_axis.set_xlabel("x lattice cell")
    profile_axis.set_ylabel("liquid mass")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_interface_exchange(
    config: InterfaceExchangeConfig,
    output_dir: Path,
    *,
    backend: str = "numpy",
    device: str = "cpu",
) -> dict[str, float | int | str]:
    config.validate()
    if backend not in ("numpy", "warp"):
        raise ValueError("backend must be 'numpy' or 'warp'")
    output_dir.mkdir(parents=True, exist_ok=True)
    fill = initial_fill(config)
    density = np.ones(fill.shape, dtype=np.float64)
    mass, _, flags, diagnostics = initialize_fsl_fields(density, fill)
    if backend == "numpy":
        static_mass = link_mass_exchange(
            equilibrium_moments(fill.shape, 0.0), mass, fill, flags
        )
        moving_mass = link_mass_exchange(
            equilibrium_moments(fill.shape, config.translation_speed), mass, fill, flags
        )
    else:
        model = HomeCoreModel(fluid_grid_res=fill.shape, device=device)
        domain = HomeCoreDomain(model)
        fluid = domain.create_state()
        fsl = FslState(model)
        advector = FslMassAdvector(model)
        domain.solver.initialize_uniform_lattice(fluid)
        fsl.initialize_from_fill_level(fluid, fill)
        advector.advect(fluid, fsl)
        advector.validate_result()
        static_mass = advector.advected_mass.numpy().astype(np.float64)
        domain.solver.initialize_uniform_lattice(
            fluid, velocity=(config.translation_speed, 0.0, 0.0)
        )
        fsl.initialize_from_fill_level(fluid, fill)
        advector.advect(fluid, fsl)
        advector.validate_result()
        moving_mass = advector.advected_mass.numpy().astype(np.float64)
    static_delta = static_mass - mass
    moving_delta = moving_mass - mass
    stage = "R3.1" if backend == "numpy" else "R3.2"
    left_index = config.slab_start - 1
    right_index = config.slab_end
    metrics: dict[str, float | int | str] = {
        "scene": (
            "r3.1-planar-interface-link-mass-oracle"
            if backend == "numpy"
            else "r3.2-planar-interface-warp-mass"
        ),
        "initial_total_mass": float(np.sum(mass, dtype=np.float64)),
        "static_max_abs_cell_delta": float(np.max(np.abs(static_delta))),
        "static_total_mass_delta": float(np.sum(static_mass) - np.sum(mass)),
        "moving_total_mass_delta": float(np.sum(moving_mass) - np.sum(mass)),
        "moving_left_interface_mean_delta": float(np.mean(moving_delta[left_index])),
        "moving_right_interface_mean_delta": float(np.mean(moving_delta[right_index])),
        "moving_max_abs_cell_delta": float(np.max(np.abs(moving_delta))),
        "direct_liquid_gas_link_count": diagnostics.direct_liquid_gas_link_count,
    }

    profile_initial = np.mean(mass[:, :, 0], axis=1)
    render_field(
        output_path=output_dir / "static-interface.png",
        title=f"{stage} static planar interface: link mass delta",
        field=static_delta,
        profile_initial=profile_initial,
        profile_final=np.mean(static_mass[:, :, 0], axis=1),
        color_limits=(-0.05, 0.05),
        color_label="mass change",
    )
    limit = 1.1 * config.translation_speed
    render_field(
        output_path=output_dir / "translating-interface.png",
        title=f"{stage} translating interface: u_x={config.translation_speed:.3f}",
        field=moving_delta,
        profile_initial=profile_initial,
        profile_final=np.mean(moving_mass[:, :, 0], axis=1),
        color_limits=(-limit, limit),
        color_label="mass change",
    )
    manifest = {
        "profile": "r3.1-cpu-oracle" if backend == "numpy" else "r3.2-warp-mass",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "backend": backend,
            "device": device if backend == "warp" else "host",
        },
        "metrics": metrics,
        "frames": ["static-interface.png", "translating-interface.png"],
        "limitations": [
            "single link-mass-exchange step",
            "fixed topology",
            "no gas-pressure boundary or HOME collision transaction",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-x", type=int, default=96)
    parser.add_argument("--resolution-y", type=int, default=32)
    parser.add_argument("--slab-start", type=int, default=24)
    parser.add_argument("--slab-end", type=int, default=72)
    parser.add_argument("--interface-fill", type=float, default=0.5)
    parser.add_argument("--translation-speed", type=float, default=0.05)
    parser.add_argument("--backend", choices=("numpy", "warp"), default="numpy")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InterfaceExchangeConfig(
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        slab_start=args.slab_start,
        slab_end=args.slab_end,
        interface_fill=args.interface_fill,
        translation_speed=args.translation_speed,
    )
    metrics = run_interface_exchange(
        config, args.output, backend=args.backend, device=args.device
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
