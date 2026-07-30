# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Authoritative VOF dam-break visual and bounded headless acceptance runner.

CPU interactive:

    uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak

CPU headless:

    uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak \
        --viewer null --num-frames 30 --csv vof-dambreak.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.examples
from wanphys._src.fluid.fluid_grid.lbm import (
    DebugVofView,
    LbmDomain,
    LbmModel,
    VofDiagnostics,
    VofInterfaceVisualizer,
    collect_vof_diagnostics,
    validate_vof_diagnostics,
)
from wanphys._src.fluid.fluid_viewer import ScreenSpaceFluidRenderer
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

D3Q19_MOVING_DIRECTIONS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (-1, 0, 1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (0, -1, 1),
    (0, -1, -1),
)


@dataclass(frozen=True)
class AuthoritativeVofDamBreakConfig:
    """Configuration shared by the interactive and headless P8 paths."""

    device: str = "cpu"
    grid_res: tuple[int, int, int] = (48, 12, 32)
    cell_size: float = 0.02
    encoding: str = "fullf"
    collision: str = "srt"
    tau: float = 0.8
    gravity_z: float = -2.0e-5
    surface_tension: float = 0.0
    dam_x_fraction: float = 1.0 / 3.0
    dam_z_fraction: float = 2.0 / 3.0
    steps_per_frame: int = 1
    runtime_profile: str = "strict"
    validation_interval: int = 60
    show_normals: bool = False


def build_authoritative_dam_break_phi(
    shape: tuple[int, int, int],
    *,
    dam_x_fraction: float = 1.0 / 3.0,
    dam_z_fraction: float = 2.0 / 3.0,
) -> np.ndarray:
    """Create a liquid column surrounded by one legal D3Q19 interface layer."""

    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value < 3 for value in shape):
        raise ValueError("P8 dam-break grid_res must contain three values >= 3")
    if not 0.0 < float(dam_x_fraction) < 1.0:
        raise ValueError("dam_x_fraction must lie in (0, 1)")
    if not 0.0 < float(dam_z_fraction) < 1.0:
        raise ValueError("dam_z_fraction must lie in (0, 1)")
    dam_x = max(1, min(shape[0] - 2, int(shape[0] * dam_x_fraction)))
    dam_z = max(1, min(shape[2] - 2, int(shape[2] * dam_z_fraction)))

    liquid = np.zeros(shape, dtype=bool)
    liquid[:dam_x, :, :dam_z] = True
    interface = np.zeros(shape, dtype=bool)
    for i, j, k in np.argwhere(liquid):
        for di, dj, dk in D3Q19_MOVING_DIRECTIONS:
            ni, nj, nk = int(i + di), int(j + dj), int(k + dk)
            if (
                0 <= ni < shape[0]
                and 0 <= nj < shape[1]
                and 0 <= nk < shape[2]
                and not liquid[ni, nj, nk]
            ):
                interface[ni, nj, nk] = True
    phi = np.zeros(shape, dtype=np.float32)
    phi[interface] = np.float32(0.5)
    phi[liquid] = np.float32(1.0)
    return phi


def _validate_config(config: AuthoritativeVofDamBreakConfig) -> None:
    device = str(config.device)
    if device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError(
            f"P8 requested {device!r}, but Warp reports CUDA unavailable"
        )
    if config.steps_per_frame <= 0:
        raise ValueError("steps_per_frame must be positive")
    if not np.isfinite(config.gravity_z):
        raise ValueError("gravity_z must be finite")
    if not np.isfinite(config.surface_tension) or config.surface_tension < 0.0:
        raise ValueError("surface_tension must be finite and non-negative")


def _build_model(config: AuthoritativeVofDamBreakConfig) -> LbmModel:
    _validate_config(config)
    return LbmModel(
        fluid_grid_res=config.grid_res,
        fluid_grid_cell_size=float(config.cell_size),
        device=config.device,
        encoding=config.encoding,
        collision=config.collision,
        interface_model="vof",
        tau=float(config.tau),
        gravity_z=float(config.gravity_z),
        vof_surface_tension=float(config.surface_tension),
        vof_runtime_profile=str(config.runtime_profile),
        vof_validation_interval=int(config.validation_interval),
        enforce_population_positivity=False,
    )


class AuthoritativeVofDamBreakScene:
    """Simulation-only P8 scene used by both visual and headless paths."""

    def __init__(self, config: AuthoritativeVofDamBreakConfig) -> None:
        self.config = config
        self.model = _build_model(config)
        self.domain = LbmDomain(self.model)
        phi = build_authoritative_dam_break_phi(
            config.grid_res,
            dam_x_fraction=config.dam_x_fraction,
            dam_z_fraction=config.dam_z_fraction,
        )
        state = self.domain.initialize_vof(phi)
        assert state.vof is not None
        self.initial_mass = float(state.vof.reference_mass)
        self.step_count = 0
        self.ledger: list[VofDiagnostics] = []

    def step(self, count: int = 1) -> VofDiagnostics | None:
        """Advance lattice steps and return the latest profile report."""

        if count <= 0:
            raise ValueError("P8 step count must be positive")
        diagnostics: VofDiagnostics | None = None
        for _ in range(int(count)):
            self.domain.step(1.0)
            self.step_count += 1
            diagnostics = self.domain.solver.last_vof_diagnostics
            if diagnostics is not None:
                self.ledger.append(diagnostics)
        return diagnostics


def format_diagnostics(step: int, diagnostics: VofDiagnostics) -> str:
    """Format one compact, stable P8 progress line."""

    return (
        f"step={step} mass={diagnostics.total_mass:.8g} "
        f"relerr={diagnostics.relative_mass_error:.3e} "
        f"phi=[{diagnostics.phi_min:.4f},{diagnostics.phi_max:.4f}] "
        f"interface={diagnostics.interface_cell_count} "
        f"pending={diagnostics.pending_excess_total:.3e} "
        f"vmax={diagnostics.max_velocity:.3e} "
        f"epoch={diagnostics.epoch}"
    )


def write_diagnostics_csv(
    path: str | Path,
    ledger: tuple[VofDiagnostics, ...] | list[VofDiagnostics],
) -> Path:
    """Write the bounded per-step ledger with deterministic columns."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("step", *VofDiagnostics.__dataclass_fields__.keys())
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for diagnostics in ledger:
            writer.writerow({"step": diagnostics.epoch, **asdict(diagnostics)})
    return output


def run_headless(
    config: AuthoritativeVofDamBreakConfig,
    *,
    num_steps: int,
    print_every: int = 0,
    csv_path: str | Path | None = None,
) -> tuple[VofDiagnostics, ...]:
    """Run a finite authoritative VOF dam-break without creating a viewer."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    scene = AuthoritativeVofDamBreakScene(config)
    for step in range(1, int(num_steps) + 1):
        diagnostics = scene.step()
        if (
            diagnostics is not None
            and print_every > 0
            and step % print_every == 0
        ):
            print(format_diagnostics(step, diagnostics))
    if not scene.ledger or scene.ledger[-1].epoch != scene.step_count:
        diagnostics = collect_vof_diagnostics(
            scene.domain.state,
            initial_mass=scene.initial_mass,
        )
        if config.runtime_profile != "off":
            validate_vof_diagnostics(
                diagnostics,
                max_lattice_speed=float(scene.model.max_lattice_speed),
            )
        scene.ledger.append(diagnostics)
    ledger = tuple(scene.ledger)
    if csv_path is not None:
        write_diagnostics_csv(csv_path, ledger)
    return ledger


class AuthoritativeVofDamBreakVisualExample:
    """Newton viewer wrapper that renders authoritative phi and geometry."""

    def __init__(
        self,
        viewer: Any,
        config: AuthoritativeVofDamBreakConfig,
        *,
        print_every: int = 30,
    ) -> None:
        self.viewer = viewer
        self.config = config
        self.print_every = int(print_every)
        self.scene = AuthoritativeVofDamBreakScene(config)
        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.5 * float(config.cell_size),
            device=self.scene.model._device,
        )
        if hasattr(viewer, "register_post_render_callback"):
            viewer.register_post_render_callback(
                lambda current: self.ssfr.render(current)
            )
        self.interface_visualizer = VofInterfaceVisualizer(
            config.grid_res,
            self.scene.model._device,
            float(config.cell_size),
        )

    @property
    def diagnostics(self) -> VofDiagnostics | None:
        return self.scene.ledger[-1] if self.scene.ledger else None

    def step(self) -> None:
        diagnostics = self.scene.step(self.config.steps_per_frame)
        if (
            diagnostics is not None
            and self.print_every > 0
            and self.scene.step_count % self.print_every == 0
        ):
            print(format_diagnostics(self.scene.step_count, diagnostics))

    def render(self) -> None:
        state = self.scene.domain.state
        assert state.vof is not None
        self.viewer.begin_frame(float(self.scene.step_count))
        if self.ssfr.available:
            self.ssfr.set_density_field(
                density=state.vof.phi,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=float(self.config.cell_size),
                threshold=0.05,
                max_steps=512,
            )
        self.interface_visualizer.render(
            self.viewer,
            DebugVofView.from_authoritative_vof(state.vof),
            state.solid_phi,
            show_normals=bool(self.config.show_normals),
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        if self.scene.step_count == 0:
            raise ValueError("P8 visual example did not advance")
        diagnostics = collect_vof_diagnostics(
            self.scene.domain.state,
            initial_mass=self.scene.initial_mass,
        )
        validate_vof_diagnostics(diagnostics)


def create_parser() -> argparse.ArgumentParser:
    """Create the P8 visual/headless CLI parser."""

    defaults = AuthoritativeVofDamBreakConfig()
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--encoding",
        choices=("fullf", "home"),
        default=defaults.encoding,
    )
    parser.add_argument(
        "--collision",
        choices=("srt", "nocm_mrt"),
        default=defaults.collision,
    )
    parser.add_argument(
        "--grid-res",
        nargs=3,
        type=int,
        default=defaults.grid_res,
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--cell-size", type=float, default=defaults.cell_size)
    parser.add_argument("--tau", type=float, default=defaults.tau)
    parser.add_argument("--gravity-z", type=float, default=defaults.gravity_z)
    parser.add_argument(
        "--surface-tension",
        type=float,
        default=defaults.surface_tension,
    )
    parser.add_argument(
        "--dam-x-fraction",
        type=float,
        default=defaults.dam_x_fraction,
    )
    parser.add_argument(
        "--dam-z-fraction",
        type=float,
        default=defaults.dam_z_fraction,
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=None,
        help="simulation substeps per rendered frame (auto: visual=4, null=1)",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=("auto", "strict", "sampled", "device", "off"),
        default="auto",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=defaults.validation_interval,
    )
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument(
        "--show-normals",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_normals,
    )
    parser.add_argument("--csv", type=str, default=None)
    return parser


def _config_from_args(
    args: argparse.Namespace,
    *,
    interactive: bool,
) -> AuthoritativeVofDamBreakConfig:
    device = "cpu" if args.device is None else str(args.device)
    profile = str(args.runtime_profile)
    if profile == "auto":
        profile = "device" if interactive else "strict"
    steps_per_frame = args.steps_per_frame
    if steps_per_frame is None:
        steps_per_frame = 4 if interactive else 1
    return AuthoritativeVofDamBreakConfig(
        device=device,
        grid_res=tuple(int(value) for value in args.grid_res),
        cell_size=float(args.cell_size),
        encoding=str(args.encoding),
        collision=str(args.collision),
        tau=float(args.tau),
        gravity_z=float(args.gravity_z),
        surface_tension=float(args.surface_tension),
        dam_x_fraction=float(args.dam_x_fraction),
        dam_z_fraction=float(args.dam_z_fraction),
        steps_per_frame=int(steps_per_frame),
        runtime_profile=profile,
        validation_interval=int(args.validation_interval),
        show_normals=bool(args.show_normals),
    )


def run_with_parser(parser: argparse.ArgumentParser) -> None:
    """Dispatch bounded null-viewer runs or the interactive Newton loop."""

    preview, _unknown = parser.parse_known_args()
    if preview.viewer == "null":
        args = parser.parse_args()
        ledger = run_headless(
            _config_from_args(args, interactive=False),
            num_steps=int(args.num_frames),
            print_every=int(args.print_every),
            csv_path=args.csv,
        )
        print(format_diagnostics(len(ledger), ledger[-1]))
        return

    viewer, args = init_fluid_viewer(parser)
    example = AuthoritativeVofDamBreakVisualExample(
        viewer,
        _config_from_args(args, interactive=True),
        print_every=int(args.print_every),
    )
    if bool(args.test) and hasattr(viewer, "_paused"):
        viewer._paused = False
    newton.examples.run(example, args)
    if args.csv and example.scene.ledger:
        write_diagnostics_csv(args.csv, example.scene.ledger)


def main() -> None:
    run_with_parser(create_parser())


__all__ = [
    "AuthoritativeVofDamBreakConfig",
    "AuthoritativeVofDamBreakScene",
    "AuthoritativeVofDamBreakVisualExample",
    "build_authoritative_dam_break_phi",
    "create_parser",
    "format_diagnostics",
    "run_headless",
    "run_with_parser",
    "write_diagnostics_csv",
]


if __name__ == "__main__":
    main()
