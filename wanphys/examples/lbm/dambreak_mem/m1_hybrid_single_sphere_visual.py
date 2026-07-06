"""M1 stable hybrid Shan-Chen dam-break visual with one MEM-coupled sphere."""

from __future__ import annotations

import argparse
from typing import Any

import newton.examples
import newton.viewer
from wanphys._src.fluid.fluid_viewer import ScreenSpaceFluidRenderer
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm._lbm_falling_sphere_scene import SPHERE_DENSITY_PRESETS
from wanphys.examples.lbm.dambreak_mem._metrics import DambreakMemMetrics, validate_m1_metrics
from wanphys.examples.lbm.dambreak_mem._render import format_m1_metrics, render_m1_frame, setup_m1_renderer
from wanphys.examples.lbm.dambreak_mem._scene import (
    DambreakMemM1Config,
    DambreakMemM1Scene,
    build_m1_scene,
    collect_m1_metrics,
    run_m1_headless,
    step_m1_scene,
)

COUPLING_MODE_ANALYTIC: str = "analytic"
COUPLING_MODE_HYBRID: str = "hybrid"
COUPLING_MODE_MEM: str = "mem"


def create_parser() -> argparse.ArgumentParser:
    """Create the M1 dam-break MEM visual CLI parser."""
    default_config: DambreakMemM1Config = DambreakMemM1Config()
    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument(
        "--coupling-mode",
        choices=(COUPLING_MODE_ANALYTIC, COUPLING_MODE_HYBRID, COUPLING_MODE_MEM),
        default=COUPLING_MODE_HYBRID,
    )
    parser.add_argument("--grid-res", nargs=3, type=int, default=default_config.grid_res, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--cell-size", type=float, default=default_config.cell_size)
    parser.add_argument("--tau", type=float, default=default_config.tau)
    parser.add_argument("--dam-x-fraction", type=float, default=default_config.dam_x_fraction)
    parser.add_argument("--dam-z-fraction", type=float, default=default_config.dam_z_fraction)
    parser.add_argument("--pool-z-fraction", type=float, default=default_config.pool_z_fraction)
    parser.add_argument("--sphere-radius", type=float, default=default_config.sphere_radius)
    parser.add_argument(
        "--sphere-start-fraction",
        nargs=3,
        type=float,
        default=default_config.sphere_start_fraction,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--force-scale", type=float, default=default_config.force_scale)
    parser.add_argument("--tracer-count", type=int, default=default_config.tracer_count)
    parser.add_argument("--analytic-force-scale", type=float, default=default_config.analytic_force_scale)
    parser.add_argument("--vertical-drag-scale", type=float, default=default_config.vertical_drag_scale)
    parser.add_argument("--flow-drag-scale", type=float, default=default_config.flow_drag_scale)
    parser.add_argument(
        "--density-preset",
        choices=tuple(sorted(SPHERE_DENSITY_PRESETS)),
        default=default_config.density_preset,
    )
    parser.add_argument("--sphere-density", type=float, default=default_config.sphere_density)
    parser.add_argument("--print-every", type=int, default=30)
    return parser


class LbmDambreakMemM1VisualExample:
    """Viewer example for the staged M1 dam-break MEM rigid feedback scene."""

    def __init__(
        self,
        viewer: Any,
        *,
        config: DambreakMemM1Config,
        print_every: int = 30,
    ) -> None:
        self.viewer: Any = viewer
        self.config: DambreakMemM1Config = config
        self.scene: DambreakMemM1Scene = build_m1_scene(config)
        self.sim_dt: float = 1.0 / 120.0
        self.print_every: int = int(print_every)
        self._last_metrics: DambreakMemMetrics = collect_m1_metrics(self.scene)
        self.ssfr: ScreenSpaceFluidRenderer | None = setup_m1_renderer(viewer, self.scene)

    @property
    def metrics(self) -> DambreakMemMetrics:
        """Most recent CPU-side diagnostics."""
        return self._last_metrics

    def step(self) -> None:
        """Advance the M1 scene and print compact diagnostics when requested."""
        metrics: DambreakMemMetrics = step_m1_scene(self.scene, dt=self.sim_dt)
        self._last_metrics: DambreakMemMetrics = metrics
        if self.print_every > 0 and int(metrics.frame) % self.print_every == 0:
            print(format_m1_metrics(metrics))

    def render(self) -> None:
        """Render one frame of the M1 scene."""
        render_m1_frame(self.viewer, self.scene, self.ssfr, float(self.scene.sim_time))

    def test_final(self) -> None:
        """Allow `--test` runs to validate finite M1 diagnostics."""
        metrics: DambreakMemMetrics = self.metrics
        if int(metrics.frame) <= 0:
            raise ValueError("M1 visual did not advance any frames")
        validate_m1_metrics(metrics, require_contact=False, require_mem=False)


def run_smoke(
    *,
    num_frames: int = 10,
    device: str | None = None,
    grid_res: tuple[int, int, int] = (32, 24, 28),
    cell_size: float = 0.04,
    tau: float = 1.0,
    dam_x_fraction: float = 0.32,
    dam_z_fraction: float = 0.85,
    pool_z_fraction: float = 0.52,
    sphere_radius: float = 0.14,
    sphere_start_fraction: tuple[float, float, float] = (0.58, 0.5, 0.76),
    force_scale: float = 15.0,
    coupling_mode: str = COUPLING_MODE_HYBRID,
    tracer_count: int = 0,
    analytic_force_scale: float = 100.0,
    vertical_drag_scale: float = 0.2,
    flow_drag_scale: float = 8.0,
    density_preset: str = "light",
    sphere_density: float | None = None,
    dt: float = 1.0 / 240.0,
    print_every: int = 0,
    require_contact: bool = False,
    require_mem: bool = False,
) -> DambreakMemMetrics:
    """Run a headless M1 smoke test and return final diagnostics."""
    config: DambreakMemM1Config = DambreakMemM1Config(
        device=device,
        grid_res=grid_res,
        cell_size=cell_size,
        tau=tau,
        dam_x_fraction=dam_x_fraction,
        dam_z_fraction=dam_z_fraction,
        pool_z_fraction=pool_z_fraction,
        sphere_radius=sphere_radius,
        sphere_start_fraction=sphere_start_fraction,
        force_scale=force_scale,
        coupling_mode=coupling_mode,
        tracer_count=tracer_count,
        analytic_force_scale=analytic_force_scale,
        vertical_drag_scale=vertical_drag_scale,
        flow_drag_scale=flow_drag_scale,
        density_preset=density_preset,
        sphere_density=sphere_density,
    )
    metrics: DambreakMemMetrics = run_m1_headless(
        config,
        num_frames=int(num_frames),
        dt=float(dt),
        print_every=int(print_every),
        require_contact=bool(require_contact),
        require_mem=bool(require_mem),
    )
    return metrics


def _unpause_viewer_for_test(viewer: Any, args: argparse.Namespace) -> None:
    """Ensure Newton test-mode runs can advance before final validation."""
    if not bool(getattr(args, "test", False)):
        return
    if hasattr(viewer, "_paused"):
        viewer._paused: bool = False


def _should_use_test_smoke(args: argparse.Namespace, viewer: Any) -> bool:
    """Return true when test mode needs a bounded headless smoke path."""
    if not bool(getattr(args, "test", False)):
        return False
    if not bool(getattr(args, "headless", False)):
        return False
    return not isinstance(viewer, newton.viewer.ViewerNull)


def run_with_parser(parser: argparse.ArgumentParser) -> None:
    """Run the visual example using an already configured parser."""
    viewer: Any
    args: argparse.Namespace
    viewer, args = init_fluid_viewer(parser)
    default_config: DambreakMemM1Config = DambreakMemM1Config()
    sphere_start_fraction: tuple[float, float, float] = tuple(
        float(value) for value in getattr(args, "sphere_start_fraction", default_config.sphere_start_fraction)
    )
    density_preset: str = str(getattr(args, "density_preset", default_config.density_preset))
    sphere_density_arg: float | None = getattr(args, "sphere_density", default_config.sphere_density)
    sphere_density: float | None = None if sphere_density_arg is None else float(sphere_density_arg)

    if _should_use_test_smoke(args, viewer):
        run_smoke(
            num_frames=int(args.num_frames),
            device=args.device,
            grid_res=(int(args.grid_res[0]), int(args.grid_res[1]), int(args.grid_res[2])),
            cell_size=float(args.cell_size),
            tau=float(args.tau),
            dam_x_fraction=float(getattr(args, "dam_x_fraction", default_config.dam_x_fraction)),
            dam_z_fraction=float(getattr(args, "dam_z_fraction", default_config.dam_z_fraction)),
            pool_z_fraction=float(getattr(args, "pool_z_fraction", default_config.pool_z_fraction)),
            sphere_radius=float(getattr(args, "sphere_radius", default_config.sphere_radius)),
            sphere_start_fraction=sphere_start_fraction,
            force_scale=float(args.force_scale),
            coupling_mode=str(args.coupling_mode),
            tracer_count=int(args.tracer_count),
            analytic_force_scale=float(args.analytic_force_scale),
            vertical_drag_scale=float(args.vertical_drag_scale),
            flow_drag_scale=float(args.flow_drag_scale),
            density_preset=density_preset,
            sphere_density=sphere_density,
            print_every=int(args.print_every),
            require_contact=False,
            require_mem=False,
        )
        if hasattr(viewer, "close"):
            viewer.close()
        return

    config: DambreakMemM1Config = DambreakMemM1Config(
        device=args.device,
        grid_res=(int(args.grid_res[0]), int(args.grid_res[1]), int(args.grid_res[2])),
        cell_size=float(args.cell_size),
        tau=float(args.tau),
        dam_x_fraction=float(getattr(args, "dam_x_fraction", default_config.dam_x_fraction)),
        dam_z_fraction=float(getattr(args, "dam_z_fraction", default_config.dam_z_fraction)),
        pool_z_fraction=float(getattr(args, "pool_z_fraction", default_config.pool_z_fraction)),
        sphere_radius=float(getattr(args, "sphere_radius", default_config.sphere_radius)),
        sphere_start_fraction=sphere_start_fraction,
        force_scale=float(args.force_scale),
        coupling_mode=str(args.coupling_mode),
        tracer_count=int(args.tracer_count),
        analytic_force_scale=float(args.analytic_force_scale),
        vertical_drag_scale=float(args.vertical_drag_scale),
        flow_drag_scale=float(args.flow_drag_scale),
        density_preset=density_preset,
        sphere_density=sphere_density,
    )
    example: LbmDambreakMemM1VisualExample = LbmDambreakMemM1VisualExample(
        viewer,
        config=config,
        print_every=int(args.print_every),
    )
    _unpause_viewer_for_test(viewer, args)
    newton.examples.run(example, args)


def main() -> None:
    """Run the M1 visual example."""
    parser: argparse.ArgumentParser = create_parser()
    run_with_parser(parser)


if __name__ == "__main__":
    main()

