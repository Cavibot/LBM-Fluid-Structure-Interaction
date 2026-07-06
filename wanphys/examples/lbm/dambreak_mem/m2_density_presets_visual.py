"""M2 density-preset Shan-Chen dam-break MEM visual with one hybrid-coupled sphere."""

from __future__ import annotations

import argparse

from wanphys.examples.lbm.dambreak_mem import m1_hybrid_single_sphere_visual as m1
from wanphys.examples.lbm.dambreak_mem._metrics import DambreakMemMetrics

M2_DEFAULT_FORCE_SCALE: float = 0.005


def create_parser() -> argparse.ArgumentParser:
    """Create the M2 density-preset visual CLI parser."""
    parser: argparse.ArgumentParser = m1.create_parser()
    parser.set_defaults(density_preset="light", force_scale=M2_DEFAULT_FORCE_SCALE)
    return parser


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
    density_preset: str = "light",
    sphere_density: float | None = None,
    force_scale: float = M2_DEFAULT_FORCE_SCALE,
    coupling_mode: str = m1.COUPLING_MODE_HYBRID,
    tracer_count: int = 0,
    analytic_force_scale: float = 100.0,
    vertical_drag_scale: float = 0.2,
    flow_drag_scale: float = 8.0,
    dt: float = 1.0 / 120.0,
    print_every: int = 0,
    require_contact: bool = False,
    require_mem: bool = False,
) -> DambreakMemMetrics:
    """Run one M2 density preset headlessly."""
    metrics: DambreakMemMetrics = m1.run_smoke(
        num_frames=int(num_frames),
        device=device,
        grid_res=grid_res,
        cell_size=float(cell_size),
        tau=float(tau),
        dam_x_fraction=float(dam_x_fraction),
        dam_z_fraction=float(dam_z_fraction),
        pool_z_fraction=float(pool_z_fraction),
        sphere_radius=float(sphere_radius),
        sphere_start_fraction=sphere_start_fraction,
        density_preset=str(density_preset),
        sphere_density=sphere_density,
        force_scale=float(force_scale),
        coupling_mode=str(coupling_mode),
        tracer_count=int(tracer_count),
        analytic_force_scale=float(analytic_force_scale),
        vertical_drag_scale=float(vertical_drag_scale),
        flow_drag_scale=float(flow_drag_scale),
        dt=float(dt),
        print_every=int(print_every),
        require_contact=bool(require_contact),
        require_mem=bool(require_mem),
    )
    return metrics


def run_density_comparison(
    *,
    num_frames: int = 180,
    device: str | None = None,
    grid_res: tuple[int, int, int] = (24, 18, 20),
    cell_size: float = 0.04,
    tau: float = 1.0,
    force_scale: float = M2_DEFAULT_FORCE_SCALE,
    tracer_count: int = 0,
    dt: float = 1.0 / 120.0,
    print_every: int = 0,
) -> dict[str, DambreakMemMetrics]:
    """Run light and heavy presets with identical M2 scene parameters."""
    light_metrics: DambreakMemMetrics = run_smoke(
        num_frames=int(num_frames),
        device=device,
        grid_res=grid_res,
        cell_size=float(cell_size),
        tau=float(tau),
        density_preset="light",
        force_scale=float(force_scale),
        tracer_count=int(tracer_count),
        dt=float(dt),
        print_every=int(print_every),
        require_contact=True,
        require_mem=True,
    )
    heavy_metrics: DambreakMemMetrics = run_smoke(
        num_frames=int(num_frames),
        device=device,
        grid_res=grid_res,
        cell_size=float(cell_size),
        tau=float(tau),
        density_preset="heavy",
        force_scale=float(force_scale),
        tracer_count=int(tracer_count),
        dt=float(dt),
        print_every=int(print_every),
        require_contact=True,
        require_mem=True,
    )
    comparison: dict[str, DambreakMemMetrics] = {
        "light": light_metrics,
        "heavy": heavy_metrics,
    }
    return comparison


def main() -> None:
    """Run the M2 visual example."""
    parser: argparse.ArgumentParser = create_parser()
    m1.run_with_parser(parser)


if __name__ == "__main__":
    main()

