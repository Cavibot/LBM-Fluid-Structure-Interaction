"""Rendering helpers for the staged M1 dam-break MEM visual example."""

from __future__ import annotations

from typing import Any

import newton.viewer
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer
from wanphys.examples.lbm.dambreak_mem._metrics import DambreakMemMetrics
from wanphys.examples.lbm.dambreak_mem._scene import DambreakMemM1Scene

FIELD_DENSITY_THRESHOLD: float = 1.0
RAY_MARCH_STEPS: int = 160


def _format_vector(values: Any, count: int = 3) -> str:
    vector: list[float] = [float(value) for value in values[:count]]
    return "[" + ", ".join(f"{value:.6g}" for value in vector) + "]"


def format_m1_metrics(metrics: DambreakMemMetrics) -> str:
    """Return a compact log line for the M1 visual diagnostics."""
    text: str = (
        f"m1 t={metrics.sim_time:7.4f} frame={metrics.frame:5d} "
        f"mode={metrics.coupling_mode} "
        f"pos={_format_vector(metrics.body_position)} "
        f"vel={_format_vector(metrics.body_velocity)} "
        f"fluid_g={metrics.fluid_gravity_z:.6g} "
        f"rigid_g={metrics.scaled_gravity_accel_z:.6g} "
        f"buoy_accel={metrics.buoyancy_accel_z:.6g} "
        f"contact={metrics.water_contact_fraction:.6f} "
        f"initial_contact={metrics.initial_water_contact_fraction:.6f} "
        f"max_contact={metrics.max_water_contact_fraction:.6f} "
        f"sphere_density={metrics.sphere_density:.3f} "
        f"density=[{metrics.density_min:.6f}, {metrics.density_max:.6f}] "
        f"min_z={metrics.min_body_z:.6f} "
        f"hdisp={metrics.horizontal_displacement:.6f} "
        f"analytic_force={_format_vector(metrics.analytic_force)} "
        f"mem_force={_format_vector(metrics.mem_force)}"
    )
    return text


def setup_m1_renderer(viewer: Any, scene: DambreakMemM1Scene) -> ScreenSpaceFluidRenderer | None:
    """Attach rigid and screen-space fluid rendering when a GL viewer is available."""
    if viewer is None:
        return None

    if not isinstance(viewer, newton.viewer.ViewerNull):
        scene.base.rigid_domain.model.setup_viewer(viewer)

    if not isinstance(viewer, FluidViewerGL):
        return None

    viewer._paused: bool = True
    ssfr: ScreenSpaceFluidRenderer = ScreenSpaceFluidRenderer(
        viewer=viewer,
        max_particles=1,
        particle_radius=0.01,
        device=scene.base.fluid_domain.model._device,
    )

    def render_ssfr(current_viewer: Any) -> None:
        ssfr.render(current_viewer)

    viewer.register_post_render_callback(render_ssfr)
    print("Press Space to unpause.")
    return ssfr


def render_m1_frame(
    viewer: Any,
    scene: DambreakMemM1Scene,
    ssfr: ScreenSpaceFluidRenderer | None,
    sim_time: float,
) -> None:
    """Render one M1 visual frame."""
    if viewer is None:
        return

    viewer.begin_frame(float(sim_time))
    viewer.log_state(scene.base.rigid_domain.state.as_newton_state())

    if ssfr is not None and ssfr.available:
        ssfr.set_density_field(
            density=scene.base.display_density,
            grid_origin=(0.0, 0.0, 0.0),
            cell_size=scene.base.config.cell_size,
            threshold=FIELD_DENSITY_THRESHOLD,
            max_steps=RAY_MARCH_STEPS,
        )

    viewer.end_frame()

