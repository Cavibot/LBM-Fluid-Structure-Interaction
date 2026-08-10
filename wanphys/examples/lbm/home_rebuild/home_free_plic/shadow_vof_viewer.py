# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare frozen link-wise HOME-Free with an independent raw-Courant PLIC shadow."""

from __future__ import annotations

import time
from typing import Any

import newton.examples
import warp as wp

from wanphys._src.fluid.fluid_viewer import HomeFreeRenderField
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    RAY_MARCH_STEPS,
    SSFR_THRESHOLD,
    HomeFreeGravityColumnViewer,
    make_viewer_config,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.shadow_vof import (
    ShadowPlicVof,
)


class ShadowPlicVofGravityColumnViewer(HomeFreeGravityColumnViewer):
    """Run the baseline and shadow VOF side by side without state feedback."""

    def __init__(
        self,
        viewer: Any,
        *,
        project_courant: bool = False,
        **kwargs: Any,
    ) -> None:
        self.project_courant = bool(project_courant)
        self.show_shadow = True
        self.shadow_frozen = False
        self.shadow_error: str | None = None
        super().__init__(viewer, **kwargs)
        mode = "projected" if self.project_courant else "raw-Courant"
        print(f"B2 shadow VOF ({mode}): V toggle baseline/shadow; no feedback")

    def _build_simulation(self) -> None:
        super()._build_simulation()
        self.shadow = ShadowPlicVof(
            self.model,
            self.walls,
            self.fsl_a.fill_level,
            project_courant=self.project_courant,
        )
        self.shadow_flags = wp.ones(
            self.render_field.source_shape,
            dtype=wp.int32,
            device=self.model._device,
        )
        self.shadow_render_field = HomeFreeRenderField(
            self.render_field.source_shape,
            device=self.model._device,
            gas_flag=0,
        )
        self.shadow_frozen = False
        self.shadow_error = None

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.V:
            self.show_shadow = not self.show_shadow
            mode = "raw-Courant PLIC shadow" if self.show_shadow else "link-wise baseline"
            print(f"B2 display: {mode}")
            return
        super()._on_key_press(symbol, modifiers)

    def step(self) -> None:
        started = time.perf_counter()
        for _ in range(self.substeps_per_frame):
            diagnostics = self.stepper.step(
                self.fluid_a,
                self.fsl_a,
                self.fluid_b,
                self.fsl_b,
                self.model.time_step,
            )
            self.fluid_a, self.fluid_b = self.fluid_b, self.fluid_a
            self.fsl_a, self.fsl_b = self.fsl_b, self.fsl_a
            self.sim_step += 1
            self.sim_time += self.model.time_step
            self.last_max_speed = max(
                self.last_max_speed, float(diagnostics.fluid.max_speed)
            )
            if not self.shadow_frozen:
                try:
                    self.shadow.advance(self.fluid_a, self.fsl_a)
                except Exception as error:
                    self.shadow_frozen = True
                    self.shadow_error = str(error)
                    print(
                        f"B2 shadow frozen at step {self.sim_step}: "
                        f"{type(error).__name__}: {error}"
                    )
        wp.synchronize_device(self.model._device)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        if self.frame_delay:
            time.sleep(self.frame_delay)
        if self.print_every and self.sim_step % self.print_every < self.substeps_per_frame:
            shadow = self.shadow.last_diagnostics
            shadow_text = "shadow=unavailable"
            if shadow is not None:
                projected = ""
                if shadow.projected_maximum_divergence is not None:
                    projected = (
                        f" projected_div={shadow.projected_maximum_divergence:.3g} "
                        f"projection_iter={shadow.projection_iteration_count}"
                    )
                shadow_text = (
                    f"shadow_drift={shadow.cumulative_relative_volume_drift:.6g} "
                    f"face_div={shadow.maximum_active_divergence:.6g} "
                    f"violations={shadow.drift_violation_count}{projected}"
                )
            print(
                f"B2 step={self.sim_step} max_speed={self.last_max_speed:.6g} "
                f"batch_ms={self.last_step_ms:.3f} {shadow_text}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        tank_color = (
            (0.55, 0.18, 0.08) if self.show_shadow else (0.18, 0.22, 0.28)
        )
        self.viewer.log_lines(
            "/home-free/tank",
            self.tank_starts,
            self.tank_ends,
            tank_color,
            width=0.008,
        )
        if self.ssfr is not None and self.ssfr.available:
            if self.show_shadow:
                density = self.shadow_render_field.update(
                    self.shadow.fill_level, self.shadow_flags
                )
            else:
                density = self.render_field.update(
                    self.fsl_a.fill_level, self.fsl_a.flags
                )
            self.ssfr.set_density_field(
                density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=self.render_cell_size,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--substeps-per-frame", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--frame-delay", type=float, default=0.0)
    parser.add_argument("--project-courant", action="store_true")
    viewer, args = init_fluid_viewer(parser)
    example = ShadowPlicVofGravityColumnViewer(
        viewer,
        config=make_viewer_config("viewer-default", args.device or "cuda:0"),
        substeps_per_frame=args.substeps_per_frame,
        print_every=args.print_every,
        frame_delay=args.frame_delay,
        project_courant=args.project_courant,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
