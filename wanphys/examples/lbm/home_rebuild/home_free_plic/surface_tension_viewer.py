# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare bulk PLIC surface tension with the frozen Viewer baseline."""

from __future__ import annotations

from typing import Any

import newton.examples

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicCapillaryWallStepper,
    PlicSurfaceTension,
)
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    HomeFreeGravityColumnViewer,
    make_viewer_config,
)


class SurfaceTensionGravityColumnViewer(HomeFreeGravityColumnViewer):
    """Viewer-default dam break with one toggleable capillary-pressure term."""

    def __init__(
        self,
        viewer: Any,
        *,
        lattice_surface_tension: float = 5.0e-4,
        **kwargs: Any,
    ) -> None:
        if lattice_surface_tension <= 0.0:
            raise ValueError("lattice surface tension must be positive")
        self.lattice_surface_tension = float(lattice_surface_tension)
        self.capillary_enabled = True
        super().__init__(viewer, **kwargs)
        print(
            "B6 surface tension: T toggles capillary/baseline and resets; "
            "wall contact remains ambient until B7"
        )

    def _build_simulation(self) -> None:
        super()._build_simulation()
        self.surface_tension = PlicSurfaceTension(
            self.model,
            self.walls,
            ambient_gas_density=self.config.gas_density,
            surface_tension=self.model.scaling.surface_tension_to_physical(
                self.lattice_surface_tension
            ),
        )
        if self.capillary_enabled:
            self.stepper = PlicCapillaryWallStepper(
                self.stepper, self.surface_tension
            )

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.T:
            self.viewer._paused = True
            self.capillary_enabled = not self.capillary_enabled
            self._build_simulation()
            state = "capillary" if self.capillary_enabled else "baseline"
            print(f"B6 reset in {state} mode; press Space to run")
            return
        super()._on_key_press(symbol, modifiers)


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--substeps-per-frame", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--frame-delay", type=float, default=0.0)
    parser.add_argument("--lattice-surface-tension", type=float, default=5.0e-4)
    viewer, args = init_fluid_viewer(parser)
    example = SurfaceTensionGravityColumnViewer(
        viewer,
        config=make_viewer_config("viewer-default", args.device or "cuda:0"),
        substeps_per_frame=args.substeps_per_frame,
        print_every=args.print_every,
        frame_delay=args.frame_delay,
        lattice_surface_tension=args.lattice_surface_tension,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
