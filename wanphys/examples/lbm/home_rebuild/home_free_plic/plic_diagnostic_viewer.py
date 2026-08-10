# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Overlay read-only PLIC geometry on the frozen link-wise HOME-Free Viewer."""

from __future__ import annotations

import math
import time
from typing import Any

import newton.examples
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicGeometryDiagnostics,
    PlicGeometryReconstructor,
    PlicGeometryState,
)
from wanphys._src.fluid.fluid_viewer import FluidViewerGL
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    RAY_MARCH_STEPS,
    SSFR_THRESHOLD,
    HomeFreeGravityColumnViewer,
    make_viewer_config,
)


@wp.kernel
def _pack_sparse_normal_lines(
    valid: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    stride: int,
    sample_ny: int,
    sample_nz: int,
    cell_size: float,
    line_length: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    yz = sample_ny * sample_nz
    sx = tid // yz
    remainder = tid - sx * yz
    sy = remainder // sample_nz
    sz = remainder - sy * sample_nz
    x = sx * stride
    y = sy * stride
    z = sz * stride
    hidden = wp.vec3(-1000.0, -1000.0, -1000.0)
    starts[tid] = hidden
    ends[tid] = hidden
    if valid[x, y, z] != 0:
        source_normal = normal[x, y, z]
        magnitude = wp.length(source_normal)
        if magnitude > 1.0e-8:
            # HOME is y-up and ViewerGL is z-up.
            render_normal = wp.vec3(
                source_normal[0], source_normal[2], source_normal[1]
            ) / magnitude
            center = wp.vec3(
                (float(x) + 0.5) * cell_size,
                (float(z) + 0.5) * cell_size,
                (float(y) + 0.5) * cell_size,
            )
            starts[tid] = center
            ends[tid] = center + render_normal * line_length


class ReadOnlyPlicObserver:
    """Reconstruct PLIC geometry without owning or modifying simulation state."""

    def __init__(self, model: Any, walls: Any) -> None:
        self.geometry = PlicGeometryState(model)
        self.reconstructor = PlicGeometryReconstructor(model, walls)
        self.last_diagnostics: PlicGeometryDiagnostics | None = None
        self.last_elapsed_ms = 0.0

    def observe(self, fsl_state: Any) -> PlicGeometryDiagnostics:
        started = time.perf_counter()
        diagnostics = self.reconstructor.reconstruct(fsl_state, self.geometry)
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_diagnostics = diagnostics
        return diagnostics


class PlicNormalRenderField:
    """Build a sparse, fixed-size ViewerGL line overlay from PLIC normals."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        *,
        device: str | wp.Device,
        stride: int = 4,
        cell_size: float = 0.02,
        line_length_cells: float = 1.8,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be positive")
        self.shape = tuple(int(value) for value in shape)
        self.device = wp.get_device(device)
        self.stride = int(stride)
        self.sample_shape = tuple(
            int(math.ceil(value / self.stride)) for value in self.shape
        )
        self.line_count = math.prod(self.sample_shape)
        self.cell_size = float(cell_size)
        self.line_length = float(line_length_cells) * self.cell_size
        self.starts = wp.empty(self.line_count, dtype=wp.vec3, device=self.device)
        self.ends = wp.empty(self.line_count, dtype=wp.vec3, device=self.device)
        self.colors = wp.full(
            self.line_count,
            value=wp.vec3(0.95, 0.2, 0.12),
            dtype=wp.vec3,
            device=self.device,
        )

    def update(self, geometry: PlicGeometryState) -> tuple[wp.array, wp.array]:
        if geometry.res != self.shape:
            raise ValueError("PLIC geometry shape does not match the normal overlay")
        wp.launch(
            _pack_sparse_normal_lines,
            dim=self.line_count,
            inputs=[
                geometry.valid,
                geometry.normal,
                self.stride,
                self.sample_shape[1],
                self.sample_shape[2],
                self.cell_size,
                self.line_length,
            ],
            outputs=[self.starts, self.ends],
            device=self.device,
            record_tape=False,
        )
        return self.starts, self.ends


class PlicDiagnosticGravityColumnViewer(HomeFreeGravityColumnViewer):
    """Frozen link-wise simulation plus a toggleable read-only PLIC overlay."""

    def __init__(
        self,
        viewer: Any,
        *,
        normal_stride: int = 4,
        **kwargs: Any,
    ) -> None:
        self.normal_stride = int(normal_stride)
        self.plic_visible = True
        super().__init__(viewer, **kwargs)
        print("B1 PLIC diagnostics: P toggle normals; PLIC does not modify the solve")

    def _build_simulation(self) -> None:
        super()._build_simulation()
        self.plic_observer = ReadOnlyPlicObserver(self.model, self.walls)
        self.plic_lines = PlicNormalRenderField(
            self.render_field.source_shape,
            device=self.model._device,
            stride=self.normal_stride,
            cell_size=self.render_cell_size if hasattr(self, "render_cell_size") else 0.02,
        )
        self._plic_observed_step = -1

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        import pyglet

        if symbol == pyglet.window.key.P:
            self.plic_visible = not self.plic_visible
            state = "visible" if self.plic_visible else "hidden"
            print(f"B1 read-only PLIC normals: {state}")
            return
        super()._on_key_press(symbol, modifiers)

    def _observe_plic(self) -> None:
        if self._plic_observed_step == self.sim_step:
            return
        diagnostics = self.plic_observer.observe(self.fsl_a)
        self._plic_observed_step = self.sim_step
        if self.print_every and self.sim_step % self.print_every < self.substeps_per_frame:
            print(
                f"B1 PLIC step={self.sim_step} interfaces={diagnostics.interface_cell_count} "
                f"invalid={diagnostics.invalid_interface_count} "
                f"max_area={diagnostics.maximum_interface_area:.6g} "
                f"observe_ms={self.plic_observer.last_elapsed_ms:.3f}"
            )

    def render(self) -> None:
        self._observe_plic()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/home-free/tank",
            self.tank_starts,
            self.tank_ends,
            (0.18, 0.22, 0.28),
            width=0.008,
        )
        if self.plic_visible:
            starts, ends = self.plic_lines.update(self.plic_observer.geometry)
            self.viewer.log_lines(
                "/home-free/plic-normals",
                starts,
                ends,
                self.plic_lines.colors,
                width=0.003,
            )
        else:
            self.viewer.log_lines(
                "/home-free/plic-normals", None, None, None
            )
        if self.ssfr is not None and self.ssfr.available:
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
    parser.add_argument("--substeps-per-frame", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--frame-delay", type=float, default=0.0)
    parser.add_argument("--normal-stride", type=int, default=4)
    viewer, args = init_fluid_viewer(parser)
    example = PlicDiagnosticGravityColumnViewer(
        viewer,
        config=make_viewer_config("viewer-default", args.device or "cuda:0"),
        substeps_per_frame=args.substeps_per_frame,
        print_every=args.print_every,
        frame_delay=args.frame_delay,
        normal_stride=args.normal_stride,
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
