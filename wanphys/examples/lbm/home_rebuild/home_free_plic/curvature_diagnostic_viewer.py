# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Show read-only bulk 3-D curvature on the frozen link-wise baseline."""

from __future__ import annotations

import math
import time
from typing import Any

import newton.examples
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicCurvatureEstimator3D,
    PlicCurvatureState,
    PlicGeometryState,
)
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    make_viewer_config,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.plic_diagnostic_viewer import (
    PlicDiagnosticGravityColumnViewer,
)


@wp.kernel
def _pack_sparse_curvature_lines(
    valid: wp.array3d(dtype=wp.int32),
    normal: wp.array3d(dtype=wp.vec3),
    curvature: wp.array3d(dtype=float),
    stride: int,
    sample_ny: int,
    sample_nz: int,
    cell_size: float,
    line_length: float,
    curvature_scale: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
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
    colors[tid] = wp.vec3(0.65, 0.68, 0.72)
    if valid[x, y, z] == 0:
        return
    source_normal = normal[x, y, z]
    magnitude = wp.length(source_normal)
    value = curvature[x, y, z]
    if magnitude <= 1.0e-8 or not wp.isfinite(value):
        return
    render_normal = wp.vec3(
        source_normal[0], source_normal[2], source_normal[1]
    ) / magnitude
    center = wp.vec3(
        (float(x) + 0.5) * cell_size,
        (float(z) + 0.5) * cell_size,
        (float(y) + 0.5) * cell_size,
    )
    normalized = wp.min(wp.abs(value) / curvature_scale, 1.0)
    starts[tid] = center
    ends[tid] = center + render_normal * line_length * (0.35 + 1.65 * normalized)
    if value < -1.0e-5:
        colors[tid] = wp.vec3(0.1, 0.55, 1.0)
    elif value > 1.0e-5:
        colors[tid] = wp.vec3(0.95, 0.28, 0.12)


class PlicCurvatureRenderField:
    def __init__(
        self,
        shape: tuple[int, int, int],
        curvature: PlicCurvatureState,
        *,
        device: str | wp.Device,
        stride: int = 3,
        cell_size: float = 0.02,
        curvature_scale: float = 0.5,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be positive")
        if curvature_scale <= 0.0:
            raise ValueError("curvature_scale must be positive")
        self.shape = tuple(int(value) for value in shape)
        self.curvature = curvature
        self.device = wp.get_device(device)
        self.stride = int(stride)
        self.sample_shape = tuple(
            int(math.ceil(value / self.stride)) for value in self.shape
        )
        self.line_count = math.prod(self.sample_shape)
        self.cell_size = float(cell_size)
        self.line_length = 1.2 * self.cell_size
        self.curvature_scale = float(curvature_scale)
        self.starts = wp.empty(self.line_count, dtype=wp.vec3, device=self.device)
        self.ends = wp.empty(self.line_count, dtype=wp.vec3, device=self.device)
        self.colors = wp.empty(self.line_count, dtype=wp.vec3, device=self.device)

    def update(self, geometry: PlicGeometryState) -> tuple[wp.array, wp.array]:
        if geometry.res != self.shape or self.curvature.model is not geometry.model:
            raise ValueError("curvature overlay fields must share one grid")
        wp.launch(
            _pack_sparse_curvature_lines,
            dim=self.line_count,
            inputs=[
                self.curvature.valid,
                geometry.normal,
                self.curvature.curvature,
                self.stride,
                self.sample_shape[1],
                self.sample_shape[2],
                self.cell_size,
                self.line_length,
                self.curvature_scale,
            ],
            outputs=[self.starts, self.ends, self.colors],
            device=self.device,
            record_tape=False,
        )
        return self.starts, self.ends


class CurvatureDiagnosticGravityColumnViewer(PlicDiagnosticGravityColumnViewer):
    """Frozen simulation with signed curvature colors and no force feedback."""

    def _build_simulation(self) -> None:
        super()._build_simulation()
        self.curvature_state = PlicCurvatureState(self.model)
        self.curvature_estimator = PlicCurvatureEstimator3D(
            self.model, self.walls, strict_bulk=False
        )
        self.plic_lines = PlicCurvatureRenderField(
            self.render_field.source_shape,
            self.curvature_state,
            device=self.model._device,
            stride=self.normal_stride,
            cell_size=(
                self.render_cell_size
                if hasattr(self, "render_cell_size")
                else 0.02
            ),
        )
        print(
            "B5 curvature: blue negative, orange positive, gray near zero; "
            "P toggles overlay; no force feedback"
        )

    def _observe_plic(self) -> None:
        if self._plic_observed_step == self.sim_step:
            return
        self.plic_observer.observe(self.fsl_a)
        started = time.perf_counter()
        diagnostics = self.curvature_estimator.reconstruct(
            self.fsl_a.fill_level,
            self.fsl_a.flags,
            self.plic_observer.geometry,
            self.curvature_state,
        )
        wp.synchronize_device(self.model._device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._plic_observed_step = self.sim_step
        if self.print_every and self.sim_step % self.print_every < self.substeps_per_frame:
            print(
                f"B5 curvature step={self.sim_step} "
                f"required={diagnostics.required_cell_count} "
                f"valid={diagnostics.valid_cell_count} "
                f"wall_skip={diagnostics.wall_contact_skipped_count} "
                f"bulk_unresolved="
                f"{diagnostics.insufficient_neighbor_count + diagnostics.ill_conditioned_count} "
                f"observe_ms={elapsed_ms:.3f}"
            )


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--substeps-per-frame", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--frame-delay", type=float, default=0.0)
    parser.add_argument("--normal-stride", type=int, default=3)
    viewer, args = init_fluid_viewer(parser)
    example = CurvatureDiagnosticGravityColumnViewer(
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
