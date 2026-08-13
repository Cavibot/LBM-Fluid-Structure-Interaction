# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only conversion of VOF-shaped views into visual primitives.

This module may derive a render mask, compact points, and transform render
coordinates.  It must not derive or modify physical phi, normal, or curvature.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .kernels.visualization import compact_interface_visuals_kernel
from ..state import DebugMockScToVofState, VofGridState


@dataclass(frozen=True)
class DebugVofView:
    """Read-only visual input resolved from one VOF-shaped state source."""

    phi: wp.array
    cell_type: wp.array
    normal: wp.array
    epoch: int
    source: str

    @classmethod
    def from_shan_chen_mock(
        cls,
        debug_mock: DebugMockScToVofState,
    ) -> DebugVofView:
        return cls(
            phi=debug_mock.phi,
            cell_type=debug_mock.cell_type,
            normal=debug_mock.normal,
            epoch=debug_mock.epoch,
            source="shan_chen_mock",
        )

    @classmethod
    def from_authoritative_vof(
        cls,
        vof: VofGridState,
    ) -> DebugVofView:
        """Adapt authoritative P6 geometry without copying or deriving fields."""

        if vof.epoch < 0 or vof.geometry_epoch != vof.epoch:
            raise ValueError(
                "authoritative VOF visual view requires current geometry"
            )
        return cls(
            phi=vof.phi,
            cell_type=vof.cell_type,
            normal=vof.normal,
            epoch=vof.epoch,
            source="authoritative_vof",
        )


@dataclass(frozen=True)
class InterfaceVisualData:
    """Compacted point and normal-line views for one interface epoch."""

    points: wp.array
    normal_starts: wp.array
    normal_ends: wp.array
    count: int


class VofInterfaceVisualizer:
    """Own reusable scratch and render an interface without changing the viewer."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        cell_size: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        point_radius_scale: float = 0.16,
        normal_length_scale: float = 0.75,
    ) -> None:
        self.shape = shape
        self.device = device
        self.cell_size = float(cell_size)
        self.origin = tuple(float(value) for value in origin)
        self.point_radius = float(point_radius_scale) * self.cell_size
        self.normal_length = float(normal_length_scale) * self.cell_size
        capacity = shape[0] * shape[1] * shape[2]
        self._count = wp.zeros(1, dtype=wp.int32, device=device)
        self._points = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._normal_ends = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._point_radii = wp.full(
            capacity,
            self.point_radius,
            dtype=wp.float32,
            device=device,
        )
        self._point_colors = wp.full(
            capacity,
            wp.vec3(1.0, 0.0, 0.0),
            dtype=wp.vec3,
            device=device,
        )

    def compact(
        self,
        view: DebugVofView,
        solid_phi: wp.array3d,
    ) -> InterfaceVisualData:
        self._count.zero_()
        wp.launch(
            compact_interface_visuals_kernel,
            dim=self.shape,
            inputs=[
                view.cell_type,
                view.phi,
                view.normal,
                solid_phi,
                self._count,
                self._points,
                self._normal_ends,
                self._point_colors,
                *self.origin,
                self.cell_size,
                self.normal_length,
            ],
            device=self.device,
        )
        count = int(self._count.numpy()[0])
        return InterfaceVisualData(
            points=self._points[:count],
            normal_starts=self._points[:count],
            normal_ends=self._normal_ends[:count],
            count=count,
        )

    def render(
        self,
        viewer: object,
        view: DebugVofView,
        solid_phi: wp.array3d,
        show_normals: bool = True,
    ) -> int:
        data = self.compact(view, solid_phi)
        if data.count == 0:
            viewer.log_points("vof/interface", points=None)
            viewer.log_lines(
                "vof/normals",
                starts=None,
                ends=None,
                colors=None,
            )
            return 0
        viewer.log_points(
            "vof/interface",
            points=data.points,
            radii=self._point_radii[:data.count],
            colors=self._point_colors[:data.count],
        )
        if show_normals:
            viewer.log_lines(
                "vof/normals",
                starts=data.normal_starts,
                ends=data.normal_ends,
                colors=(0.1, 0.9, 1.0),
                width=0.015 * self.cell_size,
            )
        else:
            viewer.log_lines(
                "vof/normals",
                starts=None,
                ends=None,
                colors=None,
            )
        return data.count


__all__ = [
    "DebugVofView",
    "InterfaceVisualData",
    "VofInterfaceVisualizer",
    "compact_interface_visuals_kernel",
]
