# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-shape SDF rasterization for the independent HOME-LBM path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import warp as wp

from wanphys._src.geometry.rigid_shape_distance import (
    RigidShapeQuery,
    RigidShapeQueryData,
    point_body_distance,
)

from .state import HomeLbmState

if TYPE_CHECKING:
    from wanphys._src.rigid.domain import RigidDomain


@wp.kernel
def _rasterize_rigid_sdf(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    query: RigidShapeQueryData,
    coupled_body_ids: wp.array(dtype=wp.int32),
    coupled_body_count: int,
    cell_size: float,
):
    i, j, k = wp.tid()
    point_world = wp.vec3(
        (float(i) + 0.5) * cell_size,
        (float(j) + 0.5) * cell_size,
        (float(k) + 0.5) * cell_size,
    )
    best_distance = float(1000.0)
    best_body = int(-1)
    for entry in range(coupled_body_count):
        body = coupled_body_ids[entry]
        distance = point_body_distance(point_world, body, query, body_q)
        if distance < best_distance:
            best_distance = distance
            best_body = body

    solid_phi[i, j, k] = best_distance
    if best_distance < 0.0:
        solid_body_id[i, j, k] = best_body
    else:
        solid_body_id[i, j, k] = -1


@wp.kernel
def _rasterize_rigid_sdf_regions(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    region_body_q: wp.array(dtype=wp.transform),
    evaluation_body_q: wp.array(dtype=wp.transform),
    body_com_local: wp.array(dtype=wp.vec3),
    query: RigidShapeQueryData,
    coupled_body_ids: wp.array(dtype=wp.int32),
    region_half_width: wp.array(dtype=wp.int32),
    coupled_body_count: int,
    maximum_region_volume: int,
    cell_size: float,
    nx: int,
    ny: int,
    nz: int,
):
    entry, local_cell = wp.tid()
    half_width = region_half_width[entry]
    side = 2 * half_width + 1
    region_volume = side * side * side
    if local_cell >= region_volume or local_cell >= maximum_region_volume:
        return

    body = coupled_body_ids[entry]
    center_world = wp.transform_point(region_body_q[body], body_com_local[body])
    center_i = int(wp.floor(center_world[0] / cell_size))
    center_j = int(wp.floor(center_world[1] / cell_size))
    center_k = int(wp.floor(center_world[2] / cell_size))
    side_squared = side * side
    offset_i = local_cell // side_squared
    remainder = local_cell - offset_i * side_squared
    offset_j = remainder // side
    offset_k = remainder - offset_j * side
    i = center_i + offset_i - half_width
    j = center_j + offset_j - half_width
    k = center_k + offset_k - half_width
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return

    # Overlapping body regions are assigned to their first entry.  This keeps
    # every launch data-race free while still evaluating all bodies per cell.
    for owner_entry in range(entry):
        owner_body = coupled_body_ids[owner_entry]
        owner_center = wp.transform_point(
            region_body_q[owner_body], body_com_local[owner_body]
        )
        owner_i = int(wp.floor(owner_center[0] / cell_size))
        owner_j = int(wp.floor(owner_center[1] / cell_size))
        owner_k = int(wp.floor(owner_center[2] / cell_size))
        owner_half_width = region_half_width[owner_entry]
        if (
            wp.abs(i - owner_i) <= owner_half_width
            and wp.abs(j - owner_j) <= owner_half_width
            and wp.abs(k - owner_k) <= owner_half_width
        ):
            return

    point_world = wp.vec3(
        (float(i) + 0.5) * cell_size,
        (float(j) + 0.5) * cell_size,
        (float(k) + 0.5) * cell_size,
    )
    best_distance = float(1000.0)
    best_body = int(-1)
    for evaluation_entry in range(coupled_body_count):
        evaluation_body = coupled_body_ids[evaluation_entry]
        distance = point_body_distance(
            point_world,
            evaluation_body,
            query,
            evaluation_body_q,
        )
        if distance < best_distance:
            best_distance = distance
            best_body = evaluation_body

    solid_phi[i, j, k] = best_distance
    if best_distance < 0.0:
        solid_body_id[i, j, k] = best_body
    else:
        solid_body_id[i, j, k] = -1


@wp.kernel
def _compute_body_radius_bounds(
    query: RigidShapeQueryData,
    coupled_body_ids: wp.array(dtype=wp.int32),
    body_com_local: wp.array(dtype=wp.vec3),
    radius_bound: wp.array(dtype=float),
):
    entry = wp.tid()
    body = coupled_body_ids[entry]
    center = body_com_local[body]
    radius = float(0.0)
    shape_entry = query.body_shape_offsets[body]
    shape_end = query.body_shape_offsets[body + 1]
    while shape_entry < shape_end:
        shape = query.body_shape_indices[shape_entry]
        lower = query.shape_collision_aabb_lower[shape]
        upper = query.shape_collision_aabb_upper[shape]
        transform = query.shape_transform[shape]
        for corner_index in range(8):
            x = lower[0]
            y = lower[1]
            z = lower[2]
            if (corner_index & 1) != 0:
                x = upper[0]
            if (corner_index & 2) != 0:
                y = upper[1]
            if (corner_index & 4) != 0:
                z = upper[2]
            corner_body = wp.transform_point(transform, wp.vec3(x, y, z))
            radius = wp.max(radius, wp.length(corner_body - center))
        shape_entry += 1
    radius_bound[body] = radius


@dataclass(frozen=True)
class RigidGeometryUpdateDiagnostics:
    """Work estimate for the most recent rigid SDF update."""

    mode: str
    launched_cell_capacity: int
    full_grid_cell_count: int

    @property
    def capacity_ratio(self) -> float:
        if self.full_grid_cell_count == 0:
            return 0.0
        return self.launched_cell_capacity / self.full_grid_cell_count


class HomeLbmRigidGeometry:
    """Static shape query plus the subset of Newton bodies coupled to HOME."""

    def __init__(
        self,
        query: RigidShapeQuery,
        body_ids: Iterable[int] | None = None,
    ) -> None:
        selected = np.asarray(
            list(range(query.body_count)) if body_ids is None else list(body_ids),
            dtype=np.int32,
        )
        if selected.ndim != 1:
            raise ValueError("body_ids must be one-dimensional")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("body_ids must not contain duplicates")
        if np.any(selected < 0) or np.any(selected >= query.body_count):
            raise ValueError("body_ids contains an index outside the rigid model")

        self.query = query
        self.device = query.device
        self._body_ids_host = selected
        self.body_ids = wp.array(selected, dtype=wp.int32, device=self.device)
        self.body_count = len(selected)
        self.radius_bound = wp.zeros(query.body_count, dtype=float, device=self.device)
        self.previous_body_q = wp.zeros(
            query.body_count, dtype=wp.transform, device=self.device
        )
        self._body_com_local: wp.array | None = None
        self._region_half_width: wp.array | None = None
        self._region_cell_size: float | None = None
        self._maximum_region_volume = 0
        self._radius_bounds_dirty = True
        self._has_previous_geometry = False
        self.last_update = RigidGeometryUpdateDiagnostics("uninitialized", 0, 0)

    @classmethod
    def from_domain(
        cls,
        domain: RigidDomain,
        body_ids: Iterable[int] | None = None,
    ) -> HomeLbmRigidGeometry:
        return cls(RigidShapeQuery.from_domain(domain), body_ids=body_ids)

    def rasterize(
        self,
        state: HomeLbmState,
        body_q: wp.array,
    ) -> RigidGeometryUpdateDiagnostics:
        """Build an exact full-grid SDF and capture the pose as update history."""

        self._validate_raster_inputs(state, body_q)
        full_grid_cell_count = state.cell_count
        self._launch_full_raster(state, body_q)
        wp.copy(self.previous_body_q, body_q)
        self._has_previous_geometry = True
        self.last_update = RigidGeometryUpdateDiagnostics(
            mode="full",
            launched_cell_capacity=full_grid_cell_count,
            full_grid_cell_count=full_grid_cell_count,
        )
        return self.last_update

    def rasterize_incremental(
        self,
        state: HomeLbmState,
        body_q: wp.array,
    ) -> RigidGeometryUpdateDiagnostics:
        """Update only old/current body regions, or use a full-grid cost fallback."""

        self._validate_raster_inputs(state, body_q)
        if not self._has_previous_geometry:
            return self.rasterize(state, body_q)
        if self.body_count == 0:
            wp.copy(self.previous_body_q, body_q)
            self.last_update = RigidGeometryUpdateDiagnostics(
                mode="incremental",
                launched_cell_capacity=0,
                full_grid_cell_count=state.cell_count,
            )
            return self.last_update
        if self._body_com_local is None:
            raise RuntimeError(
                "update_radius_bounds() must be called before incremental rasterization"
            )

        self._configure_regions(float(state.model.fluid_grid_cell_size))
        region_capacity = 2 * self.body_count * self._maximum_region_volume
        if region_capacity >= state.cell_count:
            self._launch_full_raster(state, body_q)
            mode = "full-cost-fallback"
            launched_capacity = state.cell_count
        else:
            assert self._region_half_width is not None
            for region_body_q in (self.previous_body_q, body_q):
                wp.launch(
                    _rasterize_rigid_sdf_regions,
                    dim=(self.body_count, self._maximum_region_volume),
                    inputs=[
                        state.solid_phi,
                        state.solid_body_id,
                        region_body_q,
                        body_q,
                        self._body_com_local,
                        self.query.data,
                        self.body_ids,
                        self._region_half_width,
                        self.body_count,
                        self._maximum_region_volume,
                        float(state.model.fluid_grid_cell_size),
                        state.res[0],
                        state.res[1],
                        state.res[2],
                    ],
                    device=self.device,
                )
            mode = "incremental"
            launched_capacity = region_capacity

        wp.copy(self.previous_body_q, body_q)
        self.last_update = RigidGeometryUpdateDiagnostics(
            mode=mode,
            launched_cell_capacity=launched_capacity,
            full_grid_cell_count=state.cell_count,
        )
        return self.last_update

    def _validate_raster_inputs(self, state: HomeLbmState, body_q: wp.array) -> None:
        if state.device != self.device or body_q.device != self.device:
            raise ValueError("HOME state, rigid geometry, and rigid state devices must match")
        if body_q.dtype != wp.transform or len(body_q) != self.query.body_count:
            raise ValueError("body_q must contain one transform per rigid model body")

    def _launch_full_raster(self, state: HomeLbmState, body_q: wp.array) -> None:
        wp.launch(
            _rasterize_rigid_sdf,
            dim=state.res,
            inputs=[
                state.solid_phi,
                state.solid_body_id,
                body_q,
                self.query.data,
                self.body_ids,
                self.body_count,
                float(state.model.fluid_grid_cell_size),
            ],
            device=self.device,
        )

    def _configure_regions(self, cell_size: float) -> None:
        if not self._radius_bounds_dirty and self._region_cell_size == cell_size:
            return
        wp.synchronize_device(self.device)
        radii = self.radius_bound.numpy()[self._body_ids_host]
        half_width = np.ceil(radii / cell_size).astype(np.int32) + 2
        half_width = np.maximum(half_width, 2)
        self._region_half_width = wp.array(
            half_width, dtype=wp.int32, device=self.device
        )
        maximum_side = int(2 * int(half_width.max()) + 1)
        self._maximum_region_volume = maximum_side**3
        self._region_cell_size = cell_size
        self._radius_bounds_dirty = False

    def update_radius_bounds(self, body_com_local: wp.array) -> wp.array:
        if body_com_local.device != self.device or body_com_local.dtype != wp.vec3:
            raise ValueError("body_com_local must be a vec3 array on the geometry device")
        if len(body_com_local) != self.query.body_count:
            raise ValueError("body_com_local must contain one entry per rigid model body")
        self._body_com_local = body_com_local
        self._radius_bounds_dirty = True
        self.radius_bound.zero_()
        if self.body_count:
            wp.launch(
                _compute_body_radius_bounds,
                dim=self.body_count,
                inputs=[
                    self.query.data,
                    self.body_ids,
                    body_com_local,
                    self.radius_bound,
                ],
                device=self.device,
            )
        return self.radius_bound
