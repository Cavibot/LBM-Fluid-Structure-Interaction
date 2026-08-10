# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse cut-link extraction from a cell-centred signed-distance field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .constants import D3Q27_DIRECTIONS
from .state import HomeLbmState


@wp.func
def _wrap_cut_source(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.kernel
def _count_cut_links(
    solid_phi: wp.array3d(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    count = int(0)
    if solid_phi[i, j, k] >= 0.0:
        for direction in range(1, 27):
            c = directions[direction]
            source_i = i - int(c[0])
            source_j = j - int(c[1])
            source_k = k - int(c[2])
            source_i = _wrap_cut_source(source_i, nx, periodic_x)
            source_j = _wrap_cut_source(source_j, ny, periodic_y)
            source_k = _wrap_cut_source(source_k, nz, periodic_z)
            if (
                source_i >= 0
                and source_j >= 0
                and source_k >= 0
                and solid_phi[source_i, source_j, source_k] < 0.0
            ):
                count += 1
    counts[cell] = count


@wp.kernel
def _write_cut_links(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=wp.int32),
    link_cell: wp.array(dtype=wp.int32),
    link_direction: wp.array(dtype=wp.int32),
    link_fraction: wp.array(dtype=float),
    link_body_id: wp.array(dtype=wp.int32),
    link_intersection_lattice: wp.array(dtype=wp.vec3),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if solid_phi[i, j, k] < 0.0:
        return
    cell = i * ny * nz + j * nz + k
    write_index = offsets[cell]
    phi_fluid = solid_phi[i, j, k]
    for direction in range(1, 27):
        c = directions[direction]
        source_i = i - int(c[0])
        source_j = j - int(c[1])
        source_k = k - int(c[2])
        source_i = _wrap_cut_source(source_i, nx, periodic_x)
        source_j = _wrap_cut_source(source_j, ny, periodic_y)
        source_k = _wrap_cut_source(source_k, nz, periodic_z)
        if (
            source_i >= 0
            and source_j >= 0
            and source_k >= 0
        ):
            phi_solid = solid_phi[source_i, source_j, source_k]
            if phi_solid < 0.0:
                denominator = phi_fluid - phi_solid
                fraction = 0.5
                if denominator > 1.0e-12:
                    fraction = wp.clamp(phi_fluid / denominator, 0.0, 1.0)
                link_cell[write_index] = cell
                link_direction[write_index] = direction
                link_fraction[write_index] = fraction
                link_body_id[write_index] = solid_body_id[source_i, source_j, source_k]
                link_intersection_lattice[write_index] = wp.vec3(
                    float(i) + 0.5 - fraction * c[0],
                    float(j) + 0.5 - fraction * c[1],
                    float(k) + 0.5 - fraction * c[2],
                )
                write_index += 1


@wp.kernel
def _set_wall_velocity(wall_velocity: wp.array(dtype=wp.vec3), velocity: wp.vec3):
    wall_velocity[wp.tid()] = velocity


@wp.kernel
def _set_rigid_wall_velocity(
    body_id: wp.array(dtype=wp.int32),
    intersection_lattice: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com_local: wp.array(dtype=wp.vec3),
    wall_velocity: wp.array(dtype=wp.vec3),
    invalid_body_id_count: wp.array(dtype=wp.int32),
    body_count: int,
    cell_size: float,
    time_step: float,
):
    link = wp.tid()
    body = body_id[link]
    if body < 0 or body >= body_count:
        wall_velocity[link] = wp.vec3(0.0)
        wp.atomic_add(invalid_body_id_count, 0, 1)
        return

    transform = body_q[body]
    velocity = body_qd[body]
    center_world = wp.transform_point(transform, body_com_local[body])
    point_world = intersection_lattice[link] * cell_size
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    surface_world = linear + wp.cross(angular, point_world - center_world)
    wall_velocity[link] = surface_world * (time_step / cell_size)


@dataclass
class CutLinkBuffer:
    """Cell-grouped sparse links crossing from fluid into solid."""

    counts: wp.array
    offsets: wp.array
    cell: wp.array
    direction: wp.array
    fraction: wp.array
    body_id: wp.array
    intersection_lattice: wp.array
    wall_velocity: wp.array
    impulse: wp.array
    invalid_body_id_count: wp.array
    link_count: int
    cell_count: int
    device: wp.Device
    _directions: wp.array

    @classmethod
    def build_from_sdf(cls, state: HomeLbmState) -> CutLinkBuffer:
        nx, ny, nz = state.res
        cell_count = state.cell_count
        device = state.device
        directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device
        )
        counts = wp.zeros(cell_count, dtype=wp.int32, device=device)
        offsets = wp.zeros(cell_count, dtype=wp.int32, device=device)
        wp.launch(
            _count_cut_links,
            dim=(nx, ny, nz),
            inputs=[
                state.solid_phi,
                directions,
                counts,
                int(state.model.periodic[0]),
                int(state.model.periodic[1]),
                int(state.model.periodic[2]),
                nx,
                ny,
                nz,
            ],
            device=device,
        )
        wp.utils.array_scan(counts, offsets, inclusive=False)
        wp.synchronize_device(device)
        counts_host = counts.numpy()
        offsets_host = offsets.numpy()
        link_count = int(offsets_host[-1] + counts_host[-1]) if cell_count else 0

        cell = wp.empty(link_count, dtype=wp.int32, device=device)
        direction = wp.empty(link_count, dtype=wp.int32, device=device)
        fraction = wp.empty(link_count, dtype=float, device=device)
        body_id = wp.empty(link_count, dtype=wp.int32, device=device)
        intersection_lattice = wp.empty(link_count, dtype=wp.vec3, device=device)
        wall_velocity = wp.zeros(link_count, dtype=wp.vec3, device=device)
        impulse = wp.zeros(link_count, dtype=wp.vec3, device=device)
        invalid_body_id_count = wp.zeros(1, dtype=wp.int32, device=device)
        if link_count:
            wp.launch(
                _write_cut_links,
                dim=(nx, ny, nz),
                inputs=[
                    state.solid_phi,
                    state.solid_body_id,
                    directions,
                    offsets,
                    cell,
                    direction,
                    fraction,
                    body_id,
                    intersection_lattice,
                    int(state.model.periodic[0]),
                    int(state.model.periodic[1]),
                    int(state.model.periodic[2]),
                    nx,
                    ny,
                    nz,
                ],
                device=device,
            )
        return cls(
            counts=counts,
            offsets=offsets,
            cell=cell,
            direction=direction,
            fraction=fraction,
            body_id=body_id,
            intersection_lattice=intersection_lattice,
            wall_velocity=wall_velocity,
            impulse=impulse,
            invalid_body_id_count=invalid_body_id_count,
            link_count=link_count,
            cell_count=cell_count,
            device=device,
            _directions=directions,
        )

    def set_uniform_wall_velocity(self, velocity: tuple[float, float, float]) -> None:
        if self.link_count:
            wp.launch(
                _set_wall_velocity,
                dim=self.link_count,
                inputs=[self.wall_velocity, wp.vec3(*velocity)],
                device=self.device,
            )

    def update_from_sdf(self, state: HomeLbmState) -> None:
        """Rewrite link geometry in place when the solid mask is unchanged."""

        if state.cell_count != self.cell_count or state.device != self.device:
            raise ValueError("cut-link buffer and HOME state must share grid and device")
        if self.link_count:
            nx, ny, nz = state.res
            wp.launch(
                _write_cut_links,
                dim=(nx, ny, nz),
                inputs=[
                    state.solid_phi,
                    state.solid_body_id,
                    self._directions,
                    self.offsets,
                    self.cell,
                    self.direction,
                    self.fraction,
                    self.body_id,
                    self.intersection_lattice,
                    int(state.model.periodic[0]),
                    int(state.model.periodic[1]),
                    int(state.model.periodic[2]),
                    nx,
                    ny,
                    nz,
                ],
                device=self.device,
            )

    def set_rigid_wall_velocity(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        body_com_local: wp.array,
        cell_size: float,
        time_step: float,
    ) -> None:
        body_count = len(body_q)
        if body_q.device != self.device or body_qd.device != self.device:
            raise ValueError("rigid state and cut-link devices must match")
        if body_com_local.device != self.device:
            raise ValueError("rigid centers and cut-link devices must match")
        if body_q.dtype != wp.transform or body_qd.dtype != wp.spatial_vector:
            raise ValueError("body_q/body_qd must use transform/spatial_vector dtypes")
        if body_com_local.dtype != wp.vec3:
            raise ValueError("body_com_local must use vec3 dtype")
        if len(body_qd) != body_count or len(body_com_local) != body_count:
            raise ValueError("rigid transform, velocity, and center arrays must have equal lengths")
        if cell_size <= 0.0 or time_step <= 0.0:
            raise ValueError("cell_size and time_step must be positive")

        self.invalid_body_id_count.zero_()
        if self.link_count:
            wp.launch(
                _set_rigid_wall_velocity,
                dim=self.link_count,
                inputs=[
                    self.body_id,
                    self.intersection_lattice,
                    body_q,
                    body_qd,
                    body_com_local,
                    self.wall_velocity,
                    self.invalid_body_id_count,
                    body_count,
                    float(cell_size),
                    float(time_step),
                ],
                device=self.device,
            )

    def validate_body_ids(self) -> None:
        wp.synchronize_device(self.device)
        invalid_count = int(self.invalid_body_id_count.numpy()[0])
        if invalid_count:
            raise ValueError(f"{invalid_count} cut links reference invalid rigid body IDs")

    @property
    def dense_link_capacity(self) -> int:
        return 26 * self.cell_count

    @property
    def occupancy(self) -> float:
        if self.dense_link_capacity == 0:
            return 0.0
        return self.link_count / self.dense_link_capacity
