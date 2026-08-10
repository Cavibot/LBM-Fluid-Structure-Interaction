# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative moment remapping for cells covered or uncovered by solids."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .state import HomeLbmState


@wp.func
def _wrap_neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.kernel
def _count_cell_transition_neighbors(
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    donor_counts: wp.array(dtype=wp.int32),
    fresh_count: wp.array(dtype=wp.int32),
    dead_count: wp.array(dtype=wp.int32),
    unresolved_count: wp.array(dtype=wp.int32),
    persistent_fluid_count: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    was_fluid = previous_phi[i, j, k] >= 0.0
    is_fluid = current_phi[i, j, k] >= 0.0
    cell = i * ny * nz + j * nz + k
    if was_fluid and is_fluid:
        wp.atomic_add(persistent_fluid_count, 0, 1)
    if was_fluid == is_fluid:
        donor_counts[cell] = 0
        return

    donor_count = int(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di != 0 or dj != 0 or dk != 0:
                    ni = _wrap_neighbor(i + di, nx, periodic_x)
                    nj = _wrap_neighbor(j + dj, ny, periodic_y)
                    nk = _wrap_neighbor(k + dk, nz, periodic_z)
                    if ni >= 0 and nj >= 0 and nk >= 0:
                        if previous_phi[ni, nj, nk] >= 0.0 and current_phi[ni, nj, nk] >= 0.0:
                            donor_count += 1

    donor_counts[cell] = donor_count
    if donor_count == 0:
        wp.atomic_add(unresolved_count, 0, 1)
        return
    if was_fluid:
        wp.atomic_add(dead_count, 0, 1)
    else:
        wp.atomic_add(fresh_count, 0, 1)


@wp.kernel
def _initialize_transition_component(
    moments_before: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    donor_counts: wp.array(dtype=wp.int32),
    moments_after: wp.array(dtype=float),
    conservation_delta: wp.array(dtype=float),
    component: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    donor_count = donor_counts[cell]
    if donor_count == 0:
        return

    was_fluid = previous_phi[i, j, k] >= 0.0
    is_fluid = current_phi[i, j, k] >= 0.0
    if was_fluid and not is_fluid:
        wp.atomic_add(
            conservation_delta,
            component,
            moments_before[component * stride + cell],
        )
    elif not was_fluid and is_fluid:
        value = float(0.0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di != 0 or dj != 0 or dk != 0:
                        ni = _wrap_neighbor(i + di, nx, periodic_x)
                        nj = _wrap_neighbor(j + dj, ny, periodic_y)
                        nk = _wrap_neighbor(k + dk, nz, periodic_z)
                        if ni >= 0 and nj >= 0 and nk >= 0:
                            if previous_phi[ni, nj, nk] >= 0.0 and current_phi[ni, nj, nk] >= 0.0:
                                neighbor = ni * ny * nz + nj * nz + nk
                                value += moments_before[component * stride + neighbor]
        value /= float(donor_count)
        moments_after[component * stride + cell] = value
        wp.atomic_sub(conservation_delta, component, value)


@wp.kernel
def _apply_global_conservation_correction(
    moments: wp.array(dtype=float),
    previous_phi: wp.array3d(dtype=float),
    current_phi: wp.array3d(dtype=float),
    conservation_delta: wp.array(dtype=float),
    persistent_fluid_count: wp.array(dtype=wp.int32),
    component: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if previous_phi[i, j, k] >= 0.0 and current_phi[i, j, k] >= 0.0:
        cell = i * ny * nz + j * nz + k
        correction = conservation_delta[component] / float(persistent_fluid_count[0])
        moments[component * stride + cell] += correction


@dataclass(frozen=True)
class CellTransitionDiagnostics:
    fresh_cell_count: int
    dead_cell_count: int
    unresolved_cell_count: int


class ConservativeCellRemapper:
    """Track the previous solid mask and conserve all ten moments across changes."""

    def __init__(self, state: HomeLbmState) -> None:
        self.res = state.res
        self.stride = state.cell_count
        self.device = state.device
        self.previous_phi = wp.empty_like(state.solid_phi)
        self._moments_before = wp.empty_like(state.moments)
        self._donor_counts = wp.zeros(self.stride, dtype=wp.int32, device=self.device)
        self._fresh_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._dead_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._unresolved_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._persistent_fluid_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._conservation_delta = wp.zeros(10, dtype=float, device=self.device)
        self.capture_geometry(state)

    def capture_geometry(self, state: HomeLbmState) -> None:
        self._validate_state(state)
        wp.copy(self.previous_phi, state.solid_phi)

    def remap(self, state: HomeLbmState) -> CellTransitionDiagnostics:
        self._validate_state(state)
        self._fresh_count.zero_()
        self._dead_count.zero_()
        self._unresolved_count.zero_()
        self._persistent_fluid_count.zero_()
        self._conservation_delta.zero_()
        wp.launch(
            _count_cell_transition_neighbors,
            dim=self.res,
            inputs=[
                self.previous_phi,
                state.solid_phi,
                self._donor_counts,
                self._fresh_count,
                self._dead_count,
                self._unresolved_count,
                self._persistent_fluid_count,
                int(state.model.periodic[0]),
                int(state.model.periodic[1]),
                int(state.model.periodic[2]),
                self.res[0],
                self.res[1],
                self.res[2],
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        diagnostics = CellTransitionDiagnostics(
            fresh_cell_count=int(self._fresh_count.numpy()[0]),
            dead_cell_count=int(self._dead_count.numpy()[0]),
            unresolved_cell_count=int(self._unresolved_count.numpy()[0]),
        )
        if diagnostics.unresolved_cell_count:
            raise RuntimeError(
                f"{diagnostics.unresolved_cell_count} transitioning cells have no persistent fluid neighbor"
            )
        if diagnostics.fresh_cell_count == 0 and diagnostics.dead_cell_count == 0:
            wp.copy(self.previous_phi, state.solid_phi)
            return diagnostics

        wp.copy(self._moments_before, state.moments)
        for component in range(10):
            wp.launch(
                _initialize_transition_component,
                dim=self.res,
                inputs=[
                    self._moments_before,
                    self.previous_phi,
                    state.solid_phi,
                    self._donor_counts,
                    state.moments,
                    self._conservation_delta,
                    component,
                    int(state.model.periodic[0]),
                    int(state.model.periodic[1]),
                    int(state.model.periodic[2]),
                    self.res[0],
                    self.res[1],
                    self.res[2],
                    self.stride,
                ],
                device=self.device,
            )
            wp.launch(
                _apply_global_conservation_correction,
                dim=self.res,
                inputs=[
                    state.moments,
                    self.previous_phi,
                    state.solid_phi,
                    self._conservation_delta,
                    self._persistent_fluid_count,
                    component,
                    self.res[1],
                    self.res[2],
                    self.stride,
                ],
                device=self.device,
            )
        wp.copy(self.previous_phi, state.solid_phi)
        return diagnostics

    def _validate_state(self, state: HomeLbmState) -> None:
        if state.res != self.res or state.device != self.device:
            raise ValueError("cell remapper and HOME state must share resolution and device")
