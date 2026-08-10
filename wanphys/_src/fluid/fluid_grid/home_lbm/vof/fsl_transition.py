# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fresh/dead HOME state transitions for moving FSL hydrodynamic masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..state import HomeLbmState


@dataclass(frozen=True)
class FslActiveTransitionDiagnostics:
    fresh_cell_count: int
    dead_cell_count: int
    unresolved_fresh_cell_count: int


@dataclass(frozen=True)
class FslActiveTransitionReference:
    moments: np.ndarray
    donor_count: np.ndarray
    diagnostics: FslActiveTransitionDiagnostics


def remap_fsl_active_moments(
    moments: np.ndarray,
    previous_active: np.ndarray,
    current_active: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> FslActiveTransitionReference:
    """Initialize newly hydrodynamic FSL nodes from persistent fluid donors."""

    values = np.asarray(moments, dtype=np.float64)
    previous = np.asarray(previous_active, dtype=bool)
    current = np.asarray(current_active, dtype=bool)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("FSL transition moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if previous.shape != shape or current.shape != shape:
        raise ValueError("FSL transition active masks must match the HOME grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    persistent = previous & current
    if not np.isfinite(values[persistent]).all() or np.any(
        values[..., 0][persistent] <= 0.0
    ):
        raise ValueError("persistent FSL donors must be finite with positive density")

    fresh = ~previous & current
    dead = previous & ~current
    donor_count = np.zeros(shape, dtype=np.int32)
    result = values.copy()
    unresolved = 0
    for index in map(tuple, np.argwhere(fresh)):
        donors: list[tuple[int, int, int]] = []
        for displacement in np.ndindex((3, 3, 3)):
            delta = tuple(component - 1 for component in displacement)
            if delta == (0, 0, 0):
                continue
            neighbor = [index[axis] + delta[axis] for axis in range(3)]
            valid = True
            for axis in range(3):
                if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                    if periodic[axis]:
                        neighbor[axis] %= shape[axis]
                    else:
                        valid = False
            if valid and persistent[tuple(neighbor)]:
                donors.append(tuple(neighbor))
        donor_count[index] = len(donors)
        if not donors:
            unresolved += 1
            continue
        donor_values = values[tuple(np.asarray(donors).T)]
        rho = float(np.mean(donor_values[:, 0]))
        velocity = np.mean(
            donor_values[:, 1:4] / donor_values[:, 0, None], axis=0
        )
        result[index] = (
            rho,
            rho * velocity[0],
            rho * velocity[1],
            rho * velocity[2],
            rho * velocity[0] * velocity[0],
            rho * velocity[1] * velocity[1],
            rho * velocity[2] * velocity[2],
            rho * velocity[0] * velocity[1],
            rho * velocity[0] * velocity[2],
            rho * velocity[1] * velocity[2],
        )
    diagnostics = FslActiveTransitionDiagnostics(
        fresh_cell_count=int(np.count_nonzero(fresh)),
        dead_cell_count=int(np.count_nonzero(dead)),
        unresolved_fresh_cell_count=unresolved,
    )
    if unresolved:
        raise RuntimeError(
            f"{unresolved} fresh FSL hydrodynamic cells have no persistent active donor"
        )
    return FslActiveTransitionReference(
        moments=result,
        donor_count=donor_count,
        diagnostics=diagnostics,
    )


@wp.func
def _fsl_transition_neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.kernel
def _classify_fsl_active_transitions_kernel(
    moments: wp.array(dtype=float),
    previous_active: wp.array3d(dtype=wp.int32),
    current_active: wp.array3d(dtype=wp.int32),
    donor_count: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
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
    was_active = previous_active[i, j, k] != 0
    is_active = current_active[i, j, k] != 0
    donor_count[cell] = 0
    if was_active and not is_active:
        wp.atomic_add(counts, 1, 1)
        return
    if was_active or not is_active:
        return

    wp.atomic_add(counts, 0, 1)
    donors = int(0)
    valid = bool(True)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _fsl_transition_neighbor(i + di, nx, periodic_x)
                nj = _fsl_transition_neighbor(j + dj, ny, periodic_y)
                nk = _fsl_transition_neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                if (
                    previous_active[ni, nj, nk] != 0
                    and current_active[ni, nj, nk] != 0
                ):
                    donor = ni * ny * nz + nj * nz + nk
                    rho = moments[donor]
                    valid = valid and wp.isfinite(rho) and rho > 0.0
                    for component in range(1, 10):
                        valid = valid and wp.isfinite(
                            moments[component * stride + donor]
                        )
                    donors += 1
    donor_count[cell] = donors
    if donors == 0 or not valid:
        wp.atomic_add(counts, 2, 1)


@wp.kernel
def _initialize_fresh_fsl_moments_kernel(
    moments_before: wp.array(dtype=float),
    previous_active: wp.array3d(dtype=wp.int32),
    current_active: wp.array3d(dtype=wp.int32),
    donor_count: wp.array(dtype=wp.int32),
    moments_after: wp.array(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if previous_active[i, j, k] != 0 or current_active[i, j, k] == 0:
        return
    cell = i * ny * nz + j * nz + k
    donors = donor_count[cell]
    rho_sum = float(0.0)
    velocity_sum = wp.vec3(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _fsl_transition_neighbor(i + di, nx, periodic_x)
                nj = _fsl_transition_neighbor(j + dj, ny, periodic_y)
                nk = _fsl_transition_neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                if (
                    previous_active[ni, nj, nk] != 0
                    and current_active[ni, nj, nk] != 0
                ):
                    donor = ni * ny * nz + nj * nz + nk
                    rho = moments_before[donor]
                    rho_sum += rho
                    velocity_sum += wp.vec3(
                        moments_before[stride + donor] / rho,
                        moments_before[2 * stride + donor] / rho,
                        moments_before[3 * stride + donor] / rho,
                    )
    inverse = 1.0 / float(donors)
    rho = rho_sum * inverse
    velocity = velocity_sum * inverse
    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]
    moments_after[cell] = rho
    moments_after[stride + cell] = rho * ux
    moments_after[2 * stride + cell] = rho * uy
    moments_after[3 * stride + cell] = rho * uz
    moments_after[4 * stride + cell] = rho * ux * ux
    moments_after[5 * stride + cell] = rho * uy * uy
    moments_after[6 * stride + cell] = rho * uz * uz
    moments_after[7 * stride + cell] = rho * ux * uy
    moments_after[8 * stride + cell] = rho * ux * uz
    moments_after[9 * stride + cell] = rho * uy * uz


class FslActiveTransitionRemapper:
    """Transactional device remapper for PLIC center-side active changes."""

    def __init__(self, state: HomeLbmState, active: wp.array) -> None:
        self.res = state.res
        self.stride = state.cell_count
        self.device = state.device
        self._validate_active(active)
        self.previous_active = wp.empty_like(active)
        self._moments_before = wp.empty_like(state.moments)
        self.donor_count = wp.zeros(
            self.stride, dtype=wp.int32, device=self.device
        )
        self._counts = wp.zeros(3, dtype=wp.int32, device=self.device)
        wp.copy(self.previous_active, active)

    def remap(
        self,
        state: HomeLbmState,
        current_active: wp.array,
    ) -> FslActiveTransitionDiagnostics:
        if state.res != self.res or state.device != self.device:
            raise ValueError("FSL active remapper and HOME state must match")
        self._validate_active(current_active)
        self._counts.zero_()
        wp.copy(self._moments_before, state.moments)
        wp.launch(
            _classify_fsl_active_transitions_kernel,
            dim=self.res,
            inputs=[
                self._moments_before,
                self.previous_active,
                current_active,
                self.donor_count,
                self._counts,
                int(state.model.periodic[0]),
                int(state.model.periodic[1]),
                int(state.model.periodic[2]),
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        diagnostics = FslActiveTransitionDiagnostics(
            fresh_cell_count=int(counts[0]),
            dead_cell_count=int(counts[1]),
            unresolved_fresh_cell_count=int(counts[2]),
        )
        if diagnostics.unresolved_fresh_cell_count:
            raise RuntimeError(
                f"{diagnostics.unresolved_fresh_cell_count} fresh FSL hydrodynamic "
                "cells have no persistent active donor"
            )
        if diagnostics.fresh_cell_count:
            wp.launch(
                _initialize_fresh_fsl_moments_kernel,
                dim=self.res,
                inputs=[
                    self._moments_before,
                    self.previous_active,
                    current_active,
                    self.donor_count,
                    state.moments,
                    int(state.model.periodic[0]),
                    int(state.model.periodic[1]),
                    int(state.model.periodic[2]),
                    *self.res,
                    self.stride,
                ],
                device=self.device,
            )
        wp.copy(self.previous_active, current_active)
        return diagnostics

    def _validate_active(self, active: wp.array) -> None:
        if active.shape != self.res or active.device != self.device:
            raise ValueError("FSL active mask must match remapper resolution and device")
        if active.dtype != wp.int32:
            raise TypeError("FSL active mask must use wp.int32")
