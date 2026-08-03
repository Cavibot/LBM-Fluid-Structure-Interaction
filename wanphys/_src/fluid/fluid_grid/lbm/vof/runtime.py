# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact device-side diagnostics for authoritative VOF runtime profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..encoding import (
    direction_x,
    direction_y,
    direction_z,
    opposite_direction,
)
from .diagnostics import VofDiagnostics

if TYPE_CHECKING:
    from ..state import LbmStateBase


@wp.func
def _wrap_once(value: int, size: int) -> int:
    mapped = value
    if mapped < 0:
        mapped += size
    elif mapped >= size:
        mapped -= size
    return mapped


@wp.kernel
def initialize_vof_device_metrics_kernel(
    metrics: wp.array(dtype=wp.float64),
    epoch: int,
    geometry_epoch: int,
    cell_count: int,
) -> None:
    """Initialize the single compact diagnostics structure on device."""

    if wp.tid() == 0:
        for index in range(33):
            metrics[index] = wp.float64(0.0)
        metrics[1] = wp.float64(1.0e30)
        metrics[2] = wp.float64(-1.0e30)
        metrics[14] = wp.float64(epoch)
        metrics[15] = wp.float64(geometry_epoch)
        metrics[18] = wp.float64(cell_count)
        metrics[19] = wp.float64(cell_count)


@wp.kernel
def reduce_vof_device_diagnostics_kernel(
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    pending_excess: wp.array3d(dtype=float),
    pending_receiver_count: wp.array3d(dtype=wp.uint8),
    cell_type: wp.array3d(dtype=wp.uint8),
    density: wp.array3d(dtype=float),
    velocity_x: wp.array3d(dtype=float),
    velocity_y: wp.array3d(dtype=float),
    velocity_z: wp.array3d(dtype=float),
    curvature: wp.array3d(dtype=float),
    normal: wp.array3d(dtype=wp.vec3),
    plic_offset: wp.array3d(dtype=float),
    metrics: wp.array(dtype=wp.float64),
    atmosphere_pressure: float,
    surface_tension: float,
    mass_tolerance: float,
    phi_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Reduce material invariants without copying full fields to the host."""

    i, j, k = wp.tid()
    local_mass = mass[i, j, k]
    local_phi = phi[i, j, k]
    local_pending = pending_excess[i, j, k]
    local_count = pending_receiver_count[i, j, k]
    local_type = cell_type[i, j, k]
    local_density = density[i, j, k]
    ux = velocity_x[i, j, k]
    uy = velocity_y[i, j, k]
    uz = velocity_z[i, j, k]
    local_curvature = curvature[i, j, k]
    local_normal = normal[i, j, k]
    local_offset = plic_offset[i, j, k]

    wp.atomic_add(
        metrics,
        0,
        wp.float64(local_mass) + wp.float64(local_pending),
    )
    wp.atomic_min(metrics, 1, wp.float64(local_phi))
    wp.atomic_max(metrics, 2, wp.float64(local_phi))
    speed_squared = ux * ux + uy * uy + uz * uz
    wp.atomic_max(metrics, 3, wp.float64(speed_squared))
    wp.atomic_max(metrics, 4, wp.float64(wp.abs(local_pending)))
    wp.atomic_add(metrics, 13, wp.float64(local_pending))

    invalid_finite = int(0)
    if not wp.isfinite(local_mass):
        invalid_finite += 1
    if not wp.isfinite(local_phi):
        invalid_finite += 1
    if not wp.isfinite(local_pending):
        invalid_finite += 1
    if not wp.isfinite(local_density):
        invalid_finite += 1
    if not wp.isfinite(ux):
        invalid_finite += 1
    if not wp.isfinite(uy):
        invalid_finite += 1
    if not wp.isfinite(uz):
        invalid_finite += 1
    if not wp.isfinite(local_curvature):
        invalid_finite += 1
    if not wp.isfinite(local_normal[0]):
        invalid_finite += 1
    if not wp.isfinite(local_normal[1]):
        invalid_finite += 1
    if not wp.isfinite(local_normal[2]):
        invalid_finite += 1
    if not wp.isfinite(local_offset):
        invalid_finite += 1
    if invalid_finite > 0:
        wp.atomic_add(metrics, 6, wp.float64(invalid_finite))

    legal_type = (
        local_type == wp.uint8(0)
        or local_type == wp.uint8(1)
        or local_type == wp.uint8(2)
    )
    if not legal_type:
        wp.atomic_add(metrics, 10, wp.float64(1.0))
    if local_type == wp.uint8(1):
        wp.atomic_add(metrics, 8, wp.float64(1.0))
    if local_type != wp.uint8(0) and (
        not wp.isfinite(local_density) or local_density <= 0.0
    ):
        wp.atomic_add(metrics, 7, wp.float64(1.0))
    if (
        local_phi < 0.0 - phi_tolerance
        or local_phi > 1.0 + phi_tolerance
    ):
        wp.atomic_add(metrics, 11, wp.float64(1.0))

    if local_type == wp.uint8(1):
        rho_g = 3.0 * (
            atmosphere_pressure
            - 2.0 * surface_tension * local_curvature
        )
        if not wp.isfinite(rho_g) or rho_g <= 0.0:
            wp.atomic_add(metrics, 12, wp.float64(1.0))

    actual_receiver_count = int(0)
    for q in range(1, 19):
        ni = i - direction_x(q)
        nj = j - direction_y(q)
        nk = k - direction_z(q)
        outside = bool(False)
        if ni < 0 or ni >= nx:
            if periodic_x != 0:
                ni = _wrap_once(ni, nx)
            else:
                outside = True
        if nj < 0 or nj >= ny:
            if periodic_y != 0:
                nj = _wrap_once(nj, ny)
            else:
                outside = True
        if nk < 0 or nk >= nz:
            if periodic_z != 0:
                nk = _wrap_once(nk, nz)
            else:
                outside = True
        if not outside:
            neighbor_type = cell_type[ni, nj, nk]
            if neighbor_type == wp.uint8(1):
                actual_receiver_count += 1
            if q < opposite_direction(q):
                invalid_pair = (
                    local_type == wp.uint8(0)
                    and neighbor_type == wp.uint8(2)
                ) or (
                    local_type == wp.uint8(2)
                    and neighbor_type == wp.uint8(0)
                )
                if invalid_pair:
                    wp.atomic_add(metrics, 5, wp.float64(1.0))

    material_pending = wp.abs(local_pending) > mass_tolerance
    linear_index = i * ny * nz + j * nz + k
    if material_pending and int(local_count) == 0 and actual_receiver_count == 0:
        wp.atomic_add(metrics, 16, wp.float64(1.0))
        wp.atomic_min(metrics, 18, wp.float64(linear_index))
    elif material_pending and int(local_count) != actual_receiver_count:
        wp.atomic_add(metrics, 9, wp.float64(1.0))
        wp.atomic_add(metrics, 17, wp.float64(1.0))
        wp.atomic_min(metrics, 19, wp.float64(linear_index))


@wp.kernel
def describe_first_invalid_pending_kernel(
    pending_excess: wp.array3d(dtype=float),
    pending_receiver_count: wp.array3d(dtype=wp.uint8),
    previous_cell_type: wp.array3d(dtype=wp.uint8),
    proposed_cell_type: wp.array3d(dtype=wp.uint8),
    final_cell_type: wp.array3d(dtype=wp.uint8),
    metrics: wp.array(dtype=wp.float64),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Describe the lowest-index invalid pending route after reduction."""

    if wp.tid() != 0:
        return
    linear_index = int(metrics[19])
    if metrics[16] > wp.float64(0.0):
        linear_index = int(metrics[18])
    if linear_index >= nx * ny * nz:
        return

    i = linear_index // (ny * nz)
    remainder = linear_index - i * ny * nz
    j = remainder // nz
    k = remainder - j * nz
    actual_receiver_count = int(0)
    valid_mask = int(0)
    previous_interface_mask = int(0)
    previous_liquid_mask = int(0)
    proposed_interface_mask = int(0)
    proposed_liquid_mask = int(0)
    final_interface_mask = int(0)
    final_liquid_mask = int(0)
    for q in range(1, 19):
        ni = i - direction_x(q)
        nj = j - direction_y(q)
        nk = k - direction_z(q)
        outside = bool(False)
        if ni < 0 or ni >= nx:
            if periodic_x != 0:
                ni = _wrap_once(ni, nx)
            else:
                outside = True
        if nj < 0 or nj >= ny:
            if periodic_y != 0:
                nj = _wrap_once(nj, ny)
            else:
                outside = True
        if nk < 0 or nk >= nz:
            if periodic_z != 0:
                nk = _wrap_once(nk, nz)
            else:
                outside = True
        if not outside:
            bit = int(1) << (q - 1)
            valid_mask = valid_mask | bit
            previous_neighbor = previous_cell_type[ni, nj, nk]
            proposed_neighbor = proposed_cell_type[ni, nj, nk]
            final_neighbor = final_cell_type[ni, nj, nk]
            if previous_neighbor == wp.uint8(1):
                previous_interface_mask = previous_interface_mask | bit
            elif previous_neighbor == wp.uint8(2):
                previous_liquid_mask = previous_liquid_mask | bit
            if proposed_neighbor == wp.uint8(1):
                proposed_interface_mask = proposed_interface_mask | bit
            elif proposed_neighbor == wp.uint8(2):
                proposed_liquid_mask = proposed_liquid_mask | bit
            if final_neighbor == wp.uint8(1):
                actual_receiver_count += 1
                final_interface_mask = final_interface_mask | bit
            elif final_neighbor == wp.uint8(2):
                final_liquid_mask = final_liquid_mask | bit

    metrics[20] = wp.float64(pending_excess[i, j, k])
    metrics[21] = wp.float64(pending_receiver_count[i, j, k])
    metrics[22] = wp.float64(actual_receiver_count)
    metrics[23] = wp.float64(final_cell_type[i, j, k])
    metrics[24] = wp.float64(previous_cell_type[i, j, k])
    metrics[25] = wp.float64(proposed_cell_type[i, j, k])
    metrics[26] = wp.float64(valid_mask)
    metrics[27] = wp.float64(previous_interface_mask)
    metrics[28] = wp.float64(previous_liquid_mask)
    metrics[29] = wp.float64(proposed_interface_mask)
    metrics[30] = wp.float64(proposed_liquid_mask)
    metrics[31] = wp.float64(final_interface_mask)
    metrics[32] = wp.float64(final_liquid_mask)


class VofDeviceDiagnostics:
    """Own one compact device reduction buffer and one host readback."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
        *,
        atmosphere_pressure: float,
        surface_tension: float,
        mass_tolerance: float = 2.0e-6,
        phi_tolerance: float = 3.0e-6,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)
        self.atmosphere_pressure = float(atmosphere_pressure)
        self.surface_tension = float(surface_tension)
        self.mass_tolerance = float(mass_tolerance)
        self.phi_tolerance = float(phi_tolerance)
        self._metrics = wp.zeros(33, dtype=wp.float64, device=device)

    @property
    def metrics(self) -> wp.array:
        """Return the fixed-size device diagnostics structure."""

        return self._metrics

    def collect(
        self,
        state: LbmStateBase,
        *,
        previous_cell_type: wp.array | None = None,
        proposed_cell_type: wp.array | None = None,
    ) -> VofDiagnostics:
        """Launch compact reductions and copy only 33 float64 values."""

        if state.vof is None:
            raise ValueError("device VOF diagnostics require state.vof")
        if tuple(int(value) for value in state.res) != self.shape:
            raise ValueError(
                f"device VOF diagnostics shape {state.res} != {self.shape}"
            )
        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        if previous_cell_type is None:
            previous_cell_type = state.vof.cell_type
        if proposed_cell_type is None:
            proposed_cell_type = state.vof.cell_type
        wp.launch(
            initialize_vof_device_metrics_kernel,
            dim=1,
            inputs=[
                self._metrics,
                int(state.vof.epoch),
                int(state.vof.geometry_epoch),
                nx * ny * nz,
            ],
            device=self.device,
        )
        wp.launch(
            reduce_vof_device_diagnostics_kernel,
            dim=self.shape,
            inputs=[
                state.vof.mass,
                state.vof.phi,
                state.vof.pending_excess,
                state.vof.pending_receiver_count,
                state.vof.cell_type,
                state.density,
                state.velocity_x,
                state.velocity_y,
                state.velocity_z,
                state.vof.curvature,
                state.vof.normal,
                state.vof.plic_offset,
                self._metrics,
                self.atmosphere_pressure,
                self.surface_tension,
                self.mass_tolerance,
                self.phi_tolerance,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        wp.launch(
            describe_first_invalid_pending_kernel,
            dim=1,
            inputs=[
                state.vof.pending_excess,
                state.vof.pending_receiver_count,
                previous_cell_type,
                proposed_cell_type,
                state.vof.cell_type,
                self._metrics,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        values = np.asarray(self._metrics.numpy(), dtype=np.float64)
        total_mass = float(values[0])
        reference_mass = float(state.vof.reference_mass)
        relative_error = abs(total_mass - reference_mass) / max(
            abs(reference_mass),
            1.0e-30,
        )
        first_linear = int(round(values[19]))
        if values[16] > 0.0:
            first_linear = int(round(values[18]))
        first_cell = None
        if first_linear < nx * ny * nz:
            first_cell = (
                first_linear // (ny * nz),
                (first_linear % (ny * nz)) // nz,
                first_linear % nz,
            )
        return VofDiagnostics(
            total_mass=total_mass,
            initial_mass=reference_mass,
            boundary_mass_flux=0.0,
            relative_mass_error=relative_error,
            phi_min=float(values[1]),
            phi_max=float(values[2]),
            invalid_liquid_gas_adjacency_count=int(round(values[5])),
            non_finite_count=int(round(values[6])),
            nonpositive_active_density_count=int(round(values[7])),
            max_velocity=float(np.sqrt(max(0.0, values[3]))),
            interface_cell_count=int(round(values[8])),
            epoch=int(round(values[14])),
            geometry_epoch=int(round(values[15])),
            pending_excess_total=float(values[13]),
            max_abs_pending_excess=float(values[4]),
            invalid_pending_excess_count=int(round(values[9])),
            pending_zero_receiver_count=int(round(values[16])),
            pending_receiver_mismatch_count=int(round(values[17])),
            first_invalid_pending_cell=first_cell,
            first_invalid_pending_excess=float(values[20]),
            first_invalid_pending_stored_receiver_count=int(round(values[21])),
            first_invalid_pending_actual_receiver_count=int(round(values[22])),
            first_invalid_pending_cell_type=int(round(values[23])),
            first_invalid_pending_previous_cell_type=int(round(values[24])),
            first_invalid_pending_proposed_cell_type=int(round(values[25])),
            first_invalid_pending_neighbor_valid_mask=int(round(values[26])),
            first_invalid_pending_previous_interface_mask=int(round(values[27])),
            first_invalid_pending_previous_liquid_mask=int(round(values[28])),
            first_invalid_pending_proposed_interface_mask=int(round(values[29])),
            first_invalid_pending_proposed_liquid_mask=int(round(values[30])),
            first_invalid_pending_final_interface_mask=int(round(values[31])),
            first_invalid_pending_final_liquid_mask=int(round(values[32])),
            illegal_cell_type_count=int(round(values[10])),
            out_of_range_phi_count=int(round(values[11])),
            invalid_surface_pressure_count=int(round(values[12])),
        )


__all__ = [
    "VofDeviceDiagnostics",
    "describe_first_invalid_pending_kernel",
    "initialize_vof_device_metrics_kernel",
    "reduce_vof_device_diagnostics_kernel",
]
