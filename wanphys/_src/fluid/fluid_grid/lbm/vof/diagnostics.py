# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only P7 acceptance diagnostics for committed VOF states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..state import LbmStateBase


@dataclass(frozen=True)
class VofDiagnostics:
    """Closed-domain conservation, topology and admissibility ledger."""

    total_mass: float
    initial_mass: float
    boundary_mass_flux: float
    relative_mass_error: float
    phi_min: float
    phi_max: float
    invalid_liquid_gas_adjacency_count: int
    non_finite_count: int
    nonpositive_active_density_count: int
    max_velocity: float
    interface_cell_count: int
    epoch: int
    geometry_epoch: int
    pending_excess_total: float = 0.0
    max_abs_pending_excess: float = 0.0
    invalid_pending_excess_count: int = 0
    illegal_cell_type_count: int = 0
    out_of_range_phi_count: int = 0
    invalid_surface_pressure_count: int = 0


def _direction_pairs(
    values: np.ndarray,
    direction: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray]:
    left = values
    right = values
    for axis, (delta, wraps) in enumerate(zip(direction, periodic, strict=True)):
        if delta == 0:
            continue
        if wraps:
            right = np.roll(right, -delta, axis=axis)
            continue
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        if delta > 0:
            left_slice[axis] = slice(0, -delta)
            right_slice[axis] = slice(delta, None)
        else:
            left_slice[axis] = slice(-delta, None)
            right_slice[axis] = slice(0, delta)
        left = left[tuple(left_slice)]
        right = right[tuple(right_slice)]
    return left, right


def collect_vof_diagnostics(
    state: LbmStateBase,
    *,
    initial_mass: float | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> VofDiagnostics:
    """Collect a deterministic host ledger without mutating solver state."""

    if state.vof is None:
        raise ValueError("VOF diagnostics require authoritative state.vof")
    mass = np.asarray(state.vof.mass.numpy()).copy()
    phi = np.asarray(state.vof.phi.numpy()).copy()
    pending_excess = np.asarray(state.vof.pending_excess.numpy()).copy()
    pending_receiver_count = np.asarray(
        state.vof.pending_receiver_count.numpy()
    ).copy()
    cell_type = np.asarray(state.vof.cell_type.numpy()).copy()
    density = np.asarray(state.density.numpy()).copy()
    velocity = tuple(
        np.asarray(field.numpy()).copy()
        for field in (state.velocity_x, state.velocity_y, state.velocity_z)
    )
    curvature = np.asarray(state.vof.curvature.numpy()).copy()
    normal = np.asarray(state.vof.normal.numpy()).copy()
    offset = np.asarray(state.vof.plic_offset.numpy()).copy()

    pending_total = float(np.sum(pending_excess, dtype=np.float64))
    total_mass = float(np.sum(mass, dtype=np.float64)) + pending_total
    reference_mass = (
        float(state.vof.reference_mass)
        if initial_mass is None
        else float(initial_mass)
    )
    denominator = max(abs(reference_mass), 1.0e-30)
    relative_error = abs(total_mass - reference_mass) / denominator

    invalid_links = 0
    unique_d3q19_links = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
    )
    for direction in unique_d3q19_links:
        left, right = _direction_pairs(cell_type, direction, periodic)
        invalid_links += int(
            np.count_nonzero(
                ((left == 2) & (right == 0))
                | ((left == 0) & (right == 2))
            )
        )

    finite_fields = (
        mass,
        phi,
        pending_excess,
        density,
        curvature,
        normal,
        offset,
        *velocity,
    )
    non_finite = sum(
        int(values.size - np.count_nonzero(np.isfinite(values)))
        for values in finite_fields
    )
    active = cell_type != 0
    nonpositive_density = int(
        np.count_nonzero((~np.isfinite(density) | (density <= 0.0)) & active)
    )
    speed = np.sqrt(sum(component * component for component in velocity))
    invalid_pending = int(
        np.count_nonzero(
            (pending_receiver_count > 18)
            | (
                (np.abs(pending_excess) > 2.0e-6)
                & (pending_receiver_count == 0)
            )
        )
    )
    legal_type = np.isin(cell_type, np.array([0, 1, 2], dtype=np.uint8))
    illegal_type_count = int(cell_type.size - np.count_nonzero(legal_type))
    out_of_range_phi = int(
        np.count_nonzero((phi < -3.0e-6) | (phi > 1.0 + 3.0e-6))
    )
    interface = cell_type == 1
    rho_g = 3.0 * (
        float(state.model.vof_atmosphere_pressure)
        - 2.0
        * float(state.model.vof_surface_tension)
        * curvature[interface]
    )
    invalid_surface_pressure = int(
        np.count_nonzero((~np.isfinite(rho_g)) | (rho_g <= 0.0))
    )

    return VofDiagnostics(
        total_mass=total_mass,
        initial_mass=reference_mass,
        boundary_mass_flux=0.0,
        relative_mass_error=relative_error,
        phi_min=float(np.min(phi)),
        phi_max=float(np.max(phi)),
        invalid_liquid_gas_adjacency_count=invalid_links,
        non_finite_count=non_finite,
        nonpositive_active_density_count=nonpositive_density,
        max_velocity=float(np.max(speed)),
        interface_cell_count=int(np.count_nonzero(cell_type == 1)),
        epoch=int(state.vof.epoch),
        geometry_epoch=int(state.vof.geometry_epoch),
        pending_excess_total=pending_total,
        max_abs_pending_excess=float(np.max(np.abs(pending_excess))),
        invalid_pending_excess_count=invalid_pending,
        illegal_cell_type_count=illegal_type_count,
        out_of_range_phi_count=out_of_range_phi,
        invalid_surface_pressure_count=invalid_surface_pressure,
    )


def validate_vof_diagnostics(
    diagnostics: VofDiagnostics,
    *,
    mass_tolerance: float = 5.0e-6,
    phi_tolerance: float = 3.0e-6,
    max_lattice_speed: float = 0.4,
) -> None:
    """Apply the frozen P7 closed-domain acceptance thresholds."""

    if diagnostics.boundary_mass_flux != 0.0:
        raise ValueError("P7 closed-domain boundary mass flux must be zero")
    if diagnostics.relative_mass_error > mass_tolerance:
        raise ValueError(
            "P7 closed-domain relative mass error exceeds tolerance: "
            f"{diagnostics.relative_mass_error} > {mass_tolerance}"
        )
    if diagnostics.phi_min < -phi_tolerance or diagnostics.phi_max > (
        1.0 + phi_tolerance
    ):
        raise ValueError(
            "P7 committed phi lies outside [0, 1]: "
            f"[{diagnostics.phi_min}, {diagnostics.phi_max}]"
        )
    if diagnostics.invalid_pending_excess_count != 0:
        raise ValueError("FIX1 pending excess routing is invalid")
    if diagnostics.illegal_cell_type_count != 0:
        raise ValueError("FIX1 committed cell_type contains an unknown value")
    if diagnostics.out_of_range_phi_count != 0:
        raise ValueError("FIX1 committed phi has out-of-range cells")
    if diagnostics.invalid_surface_pressure_count != 0:
        raise ValueError("FIX1 interface surface pressure is invalid")
    if diagnostics.invalid_liquid_gas_adjacency_count != 0:
        raise ValueError("P7 committed topology contains LIQUID-GAS adjacency")
    if diagnostics.non_finite_count != 0:
        raise ValueError("P7 committed state contains NaN or Inf")
    if diagnostics.nonpositive_active_density_count != 0:
        raise ValueError("P7 active density must be finite and positive")
    if diagnostics.max_velocity > max_lattice_speed + 2.0e-6:
        raise ValueError("P7 max velocity exceeds the low-Mach contract")
    if diagnostics.geometry_epoch != diagnostics.epoch:
        raise ValueError("P7 committed geometry epoch is stale")


__all__ = [
    "VofDiagnostics",
    "collect_vof_diagnostics",
    "validate_vof_diagnostics",
]
