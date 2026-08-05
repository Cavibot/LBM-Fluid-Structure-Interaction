"""Device diagnostics orchestration entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .kernels.reduction import (
    describe_first_invalid_pending_kernel,
    initialize_vof_device_metrics_kernel,
    reduce_vof_device_diagnostics_kernel,
)
from .host import VofDiagnostics

if TYPE_CHECKING:
    from ...state import LbmStateBase

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


__all__ = ["VofDeviceDiagnostics"]
