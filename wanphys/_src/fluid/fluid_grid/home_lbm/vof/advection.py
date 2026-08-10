# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Device owner for one conservative HOME-FREE mass-advection pass."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import kernels
from .state import HomeFreeState


class HomeFreeMassAdvector:
    """Evaluate link-wise liquid mass exchange without committing topology."""

    def __init__(self, model: HomeLbmModel) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.advected_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.incoming_excess_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.incoming_excess_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.advected_momentum: wp.array | None = None
        self.target_momentum: wp.array | None = None
        self._home_internal_link_momentum: wp.array | None = None
        self.reference_pressure_momentum: wp.array | None = None
        self._empty_momentum = wp.zeros(1, dtype=float, device=self.device)
        self.normalized_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.normalized_fill = wp.zeros(self.res, dtype=float, device=self.device)
        self.excess_share = wp.zeros(self.res, dtype=float, device=self.device)
        self.excess_momentum_share = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.active_neighbor_count = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=self.device)
        self._direct_liquid_gas_link_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_advected_momentum_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_target_momentum_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_normalized_cell_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_momentum_cell_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self.interface_home_density_increment: wp.array | None = None
        self.interface_home_momentum_increment: wp.array | None = None
        self.interface_momentum_candidate: wp.array | None = None
        self.internal_link_advected_momentum: wp.array | None = None
        self.home_internal_link_momentum: wp.array | None = None
        self.home_destination_density: wp.array | None = None
        self.home_destination_momentum: wp.array | None = None
        self._invalid_defect_diagnostic_count: wp.array | None = None
        self._advected_fluid_state: HomeLbmState | None = None
        self._advected_free_surface_state: HomeFreeState | None = None
        self._target_destination_fluid_state: HomeLbmState | None = None

    def enable_momentum_defect_diagnostics(self) -> None:
        """Allocate optional scratch used to measure the VOF/HOME defect."""

        if self.interface_home_density_increment is not None:
            return
        self.enable_conservative_momentum_transport()
        self.interface_home_density_increment = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.interface_home_momentum_increment = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.interface_momentum_candidate = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.internal_link_advected_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.home_destination_density = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.home_destination_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self._invalid_defect_diagnostic_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )

    def enable_conservative_momentum_transport(self) -> None:
        """Allocate exact link-momentum scratch for the experimental joint commit."""

        if self.advected_momentum is not None:
            return
        self.advected_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.target_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self._home_internal_link_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.reference_pressure_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.home_internal_link_momentum = self._home_internal_link_momentum

    def advect(
        self,
        fluid_state: HomeLbmState,
        free_surface_state: HomeFreeState,
        *,
        pressure_reference_density: float = 1.0,
    ) -> wp.array:
        if fluid_state.res != self.res or free_surface_state.res != self.res:
            raise ValueError("HOME, HOME-FREE, and advector resolutions must match")
        if (
            not np.isfinite(pressure_reference_density)
            or pressure_reference_density <= 0.0
        ):
            raise ValueError("pressure_reference_density must be finite and positive")
        self._direct_liquid_gas_link_count.zero_()
        self._invalid_advected_momentum_count.zero_()
        wp.launch(
            kernels.advect_mass_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                free_surface_state.mass,
                free_surface_state.fill_level,
                free_surface_state.excess_mass,
                free_surface_state.excess_momentum,
                free_surface_state.flags,
                self._directions,
                self._weights,
                self._opposites,
                self.advected_mass,
                self.incoming_excess_mass,
                self.incoming_excess_momentum,
                self.advected_momentum
                if self.advected_momentum is not None
                else self._empty_momentum,
                self._home_internal_link_momentum
                if self._home_internal_link_momentum is not None
                else self._empty_momentum,
                self.reference_pressure_momentum
                if self.reference_pressure_momentum is not None
                else self._empty_momentum,
                self._direct_liquid_gas_link_count,
                self._invalid_advected_momentum_count,
                int(self.advected_momentum is not None),
                float(pressure_reference_density),
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                self.res[0],
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        if self.interface_home_density_increment is not None:
            assert self.interface_home_momentum_increment is not None
            assert self.interface_momentum_candidate is not None
            assert self.internal_link_advected_momentum is not None
            assert self.home_internal_link_momentum is not None
            assert self._invalid_defect_diagnostic_count is not None
            self._invalid_defect_diagnostic_count.zero_()
            wp.launch(
                kernels.analyze_momentum_defect_kernel,
                dim=self.res,
                inputs=[
                    fluid_state.moments,
                    free_surface_state.mass,
                    free_surface_state.fill_level,
                    free_surface_state.flags,
                    self._directions,
                    self._weights,
                    self._opposites,
                    self.interface_home_density_increment,
                    self.interface_home_momentum_increment,
                    self.interface_momentum_candidate,
                    self.internal_link_advected_momentum,
                    self.home_internal_link_momentum,
                    self._invalid_defect_diagnostic_count,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    self.res[0],
                    self.res[1],
                    self.res[2],
                    self.stride,
                ],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        invalid_links = int(self._direct_liquid_gas_link_count.numpy()[0])
        invalid_momentum = int(
            self._invalid_advected_momentum_count.numpy()[0]
        )
        invalid_defect = (
            int(self._invalid_defect_diagnostic_count.numpy()[0])
            if self._invalid_defect_diagnostic_count is not None
            else 0
        )
        if invalid_links:
            raise RuntimeError(
                "HOME-FREE mass advection found "
                f"{invalid_links} direct liquid-gas links; interface topology is invalid"
            )
        if invalid_momentum:
            raise FloatingPointError(
                "HOME-FREE queue-momentum transport found "
                f"{invalid_momentum} invalid active cells or links"
            )
        if invalid_defect:
            raise FloatingPointError(
                "HOME-Free momentum-defect diagnostics found "
                f"{invalid_defect} invalid active source cells"
            )
        self._advected_fluid_state = fluid_state
        self._advected_free_surface_state = free_surface_state
        self._target_destination_fluid_state = None
        return self.advected_mass

    def finalize_momentum_target(
        self,
        destination_fluid_state: HomeLbmState,
        *,
        capillary_momentum_correction: wp.array | None = None,
    ) -> wp.array:
        """Combine exact link transport, queues, boundaries, and liquid body force."""

        if (
            self._advected_fluid_state is None
            or self._advected_free_surface_state is None
        ):
            raise RuntimeError("momentum finalization requires a preceding advect() call")
        if (
            self.advected_momentum is None
            or self.target_momentum is None
            or self._home_internal_link_momentum is None
            or self.reference_pressure_momentum is None
        ):
            raise RuntimeError(
                "momentum finalization requires "
                "enable_conservative_momentum_transport() before advect()"
            )
        if destination_fluid_state.res != self.res:
            raise ValueError("destination HOME state resolution must match the advector")
        if capillary_momentum_correction is not None and (
            tuple(capillary_momentum_correction.shape) != (3 * self.stride,)
            or capillary_momentum_correction.device != self.device
        ):
            raise ValueError(
                "capillary_momentum_correction must match the advector shape and device"
            )
        self._invalid_target_momentum_count.zero_()
        wp.launch(
            kernels.finalize_advected_momentum_kernel,
            dim=self.res,
            inputs=[
                destination_fluid_state.moments,
                self.advected_mass,
                self.advected_momentum,
                self.incoming_excess_momentum,
                self._home_internal_link_momentum,
                self.reference_pressure_momentum,
                capillary_momentum_correction
                if capillary_momentum_correction is not None
                else self._empty_momentum,
                int(capillary_momentum_correction is not None),
                self._advected_free_surface_state.flags,
                self.target_momentum,
                self._invalid_target_momentum_count,
                wp.vec3(*self.model.lattice_acceleration),
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_target_momentum_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                "HOME-FREE momentum target found "
                f"{invalid} invalid active cells"
            )
        self._target_destination_fluid_state = destination_fluid_state
        return self.target_momentum

    def apply_queue_momentum_correction(
        self,
        destination_fluid_state: HomeLbmState,
    ) -> None:
        """Replace consumed queue mass's local velocity with its donor momentum."""

        if (
            self._advected_fluid_state is None
            or self._advected_free_surface_state is None
        ):
            raise RuntimeError("queue correction requires a preceding advect() call")
        if destination_fluid_state.res != self.res:
            raise ValueError("destination HOME state resolution must match the advector")
        if self.home_destination_density is not None:
            assert self.home_destination_momentum is not None
            wp.launch(
                kernels.capture_density_momentum_kernel,
                dim=self.res,
                inputs=[
                    destination_fluid_state.moments,
                    self.home_destination_density,
                    self.home_destination_momentum,
                    self.res[1],
                    self.res[2],
                    self.stride,
                ],
                device=self.device,
            )
        self._invalid_momentum_cell_count.zero_()
        wp.launch(
            kernels.apply_queue_momentum_correction_kernel,
            dim=self.res,
            inputs=[
                destination_fluid_state.moments,
                self.advected_mass,
                self.incoming_excess_mass,
                self.incoming_excess_momentum,
                self._advected_free_surface_state.flags,
                self._invalid_momentum_cell_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_momentum_cell_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                "HOME-FREE queue-momentum correction failed for "
                f"{invalid} cells"
            )

    def normalize(
        self,
        fluid_state: HomeLbmState,
        free_surface_state: HomeFreeState,
    ) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """Normalize the last advection result into pre-topology scratch fields."""

        if (
            fluid_state is not self._advected_fluid_state
            or free_surface_state is not self._advected_free_surface_state
        ):
            raise RuntimeError("normalize requires advect() on the same state objects first")
        self._invalid_normalized_cell_count.zero_()
        wp.launch(
            kernels.normalize_mass_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                self.advected_mass,
                free_surface_state.flags,
                self._directions,
                self.normalized_mass,
                self.normalized_fill,
                self.excess_share,
                self.excess_momentum_share,
                self.active_neighbor_count,
                self._invalid_normalized_cell_count,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                self.res[0],
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid_cells = int(self._invalid_normalized_cell_count.numpy()[0])
        if invalid_cells:
            raise FloatingPointError(
                f"HOME-FREE mass normalization found {invalid_cells} invalid cells"
            )
        return (
            self.normalized_mass,
            self.normalized_fill,
            self.excess_share,
            self.active_neighbor_count,
        )
