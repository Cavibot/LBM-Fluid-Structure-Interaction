# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Owner for deterministic HOME-FREE topology classification buffers."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..constants import D3Q27_DIRECTIONS
from . import kernels as mass_kernels
from ..model import HomeLbmModel
from ..state import HomeLbmState
from .state import HomeFreeState
from . import topology_kernels


class HomeFreeTopologyUpdater:
    def __init__(self, model: HomeLbmModel, fill_epsilon: float = 1.0e-4) -> None:
        if not 0.0 <= fill_epsilon < 0.5:
            raise ValueError("fill_epsilon must be in [0, 0.5)")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.device = model._device
        self.fill_epsilon = float(fill_epsilon)
        self.candidates = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.liquid_protected = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.transitions = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.resolved_flags = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.committed_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.committed_fill = wp.zeros(self.res, dtype=float, device=self.device)
        self.committed_excess_share = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.committed_excess_momentum_share = wp.zeros(
            3 * int(np.prod(self.res)), dtype=float, device=self.device
        )
        self.active_neighbor_count = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._invalid_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._new_interface_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_new_interface_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._classified_fluid_state: HomeLbmState | None = None
        self._classified_free_surface_state: HomeFreeState | None = None
        self._classified_advected_mass: wp.array | None = None

    def classify(
        self,
        fluid_state: HomeLbmState,
        free_surface_state: HomeFreeState,
        advected_mass: wp.array,
    ) -> tuple[wp.array, wp.array]:
        if fluid_state.res != self.res or free_surface_state.res != self.res:
            raise ValueError("HOME, HOME-FREE, and topology resolutions must match")
        if tuple(advected_mass.shape) != self.res or advected_mass.device != self.device:
            raise ValueError("advected_mass must match topology shape and device")
        periodic = tuple(int(value) for value in self.model.periodic)
        common = [*periodic, *self.res]
        self._invalid_cell_count.zero_()
        wp.launch(
            topology_kernels.mark_topology_candidates_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                advected_mass,
                free_surface_state.flags,
                self.candidates,
                self._invalid_cell_count,
                self.fill_epsilon,
                *common,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.protect_liquid_growth_kernel,
            dim=self.res,
            inputs=[
                free_surface_state.flags,
                self.candidates,
                self.liquid_protected,
                *common,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.protect_gas_growth_kernel,
            dim=self.res,
            inputs=[
                free_surface_state.flags,
                self.liquid_protected,
                self.transitions,
                *common,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.apply_topology_transitions_kernel,
            dim=self.res,
            inputs=[free_surface_state.flags, self.transitions, self.resolved_flags],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_cell_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                f"HOME-FREE topology classification found {invalid} invalid cells"
            )
        self._classified_fluid_state = fluid_state
        self._classified_free_surface_state = free_surface_state
        self._classified_advected_mass = advected_mass
        return self.transitions, self.resolved_flags

    def commit(
        self,
        fluid_state: HomeLbmState,
        free_surface_state: HomeFreeState,
        advected_mass: wp.array,
        destination: HomeFreeState,
        *,
        advected_momentum: wp.array | None = None,
    ) -> HomeFreeState:
        """Commit resolved fields using final-topology recipient counts.

        Supplying ``advected_momentum`` enables the exact joint transaction;
        omitting it preserves the validated scalar HOME-Free production path.
        """

        if (
            fluid_state is not self._classified_fluid_state
            or free_surface_state is not self._classified_free_surface_state
            or advected_mass is not self._classified_advected_mass
        ):
            raise RuntimeError("commit requires classify() on the same state objects and mass buffer")
        if destination.res != self.res:
            raise ValueError("destination resolution must match topology updater")
        if advected_momentum is not None and (
            tuple(advected_momentum.shape) != (3 * int(np.prod(self.res)),)
            or advected_momentum.device != self.device
        ):
            raise ValueError("advected_momentum must match topology shape and device")
        self._new_interface_count.zero_()
        self._invalid_new_interface_count.zero_()
        periodic = tuple(int(value) for value in self.model.periodic)
        wp.launch(
            topology_kernels.validate_new_interface_donors_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                free_surface_state.flags,
                self.transitions,
                self._new_interface_count,
                self._invalid_new_interface_count,
                *periodic,
                *self.res,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid_new = int(self._invalid_new_interface_count.numpy()[0])
        if invalid_new:
            raise RuntimeError(
                f"HOME-FREE cannot initialize {invalid_new} fresh interface cells "
                "from finite liquid/interface neighbors"
            )
        if int(self._new_interface_count.numpy()[0]):
            wp.launch(
                topology_kernels.initialize_new_interface_moments_kernel,
                dim=self.res,
                inputs=[
                    fluid_state.moments,
                    free_surface_state.flags,
                    self.transitions,
                    *periodic,
                    *self.res,
                    int(np.prod(self.res)),
                ],
                device=self.device,
            )
        self._invalid_cell_count.zero_()
        normalization_inputs = [fluid_state.moments, advected_mass]
        normalization_kernel = mass_kernels.normalize_mass_kernel
        if advected_momentum is not None:
            normalization_inputs.append(advected_momentum)
            normalization_kernel = mass_kernels.normalize_mass_momentum_kernel
        normalization_inputs.extend(
            [
                self.resolved_flags,
                self._directions,
                self.committed_mass,
                self.committed_fill,
                self.committed_excess_share,
                self.committed_excess_momentum_share,
                self.active_neighbor_count,
                self._invalid_cell_count,
                *periodic,
                *self.res,
                int(np.prod(self.res)),
            ]
        )
        wp.launch(
            normalization_kernel,
            dim=self.res,
            inputs=normalization_inputs,
            device=self.device,
        )
        wp.copy(destination.mass, self.committed_mass)
        wp.copy(destination.fill_level, self.committed_fill)
        wp.copy(destination.excess_mass, self.committed_excess_share)
        wp.copy(
            destination.excess_momentum,
            self.committed_excess_momentum_share,
        )
        wp.copy(destination.flags, self.resolved_flags)
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_cell_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                f"HOME-FREE topology commit found {invalid} invalid cells"
            )
        return destination
