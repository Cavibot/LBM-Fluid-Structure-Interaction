# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology contract for geometric PLIC volume transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from .flags import HomeFreeCellFlag
from ..model import HomeLbmModel
from ..state import HomeLbmState
from .state import HomeFreeState
from . import geometric_topology_kernels


ENDPOINT_MATCHING_ROUNDS = 16


@dataclass(frozen=True)
class GeometricPlicTopologyDiagnostics:
    gas_to_interface_count: int
    liquid_to_interface_count: int
    interface_to_gas_count: int
    interface_to_liquid_count: int
    snapped_cell_count: int
    snapped_volume_delta: float
    separation_closure_cell_count: int


@dataclass(frozen=True)
class GeometricPlicTopologyReference:
    moments: np.ndarray
    mass: np.ndarray
    fill_level: np.ndarray
    flags: np.ndarray
    donor_count: np.ndarray
    diagnostics: GeometricPlicTopologyDiagnostics


class HomeFreeGeometricTopologyResolver:
    """Transactional device owner for geometric PLIC topology changes."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        endpoint_tolerance: float = 4.0e-7,
    ) -> None:
        if not math.isfinite(endpoint_tolerance) or not 0.0 <= endpoint_tolerance < 0.5:
            raise ValueError("endpoint_tolerance must be finite and in [0, 0.5)")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.endpoint_tolerance = float(endpoint_tolerance)
        self._classified_flags = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._classified_fill = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_kind = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._closure_partner = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._closure_proposal = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._closure_owner = wp.zeros(
            self.stride, dtype=wp.int32, device=self.device
        )
        self._closure_reserved = wp.zeros(
            self.stride, dtype=wp.int32, device=self.device
        )
        self._closure_partner_volume_delta = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_base_fill = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_self_mass_delta = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_partner_mass_delta = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_incoming_volume = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_incoming_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._closure_mass_delta = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._direct_repair_partner = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._direct_repair_self_volume = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._direct_repair_self_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._direct_repair_partner_volume = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._direct_repair_partner_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._direct_repair_incoming_volume = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self._direct_repair_incoming_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.resolved_flags = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.normalized_fill = wp.zeros(self.res, dtype=float, device=self.device)
        self.mass = wp.zeros(self.res, dtype=float, device=self.device)
        self._empty_transported_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.donor_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.validation_error = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self.mass_validation_error = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._moments_before = wp.zeros(
            10 * self.stride, dtype=float, device=self.device
        )
        self._moments_after = wp.zeros_like(self._moments_before)
        self._counts = wp.zeros(14, dtype=wp.int32, device=self.device)
        self._snapped_volume_delta = wp.zeros(1, dtype=float, device=self.device)
        self.last_diagnostics: GeometricPlicTopologyDiagnostics | None = None

    def resolve(
        self,
        fluid: HomeLbmState,
        source: HomeFreeState,
        transported_fill: wp.array,
        destination: HomeFreeState,
        *,
        transported_mass: wp.array | None = None,
    ) -> GeometricPlicTopologyDiagnostics:
        """Validate and atomically commit one geometric topology update."""

        self._validate_state(fluid, source, destination)
        self._validate_array(transported_fill, self.res, wp.float32, "transported_fill")
        if transported_mass is not None:
            self._validate_array(
                transported_mass, self.res, wp.float32, "transported_mass"
            )
        self._counts.zero_()
        self._closure_mass_delta.zero_()
        self._snapped_volume_delta.zero_()
        wp.copy(self._moments_before, fluid.moments)
        wp.copy(self._moments_after, self._moments_before)
        # Independent geometric mass projects onto the admissible set
        # {0} U [E, 1-E] U {1}. The E/2 transition minimizes the conservative
        # volume correction: values nearer an endpoint change phase, while values
        # nearer the interface floor remain interface and are paired to E/1-E.
        # The legacy rho*C path retains its reference endpoint-snapping contract.
        transition_tolerance = (
            self.endpoint_tolerance
            if transported_mass is None
            else 0.5 * self.endpoint_tolerance
        )
        wp.launch(
            geometric_topology_kernels.classify_geometric_topology_kernel,
            dim=self.res,
            inputs=[
                source.flags,
                transported_fill,
                self._classified_flags,
                self._classified_fill,
                self.donor_count,
                self._counts,
                self._snapped_volume_delta,
                transition_tolerance,
            ],
            device=self.device,
        )
        wp.launch(
            geometric_topology_kernels.close_geometric_topology_separation_kernel,
            dim=self.res,
            inputs=[
                source.flags,
                transported_fill,
                self._classified_flags,
                self._classified_fill,
                self.resolved_flags,
                self.normalized_fill,
                self._closure_kind,
                self._counts,
                self._snapped_volume_delta,
                self.endpoint_tolerance,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
            ],
            device=self.device,
        )
        if transported_mass is not None:
            self._closure_incoming_volume.zero_()
            self._closure_incoming_mass.zero_()
            wp.launch(
                geometric_topology_kernels.initialize_conservative_geometric_closure_kernel,
                dim=self.res,
                inputs=[
                    transported_fill,
                    self._classified_flags,
                    self.normalized_fill,
                    self._closure_base_fill,
                    self._closure_kind,
                    self._closure_partner,
                    self._closure_proposal,
                    self._closure_reserved,
                    self._closure_partner_volume_delta,
                    self._closure_self_mass_delta,
                    self._closure_partner_mass_delta,
                    self._counts,
                    self.res[1],
                    self.res[2],
                ],
                device=self.device,
            )
            wp.launch(
                geometric_topology_kernels.prepare_regular_conservative_geometric_closure_kernel,
                dim=self.res,
                inputs=[
                    transported_fill,
                    transported_mass,
                    self._classified_flags,
                    self.normalized_fill,
                    self._closure_kind,
                    self._closure_partner,
                    self._closure_partner_volume_delta,
                    self._closure_self_mass_delta,
                    self._closure_partner_mass_delta,
                    self._counts,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_topology_kernels.gather_regular_conservative_geometric_closure_kernel,
                dim=self.res,
                inputs=[
                    self._closure_kind,
                    self._closure_partner,
                    self._closure_partner_volume_delta,
                    self._closure_partner_mass_delta,
                    self._closure_incoming_volume,
                    self._closure_incoming_mass,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                ],
                device=self.device,
            )
            for _ in range(ENDPOINT_MATCHING_ROUNDS):
                wp.launch(
                    geometric_topology_kernels.reset_endpoint_closure_proposals_kernel,
                    dim=self.res,
                    inputs=[
                        self._closure_owner,
                        self._closure_proposal,
                        self._closure_reserved,
                        self.res[1],
                        self.res[2],
                    ],
                    device=self.device,
                )
                wp.launch(
                    geometric_topology_kernels.propose_endpoint_conservative_geometric_closure_kernel,
                    dim=self.res,
                    inputs=[
                        transported_fill,
                        transported_mass,
                        self._classified_flags,
                        self.normalized_fill,
                        self._closure_kind,
                        self._closure_partner,
                        self._closure_proposal,
                        self._closure_owner,
                        self._closure_reserved,
                        self._closure_incoming_volume,
                        self._closure_incoming_mass,
                        self._closure_base_fill,
                        self._closure_partner_volume_delta,
                        self._closure_self_mass_delta,
                        self._closure_partner_mass_delta,
                        self.endpoint_tolerance,
                        int(self.model.periodic[0]),
                        int(self.model.periodic[1]),
                        int(self.model.periodic[2]),
                        *self.res,
                    ],
                    device=self.device,
                )
                wp.launch(
                    geometric_topology_kernels.accept_endpoint_conservative_geometric_closure_kernel,
                    dim=self.res,
                    inputs=[
                        self._closure_kind,
                        self._closure_proposal,
                        self._closure_owner,
                        self._closure_reserved,
                        self._closure_partner,
                        self._closure_partner_volume_delta,
                        self._closure_partner_mass_delta,
                        self._closure_incoming_volume,
                        self._closure_incoming_mass,
                        self.res[1],
                        self.res[2],
                    ],
                    device=self.device,
                )
            wp.launch(
                geometric_topology_kernels.count_unmatched_endpoint_closures_kernel,
                dim=self.res,
                inputs=[
                    self._closure_kind,
                    self._closure_partner,
                    self._counts,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_topology_kernels.apply_conservative_geometric_closure_kernel,
                dim=self.res,
                inputs=[
                    source.flags,
                    transported_fill,
                    self.resolved_flags,
                    self.normalized_fill,
                    self._closure_base_fill,
                    self._closure_self_mass_delta,
                    self._closure_incoming_volume,
                    self._closure_incoming_mass,
                    self._closure_mass_delta,
                    self._counts,
                    self._snapped_volume_delta,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_topology_kernels.prepare_direct_separation_repair_kernel,
                dim=self.res,
                inputs=[
                    source.flags,
                    transported_mass,
                    self.resolved_flags,
                    self.normalized_fill,
                    self._closure_mass_delta,
                    self._direct_repair_partner,
                    self._direct_repair_self_volume,
                    self._direct_repair_self_mass,
                    self._direct_repair_partner_volume,
                    self._direct_repair_partner_mass,
                    self.endpoint_tolerance,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_topology_kernels.gather_direct_separation_repair_kernel,
                dim=self.res,
                inputs=[
                    self._direct_repair_partner,
                    self._direct_repair_partner_volume,
                    self._direct_repair_partner_mass,
                    self._direct_repair_incoming_volume,
                    self._direct_repair_incoming_mass,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_topology_kernels.apply_direct_separation_repair_kernel,
                dim=self.res,
                inputs=[
                    source.flags,
                    transported_fill,
                    self.resolved_flags,
                    self.normalized_fill,
                    self._closure_mass_delta,
                    self._direct_repair_self_volume,
                    self._direct_repair_self_mass,
                    self._direct_repair_incoming_volume,
                    self._direct_repair_incoming_mass,
                    self._counts,
                    self._snapped_volume_delta,
                ],
                device=self.device,
            )
        wp.launch(
            geometric_topology_kernels.count_geometric_topology_donors_kernel,
            dim=self.res,
            inputs=[
                self._moments_before,
                source.flags,
                self.resolved_flags,
                self.donor_count,
                self._counts,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        self._raise_preinitialization_error(
            counts,
            source=source,
            transported_fill=transported_fill,
            transported_mass=transported_mass,
        )
        if counts[0]:
            wp.launch(
                geometric_topology_kernels.initialize_fresh_geometric_interface_kernel,
                dim=self.res,
                inputs=[
                    self._moments_before,
                    source.flags,
                    self.resolved_flags,
                    self.donor_count,
                    self._moments_after,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                    self.stride,
                ],
                device=self.device,
            )
        wp.launch(
            geometric_topology_kernels.validate_geometric_topology_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                self.resolved_flags,
                self.validation_error,
                self._counts,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        if counts[9]:
            invalid = np.argwhere(self.validation_error.numpy() & 1)
            raise RuntimeError(
                "resolved geometric topology has invalid HOME moments in "
                f"{int(counts[9])} cells; first indices {invalid[:8].tolist()}"
            )
        if counts[10]:
            examples = self._direct_link_examples(
                source.flags.numpy(),
                source.fill_level.numpy(),
                transported_fill.numpy(),
            )
            raise RuntimeError(
                "resolved geometric topology has direct liquid-gas links at "
                f"{int(counts[10])} cells; first pairs {examples}"
            )
        self.mass_validation_error.zero_()
        wp.launch(
            geometric_topology_kernels.build_geometric_topology_mass_kernel,
            dim=self.res,
            inputs=[
                self._moments_after,
                (
                    self._empty_transported_mass
                    if transported_mass is None
                    else transported_mass
                ),
                self._closure_mass_delta,
                self.resolved_flags,
                self.normalized_fill,
                self.mass,
                self.mass_validation_error,
                self._counts,
                int(transported_mass is not None),
                self.endpoint_tolerance,
                self.res[1],
                self.res[2],
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        if counts[12]:
            errors = self.mass_validation_error.numpy()
            invalid_active = int(np.count_nonzero(errors & 1))
            invalid_empty = int(np.count_nonzero(errors & 2))
            resolved_flags = self.resolved_flags.numpy()
            normalized_fill = self.normalized_fill.numpy()
            transported_mass_values = (
                self._empty_transported_mass
                if transported_mass is None
                else transported_mass
            ).numpy()
            examples = []
            for raw_index in np.argwhere(errors != 0)[:8]:
                index = tuple(int(value) for value in raw_index)
                examples.append(
                    (
                        index,
                        int(resolved_flags[index]),
                        float(normalized_fill[index]),
                        float(transported_mass_values[index]),
                        int(errors[index]),
                    )
                )
            raise RuntimeError(
                "resolved geometric topology found "
                f"{int(counts[12])} invalid transported-mass cells "
                f"(active={invalid_active}, empty={invalid_empty}); "
                f"first entries {examples}; active cells have nonpositive mass "
                "or empty cells carrying mass"
            )
        diagnostics = GeometricPlicTopologyDiagnostics(
            gas_to_interface_count=int(counts[0]),
            liquid_to_interface_count=int(counts[1]),
            interface_to_gas_count=int(counts[2]),
            interface_to_liquid_count=int(counts[3]),
            snapped_cell_count=int(counts[4]),
            snapped_volume_delta=float(self._snapped_volume_delta.numpy()[0]),
            separation_closure_cell_count=int(counts[11]),
        )
        wp.copy(fluid.moments, self._moments_after)
        wp.copy(destination.mass, self.mass)
        wp.copy(destination.fill_level, self.normalized_fill)
        destination.excess_mass.zero_()
        destination.excess_momentum.zero_()
        wp.copy(destination.flags, self.resolved_flags)
        self.last_diagnostics = diagnostics
        return diagnostics

    def _direct_link_examples(
        self,
        source_flags: np.ndarray,
        source_fill: np.ndarray,
        transported_fill: np.ndarray,
    ) -> list[tuple[object, ...]]:
        flags = self.resolved_flags.numpy()
        fill = self.normalized_fill.numpy()
        invalid = self.validation_error.numpy()
        examples: list[tuple[object, ...]] = []
        for raw_index in np.argwhere(invalid & 2):
            index = tuple(int(value) for value in raw_index)
            flag = int(flags[index])
            forbidden = int(
                HomeFreeCellFlag.LIQUID
                if flag == int(HomeFreeCellFlag.GAS)
                else HomeFreeCellFlag.GAS
            )
            for delta in np.ndindex(3, 3, 3):
                offset = tuple(value - 1 for value in delta)
                if offset == (0, 0, 0):
                    continue
                neighbor = []
                valid = True
                for axis in range(3):
                    coordinate = index[axis] + offset[axis]
                    if self.model.periodic[axis]:
                        coordinate %= self.res[axis]
                    elif coordinate < 0 or coordinate >= self.res[axis]:
                        valid = False
                        break
                    neighbor.append(coordinate)
                if not valid:
                    continue
                neighbor_index = tuple(neighbor)
                if int(flags[neighbor_index]) == forbidden:
                    examples.append(
                        (
                            index,
                            int(source_flags[index]),
                            float(source_fill[index]),
                            float(transported_fill[index]),
                            flag,
                            float(fill[index]),
                            neighbor_index,
                            int(source_flags[neighbor_index]),
                            float(source_fill[neighbor_index]),
                            float(transported_fill[neighbor_index]),
                            forbidden,
                            float(fill[neighbor_index]),
                        )
                    )
                    break
            if len(examples) == 8:
                break
        return examples

    def _conservative_closure_examples(
        self,
        source: HomeFreeState,
        transported_fill: wp.array,
        transported_mass: wp.array,
    ) -> list[tuple[object, ...]]:
        kind = self._closure_kind.numpy()
        partner = self._closure_partner.numpy()
        base = self._closure_base_fill.numpy()
        incoming_volume = self._closure_incoming_volume.numpy()
        after = base + incoming_volume
        unmatched = (kind != 0) & (partner < 0)
        changed = (base != self.normalized_fill.numpy()) | (incoming_volume != 0.0)
        invalid_after = changed & (~np.isfinite(after) | (after <= 0.0) | (after >= 1.0))
        indices = np.argwhere(unmatched | invalid_after)
        source_flags = source.flags.numpy()
        source_fill = source.fill_level.numpy()
        transported_fill_values = transported_fill.numpy()
        transported_mass_values = transported_mass.numpy()
        classified_flags = self._classified_flags.numpy()
        classified_fill = self._classified_fill.numpy()
        normalized_fill = self.normalized_fill.numpy()
        incoming_mass = self._closure_incoming_mass.numpy()
        self_mass_delta = self._closure_self_mass_delta.numpy()
        examples: list[tuple[object, ...]] = []
        for raw_index in indices[:8]:
            index = tuple(int(value) for value in raw_index)
            examples.append(
                (
                    index,
                    int(kind[index]),
                    int(partner[index]),
                    int(source_flags[index]),
                    float(source_fill[index]),
                    float(transported_fill_values[index]),
                    float(transported_mass_values[index]),
                    int(classified_flags[index]),
                    float(classified_fill[index]),
                    float(normalized_fill[index]),
                    float(base[index]),
                    float(incoming_volume[index]),
                    float(incoming_mass[index]),
                    float(self_mass_delta[index]),
                    float(after[index]),
                )
            )
        return examples

    def _raise_preinitialization_error(
        self,
        counts: np.ndarray,
        *,
        source: HomeFreeState,
        transported_fill: wp.array,
        transported_mass: wp.array | None,
    ) -> None:
        if counts[5]:
            raise ValueError(
                f"geometric topology has {int(counts[5])} invalid source/fill cells"
            )
        if counts[6]:
            raise RuntimeError(
                f"geometric PLIC transport placed liquid volume in {int(counts[6])} solid cells"
            )
        if counts[7]:
            raise RuntimeError(
                f"geometric topology found {int(counts[7])} forbidden one-step phase jumps"
            )
        if counts[8]:
            raise RuntimeError(
                f"geometric topology found {int(counts[8])} invalid or unresolved fluid donors"
            )
        if counts[13]:
            examples = (
                []
                if transported_mass is None
                else self._conservative_closure_examples(
                    source, transported_fill, transported_mass
                )
            )
            raise RuntimeError(
                "geometric topology found "
                f"{int(counts[13])} invalid conservative separation-closure pairs; "
                "entries are (index, kind, partner, source_flag, source_fill, "
                "transported_fill, transported_mass, classified_flag, "
                "classified_fill, normalized_fill, base_fill, incoming_volume, "
                f"incoming_mass, self_mass_delta, after_fill): {examples}"
            )

    def _validate_state(
        self,
        fluid: HomeLbmState,
        source: HomeFreeState,
        destination: HomeFreeState,
    ) -> None:
        for state, name in ((fluid, "fluid"), (source, "source"), (destination, "destination")):
            if state.res != self.res or state.device != self.device:
                raise ValueError(f"{name} state must match geometric topology model")

    def _validate_array(
        self,
        array: wp.array,
        shape: tuple[int, int, int],
        dtype: object,
        name: str,
    ) -> None:
        if tuple(array.shape) != shape or array.device != self.device:
            raise ValueError(f"{name} must match geometric topology shape and device")
        if array.dtype != dtype:
            raise TypeError(f"{name} has the wrong Warp dtype")


def resolve_geometric_plic_topology(
    moments: np.ndarray,
    source_flags: np.ndarray,
    transported_fill: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    endpoint_tolerance: float = 4.0e-7,
) -> GeometricPlicTopologyReference:
    """Resolve geometric fill into a sharp topology without mass heuristics."""

    values = np.asarray(moments, dtype=np.float64)
    source = np.asarray(source_flags, dtype=np.int32)
    fill = np.asarray(transported_fill, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("geometric topology moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if source.shape != shape or fill.shape != shape:
        raise ValueError("geometric topology fields must share one grid shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not math.isfinite(endpoint_tolerance) or not 0.0 <= endpoint_tolerance < 0.5:
        raise ValueError("endpoint_tolerance must be finite and in [0, 0.5)")
    if np.any(source < int(HomeFreeCellFlag.GAS)) or np.any(
        source > int(HomeFreeCellFlag.SOLID)
    ):
        raise ValueError("geometric topology source flags are invalid")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("geometric topology fill must be finite and in [0, 1]")

    solid = source == int(HomeFreeCellFlag.SOLID)
    if np.any(fill[solid] > endpoint_tolerance):
        raise RuntimeError("geometric PLIC transport placed liquid volume in solid cells")
    normalized_fill = fill.copy()
    snap_to_gas = ~solid & (normalized_fill <= endpoint_tolerance)
    snap_to_liquid = ~solid & (normalized_fill >= 1.0 - endpoint_tolerance)
    normalized_fill[solid | snap_to_gas] = 0.0
    normalized_fill[snap_to_liquid] = 1.0
    target = np.full(shape, int(HomeFreeCellFlag.INTERFACE), dtype=np.int32)
    target[solid] = int(HomeFreeCellFlag.SOLID)
    target[snap_to_gas] = int(HomeFreeCellFlag.GAS)
    target[snap_to_liquid] = int(HomeFreeCellFlag.LIQUID)

    def neighbor_index(
        index: tuple[int, int, int], delta: tuple[int, int, int]
    ) -> tuple[int, int, int] | None:
        neighbor = [index[axis] + delta[axis] for axis in range(3)]
        for axis in range(3):
            if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                if periodic[axis]:
                    neighbor[axis] %= shape[axis]
                else:
                    return None
        return tuple(neighbor)

    classified_target = target.copy()
    closure = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        source_flag = int(source[index])
        target_flag = int(classified_target[index])
        if target_flag == int(HomeFreeCellFlag.GAS) and source_flag != int(
            HomeFreeCellFlag.SOLID
        ):
            forbidden = int(HomeFreeCellFlag.LIQUID)
            restored_fill = max(float(fill[index]), endpoint_tolerance)
        elif target_flag == int(HomeFreeCellFlag.LIQUID):
            forbidden = int(HomeFreeCellFlag.GAS)
            restored_fill = min(float(fill[index]), 1.0 - endpoint_tolerance)
        else:
            continue
        for displacement in np.ndindex((3, 3, 3)):
            delta = tuple(component - 1 for component in displacement)
            if delta == (0, 0, 0):
                continue
            neighbor = neighbor_index(index, delta)
            if neighbor is not None and classified_target[neighbor] == forbidden:
                neighbor_source = int(source[neighbor])
                restore_self = source_flag == int(HomeFreeCellFlag.INTERFACE) and (
                    (target_flag == int(HomeFreeCellFlag.GAS) and fill[index] > 0.0)
                    or (
                        target_flag == int(HomeFreeCellFlag.LIQUID)
                        and fill[index] < 1.0
                    )
                )
                close_transition = (
                    target_flag == int(HomeFreeCellFlag.GAS)
                    and source_flag == int(HomeFreeCellFlag.GAS)
                    and neighbor_source == int(HomeFreeCellFlag.INTERFACE)
                    and fill[neighbor] == 1.0
                ) or (
                    target_flag == int(HomeFreeCellFlag.LIQUID)
                    and source_flag == int(HomeFreeCellFlag.LIQUID)
                    and neighbor_source == int(HomeFreeCellFlag.INTERFACE)
                    and fill[neighbor] == 0.0
                )
                if not restore_self and not close_transition:
                    continue
                target[index] = int(HomeFreeCellFlag.INTERFACE)
                normalized_fill[index] = restored_fill
                closure[index] = True
                break

    source_gas = source == int(HomeFreeCellFlag.GAS)
    source_interface = source == int(HomeFreeCellFlag.INTERFACE)
    source_liquid = source == int(HomeFreeCellFlag.LIQUID)
    target_gas = target == int(HomeFreeCellFlag.GAS)
    target_interface = target == int(HomeFreeCellFlag.INTERFACE)
    target_liquid = target == int(HomeFreeCellFlag.LIQUID)
    if np.any(source_gas & target_liquid):
        raise RuntimeError("geometric topology forbids a one-step gas-to-liquid jump")
    if np.any(source_liquid & target_gas):
        raise RuntimeError("geometric topology forbids a one-step liquid-to-gas jump")

    persistent_donor = (source_interface | source_liquid) & (
        target_interface | target_liquid
    )
    if not np.isfinite(values[persistent_donor]).all() or np.any(
        values[..., 0][persistent_donor] <= 0.0
    ):
        raise ValueError("persistent geometric topology donors must be finite")
    fresh_interface = source_gas & target_interface
    result_moments = values.copy()
    donor_count = np.zeros(shape, dtype=np.int32)

    for index in map(tuple, np.argwhere(fresh_interface)):
        donors: list[tuple[int, int, int]] = []
        for displacement in np.ndindex((3, 3, 3)):
            delta = tuple(component - 1 for component in displacement)
            if delta == (0, 0, 0):
                continue
            neighbor = neighbor_index(index, delta)
            if neighbor is not None and persistent_donor[neighbor]:
                donors.append(neighbor)
        donor_count[index] = len(donors)
        if not donors:
            raise RuntimeError(
                f"fresh geometric interface cell {index} has no persistent fluid donor"
            )
        donor_values = values[tuple(np.asarray(donors).T)]
        rho = float(np.mean(donor_values[:, 0]))
        velocity = np.mean(
            donor_values[:, 1:4] / donor_values[:, 0, None], axis=0
        )
        result_moments[index] = (
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

    active_target = target_interface | target_liquid
    if not np.isfinite(result_moments[active_target]).all() or np.any(
        result_moments[..., 0][active_target] <= 0.0
    ):
        raise RuntimeError("resolved geometric fluid cells have invalid HOME moments")
    for index in np.ndindex(shape):
        if target[index] not in (
            int(HomeFreeCellFlag.GAS),
            int(HomeFreeCellFlag.LIQUID),
        ):
            continue
        forbidden = (
            int(HomeFreeCellFlag.LIQUID)
            if target[index] == int(HomeFreeCellFlag.GAS)
            else int(HomeFreeCellFlag.GAS)
        )
        for displacement in np.ndindex((3, 3, 3)):
            delta = tuple(component - 1 for component in displacement)
            if delta == (0, 0, 0):
                continue
            neighbor = neighbor_index(index, delta)
            if neighbor is not None and target[neighbor] == forbidden:
                raise RuntimeError("resolved geometric topology contains a direct liquid-gas link")

    mass = np.zeros(shape, dtype=np.float64)
    mass[active_target] = (
        result_moments[..., 0][active_target] * normalized_fill[active_target]
    )
    snapped = solid | snap_to_gas | snap_to_liquid
    changed_by_snap = snapped & (normalized_fill != fill)
    diagnostics = GeometricPlicTopologyDiagnostics(
        gas_to_interface_count=int(np.count_nonzero(fresh_interface)),
        liquid_to_interface_count=int(np.count_nonzero(source_liquid & target_interface)),
        interface_to_gas_count=int(np.count_nonzero(source_interface & target_gas)),
        interface_to_liquid_count=int(
            np.count_nonzero(source_interface & target_liquid)
        ),
        snapped_cell_count=int(np.count_nonzero(changed_by_snap)),
        snapped_volume_delta=float(
            np.sum(normalized_fill - fill, dtype=np.float64)
        ),
        separation_closure_cell_count=int(np.count_nonzero(closure)),
    )
    return GeometricPlicTopologyReference(
        moments=result_moments,
        mass=mass,
        fill_level=normalized_fill,
        flags=target,
        donor_count=donor_count,
        diagnostics=diagnostics,
    )
