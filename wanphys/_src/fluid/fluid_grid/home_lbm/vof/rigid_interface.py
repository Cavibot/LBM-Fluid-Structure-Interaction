# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-body load exchange between HOME-Free and rigid geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import warp as wp

from ..constants import D3Q27_DIRECTIONS, D3Q27_WEIGHTS
from ..cut_link import CutLinkBuffer
from ..diagnostics import HomeLbmDiagnostics
from ..rigid_geometry import HomeLbmRigidGeometry, RigidGeometryUpdateDiagnostics
from ..wrench import FluidWrenchBuffer
from .domain import HomeFreeDomain
from .rigid_transition import (
    HomeFreeRigidTransitionDiagnostics,
    HomeFreeRigidTransitionRemapper,
)

if TYPE_CHECKING:
    from wanphys._src.rigid.domain import RigidDomain


@wp.kernel
def _classify_phase_owned_links(
    link_cell: wp.array(dtype=wp.int32),
    body_id: wp.array(dtype=wp.int32),
    flags: wp.array3d(dtype=wp.int32),
    impulse: wp.array(dtype=wp.vec3),
    wet_link_count_by_body: wp.array(dtype=wp.int32),
    dry_link_count_by_body: wp.array(dtype=wp.int32),
    invalid_flag_count: wp.array(dtype=wp.int32),
    dry_nonzero_impulse_count: wp.array(dtype=wp.int32),
    body_count: int,
    ny: int,
    nz: int,
):
    link = wp.tid()
    cell = link_cell[link]
    i = cell // (ny * nz)
    remainder = cell - i * ny * nz
    j = remainder // nz
    k = remainder - j * nz
    flag = flags[i, j, k]
    body = body_id[link]
    if body < 0 or body >= body_count:
        wp.atomic_add(invalid_flag_count, 0, 1)
        return
    if flag == 1 or flag == 2:
        wp.atomic_add(wet_link_count_by_body, body, 1)
    elif flag == 0:
        value = impulse[link]
        if value[0] != 0.0 or value[1] != 0.0 or value[2] != 0.0:
            wp.atomic_add(dry_nonzero_impulse_count, 0, 1)
        wp.atomic_add(dry_link_count_by_body, body, 1)
    else:
        wp.atomic_add(invalid_flag_count, 0, 1)


@wp.kernel
def _remove_ambient_pressure_from_wet_links(
    body_id: wp.array(dtype=wp.int32),
    link_cell: wp.array(dtype=wp.int32),
    link_direction: wp.array(dtype=wp.int32),
    flags: wp.array3d(dtype=wp.int32),
    directions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    impulse: wp.array(dtype=wp.vec3),
    ambient_pressure_correction: wp.array(dtype=wp.float64),
    ambient_density: float,
    ny: int,
    nz: int,
):
    link = wp.tid()
    cell = link_cell[link]
    i = cell // (ny * nz)
    remainder = cell - i * ny * nz
    j = remainder // nz
    k = remainder - j * nz
    flag = flags[i, j, k]
    if flag == 1 or flag == 2:
        direction = link_direction[link]
        correction = (
            2.0 * weights[direction] * ambient_density * directions[direction]
        )
        impulse[link] += correction
        wp.atomic_add(
            ambient_pressure_correction,
            0,
            wp.float64(correction[0]),
        )
        wp.atomic_add(
            ambient_pressure_correction,
            1,
            wp.float64(correction[1]),
        )
        wp.atomic_add(
            ambient_pressure_correction,
            2,
            wp.float64(correction[2]),
        )


@dataclass(frozen=True)
class HomeFreeRigidLoadDiagnostics:
    """Phase ownership of cut-link impulses in one frozen-body step."""

    wet_link_count: int
    dry_link_count: int
    invalid_flag_count: int
    dry_nonzero_impulse_count: int
    ambient_pressure_correction_lattice: tuple[float, float, float]


@dataclass(frozen=True)
class HomeFreeRigidStaticStepDiagnostics:
    """Fluid and load diagnostics for one frozen-body HOME-Free step."""

    fluid: HomeLbmDiagnostics
    load: HomeFreeRigidLoadDiagnostics


@dataclass(frozen=True)
class HomeFreeRigidPrepareDiagnostics:
    """Geometry and coupled VOF transition diagnostics for a sampled pose."""

    geometry: RigidGeometryUpdateDiagnostics
    transitions: HomeFreeRigidTransitionDiagnostics


@dataclass(frozen=True)
class HomeFreeRigidDynamicStepDiagnostics:
    """One prepared moving-boundary HOME-Free fluid step."""

    geometry: RigidGeometryUpdateDiagnostics
    transitions: HomeFreeRigidTransitionDiagnostics
    fluid: HomeLbmDiagnostics
    load: HomeFreeRigidLoadDiagnostics


class HomeFreeRigidInterface:
    """Install frozen rigid cut links and collect gauge-pressure loads.

    This class deliberately rejects body motion. Moving solids require a
    coupled remap of HOME moments, liquid mass, fill, queued excess, and flags;
    applying the single-phase moment remapper would violate HOME-Free state.
    """

    def __init__(
        self,
        fluid_domain: HomeFreeDomain,
        rigid_domain: RigidDomain,
        body_ids: Iterable[int] | None = None,
    ) -> None:
        if fluid_domain.model._device != rigid_domain.model.device:
            raise ValueError("HOME-Free and rigid domains must use the same device")
        if rigid_domain._state_in is None:
            rigid_domain.create_state()
        state = fluid_domain.state
        free = state.free_surface
        if (
            np.any(free.flags.numpy() != 0)
            or np.any(free.fill_level.numpy() != 0.0)
            or np.any(free.mass.numpy() != 0.0)
            or np.any(free.excess_mass.numpy() != 0.0)
            or np.any(free.excess_momentum.numpy() != 0.0)
        ):
            raise RuntimeError(
                "construct HomeFreeRigidInterface before initializing the free surface"
            )

        self.fluid_domain = fluid_domain
        self.rigid_domain = rigid_domain
        self.device = fluid_domain.model._device
        self.body_com_local = rigid_domain.model.body_com
        self.geometry = HomeLbmRigidGeometry.from_domain(rigid_domain, body_ids=body_ids)
        self.geometry.update_radius_bounds(self.body_com_local)
        self.wrench = FluidWrenchBuffer(rigid_domain.model.body_count, device=self.device)
        rigid_state = rigid_domain.state
        self.last_geometry_update: RigidGeometryUpdateDiagnostics = self.geometry.rasterize(
            state.fluid, rigid_state.body_q
        )
        self.cut_links = CutLinkBuffer.build_from_sdf(state.fluid)
        self.cut_links.set_rigid_wall_velocity(
            rigid_state.body_q,
            rigid_state.body_qd,
            self.body_com_local,
            fluid_domain.model.fluid_grid_cell_size,
            fluid_domain.model.time_step,
        )
        self.cut_links.validate_body_ids()
        fluid_domain.solver.set_cut_links(self.cut_links)

        self._initial_body_q = rigid_state.body_q.numpy().copy()
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._wet_link_count_by_body = wp.zeros(
            rigid_domain.model.body_count, dtype=wp.int32, device=self.device
        )
        self._dry_link_count_by_body = wp.zeros(
            rigid_domain.model.body_count, dtype=wp.int32, device=self.device
        )
        self._invalid_flag_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._dry_nonzero_impulse_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._ambient_pressure_correction = wp.zeros(
            3, dtype=wp.float64, device=self.device
        )
        self._load_flags = wp.empty_like(state.free_surface.flags)
        self._impulses_corrected = False
        self._remapper: HomeFreeRigidTransitionRemapper | None = None
        self._prepared: HomeFreeRigidPrepareDiagnostics | None = None

    def step_static(self, dt: float | None = None) -> HomeFreeRigidStaticStepDiagnostics:
        """Advance HOME-Free once while proving that the rigid pose is frozen."""

        rigid_state = self.rigid_domain.state
        if not np.array_equal(rigid_state.body_q.numpy(), self._initial_body_q):
            raise RuntimeError(
                "moving HOME-Free solids require the coupled VOF transition remapper"
            )
        velocity = rigid_state.body_qd.numpy()
        if np.any(velocity != 0.0):
            raise RuntimeError(
                "moving HOME-Free solids require the coupled VOF transition remapper"
            )
        fluid_dt = self.fluid_domain.model.time_step if dt is None else float(dt)
        geometric_stepper = getattr(self.fluid_domain, "geometric_stepper", None)
        if geometric_stepper is None:
            wp.copy(self._load_flags, self.fluid_domain.free_surface_state.flags)
        self.fluid_domain.step(fluid_dt)
        if geometric_stepper is not None:
            wp.copy(self._load_flags, geometric_stepper.working_free_surface.flags)
        self._impulses_corrected = False
        load = self._correct_ambient_pressure()
        self.wrench.accumulate_from_rigid(
            self.cut_links,
            rigid_state.body_q,
            self.body_com_local,
            self.fluid_domain.model.scaling,
        )
        assert self.fluid_domain.last_diagnostics is not None
        return HomeFreeRigidStaticStepDiagnostics(
            fluid=self.fluid_domain.last_diagnostics,
            load=load,
        )

    def prepare_step(
        self,
        body_q_sample: wp.array,
        body_qd_sample: wp.array,
    ) -> HomeFreeRigidPrepareDiagnostics:
        """Rasterize a sampled moving pose and transactionally remap HOME-Free."""

        if self._prepared is not None:
            raise RuntimeError("prepared HOME-Free rigid step has not been advanced")
        state = self.fluid_domain.state
        if self._remapper is None:
            self._remapper = HomeFreeRigidTransitionRemapper(
                self.fluid_domain.model,
                state.fluid,
                state.free_surface,
                allow_independent_mass=bool(
                    getattr(
                        self.fluid_domain,
                        "uses_independent_geometric_mass",
                        False,
                    )
                ),
            )
        geometry = self.geometry.rasterize_incremental(state.fluid, body_q_sample)
        transitions = self._remapper.remap(state.fluid, state.free_surface)
        reuse_topology = (
            transitions.fresh_cell_count == 0 and transitions.dead_cell_count == 0
        )
        if reuse_topology:
            self.cut_links.update_from_sdf(state.fluid)
        else:
            self.cut_links = CutLinkBuffer.build_from_sdf(state.fluid)
        self.cut_links.set_rigid_wall_velocity(
            body_q_sample,
            body_qd_sample,
            self.body_com_local,
            self.fluid_domain.model.fluid_grid_cell_size,
            self.fluid_domain.model.time_step,
        )
        self.cut_links.validate_body_ids()
        self.fluid_domain.solver.set_cut_links(self.cut_links)
        self.last_geometry_update = geometry
        self._impulses_corrected = False
        self._prepared = HomeFreeRigidPrepareDiagnostics(
            geometry=geometry,
            transitions=transitions,
        )
        return self._prepared

    def step_prepared(
        self,
        body_q_sample: wp.array,
        dt: float | None = None,
        *,
        retain_prepared: bool = False,
    ) -> HomeFreeRigidDynamicStepDiagnostics:
        """Advance fluid and collect load for the last successfully prepared pose."""

        if self._prepared is None:
            raise RuntimeError("prepare_step() must succeed before step_prepared()")
        prepared = self._prepared
        fluid_dt = self.fluid_domain.model.time_step if dt is None else float(dt)
        geometric_stepper = getattr(self.fluid_domain, "geometric_stepper", None)
        if geometric_stepper is None:
            wp.copy(self._load_flags, self.fluid_domain.free_surface_state.flags)
        self.fluid_domain.step(fluid_dt)
        if geometric_stepper is not None:
            wp.copy(self._load_flags, geometric_stepper.working_free_surface.flags)
        load = self._correct_ambient_pressure()
        self.wrench.accumulate_from_rigid(
            self.cut_links,
            body_q_sample,
            self.body_com_local,
            self.fluid_domain.model.scaling,
        )
        assert self.fluid_domain.last_diagnostics is not None
        if not retain_prepared:
            self._prepared = None
        return HomeFreeRigidDynamicStepDiagnostics(
            geometry=prepared.geometry,
            transitions=prepared.transitions,
            fluid=self.fluid_domain.last_diagnostics,
            load=load,
        )

    def finish_prepared_step(self) -> None:
        """Close a retained prepared transaction after its final evaluation."""

        if self._prepared is None:
            raise RuntimeError("no retained HOME-Free rigid step is active")
        self._prepared = None

    def set_prepared_wall_velocity(
        self,
        body_q_sample: wp.array,
        body_qd_sample: wp.array,
        dt: float,
    ) -> None:
        """Update only wall velocity while retaining frozen geometry/topology."""

        if self._prepared is None:
            raise RuntimeError("no retained HOME-Free rigid step is active")
        self.cut_links.set_rigid_wall_velocity(
            body_q_sample,
            body_qd_sample,
            self.body_com_local,
            self.fluid_domain.model.fluid_grid_cell_size,
            dt,
        )
        self._impulses_corrected = False

    def abort_prepared_step(self) -> None:
        """Discard retained transaction metadata during an outer rollback."""

        self._prepared = None

    def _correct_ambient_pressure(self) -> HomeFreeRigidLoadDiagnostics:
        if self._impulses_corrected:
            raise RuntimeError("cut-link impulses were already pressure-corrected")
        self._wet_link_count_by_body.zero_()
        self._dry_link_count_by_body.zero_()
        self._invalid_flag_count.zero_()
        self._dry_nonzero_impulse_count.zero_()
        self._ambient_pressure_correction.zero_()
        if self.cut_links.link_count:
            wp.launch(
                _classify_phase_owned_links,
                dim=self.cut_links.link_count,
                inputs=[
                    self.cut_links.cell,
                    self.cut_links.body_id,
                    self._load_flags,
                    self.cut_links.impulse,
                    self._wet_link_count_by_body,
                    self._dry_link_count_by_body,
                    self._invalid_flag_count,
                    self._dry_nonzero_impulse_count,
                    self.rigid_domain.model.body_count,
                    int(self.fluid_domain.model.ny),
                    int(self.fluid_domain.model.nz),
                ],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        wet_by_body = self._wet_link_count_by_body.numpy()
        dry_by_body = self._dry_link_count_by_body.numpy()
        diagnostics = HomeFreeRigidLoadDiagnostics(
            wet_link_count=int(np.sum(wet_by_body, dtype=np.int64)),
            dry_link_count=int(np.sum(dry_by_body, dtype=np.int64)),
            invalid_flag_count=int(self._invalid_flag_count.numpy()[0]),
            dry_nonzero_impulse_count=int(
                self._dry_nonzero_impulse_count.numpy()[0]
            ),
            ambient_pressure_correction_lattice=(0.0, 0.0, 0.0),
        )
        if diagnostics.invalid_flag_count:
            raise RuntimeError(
                f"{diagnostics.invalid_flag_count} cut links are owned by solid/invalid VOF cells"
            )
        if diagnostics.dry_nonzero_impulse_count:
            raise RuntimeError(
                f"{diagnostics.dry_nonzero_impulse_count} dry cut links received fluid impulse"
            )
        if diagnostics.wet_link_count:
            wp.launch(
                _remove_ambient_pressure_from_wet_links,
                dim=self.cut_links.link_count,
                inputs=[
                    self.cut_links.body_id,
                    self.cut_links.cell,
                    self.cut_links.direction,
                    self._load_flags,
                    self._directions,
                    self._weights,
                    self.cut_links.impulse,
                    self._ambient_pressure_correction,
                    self.fluid_domain.gas_density,
                    int(self.fluid_domain.model.ny),
                    int(self.fluid_domain.model.nz),
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            diagnostics = HomeFreeRigidLoadDiagnostics(
                wet_link_count=diagnostics.wet_link_count,
                dry_link_count=diagnostics.dry_link_count,
                invalid_flag_count=diagnostics.invalid_flag_count,
                dry_nonzero_impulse_count=diagnostics.dry_nonzero_impulse_count,
                ambient_pressure_correction_lattice=tuple(
                    float(value)
                    for value in self._ambient_pressure_correction.numpy()
                ),
            )
        self._impulses_corrected = True
        return diagnostics
