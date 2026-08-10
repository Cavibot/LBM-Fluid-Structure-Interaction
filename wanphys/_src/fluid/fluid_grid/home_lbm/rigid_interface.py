# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Verified geometry/load exchange surface between HOME-LBM and rigid bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import warp as wp

from .cut_link import CutLinkBuffer
from .domain import HomeLbmDomain
from .rigid_geometry import HomeLbmRigidGeometry, RigidGeometryUpdateDiagnostics
from .transitions import CellTransitionDiagnostics, ConservativeCellRemapper
from .wrench import FluidWrenchBuffer

if TYPE_CHECKING:
    from wanphys._src.rigid.domain import RigidDomain


class HomeLbmRigidInterface:
    """Prepare midpoint boundaries and collect loads without advancing either domain."""

    def __init__(
        self,
        fluid_domain: HomeLbmDomain,
        rigid_domain: RigidDomain,
        body_ids: Iterable[int] | None = None,
    ) -> None:
        if fluid_domain.model._device != rigid_domain.model.device:
            raise ValueError("HOME and rigid domains must use the same device")
        if fluid_domain._state_in is None:
            fluid_domain.create_state()
        if rigid_domain._state_in is None:
            rigid_domain.create_state()

        self.fluid_domain = fluid_domain
        self.rigid_domain = rigid_domain
        self.device = fluid_domain.model._device
        self.body_com_local = rigid_domain.model.body_com
        self.geometry = HomeLbmRigidGeometry.from_domain(rigid_domain, body_ids=body_ids)
        self.geometry.update_radius_bounds(self.body_com_local)
        self.wrench = FluidWrenchBuffer(rigid_domain.model.body_count, device=self.device)

        rigid_state = rigid_domain.state
        self.last_geometry_update: RigidGeometryUpdateDiagnostics = self.geometry.rasterize(
            fluid_domain.state, rigid_state.body_q
        )
        self.remapper = ConservativeCellRemapper(fluid_domain.state)
        self.cut_links = self._build_links(rigid_state.body_q, rigid_state.body_qd)
        self.fluid_domain.solver.set_cut_links(self.cut_links)

    def prepare_step(
        self,
        body_q_sample: wp.array,
        body_qd_sample: wp.array,
    ) -> CellTransitionDiagnostics:
        """Rasterize one pose sample and install its cut links on the HOME solver."""

        self.last_geometry_update = self.geometry.rasterize_incremental(
            self.fluid_domain.state, body_q_sample
        )
        transitions = self.remapper.remap(self.fluid_domain.state)
        reuse_topology = (
            transitions.fresh_cell_count == 0 and transitions.dead_cell_count == 0
        )
        self.cut_links = self._build_links(
            body_q_sample,
            body_qd_sample,
            reuse_topology=reuse_topology,
        )
        self.fluid_domain.solver.set_cut_links(self.cut_links)
        return transitions

    def collect_wrench(
        self,
        body_q_sample: wp.array,
        body_f: wp.array | None = None,
    ) -> FluidWrenchBuffer:
        """Reduce the last HOME boundary impulses and optionally add physical loads."""

        self.wrench.accumulate_from_rigid(
            self.cut_links,
            body_q_sample,
            self.body_com_local,
            self.fluid_domain.model.scaling,
        )
        if body_f is not None:
            self.wrench.add_to_rigid_forces(body_f, self.fluid_domain.model.scaling)
        return self.wrench

    def _build_links(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        reuse_topology: bool = False,
    ) -> CutLinkBuffer:
        if reuse_topology:
            links = self.cut_links
            links.update_from_sdf(self.fluid_domain.state)
        else:
            links = CutLinkBuffer.build_from_sdf(self.fluid_domain.state)
        links.set_rigid_wall_velocity(
            body_q,
            body_qd,
            self.body_com_local,
            self.fluid_domain.model.fluid_grid_cell_size,
            self.fluid_domain.model.time_step,
        )
        links.validate_body_ids()
        return links
