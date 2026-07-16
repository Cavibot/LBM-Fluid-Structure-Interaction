# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM time-varying simulation state.

Defines :class:`HomeFslbmState`, the GPU-resident state mirroring the
``mrFlow3D`` structure from the reference implementation, plus coupling
fields (solid SDF, solid velocities) for fluid–structure interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from wanphys._src.core.domain import DomainState

from . import constants as C

if TYPE_CHECKING:
    from .model import HomeFslbmModel


class HomeFslbmState(DomainState):
    """GPU-resident state for a HOME-FSLBM free-surface simulation.

    Owns all dynamic arrays: HOME moments, free-surface fields (VOF mass,
    volume fraction, flag bitfield), bubble tracking (tag matrices, CCL
    I/O, double-precision volume/rho arrays), dissolved gas (D3Q7
    distributions, concentration), and coupling fields (solid SDF, solid
    wall velocities).

    Inherits :class:`DomainState` (not ``FluidGridStateBase``) because
    HOME-FSLBM manages its own field set independently of MAC staggered
    grid conventions.  Audit item B9.

    Parameters
    ----------
    model:
        Static HOME-FSLBM configuration.
    requires_grad:
        If ``True``, allocate arrays with gradient tracking enabled.
    """

    def __init__(self, model: HomeFslbmModel, requires_grad: bool = False) -> None:
        self.model: HomeFslbmModel = model
        nx: int = int(model.nx)
        ny: int = int(model.ny)
        nz: int = int(model.nz)
        self.res: tuple[int, int, int] = (nx, ny, nz)
        self.device: wp.Device = model._device
        self.requires_grad: bool = requires_grad

        # Stride for flattening 3D → 1D
        self._stride: int = nx * ny * nz

        # ------------------------------------------------------------------
        # HOME-LBM moments (10 per node, flat arrays)
        # Ref: ``mrFlow3D.h:38-39``
        # ------------------------------------------------------------------
        self.f_mom: wp.array = wp.zeros(
            C.NUM_MOMENTS * self._stride,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.f_mom_post: wp.array = wp.zeros(
            C.NUM_MOMENTS * self._stride,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )

        # ------------------------------------------------------------------
        # Free-surface fields
        # Ref: ``mrFlow3D.h:36,47-49,42-44``
        # ------------------------------------------------------------------
        self.flag: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.uint8, device=self.device
        )
        self.mass: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.massex: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.phi: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_x: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_y: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_z: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

        # ------------------------------------------------------------------
        # Dissolved gas (D3Q7 CMR-MRT)
        # Ref: ``mrFlow3D.h:71-76``
        # ------------------------------------------------------------------
        self.g_mom: wp.array = wp.zeros(
            C.NUM_DIRS_GAS * self._stride,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.g_mom_post: wp.array = wp.zeros(
            C.NUM_DIRS_GAS * self._stride,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.delta_g: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.c_value: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.src: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

        # ------------------------------------------------------------------
        # Volume fraction change accumulator
        # Ref: ``mrFlow3D.h:66``
        # ------------------------------------------------------------------
        self.delta_phi: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

        # ------------------------------------------------------------------
        # Bubble tracking (YACCLAB CCL)
        # Ref: ``mrFlow3D.h:55-63,66-67``
        # ------------------------------------------------------------------
        self.tag_matrix: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.int32, device=self.device
        )
        self.previous_tag: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.int32, device=self.device
        )
        self.previous_merge_tag: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.int32, device=self.device
        )
        self.input_matrix: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.uint8, device=self.device
        )
        self.label_matrix: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.int32, device=self.device
        )

        # merge_detector: per-cell bool → wp.int32 (0/1)
        # Ref: ``mrFlow3D.h:61`` — ``bool* merge_detector``
        self.merge_detector: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.int32, device=self.device
        )

        # Host-side merge/split flags (``mrFlow3D.h:62-63``)
        self.merge_flag: int = 0
        self.split_flag: int = 0

        # ------------------------------------------------------------------
        # Disjoining pressure
        # Ref: ``mrFlow3D.h:69``
        # ------------------------------------------------------------------
        self.disjoin_force: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

        # ------------------------------------------------------------------
        # Islet flag (isolated bubble removal for inlet)
        # Ref: ``mrFlow3D.h:70``
        # ------------------------------------------------------------------
        self.islet: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=wp.int32, device=self.device
        )

        # ------------------------------------------------------------------
        # Bubble property arrays (double precision — audit item B2)
        # Ref: ``mrFlow3D.h:17-27`` — ``mlBubble3D``
        # ------------------------------------------------------------------
        _max_bubbles: int = int(model.max_bubbles)
        self.bubble_volume: wp.array = wp.zeros(
            _max_bubbles, dtype=wp.float64, device=self.device
        )
        self.bubble_init_volume: wp.array = wp.zeros(
            _max_bubbles, dtype=wp.float64, device=self.device
        )
        self.bubble_rho: wp.array = wp.zeros(
            _max_bubbles, dtype=wp.float64, device=self.device
        )
        self.bubble_label_init_volume: wp.array = wp.zeros(
            _max_bubbles, dtype=wp.float64, device=self.device
        )
        self.bubble_label_volume: wp.array = wp.zeros(
            _max_bubbles, dtype=wp.float64, device=self.device
        )

        # Host-side bookkeeping (``mrFlow3D.h:24-26``)
        self.label_num: int = -1
        self.bubble_count: int = 0

        # ------------------------------------------------------------------
        # Solid boundary coupling fields
        # (compatible with FluidGridStateBase convention for FSI)
        # ------------------------------------------------------------------
        self.solid_phi: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.solid_phi.fill_(1000.0)

        self.solid_body_id: wp.array3d = wp.full(
            (nx, ny, nz), -1, dtype=wp.int32, device=self.device
        )

        # MAC staggered solid-wall velocities (for moving-wall bounce-back)
        self.vel_solid_u: wp.array3d = wp.zeros(
            (nx + 1, ny, nz), dtype=float, device=self.device
        )
        self.vel_solid_v: wp.array3d = wp.zeros(
            (nx, ny + 1, nz), dtype=float, device=self.device
        )
        self.vel_solid_w: wp.array3d = wp.zeros(
            (nx, ny, nz + 1), dtype=float, device=self.device
        )

    # ------------------------------------------------------------------
    # DomainState protocol
    # ------------------------------------------------------------------

    def clear_forces(self) -> None:
        """HOME-FSLBM forces are reset in the solver pipeline."""
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Zero all dynamic arrays and reset solid SDF to a large value."""
        self.f_mom.zero_()
        self.f_mom_post.zero_()
        self.flag.zero_()
        self.mass.zero_()
        self.massex.zero_()
        self.phi.zero_()
        self.force_x.zero_()
        self.force_y.zero_()
        self.force_z.zero_()
        self.g_mom.zero_()
        self.g_mom_post.zero_()
        self.delta_g.zero_()
        self.c_value.zero_()
        self.src.zero_()
        self.delta_phi.zero_()
        self.tag_matrix.zero_()
        self.previous_tag.zero_()
        self.previous_merge_tag.zero_()
        self.input_matrix.zero_()
        self.label_matrix.zero_()
        self.merge_detector.zero_()
        self.merge_flag = 0
        self.split_flag = 0
        self.disjoin_force.zero_()
        self.islet.zero_()
        self.bubble_volume.zero_()
        self.bubble_init_volume.zero_()
        self.bubble_rho.zero_()
        self.bubble_label_init_volume.zero_()
        self.bubble_label_volume.zero_()
        self.label_num = -1
        self.bubble_count = 0
        self.solid_phi.fill_(1000.0)
        self.solid_body_id.fill_(-1)
        self.vel_solid_u.zero_()
        self.vel_solid_v.zero_()
        self.vel_solid_w.zero_()

    def clone(self) -> "HomeFslbmState":
        """Deep-copy all GPU arrays into a new :class:`HomeFslbmState`."""
        new_state: HomeFslbmState = HomeFslbmState(
            self.model, requires_grad=self.requires_grad
        )
        wp.copy(new_state.f_mom, self.f_mom)
        wp.copy(new_state.f_mom_post, self.f_mom_post)
        wp.copy(new_state.flag, self.flag)
        wp.copy(new_state.mass, self.mass)
        wp.copy(new_state.massex, self.massex)
        wp.copy(new_state.phi, self.phi)
        wp.copy(new_state.force_x, self.force_x)
        wp.copy(new_state.force_y, self.force_y)
        wp.copy(new_state.force_z, self.force_z)
        wp.copy(new_state.g_mom, self.g_mom)
        wp.copy(new_state.g_mom_post, self.g_mom_post)
        wp.copy(new_state.delta_g, self.delta_g)
        wp.copy(new_state.c_value, self.c_value)
        wp.copy(new_state.src, self.src)
        wp.copy(new_state.delta_phi, self.delta_phi)
        wp.copy(new_state.tag_matrix, self.tag_matrix)
        wp.copy(new_state.previous_tag, self.previous_tag)
        wp.copy(new_state.previous_merge_tag, self.previous_merge_tag)
        wp.copy(new_state.input_matrix, self.input_matrix)
        wp.copy(new_state.label_matrix, self.label_matrix)
        wp.copy(new_state.merge_detector, self.merge_detector)
        new_state.merge_flag = self.merge_flag
        new_state.split_flag = self.split_flag
        wp.copy(new_state.disjoin_force, self.disjoin_force)
        wp.copy(new_state.islet, self.islet)
        wp.copy(new_state.bubble_volume, self.bubble_volume)
        wp.copy(new_state.bubble_init_volume, self.bubble_init_volume)
        wp.copy(new_state.bubble_rho, self.bubble_rho)
        wp.copy(new_state.bubble_label_init_volume, self.bubble_label_init_volume)
        wp.copy(new_state.bubble_label_volume, self.bubble_label_volume)
        new_state.label_num = self.label_num
        new_state.bubble_count = self.bubble_count
        wp.copy(new_state.solid_phi, self.solid_phi)
        wp.copy(new_state.solid_body_id, self.solid_body_id)
        wp.copy(new_state.vel_solid_u, self.vel_solid_u)
        wp.copy(new_state.vel_solid_v, self.vel_solid_v)
        wp.copy(new_state.vel_solid_w, self.vel_solid_w)
        return new_state
