# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-body reduction of Galilean-invariant cut-link momentum exchange."""

from __future__ import annotations

import numpy as np
import warp as wp

from .cut_link import CutLinkBuffer
from .scaling import LatticeScaling


@wp.kernel
def _accumulate_link_wrenches(
    body_id: wp.array(dtype=wp.int32),
    intersection_lattice: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    body_com_lattice: wp.array(dtype=wp.vec3),
    force_lattice: wp.array(dtype=wp.vec3),
    torque_lattice: wp.array(dtype=wp.vec3),
    body_count: int,
):
    link = wp.tid()
    body = body_id[link]
    if body >= 0 and body < body_count:
        link_impulse = impulse[link]
        lever = intersection_lattice[link] - body_com_lattice[body]
        wp.atomic_add(force_lattice, body, link_impulse)
        wp.atomic_add(torque_lattice, body, wp.cross(lever, link_impulse))


@wp.kernel
def _compute_body_centers_lattice(
    body_q: wp.array(dtype=wp.transform),
    body_com_local: wp.array(dtype=wp.vec3),
    body_com_lattice: wp.array(dtype=wp.vec3),
    inverse_cell_size: float,
):
    body = wp.tid()
    center_world = wp.transform_point(body_q[body], body_com_local[body])
    body_com_lattice[body] = center_world * inverse_cell_size


@wp.kernel
def _add_physical_wrenches(
    force_lattice: wp.array(dtype=wp.vec3),
    torque_lattice: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    force_unit: float,
    torque_unit: float,
):
    body = wp.tid()
    force = force_lattice[body] * force_unit
    torque = torque_lattice[body] * torque_unit
    wp.atomic_add(body_f, body, wp.spatial_vector(force, torque))


class FluidWrenchBuffer:
    """Fluid loads kept separate from rigid-body accumulators until coupling."""

    def __init__(self, body_count: int, device: wp.DeviceLike = None) -> None:
        if body_count < 0:
            raise ValueError(f"body_count must be non-negative, got {body_count}")
        self.body_count = int(body_count)
        self.device = wp.get_device(device)
        self.force_lattice = wp.zeros(self.body_count, dtype=wp.vec3, device=self.device)
        self.torque_lattice = wp.zeros(self.body_count, dtype=wp.vec3, device=self.device)
        self.body_com_lattice = wp.zeros(self.body_count, dtype=wp.vec3, device=self.device)

    def clear(self) -> None:
        self.force_lattice.zero_()
        self.torque_lattice.zero_()

    def accumulate(
        self,
        cut_links: CutLinkBuffer,
        body_com_lattice: wp.array,
        clear: bool = True,
    ) -> None:
        if cut_links.device != self.device:
            raise ValueError("cut-link and wrench-buffer devices must match")
        if body_com_lattice.device != self.device:
            raise ValueError("body centers and wrench-buffer devices must match")
        if body_com_lattice.dtype != wp.vec3 or len(body_com_lattice) != self.body_count:
            raise ValueError("body_com_lattice must be a vec3 array with one entry per body")
        if clear:
            self.clear()
        if cut_links.link_count:
            wp.launch(
                _accumulate_link_wrenches,
                dim=cut_links.link_count,
                inputs=[
                    cut_links.body_id,
                    cut_links.intersection_lattice,
                    cut_links.impulse,
                    body_com_lattice,
                    self.force_lattice,
                    self.torque_lattice,
                    self.body_count,
                ],
                device=self.device,
            )

    def lattice_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        wp.synchronize_device(self.device)
        return self.force_lattice.numpy(), self.torque_lattice.numpy()

    def physical_numpy(self, scaling: LatticeScaling) -> tuple[np.ndarray, np.ndarray]:
        force, torque = self.lattice_numpy()
        return force * scaling.force_unit, torque * scaling.torque_unit

    def accumulate_from_rigid(
        self,
        cut_links: CutLinkBuffer,
        body_q: wp.array,
        body_com_local: wp.array,
        scaling: LatticeScaling,
        clear: bool = True,
    ) -> None:
        if body_q.device != self.device or body_com_local.device != self.device:
            raise ValueError("rigid geometry and wrench-buffer devices must match")
        if body_q.dtype != wp.transform or body_com_local.dtype != wp.vec3:
            raise ValueError("body_q/body_com_local must use transform/vec3 dtypes")
        if len(body_q) != self.body_count or len(body_com_local) != self.body_count:
            raise ValueError("rigid geometry arrays must contain one entry per body")
        if self.body_count:
            wp.launch(
                _compute_body_centers_lattice,
                dim=self.body_count,
                inputs=[
                    body_q,
                    body_com_local,
                    self.body_com_lattice,
                    1.0 / scaling.cell_size,
                ],
                device=self.device,
            )
        self.accumulate(cut_links, self.body_com_lattice, clear=clear)

    def add_to_rigid_forces(self, body_f: wp.array, scaling: LatticeScaling) -> None:
        if body_f.device != self.device or body_f.dtype != wp.spatial_vector:
            raise ValueError("body_f must be a spatial-vector array on the wrench-buffer device")
        if len(body_f) != self.body_count:
            raise ValueError("body_f must contain one entry per body")
        if self.body_count:
            wp.launch(
                _add_physical_wrenches,
                dim=self.body_count,
                inputs=[
                    self.force_lattice,
                    self.torque_lattice,
                    body_f,
                    scaling.force_unit,
                    scaling.torque_unit,
                ],
                device=self.device,
            )
