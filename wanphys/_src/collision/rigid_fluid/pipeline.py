# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Native particle-shape collision pipeline for rigid-fluid coupling."""

from __future__ import annotations

from typing import Any

import warp as wp
from newton import Contacts

from .config import RigidFluidCollisionConfig, resolve_rigid_fluid_collision_config
from wanphys._src.collision.particles.kernels import create_soft_contacts


class RigidFluidCollisionPipeline:
    """Generate soft particle-shape contacts for a rigid/fluid domain pair."""

    def __init__(
        self,
        rigid_model: Any,
        particle_model: Any,
        *,
        config: RigidFluidCollisionConfig | None = None,
    ):
        self.rigid_model = rigid_model
        self.particle_model = particle_model
        self.device = rigid_model.device
        self.config = resolve_rigid_fluid_collision_config(config)
        self.soft_contact_margin = float(self.config.soft_contact_margin)

        self.shape_count = int(rigid_model.shape_count)
        self.particle_count = int(particle_model.particle_count)
        if self.config.soft_contact_max is None:
            soft_contact_max = self.shape_count * self.particle_count
        else:
            soft_contact_max = int(self.config.soft_contact_max)
        self._soft_contact_max = soft_contact_max

    @property
    def soft_contact_max(self) -> int:
        return self._soft_contact_max

    def contacts(self) -> Contacts:
        return Contacts(
            0,
            self.soft_contact_max,
            requires_grad=False,
            device=self.device,
            per_contact_shape_properties=False,
        )

    def collide(
        self,
        rigid_state: Any,
        fluid_state: Any,
        contacts: Contacts | None = None,
        *,
        soft_contact_margin: float | None = None,
    ) -> Contacts:
        if contacts is None:
            contacts = self.contacts()

        contacts.clear()

        soft_contact_margin = soft_contact_margin if soft_contact_margin is not None else self.soft_contact_margin

        if self.soft_contact_max == 0:
            return contacts

        wp.launch(
            kernel=create_soft_contacts,
            dim=self.soft_contact_max,
            inputs=[
                fluid_state.particle_q,
                self.particle_model.particle_radius,
                self.particle_model.particle_flags,
                self.particle_model.particle_world_ids,
                rigid_state.body_q,
                self.rigid_model.shape_transform,
                self.rigid_model.shape_body,
                self.rigid_model.shape_type,
                self.rigid_model.shape_scale,
                self.rigid_model.shape_source_ptr,
                self.rigid_model.shape_world,
                soft_contact_margin,
                self.soft_contact_max,
                self.shape_count,
                self.rigid_model.shape_flags,
                self.rigid_model.shape_heightfield_data,
                self.rigid_model.heightfield_elevation_data,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_tids,
            ],
            device=self.device,
        )

        return contacts