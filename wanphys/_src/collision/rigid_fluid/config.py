# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration for native rigid-fluid collision detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RigidFluidCollisionConfig:
    """Configuration for a rigid-fluid domain-pair collision query."""

    soft_contact_margin: float = 0.01
    soft_contact_max: int | None = None

    def native_cache_key(self) -> tuple[object, ...]:
        """Return a stable cache key for rigid-fluid pipeline reuse."""

        return (self.soft_contact_margin, self.soft_contact_max)


def resolve_rigid_fluid_collision_config(
    config: RigidFluidCollisionConfig | None = None,
) -> RigidFluidCollisionConfig:
    """Return the effective rigid-fluid collision config."""

    return config if config is not None else RigidFluidCollisionConfig()
