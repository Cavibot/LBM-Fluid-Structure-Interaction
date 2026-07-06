# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Cache key helpers for the outer collision pipeline API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wanphys._src.collision.rigid.config import RigidCollisionConfig, resolve_rigid_collision_config
from wanphys._src.collision.rigid_fluid import RigidFluidCollisionConfig, resolve_rigid_fluid_collision_config

if TYPE_CHECKING:
    from wanphys._src.core.domain import Domain
    from wanphys._src.rigid.domain import RigidDomain


def rigid_pipeline_cache_key(
    domain: RigidDomain,
    config: RigidCollisionConfig | None = None,
) -> tuple[int, tuple[object, ...]]:
    config = resolve_rigid_collision_config(config)
    implicit_pairs_id = None
    if config.broad_phase == "explicit" and config.shape_pairs_filtered is None:
        implicit_pairs_id = id(getattr(domain.model, "shape_contact_pairs", None))
    return (id(domain), config.native_cache_key() + (implicit_pairs_id,))


def rigid_fluid_pipeline_cache_key(
    rigid_domain: RigidDomain,
    fluid_domain: Domain,
    config: RigidFluidCollisionConfig | None = None,
) -> tuple[int, int, tuple[object, ...]]:
    effective_config = resolve_rigid_fluid_collision_config(config)
    return (id(rigid_domain), id(fluid_domain), effective_config.native_cache_key())


def rigid_bridge_cache_key(domain: RigidDomain) -> int:
    return id(domain)


def rigid_fluid_bridge_cache_key(
    rigid_domain: RigidDomain,
    fluid_domain: Domain,
) -> tuple[int, int]:
    return (id(rigid_domain), id(fluid_domain))
