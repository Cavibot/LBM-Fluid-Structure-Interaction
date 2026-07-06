# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration for native rigid collision detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF


@dataclass
class RigidCollisionConfig:
    """Configuration for a rigid-domain collision query."""

    broad_phase: Literal["explicit", "nxn", "sap", "bvh", "hash"] = "explicit"
    reduce_contacts: bool = True
    rigid_contact_max: int | None = None
    max_triangle_pairs: int = 1000000
    max_heightfield_cell_pairs: int = 1000000
    shape_pairs_filtered: Any | None = None
    sdf_hydroelastic_config: HydroelasticSDF.Config | None = None
    requires_grad: bool | None = None

    def native_cache_key(self) -> tuple[object, ...]:
        """Return a stable cache key for rigid collision pipeline reuse."""

        return (
            self.broad_phase,
            self.reduce_contacts,
            self.rigid_contact_max,
            self.max_triangle_pairs,
            self.max_heightfield_cell_pairs,
            id(self.shape_pairs_filtered) if self.shape_pairs_filtered is not None else None,
            _hydroelastic_config_cache_key(self.sdf_hydroelastic_config),
            self.requires_grad,
        )


def resolve_rigid_collision_config(
    config: RigidCollisionConfig | None = None,
) -> RigidCollisionConfig:
    """Return the effective rigid collision config."""

    return config if config is not None else RigidCollisionConfig()


def _hydroelastic_config_cache_key(config: HydroelasticSDF.Config | None) -> tuple[object, ...] | None:
    if config is None:
        return None
    return tuple(sorted(asdict(config).items()))