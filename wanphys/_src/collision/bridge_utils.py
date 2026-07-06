# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers for Newton bridge collisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wanphys._src.core.domain import Domain
    from wanphys._src.rigid.domain import RigidDomain


def build_rigid_newton_bridge(domain: RigidDomain):
    import newton as _newton

    rigid_nm = domain.model._newton_backend
    device = rigid_nm.device

    bridge = _newton.Model(device=device)
    bridge.body_count = rigid_nm.body_count
    bridge.shape_count = rigid_nm.shape_count
    for attr in [
        "shape_transform",
        "shape_body",
        "shape_type",
        "shape_scale",
        "shape_source_ptr",
        "shape_world",
        "shape_flags",
        "shape_material_kh",
        "shape_collision_radius",
        "shape_margin",
        "shape_gap",
        "shape_heightfield_data",
        "heightfield_elevation_data",
        "sdf_data",
        "shape_sdf_index",
        "sdf_block_coords",
        "sdf_index2blocks",
        "shape_collision_aabb_lower",
        "shape_collision_aabb_upper",
        "shape_contact_pairs",
        "shape_contact_pair_count",
        "_shape_voxel_resolution",
    ]:
        val = getattr(rigid_nm, attr, None)
        if val is not None:
            setattr(bridge, attr, val)

    bridge.particle_count = 0
    bridge_state = _newton.State()
    return bridge, bridge_state


def sync_rigid_newton_bridge_state(bridge_state, domain: RigidDomain) -> None:
    state = domain.state
    bridge_state.body_q = state.body_q
    bridge_state.body_qd = state.body_qd


def build_rigid_fluid_newton_bridge(rigid_domain: RigidDomain, fluid_domain: Domain):
    import newton as _newton
    import warp as wp

    rigid_nm = rigid_domain.model._newton_backend
    fluid_model = fluid_domain.model
    device = rigid_nm.device

    bridge = _newton.Model(device=device)
    bridge.body_count = rigid_nm.body_count
    bridge.shape_count = rigid_nm.shape_count
    for attr in [
        "shape_transform",
        "shape_body",
        "shape_type",
        "shape_scale",
        "shape_source_ptr",
        "shape_world",
        "shape_flags",
        "shape_collision_radius",
        "shape_margin",
        "shape_gap",
        "shape_heightfield_data",
        "heightfield_elevation_data",
        "shape_collision_aabb_lower",
        "shape_collision_aabb_upper",
    ]:
        val = getattr(rigid_nm, attr, None)
        if val is not None:
            setattr(bridge, attr, val)

    bridge.particle_count = fluid_model.particle_count
    for attr in ["particle_radius", "particle_mass", "particle_flags"]:
        val = getattr(fluid_model, attr, None)
        if val is not None:
            setattr(bridge, attr, val)

    particle_world_ids = getattr(fluid_model, "particle_world_ids", None)
    if particle_world_ids is not None:
        bridge.particle_world = particle_world_ids
    else:
        bridge.particle_world = wp.zeros(fluid_model.particle_count, dtype=wp.int32, device=device)
    bridge.shape_contact_pairs = wp.empty(0, dtype=wp.vec2i, device=device)
    bridge.shape_contact_pair_count = 0

    bridge_state = _newton.State()
    return bridge, bridge_state


def sync_rigid_fluid_newton_bridge_state(bridge_state, rigid_domain: RigidDomain, fluid_domain: Domain) -> None:
    rigid_state = rigid_domain.state
    fluid_state = fluid_domain.state
    bridge_state.body_q = rigid_state.body_q
    bridge_state.body_qd = rigid_state.body_qd
    bridge_state.particle_q = fluid_state.particle_q
    bridge_state.particle_qd = fluid_state.particle_qd
