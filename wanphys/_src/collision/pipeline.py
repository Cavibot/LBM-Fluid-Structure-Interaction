# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Centralized collision pipeline for WanPhys.

CollisionPipeline provides three classmethods for domain-specific collision
detection.  All examples and domain code call these directly — there is no
instance API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wanphys._src.collision.bridge_utils import (
    build_rigid_newton_bridge,
    build_rigid_fluid_newton_bridge,
    sync_rigid_newton_bridge_state,
    sync_rigid_fluid_newton_bridge_state,
)

from wanphys._src.collision.cache_utils import (
    rigid_bridge_cache_key,
    rigid_fluid_bridge_cache_key,
    rigid_fluid_pipeline_cache_key,
    rigid_pipeline_cache_key,
)

from wanphys._src.collision.contact_merge import build_merged_contacts

from wanphys._src.collision.rigid import (
    RigidCollisionConfig, 
    RigidCollisionPipeline,
    resolve_rigid_collision_config, 
)

from wanphys._src.collision.rigid_fluid import (
    RigidFluidCollisionConfig,
    RigidFluidCollisionPipeline,
    resolve_rigid_fluid_collision_config,
)

if TYPE_CHECKING:
    from newton import Contacts
    from wanphys._src.core.domain import Domain
    from wanphys._src.rigid.domain import RigidDomain


class CollisionPipeline:
    """Centralized collision detection pipeline.

    All methods are classmethods — no instance creation is needed.

    Example::

        for _ in range(1000):
            contacts_rigid = CollisionPipeline.collide_rigid(rigid_domain)
            contacts_fluid = CollisionPipeline.collide_particles(fluid_domain)
            rigid_domain.step(dt, contacts=contacts_rigid)
            fluid_domain.step(dt, contacts=contacts_fluid)
    """

    # ------------------------------------------------------------------
    # Domain-specific collision API
    # ------------------------------------------------------------------

    # Cache for bridge models keyed by (rigid_domain_id, fluid_domain_id)
    _rigid_fluid_bridge_cache: dict[tuple[int, int], tuple[Any, Any]] = {}
    # Cache for rigid-only bridge models keyed by domain id
    _rigid_bridge_cache: dict[int, tuple[Any, Any]] = {}
    # Cache for rigid collision pipelines keyed by domain/config
    _rigid_pipeline_cache: dict[tuple[int, tuple[object, ...]], RigidCollisionPipeline] = {}
    # Cache for rigid-fluid collision pipelines keyed by domain pair/config
    _rigid_fluid_pipeline_cache: dict[tuple[int, int, tuple[object, ...]], RigidFluidCollisionPipeline] = {}
    # Cache for static rigid shape query data keyed by domain id
    _rigid_shape_query_cache: dict[int, Any] = {}
    
    @classmethod
    def _get_rigid_pipeline(
        cls,
        domain: RigidDomain,
        config: RigidCollisionConfig | None = None,
    ) -> RigidCollisionPipeline:
        effective_config = resolve_rigid_collision_config(config)
        cache_key = rigid_pipeline_cache_key(domain, effective_config)
        pipeline = cls._rigid_pipeline_cache.get(cache_key)

        if pipeline is None:
            pipeline = RigidCollisionPipeline(
                domain.model,
                reduce_contacts=effective_config.reduce_contacts,
                rigid_contact_max=effective_config.rigid_contact_max,
                max_triangle_pairs=effective_config.max_triangle_pairs,
                max_heightfield_cell_pairs=effective_config.max_heightfield_cell_pairs,
                shape_pairs_filtered=effective_config.shape_pairs_filtered,
                requires_grad=effective_config.requires_grad,
                broad_phase=effective_config.broad_phase,
                sdf_hydroelastic_config=effective_config.sdf_hydroelastic_config,
            )
            cls._rigid_pipeline_cache[cache_key] = pipeline

        return pipeline

    @classmethod
    def _get_rigid_fluid_pipeline(
        cls,
        rigid_domain: RigidDomain,
        fluid_domain: Domain,
        config: RigidFluidCollisionConfig | None = None,
    ) -> RigidFluidCollisionPipeline:
        effective_config = resolve_rigid_fluid_collision_config(config)
        cache_key = rigid_fluid_pipeline_cache_key(rigid_domain, fluid_domain, effective_config)
        pipeline = cls._rigid_fluid_pipeline_cache.get(cache_key)
        if pipeline is None:
            pipeline = RigidFluidCollisionPipeline(
                rigid_domain.model,
                fluid_domain.model,
                config=effective_config,
            )
            cls._rigid_fluid_pipeline_cache[cache_key] = pipeline
        return pipeline

    @classmethod
    def get_shape_query(
        cls,
        domain: RigidDomain,
        rebuild: bool = False,
    ) -> Any:
        """Get cached static data for Warp-callable rigid shape queries.

        The returned object's ``data`` field is static model/query data shared
        by point-to-shape query operations. Kernels should pass the current
        ``domain.state.body_q`` explicitly to operations such as
        ``wanphys.geometry.point_shape_distance`` or
        ``wanphys.geometry.point_body_distance``.
        """

        from wanphys._src.geometry.rigid_shape_distance import RigidShapeQuery

        cache_key = id(domain)
        cache = cls._rigid_shape_query_cache.get(cache_key)
        if (
            not rebuild
            and cache is not None
            and getattr(cache, "model_id", None) == id(domain.model)
            and cache.body_count == int(domain.model.body_count)
            and cache.shape_count == int(domain.model.shape_count)
        ):
            return cache

        cache = RigidShapeQuery.from_domain(domain)
        cls._rigid_shape_query_cache[cache_key] = cache
        return cache

    @classmethod
    def collide_rigid_fluid(
        cls,
        rigid_domain: RigidDomain,
        fluid_domain: Domain,
        config: RigidFluidCollisionConfig | None = None,
    ) -> Any:
        """Run native rigid-fluid collision detection."""
        pipeline = cls._get_rigid_fluid_pipeline(rigid_domain, fluid_domain, config=config)
        return pipeline.collide(rigid_domain.state, fluid_domain.state)

    @classmethod
    def collide_rigid_fluid_newton(cls, rigid_domain: RigidDomain, fluid_domain: Domain) -> Any:
        """Run the legacy bridge-based rigid-fluid collision path.

        Builds (and caches) a bridge Newton model+state from data arrays of
        both domains, runs Newton's soft contact detection, returns raw Contacts.

        The bridge model is built once per domain pair and reused on subsequent
        calls.  Only the state arrays (body_q/qd, particle_q/qd) are updated
        each call — these are zero-copy aliases, so no data is copied.

        Args:
            rigid_domain: RigidDomain with shapes/bodies.
            fluid_domain: WCSPHDomain (or any particle fluid domain).

        Returns:
            Raw Newton Contacts object from bridge model collision.
        """
        cache_key = rigid_fluid_bridge_cache_key(rigid_domain, fluid_domain)
        cached = cls._rigid_fluid_bridge_cache.get(cache_key)

        if cached is None:
            cached = build_rigid_fluid_newton_bridge(rigid_domain, fluid_domain)
            cls._rigid_fluid_bridge_cache[cache_key] = cached

        bridge, bridge_state = cached
        sync_rigid_fluid_newton_bridge_state(bridge_state, rigid_domain, fluid_domain)
        return bridge.collide(bridge_state)

    @classmethod
    def collide_rigid_newton(cls, domain: RigidDomain) -> Any:
        """Run the legacy bridge-based rigid collision path."""

        cache_key = rigid_bridge_cache_key(domain)
        cached = cls._rigid_bridge_cache.get(cache_key)

        if cached is None:
            cached = build_rigid_newton_bridge(domain)
            cls._rigid_bridge_cache[cache_key] = cached

        bridge, bridge_state = cached
        sync_rigid_newton_bridge_state(bridge_state, domain)
        return bridge.collide(bridge_state)

    @classmethod
    def collide_rigid(
        cls,
        domain: RigidDomain,
        config: RigidCollisionConfig | None = None,
    ) -> Any:
        """Run collision detection for a rigid body domain.

        Args:
            domain: RigidDomain with initialized state.
            config: Optional rigid collision configuration. When omitted,
                default values are used.

        Returns:
            Raw Newton Contacts object.
        """
        pipeline = cls._get_rigid_pipeline(domain, config=config)
        return pipeline.collide(domain.state)

    @classmethod
    def collide_particles(cls, domain: Domain) -> Any:
        """Run particle-shape intra-domain collision for a particle fluid domain.

        Delegates to the domain's model adapter's collide() method.
        Returns None if the domain has no shapes to collide against.

        Args:
            domain: Particle fluid domain (PBFDomain, WCSPHDomain, etc.)
                with a ``_model_adapter`` that has ``collide()`` and
                ``shape_count``.

        Returns:
            Raw Newton Contacts object, or None if there are no shapes.
        """
        adapter = getattr(domain, "_model_adapter", None)
        if adapter is not None and hasattr(adapter, "collide"):
            if getattr(adapter, "shape_count", 0) > 0:
                return adapter.collide(domain.state)
        return None

    @staticmethod
    def merge_contacts(
        *,
        rigid_src: Contacts | None = None,
        soft_src: Contacts | None = None,
    ) -> Contacts:
        return build_merged_contacts(rigid_src=rigid_src, soft_src=soft_src)
