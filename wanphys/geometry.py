# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision detection and geometry utilities for deformable bodies.

This module provides triangle mesh collision detection with BVH acceleration
and mesh-to-point cloud conversion utilities for cloth and deformable body
simulations.

Core components:
- CollisionTriMeshStyle3D: Main Style3D-like collision detection class for cloth/deformable bodies
- BVH: Bounding Volume Hierarchy for broad-phase acceleration
- Mesh sampling: Convert triangle meshes to point clouds

Example:
    >>> from wanphys.geometry import CollisionTriMeshStyle3D
    >>> # Create collision backend for cloth simulation
    >>> collision = CollisionTriMeshStyle3D(...)
    >>> collision.build(model, device)
    >>> collision.generate_candidates(state, params, dt, out_pairs, out_pair_count)
"""

from wanphys._src.geometry import CollisionTriMeshStyle3D
from wanphys._src.geometry import CollisionTriMeshVBD
from wanphys._src.geometry import RigidShapeQuery, RigidShapeQueryData, point_body_distance, point_shape_distance

__all__ = [
    "CollisionTriMeshStyle3D",
    "CollisionTriMeshVBD",
    "RigidShapeQueryData",
    "RigidShapeQuery",
    "point_body_distance",
    "point_shape_distance",
]
