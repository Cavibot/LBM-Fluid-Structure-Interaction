# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys - Multi-domain physics simulation framework.

WanPhys extends Newton with a modular domain architecture that allows
different physics domains (rigid body, fluid, cloth, etc.) to be
simulated together with explicit coupling.

Example:
    >>> from wanphys.rigid import RigidDomain, RigidModelBuilder
    >>> from wanphys.collision import CollisionPipeline
    >>>
    >>> builder = RigidModelBuilder()
    >>> model = builder.finalize()
    >>> domain = RigidDomain(model)
    >>> domain.create_state()
    >>> contacts = CollisionPipeline.collide_rigid(domain)
    >>> domain.step(dt=1/60, contacts=contacts)
"""

from wanphys.collision import CollisionPipeline
from wanphys.geometry import CollisionTriMeshStyle3D, CollisionTriMeshVBD
from wanphys.utils import load_mesh_file, load_point_cloud

__version__ = "0.1.0"

__all__ = [
    "CollisionPipeline",
    "CollisionTriMeshStyle3D",
    "CollisionTriMeshVBD",
    "load_mesh_file",
    "load_point_cloud",
]
