# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Force evaluation sub-package for the symplectic integrator.

Each module provides GPU kernels and Python launchers for a category of
force contributions:

* :mod:`.elastic` – spring-dashpot, membrane/solid FEM, hinge bending
* :mod:`.collision` – particle proximity, mesh-particle, rigid penalty,
  particle-shape contacts
* :mod:`.articulation` – joint constraint forces (unified & dispatched)
* :mod:`.actuator` – muscle / tendon waypoint actuators
* :mod:`.material` – hyperelastic constitutive (material) laws
"""

from .actuator import apply_muscle_actuators
from .articulation import (
    ArticulationDispatcher,
    apply_articulation_forces,
    apply_articulation_forces_dispatched,
)
from .collision import (
    apply_mesh_particle_contact,
    apply_particle_proximity,
    apply_particle_shape_contact,
    apply_rigid_contacts,
)
from .elastic import (
    apply_hinge_bending,
    apply_membrane_stress,
    apply_solid_stress,
    apply_spring_dashpot,
)
from .material import MaterialLaw

__all__ = [
    # material
    "MaterialLaw",
    # elastic
    "apply_spring_dashpot",
    "apply_membrane_stress",
    "apply_hinge_bending",
    "apply_solid_stress",
    # articulation
    "ArticulationDispatcher",
    "apply_articulation_forces",
    "apply_articulation_forces_dispatched",
    # collision
    "apply_particle_proximity",
    "apply_mesh_particle_contact",
    "apply_rigid_contacts",
    "apply_particle_shape_contact",
    # actuator
    "apply_muscle_actuators",
]
