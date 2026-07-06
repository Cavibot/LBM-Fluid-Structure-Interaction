# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys symplectic Euler solver package.

Public API
----------
* :class:`SymplecticEulerSolver` — the main integrator.
* :class:`MaterialLaw` — hyperelastic material law catalogue.
* :class:`ArticulationDispatcher` — per-type joint kernel dispatcher.
"""

from .forces import (
    ArticulationDispatcher,
    MaterialLaw,
    apply_articulation_forces,
    apply_articulation_forces_dispatched,
)
from .integrator import SymplecticEulerSolver

__all__: list[str] = [
    "SymplecticEulerSolver",
    "MaterialLaw",
    "ArticulationDispatcher",
    "apply_articulation_forces",
    "apply_articulation_forces_dispatched",
]

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (will be removed in a future release)
# ---------------------------------------------------------------------------
WanPhysSemiImplicitSolver = SymplecticEulerSolver
ConstitutiveModel = MaterialLaw
JointProcessor = ArticulationDispatcher
