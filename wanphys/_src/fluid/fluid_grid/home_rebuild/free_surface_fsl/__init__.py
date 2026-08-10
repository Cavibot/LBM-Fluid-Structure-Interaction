# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful link-wise free-surface layer for the isolated HOME core."""

from .advection import FslMassAdvector, FslMassExchangeDiagnostics
from .flags import FslCellFlag
from .pressure import FslOnlyMissingDiagnostics, FslOnlyMissingStreamer
from .stepper import (
    FslDynamicTopologyDiagnostics,
    FslDynamicTopologyStepper,
    FslFixedTopologyDiagnostics,
    FslFixedTopologyStepper,
)
from .reference import (
    FslStateDiagnostics,
    FslOnlyMissingStreamResult,
    classify_fill,
    collide_streamed_moments,
    gas_pressure_boundary_populations,
    initialize_fsl_fields,
    link_mass_exchange,
    only_missing_stream_moments,
    reconstruct_populations,
    validate_fsl_fields,
)
from .state import FslState
from .topology import FslTopologyUpdater, FslWarpTopologyDiagnostics
from .topology_reference import (
    FslTopologyDiagnostics,
    FslTopologyResult,
    resolve_fsl_topology,
)
from .wall_reference import (
    FslHydrostaticFields,
    FslWallStreamResult,
    axis_aligned_wall_mask,
    closed_box_wall_mask,
    initialize_hydrostatic_fields,
    validate_closed_wall_mask,
    validate_wall_mask,
    wall_link_mass_exchange,
    wall_only_missing_stream_moments,
)
from .walls import FslWallMask
from .wall_pipeline import (
    FslWallActiveCollider,
    FslWallMassAdvector,
    FslWallMassDiagnostics,
    FslWallOnlyMissingStreamer,
    FslWallStreamDiagnostics,
)
from .wall_topology import FslWallTopologyUpdater
from .wall_stepper import FslWallDynamicDiagnostics, FslWallDynamicTopologyStepper

__all__ = [
    "FslCellFlag",
    "FslMassAdvector",
    "FslMassExchangeDiagnostics",
    "FslOnlyMissingStreamResult",
    "FslOnlyMissingDiagnostics",
    "FslOnlyMissingStreamer",
    "FslFixedTopologyDiagnostics",
    "FslFixedTopologyStepper",
    "FslDynamicTopologyDiagnostics",
    "FslDynamicTopologyStepper",
    "FslState",
    "FslTopologyUpdater",
    "FslWarpTopologyDiagnostics",
    "FslTopologyDiagnostics",
    "FslTopologyResult",
    "resolve_fsl_topology",
    "FslStateDiagnostics",
    "classify_fill",
    "collide_streamed_moments",
    "gas_pressure_boundary_populations",
    "initialize_fsl_fields",
    "link_mass_exchange",
    "only_missing_stream_moments",
    "reconstruct_populations",
    "validate_fsl_fields",
    "FslHydrostaticFields",
    "FslWallMask",
    "FslWallStreamResult",
    "axis_aligned_wall_mask",
    "closed_box_wall_mask",
    "initialize_hydrostatic_fields",
    "validate_closed_wall_mask",
    "validate_wall_mask",
    "wall_link_mass_exchange",
    "wall_only_missing_stream_moments",
    "FslWallActiveCollider",
    "FslWallMassAdvector",
    "FslWallMassDiagnostics",
    "FslWallOnlyMissingStreamer",
    "FslWallStreamDiagnostics",
    "FslWallTopologyUpdater",
    "FslWallDynamicDiagnostics",
    "FslWallDynamicTopologyStepper",
]
