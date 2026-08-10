# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated PLIC geometry for the HOME rebuild experiments."""

from importlib import import_module

from .advection import PlicAxisAdvectionDiagnostics, PlicAxisAdvector
from .advection_reference import (
    PlicAxisAdvectionResult,
    advect_plic_axis,
    plic_swept_slab_volume,
)
from .domain import PlicDomain
from .curvature import (
    PlicCurvatureDiagnostics,
    PlicCurvatureEstimator,
    PlicCurvatureEstimator3D,
    PlicCurvatureState,
)
from .geometry import PlicGeometryDiagnostics, PlicGeometryReconstructor, PlicGeometryState
from .geometry_reference import (
    PlicGeometryFields,
    plane_interface_area,
    plane_volume_fraction,
    plic_plane_offset,
    reconstruct_plic_geometry,
    youngs_interface_normal,
)
from .transport import (
    PlicConservativeTransportDiagnostics,
    PlicGeometricTransport,
    PlicTransportDiagnostics,
    PlicVolumeMassTransportDiagnostics,
)
from .surface_tension import (
    PlicCapillaryWallDiagnostics,
    PlicCapillaryWallStepper,
    PlicSurfaceTension,
    PlicSurfaceTensionDiagnostics,
)

_LAZY_EXPORTS = {
    "CourantProjectionResult": (".courant_reference", "CourantProjectionResult"),
    "FaceCourantFields": (".courant_reference", "FaceCourantFields"),
    "active_divergence": (".courant_reference", "active_divergence"),
    "build_face_courant_reference": (
        ".courant_reference",
        "build_face_courant_reference",
    ),
    "project_face_courant_reference": (
        ".courant_reference",
        "project_face_courant_reference",
    ),
    "CourantProjectionDiagnostics": (
        ".courant",
        "CourantProjectionDiagnostics",
    ),
    "FaceCourantDiagnostics": (".courant", "FaceCourantDiagnostics"),
    "HomeCourantProjector": (".courant", "HomeCourantProjector"),
    "HomeFaceCourantBuilder": (".courant", "HomeFaceCourantBuilder"),
    "PlicTopologyDiagnostics": (
        ".topology_reference",
        "PlicTopologyDiagnostics",
    ),
    "PlicTopologyResult": (".topology_reference", "PlicTopologyResult"),
    "resolve_plic_topology_reference": (
        ".topology_reference",
        "resolve_plic_topology_reference",
    ),
    "PlicTopologyResolver": (".topology", "PlicTopologyResolver"),
    "GeometricHomeFslDiagnostics": (".stepper", "GeometricHomeFslDiagnostics"),
    "GeometricHomeFslStepper": (".stepper", "GeometricHomeFslStepper"),
    "ProjectedVelocityGeometricHomeFslStepper": (
        ".stepper",
        "ProjectedVelocityGeometricHomeFslStepper",
    ),
    "GvofFslWallDiagnostics": (".gvof_stepper", "GvofFslWallDiagnostics"),
    "GvofFslWallStepper": (".gvof_stepper", "GvofFslWallStepper"),
    "ProjectedGvofFslWallDiagnostics": (
        ".projected_gvof_stepper",
        "ProjectedGvofFslWallDiagnostics",
    ),
    "ProjectedGvofFslWallStepper": (
        ".projected_gvof_stepper",
        "ProjectedGvofFslWallStepper",
    ),
    "ProjectedGeometricFslDiagnostics": (
        ".projected_geometric_stepper",
        "ProjectedGeometricFslDiagnostics",
    ),
    "ProjectedGeometricFslStepper": (
        ".projected_geometric_stepper",
        "ProjectedGeometricFslStepper",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "CourantProjectionDiagnostics",
    "CourantProjectionResult",
    "FaceCourantDiagnostics",
    "FaceCourantFields",
    "GeometricHomeFslDiagnostics",
    "GeometricHomeFslStepper",
    "GvofFslWallDiagnostics",
    "GvofFslWallStepper",
    "HomeCourantProjector",
    "HomeFaceCourantBuilder",
    "PlicAxisAdvectionDiagnostics",
    "PlicAxisAdvectionResult",
    "PlicAxisAdvector",
    "PlicCapillaryWallDiagnostics",
    "PlicCapillaryWallStepper",
    "PlicConservativeTransportDiagnostics",
    "PlicCurvatureDiagnostics",
    "PlicCurvatureEstimator",
    "PlicCurvatureEstimator3D",
    "PlicCurvatureState",
    "PlicDomain",
    "PlicGeometricTransport",
    "PlicGeometryDiagnostics",
    "PlicGeometryFields",
    "PlicGeometryReconstructor",
    "PlicGeometryState",
    "PlicTopologyDiagnostics",
    "PlicTopologyResolver",
    "PlicTopologyResult",
    "PlicTransportDiagnostics",
    "PlicSurfaceTension",
    "PlicSurfaceTensionDiagnostics",
    "PlicVolumeMassTransportDiagnostics",
    "ProjectedGvofFslWallDiagnostics",
    "ProjectedGvofFslWallStepper",
    "ProjectedGeometricFslDiagnostics",
    "ProjectedGeometricFslStepper",
    "ProjectedVelocityGeometricHomeFslStepper",
    "active_divergence",
    "advect_plic_axis",
    "build_face_courant_reference",
    "plane_interface_area",
    "plane_volume_fraction",
    "plic_plane_offset",
    "plic_swept_slab_volume",
    "project_face_courant_reference",
    "reconstruct_plic_geometry",
    "resolve_plic_topology_reference",
    "youngs_interface_normal",
]
