# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Sharp volume-of-fluid state for HOME-FREE LBM."""

from .advection import HomeFreeMassAdvector
from .domain import HomeFreeDomain as HomeFreeLegacyDomain, HomeFreeDomainState
from .courant_projection import (
    CourantProjectionReference,
    HomeFreeCourantProjectionDiagnostics,
    HomeFreeCourantProjectionResult,
    HomeFreeCourantProjector,
    project_face_courant_reference,
)
from .flags import HomeFreeCellFlag, HomeFreeTransition
from .fsl_transition import (
    FslActiveTransitionDiagnostics,
    FslActiveTransitionReference,
    FslActiveTransitionRemapper,
    remap_fsl_active_moments,
)
from .fsl_stepper import (
    HomeFreeFslResearchStepDiagnostics,
    HomeFreeFslResearchStepper,
)
from .fsl_courant import (
    FslFaceCourantDiagnostics,
    FslFaceCourantReference,
    HomeFreeFslFaceCourantBuilder,
    fsl_face_courant,
)
from .fsl_stress import (
    FslBulkStrainDiagnostics,
    FslBulkStrainReference,
    FslNormalStrainReference,
    FslStressClosureDiagnostics,
    HomeFreeFslBulkStrainBuilder,
    HomeFreeFslStressClosure,
    fsl_normal_strain_targets,
    reconstruct_fsl_bulk_strain,
)
from .geometric_advection import (
    HomeFreeGeometricAxisAdvector,
    PlicAxisAdvectionDiagnostics,
    PlicAxisAdvectionResult,
    advect_plic_axis,
    plic_swept_slab_volume,
)
from .geometric_domain import HomeFreeGeometricDomain
from .backend import HomeFreeBackend, create_home_free_domain

# Public default after M5 acceptance. The legacy implementation remains
# explicitly addressable for one compatibility release.
HomeFreeDomain = HomeFreeGeometricDomain
from .geometric_queue import (
    HomeFreeGeometricQueueDiagnostics,
    HomeFreeGeometricQueueMaterializer,
)
from .geometric_topology import (
    GeometricPlicTopologyDiagnostics,
    GeometricPlicTopologyReference,
    HomeFreeGeometricTopologyResolver,
    resolve_geometric_plic_topology,
)
from .geometric_transport import (
    HomeFreeGeometricTransport,
    HomeFreeGeometricTransportResult,
)
from .geometry import HomeFreeGeometryDiagnostics, HomeFreeInterfaceGeometry
from .om_stepper import (
    HomeFreeOmResearchStepDiagnostics,
    HomeFreeOmResearchStepper,
    HomeFreeOmTransactionHistory,
)
from .plic import (
    PlicCurvatureFit,
    PlicLinkIntersection,
    PlicFslLinkCoverage,
    PlicLinkPlaneOwner,
    PlicPullLinkCoverage,
    PlicPullLinkStatus,
    fit_plic_curvature,
    plane_volume_fraction,
    plane_interface_area,
    plic_plane_offset,
    plic_pull_link_intersection,
    plic_pull_link_coverage,
    plic_fsl_link_coverage,
    contact_angle_interface_normal,
    youngs_interface_normal,
)
from .reference import (
    FslBoundaryCoefficients,
    FslStreamResult,
    HomeFreeStateDiagnostics,
    OnlyMissingBoundaryResidual,
    advect_internal_link_mass_momentum,
    interface_home_convective_density_momentum_increment,
    advect_mass_momentum_remap,
    apply_topology_transitions,
    classify_fill_levels,
    classify_topology_transitions,
    discrete_capillary_pressure_link_momentum,
    fsl_boundary_coefficients,
    fsl_boundary_population,
    fsl_extrapolated_boundary_velocity,
    fsl_extrapolated_boundary_strain,
    fsl_stream_moments,
    gas_pressure_boundary_populations,
    home_internal_link_momentum,
    link_mass_delta,
    normalize_advected_mass_momentum_and_excess,
    normalize_mass_and_excess,
    normalize_mass_momentum_and_excess,
    only_missing_boundary_residual,
    plic_capillary_pressure_momentum,
    reference_pressure_link_momentum,
    represented_liquid_momentum,
    validate_state_fields,
)
from .rigid_interface import (
    HomeFreeRigidDynamicStepDiagnostics,
    HomeFreeRigidInterface,
    HomeFreeRigidLoadDiagnostics,
    HomeFreeRigidPrepareDiagnostics,
    HomeFreeRigidStaticStepDiagnostics,
)
from .rigid_coupling import HomeFreeRigidCoupling, HomeFreeRigidStepDiagnostics
from .rigid_transition_reference import (
    HomeFreeRigidPhaseRemap,
    classify_moving_solid_phases,
)
from .rigid_transition import (
    HomeFreeRigidTransitionDiagnostics,
    HomeFreeRigidTransitionRemapper,
)
from .state import HomeFreeState
from .surface_tension import (
    HomeFreeSurfaceTension,
    HomeFreeSurfaceTensionDiagnostics,
)
from .topology import HomeFreeTopologyUpdater

__all__ = [
    "HomeFreeCellFlag",
    "HomeFreeCourantProjectionDiagnostics",
    "HomeFreeCourantProjectionResult",
    "HomeFreeCourantProjector",
    "HomeFreeDomain",
    "HomeFreeDomainState",
    "HomeFreeLegacyDomain",
    "HomeFreeBackend",
    "create_home_free_domain",
    "HomeFreeGeometryDiagnostics",
    "HomeFreeGeometricAxisAdvector",
    "HomeFreeGeometricDomain",
    "HomeFreeGeometricQueueDiagnostics",
    "HomeFreeGeometricQueueMaterializer",
    "HomeFreeGeometricTransport",
    "HomeFreeGeometricTransportResult",
    "HomeFreeGeometricTopologyResolver",
    "HomeFreeFslResearchStepDiagnostics",
    "HomeFreeFslResearchStepper",
    "HomeFreeFslFaceCourantBuilder",
    "HomeFreeFslBulkStrainBuilder",
    "HomeFreeFslStressClosure",
    "HomeFreeInterfaceGeometry",
    "HomeFreeMassAdvector",
    "HomeFreeOmResearchStepDiagnostics",
    "HomeFreeOmResearchStepper",
    "HomeFreeOmTransactionHistory",
    "HomeFreeRigidInterface",
    "HomeFreeRigidDynamicStepDiagnostics",
    "HomeFreeRigidCoupling",
    "HomeFreeRigidLoadDiagnostics",
    "HomeFreeRigidPrepareDiagnostics",
    "HomeFreeRigidPhaseRemap",
    "HomeFreeRigidStaticStepDiagnostics",
    "HomeFreeRigidStepDiagnostics",
    "HomeFreeRigidTransitionDiagnostics",
    "HomeFreeRigidTransitionRemapper",
    "HomeFreeTransition",
    "FslActiveTransitionDiagnostics",
    "FslActiveTransitionReference",
    "FslActiveTransitionRemapper",
    "HomeFreeState",
    "HomeFreeStateDiagnostics",
    "FslBoundaryCoefficients",
    "FslFaceCourantDiagnostics",
    "FslFaceCourantReference",
    "FslBulkStrainDiagnostics",
    "FslBulkStrainReference",
    "FslNormalStrainReference",
    "FslStressClosureDiagnostics",
    "FslStreamResult",
    "CourantProjectionReference",
    "GeometricPlicTopologyDiagnostics",
    "GeometricPlicTopologyReference",
    "OnlyMissingBoundaryResidual",
    "HomeFreeSurfaceTension",
    "HomeFreeSurfaceTensionDiagnostics",
    "HomeFreeTopologyUpdater",
    "PlicCurvatureFit",
    "PlicAxisAdvectionDiagnostics",
    "PlicAxisAdvectionResult",
    "PlicLinkIntersection",
    "PlicFslLinkCoverage",
    "PlicLinkPlaneOwner",
    "PlicPullLinkCoverage",
    "PlicPullLinkStatus",
    "advect_internal_link_mass_momentum",
    "advect_plic_axis",
    "interface_home_convective_density_momentum_increment",
    "advect_mass_momentum_remap",
    "classify_fill_levels",
    "classify_moving_solid_phases",
    "contact_angle_interface_normal",
    "classify_topology_transitions",
    "discrete_capillary_pressure_link_momentum",
    "fsl_boundary_coefficients",
    "fsl_face_courant",
    "fsl_normal_strain_targets",
    "fsl_boundary_population",
    "fsl_extrapolated_boundary_velocity",
    "fsl_extrapolated_boundary_strain",
    "fsl_stream_moments",
    "gas_pressure_boundary_populations",
    "home_internal_link_momentum",
    "fit_plic_curvature",
    "apply_topology_transitions",
    "link_mass_delta",
    "normalize_advected_mass_momentum_and_excess",
    "normalize_mass_and_excess",
    "normalize_mass_momentum_and_excess",
    "only_missing_boundary_residual",
    "plic_capillary_pressure_momentum",
    "reference_pressure_link_momentum",
    "remap_fsl_active_moments",
    "represented_liquid_momentum",
    "reconstruct_fsl_bulk_strain",
    "resolve_geometric_plic_topology",
    "plane_volume_fraction",
    "project_face_courant_reference",
    "plane_interface_area",
    "plic_plane_offset",
    "plic_swept_slab_volume",
    "plic_pull_link_intersection",
    "plic_pull_link_coverage",
    "plic_fsl_link_coverage",
    "validate_state_fields",
    "youngs_interface_normal",
]
