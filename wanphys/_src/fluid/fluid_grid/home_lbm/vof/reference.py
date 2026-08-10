# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy specification for the committed sharp-VOF state contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from ..reference import (
    equilibrium_convective_populations,
    extract_moments,
    reconstruct_populations,
)
from .flags import HomeFreeCellFlag, HomeFreeTransition
from .plic import (
    PlicFslLinkCoverage,
    PlicLinkPlaneOwner,
    PlicPullLinkStatus,
    plane_interface_area,
)


@dataclass(frozen=True)
class HomeFreeStateDiagnostics:
    liquid_cells: int
    interface_cells: int
    gas_cells: int
    solid_cells: int
    direct_liquid_gas_links: int
    minimum_fill: float
    maximum_fill: float
    liquid_mass: float


@dataclass(frozen=True)
class OnlyMissingBoundaryResidual:
    """Exact density/momentum residual of one OM free-surface pull step.

    Every field except ``gas_link_count`` has shape ``grid + (4,)`` and stores
    ``rho, jx, jy, jz``.  ``pressure`` contains isotropic active-link pressure
    transport and the prescribed gas-pressure jump.  ``equilibrium_transport``
    contains the remaining equilibrium-population transport, including any
    kinetic closure mismatch at a gas link.  ``non_equilibrium`` contains the
    transported HOME stress remainder.  Their sum is ``total``.
    """

    pressure: np.ndarray
    equilibrium_transport: np.ndarray
    non_equilibrium: np.ndarray
    total: np.ndarray
    gas_link_count: np.ndarray


@dataclass(frozen=True)
class FslBoundaryCoefficients:
    """Bogner multi-reflection coefficients in WanPhys relaxation notation."""

    a0: float
    opposite_a0: float
    a1: float
    nonequilibrium: float
    strain: float


@dataclass(frozen=True)
class FslStreamResult:
    """One force-free FSL pull stream before HOME collision."""

    moments: np.ndarray
    populations: np.ndarray
    boundary_link_count: int
    exact_link_count: int
    extrapolated_link_count: int
    only_missing_fallback_count: int


def fsl_boundary_coefficients(
    fraction: float,
    shear_omega: float,
    weight: float,
) -> FslBoundaryCoefficients:
    """Return Bogner Eq. (25)/Table 1 coefficients for one FSL link.

    Bogner uses negative collision rates ``lambda_plus``.  WanPhys stores the
    positive relaxation rate ``omega_plus=-lambda_plus``; consequently the
    local even non-equilibrium coefficient is
    ``omega_plus*(3/2-fraction)``.  The strain coefficient includes the D3Q27
    factor ``w_q/c_s^2 = 3*w_q``.
    """

    delta = float(fraction)
    omega = float(shear_omega)
    lattice_weight = float(weight)
    if not np.isfinite(delta) or not 0.0 <= delta <= 1.0:
        raise ValueError("FSL boundary fraction must be finite and in [0, 1]")
    if not np.isfinite(omega) or not 0.0 < omega < 2.0:
        raise ValueError("FSL shear_omega must be finite and in (0, 2)")
    if not np.isfinite(lattice_weight) or lattice_weight <= 0.0:
        raise ValueError("FSL lattice weight must be finite and positive")
    lambda_plus = 1.0 / omega - 0.5
    return FslBoundaryCoefficients(
        a0=0.5 - delta,
        opposite_a0=0.5,
        a1=delta - 1.0,
        nonequilibrium=omega * (1.5 - delta),
        strain=-3.0 * lambda_plus * lattice_weight,
    )


def fsl_boundary_population(
    local_moments: np.ndarray,
    interior_moments: np.ndarray,
    incoming_direction: int,
    fraction: float,
    boundary_density: float,
    shear_omega: float,
    *,
    boundary_velocity: np.ndarray | None = None,
    boundary_strain: np.ndarray | None = None,
) -> float:
    """Evaluate Bogner's second-order FSL closure for one incoming PDF.

    ``incoming_direction`` follows the WanPhys pull convention: its source is
    on the gas side at ``-c_q``.  Bogner's outward boundary direction is thus
    ``c_qbar=-c_q``.  ``interior_moments`` belongs to the support node at
    ``+c_q``.  Both moment vectors represent post-collision HOME states.

    ``boundary_strain`` is the symmetric momentum shear-rate tensor ``S_b``
    from Bogner Eq. (21).  Passing no tensor selects the paper's documented
    simplified free-surface condition ``D=0``; stress extrapolation is kept out
    of this link oracle deliberately.
    """

    local = np.asarray(local_moments, dtype=np.float64)
    interior = np.asarray(interior_moments, dtype=np.float64)
    if local.shape != (10,) or interior.shape != (10,):
        raise ValueError("FSL local and interior moments must each have shape (10,)")
    if (
        not np.isfinite(local).all()
        or not np.isfinite(interior).all()
        or local[0] <= 0.0
        or interior[0] <= 0.0
    ):
        raise ValueError("FSL moment vectors must be finite with positive density")
    q = int(incoming_direction)
    if q <= 0 or q >= 27 or q != incoming_direction:
        raise ValueError("incoming_direction must be a non-rest D3Q27 index")
    rho_boundary = float(boundary_density)
    if not np.isfinite(rho_boundary) or rho_boundary <= 0.0:
        raise ValueError("FSL boundary_density must be finite and positive")

    if boundary_velocity is None:
        velocity = local[1:4] / local[0]
    else:
        velocity = np.asarray(boundary_velocity, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("FSL boundary_velocity must contain three finite values")
    if boundary_strain is None:
        strain = np.zeros((3, 3), dtype=np.float64)
    else:
        strain = np.asarray(boundary_strain, dtype=np.float64)
        if strain.shape != (3, 3) or not np.isfinite(strain).all():
            raise ValueError("FSL boundary_strain must be a finite 3x3 tensor")
        if not np.allclose(strain, strain.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("FSL boundary_strain must be symmetric")

    opposite = int(D3Q27_OPPOSITE[q])
    c = D3Q27_DIRECTIONS[q]
    weight = float(D3Q27_WEIGHTS[q])
    coefficients = fsl_boundary_coefficients(fraction, shear_omega, weight)
    local_populations = reconstruct_populations(local)
    interior_populations = reconstruct_populations(interior)

    local_velocity = local[1:4] / local[0]
    local_cu = float(np.dot(c, local_velocity))
    local_even_equilibrium = weight * local[0] * (
        1.0 + 4.5 * local_cu * local_cu - 1.5 * np.dot(local_velocity, local_velocity)
    )
    local_even_nonequilibrium = (
        0.5 * (local_populations[q] + local_populations[opposite])
        - local_even_equilibrium
    )

    boundary_cu = float(np.dot(c, velocity))
    boundary_even_equilibrium = weight * rho_boundary * (
        1.0 + 4.5 * boundary_cu * boundary_cu - 1.5 * np.dot(velocity, velocity)
    )
    strain_contraction = float(c @ strain @ c)
    return float(
        coefficients.a0 * local_populations[opposite]
        + coefficients.opposite_a0 * local_populations[q]
        + coefficients.a1 * interior_populations[opposite]
        + coefficients.nonequilibrium * local_even_nonequilibrium
        + boundary_even_equilibrium
        + coefficients.strain * strain_contraction
    )


def fsl_extrapolated_boundary_velocity(
    moments: np.ndarray,
    coverage: PlicFslLinkCoverage,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    no_support_policy: str = "error",
) -> np.ndarray:
    """Linearly extrapolate fluid velocity to each PLIC boundary point."""

    values = np.asarray(moments, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("FSL moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    active = np.asarray(coverage.hydrodynamic_active, dtype=bool)
    if active.shape != shape or np.asarray(coverage.status).shape != shape + (27,):
        raise ValueError("FSL coverage must match the HOME moment grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if no_support_policy not in ("error", "only_missing"):
        raise ValueError("FSL no_support_policy must be 'error' or 'only_missing'")
    if not np.isfinite(values[active]).all() or np.any(values[..., 0][active] <= 0.0):
        raise ValueError("active FSL moments must be finite with positive density")

    velocity = np.zeros(shape + (3,), dtype=np.float64)
    velocity[active] = values[..., 1:4][active] / values[..., 0, None][active]
    boundary_velocity = np.full(shape + (27, 3), np.nan, dtype=np.float64)
    directions = D3Q27_DIRECTIONS.astype(np.int32)

    def neighbor(index: tuple[int, int, int], displacement: np.ndarray) -> tuple[int, int, int]:
        result = [index[axis] + int(displacement[axis]) for axis in range(3)]
        for axis in range(3):
            if result[axis] < 0 or result[axis] >= shape[axis]:
                if periodic[axis]:
                    result[axis] %= shape[axis]
                else:
                    raise ValueError(
                        "FSL velocity extrapolation encountered a non-periodic support link"
                    )
        return tuple(result)

    for index in map(tuple, np.argwhere(active)):
        for q in range(1, 27):
            link_status = PlicPullLinkStatus(int(coverage.status[index + (q,)]))
            if link_status in (
                PlicPullLinkStatus.VALID,
                PlicPullLinkStatus.EXTRAPOLATED,
            ):
                support_index = neighbor(index, directions[q])
                if not active[support_index]:
                    raise RuntimeError(
                        "FSL coverage marked a velocity-extrapolation link usable without support"
                    )
                delta = float(coverage.fraction[index + (q,)])
                boundary_velocity[index + (q,)] = (
                    (1.0 + delta) * velocity[index]
                    - delta * velocity[support_index]
                )
            elif link_status == PlicPullLinkStatus.NO_SUPPORT:
                if no_support_policy == "error":
                    raise ValueError(
                        "FSL velocity extrapolation encountered NO_SUPPORT"
                    )
                boundary_velocity[index + (q,)] = velocity[index]
    return boundary_velocity


def fsl_extrapolated_boundary_strain(
    bulk_strain: np.ndarray,
    interface_normal: np.ndarray,
    coverage: PlicFslLinkCoverage,
    normal_strain: float | np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    no_support_policy: str = "error",
) -> np.ndarray:
    """Extrapolate and project the full Bogner free-surface strain tensor.

    Tangential-normal components are set to zero, tangential-tangential
    components remain extrapolated from the bulk, and ``normal_strain`` sets
    ``n dot S_b dot n`` explicitly according to the caller's Eq. (2a) model.
    """

    strain = np.asarray(bulk_strain, dtype=np.float64)
    normals = np.asarray(interface_normal, dtype=np.float64)
    if strain.ndim != 5 or strain.shape[-2:] != (3, 3):
        raise ValueError("FSL bulk_strain must have grid shape + (3, 3)")
    shape = strain.shape[:-2]
    if normals.shape != shape + (3,):
        raise ValueError("FSL interface_normal must have grid shape + (3,)")
    active = np.asarray(coverage.hydrodynamic_active, dtype=bool)
    if active.shape != shape or np.asarray(coverage.status).shape != shape + (27,):
        raise ValueError("FSL coverage must match the bulk strain grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if no_support_policy not in ("error", "only_missing"):
        raise ValueError("FSL no_support_policy must be 'error' or 'only_missing'")
    if not np.isfinite(strain[active]).all() or not np.allclose(
        strain[active],
        np.swapaxes(strain[active], -1, -2),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("active FSL bulk_strain tensors must be finite and symmetric")
    target_normal_strain = np.asarray(normal_strain, dtype=np.float64)
    try:
        target_normal_strain = np.broadcast_to(target_normal_strain, shape + (27,))
    except ValueError as error:
        raise ValueError("FSL normal_strain must broadcast to grid shape + (27,)") from error

    boundary_strain = np.full(shape + (27, 3, 3), np.nan, dtype=np.float64)
    directions = D3Q27_DIRECTIONS.astype(np.int32)

    def neighbor(index: tuple[int, int, int], displacement: np.ndarray) -> tuple[int, int, int]:
        result = [index[axis] + int(displacement[axis]) for axis in range(3)]
        for axis in range(3):
            if result[axis] < 0 or result[axis] >= shape[axis]:
                if periodic[axis]:
                    result[axis] %= shape[axis]
                else:
                    raise ValueError(
                        "FSL strain extrapolation encountered a non-periodic support link"
                    )
        return tuple(result)

    for index in map(tuple, np.argwhere(active)):
        for q in range(1, 27):
            link_status = PlicPullLinkStatus(int(coverage.status[index + (q,)]))
            if link_status == PlicPullLinkStatus.NO_SUPPORT:
                if no_support_policy == "error":
                    raise ValueError("FSL strain extrapolation encountered NO_SUPPORT")
                boundary_strain[index + (q,)] = np.zeros((3, 3), dtype=np.float64)
                continue
            if link_status not in (
                PlicPullLinkStatus.VALID,
                PlicPullLinkStatus.EXTRAPOLATED,
            ):
                continue
            support_index = neighbor(index, directions[q])
            if not active[support_index]:
                raise RuntimeError(
                    "FSL coverage marked a strain-extrapolation link usable without support"
                )
            owner = PlicLinkPlaneOwner(int(coverage.plane_owner[index + (q,)]))
            source_index = neighbor(index, -directions[q])
            owner_index = index if owner == PlicLinkPlaneOwner.DESTINATION else source_index
            n = normals[owner_index]
            magnitude = float(np.linalg.norm(n))
            target = float(target_normal_strain[index + (q,)])
            if not np.isfinite(n).all() or magnitude <= 0.0 or not np.isfinite(target):
                raise ValueError("FSL boundary strain requires finite owner normals and targets")
            n = n / magnitude
            delta = float(coverage.fraction[index + (q,)])
            extrapolated = (
                (1.0 + delta) * strain[index] - delta * strain[support_index]
            )
            normal_vector = extrapolated @ n
            normal_component = float(n @ normal_vector)
            tangent_vector = normal_vector - normal_component * n
            boundary_strain[index + (q,)] = (
                extrapolated
                - np.outer(n, tangent_vector)
                - np.outer(tangent_vector, n)
                + (target - normal_component) * np.outer(n, n)
            )
    return boundary_strain


def fsl_stream_moments(
    moments: np.ndarray,
    flags: np.ndarray,
    coverage: PlicFslLinkCoverage,
    boundary_density: float | np.ndarray,
    boundary_velocity: np.ndarray,
    shear_omega: float,
    *,
    boundary_strain: np.ndarray | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    no_support_policy: str = "error",
) -> FslStreamResult:
    """Apply the owner-aware Bogner closure to one reference pull stream.

    The supplied velocity and optional strain fields live on PLIC owner cells.
    This function does not invent an interface extrapolation rule.  It also
    requires the interior support node ``x_b+c_q`` used by Eq. (11) to be an
    active hydrodynamic node.
    """

    values = np.asarray(moments, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("FSL moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if cell_flags.shape != shape:
        raise ValueError("FSL flags must match the HOME moment grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if no_support_policy not in ("error", "only_missing"):
        raise ValueError("FSL no_support_policy must be 'error' or 'only_missing'")
    active = np.asarray(coverage.hydrodynamic_active, dtype=bool)
    if active.shape != shape:
        raise ValueError("FSL coverage active mask must match the HOME grid")
    for field_name, field in (
        ("fraction", coverage.fraction),
        ("status", coverage.status),
        ("plane_owner", coverage.plane_owner),
    ):
        if np.asarray(field).shape != shape + (27,):
            raise ValueError(f"FSL coverage {field_name} must have grid shape + (27,)")
    if not np.isfinite(values[active]).all() or np.any(values[..., 0][active] <= 0.0):
        raise ValueError("active FSL moments must be finite with positive density")

    prescribed_density = np.asarray(boundary_density, dtype=np.float64)
    try:
        prescribed_density = np.broadcast_to(prescribed_density, shape)
    except ValueError as error:
        raise ValueError("FSL boundary_density must broadcast to the HOME grid") from error
    if not np.isfinite(prescribed_density).all() or np.any(prescribed_density <= 0.0):
        raise ValueError("FSL boundary_density must be finite and positive")
    prescribed_velocity = np.asarray(boundary_velocity, dtype=np.float64)
    if prescribed_velocity.shape != shape + (27, 3):
        raise ValueError("FSL boundary_velocity must have grid shape + (27, 3)")
    if boundary_strain is None:
        prescribed_strain = np.zeros(shape + (27, 3, 3), dtype=np.float64)
    else:
        prescribed_strain = np.asarray(boundary_strain, dtype=np.float64)
        if prescribed_strain.shape != shape + (27, 3, 3):
            raise ValueError("FSL boundary_strain must have grid shape + (27, 3, 3)")

    reconstructed = reconstruct_populations(values)
    streamed = np.full(shape + (27,), np.nan, dtype=np.float64)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    boundary_links = 0
    exact_links = 0
    extrapolated_links = 0
    only_missing_fallbacks = 0

    def neighbor(index: tuple[int, int, int], displacement: np.ndarray) -> tuple[int, int, int]:
        result = [index[axis] + int(displacement[axis]) for axis in range(3)]
        for axis in range(3):
            if result[axis] < 0 or result[axis] >= shape[axis]:
                if periodic[axis]:
                    result[axis] %= shape[axis]
                else:
                    raise ValueError("FSL stream encountered a non-periodic domain link")
        return tuple(result)

    for index in map(tuple, np.argwhere(active)):
        for q, c in enumerate(directions):
            source_index = neighbor(index, -c)
            if active[source_index]:
                streamed[index + (q,)] = reconstructed[source_index + (q,)]
                continue
            if int(cell_flags[source_index]) == int(HomeFreeCellFlag.SOLID):
                raise ValueError("FSL stream encountered a solid-owned link")
            link_status = PlicPullLinkStatus(int(coverage.status[index + (q,)]))
            if link_status not in (
                PlicPullLinkStatus.VALID,
                PlicPullLinkStatus.EXTRAPOLATED,
                PlicPullLinkStatus.NO_SUPPORT,
            ):
                raise ValueError(
                    f"FSL boundary link has unusable geometry status {link_status.name}"
                )
            owner = PlicLinkPlaneOwner(int(coverage.plane_owner[index + (q,)]))
            if owner == PlicLinkPlaneOwner.DESTINATION:
                owner_index = index
            elif owner == PlicLinkPlaneOwner.SOURCE:
                owner_index = source_index
            else:
                raise ValueError("FSL boundary link has no unique PLIC plane owner")
            if link_status == PlicPullLinkStatus.NO_SUPPORT:
                if no_support_policy == "error":
                    raise ValueError(
                        "FSL boundary link has unusable geometry status NO_SUPPORT"
                    )
                opposite = int(D3Q27_OPPOSITE[q])
                velocity = prescribed_velocity[index + (q,)]
                cu = float(np.dot(c, velocity))
                even_equilibrium = D3Q27_WEIGHTS[q] * prescribed_density[
                    owner_index
                ] * (
                    1.0
                    + 4.5 * cu * cu
                    - 1.5 * float(np.dot(velocity, velocity))
                )
                streamed[index + (q,)] = (
                    2.0 * even_equilibrium
                    - reconstructed[index + (opposite,)]
                )
                boundary_links += 1
                only_missing_fallbacks += 1
                continue
            support_index = neighbor(index, c)
            if not active[support_index]:
                raise RuntimeError("FSL coverage marked a link usable without support")
            streamed[index + (q,)] = fsl_boundary_population(
                values[index],
                values[support_index],
                q,
                float(coverage.fraction[index + (q,)]),
                float(prescribed_density[owner_index]),
                shear_omega,
                boundary_velocity=prescribed_velocity[index + (q,)],
                boundary_strain=prescribed_strain[index + (q,)],
            )
            boundary_links += 1
            if link_status == PlicPullLinkStatus.VALID:
                exact_links += 1
            else:
                extrapolated_links += 1

    result = values.copy()
    result[active] = extract_moments(streamed[active])
    return FslStreamResult(
        moments=result,
        populations=streamed,
        boundary_link_count=boundary_links,
        exact_link_count=exact_links,
        extrapolated_link_count=extrapolated_links,
        only_missing_fallback_count=only_missing_fallbacks,
    )


def gas_pressure_boundary_populations(
    interface_moments: np.ndarray,
    gas_density: float | np.ndarray,
) -> np.ndarray:
    """Reconstruct all potentially missing gas-side populations with Eq. (11).

    The caller selects only directions whose pull source is gas. The opposite
    filtered population is reconstructed from the interface cell's ten HOME
    moments; the prescribed gas pressure is represented by ``gas_density``.
    """

    moments = np.asarray(interface_moments, dtype=np.float64)
    if moments.shape[-1] != 10:
        raise ValueError("interface_moments must have a trailing dimension of 10")
    rho = moments[..., 0]
    if not np.isfinite(moments).all() or np.any(rho <= 0.0):
        raise ValueError("interface moments must be finite with positive density")
    prescribed_rho = np.asarray(gas_density, dtype=np.float64)
    try:
        prescribed_rho = np.broadcast_to(prescribed_rho, rho.shape)
    except ValueError as error:
        raise ValueError("gas_density must broadcast to the interface grid") from error
    if not np.isfinite(prescribed_rho).all() or np.any(prescribed_rho <= 0.0):
        raise ValueError("gas_density must be finite and positive")

    velocity = moments[..., 1:4] / rho[..., None]
    cu = np.einsum("...a,qa->...q", velocity, D3Q27_DIRECTIONS)
    speed_squared = np.sum(velocity * velocity, axis=-1)
    equilibrium = D3Q27_WEIGHTS * prescribed_rho[..., None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared[..., None]
    )
    filtered = reconstruct_populations(moments)
    return equilibrium + equilibrium[..., D3Q27_OPPOSITE] - filtered[..., D3Q27_OPPOSITE]


def only_missing_boundary_residual(
    moments: np.ndarray,
    flags: np.ndarray,
    gas_density: float | np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> OnlyMissingBoundaryResidual:
    """Decompose one force-free OM stream into exact ``rho/j`` residuals.

    The oracle models the production free-surface ownership rule: populations
    are pulled unchanged from active neighbors, while only populations whose
    pull source is gas are reconstructed with Eq. (11).  Collision is omitted
    because the force-free HOME collision conserves density and momentum.

    Solid links and non-periodic domain exits are deliberately rejected.  They
    belong to separate wall/cut-link ledgers and including them here would make
    a free-surface pressure residual ambiguous.
    """

    values = np.asarray(moments, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if cell_flags.shape != shape:
        raise ValueError("flags must match the HOME moment grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    legal = np.isin(cell_flags, [int(flag) for flag in HomeFreeCellFlag])
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")

    active = np.isin(
        cell_flags,
        (int(HomeFreeCellFlag.INTERFACE), int(HomeFreeCellFlag.LIQUID)),
    )
    if not np.isfinite(values[active]).all() or np.any(values[..., 0][active] <= 0.0):
        raise ValueError("active HOME moments must be finite with positive density")

    prescribed_rho = np.asarray(gas_density, dtype=np.float64)
    try:
        prescribed_rho = np.broadcast_to(prescribed_rho, shape)
    except ValueError as error:
        raise ValueError("gas_density must broadcast to the HOME grid") from error
    if not np.isfinite(prescribed_rho).all() or np.any(prescribed_rho <= 0.0):
        raise ValueError("gas_density must be finite and positive")

    # Gas/solid HOME moments are inactive and may contain stale values.  Give
    # them a harmless state so the vectorized population reconstruction only
    # imposes validity requirements on cells participating in this ledger.
    safe = values.copy()
    safe[~active] = 0.0
    safe[..., 0][~active] = 1.0
    rho = safe[..., 0]
    velocity = safe[..., 1:4] / rho[..., None]
    equilibrium_moments = np.zeros_like(safe)
    equilibrium_moments[..., 0] = rho
    equilibrium_moments[..., 1:4] = safe[..., 1:4]
    equilibrium_moments[..., 4] = rho * velocity[..., 0] * velocity[..., 0]
    equilibrium_moments[..., 5] = rho * velocity[..., 1] * velocity[..., 1]
    equilibrium_moments[..., 6] = rho * velocity[..., 2] * velocity[..., 2]
    equilibrium_moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    equilibrium_moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
    equilibrium_moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]

    populations = reconstruct_populations(safe)
    equilibrium = reconstruct_populations(equilibrium_moments)
    non_equilibrium = populations - equilibrium
    boundary = gas_pressure_boundary_populations(safe, prescribed_rho)
    boundary_equilibrium = gas_pressure_boundary_populations(
        equilibrium_moments, prescribed_rho
    )
    boundary_local_pressure = gas_pressure_boundary_populations(
        equilibrium_moments, rho
    )

    pressure_residual = np.zeros(shape + (4,), dtype=np.float64)
    equilibrium_residual = np.zeros_like(pressure_residual)
    non_equilibrium_residual = np.zeros_like(pressure_residual)
    total_residual = np.zeros_like(pressure_residual)
    gas_link_count = np.zeros(shape, dtype=np.int32)
    directions = D3Q27_DIRECTIONS.astype(np.int32)

    for index in np.ndindex(shape):
        if not active[index]:
            continue
        for q, c in enumerate(directions):
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        raise ValueError(
                            "OM pressure residual encountered a non-periodic domain link"
                        )
            source_index = tuple(source)
            source_flag = HomeFreeCellFlag(int(cell_flags[source_index]))
            if source_flag in (HomeFreeCellFlag.INTERFACE, HomeFreeCellFlag.LIQUID):
                pressure_delta = D3Q27_WEIGHTS[q] * (
                    rho[source_index] - rho[index]
                )
                equilibrium_delta = (
                    equilibrium[source_index + (q,)]
                    - D3Q27_WEIGHTS[q] * rho[source_index]
                    - equilibrium[index + (q,)]
                    + D3Q27_WEIGHTS[q] * rho[index]
                )
                non_equilibrium_delta = (
                    non_equilibrium[source_index + (q,)]
                    - non_equilibrium[index + (q,)]
                )
                direct_delta = (
                    populations[source_index + (q,)] - populations[index + (q,)]
                )
            elif source_flag == HomeFreeCellFlag.GAS:
                if int(cell_flags[index]) != int(HomeFreeCellFlag.INTERFACE):
                    raise ValueError("OM topology contains a direct liquid-gas link")
                gas_link_count[index] += 1
                pressure_delta = (
                    boundary_equilibrium[index + (q,)]
                    - boundary_local_pressure[index + (q,)]
                )
                equilibrium_delta = (
                    boundary_local_pressure[index + (q,)]
                    - equilibrium[index + (q,)]
                )
                non_equilibrium_delta = (
                    boundary[index + (q,)]
                    - boundary_equilibrium[index + (q,)]
                    - non_equilibrium[index + (q,)]
                )
                direct_delta = boundary[index + (q,)] - populations[index + (q,)]
            else:
                raise ValueError("OM pressure residual encountered a solid link")

            projection = np.empty(4, dtype=np.float64)
            projection[0] = 1.0
            projection[1:4] = c
            pressure_residual[index] += pressure_delta * projection
            equilibrium_residual[index] += equilibrium_delta * projection
            non_equilibrium_residual[index] += non_equilibrium_delta * projection
            total_residual[index] += direct_delta * projection

    return OnlyMissingBoundaryResidual(
        pressure=pressure_residual,
        equilibrium_transport=equilibrium_residual,
        non_equilibrium=non_equilibrium_residual,
        total=total_residual,
        gas_link_count=gas_link_count,
    )


def classify_topology_transitions(
    density: np.ndarray,
    advected_mass: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    fill_epsilon: float = 1.0e-4,
) -> np.ndarray:
    """Resolve HOME-FREE topology changes in three deterministic passes."""

    rho = np.asarray(density, dtype=np.float64)
    mass = np.asarray(advected_mass, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if not (rho.shape == mass.shape == cell_flags.shape) or rho.ndim != 3:
        raise ValueError("topology fields must be identically shaped 3D arrays")
    if not np.isfinite(rho).all() or not np.isfinite(mass).all():
        raise ValueError("topology density and mass must be finite")
    if not 0.0 <= fill_epsilon < 0.5:
        raise ValueError("fill_epsilon must be in [0, 0.5)")

    candidates = np.full(rho.shape, int(HomeFreeTransition.KEEP), dtype=np.int32)
    for index in np.ndindex(rho.shape):
        if int(cell_flags[index]) != HomeFreeCellFlag.INTERFACE:
            continue
        if rho[index] <= 0.0:
            raise ValueError("interface density must be positive")
        neighbor_flags = [
            int(cell_flags[neighbor])
            for neighbor in _neighbor_indices(index, rho.shape, periodic)
        ]
        has_liquid = int(HomeFreeCellFlag.LIQUID) in neighbor_flags
        has_gas = int(HomeFreeCellFlag.GAS) in neighbor_flags
        if mass[index] >= (1.0 + fill_epsilon) * rho[index] or not has_gas:
            candidates[index] = int(HomeFreeTransition.INTERFACE_TO_LIQUID)
        elif mass[index] <= -fill_epsilon * rho[index] or not has_liquid:
            candidates[index] = int(HomeFreeTransition.INTERFACE_TO_GAS)

    liquid_protected = candidates.copy()
    for index in np.ndindex(rho.shape):
        neighbor_transitions = [
            int(candidates[neighbor])
            for neighbor in _neighbor_indices(index, rho.shape, periodic)
        ]
        next_to_liquid_growth = int(HomeFreeTransition.INTERFACE_TO_LIQUID) in neighbor_transitions
        if int(cell_flags[index]) == HomeFreeCellFlag.GAS and next_to_liquid_growth:
            liquid_protected[index] = int(HomeFreeTransition.GAS_TO_INTERFACE)
        elif (
            int(candidates[index]) == HomeFreeTransition.INTERFACE_TO_GAS
            and next_to_liquid_growth
        ):
            liquid_protected[index] = int(HomeFreeTransition.KEEP)

    resolved = liquid_protected.copy()
    for index in np.ndindex(rho.shape):
        next_to_gas_growth = any(
            int(liquid_protected[neighbor]) == HomeFreeTransition.INTERFACE_TO_GAS
            for neighbor in _neighbor_indices(index, rho.shape, periodic)
        )
        if not next_to_gas_growth:
            continue
        if int(cell_flags[index]) == HomeFreeCellFlag.LIQUID:
            resolved[index] = int(HomeFreeTransition.LIQUID_TO_INTERFACE)
        elif int(liquid_protected[index]) == HomeFreeTransition.INTERFACE_TO_LIQUID:
            resolved[index] = int(HomeFreeTransition.KEEP)
    return resolved


def apply_topology_transitions(flags: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    """Map resolved scratch intents to stable cell flags."""

    result = np.asarray(flags, dtype=np.int32).copy()
    intents = np.asarray(transitions, dtype=np.int32)
    if result.shape != intents.shape:
        raise ValueError("flags and transitions must have identical shapes")
    result[intents == int(HomeFreeTransition.INTERFACE_TO_LIQUID)] = int(
        HomeFreeCellFlag.LIQUID
    )
    result[intents == int(HomeFreeTransition.INTERFACE_TO_GAS)] = int(
        HomeFreeCellFlag.GAS
    )
    result[intents == int(HomeFreeTransition.GAS_TO_INTERFACE)] = int(
        HomeFreeCellFlag.INTERFACE
    )
    result[intents == int(HomeFreeTransition.LIQUID_TO_INTERFACE)] = int(
        HomeFreeCellFlag.INTERFACE
    )
    return result


def link_mass_delta(
    cell_flag: HomeFreeCellFlag | int,
    cell_fill: float,
    neighbor_flags: np.ndarray,
    neighbor_fill: np.ndarray,
    incoming_populations: np.ndarray,
    outgoing_populations: np.ndarray,
) -> float:
    """Evaluate HOME-FREE Eqs. (9)-(10) for one cell's non-rest links.

    The population arrays must already be aligned by physical link: entry
    ``q`` is the population entering from that neighbor and the population
    leaving toward the same neighbor, respectively.
    """

    flags = np.asarray(neighbor_flags, dtype=np.int32)
    fill = np.asarray(neighbor_fill, dtype=np.float64)
    incoming = np.asarray(incoming_populations, dtype=np.float64)
    outgoing = np.asarray(outgoing_populations, dtype=np.float64)
    if not (flags.shape == fill.shape == incoming.shape == outgoing.shape):
        raise ValueError("all link arrays must have identical shapes")
    if not np.isfinite(cell_fill) or cell_fill < 0.0 or cell_fill > 1.0:
        raise ValueError("cell_fill must be finite and in [0, 1]")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("neighbor_fill must be finite and in [0, 1]")
    if not np.isfinite(incoming).all() or not np.isfinite(outgoing).all():
        raise ValueError("link populations must be finite")

    flag = HomeFreeCellFlag(int(cell_flag))
    if flag in (HomeFreeCellFlag.GAS, HomeFreeCellFlag.SOLID):
        return 0.0
    flux = incoming - outgoing
    if flag == HomeFreeCellFlag.LIQUID:
        active = (flags == int(HomeFreeCellFlag.LIQUID)) | (
            flags == int(HomeFreeCellFlag.INTERFACE)
        )
        return float(np.sum(flux[active], dtype=np.float64))

    weights = np.zeros(flags.shape, dtype=np.float64)
    weights[flags == int(HomeFreeCellFlag.LIQUID)] = 1.0
    interface = flags == int(HomeFreeCellFlag.INTERFACE)
    weights[interface] = 0.5 * (cell_fill + fill[interface])
    return float(np.sum(weights * flux, dtype=np.float64))


def advect_mass_momentum_remap(
    moments: np.ndarray,
    mass: np.ndarray,
    fill_level: np.ndarray,
    excess_mass: np.ndarray,
    excess_momentum: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analyze queue transport and the candidate interface momentum defect.

    The mass update is exactly HOME-Free Eqs. (9)-(10).  Full-cell material
    convection remains owned by HOME.  The returned incoming excess mass and
    momentum define the correction that is safe to commit after the HOME step.
    The final field is a donor-upwind interface-link candidate for analysis
    only: applying it directly duplicates HOME's material convection.  A state
    update must first subtract HOME's own discrete convective contribution.
    Pressure, viscous, body-force, and boundary impulses remain owned by the
    HOME collision/streaming step.  The paper stores only scalar liquid mass,
    so this is the project's conservative momentum extension.
    """

    home = np.asarray(moments, dtype=np.float64)
    liquid_mass = np.asarray(mass, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    excess = np.asarray(excess_mass, dtype=np.float64)
    excess_j = np.asarray(excess_momentum, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    shape = liquid_mass.shape
    if len(shape) != 3 or home.shape != shape + (10,):
        raise ValueError("moments must have mass.shape + (10,)")
    if excess_j.shape != shape + (3,):
        raise ValueError("excess_momentum must have mass.shape + (3,)")
    if any(field.shape != shape for field in (fill, excess, cell_flags)):
        raise ValueError("all scalar advection fields must have mass.shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not all(
        np.isfinite(field).all()
        for field in (home, liquid_mass, fill, excess, excess_j)
    ):
        raise ValueError("mass-momentum advection fields must be finite")
    legal = np.isin(cell_flags, [int(flag) for flag in HomeFreeCellFlag])
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")
    if np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain in [0, 1]")

    active = np.isin(
        cell_flags,
        (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
    )
    density = home[..., 0]
    if np.any(density[active] <= 0.0):
        raise ValueError("active cells require positive HOME density")
    velocity = np.zeros(shape + (3,), dtype=np.float64)
    velocity[active] = home[..., 1:4][active] / density[active, None]
    populations = reconstruct_populations(home)
    advected_mass = liquid_mass.copy()
    incoming_excess_mass = np.zeros(shape, dtype=np.float64)
    incoming_excess_momentum = np.zeros(shape + (3,), dtype=np.float64)
    interface_correction = np.zeros(shape + (3,), dtype=np.float64)

    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in np.ndindex(shape):
        cell_flag = HomeFreeCellFlag(int(cell_flags[index]))
        if cell_flag in (HomeFreeCellFlag.GAS, HomeFreeCellFlag.SOLID):
            continue
        for q in range(1, 27):
            source = [index[axis] - int(directions[q, axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid = False
            if not valid:
                continue
            source_index = tuple(source)
            advected_mass[index] += excess[source_index]
            incoming_excess_mass[index] += excess[source_index]
            incoming_excess_momentum[index] += excess_j[source_index]
            source_flag = HomeFreeCellFlag(int(cell_flags[source_index]))
            if source_flag not in (
                HomeFreeCellFlag.LIQUID,
                HomeFreeCellFlag.INTERFACE,
            ):
                continue
            exchange_weight = 1.0
            if (
                cell_flag == HomeFreeCellFlag.INTERFACE
                and source_flag == HomeFreeCellFlag.INTERFACE
            ):
                exchange_weight = 0.5 * (fill[index] + fill[source_index])
            mass_delta = exchange_weight * (
                populations[source_index + (q,)]
                - populations[index + (int(D3Q27_OPPOSITE[q]),)]
            )
            advected_mass[index] += mass_delta
            if mass_delta > 0.0 and (
                cell_flag == HomeFreeCellFlag.INTERFACE
                or source_flag == HomeFreeCellFlag.INTERFACE
            ):
                interface_correction[index] += mass_delta * (
                    velocity[source_index] - velocity[index]
                )
    return (
        advected_mass,
        incoming_excess_mass,
        incoming_excess_momentum,
        interface_correction,
    )


def interface_home_convective_density_momentum_increment(
    moments: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Extract HOME nonlinear stream input on interface-adjacent active links.

    Gas-pressure reconstruction, domain walls, and solid/cut-link reflections
    are excluded so their impulses remain in their existing ledgers.  The four
    returned components are ``delta_rho, delta_jx, delta_jy, delta_jz``.
    """

    home = np.asarray(moments, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    shape = home.shape[:-1]
    if cell_flags.shape != shape:
        raise ValueError("flags must match the HOME grid shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    legal = np.isin(cell_flags, [int(flag) for flag in HomeFreeCellFlag])
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")
    active = np.isin(
        cell_flags,
        (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
    )
    if np.any(home[..., 0][active] <= 0.0):
        raise ValueError("active HOME cells require positive density")

    convective = equilibrium_convective_populations(home)
    increment = np.zeros(shape + (4,), dtype=np.float64)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in np.ndindex(shape):
        if not active[index]:
            continue
        for q, c in enumerate(directions):
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid = False
            if not valid:
                continue
            source_index = tuple(source)
            if not active[source_index]:
                continue
            if (
                cell_flags[index] != int(HomeFreeCellFlag.INTERFACE)
                and cell_flags[source_index] != int(HomeFreeCellFlag.INTERFACE)
            ):
                continue
            value = convective[source_index + (q,)]
            increment[index + (0,)] += value
            increment[index + (slice(1, 4),)] += value * c
    return increment


def advect_internal_link_mass_momentum(
    moments: np.ndarray,
    mass: np.ndarray,
    fill_level: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact active-link LBM mass and momentum fluxes.

    For an oriented link from ``source`` to ``destination`` the update is

    ``delta_m = alpha * (f_q(source) - f_opp(destination))``

    ``delta_p = alpha * c_q * (f_q(source) + f_opp(destination))``.

    The second expression is the exact momentum crossing that lattice link,
    including convective, pressure, and non-equilibrium stress contributions.
    Gas, wall, force, and cut-link impulses are intentionally outside this
    internal-flux reference.
    """

    home = np.asarray(moments, dtype=np.float64)
    liquid_mass = np.asarray(mass, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    shape = home.shape[:-1]
    if any(field.shape != shape for field in (liquid_mass, fill, cell_flags)):
        raise ValueError("mass, fill_level, and flags must match the HOME grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not all(np.isfinite(field).all() for field in (home, liquid_mass, fill)):
        raise ValueError("link-flux fields must be finite")
    if np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain in [0, 1]")
    legal = np.isin(cell_flags, [int(flag) for flag in HomeFreeCellFlag])
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")
    active = np.isin(
        cell_flags,
        (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
    )
    density = home[..., 0]
    if np.any(density[active] <= 0.0):
        raise ValueError("active HOME cells require positive density")

    velocity = np.zeros(shape + (3,), dtype=np.float64)
    velocity[active] = home[..., 1:4][active] / density[active, None]
    populations = reconstruct_populations(home)
    advected_mass = liquid_mass.copy()
    advected_momentum = liquid_mass[..., None] * velocity
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in np.ndindex(shape):
        cell_flag = HomeFreeCellFlag(int(cell_flags[index]))
        if cell_flag not in (HomeFreeCellFlag.LIQUID, HomeFreeCellFlag.INTERFACE):
            continue
        for q in range(1, 27):
            c = directions[q]
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid = False
            if not valid:
                continue
            source_index = tuple(source)
            source_flag = HomeFreeCellFlag(int(cell_flags[source_index]))
            if source_flag not in (
                HomeFreeCellFlag.LIQUID,
                HomeFreeCellFlag.INTERFACE,
            ):
                continue
            alpha = 1.0
            if (
                cell_flag == HomeFreeCellFlag.INTERFACE
                and source_flag == HomeFreeCellFlag.INTERFACE
            ):
                alpha = 0.5 * (fill[index] + fill[source_index])
            opposite = int(D3Q27_OPPOSITE[q])
            incoming = populations[source_index + (q,)]
            outgoing = populations[index + (opposite,)]
            advected_mass[index] += alpha * (incoming - outgoing)
            advected_momentum[index] += alpha * (incoming + outgoing) * c
    return advected_mass, advected_momentum


def home_internal_link_momentum(
    moments: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Return HOME momentum after active-active links, before external inputs."""

    home = np.asarray(moments, dtype=np.float64)
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    shape = home.shape[:-1]
    _, momentum = advect_internal_link_mass_momentum(
        home,
        home[..., 0],
        np.ones(shape, dtype=np.float64),
        flags,
        periodic=periodic,
    )
    return momentum


def reference_pressure_link_momentum(
    fill_level: np.ndarray,
    flags: np.ndarray,
    pressure_reference_density: float,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Return the absolute reference-pressure part of each liquid link ledger.

    Active-active links use the same symmetric HOME-Free weight as liquid mass
    exchange.  Gas, solid, and domain-boundary links own the full reflected
    reference-pressure impulse.  Subtracting this field leaves gauge pressure
    while retaining dynamic, viscous, capillary, and moving-wall terms.
    """

    fill = np.asarray(fill_level, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if fill.ndim != 3 or cell_flags.shape != fill.shape:
        raise ValueError("fill_level and flags must have one identical 3D shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if (
        not np.isfinite(pressure_reference_density)
        or pressure_reference_density <= 0.0
    ):
        raise ValueError("pressure_reference_density must be finite and positive")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must contain finite values in [0, 1]")
    legal = np.isin(cell_flags, [int(flag) for flag in HomeFreeCellFlag])
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")

    shape = fill.shape
    active = np.isin(
        cell_flags,
        (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
    )
    result = np.zeros(shape + (3,), dtype=np.float64)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in np.ndindex(shape):
        if not active[index]:
            continue
        for q in range(1, 27):
            c = directions[q]
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid = False
            alpha = 1.0
            if valid:
                source_index = tuple(source)
                if (
                    cell_flags[index] == int(HomeFreeCellFlag.INTERFACE)
                    and cell_flags[source_index]
                    == int(HomeFreeCellFlag.INTERFACE)
                ):
                    alpha = 0.5 * (fill[index] + fill[source_index])
            result[index] += (
                alpha
                * 2.0
                * D3Q27_WEIGHTS[q]
                * pressure_reference_density
                * c
            )
    return result


def discrete_capillary_pressure_link_momentum(
    gas_density: np.ndarray,
    ambient_gas_density: float,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Return the gas-link staircase approximation of capillary pressure."""

    gas_rho = np.asarray(gas_density, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if gas_rho.ndim != 3 or cell_flags.shape != gas_rho.shape:
        raise ValueError("gas_density and flags must have one identical 3D shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not np.isfinite(gas_rho).all() or np.any(gas_rho <= 0.0):
        raise ValueError("gas_density must contain finite positive values")
    if not np.isfinite(ambient_gas_density) or ambient_gas_density <= 0.0:
        raise ValueError("ambient_gas_density must be finite and positive")

    shape = gas_rho.shape
    result = np.zeros(shape + (3,), dtype=np.float64)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in np.ndindex(shape):
        if cell_flags[index] != int(HomeFreeCellFlag.INTERFACE):
            continue
        density_jump = gas_rho[index] - ambient_gas_density
        for q in range(1, 27):
            c = directions[q]
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid = False
            if valid and cell_flags[tuple(source)] == int(HomeFreeCellFlag.GAS):
                result[index] += 2.0 * D3Q27_WEIGHTS[q] * density_jump * c
    return result


def plic_capillary_pressure_momentum(
    normal: np.ndarray,
    plane_offset: np.ndarray,
    gas_density: np.ndarray,
    ambient_gas_density: float,
    flags: np.ndarray,
) -> np.ndarray:
    """Return the PLIC area-integrated capillary traction per interface cell."""

    normals = np.asarray(normal, dtype=np.float64)
    offsets = np.asarray(plane_offset, dtype=np.float64)
    gas_rho = np.asarray(gas_density, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    shape = gas_rho.shape
    if gas_rho.ndim != 3 or normals.shape != shape + (3,):
        raise ValueError("normal must have gas_density.shape + (3,)")
    if offsets.shape != shape or cell_flags.shape != shape:
        raise ValueError("PLIC scalar fields must match gas_density shape")
    if not all(np.isfinite(field).all() for field in (normals, offsets, gas_rho)):
        raise ValueError("PLIC capillary fields must be finite")
    if np.any(gas_rho <= 0.0):
        raise ValueError("gas_density must be positive")
    if not np.isfinite(ambient_gas_density) or ambient_gas_density <= 0.0:
        raise ValueError("ambient_gas_density must be finite and positive")

    result = np.zeros(shape + (3,), dtype=np.float64)
    for index in np.ndindex(shape):
        if cell_flags[index] != int(HomeFreeCellFlag.INTERFACE):
            continue
        n = normals[index]
        magnitude = float(np.linalg.norm(n))
        if magnitude == 0.0:
            raise ValueError("interface PLIC normal must be nonzero")
        area = plane_interface_area(float(offsets[index]), n)
        pressure_jump = (gas_rho[index] - ambient_gas_density) / 3.0
        result[index] = -pressure_jump * area * n / magnitude
    return result


def normalize_mass_and_excess(
    density: np.ndarray,
    advected_mass: np.ndarray,
    flags: np.ndarray,
    active_neighbor_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp committed mass and encode excess as a per-recipient share."""

    rho = np.asarray(density, dtype=np.float64)
    mass = np.asarray(advected_mass, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    recipients = np.asarray(active_neighbor_count, dtype=np.int32)
    if not (rho.shape == mass.shape == cell_flags.shape == recipients.shape):
        raise ValueError("mass normalization fields must have identical shapes")
    if np.any(recipients < 0):
        raise ValueError("active_neighbor_count cannot be negative")
    if not np.isfinite(rho).all() or not np.isfinite(mass).all():
        raise ValueError("density and advected_mass must be finite")

    committed = np.zeros_like(mass)
    fill = np.zeros_like(mass)
    excess = np.zeros_like(mass)
    liquid = cell_flags == int(HomeFreeCellFlag.LIQUID)
    interface = cell_flags == int(HomeFreeCellFlag.INTERFACE)
    gas = cell_flags == int(HomeFreeCellFlag.GAS)
    solid = cell_flags == int(HomeFreeCellFlag.SOLID)
    if not np.all(liquid | interface | gas | solid):
        raise ValueError("flags contain an unknown HOME-FREE cell category")
    if np.any(rho[liquid | interface] <= 0.0):
        raise ValueError("active cells require positive density")
    if np.any(np.abs(mass[solid]) > 0.0):
        raise ValueError("solid cells cannot receive advected liquid mass")

    committed[liquid] = rho[liquid]
    fill[liquid] = 1.0
    excess[liquid] = mass[liquid] - rho[liquid]
    committed[interface] = np.clip(mass[interface], 0.0, rho[interface])
    fill[interface] = committed[interface] / rho[interface]
    excess[interface] = mass[interface] - committed[interface]
    excess[gas] = mass[gas]

    has_recipient = recipients > 0
    no_recipient = ~has_recipient
    committed[no_recipient] += excess[no_recipient]
    excess[no_recipient] = 0.0
    excess[has_recipient] /= recipients[has_recipient]
    return committed, fill, excess


def normalize_mass_momentum_and_excess(
    density: np.ndarray,
    momentum: np.ndarray,
    advected_mass: np.ndarray,
    flags: np.ndarray,
    active_neighbor_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize liquid mass and attach source-cell momentum to each queue share."""

    rho = np.asarray(density, dtype=np.float64)
    j = np.asarray(momentum, dtype=np.float64)
    if j.shape != rho.shape + (3,):
        raise ValueError("momentum must have density.shape + (3,)")
    committed, fill, excess = normalize_mass_and_excess(
        rho,
        advected_mass,
        flags,
        active_neighbor_count,
    )
    queued = excess != 0.0
    if np.any(rho[queued] <= 0.0) or not np.isfinite(j).all():
        raise ValueError("queued excess requires finite momentum and positive density")
    excess_momentum = np.zeros(rho.shape + (3,), dtype=np.float64)
    excess_momentum[queued] = excess[queued, None] * j[queued] / rho[queued, None]
    return committed, fill, excess, excess_momentum


def normalize_advected_mass_momentum_and_excess(
    density: np.ndarray,
    home_momentum: np.ndarray,
    advected_mass: np.ndarray,
    advected_momentum: np.ndarray,
    flags: np.ndarray,
    active_neighbor_count: np.ndarray,
    *,
    mass_epsilon: float = 1.0e-12,
    momentum_epsilon: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one exact liquid mass-momentum transaction per cell.

    The returned queue fields are per-recipient shares.  For every cell the
    represented transaction satisfies ``m_commit*u + N*e_p = P_advected``.
    A zero-mass fresh interface retains its donor-initialized HOME velocity,
    but finite momentum without transported mass is rejected.
    """

    rho = np.asarray(density, dtype=np.float64)
    home_j = np.asarray(home_momentum, dtype=np.float64)
    transported_mass = np.asarray(advected_mass, dtype=np.float64)
    transported_momentum = np.asarray(advected_momentum, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    recipients = np.asarray(active_neighbor_count, dtype=np.int32)
    if home_j.shape != rho.shape + (3,) or transported_momentum.shape != home_j.shape:
        raise ValueError("momentum fields must have density.shape + (3,)")
    if not (
        rho.shape == transported_mass.shape == cell_flags.shape == recipients.shape
    ):
        raise ValueError("transaction fields must have identical grid shapes")
    if mass_epsilon < 0.0 or momentum_epsilon < 0.0:
        raise ValueError("transaction tolerances must be nonnegative")
    if not all(
        np.isfinite(field).all()
        for field in (rho, home_j, transported_mass, transported_momentum)
    ):
        raise ValueError("mass-momentum transaction fields must be finite")

    liquid = cell_flags == int(HomeFreeCellFlag.LIQUID)
    interface = cell_flags == int(HomeFreeCellFlag.INTERFACE)
    gas = cell_flags == int(HomeFreeCellFlag.GAS)
    solid = cell_flags == int(HomeFreeCellFlag.SOLID)
    if not np.all(liquid | interface | gas | solid):
        raise ValueError("flags contain an unknown HOME-FREE cell category")
    active = liquid | interface
    if np.any(rho[active] <= 0.0):
        raise ValueError("active cells require positive density")

    raw_committed = np.zeros_like(transported_mass)
    raw_committed[liquid] = rho[liquid]
    raw_committed[interface] = np.clip(
        transported_mass[interface], 0.0, rho[interface]
    )
    raw_excess = transported_mass - raw_committed
    stranded = (recipients == 0) & (np.abs(raw_excess) > mass_epsilon)
    if np.any(stranded):
        raise ValueError("nonzero liquid excess requires an active recipient")

    committed, fill, excess = normalize_mass_and_excess(
        rho,
        transported_mass,
        cell_flags,
        recipients,
    )
    velocity = np.zeros_like(home_j)
    velocity[active] = home_j[active] / rho[active, None]
    has_mass = np.abs(transported_mass) > mass_epsilon
    momentum_norm = np.linalg.norm(transported_momentum, axis=-1)
    invalid_zero_mass = (~has_mass) & (momentum_norm > momentum_epsilon)
    if np.any(invalid_zero_mass):
        raise ValueError("finite liquid momentum requires transported mass")
    velocity[has_mass] = (
        transported_momentum[has_mass] / transported_mass[has_mass, None]
    )

    excess_momentum = excess[..., None] * velocity
    committed_home_momentum = home_j.copy()
    committed_home_momentum[active] = rho[active, None] * velocity[active]
    represented_momentum = committed[..., None] * velocity
    represented_momentum += recipients[..., None] * excess_momentum
    if not np.allclose(
        represented_momentum,
        transported_momentum,
        rtol=0.0,
        atol=max(momentum_epsilon, 1.0e-14),
    ):
        raise RuntimeError("normalized liquid momentum does not close per cell")
    return (
        committed,
        fill,
        excess,
        excess_momentum,
        committed_home_momentum,
    )


def represented_liquid_momentum(
    density: np.ndarray,
    momentum: np.ndarray,
    mass: np.ndarray,
    excess_mass: np.ndarray,
    excess_momentum: np.ndarray,
    active_neighbor_count: np.ndarray,
) -> np.ndarray:
    """Return the physical liquid momentum represented by a VOF time level.

    Both excess fields are per-recipient shares.  Committed mass moves at the
    local HOME velocity, while queued mass carries its explicitly stored donor
    momentum until the next advection pass consumes it.
    """

    rho = np.asarray(density, dtype=np.float64)
    j = np.asarray(momentum, dtype=np.float64)
    committed = np.asarray(mass, dtype=np.float64)
    excess = np.asarray(excess_mass, dtype=np.float64)
    excess_j = np.asarray(excess_momentum, dtype=np.float64)
    recipients = np.asarray(active_neighbor_count, dtype=np.int32)
    if j.shape != rho.shape + (3,) or excess_j.shape != j.shape:
        raise ValueError("momentum fields must have density.shape + (3,)")
    if not (
        rho.shape == committed.shape == excess.shape == recipients.shape
    ):
        raise ValueError("represented-momentum fields must have identical grid shapes")
    if np.any(recipients < 0):
        raise ValueError("active_neighbor_count cannot be negative")
    if not all(
        np.isfinite(field).all()
        for field in (rho, j, committed, excess, excess_j)
    ):
        raise ValueError("represented-momentum fields must be finite")

    occupied = committed != 0.0
    if np.any(rho[occupied] <= 0.0):
        raise ValueError("nonzero represented liquid mass requires positive density")
    velocity = np.zeros_like(j)
    velocity[occupied] = j[occupied] / rho[occupied, None]
    represented = committed[..., None] * velocity
    represented += recipients[..., None] * excess_j
    return np.sum(represented, axis=(0, 1, 2), dtype=np.float64)


def classify_fill_levels(
    fill_level: np.ndarray,
    solid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Classify an already sharp fill field without inventing interface cells."""

    fill = np.asarray(fill_level)
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must contain finite values in [0, 1]")
    if solid_mask is None:
        solid = np.zeros(fill.shape, dtype=bool)
    else:
        solid = np.asarray(solid_mask, dtype=bool)
        if solid.shape != fill.shape:
            raise ValueError("solid_mask and fill_level must have identical shapes")

    flags = np.full(fill.shape, int(HomeFreeCellFlag.INTERFACE), dtype=np.int32)
    flags[fill == 0.0] = int(HomeFreeCellFlag.GAS)
    flags[fill == 1.0] = int(HomeFreeCellFlag.LIQUID)
    flags[solid] = int(HomeFreeCellFlag.SOLID)
    return flags


def validate_state_fields(
    density: np.ndarray,
    mass: np.ndarray,
    fill_level: np.ndarray,
    excess_mass: np.ndarray,
    excess_momentum: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    tolerance: float = 2.0e-6,
    require_interface_separation: bool = True,
    allow_interface_endpoints: bool = False,
    require_mass_fill_consistency: bool = True,
) -> HomeFreeStateDiagnostics:
    """Validate committed fields and the one-cell sharp-interface invariant."""

    rho = np.asarray(density, dtype=np.float64)
    mass_field = np.asarray(mass, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    excess = np.asarray(excess_mass, dtype=np.float64)
    excess_j = np.asarray(excess_momentum, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    shape = rho.shape
    if len(shape) != 3:
        raise ValueError("HOME-FREE fields must be three-dimensional")
    if any(field.shape != shape for field in (mass_field, fill, excess, cell_flags)):
        raise ValueError("all HOME-FREE fields must have identical shapes")
    if excess_j.shape != shape + (3,):
        raise ValueError("excess_momentum must have grid shape + (3,)")
    if not np.isfinite(rho).all() or not np.isfinite(mass_field).all():
        raise ValueError("density and mass must be finite")
    if not all(np.isfinite(field).all() for field in (fill, excess, excess_j)):
        raise ValueError("fill_level and excess queues must be finite")
    if np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain in [0, 1] in a committed state")
    legal = np.isin(cell_flags, [int(flag) for flag in HomeFreeCellFlag])
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-FREE cell category")

    liquid = cell_flags == int(HomeFreeCellFlag.LIQUID)
    interface = cell_flags == int(HomeFreeCellFlag.INTERFACE)
    gas = cell_flags == int(HomeFreeCellFlag.GAS)
    solid = cell_flags == int(HomeFreeCellFlag.SOLID)
    active = liquid | interface
    if np.any(rho[active] <= 0.0):
        raise ValueError("liquid and interface density must be positive")
    if np.any(np.abs(fill[liquid] - 1.0) > tolerance):
        raise ValueError("liquid cells must have fill_level=1")
    if not allow_interface_endpoints and (
        np.any(fill[interface] <= 0.0) or np.any(fill[interface] >= 1.0)
    ):
        raise ValueError("interface cells must have a strict fractional fill level")
    if np.any(np.abs(fill[gas | solid]) > tolerance):
        raise ValueError("gas and solid cells must have fill_level=0")
    if require_mass_fill_consistency and np.any(
        np.abs(mass_field[active] - rho[active] * fill[active]) > tolerance
    ):
        raise ValueError("committed liquid mass must equal density*fill_level")
    if not require_mass_fill_consistency and np.any(mass_field[active] <= 0.0):
        raise ValueError("active geometric liquid mass must be positive")
    if np.any(np.abs(excess[solid]) > tolerance) or np.any(
        np.abs(excess_j[solid]) > tolerance
    ):
        raise ValueError("solid cells cannot own queued liquid mass or momentum")
    empty_queue = np.abs(excess) <= tolerance
    if np.any(np.abs(excess_j[empty_queue]) > tolerance):
        raise ValueError("zero excess mass cannot carry queued momentum")
    if np.any(np.abs(mass_field[gas | solid]) > tolerance):
        raise ValueError("gas and solid cells must have zero liquid mass")

    direct_links = _count_direct_liquid_gas_links(cell_flags, periodic)
    if require_interface_separation and direct_links:
        raise ValueError(
            f"sharp interface invariant violated by {direct_links} direct liquid-gas links"
        )
    return HomeFreeStateDiagnostics(
        liquid_cells=int(np.count_nonzero(liquid)),
        interface_cells=int(np.count_nonzero(interface)),
        gas_cells=int(np.count_nonzero(gas)),
        solid_cells=int(np.count_nonzero(solid)),
        direct_liquid_gas_links=direct_links,
        minimum_fill=float(np.min(fill)),
        maximum_fill=float(np.max(fill)),
        liquid_mass=float(np.sum(mass_field, dtype=np.float64)),
    )


def _count_direct_liquid_gas_links(
    flags: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> int:
    count = 0
    shape = flags.shape
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0))
    ]
    for offset in offsets:
        neighbor = np.roll(flags, shift=tuple(-value for value in offset), axis=(0, 1, 2))
        valid = np.ones(shape, dtype=bool)
        for axis, delta in enumerate(offset):
            if periodic[axis] or delta == 0:
                continue
            edge = [slice(None)] * 3
            edge[axis] = shape[axis] - 1 if delta > 0 else 0
            valid[tuple(edge)] = False
        count += int(np.count_nonzero(valid & _is_liquid_gas_pair(flags, neighbor)))
    return count


def _neighbor_indices(
    index: tuple[int, int, int],
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor = [index[0] + dx, index[1] + dy, index[2] + dz]
                valid = True
                for axis in range(3):
                    if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                        if periodic[axis]:
                            neighbor[axis] %= shape[axis]
                        else:
                            valid = False
                if valid:
                    yield tuple(neighbor)


def _is_liquid_gas_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    liquid = int(HomeFreeCellFlag.LIQUID)
    gas = int(HomeFreeCellFlag.GAS)
    return ((a == liquid) & (b == gas)) | ((a == gas) & (b == liquid))
