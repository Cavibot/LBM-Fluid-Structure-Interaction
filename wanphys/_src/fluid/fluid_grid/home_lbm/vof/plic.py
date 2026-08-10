# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Executable PLIC geometry specification for a unit Cartesian cell."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..constants import D3Q27_DIRECTIONS


_ACTIVE_NORMAL_RELATIVE_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class PlicCurvatureFit:
    curvature: float
    coefficients: tuple[float, float, float, float, float]
    sample_count: int
    rank: int
    condition_number: float


@dataclass(frozen=True)
class PlicLinkIntersection:
    """Intersection of a PLIC plane with one gas-side pull link."""

    fraction: float
    point: tuple[float, float, float]
    distance: float


class PlicPullLinkStatus(IntEnum):
    """Geometric availability of a gas-side PLIC pull link."""

    NOT_APPLICABLE = 0
    VALID = 1
    EXTRAPOLATED = 2
    NOT_FACING = -2
    BEHIND = -3
    OUTSIDE = -4
    NO_PLANE = -6
    NO_SUPPORT = -7


class PlicLinkPlaneOwner(IntEnum):
    """Cell whose local PLIC plane owns an FSL boundary link."""

    NONE = 0
    DESTINATION = 1
    SOURCE = 2


@dataclass(frozen=True)
class PlicPullLinkCoverage:
    """Structured coverage of PLIC intersections over OM gas links."""

    fraction: np.ndarray
    status: np.ndarray
    total_gas_links: int
    valid_links: int
    not_facing_links: int
    behind_links: int
    outside_links: int

    @property
    def valid_fraction(self) -> float:
        if self.total_gas_links == 0:
            return float("nan")
        return self.valid_links / self.total_gas_links


@dataclass(frozen=True)
class PlicFslLinkCoverage:
    """Boundary-link coverage after separating VOF and hydrodynamic nodes."""

    hydrodynamic_active: np.ndarray
    fraction: np.ndarray
    extrapolation: np.ndarray
    status: np.ndarray
    plane_owner: np.ndarray
    total_boundary_links: int
    exact_links: int
    extrapolated_links: int
    usable_links: int
    destination_owned_links: int
    source_owned_links: int
    not_facing_links: int
    behind_links: int
    outside_links: int
    no_plane_links: int
    no_support_links: int

    @property
    def exact_fraction(self) -> float:
        if self.total_boundary_links == 0:
            return float("nan")
        return self.exact_links / self.total_boundary_links

    @property
    def usable_fraction(self) -> float:
        if self.total_boundary_links == 0:
            return float("nan")
        return self.usable_links / self.total_boundary_links

    @property
    def maximum_extrapolation(self) -> float:
        if self.extrapolated_links == 0:
            return 0.0
        return float(np.nanmax(self.extrapolation))


def plane_volume_fraction(offset: float, normal: np.ndarray) -> float:
    """Return the unit-cube fraction satisfying ``normal dot x <= offset``.

    The cube is centered at the origin with extent ``[-1/2, 1/2]^3``.
    Inclusion-exclusion gives the exact CDF of a weighted sum of uniform
    variables and naturally handles axis-aligned and two-dimensional cuts.
    """

    n = _validated_normal(normal)
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    coefficients = np.abs(n)
    support = 0.5 * float(np.sum(coefficients))
    if offset <= -support:
        return 0.0
    if offset >= support:
        return 1.0
    # Evolved FP32 normals retain transverse noise near a coordinate axis.
    # Components below 1e-4 change section area by less than FP32 resolution,
    # while retaining it catastrophically cancels inclusion-exclusion terms.
    active = coefficients[
        coefficients > _ACTIVE_NORMAL_RELATIVE_TOLERANCE * np.max(coefficients)
    ]
    dimension = int(active.size)
    complement = offset > 0.0
    evaluation_offset = -offset if complement else offset
    shifted = float(evaluation_offset + 0.5 * np.sum(active))
    denominator = math.factorial(dimension) * float(np.prod(active))
    volume = 0.0
    for mask in itertools.product((0, 1), repeat=dimension):
        corner = float(np.dot(active, np.asarray(mask, dtype=np.float64)))
        distance = max(shifted - corner, 0.0)
        volume += (-1.0) ** sum(mask) * distance**dimension
    fraction = float(np.clip(volume / denominator, 0.0, 1.0))
    return 1.0 - fraction if complement else fraction


def plic_plane_offset(
    fill_level: float,
    normal: np.ndarray,
    *,
    iterations: int = 64,
) -> float:
    """Invert the unit-cube volume exactly enough for a PLIC reference plane."""

    if not math.isfinite(fill_level) or not 0.0 <= fill_level <= 1.0:
        raise ValueError("fill_level must be finite and in [0, 1]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    n = _validated_normal(normal)
    bound = 0.5 * float(np.sum(np.abs(n)))
    if fill_level == 0.0:
        return -bound
    if fill_level == 1.0:
        return bound
    lower = -bound
    upper = bound
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if plane_volume_fraction(midpoint, n) < fill_level:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def plane_interface_area(offset: float, normal: np.ndarray) -> float:
    """Return the exact area of a plane section through the unit cube.

    The result is the derivative of the clipped volume with respect to normal
    displacement.  Using the same inclusion-exclusion polynomial as
    :func:`plane_volume_fraction` keeps axis-aligned and lower-dimensional cuts
    exact without polygon tolerances.
    """

    n = _validated_normal(normal)
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    coefficients = np.abs(n)
    active = coefficients[
        coefficients > _ACTIVE_NORMAL_RELATIVE_TOLERANCE * np.max(coefficients)
    ]
    dimension = int(active.size)
    shifted = float(offset + 0.5 * np.sum(active))
    denominator = math.factorial(dimension - 1) * float(np.prod(active))
    derivative = 0.0
    for mask in itertools.product((0, 1), repeat=dimension):
        corner = float(np.dot(active, np.asarray(mask, dtype=np.float64)))
        distance = shifted - corner
        if distance > 0.0:
            derivative += (-1.0) ** sum(mask) * distance ** (dimension - 1)
    area = float(np.linalg.norm(n)) * derivative / denominator
    return max(area, 0.0)


def plic_pull_link_intersection(
    offset: float,
    normal: np.ndarray,
    population_direction: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> PlicLinkIntersection:
    """Intersect a local PLIC plane with a missing-population pull link.

    ``population_direction`` is the lattice velocity ``c_q`` used by the pull
    stream.  Its source is at ``-c_q`` relative to the destination interface
    cell, so the link ray is ``x(delta) = -delta*c_q``.  Scaling the plane
    equation leaves the fraction unchanged; scaling ``c_q`` inversely scales
    the fraction while leaving the physical point and distance unchanged.

    A valid intersection must face the gas source and lie on the PLIC section
    inside the destination unit cube.  A plane behind the lattice node, a
    tangent link, or an intersection outside that section is rejected instead
    of being replaced by the first-order half-link assumption.
    """

    status, intersection = _plic_pull_link_intersection_result(
        offset,
        normal,
        population_direction,
        tolerance=tolerance,
    )
    if status == PlicPullLinkStatus.NOT_FACING:
        raise ValueError("pull link does not face the PLIC gas half-space")
    if status == PlicPullLinkStatus.BEHIND:
        raise ValueError("PLIC plane lies behind the pull-link destination")
    if status == PlicPullLinkStatus.OUTSIDE:
        raise ValueError("pull-link intersection lies outside the local PLIC section")
    assert intersection is not None
    return intersection


def plic_pull_link_coverage(
    normal: np.ndarray,
    plane_offset: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    tolerance: float = 1.0e-12,
) -> PlicPullLinkCoverage:
    """Classify every OM gas link against its destination-cell PLIC plane."""

    normals = np.asarray(normal, dtype=np.float64)
    offsets = np.asarray(plane_offset, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if cell_flags.ndim != 3 or normals.shape != cell_flags.shape + (3,):
        raise ValueError("normal must have flags.shape + (3,)")
    if offsets.shape != cell_flags.shape:
        raise ValueError("plane_offset and flags must have identical shapes")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    legal = np.isin(cell_flags, (0, 1, 2, 3))
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")
    interface = cell_flags == 1
    if (
        not np.isfinite(normals[interface]).all()
        or not np.isfinite(offsets[interface]).all()
        or np.any(np.linalg.norm(normals[interface], axis=-1) == 0.0)
    ):
        raise ValueError("interface PLIC planes must be finite with nonzero normals")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    shape = cell_flags.shape
    fraction = np.full(shape + (27,), np.nan, dtype=np.float64)
    status = np.full(
        shape + (27,), int(PlicPullLinkStatus.NOT_APPLICABLE), dtype=np.int32
    )
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in map(tuple, np.argwhere(interface)):
        for q in range(1, 27):
            c = directions[q]
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            valid_source = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid_source = False
            if not valid_source or cell_flags[tuple(source)] != 0:
                continue
            link_status, intersection = _plic_pull_link_intersection_result(
                float(offsets[index]),
                normals[index],
                c,
                tolerance=tolerance,
            )
            status[index + (q,)] = int(link_status)
            if intersection is not None:
                fraction[index + (q,)] = intersection.fraction

    total = int(np.count_nonzero(status != int(PlicPullLinkStatus.NOT_APPLICABLE)))
    valid = int(np.count_nonzero(status == int(PlicPullLinkStatus.VALID)))
    not_facing = int(
        np.count_nonzero(status == int(PlicPullLinkStatus.NOT_FACING))
    )
    behind = int(np.count_nonzero(status == int(PlicPullLinkStatus.BEHIND)))
    outside = int(np.count_nonzero(status == int(PlicPullLinkStatus.OUTSIDE)))
    if total != valid + not_facing + behind + outside:
        raise RuntimeError("PLIC pull-link coverage does not partition OM gas links")
    return PlicPullLinkCoverage(
        fraction=fraction,
        status=status,
        total_gas_links=total,
        valid_links=valid,
        not_facing_links=not_facing,
        behind_links=behind,
        outside_links=outside,
    )


def plic_fsl_link_coverage(
    normal: np.ndarray,
    plane_offset: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    tolerance: float = 1.0e-12,
) -> PlicFslLinkCoverage:
    """Build the fluid-node/link ownership required by the FSL closure.

    VOF interface cells remain unchanged.  Hydrodynamic nodes are liquid cells
    plus interface cells whose centers lie in the PLIC liquid half-space.  A
    boundary link is destination-owned when it leaves such an interface node
    directly for gas, and source-owned when it enters an underfilled interface
    cell whose center lies in the gas half-space.
    """

    normals, offsets, cell_flags, interface = _validated_plic_coverage_fields(
        normal, plane_offset, flags, periodic, tolerance
    )
    shape = cell_flags.shape
    normal_magnitude = np.linalg.norm(normals, axis=-1)
    center_tolerance = tolerance * np.maximum(normal_magnitude, 1.0)
    hydrodynamic_active = (cell_flags == 2) | (
        interface & (offsets >= -center_tolerance)
    )
    fraction = np.full(shape + (27,), np.nan, dtype=np.float64)
    extrapolation = np.full(shape + (27,), np.nan, dtype=np.float64)
    status = np.full(
        shape + (27,), int(PlicPullLinkStatus.NOT_APPLICABLE), dtype=np.int32
    )
    owner = np.full(
        shape + (27,), int(PlicLinkPlaneOwner.NONE), dtype=np.int32
    )
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for index in map(tuple, np.argwhere(hydrodynamic_active)):
        for q in range(1, 27):
            c = directions[q]
            source = [index[axis] - int(c[axis]) for axis in range(3)]
            valid_source = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid_source = False
            if not valid_source:
                continue
            source_index = tuple(source)
            if hydrodynamic_active[source_index] or cell_flags[source_index] == 3:
                continue

            support = [index[axis] + int(c[axis]) for axis in range(3)]
            valid_support = True
            for axis in range(3):
                if support[axis] < 0 or support[axis] >= shape[axis]:
                    if periodic[axis]:
                        support[axis] %= shape[axis]
                    else:
                        valid_support = False
            has_support = valid_support and hydrodynamic_active[tuple(support)]

            if cell_flags[source_index] == 1:
                plane_index = source_index
                plane_owner = PlicLinkPlaneOwner.SOURCE
                owner_center = -c.astype(np.float64)
            elif cell_flags[source_index] == 0 and cell_flags[index] == 1:
                plane_index = index
                plane_owner = PlicLinkPlaneOwner.DESTINATION
                owner_center = np.zeros(3, dtype=np.float64)
            else:
                status[index + (q,)] = int(PlicPullLinkStatus.NO_PLANE)
                continue

            owner[index + (q,)] = int(plane_owner)
            link_status, intersection = _plic_owned_pull_link_intersection_result(
                float(offsets[plane_index]),
                normals[plane_index],
                c,
                owner_center=owner_center,
                tolerance=tolerance,
                allow_owner_extrapolation=True,
            )
            status[index + (q,)] = int(link_status)
            if intersection is not None:
                fraction[index + (q,)] = intersection.fraction
                owner_local_point = (
                    np.asarray(intersection.point, dtype=np.float64) - owner_center
                )
                extrapolation[index + (q,)] = max(
                    float(np.max(np.abs(owner_local_point))) - 0.5,
                    0.0,
                )
                if not has_support:
                    status[index + (q,)] = int(PlicPullLinkStatus.NO_SUPPORT)

    applicable = status != int(PlicPullLinkStatus.NOT_APPLICABLE)
    total = int(np.count_nonzero(applicable))
    exact = int(np.count_nonzero(status == int(PlicPullLinkStatus.VALID)))
    extrapolated = int(
        np.count_nonzero(status == int(PlicPullLinkStatus.EXTRAPOLATED))
    )
    usable_mask = np.isin(
        status,
        (int(PlicPullLinkStatus.VALID), int(PlicPullLinkStatus.EXTRAPOLATED)),
    )
    usable = int(np.count_nonzero(usable_mask))
    destination_owned = int(
        np.count_nonzero(
            usable_mask & (owner == int(PlicLinkPlaneOwner.DESTINATION))
        )
    )
    source_owned = int(
        np.count_nonzero(
            usable_mask & (owner == int(PlicLinkPlaneOwner.SOURCE))
        )
    )
    not_facing = int(
        np.count_nonzero(status == int(PlicPullLinkStatus.NOT_FACING))
    )
    behind = int(np.count_nonzero(status == int(PlicPullLinkStatus.BEHIND)))
    outside = int(np.count_nonzero(status == int(PlicPullLinkStatus.OUTSIDE)))
    no_plane = int(np.count_nonzero(status == int(PlicPullLinkStatus.NO_PLANE)))
    no_support = int(
        np.count_nonzero(status == int(PlicPullLinkStatus.NO_SUPPORT))
    )
    if total != (
        exact
        + extrapolated
        + not_facing
        + behind
        + outside
        + no_plane
        + no_support
    ):
        raise RuntimeError("PLIC FSL coverage does not partition boundary links")
    if usable != destination_owned + source_owned:
        raise RuntimeError("usable PLIC FSL links do not have exactly one plane owner")
    return PlicFslLinkCoverage(
        hydrodynamic_active=hydrodynamic_active,
        fraction=fraction,
        extrapolation=extrapolation,
        status=status,
        plane_owner=owner,
        total_boundary_links=total,
        exact_links=exact,
        extrapolated_links=extrapolated,
        usable_links=usable,
        destination_owned_links=destination_owned,
        source_owned_links=source_owned,
        not_facing_links=not_facing,
        behind_links=behind,
        outside_links=outside,
        no_plane_links=no_plane,
        no_support_links=no_support,
    )


def _plic_pull_link_intersection_result(
    offset: float,
    normal: np.ndarray,
    population_direction: np.ndarray,
    *,
    tolerance: float,
) -> tuple[PlicPullLinkStatus, PlicLinkIntersection | None]:
    return _plic_owned_pull_link_intersection_result(
        offset,
        normal,
        population_direction,
        owner_center=np.zeros(3, dtype=np.float64),
        tolerance=tolerance,
        allow_owner_extrapolation=False,
    )


def _plic_owned_pull_link_intersection_result(
    offset: float,
    normal: np.ndarray,
    population_direction: np.ndarray,
    *,
    owner_center: np.ndarray,
    tolerance: float,
    allow_owner_extrapolation: bool,
) -> tuple[PlicPullLinkStatus, PlicLinkIntersection | None]:
    n = _validated_normal(normal)
    c = np.asarray(population_direction, dtype=np.float64)
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError("population_direction must contain three finite components")
    direction_norm = float(np.linalg.norm(c))
    if direction_norm == 0.0:
        raise ValueError("population_direction must be nonzero")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    center = np.asarray(owner_center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("owner_center must contain three finite components")

    target_offset = float(offset + np.dot(n, center))
    denominator = -float(np.dot(n, c))
    directional_scale = float(np.linalg.norm(n)) * direction_norm
    if denominator <= tolerance * directional_scale:
        return PlicPullLinkStatus.NOT_FACING, None
    fraction = float(target_offset / denominator)
    if fraction < -tolerance:
        return PlicPullLinkStatus.BEHIND, None
    if fraction < 0.0:
        fraction = 0.0
    point = -fraction * c
    owner_local_point = point - center
    if fraction > 1.0 + tolerance:
        return PlicPullLinkStatus.OUTSIDE, None
    owner_outside = float(np.max(np.abs(owner_local_point))) > 0.5 + tolerance
    if owner_outside and not allow_owner_extrapolation:
        return PlicPullLinkStatus.OUTSIDE, None
    residual = abs(float(np.dot(n, owner_local_point)) - offset)
    if residual > tolerance * max(1.0, abs(offset), float(np.linalg.norm(n))):
        raise FloatingPointError("pull-link intersection does not satisfy the PLIC plane")
    link_status = (
        PlicPullLinkStatus.EXTRAPOLATED
        if owner_outside
        else PlicPullLinkStatus.VALID
    )
    return link_status, PlicLinkIntersection(
        fraction=fraction,
        point=tuple(float(component) for component in point),
        distance=fraction * direction_norm,
    )


def _validated_plic_coverage_fields(
    normal: np.ndarray,
    plane_offset: np.ndarray,
    flags: np.ndarray,
    periodic: tuple[bool, bool, bool],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray(normal, dtype=np.float64)
    offsets = np.asarray(plane_offset, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if cell_flags.ndim != 3 or normals.shape != cell_flags.shape + (3,):
        raise ValueError("normal must have flags.shape + (3,)")
    if offsets.shape != cell_flags.shape:
        raise ValueError("plane_offset and flags must have identical shapes")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    legal = np.isin(cell_flags, (0, 1, 2, 3))
    if not legal.all():
        raise ValueError("flags contain an unknown HOME-Free cell category")
    interface = cell_flags == 1
    if (
        not np.isfinite(normals[interface]).all()
        or not np.isfinite(offsets[interface]).all()
        or np.any(np.linalg.norm(normals[interface], axis=-1) == 0.0)
    ):
        raise ValueError("interface PLIC planes must be finite with nonzero normals")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return normals, offsets, cell_flags, interface


def youngs_interface_normal(fill_stencil: np.ndarray) -> np.ndarray:
    """Estimate the liquid-to-gas unit normal from a 3x3x3 VOF stencil.

    This is the 3D Parker-Youngs/Sobel stencil used by the HOME-FSLBM
    reference: face samples carry transverse weight 4, edge samples 2,
    and corner samples 1. A constant stencil has no defined interface normal
    and is rejected rather than assigned an arbitrary direction.
    """

    fill = np.asarray(fill_stencil, dtype=np.float64)
    if fill.shape != (3, 3, 3):
        raise ValueError("fill_stencil must have shape (3, 3, 3)")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_stencil must contain finite values in [0, 1]")
    transverse = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
    weights = np.outer(transverse, transverse)
    gradient = np.asarray(
        [
            np.sum(weights * (fill[2, :, :] - fill[0, :, :])),
            np.sum(weights * (fill[:, 2, :] - fill[:, 0, :])),
            np.sum(weights * (fill[:, :, 2] - fill[:, :, 0])),
        ],
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(gradient))
    if magnitude <= 1.0e-14:
        raise ValueError("interface normal is undefined for a zero-gradient stencil")
    return -gradient / magnitude


def contact_angle_interface_normal(
    raw_interface_normal: np.ndarray,
    wall_normal: np.ndarray,
    contact_angle_degrees: float,
) -> np.ndarray:
    """Impose a static contact angle on a liquid-to-gas PLIC normal.

    ``wall_normal`` points from solid into fluid and the contact angle is
    measured through the liquid.  The raw Youngs normal supplies only the
    signed direction in the wall tangent plane; the prescribed angle supplies
    the normal component exactly.
    """

    raw = _validated_normal(raw_interface_normal)
    wall = _validated_normal(wall_normal)
    if not math.isfinite(contact_angle_degrees) or not 0.0 < contact_angle_degrees < 180.0:
        raise ValueError("contact_angle_degrees must be finite and in (0, 180)")
    wall = wall / np.linalg.norm(wall)
    tangent = raw - float(np.dot(raw, wall)) * wall
    tangent_length = float(np.linalg.norm(tangent))
    if tangent_length <= 1.0e-12:
        raise ValueError(
            "contact-angle normal is undefined when the raw interface normal has no wall-tangent component"
        )
    tangent /= tangent_length
    theta = math.radians(contact_angle_degrees)
    corrected = math.cos(theta) * wall + math.sin(theta) * tangent
    corrected /= np.linalg.norm(corrected)
    return corrected


def fit_plic_curvature(
    fill_stencil: np.ndarray,
    flag_stencil: np.ndarray,
    center_normal: np.ndarray,
    *,
    max_condition_number: float = 1.0e5,
) -> PlicCurvatureFit:
    """Fit a quadratic Monge patch to neighboring PLIC interface planes."""

    fill = np.asarray(fill_stencil, dtype=np.float64)
    flags = np.asarray(flag_stencil, dtype=np.int32)
    if fill.shape != (3, 3, 3) or flags.shape != fill.shape:
        raise ValueError("fill_stencil and flag_stencil must have shape (3, 3, 3)")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_stencil must contain finite values in [0, 1]")
    if int(flags[1, 1, 1]) != 1:
        raise ValueError("the center cell must be an interface cell")
    if not math.isfinite(max_condition_number) or max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be finite and greater than one")
    normal = _validated_normal(center_normal)
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    tangent_y = np.cross(normal, axis)
    tangent_y /= np.linalg.norm(tangent_y)
    tangent_x = np.cross(tangent_y, normal)
    center_offset = plic_plane_offset(float(fill[1, 1, 1]), normal)

    design_rows: list[list[float]] = []
    heights: list[float] = []
    for index in np.ndindex(fill.shape):
        if index == (1, 1, 1) or int(flags[index]) != 1:
            continue
        displacement = np.asarray(index, dtype=np.float64) - 1.0
        neighbor_offset = plic_plane_offset(float(fill[index]), normal)
        x = float(np.dot(displacement, tangent_x))
        y = float(np.dot(displacement, tangent_y))
        z = float(np.dot(displacement, normal) + neighbor_offset - center_offset)
        design_rows.append([x * x, y * y, x * y, x, y])
        heights.append(z)
    sample_count = len(design_rows)
    if sample_count < 5:
        raise ValueError(
            f"PLIC curvature requires at least 5 interface neighbors, got {sample_count}"
        )
    design = np.asarray(design_rows, dtype=np.float64)
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(design.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 5:
        raise ValueError(f"PLIC curvature design matrix is rank deficient ({rank}/5)")
    condition_number = float(singular_values[0] / singular_values[-1])
    if condition_number > max_condition_number:
        raise ValueError(
            "PLIC curvature design matrix condition number "
            f"{condition_number} exceeds {max_condition_number}"
        )
    coefficients = np.linalg.lstsq(design, np.asarray(heights), rcond=None)[0]
    a, b, c, h, i = (float(value) for value in coefficients)
    denominator = (1.0 + h * h + i * i) ** 1.5
    curvature = (a * (1.0 + i * i) + b * (1.0 + h * h) - c * h * i) / denominator
    return PlicCurvatureFit(
        curvature=float(curvature),
        coefficients=(a, b, c, h, i),
        sample_count=sample_count,
        rank=rank,
        condition_number=condition_number,
    )


def _validated_normal(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    if n.shape != (3,) or not np.isfinite(n).all():
        raise ValueError("normal must contain three finite components")
    if float(np.max(np.abs(n))) == 0.0:
        raise ValueError("normal must be nonzero")
    return n
