# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side boundary contracts and edge/corner conflict resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .constants import (
    BC_BOUNCE_BACK,
    BC_OUTFLOW,
    BC_PERIODIC,
    BC_PRESSURE,
    BC_VELOCITY_INLET,
)
from .contracts import BoundaryModel


FACE_NAMES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
OPEN_BOUNDARY_TYPES = frozenset((BC_VELOCITY_INLET, BC_OUTFLOW, BC_PRESSURE))
VALID_BOUNDARY_TYPES = frozenset(
    (BC_BOUNCE_BACK, BC_VELOCITY_INLET, BC_OUTFLOW, BC_PERIODIC, BC_PRESSURE)
)

BOUNDARY_MODEL_TO_TYPE = {
    BoundaryModel.BOUNCE_BACK: BC_BOUNCE_BACK,
    BoundaryModel.ZOU_HE: BC_VELOCITY_INLET,
    BoundaryModel.PRESSURE: BC_PRESSURE,
    BoundaryModel.CONVECTIVE: BC_OUTFLOW,
    BoundaryModel.PERIODIC: BC_PERIODIC,
    BoundaryModel.MOVING_WALL: BC_BOUNCE_BACK,
    BoundaryModel.CUT_LINK: BC_BOUNCE_BACK,
}
BOUNDARY_TYPE_TO_MODEL = {
    BC_BOUNCE_BACK: BoundaryModel.BOUNCE_BACK,
    BC_VELOCITY_INLET: BoundaryModel.ZOU_HE,
    BC_OUTFLOW: BoundaryModel.CONVECTIVE,
    BC_PERIODIC: BoundaryModel.PERIODIC,
    BC_PRESSURE: BoundaryModel.PRESSURE,
}


def normalize_boundary_type(value: int | str | BoundaryModel) -> tuple[int, BoundaryModel]:
    """Normalize the string public API while retaining integer kernel metadata."""

    if isinstance(value, int):
        try:
            return value, BOUNDARY_TYPE_TO_MODEL[value]
        except KeyError as exc:
            raise ValueError(f"Unknown LBM boundary type {value}") from exc
    try:
        model = BoundaryModel(str(value).lower())
    except ValueError as exc:
        expected = ", ".join(item.value for item in BoundaryModel)
        raise ValueError(
            f"Unknown LBM boundary model {value!r}; expected one of: {expected}"
        ) from exc
    if model is BoundaryModel.SURFACE:
        raise NotImplementedError(
            "SurfaceCompletion is interface-only and has no implementation"
        )
    return BOUNDARY_MODEL_TO_TYPE[model], model


@dataclass(frozen=True)
class BoundaryConflictSummary:
    """Aggregated fallback information produced during boundary setup."""

    cell_count: int
    representative_cells: tuple[tuple[int, int, int], ...]
    open_faces: tuple[str, ...]

    @property
    def warning_message(self) -> str:
        samples = ", ".join(str(cell) for cell in self.representative_cells)
        return (
            "LBM open-boundary edge/corner conflict: "
            f"{self.cell_count} cells touching {', '.join(self.open_faces)} "
            "were forced to static bounce-back"
            + (f"; representative cells: {samples}" if samples else "")
        )


@dataclass(frozen=True)
class BoundaryResolution:
    face_types: tuple[int, int, int, int, int, int]
    open_faces: tuple[int, ...]
    conflict: BoundaryConflictSummary | None

    @property
    def has_open_boundaries(self) -> bool:
        return bool(self.open_faces)


def _incident_faces(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[int, ...]:
    result: list[int] = []
    if i == 0:
        result.append(0)
    if i == nx - 1:
        result.append(1)
    if j == 0:
        result.append(2)
    if j == ny - 1:
        result.append(3)
    if k == 0:
        result.append(4)
    if k == nz - 1:
        result.append(5)
    return tuple(result)


def resolve_boundary_faces(
    face_types: tuple[int, int, int, int, int, int],
    resolution: tuple[int, int, int],
    *,
    representative_limit: int = 8,
) -> BoundaryResolution:
    """Resolve face metadata and identify cells that must fall back to walls.

    Face-interior open boundaries are valid.  An open boundary on a geometric
    edge or corner is intentionally not jointly closed in the first version;
    those complete cells are handled as static bounce-back by the device
    boundary kernel.
    """

    if len(face_types) != 6:
        raise ValueError(f"bc_types must contain six face entries, got {face_types!r}")
    unknown = tuple(value for value in face_types if value not in VALID_BOUNDARY_TYPES)
    if unknown:
        raise ValueError(f"Unknown LBM boundary type(s): {unknown!r}")
    for first, second in ((0, 1), (2, 3), (4, 5)):
        if (face_types[first] == BC_PERIODIC) != (face_types[second] == BC_PERIODIC):
            raise ValueError(
                "Periodic boundary faces must be paired on an axis: "
                f"{FACE_NAMES[first]}={face_types[first]}, "
                f"{FACE_NAMES[second]}={face_types[second]}"
            )
    nx, ny, nz = (int(value) for value in resolution)
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(f"LBM resolution must be positive, got {resolution!r}")

    open_faces = tuple(index for index, value in enumerate(face_types) if value in OPEN_BOUNDARY_TYPES)
    if not open_faces:
        return BoundaryResolution(tuple(face_types), (), None)
    for face in open_faces:
        normal_size = nx if face in (0, 1) else ny if face in (2, 3) else nz
        if normal_size < 2:
            raise ValueError(
                f"Open boundary {FACE_NAMES[face]} requires at least two cells "
                f"along its normal axis, got resolution={resolution!r}"
            )

    conflicts: set[tuple[int, int, int]] = set()
    for face in open_faces:
        if face in (0, 1):
            i = 0 if face == 0 else nx - 1
            candidates = ((i, j, k) for j in range(ny) for k in range(nz))
        elif face in (2, 3):
            j = 0 if face == 2 else ny - 1
            candidates = ((i, j, k) for i in range(nx) for k in range(nz))
        else:
            k = 0 if face == 4 else nz - 1
            candidates = ((i, j, k) for i in range(nx) for j in range(ny))
        for cell in candidates:
            if len(_incident_faces(*cell, nx, ny, nz)) > 1:
                conflicts.add(cell)

    summary = BoundaryConflictSummary(
        cell_count=len(conflicts),
        representative_cells=tuple(sorted(conflicts)[:representative_limit]),
        open_faces=tuple(FACE_NAMES[index] for index in open_faces),
    )
    return BoundaryResolution(tuple(face_types), open_faces, summary)


class SurfaceCompletion(ABC):
    """Reserved interface for future fluid/gas surface population closure."""

    @abstractmethod
    def complete(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


class UnavailableSurfaceCompletion(SurfaceCompletion):
    """Explicit fail-fast implementation used until surface closure exists."""

    def complete(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError(
            "SurfaceCompletion is interface-only; free-surface population "
            "completion is not implemented"
        )
