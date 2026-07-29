# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Public configuration axes and capability contracts for the LBM solver.

This module deliberately contains no Warp code.  It is the single host-side
source of truth used by configuration validation, solver dispatch, tests and
documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Encoding(_StringEnum):
    FULLF = "fullf"
    HOME = "home"


class Collision(_StringEnum):
    SRT = "srt"
    TRT = "trt"
    RAW_MRT = "raw_mrt"
    NOCM_MRT = "nocm_mrt"


class CollisionSpace(_StringEnum):
    POPULATION = "population"
    EVEN_ODD_POPULATION = "even_odd_population"
    RAW_MOMENT = "raw_moment"
    NOCM_MOMENT = "nocm_moment"
    HOME_ENCODED_MOMENT = "home_encoded_moment"


class InterfaceModel(_StringEnum):
    OFF = "off"
    SHAN_CHEN = "shan_chen"
    VOF = "vof"


class ForceModel(_StringEnum):
    NONE = "none"
    GRAVITY = "gravity"
    SHAN_CHEN = "shan_chen"
    GRAVITY_SHAN_CHEN = "gravity+shan_chen"


class BoundaryModel(_StringEnum):
    PERIODIC = "periodic"
    BOUNCE_BACK = "bounce_back"
    ZOU_HE = "zou_he"
    PRESSURE = "pressure"
    CONVECTIVE = "convective"
    MOVING_WALL = "moving_wall"
    CUT_LINK = "cut_link"
    SURFACE = "surface"


class CapabilityStatus(_StringEnum):
    SUPPORTED = "supported"
    PLANNED = "planned"
    FAIL_FAST = "fail-fast"
    RESEARCH = "research"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class CollisionContract:
    """The representation manual every collision implementation must expose."""

    collision: Collision
    encoding: Encoding
    space: CollisionSpace
    pre_collision_input: str
    force_translation: str
    native_output: str


@dataclass(frozen=True)
class CapabilityKey:
    encoding: Encoding
    collision: Collision
    force_model: ForceModel


@dataclass(frozen=True)
class CollisionContext:
    """Host-side bundle passed conceptually to every collision executor.

    Warp kernels still receive the device arrays as positional inputs, but the
    solver owns exactly one context with this representation-independent
    layout.
    """

    populations: Any
    moments: tuple[Any, ...]
    rho: Any
    momentum: tuple[Any, Any, Any]
    velocity: tuple[Any, Any, Any]
    force: tuple[Any, Any, Any]


_COLLISION_ALIASES: Final[dict[str, Collision]] = {
    "home_nocm": Collision.NOCM_MRT,
}


def normalize_encoding(value: str | Encoding) -> Encoding:
    try:
        return Encoding(str(value).lower())
    except ValueError as exc:
        expected = ", ".join(item.value for item in Encoding)
        raise ValueError(f"Unknown LBM encoding {value!r}; expected one of: {expected}") from exc


def normalize_collision(value: str | Collision) -> tuple[Collision, bool]:
    """Return the canonical collision and whether a deprecated alias was used."""

    lowered = str(value).lower()
    alias = _COLLISION_ALIASES.get(lowered)
    if alias is not None:
        return alias, True
    try:
        return Collision(lowered), False
    except ValueError as exc:
        expected = ", ".join(item.value for item in Collision)
        raise ValueError(f"Unknown LBM collision {value!r}; expected one of: {expected}") from exc


def normalize_interface_model(value: str | InterfaceModel) -> InterfaceModel:
    try:
        return InterfaceModel(str(value).lower())
    except ValueError as exc:
        expected = ", ".join(item.value for item in InterfaceModel)
        raise ValueError(
            f"Unknown LBM interface_model {value!r}; expected one of: {expected}"
        ) from exc


def resolve_force_model(
    interface_model: str | InterfaceModel,
    gravity: tuple[float, float, float],
) -> ForceModel:
    """Derive the internal force pipeline from interface physics and gravity."""

    resolved_interface = normalize_interface_model(interface_model)
    has_gravity = any(component != 0.0 for component in gravity)
    if resolved_interface is InterfaceModel.SHAN_CHEN:
        if has_gravity:
            return ForceModel.GRAVITY_SHAN_CHEN
        return ForceModel.SHAN_CHEN
    if has_gravity:
        return ForceModel.GRAVITY
    return ForceModel.NONE


def _contract_registry() -> dict[tuple[Encoding, Collision], CollisionContract]:
    return {
        (Encoding.FULLF, Collision.SRT): CollisionContract(
            Collision.SRT,
            Encoding.FULLF,
            CollisionSpace.POPULATION,
            "complete f*[Q]",
            "F -> Guo population source S_i",
            "fPost[Q] -> FullF",
        ),
        (Encoding.HOME, Collision.SRT): CollisionContract(
            Collision.SRT,
            Encoding.HOME,
            CollisionSpace.POPULATION,
            "HOME reconstruct -> complete f*[Q]",
            "F -> Guo population source S_i",
            "fPost[Q] -> HOME projection",
        ),
        (Encoding.FULLF, Collision.TRT): CollisionContract(
            Collision.TRT,
            Encoding.FULLF,
            CollisionSpace.EVEN_ODD_POPULATION,
            "complete f*[Q]",
            "F -> even/odd Guo source",
            "fPost[Q] -> FullF",
        ),
        (Encoding.HOME, Collision.TRT): CollisionContract(
            Collision.TRT,
            Encoding.HOME,
            CollisionSpace.EVEN_ODD_POPULATION,
            "HOME reconstruct -> complete f*[Q]",
            "F -> even/odd Guo source",
            "fPost[Q] -> HOME projection",
        ),
        (Encoding.FULLF, Collision.RAW_MRT): CollisionContract(
            Collision.RAW_MRT,
            Encoding.FULLF,
            CollisionSpace.RAW_MOMENT,
            "complete f*[Q] -> raw moments",
            "F -> raw-moment source",
            "inverse transform -> fPost[Q] -> FullF",
        ),
        (Encoding.FULLF, Collision.NOCM_MRT): CollisionContract(
            Collision.NOCM_MRT,
            Encoding.FULLF,
            CollisionSpace.NOCM_MOMENT,
            "complete f*[Q] -> raw -> NOCM moments",
            "F -> NOCM source",
            "inverse transforms -> fPost[Q] -> FullF",
        ),
        (Encoding.HOME, Collision.NOCM_MRT): CollisionContract(
            Collision.NOCM_MRT,
            Encoding.HOME,
            CollisionSpace.HOME_ENCODED_MOMENT,
            "complete f*[Q] -> retained HOME/NOCM moments",
            "F -> HOME/NOCM source",
            "post-collision retained moments -> HOME",
        ),
    }


COLLISION_CONTRACTS: Final[dict[tuple[Encoding, Collision], CollisionContract]] = (
    _contract_registry()
)


def collision_contract(
    encoding: str | Encoding,
    collision: str | Collision,
) -> CollisionContract:
    resolved_encoding = normalize_encoding(encoding)
    resolved_collision, _ = normalize_collision(collision)
    key = (resolved_encoding, resolved_collision)
    try:
        return COLLISION_CONTRACTS[key]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported LBM encoding/collision combination: "
            f"{resolved_encoding.value}+{resolved_collision.value}"
        ) from exc


def capability_status(
    encoding: str | Encoding,
    collision: str | Collision,
    force_model: str | ForceModel,
) -> CapabilityStatus:
    """Return the status of one concrete encoding/collision/force path."""

    resolved_encoding = normalize_encoding(encoding)
    resolved_collision, _ = normalize_collision(collision)
    resolved_force = ForceModel(str(force_model).lower())
    if (resolved_encoding, resolved_collision) not in COLLISION_CONTRACTS:
        return CapabilityStatus.FAIL_FAST
    if resolved_force in (ForceModel.SHAN_CHEN, ForceModel.GRAVITY_SHAN_CHEN):
        if (
            resolved_encoding is Encoding.FULLF
            and resolved_collision in (Collision.SRT, Collision.TRT)
        ):
            return CapabilityStatus.SUPPORTED
        return CapabilityStatus.RESEARCH
    return CapabilityStatus.SUPPORTED


def validate_capability(
    encoding: str | Encoding,
    collision: str | Collision,
    force_model: str | ForceModel,
) -> CapabilityStatus:
    """Validate a path and return its non-failing capability status."""

    contract = collision_contract(encoding, collision)
    status = capability_status(contract.encoding, contract.collision, force_model)
    if status is CapabilityStatus.FAIL_FAST:
        raise NotImplementedError(
            f"Unsupported LBM capability: encoding={contract.encoding.value}, "
            f"collision={contract.collision.value}, force_model={force_model}"
        )
    return status


BOUNDARY_CAPABILITIES: Final[dict[BoundaryModel, CapabilityStatus]] = {
    BoundaryModel.PERIODIC: CapabilityStatus.SUPPORTED,
    BoundaryModel.BOUNCE_BACK: CapabilityStatus.SUPPORTED,
    BoundaryModel.ZOU_HE: CapabilityStatus.SUPPORTED,
    BoundaryModel.PRESSURE: CapabilityStatus.SUPPORTED,
    BoundaryModel.CONVECTIVE: CapabilityStatus.SUPPORTED,
    BoundaryModel.MOVING_WALL: CapabilityStatus.SUPPORTED,
    BoundaryModel.CUT_LINK: CapabilityStatus.SUPPORTED,
    BoundaryModel.SURFACE: CapabilityStatus.PLANNED,
}


def boundary_capability_status(boundary: str | BoundaryModel) -> CapabilityStatus:
    try:
        resolved = BoundaryModel(str(boundary).lower())
    except ValueError as exc:
        expected = ", ".join(item.value for item in BoundaryModel)
        raise ValueError(
            f"Unknown LBM boundary model {boundary!r}; expected one of: {expected}"
        ) from exc
    return BOUNDARY_CAPABILITIES[resolved]
