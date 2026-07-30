# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Representation-independent contracts for LBM volume-of-fluid state."""

from enum import Enum, IntEnum


class VofCellType(IntEnum):
    """Liquid/gas occupancy classification, independent of solid geometry."""

    GAS = 0
    INTERFACE = 1
    LIQUID = 2


class VofMassScheme(str, Enum):
    """Audited VOF mass-exchange discretizations."""

    FSLBM_NEIGHBOR = "fslbm_neighbor"


class VofRuntimeProfile(str, Enum):
    """Validation and synchronization policy for authoritative VOF steps."""

    STRICT = "strict"
    SAMPLED = "sampled"
    DEVICE = "device"
    OFF = "off"


def normalize_vof_mass_scheme(value: str | VofMassScheme) -> VofMassScheme:
    """Return the sole P2 scheme or reject an unaudited alternative."""

    if isinstance(value, VofMassScheme):
        return value
    try:
        return VofMassScheme(str(value).lower())
    except ValueError as exc:
        raise ValueError(
            "Unknown VOF mass scheme "
            f"{value!r}; expected {VofMassScheme.FSLBM_NEIGHBOR.value!r}"
        ) from exc


def normalize_vof_runtime_profile(
    value: str | VofRuntimeProfile,
) -> VofRuntimeProfile:
    """Return a frozen VOF runtime profile or reject an unknown policy."""

    if isinstance(value, VofRuntimeProfile):
        return value
    try:
        return VofRuntimeProfile(str(value).lower())
    except ValueError as exc:
        expected = ", ".join(repr(item.value) for item in VofRuntimeProfile)
        raise ValueError(
            f"Unknown VOF runtime profile {value!r}; expected one of {expected}"
        ) from exc


__all__ = [
    "VofCellType",
    "VofMassScheme",
    "VofRuntimeProfile",
    "normalize_vof_mass_scheme",
    "normalize_vof_runtime_profile",
]
