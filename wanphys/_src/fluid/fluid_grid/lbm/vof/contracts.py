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


__all__ = [
    "VofCellType",
    "VofMassScheme",
    "normalize_vof_mass_scheme",
]
