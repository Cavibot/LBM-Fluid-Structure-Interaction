# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Representation-independent contracts for LBM volume-of-fluid state."""

from enum import IntEnum


class VofCellType(IntEnum):
    """Liquid/gas occupancy classification, independent of solid geometry."""

    GAS = 0
    INTERFACE = 1
    LIQUID = 2


__all__ = ["VofCellType"]
