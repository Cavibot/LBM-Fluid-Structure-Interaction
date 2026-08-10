# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Committed cell categories used by the paper FSL baseline."""

from enum import IntEnum


class FslCellFlag(IntEnum):
    GAS = 0
    INTERFACE = 1
    LIQUID = 2
