# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Committed HOME-FREE cell categories."""

from enum import IntEnum


class HomeFreeCellFlag(IntEnum):
    """Stable cell states; topology transitions use separate scratch flags."""

    GAS = 0
    INTERFACE = 1
    LIQUID = 2
    SOLID = 3


class HomeFreeTransition(IntEnum):
    """Scratch-only topology intents resolved before a state is committed."""

    KEEP = 0
    INTERFACE_TO_LIQUID = 1
    INTERFACE_TO_GAS = 2
    GAS_TO_INTERFACE = 3
    LIQUID_TO_INTERFACE = 4
