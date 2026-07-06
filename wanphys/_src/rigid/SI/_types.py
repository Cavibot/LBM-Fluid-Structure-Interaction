# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys-owned type definitions for the SequentialImpulse solver."""

from __future__ import annotations

from enum import IntEnum


class JointType(IntEnum):
    """Enumeration of joint types."""

    PRISMATIC = 0
    REVOLUTE = 1
    BALL = 2
    FIXED = 3
    FREE = 4
    DISTANCE = 5
    D6 = 6
    CABLE = 7
