# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Executable reference for distance-aware HOME-LBM wall closure."""

from __future__ import annotations

from .constants import CS2


def linear_bouzidi_reflection(
    incoming: float,
    outgoing: float,
    support_incoming: float | None,
    fraction: float,
    weight: float,
    density: float,
    direction_dot_wall_velocity: float,
) -> float:
    """Return the reflected population for one cut link.

    ``fraction`` is the distance from the fluid node to the wall divided by
    the link length.  ``incoming`` travels from the fluid node toward the
    wall, while ``outgoing`` travels in the reflected direction.  The
    formulas are the linear branches of Bouzidi et al. with the moving-wall
    correction written in lattice units.
    """

    q = float(fraction)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"cut-link fraction must be in [0, 1], got {q}")
    if density <= 0.0:
        raise ValueError(f"density must be positive, got {density}")

    wall_correction = (
        2.0 * float(weight) * float(density) * float(direction_dot_wall_velocity) / CS2
    )
    if q < 0.5:
        if support_incoming is None:
            raise ValueError("fraction < 0.5 requires a second fluid support node")
        return (
            2.0 * q * float(incoming)
            + (1.0 - 2.0 * q) * float(support_incoming)
            + wall_correction
        )

    inverse_distance = 1.0 / (2.0 * q)
    return inverse_distance * (float(incoming) + wall_correction) + (
        1.0 - inverse_distance
    ) * float(outgoing)
