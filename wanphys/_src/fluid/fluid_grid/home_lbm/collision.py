# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference moment collision for single-phase HOME-LBM."""

from __future__ import annotations

import numpy as np

from .constants import MOMENT_COUNT


def collide_central_moments(
    moments: np.ndarray,
    shear_omega: float,
    force_density: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the closed-form NOCM collision from paper Eqs. (21-23).

    Input momentum is the raw post-streaming first moment. With a force, the
    temporary physical velocity from paper Eq. (7) includes ``F/2`` and the
    stored post-collision momentum receives the other half through Eq. (22).
    """

    values = np.asarray(moments, dtype=np.float64)
    if values.shape[-1] != MOMENT_COUNT:
        raise ValueError(f"expected last dimension of size {MOMENT_COUNT}, got {values.shape}")
    if np.any(values[..., 0] <= 0.0):
        raise ValueError("HOME collision requires strictly positive density")
    if not 0.0 < shear_omega < 2.0:
        raise ValueError(f"shear_omega must be in (0, 2), got {shear_omega}")

    result = values.copy()
    rho = values[..., 0]
    if force_density is None:
        force = np.zeros(values.shape[:-1] + (3,), dtype=np.float64)
    else:
        force = np.broadcast_to(np.asarray(force_density, dtype=np.float64), values.shape[:-1] + (3,))
        if not np.isfinite(force).all():
            raise ValueError("force_density must contain only finite values")
    velocity = (values[..., 1:4] + 0.5 * force) / rho[..., None]
    second = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    second[..., 0, 0] = values[..., 4] / rho
    second[..., 1, 1] = values[..., 5] / rho
    second[..., 2, 2] = values[..., 6] / rho
    second[..., 0, 1] = second[..., 1, 0] = values[..., 7] / rho
    second[..., 0, 2] = second[..., 2, 0] = values[..., 8] / rho
    second[..., 1, 2] = second[..., 2, 1] = values[..., 9] / rho

    equilibrium = np.einsum("...a,...b->...ab", velocity, velocity)
    trace_second = np.trace(second, axis1=-2, axis2=-1)
    trace_equilibrium = np.einsum("...a,...a->...", velocity, velocity)
    identity = np.eye(3, dtype=np.float64)
    deviatoric_nonequilibrium = (
        second
        - trace_second[..., None, None] * identity / 3.0
        - equilibrium
        + trace_equilibrium[..., None, None] * identity / 3.0
    )
    keep = 1.0 - shear_omega
    post_second = equilibrium + keep * deviatoric_nonequilibrium
    force_velocity = force * velocity
    force_power = np.einsum("...a,...a->...", force, velocity)
    for axis in range(3):
        post_second[..., axis, axis] += (
            force_velocity[..., axis]
            + keep * (3.0 * force_velocity[..., axis] - force_power) / 3.0
        ) / rho
    off_diagonal_factor = (1.0 - 0.5 * shear_omega) / rho
    post_second[..., 0, 1] += off_diagonal_factor * (
        force[..., 0] * velocity[..., 1] + force[..., 1] * velocity[..., 0]
    )
    post_second[..., 1, 0] = post_second[..., 0, 1]
    post_second[..., 0, 2] += off_diagonal_factor * (
        force[..., 0] * velocity[..., 2] + force[..., 2] * velocity[..., 0]
    )
    post_second[..., 2, 0] = post_second[..., 0, 2]
    post_second[..., 1, 2] += off_diagonal_factor * (
        force[..., 1] * velocity[..., 2] + force[..., 2] * velocity[..., 1]
    )
    post_second[..., 2, 1] = post_second[..., 1, 2]

    result[..., 1:4] = values[..., 1:4] + force
    result[..., 4] = rho * post_second[..., 0, 0]
    result[..., 5] = rho * post_second[..., 1, 1]
    result[..., 6] = rho * post_second[..., 2, 2]
    result[..., 7] = rho * post_second[..., 0, 1]
    result[..., 8] = rho * post_second[..., 0, 2]
    result[..., 9] = rho * post_second[..., 1, 2]
    return result
