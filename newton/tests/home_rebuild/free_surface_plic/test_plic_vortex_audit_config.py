# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

from wanphys.examples.lbm.home_rebuild.geometric_vof.plic_vortex_convergence import (
    PlicVortexAuditConfig,
    _fitted_order,
    _observed_orders,
    _shape_anisotropy,
)


def test_vortex_audit_config_has_ordered_resolution_and_courant_families() -> None:
    config = PlicVortexAuditConfig()

    assert config.resolutions == tuple(sorted(config.resolutions))
    assert len(config.resolutions) >= 3
    assert len(config.courant_limits) >= 2
    assert max(config.courant_limits) >= 0.35


def test_observed_order_uses_resolution_ratio() -> None:
    cases = [
        {
            "resolution": 32,
            "courant_limit": 0.18,
            "relative_l1_shape_error": 0.04,
        },
        {
            "resolution": 64,
            "courant_limit": 0.18,
            "relative_l1_shape_error": 0.01,
        },
    ]

    assert math.isclose(_observed_orders(cases, 0.18)[0], 2.0)
    assert math.isclose(_fitted_order(cases, 0.18), 2.0)


def test_shape_anisotropy_distinguishes_circle_and_stretched_body() -> None:
    y, x = np.indices((41, 41))
    circle = (((x - 20) ** 2 + (y - 20) ** 2) <= 8**2).astype(np.float32)
    ellipse = (((x - 20) / 3) ** 2 + ((y - 20) / 10) ** 2 <= 1).astype(
        np.float32
    )
    circle_volume = np.zeros((41, 41, 3), dtype=np.float32)
    ellipse_volume = np.zeros_like(circle_volume)
    circle_volume[:, :, 1] = circle
    ellipse_volume[:, :, 1] = ellipse

    assert _shape_anisotropy(circle_volume) < 1.05
    assert _shape_anisotropy(ellipse_volume) > 3.0
