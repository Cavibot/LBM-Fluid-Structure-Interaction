# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the isolated Laplace-droplet Viewer scene."""

import math

import numpy as np
import pytest

from wanphys.examples.lbm.home_rebuild.surface_tension.laplace_droplet_3d_viewer import (
    make_laplace_droplet_config,
    sampled_sphere_vof,
)


def test_laplace_viewer_config_preserves_the_accepted_physical_regime() -> None:
    config = make_laplace_droplet_config("cpu")

    assert config.resolution == 33
    assert config.radius == pytest.approx(10.25)
    assert config.kinematic_viscosity == pytest.approx(0.5)
    assert config.surface_tension == pytest.approx(0.001)
    assert config.device == "cpu"
    assert config.surface_tension / config.radius < 1.0e-3


def test_sampled_sphere_is_sharp_symmetric_and_volume_accurate() -> None:
    shape = (17, 17, 17)
    radius = 5.25
    fill = sampled_sphere_vof(shape, radius, samples_per_axis=16)

    assert fill.dtype == np.float32
    assert float(np.min(fill)) == 0.0
    assert float(np.max(fill)) == 1.0
    assert np.any((fill > 0.0) & (fill < 1.0))
    np.testing.assert_array_equal(fill, fill[::-1, :, :])
    np.testing.assert_array_equal(fill, fill[:, ::-1, :])
    np.testing.assert_array_equal(fill, fill[:, :, ::-1])
    represented_volume = float(np.sum(fill, dtype=np.float64))
    analytic_volume = 4.0 * math.pi * radius**3 / 3.0
    assert abs(represented_volume - analytic_volume) / analytic_volume < 2.0e-4
