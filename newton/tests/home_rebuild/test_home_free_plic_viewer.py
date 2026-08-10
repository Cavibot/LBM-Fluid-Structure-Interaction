# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the qualified projected-geometric Viewer scene."""

import pytest

from wanphys.examples.lbm.home_rebuild.home_free_plic.geometric_dambreak_viewer import (
    make_plic_viewer_config,
)


def test_plic_viewer_uses_the_cuda_qualified_scene() -> None:
    config = make_plic_viewer_config("cpu")

    assert config.mode == "projected-geometric"
    assert (config.resolution_x, config.resolution_y, config.resolution_z) == (
        64,
        56,
        32,
    )
    assert (config.column_end_x, config.column_end_y, config.column_offset_x) == (
        24,
        34,
        1,
    )
    assert config.physical_viscosity / config.cell_size**2 == pytest.approx(0.02)
    assert config.physical_gravity_y / config.cell_size == pytest.approx(-5.0e-5)
    assert config.interface_roundoff_tolerance == 5.0e-6
