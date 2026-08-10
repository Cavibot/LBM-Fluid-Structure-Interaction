# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import FslWallMask
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.geometric_dambreak_preview import (
    GeometricDambreakConfig,
)


def test_geometric_dambreak_matches_linkwise_lattice_scaling() -> None:
    config = GeometricDambreakConfig(device="cpu")
    model = HomeCoreModel(
        fluid_grid_res=(
            config.resolution_x,
            config.resolution_y,
            config.resolution_z,
        ),
        fluid_grid_cell_size=config.cell_size,
        time_step=config.time_step,
        device=config.device,
        kinematic_viscosity=config.physical_viscosity,
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
    )

    assert model.lattice_viscosity == pytest.approx(0.02)
    assert model.lattice_acceleration == pytest.approx((0.0, -5.0e-5, 0.0))


def test_refined_scene_preserves_physical_scaling() -> None:
    config = GeometricDambreakConfig(
        resolution_x=128,
        resolution_y=112,
        resolution_z=64,
        cell_size=0.05,
        time_step=0.5,
        column_end_x=48,
        column_end_y=68,
        column_offset_x=2,
        device="cpu",
    )
    model = HomeCoreModel(
        fluid_grid_res=(
            config.resolution_x,
            config.resolution_y,
            config.resolution_z,
        ),
        fluid_grid_cell_size=config.cell_size,
        time_step=config.time_step,
        device=config.device,
        kinematic_viscosity=config.physical_viscosity,
        body_acceleration=(0.0, config.physical_gravity_y, 0.0),
    )

    assert config.column_offset_x * config.cell_size == pytest.approx(0.1)
    assert model.lattice_viscosity == pytest.approx(0.04)
    assert model.lattice_acceleration == pytest.approx((0.0, -2.5e-5, 0.0))


def test_offset_column_starts_dry_and_preserves_initial_volume() -> None:
    model = HomeCoreModel(fluid_grid_res=(96, 56, 1), device="cpu")
    walls = FslWallMask.periodic_depth_channel(model)
    baseline = _column_fill(
        GravityColumnConfig(
            resolution_x=96,
            resolution_y=56,
            resolution_z=1,
            periodic_depth=True,
            column_end_x=24,
            column_end_y=34,
            device="cpu",
        ),
        walls,
    )
    offset = _column_fill(
        GravityColumnConfig(
            resolution_x=96,
            resolution_y=56,
            resolution_z=1,
            periodic_depth=True,
            column_end_x=24,
            column_end_y=34,
            column_offset_x=3,
            device="cpu",
        ),
        walls,
    )

    assert np.all(offset[1:4, :, :] == 0.0)
    assert np.max(offset[4, :, :]) == pytest.approx(0.25)
    assert np.sum(offset) == pytest.approx(np.sum(baseline))

    gas = (offset == 0.0) & ~walls.host
    liquid = offset == 1.0
    for axis in (0, 1):
        assert not np.any(liquid & np.roll(gas, 1, axis=axis))
        assert not np.any(liquid & np.roll(gas, -1, axis=axis))


def test_offset_column_rejects_negative_or_out_of_bounds_gap() -> None:
    model = HomeCoreModel(fluid_grid_res=(32, 24, 1), device="cpu")
    walls = FslWallMask.periodic_depth_channel(model)
    with pytest.raises(ValueError, match="nonnegative"):
        _column_fill(
            GravityColumnConfig(
                resolution_x=32,
                resolution_y=24,
                resolution_z=1,
                periodic_depth=True,
                column_end_x=8,
                column_end_y=10,
                column_offset_x=-1,
                device="cpu",
            ),
            walls,
        )
    with pytest.raises(ValueError, match="inside the tank"):
        _column_fill(
            GravityColumnConfig(
                resolution_x=32,
                resolution_y=24,
                resolution_z=1,
                periodic_depth=True,
                column_end_x=20,
                column_end_y=10,
                column_offset_x=10,
                device="cpu",
            ),
            walls,
        )
