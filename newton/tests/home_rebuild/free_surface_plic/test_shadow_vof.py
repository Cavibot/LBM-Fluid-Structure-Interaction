# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallDynamicTopologyStepper,
    FslWallMask,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.shadow_vof import (
    ShadowPlicVof,
)


def _fingerprint(fluid: HomeCoreState, fsl: FslState) -> list[np.ndarray]:
    return [
        fluid.moments.numpy().copy(),
        fsl.mass.numpy().copy(),
        fsl.fill_level.numpy().copy(),
        fsl.excess_mass.numpy().copy(),
        fsl.flags.numpy().copy(),
    ]


def test_shadow_vof_does_not_modify_baseline_state() -> None:
    model = HomeCoreModel(
        fluid_grid_res=(10, 9, 8),
        fluid_grid_cell_size=0.1,
        kinematic_viscosity=2.0e-4,
        body_acceleration=(0.0, -5.0e-6, 0.0),
        device="cpu",
    )
    walls = FslWallMask.closed_box(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    config = GravityColumnConfig(
        resolution_x=10,
        resolution_y=9,
        resolution_z=8,
        column_end_x=3,
        column_end_y=6,
        physical_viscosity=2.0e-4,
        physical_gravity_y=-5.0e-6,
        sample_steps=(0,),
        device="cpu",
    )
    fill = _column_fill(config, walls)
    walls.initialize_hydrostatic(
        fluid_a,
        fsl_a,
        fill,
        gas_density=1.0,
        gravity_axis=1,
        surface_coordinate=7.0,
    )
    fluid_b.copy_from(fluid_a)
    fsl_b.copy_from(fsl_a)
    stepper = FslWallDynamicTopologyStepper(model, walls)
    shadow = ShadowPlicVof(model, walls, fsl_a.fill_level)
    stepper.step(fluid_a, fsl_a, fluid_b, fsl_b, model.time_step)
    before = _fingerprint(fluid_b, fsl_b)

    diagnostics = shadow.advance(fluid_b, fsl_b)

    after = _fingerprint(fluid_b, fsl_b)
    assert diagnostics.step == 1
    assert diagnostics.maximum_courant >= 0.0
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_projected_shadow_vof_does_not_modify_baseline_state() -> None:
    model = HomeCoreModel(
        fluid_grid_res=(10, 9, 8),
        fluid_grid_cell_size=0.1,
        kinematic_viscosity=2.0e-4,
        body_acceleration=(0.0, -5.0e-6, 0.0),
        device="cpu",
    )
    walls = FslWallMask.closed_box(model)
    fluid_a = HomeCoreState(model)
    fluid_b = HomeCoreState(model)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    config = GravityColumnConfig(
        resolution_x=10,
        resolution_y=9,
        resolution_z=8,
        column_end_x=3,
        column_end_y=6,
        physical_viscosity=2.0e-4,
        physical_gravity_y=-5.0e-6,
        sample_steps=(0,),
        device="cpu",
    )
    fill = _column_fill(config, walls)
    walls.initialize_hydrostatic(
        fluid_a,
        fsl_a,
        fill,
        gas_density=1.0,
        gravity_axis=1,
        surface_coordinate=7.0,
    )
    fluid_b.copy_from(fluid_a)
    fsl_b.copy_from(fsl_a)
    FslWallDynamicTopologyStepper(model, walls).step(
        fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
    )
    before = _fingerprint(fluid_b, fsl_b)
    shadow = ShadowPlicVof(
        model, walls, fsl_a.fill_level, project_courant=True
    )

    diagnostics = shadow.advance(fluid_b, fsl_b)

    after = _fingerprint(fluid_b, fsl_b)
    assert diagnostics.projected_maximum_divergence is not None
    assert diagnostics.projected_maximum_divergence <= 5.0e-7
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_array_equal(actual, expected)
