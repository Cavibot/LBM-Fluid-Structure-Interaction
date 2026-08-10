# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallMask,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    _column_fill,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.plic_diagnostic_viewer import (
    PlicNormalRenderField,
    ReadOnlyPlicObserver,
)


def _state_arrays(fluid: HomeCoreState, fsl: FslState) -> list[np.ndarray]:
    return [
        fluid.moments.numpy().copy(),
        fsl.mass.numpy().copy(),
        fsl.fill_level.numpy().copy(),
        fsl.excess_mass.numpy().copy(),
        fsl.flags.numpy().copy(),
    ]


def test_plic_observer_does_not_modify_home_or_fsl_state() -> None:
    model = HomeCoreModel(
        fluid_grid_res=(8, 8, 8),
        body_acceleration=(0.0, -5.0e-6, 0.0),
        device="cpu",
    )
    walls = FslWallMask.closed_box(model)
    fluid = HomeCoreState(model)
    fsl = FslState(model)
    config = GravityColumnConfig(
        resolution_x=8,
        resolution_y=8,
        resolution_z=8,
        column_end_x=2,
        column_end_y=5,
        sample_steps=(0,),
        device="cpu",
    )
    fill = _column_fill(config, walls)
    walls.initialize_hydrostatic(
        fluid,
        fsl,
        fill,
        gas_density=1.0,
        gravity_axis=1,
        surface_coordinate=6.0,
    )
    before = _state_arrays(fluid, fsl)

    observer = ReadOnlyPlicObserver(model, walls)
    diagnostics = observer.observe(fsl)

    after = _state_arrays(fluid, fsl)
    assert diagnostics.interface_cell_count > 0
    assert diagnostics.invalid_interface_count == 0
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_sparse_normal_overlay_rotates_y_up_to_viewer_z_up() -> None:
    model = HomeCoreModel(fluid_grid_res=(4, 4, 4), device="cpu")
    walls = FslWallMask.closed_box(model)
    observer = ReadOnlyPlicObserver(model, walls)
    valid = np.zeros((4, 4, 4), dtype=np.int32)
    normal = np.zeros((4, 4, 4, 3), dtype=np.float32)
    valid[0, 0, 0] = 1
    normal[0, 0, 0] = (0.0, 1.0, 0.0)
    observer.geometry.valid.assign(valid)
    observer.geometry.normal.assign(normal)
    overlay = PlicNormalRenderField(
        (4, 4, 4), device="cpu", stride=2, cell_size=0.1
    )

    starts, ends = overlay.update(observer.geometry)
    delta = ends.numpy()[0] - starts.numpy()[0]

    np.testing.assert_allclose(starts.numpy()[0], (0.05, 0.05, 0.05))
    np.testing.assert_allclose(delta, (0.0, 0.0, 0.18), atol=1.0e-6)
