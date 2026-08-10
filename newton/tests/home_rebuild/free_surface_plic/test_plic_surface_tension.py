# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl.wall_pipeline import (
    FslWallOnlyMissingStreamer,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicSurfaceTension,
)


def _sphere_fill(
    shape: tuple[int, int, int], *, liquid_inside: bool = True
) -> np.ndarray:
    center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
    coordinates = np.indices(shape, dtype=np.float64)
    distance = np.sqrt(
        sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
    )
    inside = np.clip(0.5 + 5.0 - distance, 0.0, 1.0)
    return (inside if liquid_inside else 1.0 - inside).astype(np.float32)


def _fsl_state(model: HomeCoreModel, walls: FslWallMask, fill: np.ndarray) -> FslState:
    fill = fill.copy()
    fill[walls.host] = 0.0
    flags = np.full(fill.shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill >= 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    state = FslState(model)
    state.fill_level.assign(fill)
    state.mass.assign(fill)
    state.flags.assign(flags)
    return state


def _rest_fluid(model: HomeCoreModel) -> HomeCoreState:
    state = HomeCoreState(model)
    moments = np.zeros((model.nx, model.ny, model.nz, 10), dtype=np.float32)
    moments[..., 0] = 1.0
    state.moments.assign(np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1)))
    return state


def _wall_droplet_fill(shape: tuple[int, int, int]) -> np.ndarray:
    center = np.asarray((8.0, 1.0, 8.0), dtype=np.float64)
    coordinates = np.indices(shape, dtype=np.float64)
    distance = np.sqrt(
        sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
    )
    return np.clip(0.5 + (5.0 - distance) / 2.0, 0.0, 1.0).astype(
        np.float32
    )


def test_spherical_laplace_density_matches_curvature_and_scaling() -> None:
    model = HomeCoreModel(
        fluid_grid_res=(17, 17, 17),
        fluid_grid_cell_size=0.02,
        time_step=0.001,
        reference_density=1000.0,
        device="cpu",
    )
    walls = FslWallMask.closed_box(model)
    state = _fsl_state(model, walls, _sphere_fill(walls.res))
    lattice_gamma = 0.01
    surface = PlicSurfaceTension(
        model,
        walls,
        surface_tension=model.scaling.surface_tension_to_physical(lattice_gamma),
    )

    diagnostics = surface.update(state)

    required = surface.curvature.required.numpy() != 0
    valid = surface.curvature.valid.numpy() != 0
    expected = 1.0 - 6.0 * lattice_gamma * surface.curvature.curvature.numpy()
    np.testing.assert_allclose(
        surface.gas_density.numpy()[required & valid],
        expected[required & valid],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert diagnostics.invalid_gas_density_count == 0
    assert diagnostics.curvature.insufficient_neighbor_count == 0
    assert diagnostics.curvature.ill_conditioned_count == 0
    assert diagnostics.minimum_interface_gas_density > 0.0


def test_nonpositive_laplace_density_is_rejected() -> None:
    model = HomeCoreModel(fluid_grid_res=(17, 17, 17), device="cpu")
    walls = FslWallMask.closed_box(model)
    state = _fsl_state(
        model, walls, _sphere_fill(walls.res, liquid_inside=False)
    )
    surface = PlicSurfaceTension(
        model,
        walls,
        surface_tension=model.scaling.surface_tension_to_physical(2.0),
    )

    with pytest.raises(FloatingPointError, match="nonpositive"):
        surface.update(state)


@pytest.mark.parametrize("contact_angle", (60.0, 90.0, 120.0))
def test_configured_contact_angle_resolves_wall_curvature(
    contact_angle: float,
) -> None:
    model = HomeCoreModel(fluid_grid_res=(17, 12, 17), device="cpu")
    walls = FslWallMask.closed_box(model)
    state = _fsl_state(model, walls, _wall_droplet_fill(walls.res))
    surface = PlicSurfaceTension(
        model,
        walls,
        surface_tension=model.scaling.surface_tension_to_physical(5.0e-4),
        contact_angle_degrees=contact_angle,
    )

    diagnostics = surface.update(state)

    assert diagnostics.geometry.wetting_cell_count > 0
    assert diagnostics.curvature.wall_contact_skipped_count == 0
    assert diagnostics.curvature.wall_contact_fitted_count > 0
    assert diagnostics.curvature.insufficient_neighbor_count == 0
    assert diagnostics.curvature.ill_conditioned_count == 0


def test_uniform_density_field_is_identical_to_scalar_boundary() -> None:
    model = HomeCoreModel(fluid_grid_res=(9, 9, 9), device="cpu")
    walls = FslWallMask.closed_box(model)
    fill = np.zeros(walls.res, dtype=np.float32)
    fill[2:7, 2:7, 2:7] = 1.0
    fill[2, 2:7, 2:7] = 0.5
    fsl = _fsl_state(model, walls, fill)
    fluid = _rest_fluid(model)
    scalar = FslWallOnlyMissingStreamer(model, walls, gas_density=1.02)
    field = FslWallOnlyMissingStreamer(model, walls, gas_density=1.0)
    density = wp.full(walls.res, 1.02, dtype=float, device=model._device)

    scalar_result = scalar.stream(fluid, fsl).moments.numpy().copy()
    field_result = field.stream(
        fluid, fsl, gas_density_field=density
    ).moments.numpy().copy()

    np.testing.assert_array_equal(field_result, scalar_result)
