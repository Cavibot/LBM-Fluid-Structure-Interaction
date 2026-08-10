# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only HOME-Free ViewerGL density adapter."""

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_viewer.home_free import HomeFreeRenderField
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_3d_viewer import (
    HOME_CELL_SIZE,
    HOME_PHYSICAL_GRAVITY,
    HOME_PHYSICAL_VISCOSITY,
    HIGH_LATTICE_GRAVITY,
    _gravity_ramp_steps,
    _look_at_angles,
    _maximum_lattice_speed,
    _simulation_cell_size,
    _tank_edges,
    make_viewer_config,
)


def test_home_free_render_field_rotates_y_up_without_mutating_state() -> None:
    fill_np = np.asarray(
        [
            [[0.0, 0.25], [0.5, 0.75], [1.0, 0.1]],
            [[0.2, 0.4], [0.6, 0.8], [1.0, 0.3]],
        ],
        dtype=np.float32,
    )
    flags_np = np.ones(fill_np.shape, dtype=np.int32)
    flags_np[0, 1, 1] = 0
    fill = wp.array(fill_np, dtype=float, device="cpu")
    flags = wp.array(flags_np, dtype=wp.int32, device="cpu")
    fill_before = fill.numpy().copy()
    flags_before = flags.numpy().copy()

    adapter = HomeFreeRenderField(fill_np.shape, device="cpu", gas_flag=0)
    result = adapter.update(fill, flags).numpy()

    expected = fill_np.copy()
    expected[flags_np == 0] = 0.0
    np.testing.assert_allclose(result, np.transpose(expected, (0, 2, 1)))
    np.testing.assert_array_equal(fill.numpy(), fill_before)
    np.testing.assert_array_equal(flags.numpy(), flags_before)


def test_home_free_render_field_rejects_mismatched_shape() -> None:
    adapter = HomeFreeRenderField((2, 3, 4), device="cpu")
    fill = wp.zeros((2, 3, 5), dtype=float, device="cpu")
    flags = wp.zeros((2, 3, 5), dtype=wp.int32, device="cpu")

    try:
        adapter.update(fill, flags)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("shape mismatch must fail")


def test_home_free_render_field_preserves_z_up_fsi_layout() -> None:
    fill = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 23.0
    flags = np.ones((2, 3, 4), dtype=np.int32)
    flags[0, 0, 0] = 0
    adapter = HomeFreeRenderField(
        (2, 3, 4), device="cpu", source_up_axis=2
    )

    actual = adapter.update(
        wp.array(fill, dtype=float, device="cpu"),
        wp.array(flags, dtype=wp.int32, device="cpu"),
    ).numpy()

    expected = fill.copy()
    expected[0, 0, 0] = 0.0
    assert adapter.render_shape == (2, 3, 4)
    np.testing.assert_allclose(actual, expected)


def test_home_free_render_field_rejects_unknown_up_axis() -> None:
    with pytest.raises(ValueError, match="source_up_axis"):
        HomeFreeRenderField((2, 3, 4), device="cpu", source_up_axis=0)


def test_home_free_viewer_profiles_preserve_linkwise_scene_parameters() -> None:
    preview = make_viewer_config("preview", "cpu")
    formal = make_viewer_config("formal", "cpu")

    assert (preview.resolution_x, preview.resolution_y, preview.resolution_z) == (
        96,
        56,
        16,
    )
    assert (preview.column_end_x, preview.column_end_y) == (24, 35)
    assert preview.physical_gravity_y == -5.0e-6
    assert preview.physical_viscosity == 2.0e-4
    assert (formal.resolution_x, formal.resolution_y, formal.resolution_z) == (
        160,
        96,
        64,
    )
    assert (formal.column_end_x, formal.column_end_y) == (40, 60)
    assert formal.physical_gravity_y == -5.0e-6
    assert formal.physical_viscosity == 2.0e-4


def test_viewer_default_profile_matches_legacy_lattice_parameters() -> None:
    config = make_viewer_config("viewer-default", "cpu")

    assert (config.resolution_x, config.resolution_y, config.resolution_z) == (
        128,
        128,
        128,
    )
    assert config.column_end_x == 31
    assert config.column_end_y == 125
    assert _simulation_cell_size(config) == HOME_CELL_SIZE
    assert _maximum_lattice_speed(config) == 0.2
    assert (
        config.physical_viscosity / HOME_CELL_SIZE**2
        == pytest.approx(0.02)
    )
    assert (
        config.physical_gravity_y / HOME_CELL_SIZE
        == pytest.approx(-5.0e-5)
    )
    assert config.physical_viscosity == HOME_PHYSICAL_VISCOSITY
    assert config.physical_gravity_y == HOME_PHYSICAL_GRAVITY


def test_high_gravity_profile_changes_only_gravity_controls() -> None:
    stable = make_viewer_config("viewer-default", "cpu")
    high = make_viewer_config("viewer-high-gravity", "cpu")

    assert (high.resolution_x, high.resolution_y, high.resolution_z) == (
        stable.resolution_x,
        stable.resolution_y,
        stable.resolution_z,
    )
    assert (high.column_end_x, high.column_end_y) == (
        stable.column_end_x,
        stable.column_end_y,
    )
    assert high.physical_viscosity == stable.physical_viscosity
    assert high.physical_gravity_y / HOME_CELL_SIZE == pytest.approx(
        HIGH_LATTICE_GRAVITY
    )
    assert _gravity_ramp_steps(high) == 60
    assert _maximum_lattice_speed(high) == 0.55


def test_look_at_angles_point_toward_target_in_z_up_viewer() -> None:
    pitch, yaw = _look_at_angles((1.0, -1.0, 1.0), (0.0, 0.0, 0.0))

    assert pitch == pytest.approx(-35.2643897)
    assert yaw == pytest.approx(135.0)


def test_tank_edges_build_twelve_segments() -> None:
    starts, ends = _tank_edges((2.0, 3.0, 4.0))

    assert starts.shape == (12, 3)
    assert ends.shape == (12, 3)
