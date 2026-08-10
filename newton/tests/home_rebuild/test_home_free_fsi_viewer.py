# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the isolated HOME-Free FSI Viewer scene."""

import newton.viewer
import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeGeometricDomain,
    HomeFreeLegacyDomain,
)
from wanphys.examples.lbm.home_rebuild.fsi.sphere_entry_3d_viewer import (
    SphereEntryViewer,
    make_sphere_entry_config,
)


def test_fsi_viewer_uses_the_cuda_qualified_entry_case() -> None:
    config = make_sphere_entry_config("cpu")

    assert config.resolution == (24, 24, 22)
    assert config.interface_index == 10
    assert config.sphere_center == (12.0, 12.0, 14.8)
    assert config.sphere_radius == 3.0
    assert config.initial_speed_z == -0.15
    assert config.gravity_z == -2.0e-4
    assert config.maximum_steps == 200
    assert config.profile == "balanced"


def test_fsi_viewer_profiles_select_explicit_numerical_paths() -> None:
    balanced = SphereEntryViewer(
        newton.viewer.ViewerNull(num_frames=1),
        config=make_sphere_entry_config("cpu", profile="balanced"),
        print_every=0,
    )
    fast = SphereEntryViewer(
        newton.viewer.ViewerNull(num_frames=1),
        config=make_sphere_entry_config("cpu", profile="fast"),
        print_every=0,
    )
    strict = SphereEntryViewer(
        newton.viewer.ViewerNull(num_frames=1),
        config=make_sphere_entry_config("cpu", profile="strict"),
        print_every=0,
    )

    assert type(balanced.fluid) is HomeFreeGeometricDomain
    assert balanced.fluid.geometric_stepper.geometric_transport.courant_projector is None
    assert type(fast.fluid) is HomeFreeLegacyDomain
    assert type(strict.fluid) is HomeFreeGeometricDomain
    assert strict.fluid.geometric_stepper.geometric_transport.courant_projector is not None


def test_fsi_viewer_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="fast, balanced, or strict"):
        make_sphere_entry_config("cpu", profile="automatic")


def test_fsi_viewer_null_advances_one_coupled_step() -> None:
    viewer = newton.viewer.ViewerNull(num_frames=1)
    example = SphereEntryViewer(
        viewer,
        config=make_sphere_entry_config("cpu"),
        print_every=0,
    )

    initial_position = np.asarray(
        example.rigid.state.get_body_position(example.body), dtype=np.float64
    )
    example.step()
    final_position = np.asarray(
        example.rigid.state.get_body_position(example.body), dtype=np.float64
    )

    assert example.sim_step == 1
    assert final_position[2] < initial_position[2]
    assert example.maximum_mass_drift < 2.0e-5


def test_fsi_viewer_reset_reuses_the_bound_viewer_model() -> None:
    viewer = newton.viewer.ViewerNull(num_frames=1)
    example = SphereEntryViewer(
        viewer,
        config=make_sphere_entry_config("cpu", profile="fast"),
        print_every=0,
    )
    example.step()

    example._reset_simulation()

    reset_position = np.asarray(
        example.rigid.state.get_body_position(example.body), dtype=np.float64
    )
    assert example.sim_step == 0
    assert reset_position[2] == pytest.approx(example.config.sphere_center[2])
    example.step()
    assert example.sim_step == 1
