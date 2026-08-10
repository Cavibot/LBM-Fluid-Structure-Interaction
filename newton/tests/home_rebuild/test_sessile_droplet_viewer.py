# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the sessile-droplet visual gate."""

import newton.viewer

from wanphys.examples.lbm.home_rebuild.surface_tension.sessile_droplet_3d_viewer import (
    SessileDropletViewer,
    make_sessile_droplet_config,
    sessile_hemisphere_fill,
    sessile_metrics,
)


def test_sessile_scene_matches_the_visual_acceptance_case() -> None:
    config = make_sessile_droplet_config("cpu")
    fill = sessile_hemisphere_fill(config)
    metrics = sessile_metrics(fill)

    assert config.resolution == (25, 25, 20)
    assert config.radius == 6.25
    assert config.surface_tension == 0.005
    assert config.contact_angle_degrees == 120.0
    assert metrics["height"] == 7
    assert 4.48 < float(metrics["bottom_rms_radius"]) < 4.50


def test_sessile_viewer_null_advances_neutral_contact_line() -> None:
    viewer = newton.viewer.ViewerNull(num_frames=1)
    example = SessileDropletViewer(
        viewer,
        config=make_sessile_droplet_config(
            "cpu", contact_angle_degrees=90.0
        ),
        print_every=0,
    )

    example.step()

    assert example.sim_step == 1
    assert example.maximum_speed < 0.02
    assert example.maximum_volume_drift < 1.0e-2


def test_sessile_viewer_can_batch_steps_between_rendered_frames() -> None:
    viewer = newton.viewer.ViewerNull(num_frames=1)
    example = SessileDropletViewer(
        viewer,
        config=make_sessile_droplet_config(
            "cpu", contact_angle_degrees=90.0
        ),
        print_every=0,
        steps_per_frame=4,
    )

    example.step()

    assert example.sim_step == 4
