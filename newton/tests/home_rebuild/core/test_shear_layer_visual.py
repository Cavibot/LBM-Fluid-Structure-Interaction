# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from wanphys.examples.lbm.home_rebuild.home_core.shear_layer import (
    ShearLayerConfig,
    frame_metrics,
    shear_layer_initial_state,
)


def test_config_requires_resolved_layer_and_ordered_samples() -> None:
    with pytest.raises(ValueError, match="interface is resolved"):
        ShearLayerConfig(resolution=64, layer_thickness=0.01).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        ShearLayerConfig(sample_steps=(0, 2, 1)).validate()


def test_initial_state_is_periodic_subsonic_and_counter_flowing() -> None:
    config = ShearLayerConfig(resolution=64, stream_speed=0.08, sample_steps=(0, 1))
    moments = shear_layer_initial_state(config)
    rho = moments[:, :, 0, 0]
    velocity = moments[:, :, 0, 1:3] / rho[..., None]

    assert np.max(np.linalg.norm(velocity, axis=-1)) < 0.1
    assert np.mean(velocity[:, 32, 0]) > 0.07
    assert np.mean(velocity[:, 0, 0]) < -0.07
    assert np.max(np.abs(velocity[0, :, 1] - velocity[-1, :, 1])) < 4.0e-4

    initial_mass = float(np.sum(rho))
    initial_energy = float(0.5 * np.sum(rho * np.sum(velocity * velocity, axis=-1)))
    metrics = frame_metrics(
        step=0,
        moments=moments,
        initial_mass=initial_mass,
        initial_energy=initial_energy,
    )
    assert metrics["relative_mass_drift"] == pytest.approx(0.0)
    assert metrics["relative_energy_change"] == pytest.approx(0.0)
    assert metrics["enstrophy"] > 0.0
