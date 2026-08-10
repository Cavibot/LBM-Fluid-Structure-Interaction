# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from wanphys.examples.lbm.home_rebuild.home_core.taylor_green import (
    TaylorGreenConfig,
    frame_metrics,
    projected_amplitude,
    taylor_green_initial_state,
)


def test_config_rejects_ambiguous_sampling() -> None:
    with pytest.raises(ValueError, match="start at zero"):
        TaylorGreenConfig(sample_steps=(1, 2)).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        TaylorGreenConfig(sample_steps=(0, 2, 2)).validate()


def test_initial_state_recovers_declared_amplitude_and_metrics() -> None:
    config = TaylorGreenConfig(
        resolution=16,
        viscosity=0.08,
        initial_amplitude=0.03,
        sample_steps=(0, 1),
        device="cpu",
    )
    moments, basis_x, basis_y = taylor_green_initial_state(config)
    assert projected_amplitude(moments, basis_x, basis_y) == pytest.approx(
        config.initial_amplitude, abs=1.0e-15
    )
    metrics = frame_metrics(
        step=0,
        moments=moments,
        basis_x=basis_x,
        basis_y=basis_y,
        config=config,
        initial_mass=float(np.sum(moments[..., 0])),
    )
    assert metrics["relative_mass_drift"] == pytest.approx(0.0)
    assert metrics["relative_amplitude_error"] == pytest.approx(0.0, abs=1.0e-15)
    assert metrics["max_abs_vorticity"] > 0.0
