# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for the isolated low-viscosity scan."""

import pytest

from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_stability_scan import (
    StabilityScanConfig,
    make_scan_model,
)


def test_scan_model_preserves_requested_lattice_parameters() -> None:
    config = StabilityScanConfig(device="cpu", steps=1)
    model = make_scan_model(config, 0.005)

    assert model.lattice_viscosity == pytest.approx(0.005)
    assert model.lattice_acceleration == pytest.approx((0.0, -5.0e-5, 0.0))
    assert model.kinematic_viscosity == pytest.approx(5.0e-5)
    assert model.body_acceleration == pytest.approx((0.0, -5.0e-6, 0.0))


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"sample_every": 0},
        {"lattice_viscosities": ()},
        {"lattice_viscosities": (0.0,)},
    ],
)
def test_scan_config_rejects_invalid_ranges(changes: dict[str, object]) -> None:
    values = StabilityScanConfig().__dict__ | changes
    with pytest.raises(ValueError):
        StabilityScanConfig(**values).validate()
