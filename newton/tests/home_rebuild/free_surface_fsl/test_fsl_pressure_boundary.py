# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_rebuild.core.constants import D3Q27_WEIGHTS
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    classify_fill,
    gas_pressure_boundary_populations,
    only_missing_stream_moments,
)


def _equilibrium_moments(
    shape: tuple[int, int, int], velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    rho = np.ones(shape, dtype=np.float64)
    u = np.asarray(velocity, dtype=np.float64)
    moments = np.zeros(shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * u
    moments[..., 4] = rho * u[0] * u[0]
    moments[..., 5] = rho * u[1] * u[1]
    moments[..., 6] = rho * u[2] * u[2]
    moments[..., 7] = rho * u[0] * u[1]
    moments[..., 8] = rho * u[0] * u[2]
    moments[..., 9] = rho * u[1] * u[2]
    return moments


def _periodic_slab_flags(shape: tuple[int, int, int]) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float64)
    fill[3:13, :, :] = 1.0
    fill[2, :, :] = 0.5
    fill[13, :, :] = 0.5
    return classify_fill(fill)


def test_resting_pressure_boundary_recovers_d3q27_equilibrium() -> None:
    moments = _equilibrium_moments((1, 1, 1))[0, 0, 0]
    reconstructed = gas_pressure_boundary_populations(moments, gas_density=1.0)
    np.testing.assert_allclose(reconstructed, D3Q27_WEIGHTS, rtol=0.0, atol=2.0e-16)


def test_prescribed_gas_pressure_changes_rest_populations_analytically() -> None:
    moments = _equilibrium_moments((1, 1, 1))[0, 0, 0]
    reconstructed = gas_pressure_boundary_populations(moments, gas_density=1.1)
    np.testing.assert_allclose(reconstructed, 1.2 * D3Q27_WEIGHTS, rtol=0.0, atol=3.0e-16)


def test_pressure_boundary_rejects_nonpositive_gas_density() -> None:
    moments = _equilibrium_moments((1, 1, 1))[0, 0, 0]
    with pytest.raises(ValueError, match="finite and positive"):
        gas_pressure_boundary_populations(moments, gas_density=0.0)


def test_only_missing_stream_preserves_static_periodic_slab() -> None:
    shape = (16, 8, 1)
    moments = _equilibrium_moments(shape)
    flags = _periodic_slab_flags(shape)
    result = only_missing_stream_moments(moments, flags, gas_density=1.0)
    active = flags != int(FslCellFlag.GAS)
    np.testing.assert_allclose(result.moments[active], moments[active], rtol=0.0, atol=8.0e-16)
    assert int(np.sum(result.gas_link_count)) == 2 * shape[1] * 9
    assert np.all(result.gas_link_count[flags == int(FslCellFlag.INTERFACE)] == 9)
    assert np.count_nonzero(result.gas_link_count[flags != int(FslCellFlag.INTERFACE)]) == 0


def test_only_missing_stream_exposes_pressure_response_on_interface_only() -> None:
    shape = (16, 8, 1)
    moments = _equilibrium_moments(shape)
    flags = _periodic_slab_flags(shape)
    result = only_missing_stream_moments(moments, flags, gas_density=1.02)
    interface = flags == int(FslCellFlag.INTERFACE)
    liquid = flags == int(FslCellFlag.LIQUID)
    assert np.all(result.moments[interface, 0] > 1.0)
    np.testing.assert_allclose(result.moments[liquid], moments[liquid], rtol=0.0, atol=8.0e-16)
    assert np.max(np.abs(result.moments[interface, 1])) > 1.0e-3


def test_only_missing_stream_rejects_direct_liquid_gas_link() -> None:
    shape = (8, 4, 1)
    moments = _equilibrium_moments(shape)
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[3:5, :, :] = int(FslCellFlag.LIQUID)
    with pytest.raises(ValueError, match="direct liquid-gas link"):
        only_missing_stream_moments(moments, flags)
