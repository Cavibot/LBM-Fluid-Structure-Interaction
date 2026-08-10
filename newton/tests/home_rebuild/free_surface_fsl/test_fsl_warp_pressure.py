# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslOnlyMissingStreamer,
    FslState,
    only_missing_stream_moments,
)


def _slab_fill(shape: tuple[int, int, int]) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    fill[3 : shape[0] - 3, :, :] = 1.0
    fill[2, :, :] = 0.5
    fill[shape[0] - 3, :, :] = 0.5
    return fill


def _smooth_moments(shape: tuple[int, int, int]) -> np.ndarray:
    x = 2.0 * np.pi * (np.arange(shape[0]) + 0.5) / shape[0]
    y = 2.0 * np.pi * (np.arange(shape[1]) + 0.5) / shape[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rho = 1.0 + 0.001 * np.cos(xx) * np.cos(yy)
    ux = 0.025 + 0.004 * np.sin(xx) * np.cos(yy)
    uy = -0.003 * np.cos(xx) * np.sin(yy)
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[:, :, 0, 0] = rho
    moments[:, :, 0, 1] = rho * ux
    moments[:, :, 0, 2] = rho * uy
    moments[:, :, 0, 4] = rho * ux * ux
    moments[:, :, 0, 5] = rho * uy * uy
    moments[:, :, 0, 7] = rho * ux * uy
    return moments


def _upload_moments(state: object, moments: np.ndarray) -> None:
    soa = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(soa, dtype=float, device=state.device))


def _download_moments(state: object) -> np.ndarray:
    shape = state.res
    return state.moments.numpy().reshape(10, -1).T.reshape(shape + (10,))


def _run_oracle_alignment(device: str) -> None:
    shape = (20, 12, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    moments = _smooth_moments(shape)
    _upload_moments(fluid, moments)
    fsl = FslState(model)
    fsl.initialize_from_fill_level(fluid, _slab_fill(shape))
    flags = fsl.flags.numpy()
    expected = only_missing_stream_moments(
        moments.astype(np.float64), flags, gas_density=1.015
    )

    streamer = FslOnlyMissingStreamer(model, gas_density=1.015)
    actual_state = streamer.stream(fluid, fsl)
    diagnostics = streamer.validate_result()
    actual = _download_moments(actual_state).astype(np.float64)

    np.testing.assert_allclose(actual, expected.moments, rtol=4.0e-6, atol=4.0e-7)
    np.testing.assert_array_equal(streamer.gas_link_count.numpy(), expected.gas_link_count)
    assert diagnostics.invalid_cell_count == 0
    assert diagnostics.direct_liquid_gas_link_count == 0
    assert diagnostics.gas_link_count == int(np.sum(expected.gas_link_count))


def test_warp_cpu_pressure_stream_matches_numpy_oracle() -> None:
    _run_oracle_alignment("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device is required")
def test_warp_cuda_pressure_stream_matches_numpy_oracle() -> None:
    _run_oracle_alignment("cuda:0")


def test_warp_static_slab_preserves_equilibrium_and_counts_missing_links() -> None:
    shape = (16, 8, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid)
    fsl = FslState(model)
    fsl.initialize_from_fill_level(fluid, _slab_fill(shape))

    streamer = FslOnlyMissingStreamer(model)
    output = streamer.stream(fluid, fsl)
    diagnostics = streamer.validate_result()
    moments = _download_moments(output)
    active = fsl.flags.numpy() != int(FslCellFlag.GAS)
    np.testing.assert_allclose(moments[active, 0], 1.0, rtol=0.0, atol=3.0e-7)
    np.testing.assert_allclose(moments[active, 1:], 0.0, rtol=0.0, atol=2.0e-7)
    assert diagnostics.gas_link_count == 2 * shape[1] * 9


def test_warp_pressure_stream_reports_direct_liquid_gas_links() -> None:
    shape = (12, 6, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid)
    fsl = FslState(model)
    fsl.initialize_from_fill_level(fluid, _slab_fill(shape))
    flags = fsl.flags.numpy()
    flags[2, :, :] = int(FslCellFlag.GAS)
    fsl.flags.assign(flags)

    streamer = FslOnlyMissingStreamer(model)
    streamer.stream(fluid, fsl)
    with pytest.raises(RuntimeError, match="direct liquid-gas links"):
        streamer.validate_result()
