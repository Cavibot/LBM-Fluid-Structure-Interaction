# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslMassAdvector,
    FslState,
    link_mass_exchange,
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
    ux = 0.04 + 0.006 * np.sin(xx) * np.cos(yy)
    uy = -0.004 * np.cos(xx) * np.sin(yy)
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


def _run_oracle_alignment(device: str) -> None:
    shape = (20, 12, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    moments = _smooth_moments(shape)
    _upload_moments(fluid, moments)
    fsl = FslState(model)
    fill = _slab_fill(shape)
    fsl.initialize_from_fill_level(fluid, fill)

    expected = link_mass_exchange(
        moments.astype(np.float64),
        fsl.mass.numpy().astype(np.float64),
        fsl.fill_level.numpy().astype(np.float64),
        fsl.flags.numpy(),
    )
    advector = FslMassAdvector(model)
    advector.advect(fluid, fsl)
    diagnostics = advector.validate_result()
    actual = advector.advected_mass.numpy().astype(np.float64)

    np.testing.assert_allclose(actual, expected, rtol=2.5e-6, atol=3.0e-7)
    assert diagnostics.invalid_cell_count == 0
    assert diagnostics.direct_liquid_gas_link_count == 0
    assert abs(diagnostics.relative_mass_drift) < 2.0e-7


def test_warp_cpu_matches_numpy_oracle() -> None:
    _run_oracle_alignment("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device is required")
def test_warp_cuda_matches_numpy_oracle() -> None:
    _run_oracle_alignment("cuda:0")


def test_static_and_translating_planes_preserve_expected_exchange() -> None:
    shape = (16, 8, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    fsl = FslState(model)
    fill = _slab_fill(shape)
    advector = FslMassAdvector(model)

    domain.solver.initialize_uniform_lattice(fluid)
    fsl.initialize_from_fill_level(fluid, fill)
    advector.advect(fluid, fsl)
    static = advector.validate_result()
    assert static.max_abs_cell_delta == pytest.approx(0.0)
    assert static.relative_mass_drift == pytest.approx(0.0)

    domain.solver.initialize_uniform_lattice(fluid, velocity=(0.05, 0.0, 0.0))
    fsl.initialize_from_fill_level(fluid, fill)
    advector.advect(fluid, fsl)
    moving = advector.validate_result()
    delta = advector.advected_mass.numpy() - fsl.mass.numpy()
    assert float(np.mean(delta[2])) == pytest.approx(-0.05, abs=2.0e-7)
    assert float(np.mean(delta[shape[0] - 3])) == pytest.approx(0.05, abs=2.0e-7)
    assert moving.relative_mass_drift == pytest.approx(0.0, abs=2.0e-7)


def test_device_kernel_reports_direct_liquid_gas_links() -> None:
    shape = (12, 6, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid)
    fsl = FslState(model)
    fill = _slab_fill(shape)
    fsl.initialize_from_fill_level(fluid, fill)
    flags = fsl.flags.numpy()
    mass = fsl.mass.numpy()
    fill = fsl.fill_level.numpy()
    flags[2, :, :] = int(FslCellFlag.GAS)
    mass[2, :, :] = 0.0
    fill[2, :, :] = 0.0
    fsl.flags.assign(flags)
    fsl.mass.assign(mass)
    fsl.fill_level.assign(fill)

    advector = FslMassAdvector(model)
    advector.advect(fluid, fsl)
    with pytest.raises(RuntimeError, match="direct liquid-gas links"):
        advector.validate_result()
