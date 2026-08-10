# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import HomeLbmDomain, HomeLbmModel
from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel


def _initial_moments(resolution: int) -> np.ndarray:
    coordinates = 2.0 * np.pi * (np.arange(resolution) + 0.5) / resolution
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    velocity = np.zeros((resolution, resolution, 1, 3), dtype=np.float64)
    velocity[:, :, 0, 0] = 0.06 * np.sin(x) * np.cos(y)
    velocity[:, :, 0, 1] = -0.06 * np.cos(x) * np.sin(y)
    rho = 1.0 + 0.002 * np.cos(2.0 * x) * np.cos(y)
    rho = rho[:, :, None]
    moments = np.zeros(rho.shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * velocity
    moments[..., 4] = rho * velocity[..., 0] ** 2
    moments[..., 5] = rho * velocity[..., 1] ** 2
    moments[..., 6] = rho * velocity[..., 2] ** 2
    moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
    moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
    return moments


def _upload(state: object, moments: np.ndarray) -> None:
    values = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(values, dtype=float, device=state.device))


def _download(state: object) -> np.ndarray:
    return state.moments.numpy().astype(np.float64)


def _compare_migration(device: str) -> None:
    resolution = 24
    common = dict(
        fluid_grid_res=(resolution, resolution, 1),
        fluid_grid_cell_size=1.0,
        time_step=1.0,
        kinematic_viscosity=0.003,
        body_acceleration=(2.0e-7, -1.0e-7, 0.0),
        periodic=(True, True, True),
        device=device,
    )
    old_domain = HomeLbmDomain(HomeLbmModel(**common))
    new_domain = HomeCoreDomain(HomeCoreModel(**common))
    initial = _initial_moments(resolution)
    _upload(old_domain.create_state(), initial)
    _upload(new_domain.create_state(), initial)

    for _ in range(20):
        old_domain.step(1.0)
        new_domain.step(1.0)
    wp.synchronize_device(device)

    old_values = _download(old_domain.state)
    new_values = _download(new_domain.state)
    np.testing.assert_allclose(new_values, old_values, rtol=2.0e-6, atol=2.0e-7)
    diagnostics = new_domain.solver.validate_state(new_domain.state)
    assert diagnostics.invalid_cell_count == 0


def test_new_core_has_no_import_dependency_on_old_home_lbm() -> None:
    core_dir = Path(__file__).parents[4] / "wanphys/_src/fluid/fluid_grid/home_rebuild/core"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in core_dir.glob("*.py"))
    assert "home_lbm" not in sources


def test_model_rejects_unimplemented_boundary_composition() -> None:
    with pytest.raises(ValueError, match="periodic-only"):
        HomeCoreModel(fluid_grid_res=(8, 8, 1), periodic=(True, False, True), device="cpu")


def test_core_rejects_invalid_speed_step_and_density() -> None:
    model = HomeCoreModel(fluid_grid_res=(8, 8, 1), max_lattice_speed=0.1, device="cpu")
    domain = HomeCoreDomain(model)
    state = domain.create_state()
    with pytest.raises(ValueError, match="exceeds configured limit"):
        domain.solver.initialize_uniform_lattice(state, velocity=(0.11, 0.0, 0.0))

    domain.solver.initialize_uniform_lattice(state)
    with pytest.raises(ValueError, match="does not match"):
        domain.step(0.5)

    values = np.zeros(10 * state.cell_count, dtype=np.float32)
    values[: state.cell_count] = 1.0
    values[0] = -1.0
    state.moments.assign(values)
    with pytest.raises(FloatingPointError, match="invalid cells"):
        domain.solver.validate_state(state)


def test_cpu_matches_audited_home_candidate() -> None:
    _compare_migration("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device is required")
def test_cuda_matches_audited_home_candidate() -> None:
    _compare_migration("cuda:0")
