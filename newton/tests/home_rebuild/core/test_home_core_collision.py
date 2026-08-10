# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    collide_streamed_moments,
)


def _streamed_moments(shape: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(734)
    rho = 1.0 + 0.001 * rng.normal(size=shape)
    velocity = 0.02 * rng.normal(size=shape + (3,))
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * velocity
    moments[..., 4] = rho * velocity[..., 0] ** 2 + 0.0002 * rng.normal(size=shape)
    moments[..., 5] = rho * velocity[..., 1] ** 2 + 0.0002 * rng.normal(size=shape)
    moments[..., 6] = rho * velocity[..., 2] ** 2 + 0.0002 * rng.normal(size=shape)
    moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
    moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
    moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
    return moments


def _upload(state: object, moments: np.ndarray) -> None:
    soa = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(soa, dtype=float, device=state.device))


def _download(state: object) -> np.ndarray:
    return state.moments.numpy().reshape(10, -1).T.reshape(state.res + (10,))


def _run_collision_oracle(device: str) -> None:
    shape = (11, 7, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=0.013,
        body_acceleration=(2.0e-5, -1.0e-5, 0.0),
    )
    domain = HomeCoreDomain(model)
    streamed = domain.create_state()
    output = domain.create_state()
    moments = _streamed_moments(shape)
    _upload(streamed, moments)
    expected = collide_streamed_moments(
        moments,
        shear_omega=model.shear_omega,
        acceleration=model.lattice_acceleration,
    )
    domain.solver.collide(streamed, output, model.time_step)
    diagnostics = domain.solver.validate_state(output)
    np.testing.assert_allclose(_download(output), expected, rtol=3.0e-6, atol=3.0e-7)
    assert diagnostics.invalid_cell_count == 0


def test_cpu_collision_only_matches_numpy_oracle() -> None:
    _run_collision_oracle("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device is required")
def test_cuda_collision_only_matches_numpy_oracle() -> None:
    _run_collision_oracle("cuda:0")


def test_collision_only_rejects_wrong_time_step_without_launching() -> None:
    model = HomeCoreModel(fluid_grid_res=(8, 6, 1), device="cpu")
    domain = HomeCoreDomain(model)
    source = domain.create_state()
    output = domain.create_state()
    with pytest.raises(ValueError, match="does not match configured"):
        domain.solver.collide(source, output, 0.5 * model.time_step)
