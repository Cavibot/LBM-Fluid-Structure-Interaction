# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslFixedTopologyStepper,
    FslState,
    collide_streamed_moments,
    link_mass_exchange,
    only_missing_stream_moments,
)


def _slab_fill(
    shape: tuple[int, int, int], *, interface_fill: float = 0.5
) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    start = shape[0] // 4
    end = 3 * shape[0] // 4
    fill[start:end, :, :] = 1.0
    fill[start - 1, :, :] = interface_fill
    fill[end, :, :] = interface_fill
    return fill


def _smooth_moments(shape: tuple[int, int, int]) -> np.ndarray:
    x = 2.0 * np.pi * (np.arange(shape[0]) + 0.5) / shape[0]
    y = 2.0 * np.pi * (np.arange(shape[1]) + 0.5) / shape[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rho = 1.0 + 0.0005 * np.cos(xx) * np.cos(yy)
    ux = 0.012 + 0.002 * np.sin(xx) * np.cos(yy)
    uy = -0.001 * np.cos(xx) * np.sin(yy)
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[:, :, 0, 0] = rho
    moments[:, :, 0, 1] = rho * ux
    moments[:, :, 0, 2] = rho * uy
    moments[:, :, 0, 4] = rho * ux * ux
    moments[:, :, 0, 5] = rho * uy * uy
    moments[:, :, 0, 7] = rho * ux * uy
    return moments


def _upload(state: object, moments: np.ndarray) -> None:
    soa = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    wp.copy(state.moments, wp.array(soa, dtype=float, device=state.device))


def _download(state: object) -> np.ndarray:
    return state.moments.numpy().reshape(10, -1).T.reshape(state.res + (10,))


def _states(model: HomeCoreModel) -> tuple[object, FslState, object, FslState]:
    domain = HomeCoreDomain(model)
    source_fluid = domain.create_state()
    destination_fluid = domain.create_state()
    source_fsl = FslState(model)
    destination_fsl = FslState(model)
    return source_fluid, source_fsl, destination_fluid, destination_fsl


def _run_one_step_oracle(device: str) -> None:
    shape = (24, 10, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape,
        device=device,
        kinematic_viscosity=0.017,
        body_acceleration=(1.0e-6, -2.0e-6, 0.0),
    )
    fluid_in, fsl_in, fluid_out, fsl_out = _states(model)
    moments = _smooth_moments(shape)
    _upload(fluid_in, moments)
    fsl_in.initialize_from_fill_level(fluid_in, _slab_fill(shape))
    flags = fsl_in.flags.numpy()
    expected_mass = link_mass_exchange(
        moments.astype(np.float64),
        fsl_in.mass.numpy().astype(np.float64),
        fsl_in.fill_level.numpy().astype(np.float64),
        flags,
    )
    streamed = only_missing_stream_moments(
        moments.astype(np.float64), flags, gas_density=1.005
    )
    expected_fluid = collide_streamed_moments(
        streamed.moments,
        shear_omega=model.shear_omega,
        acceleration=model.lattice_acceleration,
    )

    stepper = FslFixedTopologyStepper(model, gas_density=1.005)
    diagnostics = stepper.step(
        fluid_in, fsl_in, fluid_out, fsl_out, model.time_step
    )
    actual_fluid = _download(fluid_out).astype(np.float64)
    np.testing.assert_allclose(actual_fluid, expected_fluid, rtol=5.0e-6, atol=5.0e-7)
    np.testing.assert_allclose(
        fsl_out.mass.numpy(), expected_mass, rtol=3.0e-6, atol=3.0e-7
    )
    interface = flags == int(FslCellFlag.INTERFACE)
    expected_fill = expected_mass[interface] / expected_fluid[..., 0][interface]
    np.testing.assert_allclose(
        fsl_out.fill_level.numpy()[interface], expected_fill, rtol=4.0e-6, atol=4.0e-7
    )
    assert diagnostics.invalid_commit_cell_count == 0
    assert diagnostics.phase_crossing_cell_count == 0
    assert abs(diagnostics.relative_total_mass_drift) < 3.0e-7


def test_cpu_fixed_topology_step_matches_composed_numpy_oracle() -> None:
    _run_one_step_oracle("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device is required")
def test_cuda_fixed_topology_step_matches_composed_numpy_oracle() -> None:
    _run_one_step_oracle("cuda:0")


def _run_static_multistep(device: str) -> tuple[np.ndarray, np.ndarray, float]:
    shape = (32, 12, 1)
    model = HomeCoreModel(
        fluid_grid_res=shape, device=device, kinematic_viscosity=0.02
    )
    domain = HomeCoreDomain(model)
    fluid_a = domain.create_state()
    fluid_b = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid_a)
    fsl_a = FslState(model)
    fsl_b = FslState(model)
    initial_fill = _slab_fill(shape)
    fsl_a.initialize_from_fill_level(fluid_a, initial_fill)
    initial_total_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    stepper = FslFixedTopologyStepper(model)
    for _ in range(100):
        diagnostics = stepper.step(
            fluid_a, fsl_a, fluid_b, fsl_b, model.time_step
        )
        fluid_a, fluid_b = fluid_b, fluid_a
        fsl_a, fsl_b = fsl_b, fsl_a
    final_total_mass = float(
        np.sum(fsl_a.mass.numpy(), dtype=np.float64)
        + np.sum(fsl_a.excess_mass.numpy(), dtype=np.float64)
    )
    assert diagnostics.fluid.invalid_cell_count == 0
    assert diagnostics.fluid.max_speed < 2.0e-6
    assert abs((final_total_mass - initial_total_mass) / initial_total_mass) < 3.0e-6
    return _download(fluid_a), fsl_a.fill_level.numpy(), final_total_mass


def test_cpu_static_planar_interface_survives_100_transactions() -> None:
    _run_static_multistep("cpu")


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device is required")
def test_cuda_static_planar_interface_matches_cpu_after_100_transactions() -> None:
    cpu_fluid, cpu_fill, cpu_mass = _run_static_multistep("cpu")
    cuda_fluid, cuda_fill, cuda_mass = _run_static_multistep("cuda:0")
    np.testing.assert_allclose(cuda_fluid, cpu_fluid, rtol=3.0e-6, atol=3.0e-7)
    np.testing.assert_allclose(cuda_fill, cpu_fill, rtol=1.0e-5, atol=5.0e-6)
    assert cuda_mass == pytest.approx(cpu_mass, rel=3.0e-7)


def test_phase_crossing_failure_does_not_modify_caller_outputs() -> None:
    shape = (24, 8, 1)
    model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
    domain = HomeCoreDomain(model)
    fluid_in = domain.create_state()
    fluid_out = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid_in)
    fsl_in = FslState(model)
    fsl_out = FslState(model)
    fsl_in.initialize_from_fill_level(
        fluid_in, _slab_fill(shape, interface_fill=0.05)
    )
    fluid_before = fluid_out.moments.numpy().copy()
    mass_before = fsl_out.mass.numpy().copy()
    fill_before = fsl_out.fill_level.numpy().copy()
    flags_before = fsl_out.flags.numpy().copy()

    stepper = FslFixedTopologyStepper(model, fill_epsilon=0.1)
    with pytest.raises(RuntimeError, match="requires a topology transition"):
        stepper.step(fluid_in, fsl_in, fluid_out, fsl_out, model.time_step)
    np.testing.assert_array_equal(fluid_out.moments.numpy(), fluid_before)
    np.testing.assert_array_equal(fsl_out.mass.numpy(), mass_before)
    np.testing.assert_array_equal(fsl_out.fill_level.numpy(), fill_before)
    np.testing.assert_array_equal(fsl_out.flags.numpy(), flags_before)
    assert stepper.last_diagnostics is None
