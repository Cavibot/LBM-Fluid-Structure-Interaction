# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-Free gas-pressure boundary agreement with paper Eq. (11)."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_OPPOSITE,
    D3Q27_WEIGHTS,
    HomeFreeCellFlag,
    HomeLbmModel,
    HomeLbmSolver,
    HomeLbmState,
    collide_central_moments,
    extract_moments,
    gas_pressure_boundary_populations,
    only_missing_boundary_residual,
    reconstruct_populations,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof import kernels as vof_kernels


def _equilibrium_moments(rho: np.ndarray, velocity: np.ndarray) -> np.ndarray:
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


def _upload(state: HomeLbmState, moments: np.ndarray) -> None:
    state.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    )


def _download(state: HomeLbmState) -> np.ndarray:
    return (
        state.moments.numpy()
        .reshape(10, -1)
        .T.reshape(state.res + (10,))
        .astype(np.float64)
    )


def _reference_step(
    moments: np.ndarray,
    flags: np.ndarray,
    gas_density: float | np.ndarray,
    shear_omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = flags.shape
    filtered = reconstruct_populations(moments)
    pressure = gas_pressure_boundary_populations(moments, gas_density)
    streamed = np.zeros_like(filtered)
    result = moments.copy()
    gas_boundary_impulse = np.zeros(3, dtype=np.float64)
    for index in np.ndindex(shape):
        if int(flags[index]) == HomeFreeCellFlag.GAS:
            continue
        for q, c in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
            source = tuple((index[axis] - int(c[axis])) % shape[axis] for axis in range(3))
            if int(flags[source]) == HomeFreeCellFlag.GAS:
                if int(flags[index]) != HomeFreeCellFlag.INTERFACE:
                    raise ValueError("reference topology contains a direct liquid-gas link")
                streamed[index + (q,)] = pressure[index + (q,)]
                opposite = int(np.flatnonzero(np.all(D3Q27_DIRECTIONS == -c, axis=1))[0])
                gas_boundary_impulse += (
                    pressure[index + (q,)] + filtered[index + (opposite,)]
                ) * c
            else:
                streamed[index + (q,)] = filtered[source + (q,)]
        result[index] = collide_central_moments(
            extract_moments(streamed[index]), shear_omega
        )
    return result, gas_boundary_impulse


class TestHomeFreePressureBoundary(unittest.TestCase):
    @staticmethod
    def _slab_flags(shape: tuple[int, int, int]) -> np.ndarray:
        flags = np.full(shape, int(HomeFreeCellFlag.GAS), dtype=np.int32)
        flags[0] = int(HomeFreeCellFlag.INTERFACE)
        flags[1:3] = int(HomeFreeCellFlag.LIQUID)
        flags[3] = int(HomeFreeCellFlag.INTERFACE)
        return flags

    def _run_reference_case(self, device: str, *, use_density_field: bool) -> None:
        shape = (6, 3, 2)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            kinematic_viscosity=0.19,
            periodic=(True, True, True),
            device=device,
        )
        rng = np.random.default_rng(104729)
        rho = rng.uniform(0.97, 1.03, size=shape)
        velocity = rng.uniform(-0.02, 0.02, size=shape + (3,))
        moments = _equilibrium_moments(rho, velocity)
        moments[..., 4:10] += rng.uniform(-8.0e-4, 8.0e-4, size=shape + (6,))
        flags = np.empty(shape, dtype=np.int32)
        flags[0] = int(HomeFreeCellFlag.INTERFACE)
        flags[1:3] = int(HomeFreeCellFlag.LIQUID)
        flags[3] = int(HomeFreeCellFlag.INTERFACE)
        flags[4:6] = int(HomeFreeCellFlag.GAS)
        gas_density = 0.985
        density_field = np.full(shape, gas_density, dtype=np.float32)
        density_field[0] += np.linspace(-0.006, 0.004, shape[1])[:, None]
        density_field[3] += np.linspace(0.005, -0.003, shape[1])[:, None]
        prescribed_density = density_field if use_density_field else gas_density
        expected, expected_gas_impulse = _reference_step(
            moments, flags, prescribed_density, model.shear_omega
        )

        state_in = HomeLbmState(model)
        state_out = HomeLbmState(model)
        _upload(state_in, moments)
        device_flags = wp.array(flags, dtype=wp.int32, device=device)
        device_density = wp.array(density_field, dtype=float, device=device)
        solver = HomeLbmSolver(model)
        solver.step(
            state_in,
            state_out,
            1.0,
            free_surface_flags=device_flags,
            gas_density=gas_density,
            gas_density_field=device_density if use_density_field else None,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(_download(state_out), expected, rtol=4.0e-5, atol=4.0e-6)
        diagnostics = solver.validate_state(
            state_out,
            free_surface_flags=device_flags,
            gas_density_field=device_density if use_density_field else None,
        )
        self.assertEqual(diagnostics.invalid_cell_count, 0)
        self.assertEqual(diagnostics.direct_liquid_gas_link_count, 0)
        self.assertEqual(diagnostics.invalid_gas_density_count, 0)
        np.testing.assert_allclose(
            diagnostics.gas_boundary_impulse_lattice,
            expected_gas_impulse,
            rtol=2.0e-6,
            atol=2.0e-6,
        )

    def _run_only_missing_warp_oracle(self, device: str) -> None:
        shape = (7, 4, 3)
        stride = int(np.prod(shape))
        flags = self._slab_flags(shape)
        rng = np.random.default_rng(2971215073)
        rho = rng.uniform(0.98, 1.02, size=shape)
        velocity = rng.uniform(-0.015, 0.015, size=shape + (3,))
        moments = _equilibrium_moments(rho, velocity)
        moments[..., 4:10] += rng.uniform(-6.0e-4, 6.0e-4, size=shape + (6,))
        gas_density = rng.uniform(0.985, 1.015, size=shape).astype(np.float32)
        expected = only_missing_boundary_residual(
            moments,
            flags,
            gas_density,
            periodic=(True, True, True),
        )

        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            device=device,
        )
        state = HomeLbmState(model)
        _upload(state, moments)
        device_flags = wp.array(flags, dtype=wp.int32, device=device)
        device_gas_density = wp.array(gas_density, dtype=float, device=device)
        directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device
        )
        weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=device
        )
        opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=device)
        outputs = [
            wp.zeros(4 * stride, dtype=float, device=device) for _ in range(4)
        ]
        gas_link_count = wp.zeros(shape, dtype=wp.int32, device=device)
        invalid = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch(
            vof_kernels.only_missing_boundary_residual_kernel,
            dim=shape,
            inputs=[
                state.moments,
                device_flags,
                device_gas_density,
                directions,
                weights,
                opposites,
                *outputs,
                gas_link_count,
                invalid,
                1,
                1,
                1,
                *shape,
                stride,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(invalid.numpy()[0]), 0)
        actual = [
            np.moveaxis(output.numpy().reshape(4, *shape), 0, -1)
            for output in outputs
        ]
        for observed, reference in zip(
            actual,
            (
                expected.pressure,
                expected.equilibrium_transport,
                expected.non_equilibrium,
                expected.total,
            ),
            strict=True,
        ):
            np.testing.assert_allclose(
                observed, reference, rtol=5.0e-5, atol=2.0e-7
            )
        np.testing.assert_array_equal(gas_link_count.numpy(), expected.gas_link_count)

        invalid_flags = flags.copy()
        invalid_flags[2] = int(HomeFreeCellFlag.SOLID)
        device_flags.assign(invalid_flags)
        invalid.zero_()
        wp.launch(
            vof_kernels.only_missing_boundary_residual_kernel,
            dim=shape,
            inputs=[
                state.moments,
                device_flags,
                device_gas_density,
                directions,
                weights,
                opposites,
                *outputs,
                gas_link_count,
                invalid,
                1,
                1,
                1,
                *shape,
                stride,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        self.assertGreater(int(invalid.numpy()[0]), 0)

    def test_cpu_matches_cellwise_eq11_reference(self) -> None:
        self._run_reference_case("cpu", use_density_field=False)

    def test_cpu_only_missing_warp_oracle_matches_numpy(self) -> None:
        self._run_only_missing_warp_oracle("cpu")

    def test_cpu_local_gas_density_matches_cellwise_eq12_reference(self) -> None:
        self._run_reference_case("cpu", use_density_field=True)

    def test_only_missing_uniform_equilibrium_is_an_exact_fixed_point(self) -> None:
        shape = (7, 4, 3)
        flags = self._slab_flags(shape)
        rho = np.full(shape, 1.017, dtype=np.float64)
        velocity = np.empty(shape + (3,), dtype=np.float64)
        velocity[...] = (0.013, -0.007, 0.004)
        moments = _equilibrium_moments(rho, velocity)

        residual = only_missing_boundary_residual(
            moments,
            flags,
            1.017,
            periodic=(True, True, True),
        )

        np.testing.assert_allclose(residual.total, 0.0, atol=2.0e-16)
        np.testing.assert_allclose(residual.pressure, 0.0, atol=2.0e-16)
        np.testing.assert_allclose(
            residual.equilibrium_transport, 0.0, atol=2.0e-16
        )
        np.testing.assert_allclose(residual.non_equilibrium, 0.0, atol=2.0e-16)
        self.assertTrue(np.all(residual.gas_link_count[flags == HomeFreeCellFlag.INTERFACE] > 0))
        self.assertTrue(np.all(residual.gas_link_count[flags == HomeFreeCellFlag.LIQUID] == 0))

    def test_only_missing_density_jump_is_isolated_as_pressure(self) -> None:
        shape = (7, 3, 2)
        flags = self._slab_flags(shape)
        moments = _equilibrium_moments(
            np.full(shape, 1.02, dtype=np.float64),
            np.zeros(shape + (3,), dtype=np.float64),
        )

        residual = only_missing_boundary_residual(
            moments,
            flags,
            0.99,
            periodic=(True, True, True),
        )

        np.testing.assert_allclose(residual.total, residual.pressure, atol=2.0e-16)
        np.testing.assert_allclose(
            residual.equilibrium_transport, 0.0, atol=2.0e-16
        )
        np.testing.assert_allclose(residual.non_equilibrium, 0.0, atol=2.0e-16)
        self.assertGreater(float(np.max(np.linalg.norm(residual.total[..., 1:4], axis=-1))), 0.0)
        np.testing.assert_allclose(
            residual.total[flags == HomeFreeCellFlag.LIQUID], 0.0, atol=2.0e-16
        )

    def test_only_missing_stress_residual_is_isolated_from_pressure(self) -> None:
        shape = (7, 3, 2)
        flags = self._slab_flags(shape)
        moments = _equilibrium_moments(
            np.ones(shape, dtype=np.float64),
            np.zeros(shape + (3,), dtype=np.float64),
        )
        moments[0, ..., 4] += 0.012
        moments[0, ..., 7] -= 0.007

        residual = only_missing_boundary_residual(
            moments,
            flags,
            1.0,
            periodic=(True, True, True),
        )

        np.testing.assert_allclose(residual.pressure, 0.0, atol=2.0e-16)
        np.testing.assert_allclose(
            residual.equilibrium_transport, 0.0, atol=2.0e-16
        )
        np.testing.assert_allclose(
            residual.total, residual.non_equilibrium, atol=2.0e-16
        )
        self.assertGreater(float(np.max(np.abs(residual.non_equilibrium))), 0.0)

    def test_only_missing_decomposition_matches_reference_stream(self) -> None:
        shape = (7, 3, 2)
        flags = self._slab_flags(shape)
        rng = np.random.default_rng(433494437)
        rho = rng.uniform(0.98, 1.02, size=shape)
        velocity = rng.uniform(-0.012, 0.012, size=shape + (3,))
        moments = _equilibrium_moments(rho, velocity)
        moments[..., 4:10] += rng.uniform(-5.0e-4, 5.0e-4, size=shape + (6,))
        gas_density = rng.uniform(0.985, 1.015, size=shape)

        residual = only_missing_boundary_residual(
            moments,
            flags,
            gas_density,
            periodic=(True, True, True),
        )
        streamed, _ = _reference_step(moments, flags, gas_density, shear_omega=1.31)
        active = np.isin(flags, (HomeFreeCellFlag.INTERFACE, HomeFreeCellFlag.LIQUID))

        np.testing.assert_allclose(
            residual.total,
            residual.pressure
            + residual.equilibrium_transport
            + residual.non_equilibrium,
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            residual.total[active],
            streamed[..., :4][active] - moments[..., :4][active],
            rtol=2.0e-13,
            atol=2.0e-15,
        )

    def test_only_missing_oracle_rejects_other_boundary_ownership(self) -> None:
        shape = (4, 2, 2)
        flags = np.full(shape, int(HomeFreeCellFlag.LIQUID), dtype=np.int32)
        moments = _equilibrium_moments(
            np.ones(shape, dtype=np.float64),
            np.zeros(shape + (3,), dtype=np.float64),
        )
        with self.assertRaisesRegex(ValueError, "non-periodic domain link"):
            only_missing_boundary_residual(moments, flags, 1.0)

        flags[1] = int(HomeFreeCellFlag.SOLID)
        with self.assertRaisesRegex(ValueError, "solid link"):
            only_missing_boundary_residual(
                moments, flags, 1.0, periodic=(True, True, True)
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_matches_cellwise_eq11_reference(self) -> None:
        self._run_reference_case("cuda:0", use_density_field=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_only_missing_warp_oracle_matches_numpy(self) -> None:
        self._run_only_missing_warp_oracle("cuda:0")

    def test_all_liquid_free_surface_path_matches_single_phase_exactly(self) -> None:
        shape = (4, 3, 2)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        rng = np.random.default_rng(130363)
        rho = rng.uniform(0.98, 1.02, size=shape)
        velocity = rng.uniform(-0.015, 0.015, size=shape + (3,))
        moments = _equilibrium_moments(rho, velocity)
        plain_in, plain_out = HomeLbmState(model), HomeLbmState(model)
        free_in, free_out = HomeLbmState(model), HomeLbmState(model)
        _upload(plain_in, moments)
        _upload(free_in, moments)

        HomeLbmSolver(model).step(plain_in, plain_out, model.time_step)
        flags = wp.full(shape, int(HomeFreeCellFlag.LIQUID), dtype=wp.int32, device="cpu")
        HomeLbmSolver(model).step(
            free_in,
            free_out,
            model.time_step,
            free_surface_flags=flags,
        )
        wp.synchronize_device("cpu")

        np.testing.assert_array_equal(free_out.moments.numpy(), plain_out.moments.numpy())

    def test_direct_liquid_gas_link_is_reported_as_hard_error(self) -> None:
        shape = (3, 2, 2)
        model = HomeLbmModel(fluid_grid_res=shape, periodic=(False, False, False), device="cpu")
        state_in, state_out = HomeLbmState(model), HomeLbmState(model)
        moments = _equilibrium_moments(np.ones(shape), np.zeros(shape + (3,)))
        _upload(state_in, moments)
        flags = np.full(shape, int(HomeFreeCellFlag.GAS), dtype=np.int32)
        flags[0] = int(HomeFreeCellFlag.LIQUID)
        device_flags = wp.array(flags, dtype=wp.int32, device="cpu")
        solver = HomeLbmSolver(model)
        solver.step(state_in, state_out, model.time_step, free_surface_flags=device_flags)

        with self.assertRaisesRegex(RuntimeError, "direct liquid-gas"):
            solver.validate_state(state_out, free_surface_flags=device_flags)

    def test_reference_rejects_nonpositive_gas_density(self) -> None:
        moments = _equilibrium_moments(np.ones((1,)), np.zeros((1, 3)))
        with self.assertRaisesRegex(ValueError, "gas_density"):
            gas_pressure_boundary_populations(moments, 0.0)

    def test_nonpositive_local_gas_density_is_a_hard_error(self) -> None:
        shape = (4, 2, 2)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        state_in, state_out = HomeLbmState(model), HomeLbmState(model)
        _upload(state_in, _equilibrium_moments(np.ones(shape), np.zeros(shape + (3,))))
        flags = np.empty(shape, dtype=np.int32)
        flags[0] = int(HomeFreeCellFlag.INTERFACE)
        flags[1] = int(HomeFreeCellFlag.LIQUID)
        flags[2] = int(HomeFreeCellFlag.INTERFACE)
        flags[3] = int(HomeFreeCellFlag.GAS)
        density = np.ones(shape, dtype=np.float32)
        density[0] = -1.0
        device_flags = wp.array(flags, dtype=wp.int32, device="cpu")
        device_density = wp.array(density, dtype=float, device="cpu")
        solver = HomeLbmSolver(model)
        solver.step(
            state_in,
            state_out,
            model.time_step,
            free_surface_flags=device_flags,
            gas_density_field=device_density,
        )

        with self.assertRaisesRegex(FloatingPointError, "gas density"):
            solver.validate_state(
                state_out,
                free_surface_flags=device_flags,
                gas_density_field=device_density,
            )


if __name__ == "__main__":
    unittest.main()
