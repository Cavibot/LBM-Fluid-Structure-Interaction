"""Part 1 acceptance tests for zero-force collision-space execution."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import HomeLbmState, LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, W
from wanphys._src.fluid.fluid_grid.lbm.moments import (
    MOMENT_EXPONENTS,
    central_to_raw_moments_kernel,
    guo_source_to_raw_moments_kernel,
    host_moment_matrices,
    host_relaxation_rates,
    nocm_collision_kernel,
    populations_to_raw_moments_kernel,
    raw_moments_to_populations_kernel,
    raw_mrt_collision_kernel,
)


def _equilibrium(rho: float, velocity: np.ndarray) -> np.ndarray:
    c = np.asarray(tuple(zip(CX, CY, CZ)), dtype=np.float64)
    weights = np.asarray(W, dtype=np.float64)
    cu = c @ velocity
    return weights * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * np.dot(velocity, velocity))


def _guo_source(velocity: np.ndarray, force: np.ndarray) -> np.ndarray:
    c = np.asarray(tuple(zip(CX, CY, CZ)), dtype=np.float64)
    weights = np.asarray(W, dtype=np.float64)
    cu = c @ velocity
    return weights * (
        3.0 * ((c - velocity) @ force)
        + 9.0 * cu * (c @ force)
    )


def _shift_matrix(velocity: np.ndarray) -> np.ndarray:
    from math import comb

    matrix = np.zeros((19, 19), dtype=np.float64)
    for target, (tx, ty, tz) in enumerate(MOMENT_EXPONENTS):
        for source, (sx, sy, sz) in enumerate(MOMENT_EXPONENTS):
            if sx <= tx and sy <= ty and sz <= tz:
                matrix[target, source] = (
                    comb(tx, sx) * comb(ty, sy) * comb(tz, sz)
                    * velocity[0] ** (tx - sx)
                    * velocity[1] ** (ty - sy)
                    * velocity[2] ** (tz - sz)
                )
    return matrix


class TestPart1CollisionSpaces(unittest.TestCase):
    def test_raw_moment_basis_is_invertible(self) -> None:
        transform, inverse = host_moment_matrices()
        np.testing.assert_allclose(
            inverse.astype(np.float64) @ transform.astype(np.float64),
            np.eye(19),
            atol=2.0e-7,
            rtol=2.0e-7,
        )

    def test_mrt_profile_preserves_conserved_modes_and_has_distinct_orders(self) -> None:
        rates = host_relaxation_rates(1.25, 1.1, 0.9)
        np.testing.assert_array_equal(rates[:4], np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(rates[4:10], 1.25)
        np.testing.assert_allclose(rates[10:16], 1.1)
        np.testing.assert_allclose(rates[16:19], 0.9)

    def test_raw_and_nocm_kernels_match_independent_numpy_references(self) -> None:
        transform, inverse = host_moment_matrices()
        transform = transform.astype(np.float64)
        inverse = inverse.astype(np.float64)
        rates = host_relaxation_rates(1.25, 1.1, 0.9).astype(np.float64)
        populations = np.asarray(W, dtype=np.float64) * 1.07
        populations += np.linspace(-2.0e-4, 2.0e-4, 19)
        c = np.asarray(tuple(zip(CX, CY, CZ)), dtype=np.float64)
        rho = float(np.sum(populations))
        momentum = c.T @ populations
        force = np.array([2.0e-5, -1.0e-5, 0.5e-5], dtype=np.float64)
        velocity = (momentum + 0.5 * force) / rho
        equilibrium = _equilibrium(rho, velocity)
        population_source = _guo_source(velocity, force)
        raw = transform @ populations
        raw_equilibrium = transform @ equilibrium
        raw_source = transform @ population_source

        raw_post = raw - rates * (raw - raw_equilibrium) + (1.0 - 0.5 * rates) * raw_source
        raw_post[0] = raw[0]
        raw_post[1:4] = raw[1:4] + force
        raw_reference = inverse @ raw_post

        minus_shift = _shift_matrix(-velocity)
        plus_shift = _shift_matrix(velocity)
        central = minus_shift @ raw
        central_source = minus_shift @ raw_source
        central_equilibrium = np.zeros(19, dtype=np.float64)
        central_equilibrium[0] = rho
        central_equilibrium[4:7] = rho / 3.0
        central_equilibrium[16:19] = rho / 9.0
        central_post = (
            central
            - rates * (central - central_equilibrium)
            + (1.0 - 0.5 * rates) * central_source
        )
        central_post[0] = central[0]
        central_post[1:4] = central[1:4] + force
        nocm_reference = inverse @ (plus_shift @ central_post)

        f = wp.array(populations.astype(np.float32), dtype=float, device="cpu")
        raw_device = wp.zeros(19, dtype=float, device="cpu")
        source_device = wp.zeros(19, dtype=float, device="cpu")
        post_raw_device = wp.zeros(19, dtype=float, device="cpu")
        post_central_device = wp.zeros(19, dtype=float, device="cpu")
        inverse_device = wp.array(inverse.astype(np.float32).reshape(-1), dtype=float, device="cpu")
        rates_device = wp.array(rates.astype(np.float32), dtype=float, device="cpu")

        def scalar_field(value: float) -> wp.array3d:
            return wp.array(np.array([[[value]]], dtype=np.float32), dtype=float, device="cpu")

        rho_device = scalar_field(rho)
        ux, uy, uz = (scalar_field(float(value)) for value in velocity)
        fx, fy, fz = (scalar_field(float(value)) for value in force)
        wp.launch(populations_to_raw_moments_kernel, dim=(1, 1, 1), inputs=[f, raw_device, 1, 1, 1], device="cpu")
        wp.launch(
            guo_source_to_raw_moments_kernel,
            dim=(1, 1, 1),
            inputs=[ux, uy, uz, fx, fy, fz, source_device, 1, 1, 1],
            device="cpu",
        )
        wp.launch(
            raw_mrt_collision_kernel,
            dim=(1, 1, 1),
            inputs=[raw_device, source_device, post_raw_device, rho_device, ux, uy, uz, fx, fy, fz, rates_device, 1, 1, 1],
            device="cpu",
        )
        raw_actual = wp.zeros(19, dtype=float, device="cpu")
        wp.launch(raw_moments_to_populations_kernel, dim=(1, 1, 1), inputs=[post_raw_device, inverse_device, raw_actual, 1, 1, 1], device="cpu")
        np.testing.assert_allclose(raw_actual.numpy(), raw_reference, atol=2.0e-6, rtol=2.0e-6)

        wp.launch(
            nocm_collision_kernel,
            dim=(1, 1, 1),
            inputs=[raw_device, source_device, post_central_device, rho_device, ux, uy, uz, fx, fy, fz, rates_device, 1, 1, 1],
            device="cpu",
        )
        wp.launch(
            central_to_raw_moments_kernel,
            dim=(1, 1, 1),
            inputs=[post_central_device, post_raw_device, ux, uy, uz, 1, 1, 1],
            device="cpu",
        )
        nocm_actual = wp.zeros(19, dtype=float, device="cpu")
        wp.launch(raw_moments_to_populations_kernel, dim=(1, 1, 1), inputs=[post_raw_device, inverse_device, nocm_actual, 1, 1, 1], device="cpu")
        np.testing.assert_allclose(nocm_actual.numpy(), nocm_reference, atol=3.0e-6, rtol=3.0e-6)

    def test_all_zero_force_execution_paths_keep_rest_equilibrium(self) -> None:
        paths = (
            ("fullf", "srt"),
            ("fullf", "trt"),
            ("fullf", "raw_mrt"),
            ("fullf", "nocm_mrt"),
            ("home", "srt"),
            ("home", "trt"),
            ("home", "nocm_mrt"),
        )
        for encoding, collision in paths:
            with self.subTest(encoding=encoding, collision=collision):
                model = LbmModel(
                    fluid_grid_res=(3, 3, 3),
                    device="cpu",
                    encoding=encoding,
                    collision=collision,
                    tau=0.8,
                    bc_periodic=(True, True, True),
                )
                domain = LbmDomain(model)
                state = domain.create_state()
                domain.solver.initialize_equilibrium(state, rho0=1.0)
                domain.step(1.0)
                np.testing.assert_allclose(
                    np.sum(domain.state.density.numpy(), dtype=np.float64),
                    27.0,
                    # Six population-reconstruction paths share a deterministic
                    # float32 accumulation floor of 9.66e-6 over 27 cells.
                    # Keep the bound below four float32 epsilons per cell.
                    rtol=4.0e-7,
                    atol=1.0e-6,
                )
                self.assertTrue(np.all(np.isfinite(domain.state.velocity_x.numpy())))

    def test_home_nocm_alias_is_accepted_but_canonicalized(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                encoding="home",
                collision="home_nocm",
            )
        self.assertEqual(model.resolved_collision, "nocm_mrt")
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        self.assertIsInstance(LbmDomain(model).create_state(), HomeLbmState)


if __name__ == "__main__":
    unittest.main()
