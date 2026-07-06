# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Shan-Chen virtual-density method — solid / boundary wall forces."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmModel, LbmSolver, LbmState, kernels


class TestLbmShanChenWallForce(unittest.TestCase):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _center_force_x(
        self,
        solid_psi_scale: float = 1.0,
        boundary_psi: float = -1.0,
        G: float = -5.0,
    ) -> float:
        """Launch SC kernel on a 3³ grid with one solid cell at (2,1,1)."""
        nx: int = 3
        ny: int = 3
        nz: int = 3
        device: str = "cpu"
        rho_np: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)
        solid_phi_np: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)
        solid_phi_np[2, 1, 1] = -1.0  # solid neighbour to the +x of (1,1,1)

        rho: wp.array = wp.array(rho_np, dtype=float, device=device)
        solid_phi: wp.array = wp.array(solid_phi_np, dtype=float, device=device)
        fx: wp.array = wp.zeros((nx, ny, nz), dtype=float, device=device)
        fy: wp.array = wp.zeros((nx, ny, nz), dtype=float, device=device)
        fz: wp.array = wp.zeros((nx, ny, nz), dtype=float, device=device)

        wp.launch(
            kernels.compute_shan_chen_force_kernel,
            dim=(nx, ny, nz),
            inputs=[
                rho, solid_phi, fx, fy, fz,
                float(G),
                0,           # psi_type = PSI_RHO
                1.0,         # psi_ref
                float(solid_psi_scale),
                float(boundary_psi),
                0,           # homogeneous_early_out disabled for exact 3^3 test
                0.15,
                nx, ny, nz,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        force_x: np.ndarray = fx.numpy()
        return float(force_x[1, 1, 1])

    def _edge_gradient_force_x(self, homogeneous_early_out: int) -> float:
        """Return force at center when only an edge-neighbour density differs."""
        nx: int = 3
        ny: int = 3
        nz: int = 3
        device: str = "cpu"
        rho_np: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)
        rho_np[2, 2, 1] = 2.0
        solid_phi_np: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)

        rho: wp.array = wp.array(rho_np, dtype=float, device=device)
        solid_phi: wp.array = wp.array(solid_phi_np, dtype=float, device=device)
        fx: wp.array = wp.zeros((nx, ny, nz), dtype=float, device=device)
        fy: wp.array = wp.zeros((nx, ny, nz), dtype=float, device=device)
        fz: wp.array = wp.zeros((nx, ny, nz), dtype=float, device=device)

        wp.launch(
            kernels.compute_shan_chen_force_kernel,
            dim=(nx, ny, nz),
            inputs=[
                rho, solid_phi, fx, fy, fz,
                -1.0,
                0,
                1.0,
                1.0,
                -1.0,
                int(homogeneous_early_out),
                0.15,
                nx, ny, nz,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        force_x: np.ndarray = fx.numpy()
        return float(force_x[1, 1, 1])

    def _run_small_twophase_case(self, sc_force_stride: int) -> tuple[float, float, float]:
        """Run a small deterministic SC case and return mass, vmax, rho span."""
        with wp.ScopedDevice("cpu"):
            nx: int = 6
            ny: int = 6
            nz: int = 6
            model: LbmModel = LbmModel(
                fluid_grid_res=(nx, ny, nz),
                fluid_grid_cell_size=1.0,
                tau=0.8,
                G=-1.0,
                psi_type=0,
                sc_force_stride=sc_force_stride,
                sc_homogeneous_early_out=False,
                gravity_x=0.0,
                gravity_y=0.0,
                gravity_z=-0.0001,
                device="cpu",
            )
            solver: LbmSolver = LbmSolver(model)
            state_a: LbmState = LbmState(model)
            state_b: LbmState = LbmState(model)
            state_a.clear()
            state_b.clear()

            stride: int = nx * ny * nz
            f_np: np.ndarray = np.zeros(19 * stride, dtype=np.float32)
            for i in range(nx):
                rho_val: float = 1.1 if i < nx // 2 else 0.9
                for j in range(ny):
                    for k in range(nz):
                        idx: int = i * ny * nz + j * nz + k
                        f_np[0 * stride + idx] = (1.0 / 3.0) * rho_val
                        for direction in range(1, 7):
                            f_np[direction * stride + idx] = (1.0 / 18.0) * rho_val
                        for direction in range(7, 19):
                            f_np[direction * stride + idx] = (1.0 / 36.0) * rho_val
            wp.copy(state_a.f, wp.array(f_np, dtype=float, device="cpu"))

            state_in: LbmState = state_a
            state_out: LbmState = state_b
            for _step_index in range(4):
                solver.step(state_in, state_out, dt=1.0)
                state_in, state_out = state_out, state_in
            wp.synchronize_device("cpu")

            rho_np: np.ndarray = np.asarray(state_in.density.numpy(), dtype=np.float64)
            ux_np: np.ndarray = np.asarray(state_in.velocity_x.numpy(), dtype=np.float64)
            uy_np: np.ndarray = np.asarray(state_in.velocity_y.numpy(), dtype=np.float64)
            uz_np: np.ndarray = np.asarray(state_in.velocity_z.numpy(), dtype=np.float64)
            vel_mag: np.ndarray = np.sqrt(ux_np * ux_np + uy_np * uy_np + uz_np * uz_np)
            return float(rho_np.sum()), float(vel_mag.max()), float(rho_np.max() - rho_np.min())

    # ------------------------------------------------------------------
    # solid-body virtual-ψ tests
    # ------------------------------------------------------------------

    def test_solid_scale_below_one_pushes_away_from_solid(self) -> None:
        """ψ_solid = 0.5 < ψ_fluid = 1.0 → force points toward fluid (-x).

        With G = -5, -G = +5 > 0, the force direction follows higher ψ.
        The solid at (2,1,1) has lower ψ → net force points toward the
        fluid side (-x direction) → force_x < 0.
        """
        self.assertLess(self._center_force_x(solid_psi_scale=0.5), 0.0)

    def test_solid_scale_above_one_pulls_toward_solid(self) -> None:
        """ψ_solid = 2.0 > ψ_fluid = 1.0 → force points toward solid (+x)."""
        self.assertGreater(self._center_force_x(solid_psi_scale=2.0), 0.0)

    def test_solid_scale_one_gives_zero_force(self) -> None:
        """ψ_solid = ψ_c → mirror closure → zero net force for uniform ρ."""
        fx: float = self._center_force_x(solid_psi_scale=1.0)
        self.assertAlmostEqual(fx, 0.0, delta=1e-5)

    def test_homogeneous_early_out_can_be_disabled(self) -> None:
        """Edge-only gradients should be recoverable when early-out is off."""
        early_force: float = self._edge_gradient_force_x(homogeneous_early_out=1)
        full_force: float = self._edge_gradient_force_x(homogeneous_early_out=0)
        self.assertAlmostEqual(early_force, 0.0, delta=1e-6)
        self.assertGreater(abs(full_force), 1e-4)

    # ------------------------------------------------------------------
    # solver integration tests
    # ------------------------------------------------------------------

    def test_solver_passes_boundary_psi_to_kernel(self) -> None:
        """Solver step with G≠0 should produce density field (smoke test)."""
        with wp.ScopedDevice("cpu"):
            nx: int = 8
            ny: int = 8
            nz: int = 8
            model: LbmModel = LbmModel(
                fluid_grid_res=(nx, ny, nz),
                fluid_grid_cell_size=1.0,
                tau=0.8,
                G=-5.0,
                sc_boundary_psi=0.2,
                gravity_x=0.0, gravity_y=0.0, gravity_z=-0.001,
                device="cpu",
            )
            solver: LbmSolver = LbmSolver(model)
            state_in: LbmState = LbmState(model)
            state_out: LbmState = LbmState(model)
            state_in.clear()
            state_out.clear()
            solver.initialize_equilibrium(state_in, rho0=1.0, u0=(0.0, 0.0, 0.0))

            solver.step(state_in, state_out, dt=1.0)
            wp.synchronize_device("cpu")

            # Should not explode
            rho_np: np.ndarray = np.asarray(state_out.density.numpy(), dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(rho_np)))
            self.assertTrue(np.all(rho_np > 0.0))

    def test_model_rejects_invalid_shan_chen_parameters(self) -> None:
        """Invalid SC controls should fail fast at model construction."""
        with self.assertRaises(ValueError):
            LbmModel(fluid_grid_res=(4, 4, 4), psi_type=2)
        with self.assertRaises(ValueError):
            LbmModel(fluid_grid_res=(4, 4, 4), psi_ref=0.0)
        with self.assertRaises(ValueError):
            LbmModel(fluid_grid_res=(4, 4, 4), sc_force_stride=0)
        with self.assertRaises(ValueError):
            LbmModel(fluid_grid_res=(4, 4, 4), sc_homogeneous_rel_tol=-0.1)

    def test_sc_force_stride_remains_finite_on_small_case(self) -> None:
        """Stride 1 and 2 should remain finite with bounded mass drift."""
        mass_1: float
        vmax_1: float
        span_1: float
        mass_2: float
        vmax_2: float
        span_2: float
        mass_1, vmax_1, span_1 = self._run_small_twophase_case(sc_force_stride=1)
        mass_2, vmax_2, span_2 = self._run_small_twophase_case(sc_force_stride=2)

        for value in (mass_1, vmax_1, span_1, mass_2, vmax_2, span_2):
            self.assertTrue(np.isfinite(value))
        self.assertLess(vmax_1, 0.5)
        self.assertLess(vmax_2, 0.5)
        self.assertLess(abs(mass_2 - mass_1) / max(abs(mass_1), 1.0), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
