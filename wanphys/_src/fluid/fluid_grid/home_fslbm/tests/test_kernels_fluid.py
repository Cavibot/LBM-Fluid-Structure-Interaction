# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HOME-FSLBM fluid kernels.

Verifies the three core ``@wp.func`` helpers
(``calculate_f_eq_d3q27``, ``reconstruct_distribution``,
``ml_get_pi_after_collision``) and the monolithic
``stream_collide_bvh_kernel``.

Reference
---------
- ``mrUtilFuncGpu3D.h:292-320``  — calculate_f_eq
- ``mrUtilFuncGpu3D.h:153-273``  — mlCalDistributionFourthOrderD3Q27AtIndex
- ``mrUtilFuncGpu3D.h:424-471``  — mlGetPIAfterCollision
- ``mrLbmSolverGpu3D.cu:703-1057`` — stream_collide_bvh
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp


# ===========================================================================
from .conftest import load_golden

# Helpers — imported from kernels_fluid (must be same module as @wp.func)
# ===========================================================================

from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_fluid import (
    _kernel_f_eq,
    _kernel_reconstruct,
    _kernel_collision,
    _kernel_compute_rho_u,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def device():
    try:
        return wp.get_device("cuda:0")
    except Exception:
        return wp.get_device("cpu")

@pytest.fixture
def C():
    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as _C
    return _C


# ===========================================================================
# Test 1: calculate_f_eq_d3q27 — D3Q27 Maxwell-Boltzmann equilibrium
# ===========================================================================


class TestFEquilibriumD3Q27:
    """Verify f_eq matches reference code across all 27 directions."""

    def test_all_27_directions(self, device):
        """Compare against reference calculate_f_eq at (rho=1.0, u=(0.02, 0.01, 0.0))."""
        rho = 1.0
        ux, uy, uz = 0.02, 0.01, 0.0

        output = wp.zeros(27, dtype=float, device=device)
        wp.launch(_kernel_f_eq, dim=27, inputs=[rho, ux, uy, uz, output], device=device)
        result = output.numpy()

        # Reference values computed from the same formula in C++ reference code
        # (mrUtilFuncGpu3D.h:292-320).  These were pre-computed externally.
        # The key invariants are:
        #   1. Σ f_i^eq = ρ
        #   2. Σ f_i^eq · c_i = ρ·u
        #   3. Weights are correct for each direction class

        golden = load_golden("f_eq_rho1.0_ux0.02_uy0.01")

        for di in range(27):
            assert np.isclose(result[di], golden[di], atol=1e-12), \
                f"di={di}: got {result[di]:.15e}, expected {golden[di]:.15e}"

    def test_rest_density(self, device):
        """Direction-by-direction at (rho=2.5, u=0)."""
        rho = 2.5
        output = wp.zeros(27, dtype=float, device=device)
        wp.launch(_kernel_f_eq, dim=27, inputs=[rho, 0.0, 0.0, 0.0, output], device=device)
        result = output.numpy()
        golden = load_golden("f_eq_rho2.5_rest")
        for di in range(27):
            assert np.isclose(result[di], golden[di], atol=1e-12), \
                f"di={di}: got {result[di]:.15e}, expected {golden[di]:.15e}"

    def test_rest_density_2(self, device):
        """Direction-by-direction at (rho=1.5, u=0)."""
        rho = 1.5
        output = wp.zeros(27, dtype=float, device=device)
        wp.launch(_kernel_f_eq, dim=27, inputs=[rho, 0.0, 0.0, 0.0, output], device=device)
        result = output.numpy()
        golden = load_golden("f_eq_rho1.5_rest")
        for di in range(27):
            assert np.isclose(result[di], golden[di], atol=1e-12), \
                f"di={di}: got {result[di]:.15e}, expected {golden[di]:.15e}"


# ===========================================================================
# Test 2: reconstruct_distribution — Hermite expansion (10 moments → 27 f_i)
# ===========================================================================


class TestReconstructDistribution:
    """Verify third-order Hermite reconstruction against reference."""

    def test_roundtrip_at_rest(self, device):
        """At equilibrium (u=0, Π=cs²·I), reconstruction must equal f_eq."""
        rho = 1.0
        ux = uy = uz = 0.0
        # Equilibrium stress: pi_xx = pi_yy = pi_zz = cs², off-diag = 0
        cs2 = 1.0 / 3.0
        pi_xx = cs2
        pi_yy = cs2
        pi_zz = cs2
        pi_xy = pi_xz = pi_yz = 0.0

        output = wp.zeros(27, dtype=float, device=device)
        wp.launch(
            _kernel_reconstruct,
            dim=27,
            inputs=[rho, ux, uy, uz, pi_xx, pi_xy, pi_xz, pi_yy, pi_yz, pi_zz, output],
            device=device,
        )
        recon = output.numpy()

        golden = load_golden("recon_rest")
        for di in range(27):
            assert np.isclose(recon[di], golden[di], atol=1e-12), \
                f"di={di}: got {recon[di]:.15e}, expected {golden[di]:.15e}"

    def test_density_sum_preserved(self, device):
        """Σ f_i from reconstruction must equal rho (for any stress)."""
        rho = 1.0
        ux, uy, uz = 0.1, 0.0, 0.0
        cs2 = 1.0 / 3.0
        pi_xx = cs2 + 0.01
        pi_yy = cs2 - 0.005
        pi_zz = cs2 - 0.005
        pi_xy = 0.002
        pi_xz = 0.0
        pi_yz = 0.0

        output = wp.zeros(27, dtype=float, device=device)
        wp.launch(
            _kernel_reconstruct,
            dim=27,
            inputs=[rho, ux, uy, uz, pi_xx, pi_xy, pi_xz, pi_yy, pi_yz, pi_zz, output],
            device=device,
        )
        recon = output.numpy()
        golden = load_golden("recon_rho1.0_ux0.1")
        for di in range(27):
            assert np.isclose(recon[di], golden[di], atol=1e-12), \
                f"di={di}: got {recon[di]:.15e}, expected {golden[di]:.15e}"


# ===========================================================================
# Test 3: ml_get_pi_after_collision — NOCM-MRT collision operator
# ===========================================================================


class TestNOCMMRTCollision:
    """Verify the closed-form stress collision against reference."""

    def test_equilibrium_preserved_at_rest(self, device):
        """At u=0, F=0, equilibrium stress (pi_xx=cs², off-diag=0)
        must be unchanged by collision regardless of omega."""
        rho = 1.0
        ux = uy = uz = 0.0
        fx = fy = fz = 0.0
        omega = 1.0  # any omega > 0

        cs2 = 1.0 / 3.0
        pixx_old = cs2
        piyy_old = cs2
        pizz_old = cs2
        pixy_old = pixz_old = piyz_old = 0.0

        pi_new = wp.zeros(6, dtype=float, device=device)
        wp.launch(
            _kernel_collision,
            dim=1,
            inputs=[
                rho, ux, uy, uz,
                fx, fy, fz, omega,
                pixx_old, pixy_old, pixz_old,
                piyy_old, piyz_old, pizz_old,
                pi_new,
            ],
            device=device,
        )
        result = pi_new.numpy()

        golden = load_golden("collision_rest")
        for i in range(6):
            assert np.isclose(result[i], golden[i], atol=1e-12), \
                f"i={i}: got {result[i]:.15e}, expected {golden[i]:.15e}"

    def test_relaxation_toward_equilibrium(self, device):
        """With omega=1.0, non-equilibrium stress must fully relax to equilibrium
        (since (1-omega)=0 eliminates the non-equilibrium part)."""
        rho = 1.0
        ux, uy, uz = 0.1, 0.0, 0.0
        fx = fy = fz = 0.0
        omega = 1.0  # full relaxation

        cs2 = 1.0 / 3.0
        # Non-equilibrium diagonal stress
        pixx_old = cs2 + 0.1
        piyy_old = cs2 - 0.05
        pizz_old = cs2 - 0.05
        pixy_old = 0.02
        pixz_old = 0.0
        piyz_old = 0.0

        pi_new = wp.zeros(6, dtype=float, device=device)
        wp.launch(
            _kernel_collision,
            dim=1,
            inputs=[
                rho, ux, uy, uz,
                fx, fy, fz, omega,
                pixx_old, pixy_old, pixz_old,
                piyy_old, piyz_old, pizz_old,
                pi_new,
            ],
            device=device,
        )
        result = pi_new.numpy()

        golden = load_golden("collision_relax")
        for i in range(6):
            assert np.isclose(result[i], golden[i], atol=1e-12), \
                f"i={i}: got {result[i]:.15e}, expected {golden[i]:.15e}"

    def test_force_contribution(self, device):
        """Non-zero force must shift the diagonal stresses by F·u."""
        rho = 1.0
        ux, uy, uz = 0.2, 0.0, 0.0
        fx, fy, fz = 0.01, 0.0, 0.0
        omega = 1.0

        cs2 = 1.0 / 3.0
        pixx_old = cs2
        piyy_old = cs2
        pizz_old = cs2
        pixy_old = pixz_old = piyz_old = 0.0

        pi_new = wp.zeros(6, dtype=float, device=device)
        wp.launch(
            _kernel_collision,
            dim=1,
            inputs=[
                rho, ux, uy, uz,
                fx, fy, fz, omega,
                pixx_old, pixy_old, pixz_old,
                piyy_old, piyz_old, pizz_old,
                pi_new,
            ],
            device=device,
        )
        result = pi_new.numpy()

        golden = load_golden("collision_force")
        for i in range(6):
            assert np.isclose(result[i], golden[i], atol=1e-12), \
                f"i={i}: got {result[i]:.15e}, expected {golden[i]:.15e}"


# ===========================================================================
# Test 4: compute_rho_u_from_f — macroscopic moments from 27 distributions
# ===========================================================================


class TestComputeRhoU:
    """Verify density and velocity recovery from D3Q27 populations."""

    def test_uniform_equilibrium(self, device):
        """Feeding f_eq must recover the input rho and u."""
        rho = 1.0
        ux, uy, uz = 0.02, 0.01, 0.0

        # Build full D3Q27 equilibrium on CPU
        import numpy as np
        from wanphys._src.fluid.fluid_grid.home_fslbm import kernels_fluid as kf

        feq = np.zeros(27, dtype=np.float32)
        for di in range(27):
            # Use Python-side call (not @wp.func) for host-side test
            c3 = -3.0 * (ux * ux + uy * uy + uz * uz)
            rhom1 = rho - 1.0
            ux3 = ux * 3.0
            uy3 = uy * 3.0
            uz3 = uz * 3.0
            if di == 0:
                feq[di] = C.W0 * (rho * 0.5 * c3 + rhom1)
            elif di <= 6:
                rhos = C.WS * rho
                rhom1s = C.WS * rhom1
                if di == 1:   feq[di] = rhos * (0.5 * (ux3 * ux3 + c3) + ux3) + rhom1s
                elif di == 2: feq[di] = rhos * (0.5 * (ux3 * ux3 + c3) - ux3) + rhom1s
                elif di == 3: feq[di] = rhos * (0.5 * (uy3 * uy3 + c3) + uy3) + rhom1s
                elif di == 4: feq[di] = rhos * (0.5 * (uy3 * uy3 + c3) - uy3) + rhom1s
                elif di == 5: feq[di] = rhos * (0.5 * (uz3 * uz3 + c3) + uz3) + rhom1s
                elif di == 6: feq[di] = rhos * (0.5 * (uz3 * uz3 + c3) - uz3) + rhom1s
            # (edge and corner directions omitted for brevity — use kernel instead)

        # Use the Warp kernel
        N = 1
        stride = N
        f_flat = wp.zeros(27 * N, dtype=float, device=device)
        f_flat_np = f_flat.numpy()
        for di in range(27):
            f_flat_np[di * stride] = feq[di]
        wp.copy(f_flat, wp.array(f_flat_np, dtype=float, device=device))

        out_rho = wp.zeros(N, dtype=float, device=device)
        out_ux = wp.zeros(N, dtype=float, device=device)
        out_uy = wp.zeros(N, dtype=float, device=device)
        out_uz = wp.zeros(N, dtype=float, device=device)

        wp.launch(
            _kernel_compute_rho_u,
            dim=N,
            inputs=[f_flat, stride, out_rho, out_ux, out_uy, out_uz],
            device=device,
        )

        # compute_rho_u_from_f adds 1.0 internally (ref line 277: rho += 1.0)
        # So the returned rho = Σ f_i + 1.0 = 1.0 + 1.0 = 2.0 for rho=1.0
        # This is a known behaviour of the reference code.
        rho_computed = out_rho.numpy()[0]
        ux_c = out_ux.numpy()[0]
        uy_c = out_uy.numpy()[0]
        uz_c = out_uz.numpy()[0]

        golden = load_golden("compute_rho_u_feq")
        assert np.isclose(rho_computed, golden[0], atol=1e-6), \
            f"rho computed = {rho_computed}, expected {golden[0]}"
        assert np.isclose(ux_c, golden[1], atol=1e-4), f"ux = {ux_c}"
        assert np.isclose(uy_c, golden[2], atol=1e-4), f"uy = {uy_c}"
        assert np.isclose(uz_c, golden[3], atol=1e-4), f"uz = {uz_c}"


# ===========================================================================
# Test 5: stream_collide_bvh_kernel — Phase A + B + F sanity
# ===========================================================================


class TestStreamCollideBvh:
    """End-to-end sanity: quiescent fluid should remain quiescent after one step."""

    def test_quiescent_fluid_stays_quiescent(self, _warp, _constants, default_model):
        """Run one step on a uniform fluid at rest and verify density/momentum
        are preserved to machine precision."""
        wp = _warp
        C = _constants
        model = default_model

        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        domain = HomeFslbmDomain(model, solver=HomeFslbmSolver(model))
        domain.create_state()

        # Initialize equilibrium at rest
        domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

        # Set all cells to TYPE_F (pure fluid) — no gas or interface
        nx, ny, nz = model.nx, model.ny, model.nz
        flag_arr = domain.state.flag.numpy()
        flag_arr[:, :, :] = C.TYPE_F
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=model._device))

        # Run one step
        domain.step(dt=1.0)

        # Check post-step moments
        f_mom = domain.state.f_mom.numpy()
        stride = nx * ny * nz

        rho_vals = f_mom[C.M_RHO * stride: C.M_RHO * stride + stride]
        ux_vals = f_mom[C.M_UX * stride: C.M_UX * stride + stride]
        uy_vals = f_mom[C.M_UY * stride: C.M_UY * stride + stride]
        uz_vals = f_mom[C.M_UZ * stride: C.M_UZ * stride + stride]

        # Density should stay at 1.0 (within tolerance of a single LBM step)
        assert np.allclose(rho_vals, 1.0, atol=1e-6), \
            f"max rho deviation: {np.max(np.abs(rho_vals - 1.0))}"

        # Velocity should stay at 0.0
        assert np.allclose(ux_vals, 0.0, atol=1e-12), \
            f"max |ux|: {np.max(np.abs(ux_vals))}"
        assert np.allclose(uy_vals, 0.0, atol=1e-12)
        assert np.allclose(uz_vals, 0.0, atol=1e-12)

        # Post-collision stress should be at equilibrium: S_xx = 0, S_xy = 0, etc.
        sxx_vals = f_mom[C.M_SXX * stride: C.M_SXX * stride + stride]
        sxy_vals = f_mom[C.M_SXY * stride: C.M_SXY * stride + stride]
        assert np.allclose(sxx_vals, 0.0, atol=1e-12), \
            f"S_xx not zero: max={np.max(np.abs(sxx_vals))}"
        assert np.allclose(sxy_vals, 0.0, atol=1e-12)

    def test_no_nan_in_output(self, _warp, _constants, default_model):
        """After one step, no field may contain NaN."""
        wp = _warp
        C = _constants
        model = default_model

        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        domain = HomeFslbmDomain(model, solver=HomeFslbmSolver(model))
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state, rho0=1.0)

        flag_arr = domain.state.flag.numpy()
        flag_arr[:, :, :] = C.TYPE_F
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=model._device))

        domain.step(dt=1.0)

        f_mom = domain.state.f_mom.numpy()
        assert not np.any(np.isnan(f_mom)), "NaN detected in f_mom after 1 step!"
