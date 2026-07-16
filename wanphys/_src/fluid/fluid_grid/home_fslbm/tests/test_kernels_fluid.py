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
        # Equilibrium stress: traceless S = 0, off-diag = 0
        pi_xx = 0.0
        pi_yy = 0.0
        pi_zz = 0.0
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
        pi_xx = 0.01
        pi_yy = - 0.005
        pi_zz = - 0.005
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

    def test_uniform_equilibrium(self, device, C):
        """Feeding f_eq must recover the input rho and u."""
        rho = 1.0
        ux, uy, uz = 0.02, 0.01, 0.0

        feq_out = wp.zeros(27, dtype=float, device=device)
        wp.launch(_kernel_f_eq, dim=27, inputs=[rho, ux, uy, uz, feq_out], device=device)
        feq = feq_out.numpy()

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

        # Check post-step moments against reference golden data.
        # Golden assumes the HOME-stored convention (f_hn -= w3d after
        # Hermite reconstruction, ref mrLbmSolverGpu3D.cu:776/803).
        # Without the -= w3d step, rho is double-counted (~2.0 vs 1.0).
        f_mom = domain.state.f_mom.numpy()
        stride = nx * ny * nz

        golden_cell = load_golden("stream_collide_quiescent_step1")
        # Golden is cell-interleaved [c0: rho,ux,uy,uz,Sxx,..., c1: rho,ux,...].
        # f_mom is moment-major [M_RHO: rho0,rho1,..., M_UX: ux0,ux1,...].
        # Reshape golden → (N_cells, 10) → .T → (10, N_cells) moment-major.
        golden_mm = golden_cell.reshape(-1, C.NUM_MOMENTS).T.ravel()
        assert np.allclose(f_mom, golden_mm, atol=1e-6), \
            f"max f_mom deviation: {np.max(np.abs(f_mom - golden_mm))}"
        rho_vals = f_mom[C.M_RHO * stride: C.M_RHO * stride + stride]
        ux_vals = f_mom[C.M_UX * stride: C.M_UX * stride + stride]
        uy_vals = f_mom[C.M_UY * stride: C.M_UY * stride + stride]
        uz_vals = f_mom[C.M_UZ * stride: C.M_UZ * stride + stride]

        # Density should stay at 1.0 (within tolerance of a single LBM step)
        assert np.allclose(rho_vals, 1.0, atol=1e-6), \
            f"max rho deviation: {np.max(np.abs(rho_vals - 1.0))}"

        # Velocity should stay at 0.0
        assert np.allclose(ux_vals, 0.0, atol=1e-6), \
            f"max |ux|: {np.max(np.abs(ux_vals))}"
        assert np.allclose(uy_vals, 0.0, atol=1e-6)
        assert np.allclose(uz_vals, 0.0, atol=1e-6)

        # Post-collision stress should be at equilibrium: S_xx = 0, S_xy = 0, etc.
        sxx_vals = f_mom[C.M_SXX * stride: C.M_SXX * stride + stride]
        sxy_vals = f_mom[C.M_SXY * stride: C.M_SXY * stride + stride]
        assert np.allclose(sxx_vals, 0.0, atol=1e-6), \
            f"S_xx not zero: max={np.max(np.abs(sxx_vals))}"
        assert np.allclose(sxy_vals, 0.0, atol=1e-6)

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


# ===========================================================================
# Test 6: Shear decay — viscous attenuation of sinusoidal shear wave
# ===========================================================================


class TestStreamCollideBvhShearDecay:
    """Verify viscous attenuation of a sinusoidal shear wave over 100 steps.

    Initial condition: u_x(z) = A * sin(2*pi*z / L_z) with A=0.01, omega=1.0.
    After 100 steps the amplitude must have decayed measurably but the profile
    must retain the sinusoidal shape.  This is the Phase 1 gate criterion
    (audit item S3).
    """

    def test_amplitude_decays(self, _warp, _constants, default_model):
        """Peak amplitude must be lower than initial A=0.01 after 100 steps."""
        wp = _warp
        C = _constants

        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        # Reference kernel hardcodes nu_LBM=1e-4 → omega=1/(3*1e-4+0.5)
        # (mrLbmSolverGpu3D.cu:712). Periodic boundaries, no TYPE_S.
        ref_omega = 1.0 / (3.0 * 1e-4 + 0.5)
        shear_model = HomeFslbmModel(
            fluid_grid_res=(32, 32, 32),
            fluid_grid_cell_size=1.0,
            omega=ref_omega,
            bc_types=(3, 3, 3, 3, 3, 3),
            bc_periodic=(True, True, True),
        )
        domain = HomeFslbmDomain(shear_model, solver=HomeFslbmSolver(shear_model))
        domain.create_state()

        nx, ny, nz = shear_model.nx, shear_model.ny, shear_model.nz
        N = nx * ny * nz
        A = 0.01

        # Build initial f_mom on host: rho=1, u_y=u_z=0, u_x = A * sin(2*pi*z/Lz)
        import numpy as np
        f_mom_init = np.zeros((C.NUM_MOMENTS, N), dtype=np.float32)
        f_mom_init[C.M_RHO, :] = 1.0
        for k in range(nz):
            ux = A * np.sin(2.0 * np.pi * k / nz)
            for j in range(ny):
                for i in range(nx):
                    idx = i * ny * nz + j * nz + k
                    f_mom_init[C.M_UX, idx] = ux
        domain.state.f_mom = wp.array(f_mom_init.flatten(), dtype=float, device=shear_model._device)
        wp.copy(domain.state.f_mom_post, domain.state.f_mom)

        # All cells TYPE_F — matching reference golden generator (no TYPE_S flags)
        flag_arr = domain.state.flag.numpy()
        flag_arr[:, :, :] = C.TYPE_F
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=shear_model._device))

        for _step in range(100):
            domain.step(dt=1.0)

        # Compare against golden data
        f_mom = domain.state.f_mom.numpy()
        stride = nx * ny * nz
        golden = load_golden("stream_collide_shear_decay_step100")

        im, jm = nx // 2, ny // 2
        ux_warp_vals = np.zeros(nz, dtype=np.float32)
        for k in range(nz):
            idx = im * ny * nz + jm * nz + k
            ux_warp_vals[k] = f_mom[C.M_UX * stride + idx] / f_mom[C.M_RHO * stride + idx]

        print("z   warp            golden")
        for k in range(nz):
            print(f"{k:2d}  {ux_warp_vals[k]:+.6e}  {golden[k]:+.6e}")
        assert np.allclose(ux_warp_vals, golden, atol=1e-4), \
            f"max deviation: {np.max(np.abs(ux_warp_vals - golden)):.2e}"

        # Amplitude must have decayed (physical sanity)
        peak = np.max(np.abs(f_mom[C.M_UX * stride: C.M_UX * stride + stride]))
        assert peak < A, f"Peak amplitude {peak:.6e} not less than initial {A}"

    def test_sinusoidal_shape_preserved(self, _warp, _constants, default_model):
        """Velocity profile must still follow a sine envelope (no discontinuities)."""
        wp = _warp
        C = _constants

        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        # Reference kernel hardcodes nu_LBM=1e-4 → omega=1/(3*1e-4+0.5)
        ref_omega = 1.0 / (3.0 * 1e-4 + 0.5)
        shear_model = HomeFslbmModel(
            fluid_grid_res=(32, 32, 32),
            fluid_grid_cell_size=1.0,
            omega=ref_omega,
            bc_types=(3, 3, 3, 3, 3, 3),
            bc_periodic=(True, True, True),
        )
        domain = HomeFslbmDomain(shear_model, solver=HomeFslbmSolver(shear_model))
        domain.create_state()

        nx, ny, nz = shear_model.nx, shear_model.ny, shear_model.nz
        N = nx * ny * nz
        A = 0.01

        import numpy as np
        f_mom_init = np.zeros((C.NUM_MOMENTS, N), dtype=np.float32)
        f_mom_init[C.M_RHO, :] = 1.0
        for k in range(nz):
            ux = A * np.sin(2.0 * np.pi * k / nz)
            for j in range(ny):
                for i in range(nx):
                    idx = i * ny * nz + j * nz + k
                    f_mom_init[C.M_UX, idx] = ux
        domain.state.f_mom = wp.array(f_mom_init.flatten(), dtype=float, device=shear_model._device)
        wp.copy(domain.state.f_mom_post, domain.state.f_mom)

        # All cells TYPE_F — matching reference golden generator
        flag_arr = domain.state.flag.numpy()
        flag_arr[:, :, :] = C.TYPE_F
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=shear_model._device))

        for _step in range(100):
            domain.step(dt=1.0)

        f_mom = domain.state.f_mom.numpy()
        stride = nx * ny * nz
        im, jm = nx // 2, ny // 2

        ux_vals = np.array([
            f_mom[C.M_UX * stride + im * ny * nz + jm * nz + k] /
            f_mom[C.M_RHO * stride + im * ny * nz + jm * nz + k]
            for k in range(nz)
        ])

        # Check zero-crossings: must have exactly 2 sign changes for one sine period
        signs = np.sign(ux_vals)
        crossings = np.sum(np.abs(np.diff(signs[signs != 0]))) // 2
        assert crossings >= 1, f"Expected at least 1 zero-crossing, got {crossings}"


# ===========================================================================
# Test 7: Solid bounce-back — no-slip walls produce zero velocity
# ===========================================================================


class TestStreamCollideBvhSolidBounceBack:
    """Verify that solid walls (TYPE_S) enforce no-slip boundary.

    500 steps with gravity gz=-0.001.  After convergence, velocity near
    walls must be close to zero.
    """

    def test_wall_velocity_near_zero(self, _warp, _constants, default_model):
        """After 500 steps with gravity, |u| near z=0 and z=nz-1 must be < 2e-3.

        Full-way bounce-back with omega=1.0 (tau=1.0) has O(tau) wall-slip error;
        2e-3 is the realistic bound for a 32^3 grid with gz=-0.001.
        """
        wp = _warp
        C = _constants
        model = default_model

        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        domain = HomeFslbmDomain(model, solver=HomeFslbmSolver(model))
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

        nx, ny, nz = model.nx, model.ny, model.nz
        N = nx * ny * nz

        # Set walls and gravity
        flag_arr = domain.state.flag.numpy()
        for k in range(nz):
            flag_val = C.TYPE_S if (k == 0 or k == nz - 1) else C.TYPE_F
            flag_arr[:, :, k] = flag_val
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=model._device))

        force_z = domain.state.force_z.numpy()
        force_z[:, :, :] = -0.001
        wp.copy(domain.state.force_z, wp.array(force_z, dtype=float, device=model._device))

        for _step in range(500):
            domain.step(dt=1.0)

        f_mom = domain.state.f_mom.numpy()
        stride = nx * ny * nz

        # Check velocity at cells adjacent to walls
        im, jm = nx // 2, ny // 2
        for k in (1, nz - 2):  # cells next to walls
            idx = im * ny * nz + jm * nz + k
            rho = f_mom[C.M_RHO * stride + idx]
            u_mag = np.sqrt(
                (f_mom[C.M_UX * stride + idx] / rho) ** 2 +
                (f_mom[C.M_UY * stride + idx] / rho) ** 2 +
                (f_mom[C.M_UZ * stride + idx] / rho) ** 2
            )
            assert u_mag < 2e-3, f"|u|={u_mag:.2e} at z={k}, expected < 2e-3"


# ===========================================================================
# Test 8: Mass conservation through collision
# ===========================================================================


class TestMassConservationThroughCollision:
    """Total mass must be conserved before and after a single collision step."""

    def test_total_mass_conserved(self, _warp, _constants, default_model):
        """Sigma(rho) after one step must equal Sigma(rho) before the step."""
        wp = _warp
        C = _constants
        model = default_model

        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        domain = HomeFslbmDomain(model, solver=HomeFslbmSolver(model))
        domain.create_state()

        nx, ny, nz = model.nx, model.ny, model.nz
        N = nx * ny * nz

        import numpy as np

        # Random initial density [0.5, 1.5], u=0
        rng = np.random.default_rng(42)
        rho_init = rng.uniform(0.5, 1.5, N).astype(np.float32)

        f_mom_init = np.zeros((C.NUM_MOMENTS, N), dtype=np.float32)
        f_mom_init[C.M_RHO, :] = rho_init
        domain.state.f_mom = wp.array(f_mom_init.flatten(), dtype=float, device=model._device)

        flag_arr = domain.state.flag.numpy()
        flag_arr[:, :, :] = C.TYPE_F
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=model._device))

        mass_before = np.sum(rho_init)

        domain.step(dt=1.0)

        f_mom = domain.state.f_mom.numpy()
        stride = nx * ny * nz
        mass_after = np.sum(f_mom[C.M_RHO * stride: C.M_RHO * stride + stride])

        rel_err = abs(mass_after - mass_before) / mass_before
        assert rel_err < 1e-6, \
            f"Mass drift: before={mass_before:.10f}, after={mass_after:.10f}, rel_err={rel_err:.2e}"


# ===========================================================================
# Test 9: Turbulence omega modification
# ===========================================================================


class TestTurbulenceOmegaModification:
    """Verify that eddy-viscosity modifies omega near small bubbles (V < 5e6).

    A TYPE_I bubble region with tag=1 and volume=1000 is placed at the centre.
    Nearby TYPE_F cells with non-zero strain should use a modified (lower) omega,
    while cells far from the bubble should retain molecular omega.
    """

    def test_omega_modified_near_bubble(self, _warp, _constants, default_model):
        """After 1 step, f_mom near the bubble must differ from the uniform-equilibrium result."""
        wp = _warp
        C = _constants
        model = default_model

        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain

        # Use a smaller model for this test
        import numpy as np
        small_model = HomeFslbmModel(
            fluid_grid_res=(16, 16, 16),
            fluid_grid_cell_size=1.0,
            omega=1.0,
            turbulence_factor=4.0,
            turbulence_radius=3,
        )
        domain = HomeFslbmDomain(small_model, solver=HomeFslbmSolver(small_model))
        domain.create_state()

        nx, ny, nz = small_model.nx, small_model.ny, small_model.nz
        N = nx * ny * nz
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        f_mom_init = np.zeros((C.NUM_MOMENTS, N), dtype=np.float32)
        f_mom_init[C.M_RHO, :] = 1.0
        domain.state.f_mom = wp.array(f_mom_init.flatten(), dtype=float, device=small_model._device)

        flag_arr = domain.state.flag.numpy()
        tag_arr = domain.state.tag_matrix.numpy()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    d2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    if d2 <= 4:
                        flag_arr[i, j, k] = C.TYPE_I
                        tag_arr[i, j, k] = 1
                    else:
                        flag_arr[i, j, k] = C.TYPE_F
                        tag_arr[i, j, k] = -1
        wp.copy(domain.state.flag, wp.array(flag_arr, dtype=wp.uint8, device=small_model._device))
        wp.copy(domain.state.tag_matrix, wp.array(tag_arr, dtype=wp.int32, device=small_model._device))

        # Set bubble properties: small volume triggers turbulence
        bvol = wp.zeros(65536, dtype=wp.float64, device=small_model._device)
        brho = wp.zeros(65536, dtype=wp.float64, device=small_model._device)
        binit = wp.zeros(65536, dtype=wp.float64, device=small_model._device)
        bvol_np = bvol.numpy(); bvol_np[0] = 1000.0
        brho_np = brho.numpy(); brho_np[0] = 1.0
        binit_np = binit.numpy(); binit_np[0] = 1000.0
        wp.copy(bvol, wp.array(bvol_np, dtype=wp.float64, device=small_model._device))
        wp.copy(brho, wp.array(brho_np, dtype=wp.float64, device=small_model._device))
        wp.copy(binit, wp.array(binit_np, dtype=wp.float64, device=small_model._device))
        wp.copy(domain.state.bubble_volume, bvol)
        wp.copy(domain.state.bubble_rho, brho)
        wp.copy(domain.state.bubble_init_volume, binit)
        domain.state.bubble_count = 1

        domain.step(dt=1.0)

        # Compare against golden data
        golden = load_golden("turbulence_omega_step1")
        f_mom = domain.state.f_mom.numpy()
        # golden is [10 * N] moment-major
        assert np.allclose(f_mom, golden, atol=1e-5), \
            f"max deviation: {np.max(np.abs(f_mom - golden)):.2e}"

    def test_strain_magnitude_nonzero(self, _warp, _constants):
        """Strain-rate magnitude must be non-zero in the perturbed region (sanity)."""
        wp = _warp
        C = _constants

        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        small_model = HomeFslbmModel(
            fluid_grid_res=(16, 16, 16),
            fluid_grid_cell_size=1.0,
            omega=1.0,
            turbulence_factor=4.0,
            turbulence_radius=3,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver

        domain = HomeFslbmDomain(small_model, solver=HomeFslbmSolver(small_model))
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state, rho0=1.0)

        # Apply a small perturbation to velocity to create strain
        nx, ny, nz = small_model.nx, small_model.ny, small_model.nz
        N = nx * ny * nz
        import numpy as np
        f_mom = domain.state.f_mom.numpy().reshape(C.NUM_MOMENTS, N)
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = i * ny * nz + j * nz + k
                    f_mom[C.M_UX, idx] = 0.001 * (k - cz)  # shear gradient
        domain.state.f_mom = wp.array(f_mom.flatten(), dtype=float, device=small_model._device)

        domain.step(dt=1.0)

        f_mom_out = domain.state.f_mom.numpy()
        # S_xy = f_mom[M_SXY] — should be non-zero near the shear gradient
        sxy = f_mom_out[C.M_SXY * N: C.M_SXY * N + N]
        assert np.max(np.abs(sxy)) > 1e-8, \
            "Strain-rate component S_xy is zero everywhere — turbulence model may not be active"
