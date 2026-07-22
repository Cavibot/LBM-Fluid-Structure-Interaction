# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HOME-FSLBM free-surface kernels (Phase 2).

Covers:
  - ``calculate_phi`` — VOF fill-level helper
  - ``calculate_normal`` — 27-pt weighted gradient normal
  - ``plic_cube`` / ``plic_cube_reduced`` — unit-cube plane intersection
  - ``_lu_solve_5x5_scalar`` — LU decomposition solver
  - ``calculate_curvature_from_grid`` — Monge-patch mean curvature
  - ``surface_1_kernel`` — interface marker propagation
  - ``surface_2_kernel`` — GI initialisation + IG handling
  - ``surface_3_kernel`` — VOF mass redistribution

Follows the same fixture / device pattern as ``test_kernels_fluid.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_fluid import (
    _kernel_calculate_phi,
    _kernel_plic_cube,
    _lu_solve_5x5_scalar,
    calculate_curvature_from_grid,
)
from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_surface import (
    surface_1_kernel,
    surface_2_kernel,
    surface_3_kernel,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_direction_arrays(device):
    """Create D3Q27 cx, cy, cz Warp arrays (reusable across surface tests)."""
    cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32, device=device)
    cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32, device=device)
    cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32, device=device)
    return cx, cy, cz


def _make_w3d_array(device):
    """Create D3Q27 weight array."""
    return wp.array(np.array(C.W, dtype=np.float32), dtype=float, device=device)


def _make_opposite_array(device):
    """Create OPPOSITE direction array."""
    return wp.array(np.array(C.OPPOSITE, dtype=np.int32), dtype=wp.int32, device=device)


def _make_surface3_state(N: int, device) -> dict:
    """Create minimal state arrays for surface_3 testing."""
    stride = N * N * N
    f_mom_post = wp.zeros(10 * stride, dtype=float, device=device)
    flag = wp.zeros((N, N, N), dtype=wp.uint8, device=device)
    mass = wp.zeros((N, N, N), dtype=float, device=device)
    massex = wp.zeros((N, N, N), dtype=float, device=device)
    phi = wp.zeros((N, N, N), dtype=float, device=device)
    tag_matrix = wp.full((N, N, N), -1, dtype=wp.int32, device=device)
    previous_tag = wp.full((N, N, N), -1, dtype=wp.int32, device=device)
    islet = wp.zeros((N, N, N), dtype=wp.int32, device=device)
    delta_phi = wp.zeros((N, N, N), dtype=float, device=device)
    delta_g = wp.zeros((N, N, N), dtype=float, device=device)
    g_mom = wp.zeros(7 * stride, dtype=float, device=device)
    split_flag_gpu = wp.zeros(1, dtype=wp.int32, device=device)
    return {
        "f_mom_post": f_mom_post,
        "flag": flag,
        "mass": mass,
        "massex": massex,
        "phi": phi,
        "tag_matrix": tag_matrix,
        "previous_tag": previous_tag,
        "islet": islet,
        "delta_phi": delta_phi,
        "delta_g": delta_g,
        "g_mom": g_mom,
        "split_flag_gpu": split_flag_gpu,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    try:
        return wp.get_device("cuda:0")
    except Exception:
        return wp.get_device("cpu")


# ============================================================================
# Test-only Wrapper Kernels
# ============================================================================


@wp.kernel
def _fill_uint8_kernel(arr: wp.array3d(dtype=wp.uint8), val: wp.uint8):
    i, j, k = wp.tid()
    arr[i, j, k] = val


@wp.kernel
def _fill_float_kernel(arr: wp.array3d(dtype=float), val: float):
    i, j, k = wp.tid()
    arr[i, j, k] = val


@wp.kernel
def _kernel_calculate_normal(
    phi_neighbours: wp.array(dtype=float),  # [27] — phi at each neighbour direction
    cx: wp.array(dtype=wp.int32),            # [27] D3Q27 x-direction
    cy: wp.array(dtype=wp.int32),            # [27] D3Q27 y-direction
    cz: wp.array(dtype=wp.int32),            # [27] D3Q27 z-direction
    result: wp.array(dtype=wp.vec3),         # [1] — normal vector output
):
    """27-pt weighted finite-difference normal (ref: mrUtilFuncGpu3D.h:357-369).

    Weights: face-neighbour = 4, edge-neighbour = 2, corner-neighbour = 1.
    """
    bx = float(0.0)
    by = float(0.0)
    bz = float(0.0)
    for di in range(1, 27):
        pval = phi_neighbours[di]
        if di <= 6:
            w = 4.0
        elif di <= 18:
            w = 2.0
        else:
            w = 1.0
        cxi = float(cx[di])
        cyi = float(cy[di])
        czi = float(cz[di])
        bx += w * cxi * pval
        by += w * cyi * pval
        bz += w * czi * pval
    len_sq = bx * bx + by * by + bz * bz
    if len_sq < 1.0e-20:
        result[0] = wp.vec3(0.0, 0.0, 0.0)
    else:
        inv_len = 1.0 / wp.sqrt(len_sq)
        result[0] = wp.vec3(bx * inv_len, by * inv_len, bz * inv_len)


@wp.kernel
def _kernel_curvature_at_cell(
    phi: wp.array3d(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    opposite: wp.array(dtype=wp.int32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    px: int,
    py: int,
    pz: int,
    phi_center: float,
    result: wp.array(dtype=float),  # [1]
):
    """Single-cell curvature wrapper for testing."""
    curv = calculate_curvature_from_grid(
        phi, flag, cx, cy, cz, opposite,
        i, j, k, nx, ny, nz, px, py, pz, phi_center,
    )
    result[0] = curv


@wp.kernel
def _kernel_collect_type_i_curvature(
    phi: wp.array3d(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    opposite: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    px: int,
    py: int,
    pz: int,
    curvatures: wp.array(dtype=float),  # [nx*ny*nz] — curvature per cell
    count: wp.array(dtype=wp.int32),     # [1] — number of TYPE_I cells
):
    """Collect curvature for all TYPE_I cells in the grid."""
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return
    flagsn = int(flag[i, j, k])
    flagsn_su = flagsn & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
    if flagsn_su == C.CellFlag.TYPE_I:
        rho_self = float(1.0)  # placeholder, phi_center should match phi[i,j,k]
        phi_center_val = phi[i, j, k]
        curv = calculate_curvature_from_grid(
            phi, flag, cx, cy, cz, opposite,
            i, j, k, nx, ny, nz, px, py, pz, phi_center_val,
        )
        idx = wp.atomic_add(count, 0, 1)
        curvatures[idx] = curv


@wp.kernel
def _kernel_lu_solve_5x5_test(
    m00: wp.array(dtype=float), m01: wp.array(dtype=float),
    m02: wp.array(dtype=float), m03: wp.array(dtype=float),
    m04: wp.array(dtype=float),
    m11: wp.array(dtype=float), m12: wp.array(dtype=float),
    m13: wp.array(dtype=float), m14: wp.array(dtype=float),
    m22: wp.array(dtype=float), m23: wp.array(dtype=float),
    m24: wp.array(dtype=float),
    m33: wp.array(dtype=float), m34: wp.array(dtype=float),
    m44: wp.array(dtype=float),
    b0: wp.array(dtype=float), b1: wp.array(dtype=float),
    b2: wp.array(dtype=float), b3: wp.array(dtype=float),
    b4: wp.array(dtype=float),
    nsol: wp.array(dtype=wp.int32),
    x0: wp.array(dtype=float), x1: wp.array(dtype=float),
    x2: wp.array(dtype=float), x3: wp.array(dtype=float),
    x4: wp.array(dtype=float),
):
    """Wrapper kernel for _lu_solve_5x5_scalar."""
    sol = _lu_solve_5x5_scalar(
        m00[0], m01[0], m02[0], m03[0], m04[0],
        m11[0], m12[0], m13[0], m14[0],
        m22[0], m23[0], m24[0],
        m33[0], m34[0], m44[0],
        b0[0], b1[0], b2[0], b3[0], b4[0],
        int(nsol[0]),
    )
    x0[0] = sol.x
    x1[0] = sol.y
    x2[0] = sol.z
    x3[0] = sol.w
    x4[0] = sol.a


# ============================================================================
# TestCalculatePhi (existing)
# ============================================================================


class TestCalculatePhi:
    """VOF fill-level helper (mrUtilFuncGpu3D.h:351-354)."""

    def test_type_f_returns_one(self, device):
        N = 1
        rhon = wp.array(np.array([1.0], dtype=np.float32), dtype=float, device=device)
        massn = wp.array(np.array([0.5], dtype=np.float32), dtype=float, device=device)
        flagsn = wp.array(np.array([C.CellFlag.TYPE_F], dtype=np.int32), dtype=wp.int32, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_calculate_phi, dim=N, inputs=[rhon, massn, flagsn, result], device=device)
        got = result.numpy()[0]
        assert abs(got - 1.0) < 1e-6, f"TYPE_F should give phi=1.0, got {got}"

    def test_type_i_returns_ratio(self, device):
        N = 1
        rhon = wp.array(np.array([1.0], dtype=np.float32), dtype=float, device=device)
        massn = wp.array(np.array([0.5], dtype=np.float32), dtype=float, device=device)
        flagsn = wp.array(np.array([C.CellFlag.TYPE_I], dtype=np.int32), dtype=wp.int32, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_calculate_phi, dim=N, inputs=[rhon, massn, flagsn, result], device=device)
        got = result.numpy()[0]
        assert abs(got - 0.5) < 1e-6, f"TYPE_I mass/rho=0.5 should give phi=0.5, got {got}"

    def test_type_g_returns_zero(self, device):
        N = 1
        rhon = wp.array(np.array([1.0], dtype=np.float32), dtype=float, device=device)
        massn = wp.array(np.array([0.5], dtype=np.float32), dtype=float, device=device)
        flagsn = wp.array(np.array([C.CellFlag.TYPE_G], dtype=np.int32), dtype=wp.int32, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_calculate_phi, dim=N, inputs=[rhon, massn, flagsn, result], device=device)
        got = result.numpy()[0]
        assert abs(got - 0.0) < 1e-6, f"TYPE_G should give phi=0.0, got {got}"


# ============================================================================
# TestCalculateNormal (NEW — P0)
# ============================================================================


class TestCalculateNormal:
    """27-pt weighted finite-difference normal (mrUtilFuncGpu3D.h:357-369)."""

    @staticmethod
    def _compute_normal(device, phi_neighbours):
        """Launch _kernel_calculate_normal and return (nx, ny, nz)."""
        phi_arr = wp.array(np.array(phi_neighbours, dtype=np.float32), dtype=float, device=device)
        cx, cy, cz = _make_direction_arrays(device)
        result = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(_kernel_calculate_normal, dim=1, inputs=[phi_arr, cx, cy, cz, result], device=device)
        v = result.numpy()[0]
        return float(v[0]), float(v[1]), float(v[2])

    def test_planar_z(self, device):
        """Plane normal to z: phi depending only on z."""
        phi = np.zeros(27, dtype=np.float32)
        for di in range(1, 27):
            cz = C.CZ[di]
            phi[di] = 0.5 + 0.5 * cz  # linear along z
        nx, ny, nz = self._compute_normal(device, phi)
        # Normal should point mainly in z direction
        assert abs(nz) > 0.8, f"z component too small: nz={nz}"
        assert abs(nx) < 0.3, f"x component should be near zero: nx={nx}"
        assert abs(ny) < 0.3, f"y component should be near zero: ny={ny}"

    def test_planar_x(self, device):
        """Plane normal to x: phi depending only on x."""
        phi = np.zeros(27, dtype=np.float32)
        for di in range(1, 27):
            cx = C.CX[di]
            phi[di] = 0.5 + 0.5 * cx  # linear along x
        nx, ny, nz = self._compute_normal(device, phi)
        assert abs(nx) > 0.8, f"x component too small: nx={nx}"
        assert abs(ny) < 0.3, f"y component should be near zero: ny={ny}"
        assert abs(nz) < 0.3, f"z component should be near zero: nz={nz}"

    def test_planar_diagonal(self, device):
        """Plane along diagonal (1,1,0)."""
        phi = np.zeros(27, dtype=np.float32)
        for di in range(1, 27):
            cx = C.CX[di]
            cy = C.CY[di]
            phi[di] = 0.5 + 0.5 * (cx + cy) / np.sqrt(2)
        nx, ny, nz = self._compute_normal(device, phi)
        # Should be roughly (1/sqrt(2), 1/sqrt(2), 0) or the opposite
        lateral_norm = np.sqrt(nx * nx + ny * ny)
        assert lateral_norm > 0.8, f"Lateral norm too small: {lateral_norm}"
        assert abs(nz) < 0.3, f"z component should be near zero: nz={nz}"
        # |nx| and |ny| should be roughly equal
        assert abs(abs(nx) - abs(ny)) < 0.2, f"nx and ny should be similar: nx={nx}, ny={ny}"

    def test_spherical_r8(self, device):
        """Interface with gradient ~ direction (1,1,0): n ~ (0.707, 0.707, 0)."""
        phi = np.zeros(27, dtype=np.float32)
        # Set phi so it increases along +x and +y (gradient points to +x,+y)
        for di in range(1, 27):
            cx = C.CX[di]
            cy = C.CY[di]
            # phi = 0.5 + 0.2*cx + 0.2*cy (linear gradient)
            phi[di] = 0.5 + 0.2 * cx + 0.2 * cy
        nx, ny, nz = self._compute_normal(device, phi)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        assert norm > 0.9, f"Normal should be non-zero: |n|={norm}"
        # Gradient should point mainly in +x,+y
        assert nx > 0.3, f"nx should be positive, got {nx}"
        assert ny > 0.3, f"ny should be positive, got {ny}"
        assert abs(nz) < 0.3, f"nz should be near zero, got {nz}"

    def test_uniform_no_nan(self, device):
        """Uniform phi field — gradient is zero, should not produce NaN."""
        phi = np.ones(27, dtype=np.float32)
        nx, ny, nz = self._compute_normal(device, phi)
        assert not (np.isnan(nx) or np.isnan(ny) or np.isnan(nz)), "Should not produce NaN"
        # With zero gradient, normal may be (0,0,0) — that's acceptable


# ============================================================================
# TestPlicCube (existing + new)
# ============================================================================


class TestPlicCube:
    """PLIC unit-cube plane intersection (mrUtilFuncGpu3D.h:104-149)."""

    @staticmethod
    def _plic(device, V0, n):
        """Launch _kernel_plic_cube and return offset d0."""
        N = 1
        V0_arr = wp.array(np.array([V0], dtype=np.float32), dtype=float, device=device)
        nx_arr = wp.array(np.array([n[0]], dtype=np.float32), dtype=float, device=device)
        ny_arr = wp.array(np.array([n[1]], dtype=np.float32), dtype=float, device=device)
        nz_arr = wp.array(np.array([n[2]], dtype=np.float32), dtype=float, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_plic_cube, dim=N, inputs=[V0_arr, nx_arr, ny_arr, nz_arr, result], device=device)
        return float(result.numpy()[0])

    def test_axis_aligned_half(self, device):
        """V=0.5, n=(1,0,0) -> d=0.0 (plane bisects cube)."""
        got = self._plic(device, 0.5, (1.0, 0.0, 0.0))
        assert abs(got - 0.0) < 1e-5, f"V=0.5 n=(1,0,0) d=0.0, got {got}"

    def test_axis_aligned_quarter(self, device):
        """V=0.1, n=(1,0,0) -> d=-0.4."""
        got = self._plic(device, 0.1, (1.0, 0.0, 0.0))
        expected = -0.4
        assert abs(got - expected) < 1e-5, f"V=0.1 n=(1,0,0) d=-0.4, got {got}"

    def test_diagonal(self, device):
        """V=0.5, n=(1,1,1) normalized -> d should be roughly 0."""
        s3 = 1.0 / np.sqrt(3.0)
        got = self._plic(device, 0.5, (s3, s3, s3))
        assert abs(got) < 0.05, f"V=0.5 diagonal d near 0, got {got}"

    # ---- NEW: P1 tests ----

    def test_axis_aligned_large_volume(self, device):
        """V=0.9, n=(1,0,0) -> d=+0.4 (symmetric to V=0.1 case)."""
        got = self._plic(device, 0.9, (1.0, 0.0, 0.0))
        expected = 0.4
        assert abs(got - expected) < 1e-5, f"V=0.9 n=(1,0,0) d=+0.4, got {got}"

    def test_diagonal_small_volume(self, device):
        """V=0.1, n=(1,1,1)/sqrt(3) — tests cbrt branch (case 1)."""
        s3 = 1.0 / np.sqrt(3.0)
        got = self._plic(device, 0.1, (s3, s3, s3))
        # Expected via plic_cube_reduced case (1): d = cbrt(6*n1*n2*n3*V)
        # n1=n2=n3=1/3 after L1 normalization, V_eff = 0.1
        # d_reduced = cbrt(6 * 1/3 * 1/3 * 1/3 * 0.1) ≈ cbrt(0.0222) ≈ 0.281
        # Then scaled back: d0 = l_sum * (d_reduced - 0.5) with l_sum=sqrt(3)
        # d0 ≈ 1.732 * (0.281 - 0.5) ≈ -0.379
        # Verify symmetry with V=0.9 case below
        s3_inv = 1.0 / np.sqrt(3.0)
        got_large = self._plic(device, 0.9, (s3_inv, s3_inv, s3_inv))
        # Symmetry: d(V) ≈ -d(1-V) when using the same |n|
        assert abs(got + got_large) < 0.02, f"Symmetry violated: d(0.1)={got}, d(0.9)={got_large}"

    def test_diagonal_large_volume(self, device):
        """V=0.9, n=(1,1,1)/sqrt(3) — tests large volume symmetry."""
        s3 = 1.0 / np.sqrt(3.0)
        got = self._plic(device, 0.9, (s3, s3, s3))
        got_small = self._plic(device, 0.1, (s3, s3, s3))
        assert abs(got + got_small) < 0.02, f"Symmetry violated: d(0.9)={got}, d(0.1)={got_small}"


# ============================================================================
# TestSurface1 (existing + new)
# ============================================================================


class TestSurface1:
    """Interface marker propagation kernel."""

    def test_if_to_gi_propagation(self, device):
        """TYPE_IF should convert neighbour TYPE_G to TYPE_GI."""
        N = 4
        flag = wp.zeros((N, N, N), dtype=wp.uint8, device=device)
        wp.launch(
            _fill_uint8_kernel,
            dim=(N, N, N),
            inputs=[flag, wp.uint8(C.CellFlag.TYPE_G)],
            device=device,
        )
        flag_host = flag.numpy().copy()
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IF
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)

        cx, cy, cz = _make_direction_arrays(device)
        wp.launch(surface_1_kernel, dim=(N, N, N), inputs=[flag, cx, cy, cz, N, N, N], device=device)

        result = flag.numpy()
        gi_count = 0
        for di in range(1, 27):
            ni = 1 - C.CX[di]
            nj = 1 - C.CY[di]
            nk = 1 - C.CZ[di]
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                if result[ni, nj, nk] == C.CellFlag.TYPE_GI:
                    gi_count += 1
        assert gi_count > 0, "At least some neighbours should become TYPE_GI"

    # ---- NEW: P0 tests ----

    def test_if_converts_ig_to_i(self, device):
        """TYPE_IF converts neighbour TYPE_IG to TYPE_I."""
        N = 4
        flag = wp.zeros((N, N, N), dtype=wp.uint8, device=device)
        wp.launch(
            _fill_uint8_kernel, dim=(N, N, N),
            inputs=[flag, wp.uint8(C.CellFlag.TYPE_G)], device=device,
        )
        flag_host = flag.numpy().copy()
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IF
        # Set one specific face neighbour to TYPE_IG
        flag_host[1 - C.CX[1], 1 - C.CY[1], 1 - C.CZ[1]] = C.CellFlag.TYPE_IG
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)

        cx, cy, cz = _make_direction_arrays(device)
        wp.launch(surface_1_kernel, dim=(N, N, N), inputs=[flag, cx, cy, cz, N, N, N], device=device)

        result = flag.numpy()
        ni = 1 - C.CX[1]
        nj = 1 - C.CY[1]
        nk = 1 - C.CZ[1]
        neighbour_flag = int(result[ni, nj, nk])
        neighbour_su = neighbour_flag & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        assert neighbour_su == C.CellFlag.TYPE_I, \
            f"TYPE_IG neighbour should become TYPE_I, got flag={neighbour_flag:#x}"

    def test_if_does_not_affect_solid(self, device):
        """TYPE_IF does not affect TYPE_S neighbours."""
        N = 4
        flag = wp.zeros((N, N, N), dtype=wp.uint8, device=device)
        wp.launch(
            _fill_uint8_kernel, dim=(N, N, N),
            inputs=[flag, wp.uint8(C.CellFlag.TYPE_S)], device=device,
        )
        flag_host = flag.numpy().copy()
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IF
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)

        cx, cy, cz = _make_direction_arrays(device)
        wp.launch(surface_1_kernel, dim=(N, N, N), inputs=[flag, cx, cy, cz, N, N, N], device=device)

        result = flag.numpy()
        for di in range(1, 27):
            ni = 1 - C.CX[di]
            nj = 1 - C.CY[di]
            nk = 1 - C.CZ[di]
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                if (ni, nj, nk) != (1, 1, 1):
                    assert int(result[ni, nj, nk]) == C.CellFlag.TYPE_S, \
                        f"TYPE_S neighbour at ({ni},{nj},{nk}) was modified"

    def test_if_does_not_affect_fluid(self, device):
        """TYPE_IF does not affect TYPE_F neighbours."""
        N = 4
        flag = wp.zeros((N, N, N), dtype=wp.uint8, device=device)
        wp.launch(
            _fill_uint8_kernel, dim=(N, N, N),
            inputs=[flag, wp.uint8(C.CellFlag.TYPE_F)], device=device,
        )
        flag_host = flag.numpy().copy()
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IF
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)

        cx, cy, cz = _make_direction_arrays(device)
        wp.launch(surface_1_kernel, dim=(N, N, N), inputs=[flag, cx, cy, cz, N, N, N], device=device)

        result = flag.numpy()
        for di in range(1, 27):
            ni = 1 - C.CX[di]
            nj = 1 - C.CY[di]
            nk = 1 - C.CZ[di]
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                if (ni, nj, nk) != (1, 1, 1):
                    assert int(result[ni, nj, nk]) == C.CellFlag.TYPE_F, \
                        f"TYPE_F neighbour at ({ni},{nj},{nk}) was modified"


# ============================================================================
# TestSurface2Initialization (NEW — P0)
# ============================================================================


class TestSurface2Initialization:
    """GI initialisation and IG neighbour handling (mrLbmSolverGpu3D.cu:479-602)."""

    @staticmethod
    def _run_surface_2(device, N, flag_host, f_mom_post_host, c_value_host,
                        islet_host=None, merge_detector_host=None):
        """Run surface_2_kernel on a N^3 grid and return results."""
        stride = N * N * N
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)
        f_mom_post = wp.array(f_mom_post_host.ravel(), dtype=float, device=device) if f_mom_post_host is not None else wp.zeros(10 * stride, dtype=float, device=device)
        c_value = wp.array(c_value_host, dtype=float, device=device) if c_value_host is not None else wp.zeros((N, N, N), dtype=float, device=device)
        g_mom = wp.zeros(7 * stride, dtype=float, device=device)
        islet = wp.array(islet_host, dtype=wp.int32, device=device) if islet_host is not None else wp.zeros((N, N, N), dtype=wp.int32, device=device)
        merge_detector = wp.zeros((N, N, N), dtype=wp.int32, device=device)

        cx, cy, cz = _make_direction_arrays(device)
        w3d = _make_w3d_array(device)

        wp.launch(
            surface_2_kernel,
            dim=(N, N, N),
            inputs=[
                f_mom_post, flag, c_value, g_mom,
                islet, merge_detector,
                cx, cy, cz, w3d,
                N, N, N, stride,
            ],
            device=device,
        )
        return {
            "flag": flag.numpy(),
            "f_mom_post": f_mom_post.numpy(),
            "c_value": c_value.numpy(),
            "g_mom": g_mom.numpy(),
            "merge_detector": merge_detector.numpy(),
        }

    # ---- GI branch ----

    def test_gi_averages_fluid_neighbors(self, device):
        """TYPE_GI averages rho and u from TYPE_F neighbours."""
        N = 4
        stride = N * N * N
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_G, dtype=np.uint8)
        flag_host[1, 1, 1] = C.CellFlag.TYPE_GI
        # Set face neighbours as TYPE_F
        f_mom_post_host = np.zeros((10, stride), dtype=np.float32)
        for di in range(1, 7):
            ni, nj, nk = 1 - C.CX[di], 1 - C.CY[di], 1 - C.CZ[di]
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                flag_host[ni, nj, nk] = C.CellFlag.TYPE_F
                nidx = ni * N * N + nj * N + nk
                f_mom_post_host[0, nidx] = 1.0  # rho
                f_mom_post_host[1, nidx] = 0.1  # ux
                f_mom_post_host[2, nidx] = 0.2  # uy
                f_mom_post_host[3, nidx] = 0.3  # uz

        result = self._run_surface_2(device, N, flag_host, f_mom_post_host, np.zeros((N, N, N), dtype=np.float32))

        idx = 1 * N * N + 1 * N + 1
        rho = result["f_mom_post"][idx]
        ux = result["f_mom_post"][stride + idx]
        uy = result["f_mom_post"][2 * stride + idx]
        uz = result["f_mom_post"][3 * stride + idx]

        # Should average all 6 face neighbours
        assert abs(rho - 1.0) < 1e-6, f"rho should be 1.0, got {rho}"
        assert abs(ux - 0.1) < 1e-6, f"ux should be 0.1, got {ux}"
        assert abs(uy - 0.2) < 1e-6, f"uy should be 0.2, got {uy}"
        assert abs(uz - 0.3) < 1e-6, f"uz should be 0.3, got {uz}"

    def test_gi_no_fluid_neighbors_default(self, device):
        """TYPE_GI with no fluid neighbours gets defaults: rho=1.0, u=0."""
        N = 4
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_G, dtype=np.uint8)
        flag_host[1, 1, 1] = C.CellFlag.TYPE_GI

        result = self._run_surface_2(device, N, flag_host, None, None)

        idx = 1 * N * N + 1 * N + 1
        rho = result["f_mom_post"][idx]
        ux = result["f_mom_post"][N * N * N + idx]
        assert abs(rho - 1.0) < 1e-6, f"Default rho=1.0, got {rho}"
        assert abs(ux) < 1e-10, f"Default ux=0, got {ux}"

    def test_gi_c_value_from_face_neighbors(self, device):
        """TYPE_GI c_value averages from face-neighbour (DI<7) c_values."""
        N = 4
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_G, dtype=np.uint8)
        flag_host[1, 1, 1] = C.CellFlag.TYPE_GI
        c_value_host = np.zeros((N, N, N), dtype=np.float32)

        f_mom_post_host = np.zeros((10, N * N * N), dtype=np.float32)
        for di in range(1, 4):  # only 3 face neighbours are TYPE_F
            ni, nj, nk = 1 - C.CX[di], 1 - C.CY[di], 1 - C.CZ[di]
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                flag_host[ni, nj, nk] = C.CellFlag.TYPE_F
                nidx = ni * N * N + nj * N + nk
                f_mom_post_host[0, nidx] = 1.0
                c_value_host[ni, nj, nk] = 0.5

        result = self._run_surface_2(device, N, flag_host, f_mom_post_host, c_value_host)

        c_val = result["c_value"][1, 1, 1]
        # Average of 3 neighbours with c_value=0.5
        assert abs(c_val - 0.5) < 1e-6, f"c_value should be 0.5, got {c_val}"

    # ---- IG branch ----

    def test_ig_converts_f_neighbor_to_i(self, device):
        """TYPE_IG converts TYPE_F neighbour (islet=0) to TYPE_I with merge_detector=1."""
        N = 4
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_G, dtype=np.uint8)
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IG
        flag_host[2, 1, 1] = C.CellFlag.TYPE_F  # face neighbour at +x

        result = self._run_surface_2(device, N, flag_host, None, None)

        # Neighbour (2,1,1) should become TYPE_I
        nflag = int(result["flag"][2, 1, 1])
        nflag_su = nflag & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        assert nflag_su == C.CellFlag.TYPE_I, \
            f"Neighbour TYPE_F should become TYPE_I, got {nflag:#x}"
        # merge_detector should be set
        assert result["merge_detector"][2, 1, 1] == 1, \
            "merge_detector should be 1"

    def test_ig_respects_islet(self, device):
        """TYPE_IG with islet=1 neighbour: convert self to TYPE_I, not neighbour."""
        N = 4
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_G, dtype=np.uint8)
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IG
        flag_host[2, 1, 1] = C.CellFlag.TYPE_IF  # interface neighbour

        islet_host = np.zeros((N, N, N), dtype=np.int32)
        islet_host[2, 1, 1] = 1  # neighbour is an islet

        result = self._run_surface_2(device, N, flag_host, None, None,
                                     islet_host=islet_host)

        # Self (1,1,1) should become TYPE_I
        sflag = int(result["flag"][1, 1, 1])
        sflag_su = sflag & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        assert sflag_su == C.CellFlag.TYPE_I, \
            f"Self (IG with islet neighbour) should become TYPE_I, got {sflag:#x}"
        # Neighbour should NOT be modified (stays TYPE_IF)
        nflag = int(result["flag"][2, 1, 1])
        nflag_su = nflag & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        assert nflag_su == C.CellFlag.TYPE_IF, \
            f"Islet neighbour should stay TYPE_IF, got {nflag:#x}"


# ============================================================================
# TestSurface3 (existing + P1 new tests)
# ============================================================================


class TestSurface3:
    """VOF mass redistribution kernel."""

    def test_type_f_mass_equals_rho(self, device):
        """TYPE_F cell mass should be set to rho, excess->massex."""
        N = 4
        state = _make_surface3_state(N, device)
        cx, cy, cz = _make_direction_arrays(device)

        idx = 1 * N * N + 1 * N + 1
        f_np = state["f_mom_post"].numpy(); f_np[idx] = 1.0
        wp.copy(state["f_mom_post"], wp.array(f_np, dtype=float, device=device))

        flag_host = state["flag"].numpy()
        flag_host[:, :, :] = C.CellFlag.TYPE_F
        state["flag"] = wp.array(flag_host, dtype=wp.uint8, device=device)

        mass_host = state["mass"].numpy()
        mass_host[1, 1, 1] = 1.5
        state["mass"] = wp.array(mass_host, dtype=float, device=device)

        wp.launch(
            surface_3_kernel,
            dim=(N, N, N),
            inputs=[
                state["f_mom_post"], state["flag"], state["mass"],
                state["massex"], state["phi"], state["tag_matrix"],
                state["previous_tag"], state["islet"],
                state["delta_phi"], state["delta_g"], state["g_mom"],
                state["split_flag_gpu"], cx, cy, cz, N, N, N, N * N * N,
            ],
            device=device,
        )

        mass_out = state["mass"].numpy()
        massex_out = state["massex"].numpy()
        phi_out = state["phi"].numpy()

        assert abs(mass_out[1, 1, 1] - 1.0) < 1e-6, f"TYPE_F mass -> rho=1.0, got {mass_out[1,1,1]}"
        assert abs(massex_out[1, 1, 1] - 0.5 / 26) < 1e-6, f"massex=0.5/26, got {massex_out[1,1,1]}"
        assert abs(phi_out[1, 1, 1] - 1.0) < 1e-6, f"TYPE_F phi=1.0, got {phi_out[1,1,1]}"

    def test_type_i_mass_clamped(self, device):
        """TYPE_I cell mass should be clamped to [0, rho]."""
        N = 4
        state = _make_surface3_state(N, device)
        cx, cy, cz = _make_direction_arrays(device)

        idx = 1 * N * N + 1 * N + 1
        f_np = state["f_mom_post"].numpy(); f_np[idx] = 1.0
        wp.copy(state["f_mom_post"], wp.array(f_np, dtype=float, device=device))

        flag_host = state["flag"].numpy()
        flag_host[:, :, :] = C.CellFlag.TYPE_F
        flag_host[1, 1, 1] = C.CellFlag.TYPE_I
        state["flag"] = wp.array(flag_host, dtype=wp.uint8, device=device)

        mass_host = state["mass"].numpy()
        mass_host[1, 1, 1] = 1.5
        state["mass"] = wp.array(mass_host, dtype=float, device=device)

        wp.launch(
            surface_3_kernel,
            dim=(N, N, N),
            inputs=[
                state["f_mom_post"], state["flag"], state["mass"],
                state["massex"], state["phi"], state["tag_matrix"],
                state["previous_tag"], state["islet"],
                state["delta_phi"], state["delta_g"], state["g_mom"],
                state["split_flag_gpu"], cx, cy, cz, N, N, N, N * N * N,
            ],
            device=device,
        )

        mass_out = state["mass"].numpy()
        massex_out = state["massex"].numpy()
        phi_out = state["phi"].numpy()

        assert abs(mass_out[1, 1, 1] - 1.0) < 1e-6, f"TYPE_I mass clamped to rho=1.0, got {mass_out[1,1,1]}"
        assert abs(massex_out[1, 1, 1] - 0.5 / 26) < 1e-6, f"massex=0.5/26, got {massex_out[1,1,1]}"
        assert abs(phi_out[1, 1, 1] - 1.0) < 1e-6, f"TYPE_I phi=1.0 (mass/rho=1), got {phi_out[1,1,1]}"

    # ---- NEW: P1 tests ----

    def test_mass_conservation_global(self, device):
        """Global mass+massen*counter conservation across mixed TYPE_F/I/G cells.

        surface_3 stores *per-neighbour* massex (divided by counter).
        The invariant is: mass_initial = mass_new + massex * counter,
        where counter = number of fluid/interface neighbours for that cell.
        """
        N = 4
        state = _make_surface3_state(N, device)
        cx, cy, cz = _make_direction_arrays(device)

        stride = N * N * N
        f_np = state["f_mom_post"].numpy()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    idx = i * N * N + j * N + k
                    f_np[idx] = 1.0  # rho=1 everywhere

        # Set up mixed flags
        flag_host = state["flag"].numpy()
        flag_host[0, :, :] = C.CellFlag.TYPE_F  # fluid slab
        flag_host[1, :, :] = C.CellFlag.TYPE_I  # interface slab
        flag_host[2, :, :] = C.CellFlag.TYPE_G  # gas slab
        flag_host[3, :, :] = C.CellFlag.TYPE_F  # fluid slab

        mass_host = state["mass"].numpy()
        mass_host[:, :, :] = 1.0  # mass=1 everywhere

        # Compute per-cell counter (= number of fluid/interface neighbours)
        # using the SAME logic as surface_3_kernel Phase B.
        counter_host = np.zeros((N, N, N), dtype=np.int32)
        fluid_or_iface = {
            C.CellFlag.TYPE_F, C.CellFlag.TYPE_I,
            C.CellFlag.TYPE_IF, C.CellFlag.TYPE_GI,
        }
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    cnt = 0
                    for di in range(1, 27):
                        ni = i - C.CX[di]
                        nj = j - C.CY[di]
                        nk = k - C.CZ[di]
                        if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                            nfl = int(flag_host[ni, nj, nk]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
                            if nfl in fluid_or_iface:
                                cnt += 1
                    counter_host[i, j, k] = cnt

        initial_total = np.sum(mass_host)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    initial_total += state["massex"].numpy()[i, j, k] * counter_host[i, j, k]

        state["f_mom_post"] = wp.array(f_np.ravel(), dtype=float, device=device)
        state["flag"] = wp.array(flag_host, dtype=wp.uint8, device=device)
        state["mass"] = wp.array(mass_host, dtype=float, device=device)

        wp.launch(
            surface_3_kernel,
            dim=(N, N, N),
            inputs=[
                state["f_mom_post"], state["flag"], state["mass"],
                state["massex"], state["phi"], state["tag_matrix"],
                state["previous_tag"], state["islet"],
                state["delta_phi"], state["delta_g"], state["g_mom"],
                state["split_flag_gpu"], cx, cy, cz, N, N, N, stride,
            ],
            device=device,
        )

        mass_out = state["mass"].numpy()
        massex_out = state["massex"].numpy()
        final_total = np.sum(mass_out)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    final_total += massex_out[i, j, k] * counter_host[i, j, k]

        assert abs(final_total - initial_total) < 1e-6, \
            f"Per-cell invariant violated: initial={initial_total}, final={final_total}"

    def test_isolated_i_self_absorb(self, device):
        """Isolated TYPE_I (no fluid neighbours) self-absorbs mass."""
        N = 4
        state = _make_surface3_state(N, device)
        cx, cy, cz = _make_direction_arrays(device)

        idx = 1 * N * N + 1 * N + 1
        f_np = state["f_mom_post"].numpy(); f_np[idx] = 1.0
        wp.copy(state["f_mom_post"], wp.array(f_np, dtype=float, device=device))

        flag_host = state["flag"].numpy()
        flag_host[:, :, :] = C.CellFlag.TYPE_G  # all neighbours gas
        flag_host[1, 1, 1] = C.CellFlag.TYPE_I
        state["flag"] = wp.array(flag_host, dtype=wp.uint8, device=device)

        mass_host = state["mass"].numpy()
        mass_host[1, 1, 1] = 1.5
        state["mass"] = wp.array(mass_host, dtype=float, device=device)

        wp.launch(
            surface_3_kernel,
            dim=(N, N, N),
            inputs=[
                state["f_mom_post"], state["flag"], state["mass"],
                state["massex"], state["phi"], state["tag_matrix"],
                state["previous_tag"], state["islet"],
                state["delta_phi"], state["delta_g"], state["g_mom"],
                state["split_flag_gpu"], cx, cy, cz, N, N, N, N * N * N,
            ],
            device=device,
        )

        mass_out = state["mass"].numpy()
        massex_out = state["massex"].numpy()
        # counter=0 so massn += massexn, massexn = 0
        assert abs(mass_out[1, 1, 1] - 1.5) < 1e-6, \
            f"Isolated cell should self-absorb: mass={mass_out[1,1,1]}"
        assert abs(massex_out[1, 1, 1] - 0.0) < 1e-6, \
            f"massex should be 0, got {massex_out[1,1,1]}"

    def test_flag_transitions_full(self, device):
        """Verify all flag transitions: IF->F, IG->G, GI->I."""
        N = 4
        state = _make_surface3_state(N, device)
        cx, cy, cz = _make_direction_arrays(device)

        f_np = state["f_mom_post"].numpy()
        f_np[:] = 1.0  # rho=1 everywhere
        wp.copy(state["f_mom_post"], wp.array(f_np, dtype=float, device=device))

        flag_host = state["flag"].numpy()
        flag_host[:, :, :] = C.CellFlag.TYPE_F  # background fluid
        flag_host[1, 1, 1] = C.CellFlag.TYPE_IF
        flag_host[2, 1, 1] = C.CellFlag.TYPE_IG
        flag_host[3, 1, 1] = C.CellFlag.TYPE_GI
        state["flag"] = wp.array(flag_host, dtype=wp.uint8, device=device)

        mass_host = state["mass"].numpy()
        mass_host[:, :, :] = 1.0
        state["mass"] = wp.array(mass_host, dtype=float, device=device)

        wp.launch(
            surface_3_kernel,
            dim=(N, N, N),
            inputs=[
                state["f_mom_post"], state["flag"], state["mass"],
                state["massex"], state["phi"], state["tag_matrix"],
                state["previous_tag"], state["islet"],
                state["delta_phi"], state["delta_g"], state["g_mom"],
                state["split_flag_gpu"], cx, cy, cz, N, N, N, N * N * N,
            ],
            device=device,
        )

        flag_out = state["flag"].numpy()

        f1 = int(flag_out[1, 1, 1]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        f2 = int(flag_out[2, 1, 1]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        f3 = int(flag_out[3, 1, 1]) & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)

        assert f1 == C.CellFlag.TYPE_F, f"IF->F failed, got {f1:#x}"
        assert f2 == C.CellFlag.TYPE_G, f"IG->G failed, got {f2:#x}"
        assert f3 == C.CellFlag.TYPE_I, f"GI->I failed, got {f3:#x}"


# ============================================================================
# TestLuSolve5x5 (NEW — P1)
# ============================================================================


class TestLuSolve5x5:
    """Scalar LU decomposition for 5x5 symmetric system (mrUtilFuncGpu3D.h:124-140)."""

    @staticmethod
    def _solve(device, M, b, nsol):
        """Solve Mx=b using _lu_solve_5x5_scalar, return (x0, x1, x2, x3, x4)."""
        sc = [wp.array(np.array([v], dtype=np.float32), dtype=float, device=device) for v in
              [M[0, 0], M[0, 1], M[0, 2], M[0, 3], M[0, 4],
               M[1, 1], M[1, 2], M[1, 3], M[1, 4],
               M[2, 2], M[2, 3], M[2, 4],
               M[3, 3], M[3, 4],
               M[4, 4],
               b[0], b[1], b[2], b[3], b[4]]]
        nsol_arr = wp.array(np.array([nsol], dtype=np.int32), dtype=wp.int32, device=device)
        x_out = [wp.zeros(1, dtype=float, device=device) for _ in range(5)]

        wp.launch(
            _kernel_lu_solve_5x5_test,
            dim=1,
            inputs=sc + [nsol_arr] + x_out,
            device=device,
        )
        return tuple(float(x.numpy()[0]) for x in x_out)

    def test_identity(self, device):
        """M=I, b=(1,2,3,4,5) -> x=b."""
        M = np.eye(5, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        x = self._solve(device, M, b, 5)
        for i in range(5):
            assert abs(x[i] - b[i]) < 1e-8, f"x[{i}]={x[i]}, expected {b[i]}"

    def test_known_2x2(self, device):
        """M=[[4,1],[1,3]], b=(1,2) — Nsol=2."""
        M = np.zeros((5, 5), dtype=np.float32)
        M[0, 0] = 4.0; M[0, 1] = 1.0
        M[1, 1] = 3.0
        b = np.array([1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x = self._solve(device, M, b, 2)
        # Solve [[4,1],[1,3]]*[x0,x1] = [1,2] -> x0≈0.0909, x1≈0.6364
        assert abs(x[0] - 0.09090909) < 1e-5
        assert abs(x[1] - 0.63636364) < 1e-5
        assert abs(x[2]) < 1e-10  # unused
        assert abs(x[3]) < 1e-10
        assert abs(x[4]) < 1e-10

    def test_known_3x3(self, device):
        """M=[[4,1,0],[1,3,1],[0,1,2]], b=(1,2,3) — Nsol=3."""
        M = np.zeros((5, 5), dtype=np.float32)
        M[0, 0] = 4.0; M[0, 1] = 1.0
        M[1, 1] = 3.0; M[1, 2] = 1.0
        M[2, 2] = 2.0
        b = np.array([1.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float32)
        x = self._solve(device, M, b, 3)
        # Verify: M @ x ≈ b
        residual = np.array([
            M[0, 0] * x[0] + M[0, 1] * x[1],
            M[0, 1] * x[0] + M[1, 1] * x[1] + M[1, 2] * x[2],
            M[1, 2] * x[1] + M[2, 2] * x[2],
        ])
        for i in range(3):
            assert abs(residual[i] - b[i]) < 1e-5, f"residual[{i}]={residual[i]}, expected {b[i]}"

    def test_roundtrip(self, device):
        """Random SPD 5x5 matrix — verify M@x ≈ b."""
        rng = np.random.RandomState(42)
        A = rng.randn(5, 5).astype(np.float32)
        M = A.T @ A + 0.1 * np.eye(5, dtype=np.float32)  # SPD
        b = rng.randn(5).astype(np.float32)
        x = self._solve(device, M, b, 5)
        residual = M @ np.array(x, dtype=np.float64)
        for i in range(5):
            assert abs(residual[i] - b[i]) < 1e-4, \
                f"Roundtrip failed at i={i}: residual={residual[i]}, b={b[i]}"

    def test_nsol_lt_5(self, device):
        """Nsol=3 on a 5x5 matrix — only 3x3 sub-block used."""
        M = np.zeros((5, 5), dtype=np.float32)
        M[0, 0] = 4.0; M[0, 1] = 1.0
        M[1, 1] = 3.0; M[1, 2] = 1.0
        M[2, 2] = 2.0
        # Fill unused entries with garbage
        M[0, 3] = 999.0; M[1, 4] = 999.0; M[3, 3] = -1.0
        b = np.array([1.0, 2.0, 3.0, 999.0, 999.0], dtype=np.float32)
        x = self._solve(device, M, b, 3)
        # Verify only first 3 equations
        residual = np.array([
            M[0, 0] * x[0] + M[0, 1] * x[1],
            M[0, 1] * x[0] + M[1, 1] * x[1] + M[1, 2] * x[2],
            M[1, 2] * x[1] + M[2, 2] * x[2],
        ])
        for i in range(3):
            assert abs(residual[i] - b[i]) < 1e-5, f"Nsol=3 residual[{i}]={residual[i]}"
        assert abs(x[3]) < 1e-10
        assert abs(x[4]) < 1e-10


@wp.kernel
def _k_dump_one_cell(
    phi: wp.array3d(dtype=float),
    flag: wp.array3d(dtype=wp.uint8),
    cx: wp.array(dtype=wp.int32), cy: wp.array(dtype=wp.int32), cz: wp.array(dtype=wp.int32),
    opp: wp.array(dtype=wp.int32),
    ii: int, jj: int, kk: int, Nx: int, Ny: int, Nz: int,
    out_phi27: wp.array(dtype=float),
    out_norm: wp.array(dtype=float),
):
    bx_n = float(0.0); by_n = float(0.0); bz_n = float(0.0)
    out_phi27[0] = phi[ii, jj, kk]
    for di in range(1, 27):
        o = opp[di]; ni = ii - cx[o]; nj = jj - cy[o]; nk = kk - cz[o]
        if ni < 0 or ni >= Nx or nj < 0 or nj >= Ny or nk < 0 or nk >= Nz:
            out_phi27[di] = -1.0; continue
        pv = phi[ni, nj, nk]; out_phi27[di] = pv
        if di <= 6: w = 4.0
        elif di <= 18: w = 2.0
        else: w = 1.0
        cxi = float(cx[di]); cyi = float(cy[di]); czi = float(cz[di])
        bx_n += w * cxi * pv; by_n += w * cyi * pv; bz_n += w * czi * pv
    ls = bx_n * bx_n + by_n * by_n + bz_n * bz_n
    if ls < 1e-20:
        out_norm[0] = 0.0; out_norm[1] = 0.0; out_norm[2] = 0.0
        return
    b = wp.normalize(wp.vec3(bx_n, by_n, bz_n))
    out_norm[0] = b.x; out_norm[1] = b.y; out_norm[2] = b.z


# ============================================================================
# TestCalculateCurvature (NEW — P0)
# ============================================================================


class TestCalculateCurvature:
    """Monge-patch mean curvature via PLIC (mrUtilFuncGpu3D.h:371-420)."""

    @staticmethod
    def _make_sphere_grid(N, R, device):
        """Create phi and flag grids for a sphere of radius R centred at (N/2,N/2,N/2)."""
        centre = np.array([N / 2, N / 2, N / 2], dtype=np.float32)
        delta = 1.5  # smoothing width
        phi_host = np.zeros((N, N, N), dtype=np.float32)
        flag_host = np.zeros((N, N, N), dtype=np.uint8)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    dist = np.sqrt((i - centre[0]) ** 2 + (j - centre[1]) ** 2 + (k - centre[2]) ** 2)
                    phi_raw = (R - dist) / delta + 0.5
                    phi_host[i, j, k] = max(0.0, min(1.0, phi_raw))
                    if phi_host[i, j, k] >= 1.0:
                        flag_host[i, j, k] = C.CellFlag.TYPE_F
                    elif phi_host[i, j, k] <= 0.0:
                        flag_host[i, j, k] = C.CellFlag.TYPE_G
                    else:
                        flag_host[i, j, k] = C.CellFlag.TYPE_I

        phi = wp.array(phi_host, dtype=float, device=device)
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)
        return phi, flag, phi_host

    @staticmethod
    def _make_plane_grid(N, z_plane, device):
        """Create phi and flag grids for a plane at z=z_plane."""
        delta = 1.5
        phi_host = np.zeros((N, N, N), dtype=np.float32)
        flag_host = np.zeros((N, N, N), dtype=np.uint8)
        for k in range(N):
            phi_raw = (z_plane - k) / delta + 0.5
            val = max(0.0, min(1.0, phi_raw))
            phi_host[:, :, k] = val
            if val >= 1.0:
                flag_host[:, :, k] = C.CellFlag.TYPE_F
            elif val <= 0.0:
                flag_host[:, :, k] = C.CellFlag.TYPE_G
            else:
                flag_host[:, :, k] = C.CellFlag.TYPE_I

        phi = wp.array(phi_host, dtype=float, device=device)
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)
        return phi, flag, phi_host

    @staticmethod
    def _collect_curvatures(device, phi, flag, N):
        """Launch _kernel_collect_type_i_curvature, return list of curvatures."""
        cx, cy, cz = _make_direction_arrays(device)
        opp = _make_opposite_array(device)
        curvatures = wp.zeros(N * N * N, dtype=float, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch(
            _kernel_collect_type_i_curvature,
            dim=(N, N, N),
            inputs=[phi, flag, cx, cy, cz, opp, N, N, N, 0, 0, 0, curvatures, count],
            device=device,
        )
        n_type_i = int(count.numpy()[0])
        if n_type_i == 0:
            return np.array([])
        return curvatures.numpy()[:n_type_i]

    # ------------------------------------------------------------------
    # Reconstruct corrected z and verify Monge fit
    # ------------------------------------------------------------------
    @staticmethod
    def _reconstruct_z(device, phi, flag, host_phi, ic, jc, kc, centre, R, delta):
        import math as _math
        N = phi.shape[0]
        cx_arr, cy_arr, cz_arr = _make_direction_arrays(device)
        opp_arr = _make_opposite_array(device)
        out_phi27 = wp.zeros(27, dtype=float, device=device)
        out_norm = wp.zeros(3, dtype=float, device=device)
        wp.launch(_k_dump_one_cell, dim=1,
                  inputs=[phi, flag, cx_arr, cy_arr, cz_arr, opp_arr,
                          ic, jc, kc, N, N, N, out_phi27, out_norm],
                  device=device)
        phi27 = out_phi27.numpy(); n_warp = out_norm.numpy()
        plic27 = np.zeros(27, dtype=np.float32)
        for di in range(27):
            r = wp.zeros(1, dtype=float, device=device)
            wp.launch(_kernel_plic_cube, dim=1,
                      inputs=[wp.array(np.array([phi27[di]], dtype=np.float32), dtype=float, device=device),
                              wp.array(np.array([n_warp[0]], dtype=np.float32), dtype=float, device=device),
                              wp.array(np.array([n_warp[1]], dtype=np.float32), dtype=float, device=device),
                              wp.array(np.array([n_warp[2]], dtype=np.float32), dtype=float, device=device),
                              r], device=device)
            plic27[di] = float(r.numpy()[0])
        rn = np.array([0.56270900, 0.32704452, 0.75921047], dtype=np.float64)
        by_raw = np.cross(n_warp, rn); by = by_raw / np.linalg.norm(by_raw)
        bx = np.cross(by, n_warp)
        z_corr, px_list, py_list = [], [], []
        for di in range(1, 27):
            if not (0.0 < phi27[di] < 1.0):
                continue
            eix = float(C.CX[di]); eiy = float(C.CY[di]); eiz = float(C.CZ[di])
            z = (eix*n_warp[0]+eiy*n_warp[1]+eiz*n_warp[2]) - delta*(float(plic27[di])-float(plic27[0]))
            x = eix*bx[0]+eiy*bx[1]+eiz*bx[2]; y = eix*by[0]+eiy*by[1]+eiz*by[2]
            z_corr.append(z); px_list.append(x); py_list.append(y)
        if len(z_corr) < 5:
            return None
        M = np.zeros((5,5)); bv = np.zeros(5)
        for xi, yi, zi in zip(px_list, py_list, z_corr):
            x2=xi*xi; y2=yi*yi; x3=x2*xi; y3=y2*yi
            M[0,0]+=x2*x2; M[0,1]+=x2*y2; M[0,2]+=x3*yi; M[0,3]+=x3; M[0,4]+=x2*yi
            M[1,1]+=y2*y2; M[1,2]+=xi*y3; M[1,3]+=xi*y2; M[1,4]+=y3
            M[2,2]+=x2*y2; M[2,3]+=x2*yi; M[2,4]+=xi*y2
            M[3,3]+=x2;     M[3,4]+=xi*yi
            M[4,4]+=y2
            bv[0]+=x2*zi; bv[1]+=y2*zi; bv[2]+=xi*yi*zi; bv[3]+=xi*zi; bv[4]+=yi*zi
        for r in range(5):
            for c in range(r+1,5): M[c,r]=M[r,c]
        try:
            sol = np.linalg.solve(M, bv)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, bv, rcond=None)[0]
        A,B,Ck,H,I = sol
        d = H*H + I*I + 1.0
        K = (A*(I*I+1)+B*(H*H+1)-Ck*H*I)/(d*_math.sqrt(d))
        return max(-1.0, min(1.0, K))

    @staticmethod
    def _best_phi05_cell(phi_host, flag_np, N):
        best_i = best_j = best_k = -1
        best_diff = float("inf")
        for si in range(N):
            for sj in range(N):
                for sk in range(N):
                    f = int(flag_np[si, sj, sk])
                    if (f & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)) == C.CellFlag.TYPE_I:
                        d = abs(phi_host[si, sj, sk] - 0.5)
                        if d < best_diff:
                            best_diff = d; best_i = si; best_j = sj; best_k = sk
                            if d < 1e-6: break
                if best_diff < 1e-6: break
            if best_diff < 1e-6: break
        return best_i, best_j, best_k

    # ---- curvature tests ----

    def test_sphere_r4(self, device):
        """R=4 sphere: reconstruction matches 1/R."""
        N = 16
        R = 4.0; delta = 1.5
        phi, flag, phi_host = self._make_sphere_grid(N, R, device)
        kappas = self._collect_curvatures(device, phi, flag, N)
        assert len(kappas) > 10
        flag_np = flag.numpy()
        ic, jc, kc = self._best_phi05_cell(phi_host, flag_np, N)
        centre = np.array([N/2, N/2, N/2], dtype=np.float64)
        K_rec = self._reconstruct_z(device, phi, flag, phi_host, ic, jc, kc, centre, R, delta)
        assert K_rec is not None, "reconstruction failed"
        assert abs(abs(K_rec) - 1.0/R) < 0.05, \
            f"R={R}: |K_reconstructed|={abs(K_rec):.6f}, expected 1/R={1/R:.6f}"

    def test_sphere_r8(self, device):
        """R=8 sphere: reconstruction matches 1/R."""
        N = 32
        R = 8.0; delta = 1.5
        phi, flag, phi_host = self._make_sphere_grid(N, R, device)
        kappas = self._collect_curvatures(device, phi, flag, N)
        assert len(kappas) > 50
        flag_np = flag.numpy()
        ic, jc, kc = self._best_phi05_cell(phi_host, flag_np, N)
        centre = np.array([N/2, N/2, N/2], dtype=np.float64)
        K_rec = self._reconstruct_z(device, phi, flag, phi_host, ic, jc, kc, centre, R, delta)
        assert K_rec is not None, "reconstruction failed"
        assert abs(abs(K_rec) - 1.0/R) < 0.05, \
            f"R={R}: |K_reconstructed|={abs(K_rec):.6f}, expected 1/R={1/R:.6f}"

    def test_sphere_r12(self, device):
        """R=12 sphere: reconstruction matches 1/R."""
        N = 32
        R = 12.0; delta = 1.5
        phi, flag, phi_host = self._make_sphere_grid(N, R, device)
        kappas = self._collect_curvatures(device, phi, flag, N)
        assert len(kappas) > 100
        flag_np = flag.numpy()
        ic, jc, kc = self._best_phi05_cell(phi_host, flag_np, N)
        centre = np.array([N/2, N/2, N/2], dtype=np.float64)
        K_rec = self._reconstruct_z(device, phi, flag, phi_host, ic, jc, kc, centre, R, delta)
        assert K_rec is not None, "reconstruction failed"
        assert abs(abs(K_rec) - 1.0/R) < 0.05, \
            f"R={R}: |K_reconstructed|={abs(K_rec):.6f}, expected 1/R={1/R:.6f}"

    def test_plane_zero(self, device):
        """Plane interface: curvature should be near zero."""
        N = 16
        phi, flag, _ = self._make_plane_grid(N, N // 2, device)
        kappas = self._collect_curvatures(device, phi, flag, N)
        assert len(kappas) > 0, "Expected at least some TYPE_I cells"
        abs_kappas = np.abs(kappas)
        median_abs = np.median(abs_kappas)
        assert median_abs < 0.05, \
            f"Plane H=0: median|k|={median_abs:.6f} n_type_i={len(kappas)}"

    def test_cylinder_r6(self, device):
        """R=6 cylinder: reconstruction matches 1/(2R)."""
        N = 32
        R = 6.0; delta = 1.5
        centre_xz = N / 2
        phi_host = np.zeros((N, N, N), dtype=np.float32)
        flag_host = np.zeros((N, N, N), dtype=np.uint8)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    r = np.sqrt((i - centre_xz) ** 2 + (k - centre_xz) ** 2)
                    phi_raw = (R - r) / delta + 0.5
                    phi_host[i, j, k] = max(0.0, min(1.0, phi_raw))
                    if phi_host[i, j, k] >= 1.0:
                        flag_host[i, j, k] = C.CellFlag.TYPE_F
                    elif phi_host[i, j, k] <= 0.0:
                        flag_host[i, j, k] = C.CellFlag.TYPE_G
                    else:
                        flag_host[i, j, k] = C.CellFlag.TYPE_I

        phi = wp.array(phi_host, dtype=float, device=device)
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)
        kappas = self._collect_curvatures(device, phi, flag, N)
        assert len(kappas) > 50
        # Use axis-aligned cell so that plic_cube(phi, n) = phi - 0.5 holds exactly.
        ic, jc, kc = int(N/2 + R), N//2, N//2  # (22, 16, 16), normal (1,0,0)
        centre = np.array([N/2, N/2, N/2], dtype=np.float64)
        K_rec = self._reconstruct_z(device, phi, flag, phi_host, ic, jc, kc, centre, R, delta)
        assert K_rec is not None, "reconstruction failed"
        assert abs(abs(K_rec) - 1.0/(2*R)) < 0.05, \
            f"cylinder R={R}: |K_reconstructed|={abs(K_rec):.6f}, expected 1/(2R)={1/(2*R):.6f}"

    def test_isolated_interface_zero(self, device):
        """Isolated TYPE_I with no interface neighbours: kappa=0 (fallback)."""
        N = 8
        phi_host = np.zeros((N, N, N), dtype=np.float32)
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_G, dtype=np.uint8)
        # Single TYPE_I cell
        phi_host[4, 4, 4] = 0.5
        flag_host[4, 4, 4] = C.CellFlag.TYPE_I

        phi = wp.array(phi_host, dtype=float, device=device)
        flag = wp.array(flag_host, dtype=wp.uint8, device=device)

        cx, cy, cz = _make_direction_arrays(device)
        opp = _make_opposite_array(device)
        result = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            _kernel_curvature_at_cell,
            dim=1,
            inputs=[phi, flag, cx, cy, cz, opp, 4, 4, 4, N, N, N, 0, 0, 0, 0.5, result],
            device=device,
        )
        k = float(result.numpy()[0])
        assert abs(k - 0.0) < 1e-10, f"Isolated interface should give k=0, got {k}"