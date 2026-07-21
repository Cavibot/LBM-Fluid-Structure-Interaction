# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HOME-FSLBM free-surface kernels (Phase 2).

Covers:
  - ``calculate_phi`` — VOF fill-level helper
  - ``plic_cube`` / ``plic_cube_reduced`` — unit-cube plane intersection
  - ``surface_1_kernel`` — interface marker propagation
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
)
from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_surface import (
    surface_1_kernel,
    surface_3_kernel,
)


def _make_direction_arrays(device):
    """Create D3Q27 cx, cy, cz Warp arrays (reusable across surface tests)."""
    cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32, device=device)
    cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32, device=device)
    cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32, device=device)
    return cx, cy, cz


# ============================================================================
# Fixtures (matching test_kernels_fluid.py)
# ============================================================================


@pytest.fixture(scope="module")
def device():
    try:
        return wp.get_device("cuda:0")
    except Exception:
        return wp.get_device("cpu")


# ============================================================================
# Test-only fill kernels (not testing any @wp.func — just utilities)
# ============================================================================


@wp.kernel
def _fill_uint8_kernel(arr: wp.array3d(dtype=wp.uint8), val: wp.uint8):
    i, j, k = wp.tid()
    arr[i, j, k] = val


# ============================================================================
# Test classes
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


class TestPlicCube:
    """PLIC unit-cube plane intersection (mrUtilFuncGpu3D.h:104-149)."""

    def test_axis_aligned_half(self, device):
        """V=0.5, n=(1,0,0) -> d=0.0 (plane bisects cube)."""
        N = 1
        V0 = wp.array(np.array([0.5], dtype=np.float32), dtype=float, device=device)
        nx_v = wp.array(np.array([1.0], dtype=np.float32), dtype=float, device=device)
        ny_v = wp.array(np.array([0.0], dtype=np.float32), dtype=float, device=device)
        nz_v = wp.array(np.array([0.0], dtype=np.float32), dtype=float, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_plic_cube, dim=N, inputs=[V0, nx_v, ny_v, nz_v, result], device=device)
        got = result.numpy()[0]
        assert abs(got - 0.0) < 1e-5, f"V=0.5 n=(1,0,0) d=0.0, got {got}"

    def test_axis_aligned_quarter(self, device):
        """V=0.1, n=(1,0,0) -> d=-0.4."""
        N = 1
        V0 = wp.array(np.array([0.1], dtype=np.float32), dtype=float, device=device)
        nx_v = wp.array(np.array([1.0], dtype=np.float32), dtype=float, device=device)
        ny_v = wp.array(np.array([0.0], dtype=np.float32), dtype=float, device=device)
        nz_v = wp.array(np.array([0.0], dtype=np.float32), dtype=float, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_plic_cube, dim=N, inputs=[V0, nx_v, ny_v, nz_v, result], device=device)
        got = result.numpy()[0]
        expected = -0.4
        assert abs(got - expected) < 1e-5, f"V=0.1 n=(1,0,0) d=-0.4, got {got}"

    def test_diagonal(self, device):
        """V=0.5, n=(1,1,1) normalized -> d should be roughly 0."""
        N = 1
        s3 = 1.0 / np.sqrt(3.0)
        V0 = wp.array(np.array([0.5], dtype=np.float32), dtype=float, device=device)
        nx_v = wp.array(np.array([s3], dtype=np.float32), dtype=float, device=device)
        ny_v = wp.array(np.array([s3], dtype=np.float32), dtype=float, device=device)
        nz_v = wp.array(np.array([s3], dtype=np.float32), dtype=float, device=device)
        result = wp.zeros(N, dtype=float, device=device)
        wp.launch(_kernel_plic_cube, dim=N, inputs=[V0, nx_v, ny_v, nz_v, result], device=device)
        got = result.numpy()[0]
        assert abs(got) < 0.05, f"V=0.5 diagonal d near 0, got {got}"


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


class TestSurface3:
    """VOF mass redistribution kernel."""

    def test_type_f_mass_equals_rho(self, device):
        """TYPE_F cell mass should be set to rho, excess→massex."""
        N = 4
        state = _make_surface3_state(N, device)
        cx, cy, cz = _make_direction_arrays(device)

        idx = 1 * N * N + 1 * N + 1
        f_np = state["f_mom_post"].numpy(); f_np[idx] = 1.0
        wp.copy(state["f_mom_post"], wp.array(f_np, dtype=float, device=device))

        flag_host = state["flag"].numpy()
        # Fill all cells as TYPE_F so excess mass can distribute to neighbours
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


# ============================================================================
# Helpers
# ============================================================================


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