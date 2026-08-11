# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 unit tests: D3Q7 CMR-MRT gas kernels."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


def _ref_g_eq(rho: float, ux: float, uy: float, uz: float) -> np.ndarray:
    """Host reference for calculate_g_eq (mrUtilFuncGpu3D.h:322-338)."""
    w = np.array([0.25] + [0.125] * 6, dtype=np.float64)
    cx = np.array([0, 1, -1, 0, 0, 0, 0], dtype=np.float64)
    cy = np.array([0, 0, 0, 1, -1, 0, 0], dtype=np.float64)
    cz = np.array([0, 0, 0, 0, 0, 1, -1], dtype=np.float64)
    ux4, uy4, uz4 = ux * 4.0, uy * 4.0, uz * 4.0
    feq = np.zeros(7, dtype=np.float64)
    feq[0] = w[0] * rho
    for i in range(1, 7):
        feq[i] = w[i] * (1.0 + cx[i] * ux4 + cy[i] * uy4 + cz[i] * uz4) * rho
    return feq


def _ref_to_cmr(ux: float, uy: float, uz: float, node_in: np.ndarray) -> np.ndarray:
    """Host reference for mlConvertCmrMoment_d3q7."""
    cx = np.array([0, 1, -1, 0, 0, 0, 0], dtype=np.float64)
    cy = np.array([0, 0, 0, 1, -1, 0, 0], dtype=np.float64)
    cz = np.array([0, 0, 0, 0, 0, 1, -1], dtype=np.float64)
    out = np.zeros(7, dtype=np.float64)
    for k in range(7):
        CX = cx[k] - ux
        CY = cy[k] - uy
        CZ = cz[k] - uz
        ftemp = float(node_in[k])
        out[0] += ftemp
        out[1] += ftemp * CX
        out[2] += ftemp * CY
        out[3] += ftemp * CZ
        out[4] += ftemp * (CX * CX - CY * CY)
        out[5] += ftemp * (CX * CX - CZ * CZ)
        out[6] += ftemp * (CX * CX + CY * CY + CZ * CZ)
    return out


def _ref_from_cmr(U: float, V: float, W: float, node_in: np.ndarray) -> np.ndarray:
    """Host reference for mlConvertCmrF_d3q7."""
    k0, k1, k2, k3, k4, k5, k6 = [float(x) for x in node_in]
    out = np.zeros(7, dtype=np.float64)
    out[0] = -k0 * U * U - 2 * k1 * U - k0 * V * V - 2 * k2 * V - k0 * W * W - 2 * k3 * W + k0 - k6
    out[1] = k1 / 2 + k4 / 6 + k5 / 6 + k6 / 6 + (U * k0) / 2 + U * k1 + (U * U * k0) / 2
    out[2] = k4 / 6 - k1 / 2 + k5 / 6 + k6 / 6 - (U * k0) / 2 + U * k1 + (U * U * k0) / 2
    out[3] = k2 / 2 - k4 / 3 + k5 / 6 + k6 / 6 + (V * k0) / 2 + V * k2 + (V * V * k0) / 2
    out[4] = k5 / 6 - k4 / 3 - k2 / 2 + k6 / 6 - (V * k0) / 2 + V * k2 + (V * V * k0) / 2
    out[5] = k3 / 2 + k4 / 6 - k5 / 3 + k6 / 6 + (W * k0) / 2 + W * k3 + (W * W * k0) / 2
    out[6] = k4 / 6 - k3 / 2 - k5 / 3 + k6 / 6 - (W * k0) / 2 + W * k3 + (W * W * k0) / 2
    return out


class TestGEq:
    def test_g_eq_d3q7_rest(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import _kernel_g_eq

        wp = _wp
        out = wp.zeros(7, dtype=float)
        wp.launch(_kernel_g_eq, dim=1, inputs=[0.5, 0.0, 0.0, 0.0, out])
        wp.synchronize()
        got = out.numpy()
        ref = _ref_g_eq(0.5, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_g_eq_d3q7_moving(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import _kernel_g_eq

        wp = _wp
        out = wp.zeros(7, dtype=float)
        wp.launch(_kernel_g_eq, dim=1, inputs=[0.5, 0.1, 0.0, 0.0, out])
        wp.synchronize()
        got = out.numpy()
        ref = _ref_g_eq(0.5, 0.1, 0.0, 0.0)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestCMRTransforms:
    """Gate R3: CMR transforms match reference formulas."""

    def test_convert_to_central_moment_d3q7(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            _kernel_cmr_to_moment,
        )

        wp = _wp
        pop = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        ux, uy, uz = 0.1, 0.02, -0.03
        node = wp.array(pop, dtype=float)
        wp.launch(_kernel_cmr_to_moment, dim=1, inputs=[ux, uy, uz, node])
        wp.synchronize()
        ref = _ref_to_cmr(ux, uy, uz, pop.astype(np.float64))
        np.testing.assert_allclose(node.numpy(), ref, atol=1e-6, rtol=1e-6)

    def test_convert_from_central_moment_d3q7(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            _kernel_cmr_from_moment,
        )

        wp = _wp
        mom = np.array([1.0, 0.05, -0.02, 0.01, 0.03, -0.01, 0.2], dtype=np.float32)
        ux, uy, uz = 0.1, 0.0, 0.0
        node = wp.array(mom, dtype=float)
        wp.launch(_kernel_cmr_from_moment, dim=1, inputs=[ux, uy, uz, node])
        wp.synchronize()
        ref = _ref_from_cmr(ux, uy, uz, mom.astype(np.float64))
        np.testing.assert_allclose(node.numpy(), ref, atol=1e-6, rtol=1e-6)

    def test_cmr_mrt_roundtrip(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            _kernel_cmr_roundtrip,
        )

        wp = _wp
        rng = np.random.default_rng(42)
        pop = rng.random(7).astype(np.float32) * 0.2 + 0.05
        ux, uy, uz = 0.05, -0.02, 0.01
        node = wp.array(pop.copy(), dtype=float)
        wp.launch(_kernel_cmr_roundtrip, dim=1, inputs=[ux, uy, uz, node])
        wp.synchronize()
        np.testing.assert_allclose(node.numpy(), pop, atol=1e-5, rtol=1e-5)


class TestGasKernels:
    def test_g_reconstruction_henry_law(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            g_reconstruction_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 8
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)
        stride = N * N * N

        flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
        # Center interface with gas neighbor +x and fluid -x
        ci = cj = ck = N // 2
        flag[ci, cj, ck] = C.TYPE_I
        flag[ci + 1, cj, ck] = C.TYPE_G
        flag[:, :, 0] = C.TYPE_S
        flag[:, :, -1] = C.TYPE_S

        tag = np.full((N, N, N), -1, dtype=np.int32)
        tag[ci, cj, ck] = 1
        tag[ci + 1, cj, ck] = 1

        g_mom = np.zeros(7 * stride, dtype=np.float32)
        # Uniform dissolved gas density 0.01 at rest weights
        w = np.array([0.25] + [0.125] * 6, dtype=np.float32)
        for di in range(7):
            g_mom[di * stride : (di + 1) * stride] = w[di] * 0.01

        f_mom = np.zeros(10 * stride, dtype=np.float32)
        f_mom[0 * stride :] = 1.0

        bubble_rho = np.zeros(model.max_bubbles, dtype=np.float64)
        bubble_rho[0] = 1.0

        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32))
        wp.copy(state.g_mom, wp.array(g_mom, dtype=float))
        wp.copy(state.f_mom, wp.array(f_mom, dtype=float))
        wp.copy(state.bubble_rho, wp.array(bubble_rho, dtype=wp.float64))
        state.delta_g.zero_()

        cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32)
        cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32)
        cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32)
        opp = wp.array(np.array(C.OPPOSITE, dtype=np.int32), dtype=wp.int32)

        wp.launch(
            g_reconstruction_kernel,
            dim=(N, N, N),
            inputs=[
                state.g_mom,
                state.f_mom,
                state.flag,
                state.tag_matrix,
                state.bubble_rho,
                state.delta_g,
                cx, cy, cz, opp,
                float(C.HENRY_CONSTANT),
                N, N, N, stride,
            ],
        )
        wp.synchronize()

        # Pull neighbour for di uses (i - cx[di]); TYPE_G at ci+1 is di=2 (cx=-1).
        idx = ci * N * N + cj * N + ck
        g_after = state.g_mom.numpy()
        c_sat = C.HENRY_CONSTANT / 4.0 * 1.0
        assert c_sat == pytest.approx(2.5e-4)
        assert abs(float(g_after[2 * stride + idx]) - float(w[2] * 0.01)) > 1e-8

    def test_g_stream_collide_relaxation(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            g_stream_collide_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 8
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)
        stride = N * N * N

        # All-fluid domain (no walls) — isolates CMR collision from BC effects
        flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
        g_mom = np.zeros(7 * stride, dtype=np.float32)
        w = np.array([0.25] + [0.125] * 6, dtype=np.float32)
        for di in range(7):
            g_mom[di * stride : (di + 1) * stride] = w[di] * 0.01
        # Mild non-equilibrium bump at centre
        mid = (N // 2) * N * N + (N // 2) * N + (N // 2)
        g_mom[1 * stride + mid] *= 1.3

        f_mom = np.zeros(10 * stride, dtype=np.float32)
        f_mom[0 * stride :] = 1.0

        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.g_mom, wp.array(g_mom, dtype=float))
        wp.copy(state.f_mom, wp.array(f_mom, dtype=float))
        state.src.zero_()
        state.islet.zero_()

        cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32)
        cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32)
        cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32)

        wp.launch(
            g_stream_collide_kernel,
            dim=(N, N, N),
            inputs=[
                state.g_mom,
                state.g_mom_post,
                state.f_mom,
                state.flag,
                state.src,
                state.c_value,
                state.islet,
                cx, cy, cz,
                N, N, N, stride,
            ],
        )
        wp.synchronize()
        c = state.c_value.numpy()
        g_post = state.g_mom_post.numpy()
        assert np.isfinite(c).all()
        assert np.isfinite(g_post).all()
        assert float(np.mean(c)) == pytest.approx(0.01, rel=0.2)
        assert float(np.max(np.abs(c))) < 0.1


    def test_bubble_volume_g_update(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            bubble_volume_g_update_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 8
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)

        flag = np.full((N, N, N), C.TYPE_G, dtype=np.uint8)
        flag[4, 4, 4] = C.TYPE_I
        tag = np.full((N, N, N), -1, dtype=np.int32)
        tag[4, 4, 4] = 1
        phi = np.zeros((N, N, N), dtype=np.float32)
        phi[4, 4, 4] = 0.5
        delta_g = np.zeros((N, N, N), dtype=np.float32)
        delta_g[4, 4, 4] = 0.01
        init_vol = np.zeros(model.max_bubbles, dtype=np.float64)
        init_vol[0] = 10.0

        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32))
        wp.copy(state.phi, wp.array(phi, dtype=float))
        wp.copy(state.delta_g, wp.array(delta_g, dtype=float))
        wp.copy(state.bubble_init_volume, wp.array(init_vol, dtype=wp.float64))

        wp.launch(
            bubble_volume_g_update_kernel,
            dim=(N, N, N),
            inputs=[
                state.delta_g,
                state.phi,
                state.flag,
                state.tag_matrix,
                state.bubble_init_volume,
                N, N, N,
            ],
        )
        wp.synchronize()

        # init_volume += 0.25 * 0.01 * 0.5 = 0.00125
        assert float(state.bubble_init_volume.numpy()[0]) == pytest.approx(10.00125, rel=1e-9)
        assert float(state.delta_g.numpy()[4, 4, 4]) == 0.0
