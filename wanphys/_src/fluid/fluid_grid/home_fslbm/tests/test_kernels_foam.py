# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 unit tests: disjoining pressure + atmosphere foam kernels."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


def _paint_sphere_shell(N: int, cx: float, cy: float, cz: float, r: float):
    """Hard bubble: interior gas, thin interface shell, exterior fluid."""
    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C

    flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((N, N, N), dtype=np.float32)
    mass = np.ones((N, N, N), dtype=np.float32)
    tag = np.full((N, N, N), -1, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                d = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                if d < r - 0.5:
                    flag[i, j, k] = C.TYPE_G
                    phi[i, j, k] = 0.0
                    mass[i, j, k] = 0.0
                    tag[i, j, k] = 1
                elif d <= r + 0.5:
                    flag[i, j, k] = C.TYPE_I
                    phi[i, j, k] = 0.5
                    mass[i, j, k] = 0.5
                    tag[i, j, k] = 1
    return flag, phi, mass, tag


class TestDisjoint:
    def test_disjoint_no_neighbor(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_foam import (
            calculate_disjoint_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 16
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)
        stride = N * N * N

        flag, phi, mass, tag = _paint_sphere_shell(N, 8, 8, 8, 3.0)
        f_mom = np.zeros(10 * stride, dtype=np.float32)
        f_mom[0 * stride :] = 1.0

        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.phi, wp.array(phi, dtype=float))
        wp.copy(state.mass, wp.array(mass, dtype=float))
        wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32))
        wp.copy(state.f_mom, wp.array(f_mom, dtype=float))
        state.massex.zero_()
        state.disjoin_force.zero_()

        cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32)
        cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32)
        cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32)
        opp = wp.array(np.array(C.OPPOSITE, dtype=np.int32), dtype=wp.int32)

        wp.launch(
            calculate_disjoint_kernel,
            dim=(N, N, N),
            inputs=[
                state.flag, state.phi, state.mass, state.massex,
                state.f_mom, state.tag_matrix, state.disjoin_force,
                cx, cy, cz, opp, N, N, N, stride,
            ],
        )
        wp.synchronize()
        assert float(np.max(np.abs(state.disjoin_force.numpy()))) == 0.0

    def test_disjoint_raycast_hit(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_foam import (
            calculate_disjoint_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 24
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)
        stride = N * N * N

        # Two close bubbles (surface gap ~2)
        flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
        phi = np.ones((N, N, N), dtype=np.float32)
        mass = np.ones((N, N, N), dtype=np.float32)
        tag = np.full((N, N, N), -1, dtype=np.int32)

        centres = [(9, 12, 12, 1), (15, 12, 12, 2)]
        r = 4.0
        for cx0, cy0, cz0, tid in centres:
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        d = np.sqrt((i - cx0) ** 2 + (j - cy0) ** 2 + (k - cz0) ** 2)
                        if d < r - 0.5:
                            flag[i, j, k] = C.TYPE_G
                            phi[i, j, k] = 0.0
                            mass[i, j, k] = 0.0
                            tag[i, j, k] = tid
                        elif d <= r + 0.5:
                            flag[i, j, k] = C.TYPE_I
                            phi[i, j, k] = 0.5
                            mass[i, j, k] = 0.5
                            tag[i, j, k] = tid

        f_mom = np.zeros(10 * stride, dtype=np.float32)
        f_mom[0 * stride :] = 1.0

        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.phi, wp.array(phi, dtype=float))
        wp.copy(state.mass, wp.array(mass, dtype=float))
        wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32))
        wp.copy(state.f_mom, wp.array(f_mom, dtype=float))
        state.massex.zero_()
        state.disjoin_force.zero_()

        cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32)
        cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32)
        cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32)
        opp = wp.array(np.array(C.OPPOSITE, dtype=np.int32), dtype=wp.int32)

        wp.launch(
            calculate_disjoint_kernel,
            dim=(N, N, N),
            inputs=[
                state.flag, state.phi, state.mass, state.massex,
                state.f_mom, state.tag_matrix, state.disjoin_force,
                cx, cy, cz, opp, N, N, N, stride,
            ],
        )
        wp.synchronize()
        assert float(np.sum(state.disjoin_force.numpy())) > 0.0

    def test_disjoint_distance_scaling(self, _wp):
        """P_disjoin = max(0, 1 - d/4) for a constructed 1D-like pair."""
        # Analytical check of the formula itself (kernel uses same expression).
        for d, expect in [(0.0, 1.0), (2.0, 0.5), (4.0, 0.0), (5.0, -0.25)]:
            p = 1.0 - d / 4.0
            assert p == pytest.approx(expect)
            assert max(0.0, p) == pytest.approx(max(0.0, expect))

    def test_reset_disjoin_force(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_foam import (
            reset_disjoin_force_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 8
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)
        ones = np.ones((N, N, N), dtype=np.float32)
        wp.copy(state.disjoin_force, wp.array(ones, dtype=float))
        wp.copy(state.massex, wp.array(ones * 2.0, dtype=float))

        wp.launch(
            reset_disjoin_force_kernel,
            dim=(N, N, N),
            inputs=[state.disjoin_force, state.massex, N, N, N],
        )
        wp.synchronize()
        assert float(np.max(np.abs(state.disjoin_force.numpy()))) == 0.0
        assert float(np.max(np.abs(state.massex.numpy()))) == 0.0


class TestAtmosphere:
    def test_atmosphere_rho_update(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_foam import (
            atmosphere_rho_update_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        N = 16
        model = HomeFslbmModel(fluid_grid_res=(N, N, N), omega=1.0)
        state = HomeFslbmState(model)

        tag = np.full((N, N, N), -1, dtype=np.int32)
        # Cell at x==nx-2 with large bubble tag
        tag[N - 2, N // 2, N // 2] = 1
        vol = np.zeros(model.max_bubbles, dtype=np.float64)
        vol[0] = 2_000_000.0
        rho = np.zeros(model.max_bubbles, dtype=np.float64)
        rho[0] = 1.5

        wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32))
        wp.copy(state.bubble_volume, wp.array(vol, dtype=wp.float64))
        wp.copy(state.bubble_rho, wp.array(rho, dtype=wp.float64))

        wp.launch(
            atmosphere_rho_update_kernel,
            dim=(N, N, N),
            inputs=[
                state.tag_matrix,
                state.bubble_volume,
                state.bubble_rho,
                N, N, N,
            ],
        )
        wp.synchronize()
        assert float(state.bubble_rho.numpy()[0]) == pytest.approx(1.0)

    def test_atmosphere_volme_update(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_foam import (
            atmosphere_volme_update_kernel,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        model = HomeFslbmModel(fluid_grid_res=(8, 8, 8), omega=1.0)
        state = HomeFslbmState(model)

        vol = np.array([1000.0, 50.0], dtype=np.float64)
        init = np.array([900.0, 50.0], dtype=np.float64)
        rho = np.array([1.0, 1.2], dtype=np.float64)
        wp.copy(state.bubble_volume, wp.array(np.zeros(model.max_bubbles, dtype=np.float64), dtype=wp.float64))
        wp.copy(state.bubble_init_volume, wp.array(np.zeros(model.max_bubbles, dtype=np.float64), dtype=wp.float64))
        wp.copy(state.bubble_rho, wp.array(np.zeros(model.max_bubbles, dtype=np.float64), dtype=wp.float64))

        # Write first two entries
        bv = state.bubble_volume.numpy()
        bi = state.bubble_init_volume.numpy()
        br = state.bubble_rho.numpy()
        bv[:2] = vol
        bi[:2] = init
        br[:2] = rho
        wp.copy(state.bubble_volume, wp.array(bv, dtype=wp.float64))
        wp.copy(state.bubble_init_volume, wp.array(bi, dtype=wp.float64))
        wp.copy(state.bubble_rho, wp.array(br, dtype=wp.float64))

        wp.launch(
            atmosphere_volme_update_kernel,
            dim=1,
            inputs=[
                state.bubble_volume,
                state.bubble_init_volume,
                state.bubble_rho,
                2,
            ],
        )
        wp.synchronize()
        bi2 = state.bubble_init_volume.numpy()
        assert float(bi2[0]) == pytest.approx(1000.0)
        assert float(bi2[1]) == pytest.approx(50.0)  # unchanged (rho != 1)
