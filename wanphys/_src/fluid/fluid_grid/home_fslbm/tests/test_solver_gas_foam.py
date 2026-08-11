# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 solver integration: gas handle + foam disjoint/reset wiring."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


def _uniform_fluid_domain(N: int = 16):
    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
    from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
    from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel

    model = HomeFslbmModel(
        fluid_grid_res=(N, N, N),
        omega=1.0,
        gravity_z=0.0,
        surface_tension=C.SURFACE_TENSION,
        disjoin_factor=C.DISJOINT_FACTOR,
    )
    domain = HomeFslbmDomain(model)
    domain.create_state()
    domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

    flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
    for arr, v in ((flag, C.TYPE_S),):
        arr[0, :, :] = v
        arr[-1, :, :] = v
        arr[:, 0, :] = v
        arr[:, -1, :] = v
        arr[:, :, 0] = v
        arr[:, :, -1] = v
    import warp as wp

    st = domain.state
    wp.copy(st.flag, wp.array(flag, dtype=wp.uint8))
    wp.copy(st.phi, wp.array(np.ones((N, N, N), dtype=np.float32), dtype=float))
    wp.copy(st.mass, wp.array(np.ones((N, N, N), dtype=np.float32), dtype=float))
    tag = np.full((N, N, N), -1, dtype=np.int32)
    wp.copy(st.tag_matrix, wp.array(tag, dtype=wp.int32))

    # Seed non-equilibrium dissolved gas
    stride = N * N * N
    g = np.zeros(7 * stride, dtype=np.float32)
    w = [0.25] + [0.125] * 6
    for di, wi in enumerate(w):
        g[di * stride : (di + 1) * stride] = wi * 0.01
    g[1 * stride + (N // 2) * N * N + (N // 2) * N + (N // 2)] *= 1.5
    wp.copy(st.g_mom, wp.array(g, dtype=float))
    wp.copy(st.g_mom_post, st.g_mom)

    # Sync double buffer
    dst = domain._state_out
    for name in (
        "f_mom", "f_mom_post", "flag", "mass", "massex", "phi",
        "g_mom", "g_mom_post", "tag_matrix", "c_value", "src", "delta_g",
        "disjoin_force", "islet",
    ):
        wp.copy(getattr(dst, name), getattr(st, name))
    return domain


class TestSolverGasFoam:
    def test_gas_handle_updates_g_mom(self, _wp):
        domain = _uniform_fluid_domain(16)
        g0 = domain.state.g_mom.numpy().copy()
        domain.step(1.0)
        g1 = domain.state.g_mom.numpy()
        assert np.isfinite(g1).all()
        # CMR collision should change at least some entries
        assert float(np.max(np.abs(g1 - g0))) > 0.0
        c = domain.state.c_value.numpy()
        assert np.isfinite(c).all()

    def test_disjoint_cleared_after_step(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        import warp as wp

        N = 24
        model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            omega=1.0,
            gravity_z=0.0,
            disjoin_factor=C.DISJOINT_FACTOR,
        )
        domain = HomeFslbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state)

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
        for a in (flag,):
            a[0, :, :] = C.TYPE_S
            a[-1, :, :] = C.TYPE_S
            a[:, 0, :] = C.TYPE_S
            a[:, -1, :] = C.TYPE_S
            a[:, :, 0] = C.TYPE_S
            a[:, :, -1] = C.TYPE_S

        st = domain.state
        wp.copy(st.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(st.phi, wp.array(phi, dtype=float))
        wp.copy(st.mass, wp.array(mass, dtype=float))
        wp.copy(st.tag_matrix, wp.array(tag, dtype=wp.int32))
        st.bubble_count = 2
        vol = np.zeros(model.max_bubbles, dtype=np.float64)
        vol[0] = vol[1] = 200.0
        rho = np.ones(model.max_bubbles, dtype=np.float64)
        wp.copy(st.bubble_volume, wp.array(vol, dtype=wp.float64))
        wp.copy(st.bubble_init_volume, wp.array(vol.copy(), dtype=wp.float64))
        wp.copy(st.bubble_rho, wp.array(rho, dtype=wp.float64))

        dst = domain._state_out
        for name in (
            "f_mom", "f_mom_post", "flag", "mass", "massex", "phi",
            "tag_matrix", "bubble_volume", "bubble_init_volume", "bubble_rho",
            "g_mom", "g_mom_post", "disjoin_force",
        ):
            wp.copy(getattr(dst, name), getattr(st, name))
        dst.bubble_count = 2

        domain.step(1.0)
        # After reset_disjoin_force, force must be zero
        assert float(np.max(np.abs(domain.state.disjoin_force.numpy()))) == 0.0

    def test_f_mom_swap_not_g_post_overwrite(self, _wp):
        """surface_2 may write g_mom; end-of-step must not restore stale g_mom_post."""
        domain = _uniform_fluid_domain(12)
        # Poison g_mom_post with a sentinel distinct from g_mom
        import warp as wp

        poison = np.full_like(domain.state.g_mom.numpy(), 7.777)
        wp.copy(domain.state.g_mom_post, wp.array(poison, dtype=float))
        wp.copy(domain._state_out.g_mom_post, wp.array(poison, dtype=float))
        domain.step(1.0)
        g = domain.state.g_mom.numpy()
        # Must not be all 7.777 (would mean final swap overwrote from poisoned post)
        assert not np.allclose(g, 7.777)
