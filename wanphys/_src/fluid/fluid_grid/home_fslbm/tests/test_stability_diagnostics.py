# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase-5 diagnostic ablation: locate foam volume-oscillation drive terms."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


def _build_two_bubble_domain(*, enable_gas: bool, enable_disjoin: bool, N: int = 32):
    import warp as wp

    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
    from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
    from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
    from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
        paint_soft_bubble,
        set_solid_walls,
        sync_double_buffer,
    )

    model = HomeFslbmModel(
        fluid_grid_res=(N, N, N),
        omega=1.0,
        gravity_z=0.0,
        surface_tension=C.SURFACE_TENSION,
        disjoin_factor=C.DISJOINT_FACTOR,
        enable_gas=enable_gas,
        enable_disjoin=enable_disjoin,
    )
    domain = HomeFslbmDomain(model)
    domain.create_state()
    domain.solver.initialize_equilibrium(domain.state)

    flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((N, N, N), dtype=np.float32)
    mass = np.ones((N, N, N), dtype=np.float32)
    tag = np.full((N, N, N), -1, dtype=np.int32)
    # gap ~2 on 32^3
    paint_soft_bubble(flag, phi, mass, tag, N, N, N, 10.0, 16.0, 16.0, 4.0, tid=1)
    paint_soft_bubble(flag, phi, mass, tag, N, N, N, 20.0, 16.0, 16.0, 4.0, tid=2)
    set_solid_walls(flag, phi, mass, N, N, N)

    st = domain.state
    wp.copy(st.flag, wp.array(flag, dtype=wp.uint8))
    wp.copy(st.phi, wp.array(phi, dtype=float))
    wp.copy(st.mass, wp.array(mass, dtype=float))
    domain.solver.init_bubbles(st)
    sync_double_buffer(domain)
    return domain


def _build_single_bubble_domain(*, enable_gas: bool, N: int = 32):
    import warp as wp

    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
    from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
    from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
    from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
        paint_soft_bubble,
        set_solid_walls,
        sync_double_buffer,
    )

    model = HomeFslbmModel(
        fluid_grid_res=(N, N, N),
        omega=1.0,
        gravity_z=0.0,
        surface_tension=C.SURFACE_TENSION,
        enable_gas=enable_gas,
        enable_disjoin=False,
    )
    domain = HomeFslbmDomain(model)
    domain.create_state()
    domain.solver.initialize_equilibrium(domain.state)

    flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((N, N, N), dtype=np.float32)
    mass = np.ones((N, N, N), dtype=np.float32)
    tag = np.full((N, N, N), -1, dtype=np.int32)
    paint_soft_bubble(flag, phi, mass, tag, N, N, N, 16.0, 16.0, 16.0, 6.0, tid=1)
    set_solid_walls(flag, phi, mass, N, N, N)

    st = domain.state
    wp.copy(st.flag, wp.array(flag, dtype=wp.uint8))
    wp.copy(st.phi, wp.array(phi, dtype=float))
    wp.copy(st.mass, wp.array(mass, dtype=float))
    domain.solver.init_bubbles(st)
    sync_double_buffer(domain)
    return domain


def _run_series(domain, steps: int):
    from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
        StabilitySeries,
        capture_snapshot,
    )

    series = StabilitySeries()
    series.append(capture_snapshot(domain.state, 0))
    for t in range(1, steps + 1):
        domain.step(1.0)
        if t % 10 == 0 or t == steps:
            series.append(capture_snapshot(domain.state, t))
    return series


class TestStabilityAblation:
    def test_probe_records_volumes(self, _wp):
        domain = _build_single_bubble_domain(enable_gas=False, N=24)
        series = _run_series(domain, 20)
        assert len(series.snaps) >= 2
        assert series.snaps[0].bubble_count >= 1
        assert np.isfinite(series.snaps[0].volumes).all()

    def test_single_bubble_no_gas_bookkeeping_bounded(self, _wp):
        """σ=0 closed single bubble: geometric volume bookkeeping stays tight."""
        import warp as wp

        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
            paint_soft_bubble,
            set_solid_walls,
            sync_double_buffer,
        )

        N = 32
        model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            omega=1.0,
            gravity_z=0.0,
            surface_tension=0.0,
            enable_gas=False,
            enable_disjoin=False,
        )
        domain = HomeFslbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state)
        flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
        phi = np.ones((N, N, N), dtype=np.float32)
        mass = np.ones((N, N, N), dtype=np.float32)
        tag = np.full((N, N, N), -1, dtype=np.int32)
        paint_soft_bubble(flag, phi, mass, tag, N, N, N, 16.0, 16.0, 16.0, 6.0, tid=1)
        set_solid_walls(flag, phi, mass, N, N, N)
        st = domain.state
        wp.copy(st.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(st.phi, wp.array(phi, dtype=float))
        wp.copy(st.mass, wp.array(mass, dtype=float))
        domain.solver.init_bubbles(st)
        sync_double_buffer(domain)
        series = _run_series(domain, 100)
        drift = series.relative_volume_drift(0)
        assert np.isfinite(drift)
        assert drift < 1e-3

    def test_two_bubble_ablation_gas_off_keeps_init_early(self, _wp):
        """First 20 steps: gas-off keeps init_volume fixed; gas-on may nudge it."""
        d_off = _build_two_bubble_domain(enable_gas=False, enable_disjoin=True, N=32)
        d_on = _build_two_bubble_domain(enable_gas=True, enable_disjoin=True, N=32)

        def early_init_delta(domain):
            from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
                capture_snapshot,
            )

            s0 = capture_snapshot(domain.state, 0)
            for _ in range(20):
                domain.step(1.0)
            s1 = capture_snapshot(domain.state, 20)
            n = min(len(s0.init_volumes), len(s1.init_volumes), 2)
            if n == 0:
                return 0.0
            return float(np.max(np.abs(s1.init_volumes[:n] - s0.init_volumes[:n])))

        d_off_delta = early_init_delta(d_off)
        d_on_delta = early_init_delta(d_on)
        assert d_off_delta <= 1e-9
        assert d_on_delta >= d_off_delta - 1e-12

    def test_disjoin_off_zero_force_probe(self, _wp):
        domain = _build_two_bubble_domain(enable_gas=False, enable_disjoin=False, N=32)
        domain.step(1.0)
        assert float(np.abs(domain.state.disjoin_force.numpy()).sum()) == 0.0
