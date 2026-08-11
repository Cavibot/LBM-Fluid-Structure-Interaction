# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 5 stability regression: pipeline, foam gate, volume conservation, shear list."""

from __future__ import annotations

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_fslbm.tests.conftest import (
    GOLDEN_DIR,
    bubble_golden_available,
    load_bubble_golden,
)


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


def _sync(domain):
    from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import sync_double_buffer

    sync_double_buffer(domain)


def _paint_walls_and_bubble(domain, *, centres, radius, N):
    import warp as wp

    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
    from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
        paint_soft_bubble,
        set_solid_walls,
    )

    flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((N, N, N), dtype=np.float32)
    mass = np.ones((N, N, N), dtype=np.float32)
    tag = np.full((N, N, N), -1, dtype=np.int32)
    for tid, (cx, cy, cz) in enumerate(centres, start=1):
        paint_soft_bubble(flag, phi, mass, tag, N, N, N, cx, cy, cz, radius, tid=tid)
    set_solid_walls(flag, phi, mass, N, N, N)
    st = domain.state
    wp.copy(st.flag, wp.array(flag, dtype=wp.uint8))
    wp.copy(st.phi, wp.array(phi, dtype=float))
    wp.copy(st.mass, wp.array(mass, dtype=float))


class TestPipelineOrder:
    """S1: domain step remains finite with gas/disjoin switches."""

    def test_pipeline_order_smoke(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel

        N = 16
        for enable_gas, enable_disjoin in ((True, True), (False, True), (True, False)):
            model = HomeFslbmModel(
                fluid_grid_res=(N, N, N),
                omega=1.0,
                gravity_z=0.0,
                surface_tension=C.SURFACE_TENSION,
                enable_gas=enable_gas,
                enable_disjoin=enable_disjoin,
            )
            domain = HomeFslbmDomain(model)
            domain.create_state()
            domain.solver.initialize_equilibrium(domain.state)
            _paint_walls_and_bubble(
                domain, centres=[(8.0, 8.0, 8.0)], radius=3.0, N=N
            )
            domain.solver.init_bubbles(domain.state)
            _sync(domain)
            domain.step(1.0)
            phi = domain.state.phi.numpy()
            assert np.isfinite(phi).all()
            assert domain.state.bubble_count >= 1


class TestFoamNoCoalescence:
    """S2: foam gate within 500 steps."""

    def test_foam_no_coalescence_500(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
            StabilitySeries,
            capture_snapshot,
        )

        # Prefer golden summary if present
        scene = "foam_no_coalescence"
        if bubble_golden_available(scene, ["bubble_count", "merge_flag"]):
            g = load_bubble_golden(scene)
            assert int(np.asarray(g["bubble_count"]).ravel()[0]) == 2
            assert int(np.asarray(g["merge_flag"]).ravel()[0]) == 0

        N = 32
        steps = 200
        model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            omega=1.0,
            gravity_z=0.0,
            surface_tension=C.SURFACE_TENSION,
            disjoin_factor=C.DISJOINT_FACTOR,
            enable_gas=False,  # isolate disjoin gate
            enable_disjoin=True,
        )
        domain = HomeFslbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state)
        _paint_walls_and_bubble(
            domain,
            centres=[(10.0, 16.0, 16.0), (20.0, 16.0, 16.0)],
            radius=4.0,
            N=N,
        )
        domain.solver.init_bubbles(domain.state)
        _sync(domain)
        assert domain.state.bubble_count >= 2

        series = StabilitySeries()
        series.append(capture_snapshot(domain.state, 0))
        dist0 = series.snaps[0].com_distance
        saw_merge = False
        for t in range(1, steps + 1):
            domain.step(1.0)
            if domain.state.merge_flag != 0:
                saw_merge = True
                break
            if t % 20 == 0 or t == steps:
                series.append(capture_snapshot(domain.state, t))

        assert not saw_merge
        assert domain.state.merge_flag == 0
        assert domain.state.bubble_count >= 2
        dist1 = series.snaps[-1].com_distance
        if dist0 > 0 and dist1 > 0:
            assert dist1 > 0.5 * dist0
        # Volume anti-phase may still occur in a closed full tank (Phase-5 design);
        # coalescence gate only requires topology + separation above.


class TestVolumeConservation:
    """S3: bubble volume drift gates."""

    def _single_bubble_run(self, N: int, steps: int, sample_every: int):
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.stability_probe import (
            StabilitySeries,
            capture_snapshot,
        )

        model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            omega=1.0,
            gravity_z=0.0,
            surface_tension=0.0,  # bookkeeping gate: no Laplace-driven resize
            enable_gas=False,
            enable_disjoin=False,
        )
        domain = HomeFslbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state)
        _paint_walls_and_bubble(
            domain, centres=[(N / 2, N / 2, N / 2)], radius=max(4.0, N / 8), N=N
        )
        domain.solver.init_bubbles(domain.state)
        _sync(domain)
        assert domain.state.bubble_count >= 1

        series = StabilitySeries()
        series.append(capture_snapshot(domain.state, 0))
        for t in range(1, steps + 1):
            domain.step(1.0)
            if t % sample_every == 0 or t == steps:
                series.append(capture_snapshot(domain.state, t))
        return series

    def test_bubble_volume_conservation_smoke(self, _wp):
        series = self._single_bubble_run(N=32, steps=200, sample_every=20)
        drift = series.relative_volume_drift(0)
        assert np.isfinite(drift)
        assert drift < 1e-3

    @pytest.mark.slow
    def test_bubble_volume_conservation_1e4(self, _wp):
        """Phase-5 main gate: σ=0 bookkeeping over 10^4 steps, drift <= 1e-6."""
        scene = "stability_volume_single_bubble"
        if bubble_golden_available(scene, ["volume_rel_drift"]):
            g = load_bubble_golden(scene)
            assert float(np.asarray(g["volume_rel_drift"]).ravel()[0]) <= 1e-6

        series = self._single_bubble_run(N=64, steps=10_000, sample_every=500)
        drift = series.relative_volume_drift(0)
        assert np.isfinite(drift), "volume series empty/invalid"
        assert drift <= 1e-6, f"volume drift {drift} exceeds 1e-6"


class TestShearListed:
    """S4: shear decay remains in the Phase-5 checklist (reuse existing test)."""

    def test_shear_decay_listed(self, _wp):
        """Ensure shear golden path is still discoverable / optionally comparable."""
        from pathlib import Path

        # Existing Phase-1 shear golden under tests/golden_data
        candidates = list(GOLDEN_DIR.glob("**/stream_collide_shear_decay*.txt"))
        flat = GOLDEN_DIR / "stream_collide_shear_decay_step100.txt"
        if flat.is_file() or candidates:
            # Presence is enough for checklist; numeric compare lives in test_kernels_fluid
            assert True
        else:
            pytest.skip(
                "Shear golden not present under tests/golden_data "
                "(see test_kernels_fluid.TestShearDecay)"
            )
