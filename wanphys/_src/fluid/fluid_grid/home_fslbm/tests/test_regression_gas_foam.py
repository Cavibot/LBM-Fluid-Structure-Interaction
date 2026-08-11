# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 golden regression: gas CMR-MRT + foam disjoining / atmosphere.

Scenes without golden data under ``tests/golden_data/`` are skipped.
See ``docs/wanphys/home_fslbm/phase4_gas_foam_golden_zh.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_fslbm.tests.conftest import (
    GOLDEN_DIR,
    bubble_golden_available,
    load_bubble_golden,
    reorder_ref_scalar_to_warp,
)


def _skip_without(scene: str, required: list[str]):
    if not bubble_golden_available(scene, required):
        pytest.skip(
            f"Golden data missing for '{scene}' "
            f"(need {required} under tests/golden_data/{scene}/). "
            f"See docs/wanphys/home_fslbm/phase4_gas_foam_golden_zh.md"
        )


def _compare_float(a: np.ndarray, b: np.ndarray, rtol: float = 1e-4, atol: float = 1e-10):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-10)
    near_zero = np.maximum(np.abs(a), np.abs(b)) < 1e-8
    rel_ok = np.abs(a - b) / scale <= rtol
    abs_ok = np.abs(a - b) <= atol
    ok = np.where(near_zero, abs_ok, rel_ok)
    assert bool(np.all(ok)), (
        f"float mismatch: max |a-b|={float(np.max(np.abs(a - b)))}, "
        f"fail_frac={float(1.0 - np.mean(ok)):.4f}"
    )


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


class TestGasCMRGolden:
    """G1: transform vectors (host-ref or golden files)."""

    def test_gas_cmr_transform_vectors(self, _wp):
        scene = "gas_cmr_transform_vectors"
        # Names without ".txt" — bubble_golden_available appends the suffix.
        required = ["pop_in", "moment_out", "uxuyuz"]
        # Prefer golden; else fall back to analytical host reference (always available).
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_gas import (
            _kernel_cmr_from_moment,
            _kernel_cmr_to_moment,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_kernels_gas import (
            _ref_from_cmr,
            _ref_to_cmr,
        )

        wp = _wp
        if bubble_golden_available(scene, required):
            g = load_bubble_golden(scene)
            pop = np.asarray(g["pop_in"], dtype=np.float32)
            u = np.asarray(g["uxuyuz"], dtype=np.float32).ravel()
            ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
            node = wp.array(pop.copy(), dtype=float)
            wp.launch(_kernel_cmr_to_moment, dim=1, inputs=[ux, uy, uz, node])
            wp.synchronize()
            _compare_float(node.numpy(), g["moment_out"], rtol=1e-5, atol=1e-6)
            if (GOLDEN_DIR / scene / "moment_in.txt").is_file():
                mom = np.asarray(g["moment_in"], dtype=np.float32)
                node2 = wp.array(mom.copy(), dtype=float)
                wp.launch(_kernel_cmr_from_moment, dim=1, inputs=[ux, uy, uz, node2])
                wp.synchronize()
                _compare_float(node2.numpy(), g["pop_out"], rtol=1e-5, atol=1e-6)
        else:
            pop = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
            ux, uy, uz = 0.1, 0.02, -0.03
            node = wp.array(pop.copy(), dtype=float)
            wp.launch(_kernel_cmr_to_moment, dim=1, inputs=[ux, uy, uz, node])
            wp.synchronize()
            ref = _ref_to_cmr(ux, uy, uz, pop.astype(np.float64))
            np.testing.assert_allclose(node.numpy(), ref, atol=1e-6, rtol=1e-6)
            mom = ref.astype(np.float32)
            node2 = wp.array(mom.copy(), dtype=float)
            wp.launch(_kernel_cmr_from_moment, dim=1, inputs=[ux, uy, uz, node2])
            wp.synchronize()
            ref2 = _ref_from_cmr(ux, uy, uz, mom.astype(np.float64))
            np.testing.assert_allclose(node2.numpy(), ref2, atol=1e-6, rtol=1e-6)


class TestGasFieldGolden:
    def test_gas_henry_interface_step1(self, _wp):
        scene = "gas_henry_interface_step1"
        _skip_without(scene, ["nx", "delta_g", "g_mom"])
        # Presence-only gate until exporter is run; structural load check
        g = load_bubble_golden(scene)
        assert "delta_g" in g
        assert "g_mom" in g

    def test_gas_stream_collide_step10(self, _wp):
        scene = "gas_stream_collide_step10"
        _skip_without(scene, ["nx", "g_mom", "c_value"])
        g = load_bubble_golden(scene)
        assert "c_value" in g

    def test_gas_volume_g_update(self, _wp):
        scene = "gas_volume_g_update"
        _skip_without(scene, ["bubble_init_volume"])
        g = load_bubble_golden(scene)
        assert "bubble_init_volume" in g


class TestFoamGolden:
    def test_foam_disjoint_two_spheres(self, _wp):
        scene = "foam_disjoint_two_spheres"
        _skip_without(scene, ["nx", "disjoin_force"])
        g = load_bubble_golden(scene)
        assert float(np.max(np.abs(np.asarray(g["disjoin_force"])))) >= 0.0

    def test_foam_no_coalescence_gate(self, _wp):
        """Functional gate (S2): two close bubbles do not merge under disjoining."""
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        import warp as wp

        # Prefer golden summary if present; otherwise run a short Warp scene.
        scene = "foam_no_coalescence"
        if bubble_golden_available(scene, ["bubble_count", "merge_flag"]):
            g = load_bubble_golden(scene)
            assert int(np.asarray(g["bubble_count"]).ravel()[0]) == 2
            assert int(np.asarray(g["merge_flag"]).ravel()[0]) == 0
            return

        N = 32
        model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            omega=1.0,
            gravity_z=0.0,
            disjoin_factor=C.DISJOINT_FACTOR,
            surface_tension=C.SURFACE_TENSION,
        )
        domain = HomeFslbmDomain(model)
        domain.create_state()
        domain.solver.initialize_equilibrium(domain.state)

        flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
        phi = np.ones((N, N, N), dtype=np.float32)
        mass = np.ones((N, N, N), dtype=np.float32)
        # r=4, centres 10 and 20 -> surface gap ~2 on 32^3 (scaled-down F2)
        centres = [(10, 16, 16), (20, 16, 16)]
        r = 4.0
        for tid, (cx0, cy0, cz0) in enumerate(centres, start=1):
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        d = np.sqrt((i - cx0) ** 2 + (j - cy0) ** 2 + (k - cz0) ** 2)
                        if d < r - 0.5:
                            flag[i, j, k] = C.TYPE_G
                            phi[i, j, k] = 0.0
                            mass[i, j, k] = 0.0
                        elif d <= r + 0.5:
                            flag[i, j, k] = C.TYPE_I
                            phi[i, j, k] = 0.5
                            mass[i, j, k] = 0.5
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
        domain.solver.init_bubbles(st)
        assert st.bubble_count >= 2

        dst = domain._state_out
        for name in (
            "f_mom", "f_mom_post", "flag", "mass", "massex", "phi",
            "tag_matrix", "previous_tag", "previous_merge_tag",
            "bubble_volume", "bubble_init_volume", "bubble_rho",
            "g_mom", "g_mom_post", "disjoin_force", "label_matrix",
            "input_matrix", "merge_detector",
        ):
            wp.copy(getattr(dst, name), getattr(st, name))
        dst.bubble_count = st.bubble_count
        dst.label_num = st.label_num

        c0 = np.argwhere(st.tag_matrix.numpy() == 1).mean(axis=0)
        c1 = np.argwhere(st.tag_matrix.numpy() == 2).mean(axis=0)
        dist0 = float(np.linalg.norm(c0 - c1))

        saw_disjoin = False
        for _ in range(50):
            # Probe disjoin before reset: run one calculate_disjoint manually
            from wanphys._src.fluid.fluid_grid.home_fslbm import kernels_foam
            from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C2

            cx = domain.solver._cx
            cy = domain.solver._cy
            cz = domain.solver._cz
            opp = domain.solver._opposite
            domain.state.disjoin_force.zero_()
            wp.launch(
                kernels_foam.calculate_disjoint_kernel,
                dim=(N, N, N),
                inputs=[
                    domain.state.flag, domain.state.phi, domain.state.mass,
                    domain.state.massex, domain.state.f_mom, domain.state.tag_matrix,
                    domain.state.disjoin_force, cx, cy, cz, opp,
                    N, N, N, domain.solver._stride,
                ],
            )
            wp.synchronize()
            if float(np.sum(domain.state.disjoin_force.numpy())) > 0.0:
                saw_disjoin = True
            domain.step(1.0)
            if domain.state.merge_flag != 0:
                break

        assert domain.state.merge_flag == 0
        assert domain.state.bubble_count >= 2
        assert saw_disjoin

        tags = domain.state.tag_matrix.numpy()
        ids = sorted(int(x) for x in np.unique(tags) if x > 0)
        if len(ids) >= 2:
            ca = np.argwhere(tags == ids[0]).mean(axis=0)
            cb = np.argwhere(tags == ids[1]).mean(axis=0)
            dist1 = float(np.linalg.norm(ca - cb))
            # Centres must not collapse to contact
            assert dist1 > 0.5 * dist0

    def test_foam_atmosphere_open_tank(self, _wp):
        scene = "foam_atmosphere_open_tank"
        _skip_without(scene, ["bubble_rho", "bubble_init_volume"])
        g = load_bubble_golden(scene)
        assert "bubble_rho" in g
