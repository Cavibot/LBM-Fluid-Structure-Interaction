# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 unit tests: YACCLAB CCL + bubble tracking kernels."""

from __future__ import annotations

import numpy as np
import pytest


def _labels_topology_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """True if label fields match up to a permutation of positive IDs."""
    a = a.astype(np.int32).ravel()
    b = b.astype(np.int32).ravel()
    if a.shape != b.shape:
        return False
    # Background must agree
    if not np.array_equal(a == 0, b == 0):
        return False
    a_ids = sorted(int(x) for x in np.unique(a) if x > 0)
    b_ids = sorted(int(x) for x in np.unique(b) if x > 0)
    if len(a_ids) != len(b_ids):
        return False
    if len(a_ids) == 0:
        return True

    def partition(arr, ids):
        parts = []
        for lab in ids:
            parts.append(frozenset(np.flatnonzero(arr == lab).tolist()))
        return frozenset(parts)

    return partition(a, a_ids) == partition(b, b_ids)


def _make_three_spheres_input(n: int = 16, radius: int = 3) -> np.ndarray:
    """16³ binary image with 3 non-overlapping spheres (foreground=255)."""
    img = np.zeros((n, n, n), dtype=np.uint8)
    centres = [(4, 4, 4), (12, 4, 4), (8, 12, 12)]
    for cx, cy, cz in centres:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= radius**2:
                        img[i, j, k] = 255
    return img


def _scipy_label(img: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled, _ = ndimage.label(img > 0, structure=structure)
    return labeled.astype(np.int32)


@pytest.fixture(scope="module")
def _wp():
    import warp as wp

    wp.init()
    return wp


class TestCCL:
    """Gate A: CCL topology on 16³ / 3 spheres."""

    def test_ccl_golden_16cube_3_spheres(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            connected_component_labeling,
        )

        wp = _wp
        img_np = _make_three_spheres_input(16, 3)
        ref = _scipy_label(img_np)

        input_m = wp.array(img_np, dtype=wp.uint8)
        label_m = wp.zeros((16, 16, 16), dtype=wp.int32)
        nlab = connected_component_labeling(input_m, label_m)
        out = label_m.numpy()

        assert nlab == 3, f"expected 3 components, got {nlab}"
        assert _labels_topology_equal(out, ref), (
            "CCL topology mismatch vs 26-connected reference"
        )

    def test_ccl_deterministic(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            connected_component_labeling,
        )

        wp = _wp
        img_np = _make_three_spheres_input(16, 3)
        input_m = wp.array(img_np, dtype=wp.uint8)
        results = []
        for _ in range(5):
            label_m = wp.zeros((16, 16, 16), dtype=wp.int32)
            connected_component_labeling(input_m, label_m)
            results.append(label_m.numpy().copy())
        for r in results[1:]:
            assert np.array_equal(results[0], r), "CCL not bit-identical across runs"


class TestConvertAndInitTag:
    def test_convert_flag_to_input(self, _wp, _constants):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            convert_flag_to_input_kernel,
        )

        wp = _wp
        C = _constants
        n = 8
        flag = np.zeros((n, n, n), dtype=np.uint8)
        flag[2, 2, 2] = C.TYPE_I
        flag[3, 3, 3] = C.TYPE_G
        flag[4, 4, 4] = C.TYPE_F
        flag[5, 5, 5] = C.TYPE_S
        flag_wp = wp.array(flag, dtype=wp.uint8)
        inp = wp.zeros((n, n, n), dtype=wp.uint8)
        wp.launch(
            convert_flag_to_input_kernel,
            dim=(n, n, n),
            inputs=[flag_wp, inp, n, n, n],
        )
        out = inp.numpy()
        assert out[2, 2, 2] == 255
        assert out[3, 3, 3] == 255
        assert out[4, 4, 4] == 0
        assert out[5, 5, 5] == 0

    def test_init_tag_boundary_fluid(self, _wp, _constants):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            init_tag_kernel,
        )

        wp = _wp
        C = _constants
        n = 8
        flag = np.full((n, n, n), C.TYPE_G, dtype=np.uint8)
        flag[1, 1, 1] = C.TYPE_F
        flag[2, 2, 2] = C.TYPE_S
        flag_wp = wp.array(flag, dtype=wp.uint8)
        tag = wp.full((n, n, n), 7, dtype=wp.int32)
        prev = wp.full((n, n, n), 7, dtype=wp.int32)
        wp.launch(
            init_tag_kernel,
            dim=(n, n, n),
            inputs=[flag_wp, tag, prev, n, n, n],
        )
        t = tag.numpy()
        p = prev.numpy()
        assert t[1, 1, 1] == -1
        assert t[2, 2, 2] == -1
        assert np.all(p == -1)


class TestParseLabelAndGasLaw:
    def test_parse_label_volume_accumulation(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            parse_label_kernel,
        )

        wp = _wp
        n = 8
        label = np.zeros((n, n, n), dtype=np.int32)
        phi = np.ones((n, n, n), dtype=np.float32)
        # bubble 1: 100 cells with phi=0.3 → vol = 70
        # bubble 2: 50 cells with phi=0.3 → vol = 35  (use 50 for smaller case)
        # Plan says 100 each → 70; use 100 + 100
        coords1 = [(i, j, k) for i in range(2, 6) for j in range(2, 7) for k in range(2, 7)]
        # 4*5*5 = 100
        assert len(coords1) == 100
        for i, j, k in coords1:
            label[i, j, k] = 1
            phi[i, j, k] = 0.3
        coords2 = [(i, j, k) for i in range(0, 5) for j in range(0, 5) for k in range(0, 4)]
        # 5*5*4 = 100
        assert len(coords2) == 100
        for i, j, k in coords2:
            if label[i, j, k] == 0:
                label[i, j, k] = 2
                phi[i, j, k] = 0.3
        # recount bubble 2 cells
        n2 = int(np.sum(label == 2))
        label_wp = wp.array(label, dtype=wp.int32)
        phi_wp = wp.array(phi, dtype=float)
        vol = wp.zeros(8, dtype=wp.float64)
        lnum = wp.zeros(1, dtype=wp.int32)
        wp.launch(
            parse_label_kernel,
            dim=(n, n, n),
            inputs=[label_wp, phi_wp, vol, lnum, n, n, n],
        )
        v = vol.numpy()
        assert abs(v[0] - 70.0) < 1e-5
        assert abs(v[1] - n2 * 0.7) < 1e-5
        assert int(lnum.numpy()[0]) == 2

    def test_bubble_rho_gas_law(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            bubble_rho_update_kernel,
        )

        wp = _wp
        vol = wp.array(np.array([80.0], dtype=np.float64), dtype=wp.float64)
        init = wp.array(np.array([100.0], dtype=np.float64), dtype=wp.float64)
        rho = wp.array(np.array([1.0], dtype=np.float64), dtype=wp.float64)
        wp.launch(
            bubble_rho_update_kernel,
            dim=1,
            inputs=[vol, init, rho, 1],
        )
        assert abs(float(rho.numpy()[0]) - 1.25) < 1e-12

    def test_create_bubble_label_init(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            create_bubble_label_kernel,
        )

        wp = _wp
        label_vol = wp.array(np.array([70.0, 50.0], dtype=np.float64), dtype=wp.float64)
        bvol = wp.zeros(8, dtype=wp.float64)
        binit = wp.zeros(8, dtype=wp.float64)
        brho = wp.zeros(8, dtype=wp.float64)
        lnum = wp.array(np.array([2], dtype=np.int32), dtype=wp.int32)
        bcount = wp.zeros(1, dtype=wp.int32)
        wp.launch(
            create_bubble_label_kernel,
            dim=1,
            inputs=[bvol, binit, brho, label_vol, lnum, bcount],
        )
        assert int(bcount.numpy()[0]) == 2
        assert abs(float(bvol.numpy()[0]) - 70.0) < 1e-12
        assert abs(float(bvol.numpy()[1]) - 50.0) < 1e-12
        assert abs(float(brho.numpy()[0]) - 1.0) < 1e-12


class TestMergeSplitHelpers:
    def test_clear_detector_reset(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            clear_detector_kernel,
        )

        wp = _wp
        n = 4
        md = wp.full((n, n, n), 1, dtype=wp.int32)
        mf = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32)
        sf = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32)
        wp.launch(
            clear_detector_kernel,
            dim=(n, n, n),
            inputs=[md, mf, sf, n, n, n],
        )
        assert np.all(md.numpy() == 0)
        assert int(mf.numpy()[0]) == 0
        assert int(sf.numpy()[0]) == 0

    def test_reset_label_volume(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            reset_label_volume_kernel,
        )

        wp = _wp
        lv = wp.array(np.ones(16, dtype=np.float64), dtype=wp.float64)
        li = wp.array(np.ones(16, dtype=np.float64), dtype=wp.float64)
        wp.launch(reset_label_volume_kernel, dim=16, inputs=[lv, li, 16])
        assert np.allclose(lv.numpy(), 0.0)
        assert np.allclose(li.numpy(), 0.0)

    def test_assign_tag_resolve_conflict(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            assign_tag_kernel,
        )

        wp = _wp
        n = 4
        tag = wp.full((n, n, n), -1, dtype=wp.int32)
        prev_merge = wp.zeros((n, n, n), dtype=wp.int32)
        md = wp.zeros((n, n, n), dtype=wp.int32)
        # cell (1,1,1) needs restore
        prev_m = prev_merge.numpy()
        prev_m[1, 1, 1] = 3
        wp.copy(prev_merge, wp.array(prev_m, dtype=wp.int32))
        md_h = md.numpy()
        md_h[1, 1, 1] = 1
        wp.copy(md, wp.array(md_h, dtype=wp.int32))
        wp.launch(
            assign_tag_kernel,
            dim=(n, n, n),
            inputs=[tag, prev_merge, md, n, n, n],
        )
        assert int(tag.numpy()[1, 1, 1]) == 3

    def test_recheck_merge_distinguish_motion(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            recheck_merge_kernel,
        )

        wp = _wp
        n = 8
        tag = wp.full((n, n, n), -1, dtype=wp.int32)
        # Only bubble A nearby — motion, not merge
        th = tag.numpy()
        th[2, 2, 2] = 1
        th[3, 2, 2] = 1
        wp.copy(tag, wp.array(th, dtype=wp.int32))
        md = wp.zeros((n, n, n), dtype=wp.int32)
        md_h = md.numpy()
        md_h[4, 2, 2] = 1  # detector at empty cell near A only
        wp.copy(md, wp.array(md_h, dtype=wp.int32))
        mf = wp.zeros(1, dtype=wp.int32)
        cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32)
        cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32)
        cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32)
        wp.launch(
            recheck_merge_kernel,
            dim=(n, n, n),
            inputs=[tag, md, mf, cx, cy, cz, n, n, n],
        )
        assert int(mf.numpy()[0]) == 0

    def test_recheck_merge_detects_two_tags(self, _wp):
        from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            recheck_merge_kernel,
        )

        wp = _wp
        n = 8
        tag = wp.full((n, n, n), -1, dtype=wp.int32)
        th = tag.numpy()
        th[2, 2, 2] = 1
        th[4, 2, 2] = 2
        wp.copy(tag, wp.array(th, dtype=wp.int32))
        md = wp.zeros((n, n, n), dtype=wp.int32)
        md_h = md.numpy()
        md_h[3, 2, 2] = 1  # between two tags
        wp.copy(md, wp.array(md_h, dtype=wp.int32))
        mf = wp.zeros(1, dtype=wp.int32)
        cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32)
        cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32)
        cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32)
        wp.launch(
            recheck_merge_kernel,
            dim=(n, n, n),
            inputs=[tag, md, mf, cx, cy, cz, n, n, n],
        )
        assert int(mf.numpy()[0]) == 1


class TestInitBubbleAndMergeConservation:
    def test_init_bubbles_three_spheres(self, _wp, _constants):
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        C = _constants
        model = HomeFslbmModel(fluid_grid_res=(16, 16, 16), max_bubbles=64)
        state = HomeFslbmState(model)
        solver = HomeFslbmSolver(model)

        # Build TYPE_I spheres in fluid background (gas/interface = CCL FG)
        flag = np.full((16, 16, 16), C.TYPE_F, dtype=np.uint8)
        phi = np.ones((16, 16, 16), dtype=np.float32)
        centres = [(4, 4, 4), (12, 4, 4), (8, 12, 12)]
        for cx, cy, cz in centres:
            for i in range(16):
                for j in range(16):
                    for k in range(16):
                        if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= 9:
                            flag[i, j, k] = C.TYPE_I
                            phi[i, j, k] = 0.0
        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.phi, wp.array(phi, dtype=float))

        solver.init_bubbles(state)
        assert state.bubble_count == 3
        vols = state.bubble_volume.numpy()[:3]
        assert np.all(vols > 0)
        tags = state.tag_matrix.numpy()
        assert len(set(int(x) for x in np.unique(tags) if x > 0)) == 3

    def test_merge_detection_two_old_one_new(self, _wp, _constants):
        """Gate B: after handle_merge_split, Σ(V·ρ) conserved."""
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            handle_merge_split,
        )
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        C = _constants
        model = HomeFslbmModel(fluid_grid_res=(32, 32, 32), max_bubbles=64)
        state = HomeFslbmState(model)

        # Two adjacent interface blobs in fluid — touch → one CCL component
        flag = np.full((32, 32, 32), C.TYPE_F, dtype=np.uint8)
        phi = np.ones((32, 32, 32), dtype=np.float32)
        tag = np.full((32, 32, 32), -1, dtype=np.int32)

        # Bubble 1: sphere centre (10,16,16) r=4
        # Bubble 2: sphere centre (17,16,16) r=4  — touching
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    d1 = (i - 10) ** 2 + (j - 16) ** 2 + (k - 16) ** 2
                    d2 = (i - 17) ** 2 + (j - 16) ** 2 + (k - 16) ** 2
                    if d1 <= 16:
                        flag[i, j, k] = C.TYPE_I
                        phi[i, j, k] = 0.0
                        tag[i, j, k] = 1
                    if d2 <= 16:
                        flag[i, j, k] = C.TYPE_I
                        phi[i, j, k] = 0.0
                        tag[i, j, k] = 2

        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.phi, wp.array(phi, dtype=float))
        wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32))

        # Pre-merge bubble densities (volumes recomputed from φ by reduce)
        r1, r2 = 1.0, 1.2
        bv = np.zeros(int(state.bubble_volume.shape[0]), dtype=np.float64)
        bi = np.zeros_like(bv)
        br = np.zeros_like(bv)
        bv[0], bv[1] = 100.0, 80.0
        bi[0], bi[1] = r1 * 100.0, r2 * 80.0
        br[0], br[1] = r1, r2
        wp.copy(state.bubble_volume, wp.array(bv, dtype=wp.float64))
        wp.copy(state.bubble_init_volume, wp.array(bi, dtype=wp.float64))
        wp.copy(state.bubble_rho, wp.array(br, dtype=wp.float64))
        state.bubble_count = 2

        conserved_before = 0.0
        th = tag
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    t = int(th[i, j, k])
                    if t > 0:
                        rho_t = r1 if t == 1 else r2
                        conserved_before += (1.0 - float(phi[i, j, k])) * rho_t

        label_num_gpu = wp.zeros(1, dtype=wp.int32)
        bubble_count_gpu = wp.zeros(1, dtype=wp.int32)
        handle_merge_split(state, label_num_gpu, bubble_count_gpu)

        assert state.bubble_count == 1
        v_new = float(state.bubble_volume.numpy()[0])
        r_new = float(state.bubble_rho.numpy()[0])
        init_new = float(state.bubble_init_volume.numpy()[0])
        # Mass-like quantity Σ(1-φ)·ρ conserved into init_volume; ρ = init/V
        assert abs(init_new - conserved_before) / max(conserved_before, 1e-12) < 1e-6
        assert abs(r_new * v_new - init_new) / max(abs(init_new), 1e-12) < 1e-6
        assert abs(r_new * v_new - conserved_before) / max(conserved_before, 1e-12) < 1e-6


class TestSolverMergeSplitTrigger:
    def test_solver_handle_merge_split_trigger(self, _wp, _constants):
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _wp
        C = _constants
        model = HomeFslbmModel(fluid_grid_res=(16, 16, 16), max_bubbles=32)
        solver = HomeFslbmSolver(model)
        state_in = HomeFslbmState(model)
        state_out = HomeFslbmState(model)

        flag = np.full((16, 16, 16), C.TYPE_F, dtype=np.uint8)
        phi = np.ones((16, 16, 16), dtype=np.float32)
        # small interface regions
        flag[4:8, 4:8, 4:8] = C.TYPE_I
        flag[8:12, 4:8, 4:8] = C.TYPE_I
        phi[4:8, 4:8, 4:8] = 0.0
        phi[8:12, 4:8, 4:8] = 0.0
        wp.copy(state_in.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state_in.phi, wp.array(phi, dtype=float))

        # Force merge path
        state_in.merge_flag = 1
        state_in.bubble_count = 2
        bv = np.zeros(int(state_in.bubble_volume.shape[0]), dtype=np.float64)
        bi = np.zeros_like(bv)
        br = np.zeros_like(bv)
        bv[0], bv[1] = 50.0, 50.0
        bi[0], bi[1] = 50.0, 50.0
        br[0], br[1] = 1.0, 1.0
        wp.copy(state_in.bubble_volume, wp.array(bv, dtype=wp.float64))
        wp.copy(state_in.bubble_init_volume, wp.array(bi, dtype=wp.float64))
        wp.copy(state_in.bubble_rho, wp.array(br, dtype=wp.float64))
        th = np.full((16, 16, 16), -1, dtype=np.int32)
        th[4:8, 4:8, 4:8] = 1
        th[8:12, 4:8, 4:8] = 2
        wp.copy(state_in.tag_matrix, wp.array(th, dtype=wp.int32))

        solver.initialize_equilibrium(state_in)
        solver.step(state_in, state_out, dt=1.0)

        # Merge/split path should have run CCL and cleared flags
        assert state_out.merge_flag == 0
        assert state_out.bubble_count >= 1
