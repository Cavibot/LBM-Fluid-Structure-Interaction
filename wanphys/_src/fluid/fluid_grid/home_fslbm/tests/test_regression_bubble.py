# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 golden-data regression tests for HOME-FSLBM bubble / CCL.

Each scene subdirectory under ``tests/golden_data/`` is produced by the
reference exporter (see ``docs/wanphys/home_fslbm/phase3_bubble_golden_zh.md``).
Until the user drops the ``.txt`` dumps in place, tests are skipped.

Scene IDs B1–B13 match the Phase-3 golden design.
"""

from __future__ import annotations

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
from wanphys._src.fluid.fluid_grid.home_fslbm.tests.conftest import (
    bubble_golden_available,
    load_bubble_golden,
    reorder_ref_scalar_to_warp,
)

# ---------------------------------------------------------------------------
# Scene registry: (scene_name, required_files, kind)
# kind ∈ {"ccl", "init", "coupling"}
# ---------------------------------------------------------------------------

SCENES: dict[str, dict] = {
    # A. Static CCL
    "bubble_ccl_3spheres_r3": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 16,
        "builder": "three_spheres",
        "builder_kwargs": {"radius": 3, "centres": ((4, 4, 4), (12, 4, 4), (8, 12, 12))},
        "expect_count": 3,
    },
    "bubble_ccl_touching_face": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 16,
        "builder": "two_spheres",
        "builder_kwargs": {"radius": 3, "centres": ((6, 8, 8), (10, 8, 8))},
        "expect_count": 1,
    },
    "bubble_ccl_touching_edge": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 16,
        "builder": "two_boxes_edge",
        "builder_kwargs": {},
        "expect_count": 1,
    },
    "bubble_ccl_diagonal_gap": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 16,
        "builder": "two_spheres",
        "builder_kwargs": {"radius": 2, "centres": ((5, 8, 8), (11, 8, 8))},
        "expect_count": 2,
    },
    "bubble_ccl_odd_dims": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 17,
        "builder": "three_spheres",
        "builder_kwargs": {
            "radius": 2,
            "centres": ((4, 4, 4), (12, 4, 4), (8, 12, 12)),
        },
        "expect_count": 3,
    },
    "bubble_ccl_near_wall": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 16,
        "builder": "sphere_near_wall",
        "builder_kwargs": {"radius": 3, "centre": (3, 8, 8)},
        "expect_count": 1,
    },
    "bubble_ccl_many_small": {
        "kind": "ccl",
        "required": ["input_matrix", "label_matrix"],
        "N": 32,
        "builder": "many_small",
        "builder_kwargs": {},
        "expect_count": 8,
    },
    # B. InitBubble
    "bubble_init_two_bubbles": {
        "kind": "init",
        "required": [
            "flag",
            "phi",
            "tag_matrix",
            "bubble_count",
            "bubble_volume",
            "bubble_rho",
        ],
        "N": 32,
        "builder": "fluid_two_spheres",
        "builder_kwargs": {
            "radius": 4,
            "centres": ((10, 16, 16), (22, 16, 16)),
        },
        "expect_count": 2,
        "steps": 0,
    },
    "bubble_init_interface_shell": {
        "kind": "init",
        "required": [
            "flag",
            "phi",
            "tag_matrix",
            "bubble_count",
            "bubble_volume",
        ],
        "N": 32,
        "builder": "interface_shell",
        "builder_kwargs": {"r_inner": 4, "r_outer": 6, "centre": (16, 16, 16)},
        "expect_count": 1,
        "steps": 0,
    },
    # C. Coupling short runs
    "bubble_coupling_volume_delta_phi": {
        "kind": "coupling",
        "required": [
            "flag",
            "phi",
            "tag_matrix",
            "bubble_volume",
            "bubble_rho",
            "bubble_count",
            "steps",
        ],
        "N": 16,
        "builder": "fluid_one_sphere",
        "builder_kwargs": {"radius": 4, "centre": (8, 8, 8)},
        "expect_count": 1,
        "default_steps": 5,
        "gz": 0.0,
    },
    "bubble_two_bubbles_merge": {
        "kind": "coupling",
        "required": [
            "flag",
            "phi",
            "tag_matrix",
            "bubble_volume",
            "bubble_init_volume",
            "bubble_rho",
            "bubble_count",
            "steps",
        ],
        "N": 32,
        "builder": "fluid_two_spheres",
        "builder_kwargs": {
            "radius": 4,
            "centres": ((12, 16, 16), (18, 16, 16)),
        },
        "expect_count": None,  # may be 1 after merge
        "default_steps": 80,
        "gz": 0.0,
        "check_merge_conservation": True,
    },
    "bubble_split_via_surface": {
        "kind": "coupling",
        "required": [
            "flag",
            "phi",
            "tag_matrix",
            "bubble_count",
            "steps",
        ],
        "N": 32,
        "builder": "fluid_one_sphere",
        "builder_kwargs": {"radius": 6, "centre": (16, 16, 16)},
        "expect_count": None,
        "default_steps": 60,
        "gz": 0.0,
    },
    "bubble_translate_no_merge": {
        "kind": "coupling",
        "required": [
            "flag",
            "phi",
            "tag_matrix",
            "bubble_count",
            "merge_flag",
            "steps",
        ],
        "N": 32,
        "builder": "fluid_one_sphere",
        "builder_kwargs": {"radius": 5, "centre": (10, 16, 16)},
        "expect_count": 1,
        "default_steps": 20,
        "gz": -1.0e-4,
    },
}


def _skip_without_golden(scene: str, required: list[str]):
    if not bubble_golden_available(scene, required):
        pytest.skip(
            f"Golden data missing for '{scene}' "
            f"(need {required} under tests/golden_data/{scene}/). "
            f"See docs/wanphys/home_fslbm/phase3_bubble_golden_zh.md"
        )


# ---------------------------------------------------------------------------
# Geometry builders (must match exporter IC — documented in phase3_bubble_golden_zh.md)
# ---------------------------------------------------------------------------


def _paint_sphere(
    mask: np.ndarray, centre: tuple[int, int, int], radius: int, value: int = 255
) -> None:
    cx, cy, cz = centre
    nx, ny, nz = mask.shape
    r2 = radius * radius
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= r2:
                    mask[i, j, k] = value


def build_input_ccl(N: int, builder: str, kwargs: dict) -> np.ndarray:
    """Binary input_matrix (255 FG / 0 BG), shape (N,N,N), Warp [x,y,z]."""
    img = np.zeros((N, N, N), dtype=np.uint8)
    if builder == "three_spheres" or builder == "two_spheres":
        for c in kwargs["centres"]:
            _paint_sphere(img, c, int(kwargs["radius"]), 255)
    elif builder == "two_boxes_edge":
        # Two 3³ cubes that share only an edge (26-connected → one component)
        img[4:7, 4:7, 4:7] = 255
        img[6:9, 6:9, 4:7] = 255  # share edge along z at (6,6,*)
    elif builder == "sphere_near_wall":
        _paint_sphere(img, kwargs["centre"], int(kwargs["radius"]), 255)
        # Wall cells are BG in input_matrix (exporter handled in flag for init scenes)
    elif builder == "many_small":
        centres = [
            (4, 4, 4),
            (4, 4, 28),
            (4, 28, 4),
            (4, 28, 28),
            (28, 4, 4),
            (28, 4, 28),
            (28, 28, 4),
            (28, 28, 28),
        ]
        for c in centres:
            _paint_sphere(img, c, 2, 255)
    else:
        raise ValueError(f"unknown CCL builder {builder}")
    return img


def build_fluid_bubble_scene(
    N: int, builder: str, kwargs: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flag, phi) with TYPE_F background and TYPE_I bubbles (phi=0)."""
    flag = np.full((N, N, N), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((N, N, N), dtype=np.float32)

    def paint_bubble(centre, radius):
        cx, cy, cz = centre
        r2 = radius * radius
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= r2:
                        flag[i, j, k] = C.TYPE_I
                        phi[i, j, k] = 0.0

    if builder == "fluid_one_sphere":
        paint_bubble(kwargs["centre"], int(kwargs["radius"]))
    elif builder == "fluid_two_spheres":
        for c in kwargs["centres"]:
            paint_bubble(c, int(kwargs["radius"]))
    elif builder == "interface_shell":
        cx, cy, cz = kwargs["centre"]
        r_in2 = int(kwargs["r_inner"]) ** 2
        r_out2 = int(kwargs["r_outer"]) ** 2
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    d2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    if d2 <= r_in2:
                        flag[i, j, k] = C.TYPE_G
                        phi[i, j, k] = 0.0
                    elif d2 <= r_out2:
                        flag[i, j, k] = C.TYPE_I
                        phi[i, j, k] = 0.5
    else:
        raise ValueError(f"unknown fluid builder {builder}")

    # Solid walls on domain faces (matches common reference init)
    flag[0, :, :] = C.TYPE_S
    flag[N - 1, :, :] = C.TYPE_S
    flag[:, 0, :] = C.TYPE_S
    flag[:, N - 1, :] = C.TYPE_S
    flag[:, :, 0] = C.TYPE_S
    flag[:, :, N - 1] = C.TYPE_S
    return flag, phi


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _labels_topology_equal(a: np.ndarray, b: np.ndarray, bg: int = 0) -> bool:
    a = np.asarray(a, dtype=np.int32).ravel()
    b = np.asarray(b, dtype=np.int32).ravel()
    if a.shape != b.shape:
        return False
    if not np.array_equal(a == bg, b == bg):
        return False
    a_ids = sorted(int(x) for x in np.unique(a) if x != bg and x != -1)
    b_ids = sorted(int(x) for x in np.unique(b) if x != bg and x != -1)
    if len(a_ids) != len(b_ids):
        return False

    def parts(arr, ids):
        return frozenset(frozenset(np.flatnonzero(arr == lab).tolist()) for lab in ids)

    return parts(a, a_ids) == parts(b, b_ids)


def _compare_float(
    golden: np.ndarray,
    warped: np.ndarray,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> str | None:
    g = np.asarray(golden, dtype=np.float64).ravel()
    w = np.asarray(warped, dtype=np.float64).ravel()
    n = min(g.size, w.size)
    g, w = g[:n], w[:n]
    diff = np.abs(g - w)
    thresh = atol + rtol * np.maximum(np.abs(g), np.abs(w))
    bad = int(np.sum(diff > thresh))
    if bad:
        return f"{name}: {bad}/{n} exceed tol (rtol={rtol}, atol={atol})"
    return None


def _grid_shape(golden: dict, default_N: int) -> tuple[int, int, int]:
    if "nx" in golden:
        nx = int(golden["nx"])
        ny = int(golden.get("ny", nx))
        nz = int(golden.get("nz", nx))
        return nx, ny, nz
    # Infer cubic size from any flat field
    for key in ("input_matrix", "label_matrix", "flag", "phi", "tag_matrix"):
        if key in golden:
            n = int(np.asarray(golden[key]).size)
            side = int(round(n ** (1.0 / 3.0)))
            if side * side * side == n:
                return side, side, side
    return default_N, default_N, default_N


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _warp():
    import warp as wp

    wp.init()
    return wp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBubbleCCLGolden:
    """B1–B7: static CCL vs reference label_matrix."""

    @pytest.mark.parametrize("scene", [k for k, v in SCENES.items() if v["kind"] == "ccl"])
    def test_ccl_scene(self, _warp, scene: str):
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_bubble import (
            connected_component_labeling,
        )

        wp = _warp
        meta = SCENES[scene]
        _skip_without_golden(scene, meta["required"])
        golden = load_bubble_golden(scene)
        nx, ny, nz = _grid_shape(golden, meta["N"])
        assert nx == meta["N"]

        img = build_input_ccl(nx, meta["builder"], meta["builder_kwargs"])
        # Optional: verify exporter input matches builder (if present)
        if "input_matrix" in golden:
            g_in = reorder_ref_scalar_to_warp(golden["input_matrix"], nx, ny, nz)
            assert np.array_equal(g_in.astype(np.uint8), img), (
                f"[{scene}] Warp builder input_matrix disagrees with golden — "
                f"check exporter IC vs build_input_ccl"
            )

        input_m = wp.array(img, dtype=wp.uint8)
        label_m = wp.zeros((nx, ny, nz), dtype=wp.int32)
        nlab = connected_component_labeling(input_m, label_m)
        out = label_m.numpy()

        g_lab = reorder_ref_scalar_to_warp(golden["label_matrix"], nx, ny, nz)
        expect = meta.get("expect_count")
        if expect is not None:
            assert nlab == expect, f"[{scene}] component count {nlab} != {expect}"
            assert int(np.max(g_lab)) == expect or (
                len([x for x in np.unique(g_lab) if x > 0]) == expect
            )

        assert _labels_topology_equal(out, g_lab, bg=0), (
            f"[{scene}] label_matrix topology mismatch vs reference"
        )


class TestBubbleInitGolden:
    """B8–B9: InitBubble vs reference tags / volumes."""

    @pytest.mark.parametrize("scene", [k for k, v in SCENES.items() if v["kind"] == "init"])
    def test_init_scene(self, _warp, scene: str):
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
        from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

        wp = _warp
        meta = SCENES[scene]
        _skip_without_golden(scene, meta["required"])
        golden = load_bubble_golden(scene)
        nx, ny, nz = _grid_shape(golden, meta["N"])

        model = HomeFslbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=1.0,
            max_bubbles=256,
        )
        state = HomeFslbmState(model)
        solver = HomeFslbmSolver(model)

        flag, phi = build_fluid_bubble_scene(nx, meta["builder"], meta["builder_kwargs"])
        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.phi, wp.array(phi, dtype=float))
        solver.init_bubbles(state)

        g_tag = reorder_ref_scalar_to_warp(golden["tag_matrix"], nx, ny, nz)
        assert _labels_topology_equal(
            state.tag_matrix.numpy(), g_tag, bg=-1
        ), f"[{scene}] tag_matrix topology mismatch"

        g_count = int(golden["bubble_count"])
        assert state.bubble_count == g_count

        err = _compare_float(
            np.asarray(golden["bubble_volume"]).ravel()[:g_count],
            state.bubble_volume.numpy()[:g_count],
            "bubble_volume",
            rtol=1e-8,
            atol=1e-9,
        )
        assert err is None, f"[{scene}] {err}"

        if "bubble_rho" in golden:
            err = _compare_float(
                np.asarray(golden["bubble_rho"]).ravel()[:g_count],
                state.bubble_rho.numpy()[:g_count],
                "bubble_rho",
                rtol=1e-8,
                atol=1e-12,
            )
            assert err is None, f"[{scene}] {err}"


class TestBubbleCouplingGolden:
    """B10–B13: short coupling runs vs reference snapshots."""

    @pytest.mark.parametrize(
        "scene", [k for k, v in SCENES.items() if v["kind"] == "coupling"]
    )
    def test_coupling_scene(self, _warp, scene: str):
        from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
        from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
        from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver

        wp = _warp
        meta = SCENES[scene]
        _skip_without_golden(scene, meta["required"])
        golden = load_bubble_golden(scene)
        nx, ny, nz = _grid_shape(golden, meta["N"])
        steps = int(golden.get("steps", meta.get("default_steps", 1)))
        gz = float(meta.get("gz", 0.0))

        model = HomeFslbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=1.0,
            max_bubbles=256,
            gravity_z=gz,
            omega=1.0 / (3.0 * 1e-4 + 0.5),
        )
        solver = HomeFslbmSolver(model)
        domain = HomeFslbmDomain(model, solver=solver)
        domain.create_state()
        state = domain.state

        flag, phi = build_fluid_bubble_scene(nx, meta["builder"], meta["builder_kwargs"])
        wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
        wp.copy(state.phi, wp.array(phi, dtype=float))
        solver.initialize_equilibrium(state)
        solver.init_bubbles(state)

        # Optional: seed delta_phi from golden for volume-update scene
        if scene == "bubble_coupling_volume_delta_phi" and "delta_phi" in golden:
            dphi = reorder_ref_scalar_to_warp(golden["delta_phi"], nx, ny, nz)
            wp.copy(state.delta_phi, wp.array(dphi.astype(np.float32), dtype=float))

        for _ in range(steps):
            domain.step(1.0)

        state = domain.state
        g_tag = reorder_ref_scalar_to_warp(golden["tag_matrix"], nx, ny, nz)
        assert _labels_topology_equal(
            state.tag_matrix.numpy(), g_tag, bg=-1
        ), f"[{scene}] tag_matrix topology mismatch after {steps} steps"

        g_count = int(golden["bubble_count"])
        assert state.bubble_count == g_count, (
            f"[{scene}] bubble_count {state.bubble_count} != golden {g_count}"
        )

        if "bubble_volume" in golden:
            err = _compare_float(
                np.asarray(golden["bubble_volume"]).ravel()[:g_count],
                state.bubble_volume.numpy()[:g_count],
                "bubble_volume",
                rtol=1e-4,
                atol=1e-6,
            )
            assert err is None, f"[{scene}] {err}"

        if "bubble_rho" in golden:
            err = _compare_float(
                np.asarray(golden["bubble_rho"]).ravel()[:g_count],
                state.bubble_rho.numpy()[:g_count],
                "bubble_rho",
                rtol=1e-4,
                atol=1e-8,
            )
            assert err is None, f"[{scene}] {err}"

        if "merge_flag" in golden:
            assert int(state.merge_flag) == int(golden["merge_flag"]), (
                f"[{scene}] merge_flag mismatch"
            )

        if meta.get("check_merge_conservation") and "bubble_init_volume" in golden:
            # Σ V·ρ ≡ Σ init_volume for active bubbles
            init_g = float(np.sum(np.asarray(golden["bubble_init_volume"]).ravel()[:g_count]))
            init_w = float(np.sum(state.bubble_init_volume.numpy()[:g_count]))
            assert abs(init_w - init_g) / max(abs(init_g), 1e-12) < 1e-4


class TestBubbleGoldenRegistry:
    """Sanity: every registered scene is documented and skippable cleanly."""

    def test_registry_kinds(self):
        kinds = {v["kind"] for v in SCENES.values()}
        assert kinds == {"ccl", "init", "coupling"}
        assert len(SCENES) >= 12

    def test_missing_golden_skips(self):
        """A deliberately absent scene name must skip, not error at load."""
        assert not bubble_golden_available(
            "__no_such_bubble_scene__", ["label_matrix"]
        )
