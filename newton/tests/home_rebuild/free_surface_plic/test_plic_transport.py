# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslWallMask,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicGeometricTransport,
)


def _sample_circle(shape: tuple[int, int, int], *, samples: int = 12) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    center = np.asarray((shape[0] / 2.0, shape[1] / 2.0))
    radius = 6.0
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples - 0.5
    for i in range(shape[0]):
        for j in range(shape[1]):
            inside = 0
            for dx in offsets:
                for dy in offsets:
                    point = np.asarray((i + dx, j + dy))
                    inside += int(np.linalg.norm(point - center) <= radius)
            fill[i, j, 1] = inside / (samples * samples)
    return fill


def _translation_case(device: str) -> tuple[np.ndarray, np.ndarray, float]:
    shape = (32, 32, 3)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    closed_axes = (False, False, True)
    walls = FslWallMask(
        model,
        axis_aligned_wall_mask(shape, closed_axes=closed_axes),
        closed_axes=closed_axes,
    )
    initial = _sample_circle(shape)
    fill = wp.array(initial, dtype=float, device=device)
    cx = np.zeros((33, 32, 3), dtype=np.float32)
    cy = np.zeros((32, 33, 3), dtype=np.float32)
    cz = np.zeros((32, 32, 4), dtype=np.float32)
    cx[:, :, 1] = 0.2
    cy[:, :, 1] = 0.15
    courant = (
        wp.array(cx, dtype=float, device=device),
        wp.array(cy, dtype=float, device=device),
        wp.array(cz, dtype=float, device=device),
    )
    transport = PlicGeometricTransport(model, walls)
    maximum_drift = 0.0
    for _ in range(20):
        fill, diagnostics = transport.transport(
            fill, courant, split_order=(0, 1)
        )
        maximum_drift = max(maximum_drift, abs(diagnostics.relative_volume_drift))
    expected = np.roll(np.roll(initial, 4, axis=0), 3, axis=1)
    return fill.numpy(), expected, maximum_drift


def _vortex_courant(
    shape: tuple[int, int, int], *, maximum_courant: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    stream = np.zeros((nx, ny), dtype=np.float64)
    for i in range(1, nx):
        x = (i - 1) / (nx - 2)
        for j in range(1, ny):
            y = (j - 1) / (ny - 2)
            stream[i, j] = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
    cx = np.zeros((nx + 1, ny, nz), dtype=np.float64)
    cy = np.zeros((nx, ny + 1, nz), dtype=np.float64)
    cz = np.zeros((nx, ny, nz + 1), dtype=np.float64)
    for i in range(1, nx):
        for j in range(1, ny - 1):
            cx[i, j, 1] = stream[i, j + 1] - stream[i, j]
    for i in range(1, nx - 1):
        for j in range(1, ny):
            cy[i, j, 1] = -(stream[i + 1, j] - stream[i, j])
    scale = maximum_courant / max(
        float(np.max(np.abs(cx))), float(np.max(np.abs(cy)))
    )
    cx *= scale
    cy *= scale
    divergence = (
        cx[2:nx, 1 : ny - 1, 1]
        - cx[1 : nx - 1, 1 : ny - 1, 1]
        + cy[1 : nx - 1, 2:ny, 1]
        - cy[1 : nx - 1, 1 : ny - 1, 1]
    )
    if float(np.max(np.abs(divergence), initial=0.0)) > 2.0e-14:
        raise AssertionError("analytic vortex Courant field is not divergence-free")
    return cx.astype(np.float32), cy.astype(np.float32), cz.astype(np.float32)


def _reversible_vortex_case(device: str) -> tuple[np.ndarray, np.ndarray, float]:
    shape = (34, 34, 3)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    walls = FslWallMask.closed_box(model)
    initial = _sample_circle(shape)
    initial = np.roll(np.roll(initial, -5, axis=0), 3, axis=1)
    initial[walls.host] = 0.0
    positive = _vortex_courant(shape, maximum_courant=0.18)
    negative = tuple(-field for field in positive)
    positive_wp = tuple(
        wp.array(field, dtype=float, device=device) for field in positive
    )
    negative_wp = tuple(
        wp.array(field, dtype=float, device=device) for field in negative
    )
    transport = PlicGeometricTransport(model, walls)
    fill = wp.array(initial, dtype=float, device=device)
    maximum_drift = 0.0
    for _ in range(20):
        fill, diagnostics = transport.transport(
            fill, positive_wp, split_order=(0, 1)
        )
        maximum_drift = max(maximum_drift, abs(diagnostics.relative_volume_drift))
    for _ in range(20):
        fill, diagnostics = transport.transport(
            fill, negative_wp, split_order=(1, 0)
        )
        maximum_drift = max(maximum_drift, abs(diagnostics.relative_volume_drift))
    return fill.numpy(), initial, maximum_drift


class TestPlicGeometricTransport(unittest.TestCase):
    def test_cpu_oblique_translation_preserves_circle(self) -> None:
        actual, expected, maximum_drift = _translation_case("cpu")
        self.assertLess(maximum_drift, 4.0e-6)
        self.assertLess(float(np.mean(np.abs(actual - expected))), 5.0e-3)
        self.assertGreaterEqual(float(np.min(actual)), 0.0)
        self.assertLessEqual(float(np.max(actual)), 1.0)

    def test_cuda_matches_cpu_translation(self) -> None:
        cpu = _translation_case("cpu")
        cuda = _translation_case("cuda:0")
        np.testing.assert_allclose(cuda[0], cpu[0], rtol=4.0e-5, atol=4.0e-5)

    def test_reversible_vortex_preserves_volume_and_recovers_shape(self) -> None:
        actual, initial, maximum_drift = _reversible_vortex_case("cuda:0")
        active_error = np.abs(actual[:, :, 1] - initial[:, :, 1])
        self.assertLess(maximum_drift, 4.0e-6)
        self.assertLess(float(np.mean(active_error)), 2.0e-3)
        self.assertLess(
            float(np.sum(active_error) / np.sum(initial[:, :, 1])), 1.0e-2
        )
        self.assertGreaterEqual(float(np.min(actual)), 0.0)
        self.assertLessEqual(float(np.max(actual)), 1.0)


if __name__ == "__main__":
    unittest.main()
