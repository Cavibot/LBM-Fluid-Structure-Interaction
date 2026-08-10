# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslWallMask,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicTopologyResolver,
    resolve_plic_topology_reference,
)

def _upload_moments(state: HomeCoreState, moments: np.ndarray) -> None:
    state.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    )


def _download_moments(state: HomeCoreState) -> np.ndarray:
    return state.moments.numpy().reshape(10, -1).T.reshape(state.res + (10,))


def _warp_case(device: str) -> tuple[np.ndarray, ...]:
    shape = (6, 3, 1)
    closed_axes = (True, False, False)
    solid = axis_aligned_wall_mask(shape, closed_axes=closed_axes)
    source_flags = np.zeros(shape, dtype=np.int32)
    source_flags[1] = 2
    source_flags[2] = 1
    fill = np.zeros(shape, dtype=np.float64)
    fill[1] = 1.0
    fill[2] = 0.8
    fill[3] = 0.2
    density = 1.0 + 0.01 * np.arange(shape[0])[:, None, None]
    mass = fill * density
    momentum = mass[..., None] * np.asarray((0.03, -0.02, 0.01))
    moments = np.zeros(shape + (10,), dtype=np.float64)
    moments[..., 0] = density
    moments[..., 1] = density * 0.02
    moments[..., 2] = density * -0.01
    moments[..., 4] = density * 0.02**2
    moments[..., 5] = density * 0.01**2
    moments[..., 7] = density * -0.0002
    moments[3] = -9.0
    expected = resolve_plic_topology_reference(
        moments,
        source_flags,
        fill,
        mass,
        momentum,
        solid,
        periodic=tuple(not value for value in closed_axes),
    )
    model = HomeCoreModel(fluid_grid_res=source_flags.shape, device=device)
    walls = FslWallMask(
        model,
        solid,
        closed_axes=closed_axes,
    )
    candidate = HomeCoreState(model)
    destination = HomeCoreState(model)
    _upload_moments(candidate, moments)
    source = FslState(model)
    source.flags.assign(source_flags)
    source.fill_level.assign(np.where(source_flags == 2, 1.0, np.where(source_flags == 1, 0.8, 0.0)).astype(np.float32))
    source.mass.assign(source.fill_level.numpy())
    committed = FslState(model)
    transported_momentum = wp.array(momentum, dtype=wp.vec3, device=device)
    resolver = PlicTopologyResolver(model, walls)
    diagnostics = resolver.resolve(
        candidate,
        source,
        wp.array(fill, dtype=float, device=device),
        wp.array(mass, dtype=float, device=device),
        transported_momentum,
        destination,
        committed,
    )
    return (
        committed.flags.numpy(),
        committed.fill_level.numpy(),
        committed.mass.numpy(),
        _download_moments(destination),
        resolver.donor_count.numpy(),
        expected.flags,
        expected.fill_level,
        expected.mass,
        expected.moments,
        expected.donor_count,
        diagnostics,
    )


class TestPlicTopologyWarp(unittest.TestCase):
    def test_cpu_matches_reference(self) -> None:
        result = _warp_case("cpu")
        np.testing.assert_array_equal(result[0], result[5])
        np.testing.assert_allclose(result[1], result[6], rtol=0.0, atol=2.0e-8)
        np.testing.assert_allclose(result[2], result[7], rtol=3.0e-6, atol=3.0e-7)
        np.testing.assert_allclose(result[3], result[8], rtol=3.0e-6, atol=3.0e-7)
        np.testing.assert_array_equal(result[4], result[9])
        self.assertEqual(result[10].gas_to_interface_count, 3)

    def test_cuda_matches_cpu(self) -> None:
        cpu = _warp_case("cpu")
        cuda = _warp_case("cuda:0")
        for actual, expected in zip(cuda[:5], cpu[:5]):
            np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)

    def test_warp_repairs_new_liquid_next_to_gas_without_ledger_change(self) -> None:
        shape = (5, 5, 1)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask.periodic_depth_channel(model)
        candidate = HomeCoreState(model)
        destination = HomeCoreState(model)
        moments = np.zeros(shape + (10,), dtype=np.float32)
        moments[..., 0] = 1.0
        _upload_moments(candidate, moments)
        source = FslState(model)
        source_flags = np.zeros(shape, dtype=np.int32)
        source_flags[2, 2, 0] = 1
        source.flags.assign(source_flags)
        fill = np.zeros(shape, dtype=np.float32)
        fill[2, 2, 0] = 1.0
        mass = fill.copy()
        committed = FslState(model)
        diagnostics = PlicTopologyResolver(model, walls).resolve(
            candidate,
            source,
            wp.array(fill, dtype=float, device="cpu"),
            wp.array(mass, dtype=float, device="cpu"),
            wp.zeros(shape, dtype=wp.vec3, device="cpu"),
            destination,
            committed,
        )
        self.assertEqual(committed.flags.numpy()[2, 2, 0], 1)
        self.assertEqual(committed.fill_level.numpy()[2, 2, 0], 1.0)
        self.assertEqual(committed.mass.numpy()[2, 2, 0], 1.0)
        self.assertEqual(diagnostics.separation_repair_count, 1)

    def test_warp_conservatively_closes_sub_tolerance_endpoint(self) -> None:
        shape = (7, 5, 1)
        model = HomeCoreModel(fluid_grid_res=shape, device="cpu")
        walls = FslWallMask.periodic_depth_channel(model)
        candidate = HomeCoreState(model)
        destination = HomeCoreState(model)
        moments = np.zeros(shape + (10,), dtype=np.float32)
        moments[..., 0] = 1.0
        _upload_moments(candidate, moments)
        source = FslState(model)
        source_flags = np.zeros(shape, dtype=np.int32)
        source_flags[2:5, 1:4, 0] = 1
        source.flags.assign(source_flags)
        fill = np.zeros(shape, dtype=np.float32)
        fill[2:5, 1:4, 0] = 0.4
        fill[3, 2, 0] = 1.0e-7
        mass = 0.9 * fill
        momentum = mass[..., None] * np.asarray(
            (0.03, -0.02, 0.01), dtype=np.float32
        )
        expected = resolve_plic_topology_reference(
            moments,
            source_flags,
            fill,
            mass,
            momentum,
            walls.host,
            periodic=(False, False, True),
            endpoint_tolerance=2.0e-6,
        )
        committed = FslState(model)
        resolver = PlicTopologyResolver(
            model, walls, endpoint_tolerance=2.0e-6
        )
        diagnostics = resolver.resolve(
            candidate,
            source,
            wp.array(fill, dtype=float, device="cpu"),
            wp.array(mass, dtype=float, device="cpu"),
            wp.array(momentum, dtype=wp.vec3, device="cpu"),
            destination,
            committed,
        )

        np.testing.assert_array_equal(committed.flags.numpy(), expected.flags)
        np.testing.assert_allclose(
            committed.fill_level.numpy(), expected.fill_level, atol=2.0e-7
        )
        np.testing.assert_allclose(
            committed.mass.numpy(), expected.mass, atol=2.0e-7
        )
        np.testing.assert_allclose(
            resolver.closed_momentum.numpy(), expected.momentum, atol=2.0e-8
        )
        self.assertEqual(diagnostics.endpoint_removed_gas_count, 1)
        self.assertAlmostEqual(
            diagnostics.endpoint_redistributed_volume, 1.0e-7, places=12
        )


if __name__ == "__main__":
    unittest.main()
