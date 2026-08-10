# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel, HomeCoreState
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallActiveCollider,
    FslWallMassAdvector,
    FslWallMask,
    FslWallOnlyMissingStreamer,
    closed_box_wall_mask,
    wall_link_mass_exchange,
    wall_only_missing_stream_moments,
)


def _rest_moments(shape: tuple[int, int, int]) -> np.ndarray:
    moments = np.zeros(shape + (10,), dtype=np.float32)
    moments[..., 0] = 1.0
    return moments


def _upload_moments(state: HomeCoreState, moments: np.ndarray) -> None:
    soa = np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1))
    state.moments.assign(soa)


def _download_moments(state: HomeCoreState) -> np.ndarray:
    return state.moments.numpy().reshape(10, -1).T.reshape(state.res + (10,))


class TestFslWarpWalls(unittest.TestCase):
    def _alignment(self, device: str) -> None:
        shape = (8, 7, 6)
        solid = closed_box_wall_mask(shape)
        model = HomeCoreModel(fluid_grid_res=shape, device=device)
        walls = FslWallMask(model, solid)
        fluid = HomeCoreState(model)
        moments = _rest_moments(shape)
        _upload_moments(fluid, moments)
        fsl = FslState(model)
        flags = np.full(shape, int(FslCellFlag.LIQUID), dtype=np.int32)
        flags[solid] = int(FslCellFlag.GAS)
        mass = np.ones(shape, dtype=np.float32)
        mass[solid] = 0.0
        fsl.flags.assign(flags)
        fsl.mass.assign(mass)
        fsl.fill_level.assign(mass)

        expected_stream = wall_only_missing_stream_moments(moments, flags, solid)
        streamer = FslWallOnlyMissingStreamer(model, walls)
        streamed = streamer.stream(fluid, fsl)
        stream_diagnostics = streamer.collect_diagnostics()
        expected_mass = wall_link_mass_exchange(moments, mass, mass, flags, solid)
        advector = FslWallMassAdvector(model, walls)
        advector.advect(fluid, fsl)
        mass_diagnostics = advector.collect_diagnostics()

        np.testing.assert_allclose(
            _download_moments(streamed), expected_stream.moments,
            rtol=3.0e-6, atol=3.0e-7,
        )
        np.testing.assert_array_equal(
            streamer.wall_link_count.numpy(), expected_stream.wall_link_count
        )
        np.testing.assert_allclose(
            advector.advected_mass.numpy(), expected_mass,
            rtol=3.0e-6, atol=3.0e-7,
        )
        self.assertEqual(stream_diagnostics.invalid_cell_count, 0)
        self.assertEqual(stream_diagnostics.direct_liquid_gas_link_count, 0)
        self.assertEqual(mass_diagnostics.invalid_cell_count, 0)
        self.assertLess(abs(mass_diagnostics.relative_mass_drift), 2.0e-7)

    def test_cpu_matches_wall_oracle(self) -> None:
        self._alignment("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_matches_wall_oracle(self) -> None:
        self._alignment("cuda:0")

    def test_gravity_collision_skips_gas_and_solid_cells(self) -> None:
        shape = (7, 7, 7)
        solid = closed_box_wall_mask(shape)
        model = HomeCoreModel(
            fluid_grid_res=shape, device="cpu", body_acceleration=(0.0, -1.0e-4, 0.0)
        )
        walls = FslWallMask(model, solid)
        source = HomeCoreState(model)
        destination = HomeCoreState(model)
        moments = _rest_moments(shape)
        _upload_moments(source, moments)
        fsl = FslState(model)
        flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
        flags[3, 3, 3] = int(FslCellFlag.LIQUID)
        fsl.flags.assign(flags)
        collider = FslWallActiveCollider(model, walls)

        collider.collide(source, fsl, destination, model.time_step)
        result = _download_moments(destination)

        self.assertAlmostEqual(
            result[3, 3, 3, 2], model.lattice_acceleration[1], places=10
        )
        np.testing.assert_array_equal(result[..., 1:4][flags == int(FslCellFlag.GAS)], 0.0)
        self.assertEqual(collider.collect_diagnostics().invalid_cell_count, 0)


if __name__ == "__main__":
    unittest.main()
