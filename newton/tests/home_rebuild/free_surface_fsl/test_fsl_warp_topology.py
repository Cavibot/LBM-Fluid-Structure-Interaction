# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import (
    HomeCoreModel,
    HomeCoreSolver,
    HomeCoreState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslState,
    FslTopologyUpdater,
    resolve_fsl_topology,
)


def _moments_numpy(state: HomeCoreState) -> np.ndarray:
    return np.moveaxis(
        state.moments.numpy().reshape(10, *state.res), 0, -1
    ).astype(np.float64)


class TestFslWarpTopology(unittest.TestCase):
    def _matches_reference(self, device: str) -> None:
        shape = (8, 3, 3)
        model = HomeCoreModel(fluid_grid_res=shape, device=device)
        fluid = HomeCoreState(model)
        HomeCoreSolver(model).initialize_uniform_lattice(
            fluid, rho=1.0, velocity=(0.03, -0.01, 0.02)
        )
        fill = np.zeros(shape, dtype=np.float32)
        fill[1] = 0.5
        fill[2:5] = 1.0
        fill[5] = 0.5
        source = FslState(model)
        source.initialize_from_fill_level(fluid, fill)
        advected = source.mass.numpy().astype(np.float64)
        advected[1] = 1.1
        queued = np.zeros(shape, dtype=np.float32)
        queued[3, 1, 1] = 0.2
        source.excess_mass.assign(queued)
        expected = resolve_fsl_topology(
            _moments_numpy(fluid),
            source.flags.numpy(),
            advected,
            queued.astype(np.float64),
        )

        fluid_out = HomeCoreState(model)
        destination = FslState(model)
        updater = FslTopologyUpdater(model)
        diagnostics = updater.resolve(
            fluid,
            source,
            wp.array(advected.astype(np.float32), dtype=float, device=device),
            fluid_out,
            destination,
        )

        np.testing.assert_array_equal(destination.flags.numpy(), expected.flags)
        np.testing.assert_allclose(
            destination.mass.numpy(), expected.mass, rtol=3.0e-6, atol=3.0e-7
        )
        np.testing.assert_allclose(
            destination.fill_level.numpy(), expected.fill_level, rtol=3.0e-6, atol=3.0e-7
        )
        np.testing.assert_allclose(
            destination.excess_mass.numpy(), expected.excess_mass, rtol=3.0e-6, atol=3.0e-7
        )
        np.testing.assert_allclose(
            _moments_numpy(fluid_out), expected.moments, rtol=3.0e-6, atol=3.0e-7
        )
        np.testing.assert_array_equal(
            updater.recipient_count.numpy(), expected.recipient_count
        )
        self.assertEqual(diagnostics.interface_to_liquid_count, 9)
        self.assertEqual(diagnostics.gas_to_interface_count, 9)
        self.assertEqual(diagnostics.direct_liquid_gas_link_count, 0)
        self.assertLess(abs(diagnostics.relative_total_mass_drift), 2.0e-7)
        destination.validate(fluid_out, allow_interface_endpoints=True)

    def test_cpu_matches_reference(self) -> None:
        self._matches_reference("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_matches_reference(self) -> None:
        self._matches_reference("cuda:0")


if __name__ == "__main__":
    unittest.main()
