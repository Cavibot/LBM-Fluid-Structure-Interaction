# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic constant-speed translation of two periodic sharp interfaces."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeLbmModel,
)


class TestHomeFreeTranslation(unittest.TestCase):
    def _run_translation(self, device: str) -> None:
        shape = (12, 3, 2)
        speed = 0.02
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 0.75
        fill[1:5] = 1.0
        fill[5] = 0.25
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            periodic=(True, True, True),
            device=device,
        )
        domain = HomeFreeLegacyDomain(model)
        domain.initialize_uniform_lattice(fill, velocity=(speed, 0.0, 0.0))
        initial_mass = float(
            np.sum(domain.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        initial_flags = domain.free_surface_state.flags.numpy().copy()

        for step in range(1, 11):
            domain.step(model.time_step)
            current_fill = domain.free_surface_state.fill_level.numpy()
            np.testing.assert_allclose(
                current_fill[0], 0.75 - step * speed, rtol=0.0, atol=1.5e-6
            )
            np.testing.assert_allclose(
                current_fill[5], 0.25 + step * speed, rtol=0.0, atol=1.5e-6
            )
            np.testing.assert_array_equal(
                domain.free_surface_state.flags.numpy(), initial_flags
            )
            moments = domain.fluid_state.moments.numpy().reshape(10, -1)
            active = np.isin(initial_flags.reshape(-1), (1, 2))
            velocity_x = moments[1, active] / moments[0, active]
            np.testing.assert_allclose(
                velocity_x, speed, rtol=1.0e-5, atol=2.0e-7
            )
            counts = domain.topology.active_neighbor_count.numpy()
            represented_mass = float(
                np.sum(
                    domain.free_surface_state.mass.numpy()
                    + domain.free_surface_state.excess_mass.numpy() * counts,
                    dtype=np.float64,
                )
            )
            self.assertLess(abs(represented_mass - initial_mass), 5.0e-6)

    def test_cpu_constant_speed_translation_is_exact(self) -> None:
        self._run_translation("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_constant_speed_translation_is_exact(self) -> None:
        self._run_translation("cuda:0")


if __name__ == "__main__":
    unittest.main()
