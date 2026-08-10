# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Mass-weighted hydrostatic balance of geometric HOME-Free."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeGeometricDomain,
    HomeLbmModel,
)


class TestHomeFreeGeometricHydrostatic(unittest.TestCase):
    def _run_planar_column(self, device: str) -> None:
        shape = (6, 6, 12)
        interface_index = 6
        interface_fill = 0.5
        acceleration = -2.0e-4
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :interface_index] = 1.0
        fill[:, :, interface_index] = interface_fill
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            body_acceleration=(0.0, 0.0, acceleration),
            periodic=(True, True, False),
            kinematic_viscosity=0.2,
            max_lattice_speed=0.1,
            device=device,
        )
        domain = HomeFreeGeometricDomain(model, gas_density=1.0)
        domain.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=interface_index,
            gas_direction=1,
        )

        initial_flags = domain.free_surface_state.flags.numpy().copy()
        initial_fill = domain.free_surface_state.fill_level.numpy().copy()
        initial_mass = float(
            np.sum(domain.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        initial_moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        initial_density = initial_moments[0].reshape(shape).copy()
        lattice_acceleration = float(model.lattice_acceleration[2])
        interface_density = 1.0 / (
            1.0 + 1.5 * interface_fill * lattice_acceleration
        )
        np.testing.assert_allclose(
            initial_density[:, :, interface_index],
            interface_density,
            rtol=2.0e-7,
            atol=2.0e-7,
        )

        for _ in range(200):
            domain.step(model.time_step)

        final_flags = domain.free_surface_state.flags.numpy()
        final_fill = domain.free_surface_state.fill_level.numpy()
        final_mass = float(
            np.sum(domain.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        final_moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        final_density = final_moments[0].reshape(shape)
        active_fill = fill.reshape(-1) > 0.0
        expected_momentum = (
            0.5
            * final_moments[0, active_fill]
            * fill.reshape(-1)[active_fill]
            * lattice_acceleration
        )

        np.testing.assert_array_equal(final_flags, initial_flags)
        np.testing.assert_allclose(final_fill, initial_fill, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(
            final_density[:, :, : interface_index + 1],
            initial_density[:, :, : interface_index + 1],
            rtol=8.0e-7,
            atol=8.0e-7,
        )
        np.testing.assert_allclose(
            final_moments[3, active_fill],
            expected_momentum,
            rtol=0.0,
            atol=4.0e-7,
        )
        self.assertLess(abs(final_mass - initial_mass) / initial_mass, 1.0e-7)

    def test_cpu_planar_column_is_a_discrete_fixed_point(self) -> None:
        self._run_planar_column("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_planar_column_is_a_discrete_fixed_point(self) -> None:
        self._run_planar_column("cuda:0")


if __name__ == "__main__":
    unittest.main()
