# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Atmospheric-pressure anchoring of a planar HOME-Free surface."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeLbmModel,
)


class TestHomeFreeHydrostatic(unittest.TestCase):
    def _run_planar_hydrostatic_column(self, device: str) -> None:
        shape = (6, 6, 12)
        interface_index = 6
        acceleration = -2.0e-4
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :interface_index] = 1.0
        fill[:, :, interface_index] = 0.5
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=(0.0, 0.0, acceleration),
            periodic=(True, True, False),
            max_lattice_speed=0.1,
            device=device,
        )
        domain = HomeFreeLegacyDomain(model, gas_density=1.0)
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
        moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        initial_density = moments[0].reshape(shape)
        lattice_acceleration = model.lattice_acceleration[2]
        ratio = (1.0 + 1.5 * lattice_acceleration) / (
            1.0 - 1.5 * lattice_acceleration
        )

        # A half-filled axis-aligned PLIC plane lies at the cell center. Eq. (11)
        # applies atmospheric pressure one half-link into gas, so the interface
        # cell center is one half recurrence step below that pressure reference.
        np.testing.assert_allclose(
            initial_density[:, :, interface_index],
            ratio**-0.5,
            rtol=2.0e-7,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            initial_density[:, :, 1 : interface_index + 1]
            / initial_density[:, :, :interface_index],
            ratio,
            rtol=2.0e-7,
            atol=2.0e-7,
        )
        active = np.isin(initial_flags.reshape(-1), (1, 2))
        np.testing.assert_allclose(
            moments[3, active] / moments[0, active],
            0.5 * lattice_acceleration,
            rtol=0.0,
            atol=2.0e-8,
        )

        for _ in range(200):
            domain.step(model.time_step)

        final_flags = domain.free_surface_state.flags.numpy()
        final_fill = domain.free_surface_state.fill_level.numpy()
        final_moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        final_density = final_moments[0].reshape(shape)
        counts = domain.topology.active_neighbor_count.numpy()
        represented_mass = float(
            np.sum(
                domain.free_surface_state.mass.numpy()
                + domain.free_surface_state.excess_mass.numpy() * counts,
                dtype=np.float64,
            )
        )

        np.testing.assert_array_equal(final_flags, initial_flags)
        self.assertLess(
            float(np.max(np.abs(final_fill - initial_fill))),
            1.0e-4,
        )
        np.testing.assert_allclose(
            final_density[:, :, : interface_index + 1],
            initial_density[:, :, : interface_index + 1],
            rtol=5.0e-4,
            atol=5.0e-5,
        )
        np.testing.assert_allclose(
            final_moments[3, active] / final_moments[0, active],
            0.5 * lattice_acceleration,
            rtol=0.0,
            atol=3.0e-5,
        )
        self.assertLess(abs(represented_mass - initial_mass) / initial_mass, 5.0e-6)

    def test_cpu_planar_surface_remains_hydrostatic(self) -> None:
        self._run_planar_hydrostatic_column("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_planar_surface_remains_hydrostatic(self) -> None:
        self._run_planar_hydrostatic_column("cuda:0")

    def test_initializer_rejects_nonplanar_or_transversely_forced_input(self) -> None:
        shape = (4, 4, 6)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:, :, :3] = 1.0
        fill[:, :, 3] = 0.5
        fill[0, 0, 3] = 0.25
        model = HomeLbmModel(
            fluid_grid_res=shape,
            body_acceleration=(1.0e-4, 0.0, -1.0e-4),
            periodic=(False, False, False),
            device="cpu",
        )
        domain = HomeFreeLegacyDomain(model)
        with self.assertRaisesRegex(ValueError, "one fractional fill"):
            domain.initialize_planar_hydrostatic_lattice(
                fill,
                interface_axis=2,
                interface_index=3,
                gas_direction=1,
            )

        fill[:, :, 3] = 0.5
        with self.assertRaisesRegex(ValueError, "axis-aligned"):
            domain.initialize_planar_hydrostatic_lattice(
                fill,
                interface_axis=2,
                interface_index=3,
                gas_direction=1,
            )


if __name__ == "__main__":
    unittest.main()
