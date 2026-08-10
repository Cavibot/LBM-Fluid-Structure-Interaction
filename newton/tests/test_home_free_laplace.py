# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Static-plane and coarse spherical Laplace-pressure acceptance tests."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeLbmModel,
)


def _upload_rest_density(
    domain: HomeFreeLegacyDomain, density: np.ndarray
) -> None:
    moments = np.zeros(density.shape + (10,), dtype=np.float32)
    moments[..., 0] = density
    domain.fluid_state.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1))
    )


class TestHomeFreeLaplace(unittest.TestCase):
    def test_planar_interface_has_zero_curvature_and_no_capillary_flow(self) -> None:
        shape = (8, 4, 4)
        fill = np.zeros(shape, dtype=np.float32)
        fill[:3] = 1.0
        fill[3] = 0.5
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.3,
            periodic=(False, True, True),
            device="cpu",
        )
        domain = HomeFreeLegacyDomain(model, surface_tension=0.05)
        domain.initialize_uniform_lattice(fill)

        for _ in range(20):
            domain.step(model.time_step)

        flags = domain.free_surface_state.flags.numpy()
        interface = flags == 1
        curvature = domain.surface_tension.geometry.curvature.numpy()[interface]
        moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        active = np.isin(flags.reshape(-1), (1, 2))
        speed = np.linalg.norm(
            moments[1:4, active] / moments[0, active][None, :], axis=0
        )
        np.testing.assert_allclose(curvature, 0.0, atol=2.0e-7)
        np.testing.assert_allclose(
            domain.surface_tension.gas_density.numpy()[interface], 1.0, atol=2.0e-7
        )
        self.assertLess(float(np.max(speed)), 2.0e-6)

    def test_spherical_drop_relaxes_to_discrete_laplace_pressure(self) -> None:
        shape = (17, 17, 17)
        center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0)
        coordinates = np.indices(shape).transpose(1, 2, 3, 0)
        distance = np.linalg.norm(coordinates - center, axis=-1)
        radius = 5.25
        gamma = 0.003
        fill = np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0).astype(
            np.float32
        )
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.5,
            periodic=(False, False, False),
            max_lattice_speed=0.2,
            device="cpu",
        )
        domain = HomeFreeLegacyDomain(model, surface_tension=gamma)
        density = np.ones(shape, dtype=np.float32)
        density[fill > 0.0] = 1.0 + 6.0 * gamma / radius
        _upload_rest_density(domain, density)
        domain.free_surface_state.initialize_from_fill_level(
            domain.fluid_state, fill
        )
        initial_mass = float(
            np.sum(domain.free_surface_state.mass.numpy(), dtype=np.float64)
        )

        for _ in range(3000):
            domain.step(model.time_step)

        flags = domain.free_surface_state.flags.numpy()
        interface = flags == 1
        liquid = flags == 2
        moments = (
            domain.fluid_state.moments.numpy()
            .reshape(10, -1)
            .T.reshape(shape + (10,))
        )
        measured_pressure_jump = (float(np.mean(moments[..., 0][liquid])) - 1.0) / 3.0
        mean_curvature = float(
            np.mean(domain.surface_tension.geometry.curvature.numpy()[interface])
        )
        expected_pressure_jump = -2.0 * gamma * mean_curvature
        velocity = np.linalg.norm(
            moments[..., 1:4] / moments[..., 0, None], axis=-1
        )
        counts = domain.topology.active_neighbor_count.numpy()
        represented_mass = float(
            np.sum(
                domain.free_surface_state.mass.numpy()
                + domain.free_surface_state.excess_mass.numpy() * counts,
                dtype=np.float64,
            )
        )

        self.assertGreater(expected_pressure_jump, 0.0)
        self.assertLess(
            abs(measured_pressure_jump - expected_pressure_jump)
            / expected_pressure_jump,
            0.16,
        )
        self.assertLess(float(np.max(velocity[liquid | interface])), 1.0e-4)
        self.assertLess(abs(represented_mass - initial_mass) / initial_mass, 2.0e-5)


if __name__ == "__main__":
    unittest.main()
