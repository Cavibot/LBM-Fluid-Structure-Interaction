# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Gravity-driven liquid-column collapse with evolving HOME-Free topology."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeLegacyDomain,
    HomeLbmModel,
)


@dataclass(frozen=True)
class _CollapseMetrics:
    relative_mass_error: float
    center_x: float
    center_z: float
    front_x: int
    top_z: int
    interface_cells: int
    changed_cells: int
    maximum_speed: float


def _represented_mass(domain: HomeFreeLegacyDomain) -> np.ndarray:
    recipients = domain.topology.active_neighbor_count.numpy()
    state = domain.free_surface_state
    return state.mass.numpy() + state.excess_mass.numpy() * recipients


class TestHomeFreeColumnCollapse(unittest.TestCase):
    def _run_collapse(self, device: str) -> tuple[_CollapseMetrics, _CollapseMetrics]:
        shape = (24, 4, 16)
        fill_x = np.zeros(shape[0], dtype=np.float32)
        fill_z = np.zeros(shape[2], dtype=np.float32)
        fill_x[:8] = 1.0
        fill_x[8] = 0.5
        fill_z[:8] = 1.0
        fill_z[8] = 0.5
        fill = fill_x[:, None, None] * fill_z[None, None, :]
        fill = np.broadcast_to(fill, shape).copy()
        model = HomeLbmModel(
            fluid_grid_res=shape,
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.08,
            body_acceleration=(0.0, 0.0, -2.0e-4),
            periodic=(False, True, False),
            max_lattice_speed=0.2,
            device=device,
        )
        domain = HomeFreeLegacyDomain(model)
        domain.initialize_anchored_hydrostatic_lattice(
            fill,
            # The top PLIC plane is z=8.5; Eq. (11)'s pressure location is
            # another half link into gas. Side exposure then drives collapse.
            reference_coordinate=(12.0, 2.0, 9.0),
        )
        initial_flags = domain.free_surface_state.flags.numpy().copy()
        initial_mass = float(np.sum(_represented_mass(domain), dtype=np.float64))
        initial = self._metrics(domain, initial_flags, initial_mass)

        for _ in range(400):
            domain.step(model.time_step)

        final = self._metrics(domain, initial_flags, initial_mass)
        self.assertIsNotNone(domain.last_diagnostics)
        assert domain.last_diagnostics is not None
        self.assertEqual(domain.last_diagnostics.direct_liquid_gas_link_count, 0)
        self.assertLess(final.relative_mass_error, 5.0e-6)
        self.assertGreater(final.center_x - initial.center_x, 1.3)
        self.assertGreater(initial.center_z - final.center_z, 1.1)
        self.assertGreaterEqual(final.front_x - initial.front_x, 5)
        self.assertLessEqual(final.top_z, initial.top_z - 1)
        self.assertGreaterEqual(final.interface_cells - initial.interface_cells, 20)
        self.assertGreaterEqual(final.changed_cells, 200)
        self.assertGreater(final.maximum_speed, 1.0e-2)
        self.assertLess(final.maximum_speed, 3.0e-2)
        return initial, final

    @staticmethod
    def _metrics(
        domain: HomeFreeLegacyDomain,
        initial_flags: np.ndarray,
        initial_mass: float,
    ) -> _CollapseMetrics:
        represented = _represented_mass(domain)
        total_mass = float(np.sum(represented, dtype=np.float64))
        coordinates = np.indices(represented.shape)
        flags = domain.free_surface_state.flags.numpy()
        active = np.isin(flags, (1, 2))
        active_x = np.where(np.any(active, axis=(1, 2)))[0]
        active_z = np.where(np.any(active, axis=(0, 1)))[0]
        moments = domain.fluid_state.moments.numpy().reshape(10, -1)
        flat_active = active.reshape(-1)
        velocity = moments[1:4, flat_active] / moments[0, flat_active][None, :]
        return _CollapseMetrics(
            relative_mass_error=abs(total_mass - initial_mass) / initial_mass,
            center_x=float(
                np.sum(represented * coordinates[0], dtype=np.float64) / total_mass
            ),
            center_z=float(
                np.sum(represented * coordinates[2], dtype=np.float64) / total_mass
            ),
            front_x=int(np.max(active_x)),
            top_z=int(np.max(active_z)),
            interface_cells=int(np.sum(flags == 1)),
            changed_cells=int(np.sum(flags != initial_flags)),
            maximum_speed=float(np.max(np.linalg.norm(velocity, axis=0))),
        )

    def test_cpu_column_collapses_and_spreads(self) -> None:
        self._run_collapse("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_column_collapses_and_spreads(self) -> None:
        self._run_collapse("cuda:0")


if __name__ == "__main__":
    unittest.main()
