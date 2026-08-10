# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from wanphys.fluid import (
    HomeFreeBackend,
    HomeFreeDomain,
    HomeFreeLegacyDomain,
    HomeLbmModel,
    create_home_free_domain,
)
from wanphys._src.fluid.fluid_grid.home_lbm import HomeFreeGeometricDomain


class TestHomeFreeBackend(unittest.TestCase):
    def _model(self) -> HomeLbmModel:
        return HomeLbmModel(
            fluid_grid_res=(4, 4, 4),
            fluid_grid_cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            device="cpu",
        )

    def test_public_default_is_geometric(self) -> None:
        self.assertIs(HomeFreeDomain, HomeFreeGeometricDomain)
        domain = create_home_free_domain(self._model(), project_courant=False)
        self.assertIs(type(domain), HomeFreeGeometricDomain)

    def test_explicit_legacy_backend_remains_available(self) -> None:
        domain = create_home_free_domain(
            self._model(), backend=HomeFreeBackend.LEGACY
        )
        self.assertIs(type(domain), HomeFreeLegacyDomain)

    def test_unknown_backend_fails_without_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown HOME-Free backend"):
            create_home_free_domain(self._model(), backend="approximate")


if __name__ == "__main__":
    unittest.main()
