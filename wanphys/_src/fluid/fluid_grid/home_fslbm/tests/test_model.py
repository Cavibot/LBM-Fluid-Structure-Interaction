# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HomeFslbmModel — parameter defaults, validation, and derived properties.

Reference
---------
- ``mrFlow3D.h:78-85``  — Create() parameter list
- ``MLFluidParam3D``     — physical parameter block
- Audit items B2, B3, B4, S6
"""

from __future__ import annotations

import pytest

from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel


class TestDefaultConstruction:
    """Verify that a model constructed with defaults has correct values."""

    def test_default_omega(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m.omega == 1.0

    def test_default_turbulence_radius(self):
        """Audit item S6: turbulence_radius must default to 3, not 4."""
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m.turbulence_radius == 3, "S6: turbulence_radius must be 3"

    def test_default_max_bubbles(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m.max_bubbles == 65536

    def test_default_surface_tension(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m.surface_tension == pytest.approx(6.0 * 4e-3)

    def test_default_atmosphere(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m.atmosphere_open is False

    def test_default_bc(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m.bc_types == (0, 0, 0, 0, 0, 0)  # all bounce-back
        assert m.bc_periodic == (False, False, False)


class TestDerivedProperties:
    """Kinematic viscosity and related derived quantities."""

    def test_tau_from_omega(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8), omega=2.0)
        assert m.tau == pytest.approx(0.5)

    def test_kinematic_viscosity(self):
        """nu = (1/3) * (tau - 0.5)"""
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8), omega=1.0)
        # tau = 1.0, nu = (1/3)*(1.0-0.5) = 1/6
        assert m.kinematic_viscosity == pytest.approx(1.0 / 6.0)

    def test_kinematic_viscosity_omega_2(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8), omega=2.0)
        # tau = 0.5, nu = 0
        assert m.kinematic_viscosity == pytest.approx(0.0)

    def test_periodic_ints(self):
        m = HomeFslbmModel(
            fluid_grid_res=(8, 8, 8),
            bc_periodic=(True, False, True),
            bc_types=(3, 3, 0, 0, 3, 3),
        )
        assert m._periodic_ints == (1, 0, 1)


class TestValidation:
    """Model validation must reject invalid configurations."""

    def test_tau_stability_guard_negative(self):
        """omega <= 0 → ValueError (tau <= 0.5 unstable)."""
        with pytest.raises(ValueError):
            HomeFslbmModel(fluid_grid_res=(8, 8, 8), omega=0.0)

    def test_tau_stability_guard_zero(self):
        with pytest.raises(ValueError):
            HomeFslbmModel(fluid_grid_res=(8, 8, 8), omega=-0.1)

    def test_grid_res_positive(self):
        with pytest.raises(ValueError):
            HomeFslbmModel(fluid_grid_res=(0, 8, 8))

    def test_periodic_bc_consistency(self):
        """bc_periodic[0] == True but bc_types[0] != 3 → ValueError."""
        with pytest.raises(ValueError):
            HomeFslbmModel(
                fluid_grid_res=(8, 8, 8),
                bc_periodic=(True, False, False),
                bc_types=(0, 0, 0, 0, 0, 0),  # not periodic
            )


class TestGridResProperties:
    """Grid resolution accessors."""

    def test_resolution_tuple(self):
        m = HomeFslbmModel(fluid_grid_res=(64, 32, 16))
        assert m.resolution == (64.0, 32.0, 16.0)

    def test_nx_ny_nz(self):
        m = HomeFslbmModel(fluid_grid_res=(64, 32, 16))
        assert m.nx == 64
        assert m.ny == 32
        assert m.nz == 16

    def test_cell_size(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8), fluid_grid_cell_size=0.5)
        assert m.cell_size == pytest.approx(0.5)
        assert m.dh == pytest.approx(0.5)


class TestDeviceSelection:
    """Device parameter handling."""

    def test_device_cpu(self, _warp):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8), device="cpu")
        assert m._device.is_cpu

    def test_device_default_is_not_none(self):
        m = HomeFslbmModel(fluid_grid_res=(8, 8, 8))
        assert m._device is not None


class TestConvenienceFactories:
    """default_32 and dam_break_64 factories."""

    def test_default_32(self):
        m = HomeFslbmModel.default_32(device="cpu")
        assert m.nx == 32 and m.ny == 32 and m.nz == 32
        assert m.turbulence_radius == 3
        assert m.omega == 1.0

    def test_dam_break_64(self):
        m = HomeFslbmModel.dam_break_64(device="cpu")
        assert m.nx == 64 and m.ny == 64 and m.nz == 64
        assert m.gravity_y < 0  # downward gravity
        assert m.surface_tension == pytest.approx(6.0 * 4e-3)
