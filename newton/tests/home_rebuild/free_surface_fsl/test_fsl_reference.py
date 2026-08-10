# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreDomain, HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.core.constants import D3Q27_DIRECTIONS
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    classify_fill,
    initialize_fsl_fields,
    link_mass_exchange,
    reconstruct_populations,
    validate_fsl_fields,
)


def _equilibrium_moments(rho: np.ndarray, velocity: tuple[float, float, float]) -> np.ndarray:
    u = np.asarray(velocity, dtype=np.float64)
    moments = np.zeros(rho.shape + (10,), dtype=np.float64)
    moments[..., 0] = rho
    moments[..., 1:4] = rho[..., None] * u
    moments[..., 4] = rho * u[0] * u[0]
    moments[..., 5] = rho * u[1] * u[1]
    moments[..., 6] = rho * u[2] * u[2]
    moments[..., 7] = rho * u[0] * u[1]
    moments[..., 8] = rho * u[0] * u[2]
    moments[..., 9] = rho * u[1] * u[2]
    return moments


def _periodic_slab_fill(shape: tuple[int, int, int]) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float64)
    fill[3:11, :, :] = 1.0
    fill[2, :, :] = 0.5
    fill[11, :, :] = 0.5
    return fill


def test_fsl_package_does_not_import_old_vof_or_optional_extensions() -> None:
    package = Path(__file__).parents[4] / "wanphys/_src/fluid/fluid_grid/home_rebuild/free_surface_fsl"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in ("home_lbm", "plic", "courant", "excess_momentum", "rigid"):
        assert forbidden not in sources.lower()


def test_fill_classification_and_periodic_interface_separation() -> None:
    fill = _periodic_slab_fill((16, 6, 1))
    flags = classify_fill(fill)
    assert np.all(flags[fill == 0.0] == int(FslCellFlag.GAS))
    assert np.all(flags[fill == 0.5] == int(FslCellFlag.INTERFACE))
    assert np.all(flags[fill == 1.0] == int(FslCellFlag.LIQUID))

    rho = np.ones(fill.shape)
    mass, excess, flags, diagnostics = initialize_fsl_fields(rho, fill)
    assert diagnostics.direct_liquid_gas_link_count == 0
    assert diagnostics.total_mass == pytest.approx(float(np.sum(fill)))
    assert np.array_equal(mass, fill)
    assert np.count_nonzero(excess) == 0

    broken = flags.copy()
    broken_fill = fill.copy()
    broken_mass = mass.copy()
    broken[2, :, :] = int(FslCellFlag.GAS)
    broken_fill[2, :, :] = 0.0
    broken_mass[2, :, :] = 0.0
    with pytest.raises(ValueError, match="direct liquid-gas links"):
        validate_fsl_fields(rho, broken_mass, broken_fill, excess, broken)


def test_static_planar_interface_has_zero_link_mass_change() -> None:
    shape = (16, 8, 1)
    fill = _periodic_slab_fill(shape)
    rho = np.ones(shape)
    mass, _, flags, _ = initialize_fsl_fields(rho, fill)
    moments = _equilibrium_moments(rho, (0.0, 0.0, 0.0))
    advected = link_mass_exchange(moments, mass, fill, flags)
    np.testing.assert_allclose(advected, mass, rtol=0.0, atol=2.0e-15)


def test_translating_interface_internal_links_conserve_total_mass() -> None:
    shape = (16, 8, 1)
    fill = _periodic_slab_fill(shape)
    rho = np.ones(shape)
    mass, _, flags, _ = initialize_fsl_fields(rho, fill)
    moments = _equilibrium_moments(rho, (0.05, 0.0, 0.0))
    advected = link_mass_exchange(moments, mass, fill, flags)
    assert float(np.sum(advected)) == pytest.approx(float(np.sum(mass)), abs=2.0e-13)
    assert np.max(np.abs(advected - mass)) > 1.0e-3


def test_full_liquid_mass_exchange_reduces_to_streamed_home_density() -> None:
    shape = (9, 7, 1)
    x = np.arange(shape[0], dtype=np.float64)[:, None, None]
    rho = np.broadcast_to(1.0 + 0.01 * np.cos(2.0 * np.pi * x / shape[0]), shape).copy()
    fill = np.ones(shape)
    mass, _, flags, _ = initialize_fsl_fields(rho, fill)
    moments = _equilibrium_moments(rho, (0.02, -0.01, 0.0))
    populations = reconstruct_populations(moments)
    expected = np.zeros(shape, dtype=np.float64)
    for q, direction in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
        expected += np.roll(
            populations[..., q],
            shift=tuple(int(component) for component in direction),
            axis=(0, 1, 2),
        )
    advected = link_mass_exchange(moments, mass, fill, flags)
    np.testing.assert_allclose(advected, expected, rtol=2.0e-13, atol=2.0e-13)


def test_device_state_initializes_exact_paper_scalars() -> None:
    model = HomeCoreModel(fluid_grid_res=(16, 6, 1), device="cpu")
    domain = HomeCoreDomain(model)
    fluid = domain.create_state()
    domain.solver.initialize_uniform_lattice(fluid)
    state = FslState(model)
    fill = _periodic_slab_fill((16, 6, 1))
    diagnostics = state.initialize_from_fill_level(fluid, fill)
    assert state.persistent_scalar_count == 4 * fluid.cell_count
    assert diagnostics.invalid_cell_count == 0
    assert state.validate(fluid).max_mass_fill_error < 1.0e-7
