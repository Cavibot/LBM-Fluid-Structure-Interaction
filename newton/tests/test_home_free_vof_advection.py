# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp/reference agreement for HOME-FREE link-wise mass transport."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_OPPOSITE,
    HomeFreeMassAdvector,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmState,
    advect_internal_link_mass_momentum,
    interface_home_convective_density_momentum_increment,
    home_internal_link_momentum,
    advect_mass_momentum_remap,
    reconstruct_populations,
    reference_pressure_link_momentum,
    represented_liquid_momentum,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.flags import HomeFreeCellFlag
from wanphys._src.fluid.fluid_grid.home_lbm.vof import normalize_mass_and_excess


def _reference_advect(
    moments: np.ndarray,
    mass: np.ndarray,
    fill: np.ndarray,
    excess: np.ndarray,
    flags: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> np.ndarray:
    shape = mass.shape
    populations = reconstruct_populations(moments)
    result = mass.astype(np.float64).copy()
    for index in np.ndindex(shape):
        cell_flag = int(flags[index])
        if cell_flag in (HomeFreeCellFlag.GAS, HomeFreeCellFlag.SOLID):
            continue
        for q in range(1, 27):
            source = [index[axis] - int(D3Q27_DIRECTIONS[q, axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if source[axis] < 0 or source[axis] >= shape[axis]:
                    if periodic[axis]:
                        source[axis] %= shape[axis]
                    else:
                        valid = False
            if not valid:
                continue
            source_index = tuple(source)
            source_flag = int(flags[source_index])
            result[index] += float(excess[source_index])
            if source_flag not in (HomeFreeCellFlag.LIQUID, HomeFreeCellFlag.INTERFACE):
                continue
            weight = 1.0
            if cell_flag == HomeFreeCellFlag.INTERFACE and source_flag == HomeFreeCellFlag.INTERFACE:
                weight = 0.5 * (float(fill[index]) + float(fill[source_index]))
            result[index] += weight * (
                populations[source_index + (q,)]
                - populations[index + (int(D3Q27_OPPOSITE[q]),)]
            )
    return result


def _active_neighbor_count(
    flags: np.ndarray, periodic: tuple[bool, bool, bool]
) -> np.ndarray:
    shape = flags.shape
    counts = np.zeros(shape, dtype=np.int32)
    for index in np.ndindex(shape):
        if int(flags[index]) == HomeFreeCellFlag.SOLID:
            continue
        for c in D3Q27_DIRECTIONS[1:].astype(np.int32):
            neighbor = [index[axis] + int(c[axis]) for axis in range(3)]
            valid = True
            for axis in range(3):
                if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                    if periodic[axis]:
                        neighbor[axis] %= shape[axis]
                    else:
                        valid = False
            if valid and int(flags[tuple(neighbor)]) in (
                HomeFreeCellFlag.LIQUID,
                HomeFreeCellFlag.INTERFACE,
            ):
                counts[index] += 1
    return counts


class TestHomeFreeVofAdvection(unittest.TestCase):
    def _case(self, periodic: tuple[bool, bool, bool], device: str = "cpu") -> tuple:
        shape = (5, 4, 3)
        model = HomeLbmModel(fluid_grid_res=shape, periodic=periodic, device=device)
        fluid = HomeLbmState(model)
        free = HomeFreeState(model)
        rng = np.random.default_rng(90527)
        rho = rng.uniform(0.97, 1.03, size=shape)
        velocity = rng.uniform(-0.025, 0.025, size=shape + (3,))
        moments = np.zeros(shape + (10,), dtype=np.float64)
        moments[..., 0] = rho
        moments[..., 1:4] = rho[..., None] * velocity
        moments[..., 4] = rho * velocity[..., 0] ** 2
        moments[..., 5] = rho * velocity[..., 1] ** 2
        moments[..., 6] = rho * velocity[..., 2] ** 2
        moments[..., 7] = rho * velocity[..., 0] * velocity[..., 1]
        moments[..., 8] = rho * velocity[..., 0] * velocity[..., 2]
        moments[..., 9] = rho * velocity[..., 1] * velocity[..., 2]
        fluid.moments.assign(np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1)))

        fill = np.full(shape, 0.45, dtype=np.float32)
        if not periodic[0]:
            fill[0] = 1.0
            fill[2:] = 0.0
        free.initialize_from_fill_level(fluid, fill)
        excess = np.zeros(shape, dtype=np.float32)
        excess[1, 1, 1] = 0.003
        free.excess_mass.assign(excess)
        excess_momentum = np.zeros((3,) + shape, dtype=np.float32)
        excess_momentum[:, 1, 1, 1] = (
            excess[1, 1, 1] * velocity[1, 1, 1]
        )
        free.excess_momentum.assign(excess_momentum.reshape(3, -1).reshape(-1))
        return model, fluid, free, moments, excess

    def test_warp_matches_reference_for_nonperiodic_sharp_surface(self) -> None:
        model, fluid, free, moments, excess = self._case((False, False, False))
        expected = _reference_advect(
            moments,
            free.mass.numpy(),
            free.fill_level.numpy(),
            excess,
            free.flags.numpy(),
            model.periodic,
        )

        actual = HomeFreeMassAdvector(model).advect(fluid, free).numpy()

        np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)

        counts = _active_neighbor_count(free.flags.numpy(), model.periodic)
        expected_normalized = normalize_mass_and_excess(
            moments[..., 0], expected, free.flags.numpy(), counts
        )
        actual_normalized = HomeFreeMassAdvector(model)
        actual_normalized.advect(fluid, free)
        mass, fill, excess_share, actual_counts = actual_normalized.normalize(fluid, free)
        np.testing.assert_array_equal(actual_counts.numpy(), counts)
        for actual_field, expected_field in zip(
            (mass.numpy(), fill.numpy(), excess_share.numpy()),
            expected_normalized,
            strict=True,
        ):
            np.testing.assert_allclose(actual_field, expected_field, rtol=3.0e-6, atol=3.0e-7)
        self.assertAlmostEqual(
            float(np.sum(mass.numpy() + excess_share.numpy() * counts)),
            float(np.sum(expected)),
            places=5,
        )

    def test_warp_matches_reference_with_periodic_wrapping(self) -> None:
        model, fluid, free, moments, excess = self._case((True, True, True))
        expected = _reference_advect(
            moments,
            free.mass.numpy(),
            free.fill_level.numpy(),
            excess,
            free.flags.numpy(),
            model.periodic,
        )

        actual = HomeFreeMassAdvector(model).advect(fluid, free).numpy()

        np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)

    def test_internal_link_flux_reduces_exactly_to_full_liquid_streaming(self) -> None:
        _, _, _, moments, _ = self._case((True, True, True))
        shape = moments.shape[:-1]
        density = moments[..., 0]
        fill = np.ones(shape, dtype=np.float64)
        flags = np.full(shape, int(HomeFreeCellFlag.LIQUID), dtype=np.int32)

        advected_mass, advected_momentum = advect_internal_link_mass_momentum(
            moments,
            density,
            fill,
            flags,
            periodic=(True, True, True),
        )
        populations = reconstruct_populations(moments)
        streamed = np.empty_like(populations)
        for q, c in enumerate(D3Q27_DIRECTIONS.astype(np.int32)):
            streamed[..., q] = np.roll(
                populations[..., q],
                shift=tuple(int(component) for component in c),
                axis=(0, 1, 2),
            )
        expected_density = np.sum(streamed, axis=-1)
        expected_momentum = np.einsum(
            "...q,qa->...a", streamed, D3Q27_DIRECTIONS
        )

        np.testing.assert_allclose(
            advected_mass, expected_density, rtol=2.0e-13, atol=2.0e-13
        )
        np.testing.assert_allclose(
            advected_momentum,
            expected_momentum,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_interface_link_flux_is_globally_conservative(self) -> None:
        _, _, _, moments, _ = self._case((True, True, True))
        shape = moments.shape[:-1]
        coordinates = np.indices(shape, dtype=np.float64)
        fill = 0.2 + 0.6 * (
            0.5
            + 0.25 * np.sin(2.0 * np.pi * coordinates[0] / shape[0])
            + 0.25 * np.cos(2.0 * np.pi * coordinates[1] / shape[1])
        )
        density = moments[..., 0]
        mass = density * fill
        velocity = moments[..., 1:4] / density[..., None]
        momentum = mass[..., None] * velocity
        flags = np.full(shape, int(HomeFreeCellFlag.INTERFACE), dtype=np.int32)

        advected_mass, advected_momentum = advect_internal_link_mass_momentum(
            moments,
            mass,
            fill,
            flags,
            periodic=(True, True, True),
        )

        self.assertAlmostEqual(
            float(np.sum(advected_mass, dtype=np.float64)),
            float(np.sum(mass, dtype=np.float64)),
            places=12,
        )
        np.testing.assert_allclose(
            np.sum(advected_momentum, axis=(0, 1, 2), dtype=np.float64),
            np.sum(momentum, axis=(0, 1, 2), dtype=np.float64),
            rtol=0.0,
            atol=2.0e-13,
        )

    def _assert_queue_transport_and_reference_defect(
        self,
        periodic: tuple[bool, bool, bool],
        device: str,
    ) -> None:
        model, fluid, free, moments, _ = self._case(periodic, device=device)
        excess_momentum = np.moveaxis(
            free.excess_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        (
            expected_mass,
            expected_queue_mass,
            expected_queue_momentum,
            expected_interface,
        ) = advect_mass_momentum_remap(
            moments,
            free.mass.numpy(),
            free.fill_level.numpy(),
            free.excess_mass.numpy(),
            excess_momentum,
            free.flags.numpy(),
            periodic=periodic,
        )
        advector = HomeFreeMassAdvector(model)
        advector.enable_momentum_defect_diagnostics()

        actual_mass = advector.advect(fluid, free).numpy()
        actual_queue_momentum = np.moveaxis(
            advector.incoming_excess_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )

        np.testing.assert_allclose(actual_mass, expected_mass, rtol=3.0e-6, atol=3.0e-7)
        np.testing.assert_allclose(
            advector.incoming_excess_mass.numpy(),
            expected_queue_mass,
            rtol=4.0e-6,
            atol=4.0e-7,
        )
        np.testing.assert_allclose(
            actual_queue_momentum,
            expected_queue_momentum,
            rtol=4.0e-6,
            atol=4.0e-7,
        )
        expected_home_convection = (
            interface_home_convective_density_momentum_increment(
                moments,
                free.flags.numpy(),
                periodic=periodic,
            )
        )
        assert advector.interface_home_density_increment is not None
        assert advector.interface_home_momentum_increment is not None
        assert advector.interface_momentum_candidate is not None
        assert advector.internal_link_advected_momentum is not None
        assert advector.home_internal_link_momentum is not None
        actual_home_momentum = np.moveaxis(
            advector.interface_home_momentum_increment.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        actual_interface_candidate = np.moveaxis(
            advector.interface_momentum_candidate.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        actual_internal_momentum = np.moveaxis(
            advector.internal_link_advected_momentum.numpy().reshape(
                3, *free.res
            ),
            0,
            -1,
        )
        actual_home_internal_momentum = np.moveaxis(
            advector.home_internal_link_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        _, expected_internal_momentum = advect_internal_link_mass_momentum(
            moments,
            free.mass.numpy(),
            free.fill_level.numpy(),
            free.flags.numpy(),
            periodic=periodic,
        )
        expected_home_internal_momentum = home_internal_link_momentum(
            moments,
            free.flags.numpy(),
            periodic=periodic,
        )
        np.testing.assert_allclose(
            advector.interface_home_density_increment.numpy(),
            expected_home_convection[..., 0],
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        np.testing.assert_allclose(
            actual_home_momentum,
            expected_home_convection[..., 1:4],
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        np.testing.assert_allclose(
            actual_interface_candidate,
            expected_interface,
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        np.testing.assert_allclose(
            actual_internal_momentum,
            expected_internal_momentum,
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        np.testing.assert_allclose(
            actual_home_internal_momentum,
            expected_home_internal_momentum,
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        actual_production_momentum = np.moveaxis(
            advector.advected_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        np.testing.assert_allclose(
            actual_production_momentum,
            expected_internal_momentum,
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        expected_reference_pressure = reference_pressure_link_momentum(
            free.fill_level.numpy(),
            free.flags.numpy(),
            1.0,
            periodic=periodic,
        )
        actual_reference_pressure = np.moveaxis(
            advector.reference_pressure_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        np.testing.assert_allclose(
            actual_reference_pressure,
            expected_reference_pressure,
            rtol=4.0e-5,
            atol=4.0e-8,
        )
        target_momentum = np.moveaxis(
            advector.finalize_momentum_target(fluid).numpy().reshape(3, *free.res),
            0,
            -1,
        )
        expected_target = (
            expected_internal_momentum
            + expected_queue_momentum
            + moments[..., 1:4]
            - expected_home_internal_momentum
        )
        expected_target -= expected_reference_pressure
        source_active = np.isin(
            free.flags.numpy(),
            (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
        )
        expected_target[~source_active] = 0.0
        np.testing.assert_allclose(
            target_momentum,
            expected_target,
            rtol=5.0e-5,
            atol=5.0e-8,
        )
        counts = _active_neighbor_count(free.flags.numpy(), periodic)
        initial_momentum = represented_liquid_momentum(
            moments[..., 0],
            moments[..., 1:4],
            free.mass.numpy(),
            free.excess_mass.numpy(),
            excess_momentum,
            counts,
        )
        if not np.any(free.flags.numpy() == 2):
            source_velocity = moments[..., 1:4] / moments[..., 0, None]
            expected_momentum = (
                expected_mass[..., None] * source_velocity
                + expected_queue_momentum
                - expected_queue_mass[..., None] * source_velocity
                + expected_interface
            )
            np.testing.assert_allclose(
                np.sum(expected_momentum, axis=(0, 1, 2), dtype=np.float64),
                initial_momentum,
                rtol=2.0e-13,
                atol=2.0e-13,
            )

    def test_queue_transport_and_reference_defect_match_cpu(self) -> None:
        self._assert_queue_transport_and_reference_defect((False, False, False), "cpu")
        self._assert_queue_transport_and_reference_defect((True, True, True), "cpu")

    def test_momentum_target_scales_body_force_by_transported_mass(self) -> None:
        model, fluid, free, moments, _ = self._case((True, True, True))
        model.body_acceleration = (2.0e-4, -1.0e-4, 3.0e-4)
        advector = HomeFreeMassAdvector(model)
        advector.enable_conservative_momentum_transport()
        advector.advect(fluid, free)
        target = np.moveaxis(
            advector.finalize_momentum_target(fluid).numpy().reshape(3, *free.res),
            0,
            -1,
        )
        internal = np.moveaxis(
            advector.advected_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        queue = np.moveaxis(
            advector.incoming_excess_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        home_internal = np.moveaxis(
            advector.home_internal_link_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        acceleration = np.asarray(model.lattice_acceleration)
        expected = internal + queue + moments[..., 1:4] - home_internal
        expected -= np.moveaxis(
            advector.reference_pressure_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        expected += (
            advector.advected_mass.numpy() - moments[..., 0]
        )[..., None] * acceleration

        np.testing.assert_allclose(target, expected, rtol=5.0e-5, atol=5.0e-8)

    def test_momentum_target_adds_explicit_capillary_correction(self) -> None:
        model, fluid, free, _, _ = self._case((True, True, True))
        advector = HomeFreeMassAdvector(model)
        advector.enable_conservative_momentum_transport()
        advector.advect(fluid, free)
        baseline = np.moveaxis(
            advector.finalize_momentum_target(fluid).numpy().reshape(3, *free.res),
            0,
            -1,
        ).copy()
        rng = np.random.default_rng(8675309)
        correction = rng.normal(scale=1.0e-5, size=free.res + (3,)).astype(
            np.float32
        )
        device_correction = wp.array(
            np.moveaxis(correction, -1, 0).reshape(-1),
            dtype=float,
            device=model._device,
        )
        corrected = np.moveaxis(
            advector.finalize_momentum_target(
                fluid, capillary_momentum_correction=device_correction
            )
            .numpy()
            .reshape(3, *free.res),
            0,
            -1,
        )
        active = np.isin(
            free.flags.numpy(),
            (int(HomeFreeCellFlag.LIQUID), int(HomeFreeCellFlag.INTERFACE)),
        )
        expected = baseline.copy()
        expected[active] += correction[active]
        np.testing.assert_allclose(corrected, expected, rtol=5.0e-5, atol=5.0e-8)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_queue_transport_and_reference_defect_match(self) -> None:
        self._assert_queue_transport_and_reference_defect((False, False, False), "cuda:0")
        self._assert_queue_transport_and_reference_defect((True, True, True), "cuda:0")

    def _assert_convective_correction_preserves_home_increment(
        self, device: str
    ) -> None:
        model, source, free, moments, _ = self._case(
            (True, True, True), device=device
        )
        advector = HomeFreeMassAdvector(model)
        advector.enable_momentum_defect_diagnostics()
        advector.advect(source, free)
        destination = HomeLbmState(model)
        shifted = moments.copy()
        increment = np.array((0.003, -0.002, 0.001), dtype=np.float64)
        rho = shifted[..., 0]
        old_j = shifted[..., 1:4].copy()
        shifted[..., 4] += 2.0 * increment[0] * old_j[..., 0] + rho * increment[0] ** 2
        shifted[..., 5] += 2.0 * increment[1] * old_j[..., 1] + rho * increment[1] ** 2
        shifted[..., 6] += 2.0 * increment[2] * old_j[..., 2] + rho * increment[2] ** 2
        shifted[..., 7] += (
            increment[0] * old_j[..., 1]
            + increment[1] * old_j[..., 0]
            + rho * increment[0] * increment[1]
        )
        shifted[..., 8] += (
            increment[0] * old_j[..., 2]
            + increment[2] * old_j[..., 0]
            + rho * increment[0] * increment[2]
        )
        shifted[..., 9] += (
            increment[1] * old_j[..., 2]
            + increment[2] * old_j[..., 1]
            + rho * increment[1] * increment[2]
        )
        shifted[..., 1:4] += rho[..., None] * increment
        destination.moments.assign(
            np.ascontiguousarray(shifted.reshape(-1, 10).T.reshape(-1))
        )

        transported = advector.advected_mass.numpy().astype(np.float64)
        queued_mass = advector.incoming_excess_mass.numpy().astype(np.float64)
        queued_momentum = np.moveaxis(
            advector.incoming_excess_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        ).astype(np.float64)
        velocity_before = shifted[..., 1:4] / shifted[..., 0, None]
        expected = transported[..., None] * velocity_before
        expected += queued_momentum - queued_mass[..., None] * velocity_before
        central_before = shifted[..., 4:10].copy()
        central_before[..., 0] -= shifted[..., 1] ** 2 / rho
        central_before[..., 1] -= shifted[..., 2] ** 2 / rho
        central_before[..., 2] -= shifted[..., 3] ** 2 / rho
        central_before[..., 3] -= shifted[..., 1] * shifted[..., 2] / rho
        central_before[..., 4] -= shifted[..., 1] * shifted[..., 3] / rho
        central_before[..., 5] -= shifted[..., 2] * shifted[..., 3] / rho

        advector.apply_queue_momentum_correction(destination)

        assert advector.home_destination_density is not None
        assert advector.home_destination_momentum is not None
        captured_momentum = np.moveaxis(
            advector.home_destination_momentum.numpy().reshape(3, *free.res),
            0,
            -1,
        )
        np.testing.assert_allclose(
            advector.home_destination_density.numpy(),
            shifted[..., 0].astype(np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            captured_momentum,
            shifted[..., 1:4].astype(np.float32),
            rtol=0.0,
            atol=0.0,
        )
        after = np.moveaxis(
            destination.moments.numpy().reshape(10, *free.res), 0, -1
        ).astype(np.float64)
        velocity_after = after[..., 1:4] / after[..., 0, None]
        np.testing.assert_allclose(
            transported[..., None] * velocity_after,
            expected,
            rtol=5.0e-6,
            atol=5.0e-7,
        )
        central_after = after[..., 4:10].copy()
        central_after[..., 0] -= after[..., 1] ** 2 / after[..., 0]
        central_after[..., 1] -= after[..., 2] ** 2 / after[..., 0]
        central_after[..., 2] -= after[..., 3] ** 2 / after[..., 0]
        central_after[..., 3] -= after[..., 1] * after[..., 2] / after[..., 0]
        central_after[..., 4] -= after[..., 1] * after[..., 3] / after[..., 0]
        central_after[..., 5] -= after[..., 2] * after[..., 3] / after[..., 0]
        np.testing.assert_allclose(
            central_after,
            central_before,
            rtol=5.0e-6,
            atol=5.0e-7,
        )

    def test_convective_correction_preserves_home_velocity_increment(self) -> None:
        self._assert_convective_correction_preserves_home_increment("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_convective_correction_preserves_home_velocity_increment(self) -> None:
        self._assert_convective_correction_preserves_home_increment("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_cuda_matches_reference_with_periodic_wrapping(self) -> None:
        model, fluid, free, moments, excess = self._case(
            (True, True, True), device="cuda:0"
        )
        expected = _reference_advect(
            moments,
            free.mass.numpy(),
            free.fill_level.numpy(),
            excess,
            free.flags.numpy(),
            model.periodic,
        )

        actual = HomeFreeMassAdvector(model).advect(fluid, free).numpy()

        np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)

    def test_direct_liquid_gas_link_is_a_hard_error(self) -> None:
        shape = (3, 2, 2)
        model = HomeLbmModel(fluid_grid_res=shape, periodic=(False, False, False), device="cpu")
        fluid = HomeLbmState(model)
        moments = np.zeros((10, int(np.prod(shape))), dtype=np.float32)
        moments[0] = 1.0
        fluid.moments.assign(moments.reshape(-1))
        free = HomeFreeState(model)
        fill = np.zeros(shape, dtype=np.float32)
        fill[0] = 1.0
        free.initialize_from_fill_level(fluid, fill, validate_topology=False)

        with self.assertRaisesRegex(RuntimeError, "direct liquid-gas"):
            HomeFreeMassAdvector(model).advect(fluid, free)


if __name__ == "__main__":
    unittest.main()
