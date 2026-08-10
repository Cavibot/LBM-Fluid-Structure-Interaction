# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference-contract tests for geometric PLIC topology resolution."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm.model import HomeLbmModel
from wanphys._src.fluid.fluid_grid.home_lbm.state import HomeLbmState
from wanphys._src.fluid.fluid_grid.home_lbm.vof.flags import HomeFreeCellFlag
from wanphys._src.fluid.fluid_grid.home_lbm.vof.geometric_topology import (
    HomeFreeGeometricTopologyResolver,
    resolve_geometric_plic_topology,
)
from wanphys._src.fluid.fluid_grid.home_lbm.vof.state import HomeFreeState


GAS = int(HomeFreeCellFlag.GAS)
INTERFACE = int(HomeFreeCellFlag.INTERFACE)
LIQUID = int(HomeFreeCellFlag.LIQUID)
SOLID = int(HomeFreeCellFlag.SOLID)


def _equilibrium_moments(rho: float, velocity: np.ndarray) -> np.ndarray:
    ux, uy, uz = velocity
    return np.asarray(
        (
            rho,
            rho * ux,
            rho * uy,
            rho * uz,
            rho * ux * ux,
            rho * uy * uy,
            rho * uz * uz,
            rho * ux * uy,
            rho * ux * uz,
            rho * uy * uz,
        ),
        dtype=np.float64,
    )


def _moments(shape: tuple[int, int, int]) -> np.ndarray:
    result = np.empty(shape + (10,), dtype=np.float64)
    for index in np.ndindex(shape):
        rho = 0.98 + 0.007 * index[0] + 0.003 * index[1] + 0.002 * index[2]
        velocity = np.asarray(
            (0.012 + 0.001 * index[1], -0.006 + 0.0005 * index[2], 0.004),
            dtype=np.float64,
        )
        result[index] = _equilibrium_moments(rho, velocity)
    return result


def _upload_moments(state: HomeLbmState, moments: np.ndarray) -> None:
    state.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    )


def _download_moments(state: HomeLbmState) -> np.ndarray:
    return (
        state.moments.numpy()
        .reshape(10, -1)
        .T.reshape(state.res + (10,))
        .astype(np.float64)
    )


class TestHomeFreeGeometricTopologyReference(unittest.TestCase):
    def test_advancing_interface_uses_persistent_donors_and_closes_mass(self) -> None:
        shape = (5, 3, 3)
        source = np.full(shape, GAS, dtype=np.int32)
        source[0] = LIQUID
        source[1] = INTERFACE
        fill = np.zeros(shape, dtype=np.float64)
        fill[0] = 1.0
        fill[1] = 0.8
        fill[2] = 0.2
        moments = _moments(shape)
        moments[2] = -17.0

        result = resolve_geometric_plic_topology(moments, source, fill)

        np.testing.assert_array_equal(result.flags[0], LIQUID)
        np.testing.assert_array_equal(result.flags[1:3], INTERFACE)
        np.testing.assert_array_equal(result.flags[3:], GAS)
        self.assertEqual(result.diagnostics.gas_to_interface_count, 9)
        self.assertEqual(result.diagnostics.liquid_to_interface_count, 0)
        self.assertEqual(result.diagnostics.interface_to_gas_count, 0)
        self.assertEqual(result.diagnostics.interface_to_liquid_count, 0)
        for j in range(shape[1]):
            for k in range(shape[2]):
                donors = moments[
                    1,
                    max(0, j - 1) : min(shape[1], j + 2),
                    max(0, k - 1) : min(shape[2], k + 2),
                ].reshape(-1, 10)
                expected_rho = float(np.mean(donors[:, 0]))
                expected_velocity = np.mean(
                    donors[:, 1:4] / donors[:, 0, None], axis=0
                )
                np.testing.assert_allclose(
                    result.moments[2, j, k],
                    _equilibrium_moments(expected_rho, expected_velocity),
                    rtol=0.0,
                    atol=2.0e-15,
                )
                self.assertEqual(result.donor_count[2, j, k], donors.shape[0])
        active = np.isin(result.flags, (INTERFACE, LIQUID))
        np.testing.assert_allclose(
            result.mass[active],
            result.moments[..., 0][active] * result.fill_level[active],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(result.mass[~active], 0.0)

    def test_all_supported_transition_classes_are_counted(self) -> None:
        shape = (7, 3, 3)
        source_layers = (LIQUID, INTERFACE, GAS, GAS, INTERFACE, LIQUID, INTERFACE)
        source = np.empty(shape, dtype=np.int32)
        for i, flag in enumerate(source_layers):
            source[i] = flag
        fill_layers = (1.0, 1.0, 0.25, 0.0, 0.0, 0.65, 0.5)
        fill = np.empty(shape, dtype=np.float64)
        for i, value in enumerate(fill_layers):
            fill[i] = value

        result = resolve_geometric_plic_topology(_moments(shape), source, fill)

        expected_layers = (LIQUID, LIQUID, INTERFACE, GAS, GAS, INTERFACE, INTERFACE)
        for i, flag in enumerate(expected_layers):
            np.testing.assert_array_equal(result.flags[i], flag)
        self.assertEqual(result.diagnostics.gas_to_interface_count, 9)
        self.assertEqual(result.diagnostics.liquid_to_interface_count, 9)
        self.assertEqual(result.diagnostics.interface_to_gas_count, 9)
        self.assertEqual(result.diagnostics.interface_to_liquid_count, 9)

    def test_endpoint_snapping_reports_volume_change(self) -> None:
        shape = (3, 2, 2)
        source = np.full(shape, INTERFACE, dtype=np.int32)
        fill = np.full(shape, 0.5, dtype=np.float64)
        fill[0] = 1.0e-7
        fill[2] = 1.0 - 2.0e-7

        result = resolve_geometric_plic_topology(
            _moments(shape), source, fill, endpoint_tolerance=4.0e-7
        )

        np.testing.assert_array_equal(result.fill_level[0], 0.0)
        np.testing.assert_array_equal(result.fill_level[2], 1.0)
        self.assertEqual(result.diagnostics.snapped_cell_count, 8)
        self.assertAlmostEqual(result.diagnostics.snapped_volume_delta, 4.0e-7)
        self.assertEqual(result.diagnostics.separation_closure_cell_count, 0)

    def test_positive_endpoint_sliver_preserves_d3q27_separation(self) -> None:
        shape = (3, 3, 3)
        source = np.full(shape, INTERFACE, dtype=np.int32)
        source[1, 1, 1] = LIQUID
        fill = np.full(shape, 0.5, dtype=np.float64)
        fill[0, 0, 0] = 1.5e-7
        fill[1, 1, 1] = 1.0

        result = resolve_geometric_plic_topology(
            _moments(shape), source, fill, endpoint_tolerance=4.0e-7
        )

        self.assertEqual(result.flags[0, 0, 0], INTERFACE)
        self.assertEqual(result.fill_level[0, 0, 0], 4.0e-7)
        self.assertEqual(result.diagnostics.interface_to_gas_count, 0)
        self.assertEqual(result.diagnostics.separation_closure_cell_count, 1)
        self.assertEqual(result.diagnostics.snapped_cell_count, 1)
        self.assertAlmostEqual(
            result.diagnostics.snapped_volume_delta, 2.5e-7
        )

    def test_forbids_one_step_gas_to_liquid_jump(self) -> None:
        source = np.full((3, 2, 2), GAS, dtype=np.int32)
        fill = np.zeros(source.shape, dtype=np.float64)
        fill[1] = 1.0
        with self.assertRaisesRegex(RuntimeError, "gas-to-liquid"):
            resolve_geometric_plic_topology(_moments(source.shape), source, fill)

    def test_forbids_one_step_liquid_to_gas_jump(self) -> None:
        source = np.full((3, 2, 2), LIQUID, dtype=np.int32)
        fill = np.ones(source.shape, dtype=np.float64)
        fill[1] = 0.0
        with self.assertRaisesRegex(RuntimeError, "liquid-to-gas"):
            resolve_geometric_plic_topology(_moments(source.shape), source, fill)

    def test_fresh_interface_without_persistent_donor_fails(self) -> None:
        source = np.full((3, 3, 3), GAS, dtype=np.int32)
        fill = np.zeros(source.shape, dtype=np.float64)
        fill[1, 1, 1] = 0.25
        with self.assertRaisesRegex(RuntimeError, "no persistent fluid donor"):
            resolve_geometric_plic_topology(_moments(source.shape), source, fill)

    def test_direct_liquid_gas_target_link_fails(self) -> None:
        source = np.full((2, 2, 2), GAS, dtype=np.int32)
        source[0] = LIQUID
        fill = np.zeros(source.shape, dtype=np.float64)
        fill[0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "direct liquid-gas link"):
            resolve_geometric_plic_topology(_moments(source.shape), source, fill)

    def test_liquid_volume_in_solid_fails(self) -> None:
        source = np.full((2, 2, 2), SOLID, dtype=np.int32)
        fill = np.zeros(source.shape, dtype=np.float64)
        fill[0, 0, 0] = 1.0e-3
        with self.assertRaisesRegex(RuntimeError, "solid cells"):
            resolve_geometric_plic_topology(_moments(source.shape), source, fill)


class TestHomeFreeGeometricTopologyWarp(unittest.TestCase):
    @staticmethod
    def _advancing_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (5, 3, 3)
        source = np.full(shape, GAS, dtype=np.int32)
        source[0] = LIQUID
        source[1] = INTERFACE
        source_fill = np.zeros(shape, dtype=np.float64)
        source_fill[0] = 1.0
        source_fill[1] = 0.8
        transported = source_fill.copy()
        transported[1] = 0.8
        transported[2] = 0.2
        moments = _moments(shape)
        moments[2] = -17.0
        return moments, source, source_fill, transported

    def _matches_reference(self, device: str) -> None:
        moments, source_flags, source_fill, transported = self._advancing_case()
        expected = resolve_geometric_plic_topology(
            moments, source_flags, transported
        )
        model = HomeLbmModel(
            fluid_grid_res=source_flags.shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill.astype(np.float32))
        source.mass.assign(
            (moments[..., 0] * source_fill).astype(np.float32)
        )
        destination = HomeFreeState(model)
        transported_device = wp.array(
            transported.astype(np.float32), dtype=float, device=device
        )
        resolver = HomeFreeGeometricTopologyResolver(model)

        diagnostics = resolver.resolve(
            fluid, source, transported_device, destination
        )
        wp.synchronize_device(device)

        self.assertEqual(
            diagnostics.gas_to_interface_count,
            expected.diagnostics.gas_to_interface_count,
        )
        self.assertEqual(
            diagnostics.liquid_to_interface_count,
            expected.diagnostics.liquid_to_interface_count,
        )
        self.assertEqual(
            diagnostics.interface_to_gas_count,
            expected.diagnostics.interface_to_gas_count,
        )
        self.assertEqual(
            diagnostics.interface_to_liquid_count,
            expected.diagnostics.interface_to_liquid_count,
        )
        self.assertEqual(
            diagnostics.snapped_cell_count,
            expected.diagnostics.snapped_cell_count,
        )
        self.assertEqual(
            diagnostics.separation_closure_cell_count,
            expected.diagnostics.separation_closure_cell_count,
        )
        self.assertAlmostEqual(
            diagnostics.snapped_volume_delta,
            expected.diagnostics.snapped_volume_delta,
            delta=1.0e-13,
        )
        np.testing.assert_array_equal(destination.flags.numpy(), expected.flags)
        np.testing.assert_allclose(
            destination.fill_level.numpy(), expected.fill_level, rtol=0.0, atol=2.0e-8
        )
        np.testing.assert_allclose(
            destination.mass.numpy(), expected.mass, rtol=4.0e-6, atol=4.0e-7
        )
        np.testing.assert_array_equal(
            resolver.donor_count.numpy(), expected.donor_count
        )
        np.testing.assert_allclose(
            _download_moments(fluid), expected.moments, rtol=4.0e-6, atol=4.0e-7
        )
        np.testing.assert_array_equal(destination.excess_mass.numpy(), 0.0)
        np.testing.assert_array_equal(destination.excess_momentum.numpy(), 0.0)
        destination.validate(fluid)

    def test_warp_matches_reference_on_cpu(self) -> None:
        self._matches_reference("cpu")

    def _endpoint_closure_matches_reference(self, device: str) -> None:
        shape = (3, 3, 3)
        source_flags = np.full(shape, INTERFACE, dtype=np.int32)
        source_flags[1, 1, 1] = LIQUID
        source_fill = np.full(shape, 0.5, dtype=np.float32)
        source_fill[0, 0, 0] = 1.0e-6
        source_fill[1, 1, 1] = 1.0
        transported = source_fill.copy()
        transported[0, 0, 0] = 1.5e-7
        moments = _moments(shape)
        expected = resolve_geometric_plic_topology(
            moments,
            source_flags,
            transported,
            endpoint_tolerance=4.0e-7,
        )
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))
        destination = HomeFreeState(model)
        resolver = HomeFreeGeometricTopologyResolver(model)

        diagnostics = resolver.resolve(
            fluid,
            source,
            wp.array(transported, dtype=float, device=device),
            destination,
        )

        self.assertEqual(
            (
                diagnostics.gas_to_interface_count,
                diagnostics.liquid_to_interface_count,
                diagnostics.interface_to_gas_count,
                diagnostics.interface_to_liquid_count,
                diagnostics.snapped_cell_count,
                diagnostics.separation_closure_cell_count,
            ),
            (
                expected.diagnostics.gas_to_interface_count,
                expected.diagnostics.liquid_to_interface_count,
                expected.diagnostics.interface_to_gas_count,
                expected.diagnostics.interface_to_liquid_count,
                expected.diagnostics.snapped_cell_count,
                expected.diagnostics.separation_closure_cell_count,
            ),
        )
        self.assertAlmostEqual(
            diagnostics.snapped_volume_delta,
            expected.diagnostics.snapped_volume_delta,
            delta=1.0e-13,
        )
        np.testing.assert_array_equal(destination.flags.numpy(), expected.flags)
        np.testing.assert_allclose(
            destination.fill_level.numpy(),
            expected.fill_level,
            rtol=0.0,
            atol=2.0e-8,
        )

    def test_warp_endpoint_closure_matches_reference_on_cpu(self) -> None:
        self._endpoint_closure_matches_reference("cpu")

    def test_transported_independent_mass_is_committed_exactly(self) -> None:
        shape = (4, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        fill = np.full(shape, 0.5, dtype=np.float32)
        source.flags.assign(np.full(shape, INTERFACE, dtype=np.int32))
        source.fill_level.assign(fill)
        source.mass.assign((moments[..., 0] * fill).astype(np.float32))
        concentration = np.linspace(0.7, 1.3, num=np.prod(shape)).reshape(shape)
        transported_mass = (concentration * fill).astype(np.float32)
        destination = HomeFreeState(model)

        HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(fill, dtype=float, device="cpu"),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device="cpu"
            ),
        )

        np.testing.assert_array_equal(destination.mass.numpy(), transported_mass)
        self.assertGreater(
            float(np.max(np.abs(transported_mass - moments[..., 0] * fill))),
            0.1,
        )

    def _independent_mass_near_endpoint_stays_interface(self, device: str) -> None:
        shape = (4, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(np.full(shape, INTERFACE, dtype=np.int32))
        source_fill = np.full(shape, 0.5, dtype=np.float32)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))
        transported_fill = source_fill.copy()
        transported_fill[0, 0, 0] = np.float32(3.5762787e-7)
        transported_fill[3, 2, 2] = np.float32(1.0 - 3.5762787e-7)
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        destination = HomeFreeState(model)

        diagnostics = HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device=device
            ),
        )

        np.testing.assert_array_equal(
            destination.flags.numpy(),
            np.full(shape, INTERFACE, dtype=np.int32),
        )
        resolved_fill = destination.fill_level.numpy()
        lower_endpoint = np.float32(4.0e-7)
        upper_endpoint = np.float32(1.0 - 4.0e-7)
        self.assertGreaterEqual(float(resolved_fill[0, 0, 0]), lower_endpoint)
        self.assertLess(float(resolved_fill[0, 0, 0]), lower_endpoint + 2.0e-7)
        self.assertLessEqual(float(resolved_fill[3, 2, 2]), upper_endpoint)
        self.assertGreater(
            float(resolved_fill[3, 2, 2]), upper_endpoint - 2.0e-7
        )
        self.assertAlmostEqual(
            float(np.sum(resolved_fill, dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=2.0e-7,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=2.0e-7,
        )
        self.assertGreaterEqual(diagnostics.snapped_cell_count, 2)

    def test_independent_mass_near_endpoint_stays_interface_on_cpu(self) -> None:
        self._independent_mass_near_endpoint_stays_interface("cpu")

    def _sub_ulp_closure_uses_representable_partner(self, device: str) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source_flags = np.full(shape, INTERFACE, dtype=np.int32)
        source_flags[0, 0, 0] = LIQUID
        source_fill = np.full(shape, 0.5, dtype=np.float32)
        source_fill[0, 0, 0] = 1.0
        source_fill[2, 2, 2] = np.float32(8.0e-7)
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))

        endpoint = np.float32(4.0e-7)
        sub_endpoint = endpoint
        for _ in range(5):
            sub_endpoint = np.nextafter(
                sub_endpoint, np.float32(0.0), dtype=np.float32
            )
        transported_fill = source_fill.copy()
        transported_fill[1, 1, 1] = sub_endpoint
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        destination = HomeFreeState(model)

        HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device=device
            ),
        )
        wp.synchronize_device(device)

        resolved_fill = destination.fill_level.numpy()
        self.assertGreaterEqual(float(resolved_fill[1, 1, 1]), float(endpoint))
        self.assertLess(float(resolved_fill[1, 1, 1]), float(endpoint) + 1.0e-7)
        self.assertEqual(float(resolved_fill[0, 0, 0]), 1.0)
        self.assertEqual(float(resolved_fill[2, 2, 2]), np.float32(8.0e-7))
        unchanged_half = np.ones(shape, dtype=bool)
        unchanged_half[0, 0, 0] = False
        unchanged_half[1, 1, 1] = False
        unchanged_half[2, 2, 2] = False
        half_values = resolved_fill[unchanged_half]
        self.assertTrue(np.any(half_values < 0.5))
        self.assertTrue(np.all(half_values <= 0.5))
        self.assertAlmostEqual(
            float(np.sum(resolved_fill, dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=5.0e-8,
        )

    def test_sub_ulp_closure_uses_representable_partner_on_cpu(self) -> None:
        self._sub_ulp_closure_uses_representable_partner("cpu")

    def _aggregate_endpoint_closures_respect_partner_capacity(
        self, device: str
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (5, 5, 5)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            device=device,
        )
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source_flags = np.full(shape, GAS, dtype=np.int32)
        source_fill = np.zeros(shape, dtype=np.float32)
        upper_endpoint = np.float32(1.0 - 4.0e-7)
        target = (0, 0, 0)
        sources = (
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 0),
            (4, 0, 0),
            (4, 1, 0),
            (0, 4, 0),
            (1, 4, 0),
        )
        source_flags[target] = INTERFACE
        source_fill[target] = upper_endpoint
        transported_fill = source_fill.copy()
        above_endpoint = np.nextafter(
            upper_endpoint, np.float32(1.0), dtype=np.float32
        )
        for position in sources:
            source_flags[position] = INTERFACE
            source_fill[position] = upper_endpoint
            transported_fill[position] = above_endpoint
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        destination = HomeFreeState(model)

        HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device=device
            ),
        )
        wp.synchronize_device(device)

        resolved_fill = destination.fill_level.numpy()
        self.assertEqual(float(resolved_fill[target]), float(upper_endpoint))
        self.assertLessEqual(float(np.max(resolved_fill)), 1.0)
        self.assertGreater(np.count_nonzero(resolved_fill > 0.0), len(sources) + 1)
        self.assertAlmostEqual(
            float(np.sum(resolved_fill, dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=2.0e-6,
        )
        return resolved_fill.copy(), destination.mass.numpy().copy()

    def test_aggregate_endpoint_closures_respect_partner_capacity_on_cpu(self) -> None:
        self._aggregate_endpoint_closures_respect_partner_capacity("cpu")

    def _regular_closure_aggregate_may_reach_endpoint(
        self, device: str
    ) -> None:
        shape = (3, 3, 3)
        target = (1, 1, 1)
        sources = ((0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 1, 1))
        source_flags = np.full(shape, GAS, dtype=np.int32)
        source_flags[target] = INTERFACE
        source_fill = np.zeros(shape, dtype=np.float32)
        source_fill[target] = np.float32(1.0 - 4.0e-7)
        transported_fill = source_fill.copy()
        for index in sources:
            transported_fill[index] = np.float32(1.0e-7)
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        moments = _moments(shape)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill)
        source.mass.assign((source_fill * moments[..., 0]).astype(np.float32))
        destination = HomeFreeState(model)
        resolver = HomeFreeGeometricTopologyResolver(model)

        resolver.resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device=device
            ),
        )
        wp.synchronize_device(device)

        resolved_fill = destination.fill_level.numpy()
        self.assertEqual(int(destination.flags.numpy()[target]), INTERFACE)
        self.assertGreater(float(resolved_fill[target]), 0.999)
        self.assertLess(float(resolved_fill[target]), 1.0)
        self.assertAlmostEqual(
            float(np.sum(destination.fill_level.numpy(), dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=2.0e-7,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=2.0e-7,
        )

    def _periodic_closure_is_spanwise_translation_equivariant(
        self, device: str
    ) -> None:
        shape = (3, 4, 3)
        source_flags = np.full(shape, GAS, dtype=np.int32)
        source_fill = np.zeros(shape, dtype=np.float32)
        source_flags[1, :, 0] = INTERFACE
        source_fill[1, :, 0] = 0.5
        transported_fill = source_fill.copy()
        transported_fill[1, 0, 1] = np.float32(1.0e-7)
        moments = _moments(shape)

        def resolve(fields_shift: int) -> tuple[np.ndarray, np.ndarray]:
            flags = np.roll(source_flags, fields_shift, axis=1)
            fill = np.roll(source_fill, fields_shift, axis=1)
            transported = np.roll(transported_fill, fields_shift, axis=1)
            shifted_moments = np.roll(moments, fields_shift, axis=1)
            transported_mass = (
                transported * shifted_moments[..., 0].astype(np.float32)
            ).astype(np.float32)
            model = HomeLbmModel(
                fluid_grid_res=shape,
                periodic=(False, True, False),
                device=device,
            )
            fluid = HomeLbmState(model)
            _upload_moments(fluid, shifted_moments)
            source = HomeFreeState(model)
            source.flags.assign(flags)
            source.fill_level.assign(fill)
            source.mass.assign(
                (fill * shifted_moments[..., 0]).astype(np.float32)
            )
            destination = HomeFreeState(model)
            HomeFreeGeometricTopologyResolver(model).resolve(
                fluid,
                source,
                wp.array(transported, dtype=float, device=device),
                destination,
                transported_mass=wp.array(
                    transported_mass, dtype=float, device=device
                ),
            )
            wp.synchronize_device(device)
            return destination.fill_level.numpy(), destination.mass.numpy()

        base_fill, base_mass = resolve(0)
        shifted_fill, shifted_mass = resolve(1)
        np.testing.assert_array_equal(shifted_fill, np.roll(base_fill, 1, axis=1))
        np.testing.assert_array_equal(shifted_mass, np.roll(base_mass, 1, axis=1))

    def test_regular_closure_aggregate_may_reach_endpoint_on_cpu(self) -> None:
        self._regular_closure_aggregate_may_reach_endpoint("cpu")

    def test_periodic_closure_is_spanwise_translation_equivariant_on_cpu(self) -> None:
        self._periodic_closure_is_spanwise_translation_equivariant("cpu")

    def _regular_closure_chain_preserves_volume_and_mass(self, device: str) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device=device)
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source_flags = np.full(shape, INTERFACE, dtype=np.int32)
        source_fill = np.full(shape, 0.5, dtype=np.float32)
        thin_interface = (1, 1, 1)
        near_liquid = (1, 1, 2)
        source_flags[near_liquid] = LIQUID
        source_fill[near_liquid] = 1.0
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))

        transported_fill = source_fill.copy()
        transported_fill[thin_interface] = np.float32(1.0e-7)
        transported_fill[near_liquid] = np.float32(1.0 - 1.0e-7)
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        destination = HomeFreeState(model)

        HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device=device
            ),
        )
        wp.synchronize_device(device)

        resolved_fill = destination.fill_level.numpy()
        self.assertEqual(int(destination.flags.numpy()[thin_interface]), INTERFACE)
        self.assertEqual(int(destination.flags.numpy()[near_liquid]), INTERFACE)
        self.assertGreaterEqual(float(resolved_fill[thin_interface]), 4.0e-7)
        self.assertLessEqual(float(resolved_fill[near_liquid]), 1.0 - 4.0e-7)
        self.assertAlmostEqual(
            float(np.sum(resolved_fill, dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=2.0e-7,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=2.0e-6,
        )

    def test_regular_closure_chain_preserves_volume_and_mass_on_cpu(self) -> None:
        self._regular_closure_chain_preserves_volume_and_mass("cpu")

    def _checkerboard_endpoint_collapse_preserves_separation(
        self, device: str
    ) -> None:
        shape = (2, 2, 2)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(np.full(shape, INTERFACE, dtype=np.int32))
        source.fill_level.assign(np.full(shape, 0.5, dtype=np.float32))
        source.mass.assign(np.full(shape, 0.5, dtype=np.float32))
        transported_fill = (
            np.indices(shape, dtype=np.int32).sum(axis=0) % 2
        ).astype(np.float32)
        destination = HomeFreeState(model)

        diagnostics = HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_fill, dtype=float, device=device
            ),
        )

        resolved_fill = destination.fill_level.numpy()
        resolved_flags = destination.flags.numpy()
        self.assertFalse(np.any(resolved_flags == GAS))
        self.assertTrue(np.all(np.isin(resolved_flags, (INTERFACE, LIQUID))))
        interface_fill = resolved_fill[resolved_flags == INTERFACE]
        self.assertGreaterEqual(float(np.min(interface_fill)), 4.0e-7)
        self.assertLessEqual(float(np.max(interface_fill)), 1.0 - 4.0e-7)
        self.assertAlmostEqual(
            float(np.sum(resolved_fill, dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            places=6,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            places=6,
        )
        self.assertGreater(diagnostics.separation_closure_cell_count, 0)

    def test_checkerboard_endpoint_collapse_preserves_separation_on_cpu(self) -> None:
        self._checkerboard_endpoint_collapse_preserves_separation("cpu")

    def _radius_three_closure_matching_reaches_partner(self, device: str) -> None:
        shape = (7, 7, 7)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device=device,
        )
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        upper_endpoint = np.float32(1.0 - 4.0e-7)
        source_fill = np.full(shape, upper_endpoint, dtype=np.float32)
        source_flags = np.full(shape, INTERFACE, dtype=np.int32)
        closure_source = (3, 3, 3)
        distant_partner = (0, 3, 3)
        source_fill[distant_partner] = np.float32(0.5)
        source.flags.assign(source_flags)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))

        transported_fill = source_fill.copy()
        transported_fill[closure_source] = np.nextafter(
            upper_endpoint, np.float32(1.0), dtype=np.float32
        )
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        destination = HomeFreeState(model)

        HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device=device),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device=device
            ),
        )
        wp.synchronize_device(device)

        resolved_fill = destination.fill_level.numpy()
        self.assertEqual(
            float(resolved_fill[closure_source]), float(upper_endpoint)
        )
        self.assertGreater(float(resolved_fill[distant_partner]), 0.5)
        changed = np.argwhere(resolved_fill != transported_fill)
        self.assertEqual(changed.shape[0], 2)
        self.assertAlmostEqual(
            float(np.sum(resolved_fill, dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=2.0e-6,
        )

    def test_radius_three_closure_matching_reaches_partner_on_cpu(self) -> None:
        self._radius_three_closure_matching_reaches_partner("cpu")

    def test_independent_mass_conservatively_removes_sub_half_floor_sliver(self) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(np.full(shape, INTERFACE, dtype=np.int32))
        source_fill = np.full(shape, 0.5, dtype=np.float32)
        source.fill_level.assign(source_fill)
        source.mass.assign((moments[..., 0] * source_fill).astype(np.float32))
        transported_fill = source_fill.copy()
        transported_fill[1, 1, 1] = np.float32(1.0e-11)
        transported_mass = (transported_fill * np.float32(1.001)).astype(
            np.float32
        )
        destination = HomeFreeState(model)

        HomeFreeGeometricTopologyResolver(model).resolve(
            fluid,
            source,
            wp.array(transported_fill, dtype=float, device="cpu"),
            destination,
            transported_mass=wp.array(
                transported_mass, dtype=float, device="cpu"
            ),
        )

        self.assertEqual(int(destination.flags.numpy()[1, 1, 1]), GAS)
        self.assertEqual(float(destination.fill_level.numpy()[1, 1, 1]), 0.0)
        self.assertEqual(float(destination.mass.numpy()[1, 1, 1]), 0.0)
        self.assertAlmostEqual(
            float(np.sum(destination.fill_level.numpy(), dtype=np.float64)),
            float(np.sum(transported_fill, dtype=np.float64)),
            delta=2.0e-8,
        )
        self.assertAlmostEqual(
            float(np.sum(destination.mass.numpy(), dtype=np.float64)),
            float(np.sum(transported_mass, dtype=np.float64)),
            delta=2.0e-8,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_independent_mass_near_endpoint_stays_interface_on_cuda(self) -> None:
        self._independent_mass_near_endpoint_stays_interface("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_sub_ulp_closure_uses_representable_partner_on_cuda(self) -> None:
        self._sub_ulp_closure_uses_representable_partner("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_aggregate_endpoint_closures_respect_partner_capacity_on_cuda(self) -> None:
        self._aggregate_endpoint_closures_respect_partner_capacity("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_regular_closure_aggregate_may_reach_endpoint_on_cuda(self) -> None:
        self._regular_closure_aggregate_may_reach_endpoint("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_periodic_closure_is_spanwise_translation_equivariant_on_cuda(self) -> None:
        self._periodic_closure_is_spanwise_translation_equivariant("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_radius_three_closure_matching_reaches_partner_on_cuda(self) -> None:
        self._radius_three_closure_matching_reaches_partner("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_regular_closure_chain_preserves_volume_and_mass_on_cuda(self) -> None:
        self._regular_closure_chain_preserves_volume_and_mass("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_checkerboard_endpoint_collapse_preserves_separation_on_cuda(self) -> None:
        self._checkerboard_endpoint_collapse_preserves_separation("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_endpoint_matching_is_bitwise_deterministic_on_cuda(self) -> None:
        expected_fill, expected_mass = (
            self._aggregate_endpoint_closures_respect_partner_capacity("cuda:0")
        )
        for _ in range(3):
            actual_fill, actual_mass = (
                self._aggregate_endpoint_closures_respect_partner_capacity(
                    "cuda:0"
                )
            )
            np.testing.assert_array_equal(actual_fill, expected_fill)
            np.testing.assert_array_equal(actual_mass, expected_mass)

    def test_independent_mass_rejects_empty_mass_and_nonpositive_active_mass(self) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(np.full(shape, INTERFACE, dtype=np.int32))
        source.fill_level.assign(np.full(shape, 0.5, dtype=np.float32))
        source.mass.assign((moments[..., 0] * 0.5).astype(np.float32))
        resolver = HomeFreeGeometricTopologyResolver(model)

        transported_fill = np.full(shape, 0.5, dtype=np.float32)
        transported_fill[0, 0, 0] = 0.0
        empty_mass = np.full(shape, 0.5, dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "empty cells carrying mass"):
            resolver.resolve(
                fluid,
                source,
                wp.array(transported_fill, dtype=float, device="cpu"),
                HomeFreeState(model),
                transported_mass=wp.array(empty_mass, dtype=float, device="cpu"),
            )

        active_mass = np.full(shape, 0.5, dtype=np.float32)
        active_mass[1, 1, 1] = 0.0
        with self.assertRaisesRegex(RuntimeError, "nonpositive mass"):
            resolver.resolve(
                fluid,
                source,
                wp.array(
                    np.full(shape, 0.5, dtype=np.float32),
                    dtype=float,
                    device="cpu",
                ),
                HomeFreeState(model),
                transported_mass=wp.array(active_mass, dtype=float, device="cpu"),
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_matches_reference_on_cuda(self) -> None:
        self._matches_reference("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_endpoint_closure_matches_reference_on_cuda(self) -> None:
        self._endpoint_closure_matches_reference("cuda:0")

    def test_failed_resolution_leaves_fluid_and_destination_unchanged(self) -> None:
        shape = (3, 3, 3)
        model = HomeLbmModel(fluid_grid_res=shape, device="cpu")
        fluid = HomeLbmState(model)
        moments = _moments(shape)
        _upload_moments(fluid, moments)
        source = HomeFreeState(model)
        source.flags.assign(np.full(shape, GAS, dtype=np.int32))
        destination = HomeFreeState(model)
        destination.mass.fill_(0.31)
        destination.fill_level.fill_(0.41)
        destination.excess_mass.fill_(0.07)
        destination.excess_momentum.fill_(-0.03)
        destination.flags.fill_(SOLID)
        transported = np.zeros(shape, dtype=np.float32)
        transported[1, 1, 1] = 0.25
        transported_device = wp.array(transported, dtype=float, device="cpu")
        fluid_before = fluid.moments.numpy().copy()
        destination_before = (
            destination.mass.numpy().copy(),
            destination.fill_level.numpy().copy(),
            destination.excess_mass.numpy().copy(),
            destination.excess_momentum.numpy().copy(),
            destination.flags.numpy().copy(),
        )

        with self.assertRaisesRegex(RuntimeError, "invalid or unresolved fluid donors"):
            HomeFreeGeometricTopologyResolver(model).resolve(
                fluid, source, transported_device, destination
            )

        np.testing.assert_array_equal(fluid.moments.numpy(), fluid_before)
        for actual, expected in zip(
            (
                destination.mass.numpy(),
                destination.fill_level.numpy(),
                destination.excess_mass.numpy(),
                destination.excess_momentum.numpy(),
                destination.flags.numpy(),
            ),
            destination_before,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
