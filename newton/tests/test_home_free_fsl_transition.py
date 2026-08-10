# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fresh/dead hydrodynamic-node transitions for moving PLIC interfaces."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    D3Q27_DIRECTIONS,
    D3Q27_OPPOSITE,
    D3Q27_WEIGHTS,
    FslActiveTransitionRemapper,
    HomeFreeInterfaceGeometry,
    HomeFreeState,
    HomeLbmModel,
    HomeLbmState,
    remap_fsl_active_moments,
)
from wanphys._src.fluid.fluid_grid.home_lbm import kernels as home_kernels
from wanphys._src.fluid.fluid_grid.home_lbm.vof import geometry_kernels
from wanphys._src.fluid.fluid_grid.home_lbm.vof import kernels as vof_kernels


def _equilibrium_moments(rho: float, velocity: np.ndarray) -> np.ndarray:
    u = np.asarray(velocity, dtype=np.float64)
    return np.asarray(
        (
            rho,
            rho * u[0],
            rho * u[1],
            rho * u[2],
            rho * u[0] * u[0],
            rho * u[1] * u[1],
            rho * u[2] * u[2],
            rho * u[0] * u[1],
            rho * u[0] * u[2],
            rho * u[1] * u[2],
        ),
        dtype=np.float64,
    )


def _upload(state: HomeLbmState, moments: np.ndarray) -> None:
    state.moments.assign(
        np.ascontiguousarray(moments.reshape(-1, 10).T.reshape(-1), dtype=np.float32)
    )


def _download(state: HomeLbmState) -> np.ndarray:
    return (
        state.moments.numpy()
        .reshape(10, -1)
        .T.reshape(state.res + (10,))
        .astype(np.float64)
    )


class TestHomeFreeFslTransition(unittest.TestCase):
    def _dynamic_planar_translation(self, device: str) -> None:
        shape = (8, 3, 2)
        stride = int(np.prod(shape))
        rho = 1.017
        velocity = np.asarray((0.013, -0.007, 0.004), dtype=np.float64)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(True, True, True),
            kinematic_viscosity=0.25,
            device=device,
        )
        state_in = HomeLbmState(model)
        state_out = HomeLbmState(model)
        moments = np.empty(shape + (10,), dtype=np.float64)
        moments[...] = _equilibrium_moments(rho, velocity)
        _upload(state_in, moments)
        directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=device
        )
        weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=device
        )
        opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=device)
        active = wp.zeros(shape, dtype=wp.int32, device=device)
        fraction = wp.zeros(27 * stride, dtype=float, device=device)
        extrapolation = wp.zeros(27 * stride, dtype=float, device=device)
        status = wp.zeros(27 * stride, dtype=wp.int32, device=device)
        owner = wp.zeros(27 * stride, dtype=wp.int32, device=device)
        boundary_velocity = wp.zeros(27 * stride, dtype=wp.vec3, device=device)
        boundary_strain = wp.zeros(6 * 27 * stride, dtype=float, device=device)
        gas_density = wp.full(shape, rho, dtype=float, device=device)
        raw_moments = wp.zeros(10 * stride, dtype=float, device=device)
        invalid_velocity = wp.zeros(1, dtype=wp.int32, device=device)
        invalid_geometry = wp.zeros(2, dtype=wp.int32, device=device)
        invalid_stream = wp.zeros(1, dtype=wp.int32, device=device)
        invalid_collision = wp.zeros(1, dtype=wp.int32, device=device)
        fallback_count = wp.zeros(1, dtype=wp.int32, device=device)

        def rebuild_coverage(fill_left: float, fill_right: float) -> None:
            invalid_geometry.zero_()
            fill = np.zeros(shape, dtype=np.float32)
            fill[0] = fill_left
            fill[1:4] = 1.0
            fill[4] = fill_right
            free = HomeFreeState(model)
            free.initialize_from_fill_level(state_in, fill)
            geometry = HomeFreeInterfaceGeometry(model)
            geometry.update(free)
            wp.launch(
                geometry_kernels.classify_fsl_hydrodynamic_nodes_kernel,
                dim=shape,
                inputs=[
                    free.flags,
                    geometry.normal,
                    geometry.plane_offset,
                    geometry.valid,
                    active,
                    invalid_geometry,
                    1.0e-6,
                ],
                device=device,
            )
            wp.launch(
                geometry_kernels.construct_fsl_link_coverage_kernel,
                dim=27 * stride,
                inputs=[
                    free.flags,
                    geometry.normal,
                    geometry.plane_offset,
                    geometry.valid,
                    active,
                    directions,
                    fraction,
                    extrapolation,
                    status,
                    owner,
                    1.0e-6,
                    1,
                    1,
                    1,
                    *shape,
                ],
                device=device,
            )
            wp.synchronize_device(device)
            np.testing.assert_array_equal(invalid_geometry.numpy(), 0)

        rebuild_coverage(0.75, 0.25)
        remapper = FslActiveTransitionRemapper(state_in, active)
        fresh_total = 0
        dead_total = 0
        for fill_left, fill_right in ((0.60, 0.40), (0.49, 0.51), (0.30, 0.70)):
            rebuild_coverage(fill_left, fill_right)
            transitions = remapper.remap(state_in, active)
            fresh_total += transitions.fresh_cell_count
            dead_total += transitions.dead_cell_count
            invalid_velocity.zero_()
            invalid_stream.zero_()
            invalid_collision.zero_()
            fallback_count.zero_()
            wp.launch(
                vof_kernels.extrapolate_fsl_boundary_velocity_kernel,
                dim=27 * stride,
                inputs=[
                    state_in.moments,
                    active,
                    status,
                    fraction,
                    directions,
                    boundary_velocity,
                    invalid_velocity,
                    0,
                    1,
                    1,
                    1,
                    *shape,
                    stride,
                ],
                device=device,
            )
            wp.launch(
                vof_kernels.fsl_stream_moments_kernel,
                dim=shape,
                inputs=[
                    state_in.moments,
                    active,
                    status,
                    owner,
                    fraction,
                    gas_density,
                    boundary_velocity,
                    boundary_strain,
                    directions,
                    weights,
                    opposites,
                    raw_moments,
                    invalid_stream,
                    fallback_count,
                    model.shear_omega,
                    0,
                    0,
                    1,
                    1,
                    1,
                    *shape,
                    stride,
                ],
                device=device,
            )
            wp.launch(
                home_kernels.collide_force_free_moments_kernel,
                dim=shape,
                inputs=[
                    raw_moments,
                    active,
                    state_out.moments,
                    invalid_collision,
                    model.shear_omega,
                    shape[1],
                    shape[2],
                    stride,
                ],
                device=device,
            )
            wp.synchronize_device(device)
            self.assertEqual(int(invalid_velocity.numpy()[0]), 0)
            self.assertEqual(int(invalid_stream.numpy()[0]), 0)
            self.assertEqual(int(invalid_collision.numpy()[0]), 0)
            self.assertEqual(int(fallback_count.numpy()[0]), 0)
            actual = _download(state_out)
            current_active = active.numpy().astype(bool)
            np.testing.assert_allclose(
                actual[current_active],
                moments[current_active],
                rtol=8.0e-5,
                atol=8.0e-7,
            )
            state_in, state_out = state_out, state_in

        self.assertEqual(fresh_total, shape[1] * shape[2])
        self.assertEqual(dead_total, shape[1] * shape[2])

    @staticmethod
    def _advancing_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (5, 3, 2)
        previous = np.zeros(shape, dtype=bool)
        previous[:3] = True
        current = previous.copy()
        current[3] = True
        moments = np.empty(shape + (10,), dtype=np.float64)
        for index in np.ndindex(shape):
            rho = 0.98 + 0.004 * index[1] + 0.002 * index[2]
            velocity = np.asarray(
                (0.02 + 0.001 * index[1], -0.01, 0.006 + 0.001 * index[2])
            )
            moments[index] = _equilibrium_moments(rho, velocity)
        moments[3] = np.asarray((9.0,) * 10, dtype=np.float64)
        moments[2, ..., 4:10] += 0.01
        return moments, previous, current

    def _warp_matches_reference(self, device: str) -> None:
        moments, previous, current = self._advancing_case()
        expected = remap_fsl_active_moments(
            moments,
            previous,
            current,
            periodic=(False, True, True),
        )
        model = HomeLbmModel(
            fluid_grid_res=previous.shape,
            periodic=(False, True, True),
            device=device,
        )
        state = HomeLbmState(model)
        _upload(state, moments)
        previous_device = wp.array(
            previous.astype(np.int32), dtype=wp.int32, device=device
        )
        current_device = wp.array(
            current.astype(np.int32), dtype=wp.int32, device=device
        )
        remapper = FslActiveTransitionRemapper(state, previous_device)

        diagnostics = remapper.remap(state, current_device)
        wp.synchronize_device(device)

        self.assertEqual(diagnostics, expected.diagnostics)
        np.testing.assert_array_equal(
            remapper.donor_count.numpy().reshape(previous.shape),
            expected.donor_count,
        )
        np.testing.assert_allclose(
            _download(state), expected.moments, rtol=3.0e-6, atol=3.0e-7
        )
        fresh = ~previous & current
        actual = _download(state)
        velocity = actual[..., 1:4][fresh] / actual[..., 0, None][fresh]
        np.testing.assert_allclose(
            actual[..., 4][fresh], actual[..., 0][fresh] * velocity[:, 0] ** 2
        )

    def test_reference_initializes_fresh_nodes_from_persistent_donors(self) -> None:
        moments, previous, current = self._advancing_case()
        result = remap_fsl_active_moments(
            moments,
            previous,
            current,
            periodic=(False, True, True),
        )

        self.assertEqual(result.diagnostics.fresh_cell_count, 6)
        self.assertEqual(result.diagnostics.dead_cell_count, 0)
        self.assertEqual(result.diagnostics.unresolved_fresh_cell_count, 0)
        self.assertTrue(np.all(result.donor_count[3] > 0))
        self.assertFalse(np.allclose(result.moments[3], moments[3]))

    def test_reference_preserves_dead_node_storage_without_global_correction(self) -> None:
        moments, previous, _ = self._advancing_case()
        current = previous.copy()
        current[2] = False
        result = remap_fsl_active_moments(
            moments,
            previous,
            current,
            periodic=(False, True, True),
        )

        self.assertEqual(result.diagnostics.dead_cell_count, 6)
        self.assertEqual(result.diagnostics.fresh_cell_count, 0)
        np.testing.assert_array_equal(result.moments, moments)

    def test_reference_rejects_fresh_node_without_persistent_donor(self) -> None:
        shape = (4, 2, 2)
        moments = np.empty(shape + (10,), dtype=np.float64)
        moments[...] = _equilibrium_moments(1.0, np.zeros(3))
        previous = np.zeros(shape, dtype=bool)
        previous[0] = True
        current = previous.copy()
        current[3] = True
        with self.assertRaisesRegex(RuntimeError, "no persistent active donor"):
            remap_fsl_active_moments(moments, previous, current)

    def test_warp_failure_is_transactional(self) -> None:
        shape = (4, 2, 2)
        model = HomeLbmModel(
            fluid_grid_res=shape,
            periodic=(False, False, False),
            device="cpu",
        )
        state = HomeLbmState(model)
        moments = np.empty(shape + (10,), dtype=np.float64)
        moments[...] = _equilibrium_moments(1.0, np.zeros(3))
        _upload(state, moments)
        previous = np.zeros(shape, dtype=np.int32)
        previous[0] = 1
        current = previous.copy()
        current[3] = 1
        previous_device = wp.array(previous, dtype=wp.int32, device="cpu")
        remapper = FslActiveTransitionRemapper(state, previous_device)
        before = state.moments.numpy().copy()

        with self.assertRaisesRegex(RuntimeError, "no persistent active donor"):
            remapper.remap(
                state, wp.array(current, dtype=wp.int32, device="cpu")
            )

        np.testing.assert_array_equal(state.moments.numpy(), before)
        np.testing.assert_array_equal(remapper.previous_active.numpy(), previous)

    def test_warp_cpu_matches_reference(self) -> None:
        self._warp_matches_reference("cpu")

    def test_warp_cpu_dynamic_planar_translation(self) -> None:
        self._dynamic_planar_translation("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_matches_reference(self) -> None:
        self._warp_matches_reference("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device is required")
    def test_warp_cuda_dynamic_planar_translation(self) -> None:
        self._dynamic_planar_translation("cuda:0")


if __name__ == "__main__":
    unittest.main()
