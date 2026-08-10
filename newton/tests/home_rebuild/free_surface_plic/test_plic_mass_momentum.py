# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import (
    FslCellFlag,
    FslState,
    FslWallMask,
    axis_aligned_wall_mask,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicAxisAdvector,
    PlicGeometricTransport,
    PlicGeometryReconstructor,
    PlicGeometryState,
)
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    coupling_kernels,
)



def _sample_circle(shape: tuple[int, int, int], *, samples: int = 12) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    center = np.asarray((shape[0] / 2.0, shape[1] / 2.0))
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples - 0.5
    for i in range(shape[0]):
        for j in range(shape[1]):
            count = sum(
                np.linalg.norm(np.asarray((i + dx, j + dy)) - center) <= 6.0
                for dx in offsets
                for dy in offsets
            )
            fill[i, j, 1] = count / (samples * samples)
    return fill


def _vortex_courant(
    shape: tuple[int, int, int], *, maximum_courant: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    stream = np.zeros((nx, ny), dtype=np.float64)
    for i in range(1, nx):
        for j in range(1, ny):
            x = (i - 1) / (nx - 2)
            y = (j - 1) / (ny - 2)
            stream[i, j] = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
    cx = np.zeros((nx + 1, ny, nz), dtype=np.float64)
    cy = np.zeros((nx, ny + 1, nz), dtype=np.float64)
    cz = np.zeros((nx, ny, nz + 1), dtype=np.float64)
    for i in range(1, nx):
        for j in range(1, ny - 1):
            cx[i, j, 1] = stream[i, j + 1] - stream[i, j]
    for i in range(1, nx - 1):
        for j in range(1, ny):
            cy[i, j, 1] = -(stream[i + 1, j] - stream[i, j])
    scale = maximum_courant / max(np.max(np.abs(cx)), np.max(np.abs(cy)))
    return (
        (cx * scale).astype(np.float32),
        (cy * scale).astype(np.float32),
        cz.astype(np.float32),
    )


def _mass_momentum_case(device: str) -> tuple[np.ndarray, ...]:
    shape = (16, 3, 3)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    closed_axes = (False, True, True)
    walls = FslWallMask(
        model,
        axis_aligned_wall_mask(shape, closed_axes=closed_axes),
        closed_axes=closed_axes,
    )
    fill = np.zeros(shape, dtype=np.float32)
    fill[3, 1, 1] = 0.7
    fill[4:8, 1, 1] = 1.0
    fill[8, 1, 1] = 0.3
    density = 1.0 + 0.01 * np.arange(shape[0], dtype=np.float32)[:, None, None]
    mass = fill * density
    velocity = np.asarray((0.06, -0.02, 0.01), dtype=np.float32)
    momentum = mass[..., None] * velocity
    flags = np.full(shape, int(FslCellFlag.GAS), dtype=np.int32)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[(fill > 0.0) & (fill < 1.0)] = int(FslCellFlag.INTERFACE)
    fsl = FslState(model)
    fsl.fill_level.assign(fill)
    fsl.mass.assign(mass)
    fsl.flags.assign(flags)
    geometry = PlicGeometryState(model)
    PlicGeometryReconstructor(model, walls).reconstruct(fsl, geometry)
    courant = np.zeros((17, 3, 3), dtype=np.float32)
    courant[:, 1, 1] = 0.2
    advector = PlicAxisAdvector(model, walls, axis=0)
    advector.advect(
        fsl.fill_level,
        geometry,
        wp.array(courant, dtype=float, device=device),
    )
    updated_mass = advector.advect_mass(fsl.mass, fsl.fill_level)
    updated_momentum = advector.advect_momentum(
        wp.array(momentum, dtype=wp.vec3, device=device), fsl.mass
    )
    return (
        updated_mass.numpy(),
        updated_momentum.numpy(),
        advector.face_mass_flux.numpy(),
        advector.face_momentum_flux.numpy(),
        mass,
        momentum,
    )


def _split_mass_momentum_case(device: str) -> tuple[np.ndarray, ...]:
    shape = (34, 34, 3)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    walls = FslWallMask.closed_box(model)
    fill = _sample_circle(shape)
    fill = np.roll(np.roll(fill, -5, axis=0), 3, axis=1)
    fill[walls.host] = 0.0
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    density = 0.95 + 0.002 * x
    mass = fill * density
    velocity = np.zeros(shape + (3,), dtype=np.float32)
    velocity[..., 0] = 0.04
    velocity[..., 1] = -0.015
    velocity[..., 2] = 0.005
    momentum = mass[..., None] * velocity
    courant = tuple(
        wp.array(field, dtype=float, device=device)
        for field in _vortex_courant(shape, maximum_courant=0.12)
    )
    transport = PlicGeometricTransport(model, walls)
    result = transport.transport_conservative(
        wp.array(fill, dtype=float, device=device),
        wp.array(mass, dtype=float, device=device),
        wp.array(momentum, dtype=wp.vec3, device=device),
        courant,
        split_order=(0, 1),
    )
    return result[0].numpy(), result[1].numpy(), result[2].numpy(), result[3]


def _projected_momentum_case(device: str) -> tuple[np.ndarray, np.ndarray]:
    shape = (4, 3, 2)
    face_x = np.zeros((5, 3, 2), dtype=np.float32)
    face_y = np.zeros((4, 4, 2), dtype=np.float32)
    face_z = np.zeros((4, 3, 3), dtype=np.float32)
    face_x[:] = np.arange(5, dtype=np.float32)[:, None, None] * 0.01
    face_y[:] = np.arange(4, dtype=np.float32)[None, :, None] * -0.02
    face_z[:] = np.arange(3, dtype=np.float32)[None, None, :] * 0.03
    mass = np.full(shape, 0.6, dtype=np.float32)
    flags = np.full(shape, int(FslCellFlag.INTERFACE), dtype=np.int32)
    solid = np.zeros(shape, dtype=np.int32)
    flags[0, 0, 0] = int(FslCellFlag.GAS)
    solid[-1, -1, -1] = 1
    output = wp.zeros(shape, dtype=wp.vec3, device=device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        coupling_kernels.initialize_projected_liquid_momentum_kernel,
        dim=shape,
        inputs=[
            wp.array(face_x, dtype=float, device=device),
            wp.array(face_y, dtype=float, device=device),
            wp.array(face_z, dtype=float, device=device),
            wp.array(mass, dtype=float, device=device),
            wp.array(flags, dtype=wp.int32, device=device),
            wp.array(solid, dtype=wp.int32, device=device),
            output,
            invalid,
        ],
        device=device,
    )
    expected = np.stack(
        (
            0.5 * (face_x[:-1] + face_x[1:]),
            0.5 * (face_y[:, :-1] + face_y[:, 1:]),
            0.5 * (face_z[:, :, :-1] + face_z[:, :, 1:]),
        ),
        axis=-1,
    ) * mass[..., None]
    expected[(flags == int(FslCellFlag.GAS)) | (solid != 0)] = 0.0
    return output.numpy(), expected


class TestPlicMassMomentum(unittest.TestCase):
    def test_projected_faces_initialize_cell_momentum(self) -> None:
        actual, expected = _projected_momentum_case("cpu")
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)

    def test_projected_momentum_cuda_matches_cpu(self) -> None:
        cpu = _projected_momentum_case("cpu")
        cuda = _projected_momentum_case("cuda:0")
        np.testing.assert_allclose(cuda[0], cpu[0], rtol=2.0e-6, atol=2.0e-7)

    def test_split_transport_conserves_all_three_ledgers(self) -> None:
        fill, mass, momentum, diagnostics = _split_mass_momentum_case("cpu")
        self.assertLess(abs(diagnostics.volume_drift), 4.0e-6 * max(float(np.sum(fill)), 1.0))
        self.assertLess(abs(diagnostics.mass_drift), 2.0e-6 * max(float(np.sum(mass)), 1.0))
        self.assertLess(max(abs(value) for value in diagnostics.momentum_drift), 2.0e-6)
        self.assertGreaterEqual(float(np.min(fill)), 0.0)
        self.assertLessEqual(float(np.max(fill)), 1.0)

    def test_split_cuda_matches_cpu(self) -> None:
        cpu = _split_mass_momentum_case("cpu")
        cuda = _split_mass_momentum_case("cuda:0")
        for actual, expected in zip(cuda[:3], cpu[:3]):
            np.testing.assert_allclose(actual, expected, rtol=4.0e-5, atol=4.0e-6)

    def test_cpu_conserves_nonuniform_mass_and_uniform_velocity(self) -> None:
        updated_mass, updated_momentum, face_mass, face_momentum, mass, momentum = (
            _mass_momentum_case("cpu")
        )
        self.assertAlmostEqual(
            float(np.sum(updated_mass)), float(np.sum(mass)), places=6
        )
        np.testing.assert_allclose(
            np.sum(updated_momentum, axis=(0, 1, 2)),
            np.sum(momentum, axis=(0, 1, 2)),
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        transported = face_mass != 0.0
        np.testing.assert_allclose(
            face_momentum[transported],
            face_mass[transported][:, None] * np.asarray((0.06, -0.02, 0.01)),
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    def test_cuda_matches_cpu(self) -> None:
        cpu = _mass_momentum_case("cpu")
        cuda = _mass_momentum_case("cuda:0")
        for actual, expected in zip(cuda[:4], cpu[:4]):
            np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=3.0e-7)


if __name__ == "__main__":
    unittest.main()
