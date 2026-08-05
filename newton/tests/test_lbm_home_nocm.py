"""HOME reconstruction and NOCM closed-form reference tests."""

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import HomeLbmState, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.solver import collisions, encoding
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, W


def reference_reconstruct(m: np.ndarray) -> np.ndarray:
    rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz = map(float, m)
    ux, uy, uz = jx / rho, jy / rho, jz / rho
    axxy = -2 * rho * uy * ux**2 + 2 * sxy * ux + sxx * uy
    axyy = -2 * rho * ux * uy**2 + 2 * sxy * uy + syy * ux
    axxz = -2 * rho * uz * ux**2 + 2 * sxz * ux + sxx * uz
    axzz = -2 * rho * ux * uz**2 + 2 * sxz * uz + szz * ux
    ayyz = -2 * rho * uz * uy**2 + 2 * syz * uy + syy * uz
    ayzz = -2 * rho * uy * uz**2 + 2 * syz * uz + szz * uy
    result = np.empty(19, dtype=np.float64)
    for q, (cx, cy, cz, weight) in enumerate(zip(CX, CY, CZ, W)):
        hxx, hyy, hzz = cx * cx - 1 / 3, cy * cy - 1 / 3, cz * cz - 1 / 3
        first = 3 * (cx * jx + cy * jy + cz * jz)
        second = 4.5 * (
            hxx * sxx + hyy * syy + hzz * szz
            + 2 * cx * cy * sxy + 2 * cx * cz * sxz + 2 * cy * cz * syz
        )
        third = 13.5 * (
            hxx * cy * axxy + hyy * cx * axyy
            + hxx * cz * axxz + hzz * cx * axzz
            + hyy * cz * ayyz + hzz * cy * ayzz
        )
        result[q] = weight * (rho + first + second + third)
    return result


def reference_nocm(m: np.ndarray, omega: float) -> np.ndarray:
    rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz = map(float, m)
    ux, uy, uz = jx / rho, jy / rho, jz / rho
    pxx, pyy, pzz = sxx + rho / 3, syy + rho / 3, szz + rho / 3
    trace = (pxx + pyy + pzz) / 3
    dev = np.array([pxx, pyy, pzz]) - trace
    utrace = rho * (ux * ux + uy * uy + uz * uz) / 3
    post_p = np.empty(3)
    post_p[0] = rho / 3 + (1 - omega) * dev[0] + utrace + omega * rho * (2 * ux**2 - uy**2 - uz**2) / 3
    post_p[1] = rho / 3 + (1 - omega) * dev[1] + utrace + omega * rho * (-ux**2 + 2 * uy**2 - uz**2) / 3
    post_p[2] = rho / 3 + (1 - omega) * dev[2] + utrace + omega * rho * (-ux**2 - uy**2 + 2 * uz**2) / 3
    return np.array([
        rho, jx, jy, jz,
        post_p[0] - rho / 3, post_p[1] - rho / 3, post_p[2] - rho / 3,
        (1 - omega) * sxy + omega * rho * ux * uy,
        (1 - omega) * sxz + omega * rho * ux * uz,
        (1 - omega) * syz + omega * rho * uy * uz,
    ])


class TestLbmHomeNocm(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LbmModel(
            fluid_grid_res=(1, 1, 1), device="cpu", encoding="home",
            collision="nocm_mrt", bc_periodic=(True, True, True), tau=0.8,
        )

    def test_home_reconstruction_matches_numpy_reference_and_closes(self) -> None:
        values = np.array([1.1, 0.033, -0.022, 0.011, 0.014, -0.007, 0.004, 0.003, -0.002, 0.001], dtype=np.float32)
        state = HomeLbmState(self.model)
        for field, value in zip(state.kinetic_fields, values):
            field.fill_(float(value))
        f = wp.zeros(19, dtype=float, device="cpu")
        wp.launch(
            encoding.home_to_populations_kernel, dim=(1, 1, 1),
            inputs=[*state.kinetic_fields, f, 1, 1, 1], device="cpu",
        )
        np.testing.assert_allclose(f.numpy(), reference_reconstruct(values), atol=2e-7, rtol=2e-6)

        recovered = HomeLbmState(self.model)
        wp.launch(
            encoding.populations_to_home_kernel, dim=(1, 1, 1),
            inputs=[f, *recovered.kinetic_fields, 1, 1, 1], device="cpu",
        )
        actual = np.array([float(x.numpy()[0, 0, 0]) for x in recovered.kinetic_fields])
        np.testing.assert_allclose(actual, values, atol=3e-7, rtol=3e-6)

    def test_home_nocm_matches_numpy_reference(self) -> None:
        values = np.array([1.05, 0.021, -0.0105, 0.00525, 0.02, -0.01, 0.006, 0.004, -0.003, 0.002], dtype=np.float32)
        source = HomeLbmState(self.model)
        target = HomeLbmState(self.model)
        for field, value in zip(source.kinetic_fields, values):
            field.fill_(float(value))
        omega = 1.0 / self.model.tau
        wp.launch(
            collisions.home_nocm_collision_kernel, dim=(1, 1, 1),
            inputs=[*source.kinetic_fields, *target.kinetic_fields, omega], device="cpu",
        )
        actual = np.array([float(x.numpy()[0, 0, 0]) for x in target.kinetic_fields])
        np.testing.assert_allclose(actual, reference_nocm(values, omega), atol=3e-7, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
