"""Directional streaming and multi-step shear-wave acceptance tests."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    FullFLbmState,
    HomeLbmState,
    LbmDomain,
    LbmModel,
    streaming,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, OPPOSITE, W


def _idx(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return i * ny * nz + j * nz + k


def _wrap(value: int, size: int) -> int:
    if value < 0:
        return value + size
    if value >= size:
        return value - size
    return value


def _equilibrium(q: int, rho: float, ux: float, uy: float, uz: float) -> float:
    cx, cy, cz = CX[q], CY[q], CZ[q]
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux * ux + uy * uy + uz * uz
    return W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def _reconstruct_home(moments: np.ndarray) -> np.ndarray:
    rho, jx, jy, jz, sxx, syy, szz, sxy, sxz, syz = map(float, moments)
    ux, uy, uz = jx / rho, jy / rho, jz / rho
    axxy = -2 * rho * uy * ux**2 + 2 * sxy * ux + sxx * uy
    axyy = -2 * rho * ux * uy**2 + 2 * sxy * uy + syy * ux
    axxz = -2 * rho * uz * ux**2 + 2 * sxz * ux + sxx * uz
    axzz = -2 * rho * ux * uz**2 + 2 * sxz * uz + szz * ux
    ayyz = -2 * rho * uz * uy**2 + 2 * syz * uy + syy * uz
    ayzz = -2 * rho * uy * uz**2 + 2 * syz * uz + szz * uy
    out = np.empty(19, dtype=np.float64)
    for q, (cx, cy, cz, weight) in enumerate(zip(CX, CY, CZ, W, strict=True)):
        hxx, hyy, hzz = cx * cx - 1.0 / 3.0, cy * cy - 1.0 / 3.0, cz * cz - 1.0 / 3.0
        first = 3.0 * (cx * jx + cy * jy + cz * jz)
        second = 4.5 * (
            hxx * sxx + hyy * syy + hzz * szz + 2.0 * cx * cy * sxy + 2.0 * cx * cz * sxz + 2.0 * cy * cz * syz
        )
        third = 13.5 * (
            hxx * cy * axxy + hyy * cx * axyy + hxx * cz * axxz + hzz * cx * axzz + hyy * cz * ayyz + hzz * cy * ayzz
        )
        out[q] = weight * (rho + first + second + third)
    return out


def _moments_from_populations(f: np.ndarray) -> np.ndarray:
    rho = float(np.sum(f))
    jx = float(sum(CX[q] * f[q] for q in range(19)))
    jy = float(sum(CY[q] * f[q] for q in range(19)))
    jz = float(sum(CZ[q] * f[q] for q in range(19)))
    pxx = float(sum(CX[q] * CX[q] * f[q] for q in range(19)))
    pyy = float(sum(CY[q] * CY[q] * f[q] for q in range(19)))
    pzz = float(sum(CZ[q] * CZ[q] * f[q] for q in range(19)))
    pxy = float(sum(CX[q] * CY[q] * f[q] for q in range(19)))
    pxz = float(sum(CX[q] * CZ[q] * f[q] for q in range(19)))
    pyz = float(sum(CY[q] * CZ[q] * f[q] for q in range(19)))
    return np.array(
        [
            rho,
            jx,
            jy,
            jz,
            pxx - rho / 3.0,
            pyy - rho / 3.0,
            pzz - rho / 3.0,
            pxy,
            pxz,
            pyz,
        ],
        dtype=np.float64,
    )


def _incoming_population(
    q: int,
    i: int,
    j: int,
    k: int,
    source_f: np.ndarray,
    solid_phi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """NumPy reference for pull-streaming with periodic axes and solid BB."""
    si, sj, sk = i - CX[q], j - CY[q], k - CZ[q]
    outside = False
    if si < 0 or si >= nx:
        si = _wrap(si, nx)
    if sj < 0 or sj >= ny:
        sj = _wrap(sj, ny)
    if sk < 0 or sk >= nz:
        sk = _wrap(sk, nz)
    # All tests use fully periodic domains; outside stays False after wrap.
    del outside
    if solid_phi[si, sj, sk] >= 0.0:
        return float(source_f[q, si, sj, sk])
    return float(source_f[OPPOSITE[q], i, j, k])


class TestLbmDirectionalStreaming(unittest.TestCase):
    def setUp(self) -> None:
        self.nx = self.ny = self.nz = 4
        self.stride = self.nx * self.ny * self.nz
        self.device = "cpu"
        self.source = (1, 1, 1)
        self.q = 1  # +x
        self.amplitude = 0.17
        self.solid_phi = np.full((self.nx, self.ny, self.nz), 1000.0, dtype=np.float32)

    def _fullf_packet(self) -> np.ndarray:
        f = np.zeros((19, self.nx, self.ny, self.nz), dtype=np.float32)
        i, j, k = self.source
        f[self.q, i, j, k] = self.amplitude
        return f

    def _home_packet_moments(self) -> np.ndarray:
        """HOME cannot store a pure single-direction packet; use a local blob."""
        moments = np.zeros((10, self.nx, self.ny, self.nz), dtype=np.float32)
        i, j, k = self.source
        # Non-zero density and +x momentum so reconstruction is directionally biased.
        moments[:, i, j, k] = np.array(
            [1.2, 0.18, 0.0, 0.0, 0.04, -0.01, -0.01, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return moments

    def _reference_streamed_populations(self, source_f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(source_f, dtype=np.float64)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    for q in range(19):
                        out[q, i, j, k] = _incoming_population(
                            q, i, j, k, source_f, self.solid_phi, self.nx, self.ny, self.nz
                        )
        return out

    def _reference_streamed_moments(self, source_f: np.ndarray) -> np.ndarray:
        f_star = self._reference_streamed_populations(source_f)
        moments = np.zeros((10, self.nx, self.ny, self.nz), dtype=np.float64)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    moments[:, i, j, k] = _moments_from_populations(f_star[:, i, j, k])
        return moments

    def test_fullf_to_populations_moves_packet_and_wraps(self) -> None:
        model = LbmModel(
            fluid_grid_res=(self.nx, self.ny, self.nz),
            device=self.device,
            encoding="fullf",
            collision="srt",
            bc_periodic=(True, True, True),
        )
        state = FullFLbmState(model)
        f = self._fullf_packet()
        state.f_post.assign(f.reshape(19 * self.stride))
        f_star = wp.zeros(19 * self.stride, dtype=float, device=self.device)
        wp.launch(
            streaming.stream_fullf_to_populations_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[state.f_post, state.solid_phi, f_star, 1, 1, 1, self.nx, self.ny, self.nz, self.stride],
            device=self.device,
        )
        # Copy out of the Warp buffer view before later zero_/relaunch mutates it.
        actual = f_star.numpy().reshape(19, self.nx, self.ny, self.nz).copy()
        expected = self._reference_streamed_populations(f)

        # Directional move: packet at source moves to source + c_q.
        si, sj, sk = self.source
        di = _wrap(si + CX[self.q], self.nx)
        self.assertAlmostEqual(float(actual[self.q, di, sj, sk]), self.amplitude, places=6)
        self.assertAlmostEqual(float(actual[self.q, si, sj, sk]), 0.0, places=6)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)

        # Periodic wrap: put packet at xmax and expect wrap to xmin.
        f_wrap = np.zeros_like(f)
        f_wrap[self.q, self.nx - 1, sj, sk] = self.amplitude
        state.f_post.assign(f_wrap.reshape(19 * self.stride))
        f_star.zero_()
        wp.launch(
            streaming.stream_fullf_to_populations_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[state.f_post, state.solid_phi, f_star, 1, 1, 1, self.nx, self.ny, self.nz, self.stride],
            device=self.device,
        )
        actual_wrap = f_star.numpy().reshape(19, self.nx, self.ny, self.nz).copy()
        self.assertAlmostEqual(float(actual_wrap[self.q, 0, sj, sk]), self.amplitude, places=6)
        expected_wrap = self._reference_streamed_populations(f_wrap)
        np.testing.assert_allclose(actual_wrap, expected_wrap, atol=1e-7, rtol=1e-7)

    def test_fullf_to_moments_collects_streamed_packet(self) -> None:
        model = LbmModel(
            fluid_grid_res=(self.nx, self.ny, self.nz),
            device=self.device,
            encoding="fullf",
            collision="nocm_mrt",
            bc_periodic=(True, True, True),
        )
        state = FullFLbmState(model)
        f = self._fullf_packet()
        state.f_post.assign(f.reshape(19 * self.stride))
        moments = tuple(wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=self.device) for _ in range(10))
        wp.launch(
            streaming.stream_fullf_to_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[state.f_post, state.solid_phi, *moments, 1, 1, 1, self.nx, self.ny, self.nz, self.stride],
            device=self.device,
        )
        actual = np.stack([m.numpy().copy() for m in moments], axis=0)
        expected = self._reference_streamed_moments(f)
        si, sj, sk = self.source
        di = _wrap(si + CX[self.q], self.nx)
        self.assertAlmostEqual(float(actual[0, di, sj, sk]), self.amplitude, places=6)
        self.assertAlmostEqual(float(actual[1, di, sj, sk]), self.amplitude, places=6)
        self.assertAlmostEqual(float(actual[0, si, sj, sk]), 0.0, places=6)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_home_to_populations_matches_reconstruct_then_pull(self) -> None:
        model = LbmModel(
            fluid_grid_res=(self.nx, self.ny, self.nz),
            device=self.device,
            encoding="home",
            collision="srt",
            bc_periodic=(True, True, True),
        )
        state = HomeLbmState(model)
        moments = self._home_packet_moments()
        for field, values in zip(state.kinetic_fields, moments, strict=True):
            field.assign(values)
        f_star = wp.zeros(19 * self.stride, dtype=float, device=self.device)
        wp.launch(
            streaming.stream_home_to_populations_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[*state.kinetic_fields, state.solid_phi, f_star, 1, 1, 1, self.nx, self.ny, self.nz, self.stride],
            device=self.device,
        )
        actual = f_star.numpy().reshape(19, self.nx, self.ny, self.nz).copy()

        source_f = np.zeros((19, self.nx, self.ny, self.nz), dtype=np.float64)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if moments[0, i, j, k] == 0.0:
                        continue
                    source_f[:, i, j, k] = _reconstruct_home(moments[:, i, j, k])
        expected = self._reference_streamed_populations(source_f)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

        # Directional bias: reconstructed +x mass should leave the source cell.
        si, sj, sk = self.source
        di = _wrap(si + 1, self.nx)
        self.assertGreater(float(actual[1, di, sj, sk]), float(actual[1, si, sj, sk]))
        self.assertGreater(float(actual[1, di, sj, sk]), 0.0)

    def test_home_to_moments_matches_reconstruct_then_pull(self) -> None:
        model = LbmModel(
            fluid_grid_res=(self.nx, self.ny, self.nz),
            device=self.device,
            encoding="home",
            collision="nocm_mrt",
            bc_periodic=(True, True, True),
        )
        state = HomeLbmState(model)
        moments = self._home_packet_moments()
        for field, values in zip(state.kinetic_fields, moments, strict=True):
            field.assign(values)
        out_moments = tuple(wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=self.device) for _ in range(10))
        wp.launch(
            streaming.stream_home_to_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[*state.kinetic_fields, state.solid_phi, *out_moments, 1, 1, 1, self.nx, self.ny, self.nz],
            device=self.device,
        )
        actual = np.stack([m.numpy().copy() for m in out_moments], axis=0)

        source_f = np.zeros((19, self.nx, self.ny, self.nz), dtype=np.float64)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if moments[0, i, j, k] == 0.0:
                        continue
                    source_f[:, i, j, k] = _reconstruct_home(moments[:, i, j, k])
        expected = self._reference_streamed_moments(source_f)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        # Rest-mode mass stays at the source cell after pull-stream; only the
        # non-zero velocity populations leave. Check mass leaves and arrives.
        si, sj, sk = self.source
        di = _wrap(si + 1, self.nx)
        rho0 = float(moments[0, si, sj, sk])
        self.assertGreater(float(actual[0, di, sj, sk]), 0.0)
        self.assertLess(float(actual[0, si, sj, sk]), rho0)
        self.assertGreater(float(actual[1, di, sj, sk]), float(actual[1, si, sj, sk]))


class TestLbmShearWaveDecay(unittest.TestCase):
    def _init_shear_wave(self, domain: LbmDomain, u0: float) -> float:
        state = domain.state
        nx, ny, nz = domain.model.nx, domain.model.ny, domain.model.nz
        y = np.arange(ny, dtype=np.float64)
        uy_profile = u0 * np.sin(2.0 * math.pi * y / ny)
        mass = 0.0
        if isinstance(state, FullFLbmState):
            f = np.zeros((19, nx, ny, nz), dtype=np.float32)
            for j, ux in enumerate(uy_profile):
                for q in range(19):
                    f[q, :, j, :] = _equilibrium(q, 1.0, float(ux), 0.0, 0.0)
            state.f_post.assign(f.reshape(19 * nx * ny * nz))
            mass = float(np.sum(f, dtype=np.float64))
        else:
            assert isinstance(state, HomeLbmState)
            for field in state.kinetic_fields:
                field.zero_()
            rho = np.ones((nx, ny, nz), dtype=np.float32)
            jx = np.zeros((nx, ny, nz), dtype=np.float32)
            for j, ux in enumerate(uy_profile):
                jx[:, j, :] = float(ux)
            # Equilibrium HOME second moments: rho * (u_a u_b)
            sxx = np.zeros((nx, ny, nz), dtype=np.float32)
            for j, ux in enumerate(uy_profile):
                sxx[:, j, :] = float(ux * ux)
            state.rho.assign(rho)
            state.rho_u_x.assign(jx)
            state.rho_s_xx.assign(sxx)
            mass = float(np.sum(rho, dtype=np.float64))
        state.density.fill_(1.0)
        vel = np.zeros((nx, ny, nz), dtype=np.float32)
        for j, ux in enumerate(uy_profile):
            vel[:, j, :] = float(ux)
        state.velocity_x.assign(vel)
        return mass

    def _max_ux(self, domain: LbmDomain) -> float:
        return float(np.max(np.abs(domain.state.velocity_x.numpy())))

    def _mass(self, domain: LbmDomain) -> float:
        state = domain.state
        if isinstance(state, FullFLbmState):
            return float(np.sum(state.f_post.numpy(), dtype=np.float64))
        assert isinstance(state, HomeLbmState)
        return float(np.sum(state.rho.numpy(), dtype=np.float64))

    def test_srt_and_trt_shear_wave_decay_matches_theory(self) -> None:
        nx, ny, nz = 8, 16, 4
        u0 = 0.01
        tau = 0.8
        steps = 40
        nu = (tau - 0.5) / 3.0
        k = 2.0 * math.pi / ny
        theoretical = u0 * math.exp(-nu * k * k * steps)

        for collision, lambda_trt in (("srt", 0.0), ("trt", (tau - 0.5) ** 2)):
            with self.subTest(collision=collision):
                model = LbmModel(
                    fluid_grid_res=(nx, ny, nz),
                    device="cpu",
                    encoding="fullf",
                    collision=collision,
                    lambda_trt=lambda_trt,
                    tau=tau,
                    bc_periodic=(True, True, True),
                )
                domain = LbmDomain(model)
                domain.create_state()
                mass0 = self._init_shear_wave(domain, u0)
                for _ in range(steps):
                    domain.step(1.0)
                mass1 = self._mass(domain)
                amp = self._max_ux(domain)
                self.assertTrue(np.isfinite(domain.state.velocity_x.numpy()).all())
                # float32 multi-step mass drift is O(1e-3) absolute on this grid.
                self.assertAlmostEqual(mass0, mass1, delta=1e-3 * max(1.0, abs(mass0)))
                self.assertLess(amp, u0)
                # Small-grid acceptance window: correct order of viscous decay.
                self.assertLess(abs(amp - theoretical) / theoretical, 0.25)

    def test_home_paths_remain_finite_conserved_and_decaying(self) -> None:
        nx, ny, nz = 8, 16, 4
        u0 = 0.01
        tau = 0.8
        steps = 30
        for encoding_name, collision in (
            ("home", "nocm_mrt"),
            ("home", "srt"),
            ("fullf", "nocm_mrt"),
        ):
            with self.subTest(encoding=encoding_name, collision=collision):
                model = LbmModel(
                    fluid_grid_res=(nx, ny, nz),
                    device="cpu",
                    encoding=encoding_name,
                    collision=collision,
                    tau=tau,
                    bc_periodic=(True, True, True),
                )
                domain = LbmDomain(model)
                domain.create_state()
                mass0 = self._init_shear_wave(domain, u0)
                amp0 = self._max_ux(domain)
                for _ in range(steps):
                    domain.step(1.0)
                mass1 = self._mass(domain)
                amp1 = self._max_ux(domain)
                vel = domain.state.velocity_x.numpy()
                self.assertTrue(np.isfinite(vel).all())
                self.assertAlmostEqual(mass0, mass1, delta=1e-4 * max(1.0, abs(mass0)))
                self.assertLess(amp1, amp0)


if __name__ == "__main__":
    unittest.main()
