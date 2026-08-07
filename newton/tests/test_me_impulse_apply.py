# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""ME force→impulse apply: Δv = (F/m)·dt exactly once."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import coupling_kernels as ck


def _require_cuda() -> str:
    try:
        wp.init()
        if wp.get_cuda_device_count() <= 0:
            raise RuntimeError("no CUDA")
        return "cuda:0"
    except Exception as exc:  # noqa: BLE001
        raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc


class TestMeImpulseApply(unittest.TestCase):
    def test_linear_impulse_matches_force_times_dt(self) -> None:
        device = _require_cuda()
        n = 2
        mass = np.array([2.0, 4.0], dtype=np.float32)
        inv_m = 1.0 / mass
        # Identity inertia / inv inertia (angular unused in assert).
        Iinv = np.zeros((n, 3, 3), dtype=np.float32)
        for i in range(n):
            Iinv[i] = np.eye(3, dtype=np.float32)

        body_f = wp.zeros(n, dtype=wp.spatial_vector, device=device)
        body_qd = wp.zeros(n, dtype=wp.spatial_vector, device=device)
        body_q = wp.zeros(n, dtype=wp.transform, device=device)
        # Unit transforms at origin.
        q_np = np.zeros((n, 7), dtype=np.float32)
        q_np[:, 6] = 1.0  # quat w
        body_q.assign(wp.array(q_np, dtype=wp.transform, device=device))

        F = np.array(
            [
                [1.0, 0.0, -2.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        body_f.assign(wp.array(F, dtype=wp.spatial_vector, device=device))
        inv_m_wp = wp.array(inv_m, dtype=float, device=device)
        inv_I_wp = wp.array(Iinv, dtype=wp.mat33, device=device)

        dt = 0.01
        wp.launch(
            ck.apply_body_impulse_from_force_wrench,
            dim=n,
            inputs=[body_f, body_qd, body_q, inv_m_wp, inv_I_wp, dt, 1],
            device=device,
        )
        qd = body_qd.numpy()
        f_after = body_f.numpy()
        for i in range(n):
            expected = F[i, 0:3] * dt * inv_m[i]
            np.testing.assert_allclose(qd[i, 0:3], expected, rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(f_after[i], 0.0, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
