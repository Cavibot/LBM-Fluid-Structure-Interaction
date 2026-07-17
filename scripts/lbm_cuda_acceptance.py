#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
"""CUDA acceptance smoke for FullF/HOME LBM core."""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import FullFLbmState, HomeLbmState, LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.constants import CX, CY, CZ, W


def _equilibrium(q: int, rho: float, ux: float, uy: float, uz: float) -> float:
    cx, cy, cz = CX[q], CY[q], CZ[q]
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux * ux + uy * uy + uz * uz
    return W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def _fill_uniform(domain: LbmDomain, rho: float = 1.0) -> None:
    state = domain.state
    nx, ny, nz = domain.model.nx, domain.model.ny, domain.model.nz
    if isinstance(state, FullFLbmState):
        f = np.empty((19, nx, ny, nz), dtype=np.float32)
        for q in range(19):
            f[q] = _equilibrium(q, rho, 0.0, 0.0, 0.0)
        state.f_post.assign(f.reshape(19 * nx * ny * nz))
    else:
        assert isinstance(state, HomeLbmState)
        state.rho.fill_(rho)
        state.rho_u_x.zero_()
        state.rho_u_y.zero_()
        state.rho_u_z.zero_()
        state.rho_s_xx.zero_()
        state.rho_s_yy.zero_()
        state.rho_s_zz.zero_()
        state.rho_s_xy.zero_()
        state.rho_s_xz.zero_()
        state.rho_s_yz.zero_()
    state.density.fill_(rho)
    state.velocity_x.zero_()
    state.velocity_y.zero_()
    state.velocity_z.zero_()


def _is_finite_state(domain: LbmDomain) -> bool:
    state = domain.state
    if isinstance(state, FullFLbmState):
        return bool(np.isfinite(state.f_post.numpy()).all())
    assert isinstance(state, HomeLbmState)
    return bool(np.isfinite(state.rho.numpy()).all() and np.isfinite(state.velocity_x.numpy()).all())


def _combo_step(device: str, encoding: str, collision: str, g: float = 0.0) -> str:
    model = LbmModel(
        fluid_grid_res=(8, 8, 8),
        device=device,
        encoding=encoding,
        collision=collision,
        G=g,
        bc_periodic=(True, True, True),
        tau=0.7,
    )
    domain = LbmDomain(model)
    domain.create_state()
    _fill_uniform(domain)
    domain.step(1.0)
    if not _is_finite_state(domain):
        raise RuntimeError(f"{encoding}+{collision} produced non-finite state")
    if domain.solver._f_star is None:
        raise RuntimeError("unified boundary pipeline did not allocate f*[Q]")
    return "ok finite; unified_population_boundary_scratch=yes"


def main() -> int:
    wp.init()
    print("=== LBM CUDA acceptance ===")
    print("Warp version:", getattr(wp, "__version__", "unknown"))
    print("Devices:", wp.get_devices())
    cuda_devices = list(wp.get_cuda_devices())
    print("CUDA devices:", cuda_devices)
    if not cuda_devices:
        print("FAIL: no CUDA device available")
        return 1
    device = str(cuda_devices[0])
    print("Using device:", device)

    results: list[tuple[str, str]] = []
    combos = (
        ("fullf", "srt"),
        ("fullf", "trt"),
        ("fullf", "raw_mrt"),
        ("fullf", "nocm_mrt"),
        ("home", "srt"),
        ("home", "trt"),
        ("home", "nocm_mrt"),
    )
    all_ok = True
    for encoding, collision in combos:
        name = f"{encoding}+{collision}"
        try:
            detail = _combo_step(device, encoding, collision)
            results.append((name, f"PASS {detail}"))
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:
            all_ok = False
            results.append((name, f"FAIL {exc}"))
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()

    # Fail-fast: HOME + Raw MRT is not a declared execution path.
    try:
        LbmModel(
            fluid_grid_res=(4, 4, 4),
            device=device,
            encoding="home",
            collision="raw_mrt",
        )
        all_ok = False
        print("[FAIL] HOME+Raw MRT should raise NotImplementedError")
    except NotImplementedError:
        print("[PASS] HOME+Raw MRT fail-fast")

    # FullF Shan-Chen smoke
    try:
        model = LbmModel(
            fluid_grid_res=(12, 12, 12),
            device=device,
            encoding="fullf",
            collision="srt",
            G=-1.0,
            bc_periodic=(True, True, True),
            tau=0.7,
        )
        domain = LbmDomain(model)
        domain.create_state()
        _fill_uniform(domain, rho=1.0)
        # Small density perturbation so SC force is non-trivial.
        dens = domain.state.density.numpy()
        dens[6, 6, 6] = 1.2
        domain.state.density.assign(dens)
        for _ in range(5):
            domain.step(1.0)
        if not _is_finite_state(domain):
            raise RuntimeError("SC smoke non-finite")
        print("[PASS] FullF Shan-Chen smoke (5 steps)")
    except Exception as exc:
        all_ok = False
        print(f"[FAIL] FullF Shan-Chen smoke: {exc}")
        traceback.print_exc()

    # Multi-step shear wave on CUDA
    try:
        nx, ny, nz = 8, 16, 4
        u0, tau, steps = 0.01, 0.8, 20
        model = LbmModel(
            fluid_grid_res=(nx, ny, nz),
            device=device,
            encoding="fullf",
            collision="srt",
            tau=tau,
            bc_periodic=(True, True, True),
        )
        domain = LbmDomain(model)
        domain.create_state()
        state = domain.state
        assert isinstance(state, FullFLbmState)
        f = np.zeros((19, nx, ny, nz), dtype=np.float32)
        y = np.arange(ny, dtype=np.float64)
        for j, ux in enumerate(u0 * np.sin(2.0 * math.pi * y / ny)):
            for q in range(19):
                f[q, :, j, :] = _equilibrium(q, 1.0, float(ux), 0.0, 0.0)
        state.f_post.assign(f.reshape(19 * nx * ny * nz))
        state.density.fill_(1.0)
        vel = np.zeros((nx, ny, nz), dtype=np.float32)
        for j, ux in enumerate(u0 * np.sin(2.0 * math.pi * y / ny)):
            vel[:, j, :] = float(ux)
        state.velocity_x.assign(vel)
        for _ in range(steps):
            domain.step(1.0)
        amp = float(np.max(np.abs(domain.state.velocity_x.numpy())))
        if not np.isfinite(amp) or amp >= u0:
            raise RuntimeError(f"shear wave did not decay finite: amp={amp}")
        nu = (tau - 0.5) / 3.0
        k = 2.0 * math.pi / ny
        theory = u0 * math.exp(-nu * k * k * steps)
        rel = abs(amp - theory) / theory
        print(f"[PASS] CUDA shear-wave amp={amp:.6f} theory={theory:.6f} rel_err={rel:.3f}")
    except Exception as exc:
        all_ok = False
        print(f"[FAIL] CUDA multi-step shear-wave: {exc}")
        traceback.print_exc()

    # Memory note
    try:
        dev = wp.get_device(device)
        free = int(dev.free_memory)
        total = int(dev.total_memory)
        print(
            f"GPU memory free/total bytes: {free} / {total} "
            f"({free / (1024**3):.2f} / {total / (1024**3):.2f} GiB)"
        )
    except Exception as exc:
        print(f"GPU memory info unavailable: {exc}")

    print("=== summary ===")
    for name, detail in results:
        print(f"{name}: {detail}")
    print("overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
