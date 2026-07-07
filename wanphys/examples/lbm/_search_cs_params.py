# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless CS EOS parameter stability sweep.

Runs a small spinodal-decomposition test for each (T, tau, a) combination
and reports which ones survive without divergence.  Use the stable region
as a starting point for CS-LBM examples.

Usage:
    uv run --extra examples python -m wanphys.examples.lbm._search_cs_params
"""

from __future__ import annotations

import sys
from itertools import product

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
N: int = 64                        # small grid for fast sweeps
DH: float = 0.02
TEST_STEPS: int = 300              # steps per test
GRAVITY: float = 0.0               # no gravity — pure phase separation
OMEGA_REG: float = 0.5
LAMBDA_TRT: float = 0.03

# Parameter sweep ranges
T_VALUES: tuple[float, ...] = (0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09)
TAU_VALUES: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70)
A_VALUES: tuple[float, ...] = (0.5, 1.0)
B_VALUE: float = 4.0
G_VALUES: tuple[float, ...] = (-1.0, -3.0, -5.0)

RHO0: float = 0.12                 # initial uniform density (near critical)
NOISE_AMP: float = 0.01


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@wp.kernel
def _init_test(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    rho0: float,
    noise_amp: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    h = (i * 1664525 + j * 1013904223 + k * 22695477 + seed * 1103515245) & 0x7FFFFFFF
    n = (float(h) / 2147483648.0) - 1.0
    rho = wp.max(rho0 + noise_amp * n, 0.001)
    density[i, j, k] = rho
    f[0 * stride + idx] = (1.0 / 3.0) * rho
    f[1 * stride + idx] = (1.0 / 18.0) * rho
    f[2 * stride + idx] = (1.0 / 18.0) * rho
    f[3 * stride + idx] = (1.0 / 18.0) * rho
    f[4 * stride + idx] = (1.0 / 18.0) * rho
    f[5 * stride + idx] = (1.0 / 18.0) * rho
    f[6 * stride + idx] = (1.0 / 18.0) * rho
    f[7 * stride + idx] = (1.0 / 36.0) * rho
    f[8 * stride + idx] = (1.0 / 36.0) * rho
    f[9 * stride + idx] = (1.0 / 36.0) * rho
    f[10 * stride + idx] = (1.0 / 36.0) * rho
    f[11 * stride + idx] = (1.0 / 36.0) * rho
    f[12 * stride + idx] = (1.0 / 36.0) * rho
    f[13 * stride + idx] = (1.0 / 36.0) * rho
    f[14 * stride + idx] = (1.0 / 36.0) * rho
    f[15 * stride + idx] = (1.0 / 36.0) * rho
    f[16 * stride + idx] = (1.0 / 36.0) * rho
    f[17 * stride + idx] = (1.0 / 36.0) * rho
    f[18 * stride + idx] = (1.0 / 36.0) * rho


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def _test_one(a: float, b: float, T: float, tau: float, G: float) -> dict:
    """Run one spinodal test.  Returns dict with stability and phase-sep info."""
    result: dict = {
        "a": a, "b": b, "T": T, "tau": tau, "G": G,
        "T/Tc": round(T / (0.3773 * a / b), 3),
        "stable": False,
        "phase_separated": False,
        "rho_min": 0.0, "rho_max": 0.0,
        "diverged_at": -1,
    }

    try:
        model: LbmModel = LbmModel(
            fluid_grid_res=(N, N, N),
            fluid_grid_cell_size=DH,
            tau=tau,
            G=G,
            psi_type=2,
            psi_ref=1.0,
            cs_a=a,
            cs_b=b,
            cs_T=T,
            lambda_trt=LAMBDA_TRT,
            use_regularization=True,
            omega_reg=OMEGA_REG,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY,
            bc_periodic=(True, True, True),
        )
    except ValueError as e:
        result["diverged_at"] = -2  # model validation failed
        return result

    domain: LbmDomain = LbmDomain(model)
    domain.create_state()
    state = domain.state
    n: int = int(model.nx)
    stride: int = n * n * n

    wp.launch(
        _init_test,
        dim=(n, n, n),
        inputs=[state.f, state.density, RHO0, NOISE_AMP, 42, n, n, n, stride],
    )
    for arr_name in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
        wp.copy(getattr(domain._state_out, arr_name), getattr(state, arr_name))
    wp.synchronize_device(model._device)

    for step in range(TEST_STEPS):
        domain.step(1.0 / 60.0)

        if step % 50 == 49:
            wp.synchronize_device(model._device)
            rho_np: np.ndarray = state.density.numpy()
            if np.any(~np.isfinite(rho_np)):
                result["diverged_at"] = step + 1
                return result

    wp.synchronize_device(model._device)
    rho_np = state.density.numpy()
    result["stable"] = True
    result["rho_min"] = round(float(rho_np.min()), 4)
    result["rho_max"] = round(float(rho_np.max()), 4)
    # Phase separated if density range spans well beyond noise level
    result["phase_separated"] = (result["rho_max"] - result["rho_min"]) > 0.03

    return result


# ---------------------------------------------------------------------------
def main() -> None:
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    print(f"Device: {device.name}")
    print(f"Grid: {N}^3, steps: {TEST_STEPS}")
    print(f"T range: {T_VALUES}")
    print(f"tau range: {TAU_VALUES}")
    print(f"a range: {A_VALUES}")
    print(f"G range: {G_VALUES}")
    print(f"T_c(a=0.5)={0.3773*0.5/B_VALUE:.4f}, T_c(a=1.0)={0.3773*1.0/B_VALUE:.4f}")
    print()

    total: int = len(T_VALUES) * len(TAU_VALUES) * len(A_VALUES) * len(G_VALUES)
    stable_results: list[dict] = []
    unstable_results: list[dict] = []
    count: int = 0

    for a, T, tau, G in product(A_VALUES, T_VALUES, TAU_VALUES, G_VALUES):
        count += 1
        tc: float = 0.3773 * a / B_VALUE
        sys.stdout.write(f"\r[{count}/{total}] a={a} T={T} (T/Tc={T/tc:.2f}) tau={tau} G={G} ...")
        sys.stdout.flush()

        r: dict = _test_one(a, B_VALUE, T, tau, G)
        if r["stable"] and r["phase_separated"]:
            stable_results.append(r)
        else:
            unstable_results.append(r)

    print("\n")

    # ---- Print stable results ------------------------------------------------
    print("=" * 78)
    print(f"STABLE + PHASE-SEPARATED: {len(stable_results)} / {total}")
    print("=" * 78)
    if stable_results:
        stable_results.sort(key=lambda r: (r["a"], r["T"], r["tau"], r["G"]))
        print(f"{'a':>5} {'T':>7} {'T/Tc':>7} {'tau':>5} {'G':>5} {'r_min':>7} {'r_max':>7} {'dr':>7}")
        print("-" * 55)
        for r in stable_results:
            dr: float = round(r["rho_max"] - r["rho_min"], 4)
            print(f"{r['a']:5.1f} {r['T']:7.3f} {r['T/Tc']:7.3f} {r['tau']:5.2f} {r['G']:5.1f} "
                  f"{r['rho_min']:7.4f} {r['rho_max']:7.4f} {dr:7.4f}")

    # ---- Print unstable results (summary) -----------------------------------
    print()
    print("=" * 78)
    print(f"UNSTABLE / NO PHASE SEPARATION: {len(unstable_results)} / {total}")
    print("=" * 78)
    if unstable_results:
        # Group by failure mode
        diverged: list[dict] = [r for r in unstable_results if r["diverged_at"] >= 0]
        no_phase: list[dict] = [r for r in unstable_results if r["stable"] and not r["phase_separated"]]
        bad_model: list[dict] = [r for r in unstable_results if r["diverged_at"] == -2]
        if diverged:
            print(f"  Diverged (NaN/Inf): {len(diverged)}")
        if no_phase:
            print(f"  Stable but no phase separation: {len(no_phase)}")
        if bad_model:
            print(f"  Model validation failed (e.g. T >= Tc): {len(bad_model)}")

    print(f"\nDone. Stable region can be used as CS EOS starting parameters.")


if __name__ == "__main__":
    main()
