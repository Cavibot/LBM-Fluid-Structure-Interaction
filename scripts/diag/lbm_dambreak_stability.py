#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
"""Long-horizon dam-break stability diagnostics for every legal LBM path.

This runner intentionally bypasses the visual example.  It exercises the
current solver with one common initial condition and reports the first step at
which a numerical admissibility invariant is violated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    FullFLbmState,
    HomeLbmState,
    LbmDomain,
    LbmModel,
    LbmStateBase,
)


LEGAL_PATHS: tuple[tuple[str, str], ...] = (
    ("fullf", "srt"),
    ("fullf", "trt"),
    ("fullf", "raw_mrt"),
    ("fullf", "nocm_mrt"),
    ("home", "srt"),
    ("home", "trt"),
    ("home", "nocm_mrt"),
)


@wp.kernel
def _initialize_dam_break_fullf(
    f_post: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    dam_x: int,
    rho_water: float,
    rho_air: float,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    rho = rho_water if i < dam_x else rho_air
    density[i, j, k] = rho
    f_post[0 * stride + idx] = rho / 3.0
    for q in range(1, 7):
        f_post[q * stride + idx] = rho / 18.0
    for q in range(7, 19):
        f_post[q * stride + idx] = rho / 36.0


@wp.kernel
def _initialize_dam_break_home(
    rho: wp.array3d(dtype=float),
    density: wp.array3d(dtype=float),
    dam_x: int,
    rho_water: float,
    rho_air: float,
) -> None:
    i, j, k = wp.tid()
    value = rho_water if i < dam_x else rho_air
    rho[i, j, k] = value
    density[i, j, k] = value


def _mirror_state(domain: LbmDomain) -> None:
    source = domain.state
    target = domain._state_out
    assert target is not None
    for name in (
        "density",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "force_x",
        "force_y",
        "force_z",
        "solid_phi",
        "solid_body_id",
        "vel_solid_u",
        "vel_solid_v",
        "vel_solid_w",
    ):
        wp.copy(getattr(target, name), getattr(source, name))
    if isinstance(source, FullFLbmState):
        assert isinstance(target, FullFLbmState)
        wp.copy(target.f_post, source.f_post)
    else:
        assert isinstance(source, HomeLbmState)
        assert isinstance(target, HomeLbmState)
        for destination, field in zip(target.kinetic_fields, source.kinetic_fields):
            wp.copy(destination, field)


def initialize_dam_break(
    domain: LbmDomain,
    rho_water: float = 1.8,
    rho_air: float = 0.1,
    dam_fraction: float = 0.25,
) -> float:
    state = domain.state
    nx, ny, nz = int(domain.model.nx), int(domain.model.ny), int(domain.model.nz)
    dam_x = max(1, int(nx * dam_fraction))
    if isinstance(state, FullFLbmState):
        wp.launch(
            _initialize_dam_break_fullf,
            dim=(nx, ny, nz),
            inputs=[state.f_post, state.density, dam_x, rho_water, rho_air, ny, nz, nx * ny * nz],
            device=domain.model._device,
        )
    else:
        assert isinstance(state, HomeLbmState)
        for field in state.kinetic_fields[1:]:
            field.zero_()
        wp.launch(
            _initialize_dam_break_home,
            dim=(nx, ny, nz),
            inputs=[state.rho, state.density, dam_x, rho_water, rho_air],
            device=domain.model._device,
        )
    _mirror_state(domain)
    wp.synchronize_device(domain.model._device)
    return float(np.sum(state.density.numpy(), dtype=np.float64))


@dataclass(frozen=True)
class StabilitySample:
    step: int
    density_min: float
    density_max: float
    mass_ratio: float
    speed_max: float
    population_min: float
    negative_population_fraction: float
    kinetic_magnitude_max: float
    all_finite: bool


@dataclass(frozen=True)
class StabilityResult:
    encoding: str
    collision: str
    passed: bool
    first_bad_step: int | None
    reason: str
    final: StabilitySample


def sample_domain(domain: LbmDomain, step: int, initial_mass: float) -> StabilitySample:
    wp.synchronize_device(domain.model._device)
    state: LbmStateBase = domain.state
    density = state.density.numpy()
    ux = state.velocity_x.numpy()
    uy = state.velocity_y.numpy()
    uz = state.velocity_z.numpy()
    streamed_populations = domain.solver._f_star.numpy()
    if isinstance(state, FullFLbmState):
        stored_populations = state.f_post.numpy()
    elif domain.solver._f_post is not None:
        stored_populations = domain.solver._f_post.numpy()
    else:
        # HOME-NOCM reconstructs its post-collision moments into _f_star when
        # post-collision admissibility enforcement is enabled.
        stored_populations = streamed_populations
    populations = (streamed_populations, stored_populations)
    kinetic_arrays = (
        tuple(field.numpy() for field in state.kinetic_fields)
        if isinstance(state, HomeLbmState)
        else (state.f_post.numpy(),)
    )
    finite = bool(
        np.isfinite(density).all()
        and np.isfinite(ux).all()
        and np.isfinite(uy).all()
        and np.isfinite(uz).all()
        and all(np.isfinite(field).all() for field in populations)
        and all(np.isfinite(field).all() for field in kinetic_arrays)
    )
    if finite:
        speed_max = float(np.sqrt(ux * ux + uy * uy + uz * uz).max())
        density_min = float(density.min())
        density_max = float(density.max())
        population_min = min(float(field.min()) for field in populations)
        population_size = sum(field.size for field in populations)
        negative_fraction = float(
            sum(np.count_nonzero(field < 0.0) for field in populations) / population_size
        )
        mass_ratio = float(np.sum(density, dtype=np.float64) / initial_mass)
        kinetic_magnitude_max = max(float(np.max(np.abs(field))) for field in kinetic_arrays)
    else:
        speed_max = math.inf
        density_min = float(np.nanmin(density)) if np.isfinite(density).any() else math.nan
        density_max = float(np.nanmax(density)) if np.isfinite(density).any() else math.nan
        finite_population_minima = [
            float(np.nanmin(field))
            for field in populations
            if np.isfinite(field).any()
        ]
        population_min = min(finite_population_minima, default=math.nan)
        population_size = sum(field.size for field in populations)
        negative_fraction = float(
            sum(np.count_nonzero(field < 0.0) for field in populations) / population_size
        )
        mass_ratio = float(np.nansum(density, dtype=np.float64) / initial_mass)
        finite_kinetic_maxima = [
            float(np.nanmax(np.abs(field)))
            for field in kinetic_arrays
            if np.isfinite(field).any()
        ]
        kinetic_magnitude_max = max(finite_kinetic_maxima, default=math.inf)
    return StabilitySample(
        step=step,
        density_min=density_min,
        density_max=density_max,
        mass_ratio=mass_ratio,
        speed_max=speed_max,
        population_min=population_min,
        negative_population_fraction=negative_fraction,
        kinetic_magnitude_max=kinetic_magnitude_max,
        all_finite=finite,
    )


def _failure_reason(sample: StabilitySample, mass_tolerance: float, speed_limit: float) -> str | None:
    if not sample.all_finite:
        return "non-finite density, velocity, or population"
    if sample.density_min <= 0.0:
        return f"non-positive density ({sample.density_min:.6g})"
    if sample.population_min < -1.0e-7:
        return f"negative logical population ({sample.population_min:.6g})"
    if abs(sample.mass_ratio - 1.0) > mass_tolerance:
        return f"mass drift ({sample.mass_ratio - 1.0:+.3e})"
    if sample.speed_max > speed_limit:
        return f"lattice speed exceeded {speed_limit:g} ({sample.speed_max:.6g})"
    return None


def run_case(
    encoding: str,
    collision: str,
    *,
    resolution: int = 24,
    steps: int = 180,
    sample_every: int = 5,
    device: str = "cpu",
    gravity: float = -0.005,
    gravity_ramp_steps: int = 60,
    g_sc: float = -5.0,
    sc_force_stride: int = 2,
    regularize_all: bool = False,
    regularization_strength: float = 0.5,
    mass_tolerance: float = 5.0e-4,
    speed_limit: float = 0.6,
    max_lattice_speed: float = 0.4,
) -> StabilityResult:
    use_trt = collision == "trt"
    model = LbmModel(
        fluid_grid_res=(resolution, resolution, resolution),
        device=device,
        encoding=encoding,
        collision=collision,
        interface_model="shan_chen",
        tau=0.55,
        lambda_trt=0.03 if use_trt else 0.0,
        use_regularization=use_trt,
        omega_reg=0.5 if use_trt else 0.0,
        G=g_sc,
        psi_type=1,
        psi_ref=1.0,
        sc_boundary_psi=-1.0,
        sc_force_stride=sc_force_stride,
        gravity_z=gravity,
        max_lattice_speed=max_lattice_speed,
    )
    domain = LbmDomain(model)
    if regularize_all:
        # Diagnostic ablation: the kernel itself is collision agnostic even
        # though the public model currently exposes it only for TRT.
        model.use_regularization = True
        model.omega_reg = regularization_strength
    domain.create_state()
    initial_mass = initialize_dam_break(domain)
    target_gravity = gravity
    model.gravity_z = 0.0
    final = sample_domain(domain, 0, initial_mass)
    for step in range(1, steps + 1):
        model.gravity_z = target_gravity * min(step, gravity_ramp_steps) / gravity_ramp_steps
        domain.step(1.0)
        if step % sample_every != 0 and step != steps:
            continue
        final = sample_domain(domain, step, initial_mass)
        reason = _failure_reason(final, mass_tolerance, speed_limit)
        if reason is not None:
            return StabilityResult(encoding, collision, False, step, reason, final)
    return StabilityResult(encoding, collision, True, None, "ok", final)


def _format_result(result: StabilityResult) -> str:
    sample = result.final
    status = "PASS" if result.passed else "FAIL"
    return (
        f"[{status}] {result.encoding:5s}+{result.collision:9s} "
        f"step={sample.step:4d} rho=[{sample.density_min:.4g},{sample.density_max:.4g}] "
        f"mass={sample.mass_ratio:.7f} umax={sample.speed_max:.4g} "
        f"fmin={sample.population_min:.4g} neg={sample.negative_population_fraction:.3%} "
        f"kmax={sample.kinetic_magnitude_max:.4g} "
        f"reason={result.reason}"
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", "-n", type=int, default=24)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gravity", type=float, default=-0.005)
    parser.add_argument("--max-lattice-speed", type=float, default=0.4)
    parser.add_argument("--sc-force-stride", type=int, default=2)
    parser.add_argument("--regularize-all", action="store_true")
    parser.add_argument("--regularization-strength", type=float, default=0.5)
    parser.add_argument(
        "--path",
        action="append",
        metavar="ENCODING+COLLISION",
        help="run selected path; may be repeated (default: all legal paths)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    selected = LEGAL_PATHS
    if args.path:
        selected = tuple(tuple(value.split("+", 1)) for value in args.path)  # type: ignore[assignment]
        unknown = tuple(path for path in selected if path not in LEGAL_PATHS)
        if unknown:
            raise SystemExit(f"unsupported path(s): {unknown}")
    results = []
    for encoding, collision in selected:
        result = run_case(
            encoding,
            collision,
            resolution=args.resolution,
            steps=args.steps,
            sample_every=args.sample_every,
            device=args.device,
            gravity=args.gravity,
            max_lattice_speed=args.max_lattice_speed,
            sc_force_stride=args.sc_force_stride,
            regularize_all=args.regularize_all,
            regularization_strength=args.regularization_strength,
        )
        results.append(result)
        print(_format_result(result), flush=True)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
