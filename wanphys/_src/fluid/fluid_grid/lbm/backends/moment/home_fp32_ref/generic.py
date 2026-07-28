# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Generic HOME model presets (shared base; free-surface is a branch).

Use these kwargs when constructing ``LbmModel(..., lbm_backend='home_fp32')``
so single-phase and VOF scenes share one baseline. Showcase demos override
individual flags explicitly.
"""

from __future__ import annotations

from typing import Any

from wanphys._src.fluid.fluid_grid.lbm.model import LbmModel


def generic_home_flags() -> dict[str, Any]:
    """Return opt-in HOME heuristic flags in the **off** state.

    Used for both ``phase_mode='none'`` and ``vof_sharp``. Free-surface
    topology defaults (``vof_seal_fg``, ``vof_gamma``) are set by
    :func:`make_home_model` according to the branch.
    """
    return {
        "vof_wall_film_drain": False,
        "vof_height_eq": False,
        "vof_quiet_fill": False,
        "vof_orphan_reabsorb": False,
        "vof_bubble_pressure": False,
        "vof_bubble_disjoint": False,
        "vof_bubble_small_sigma": False,
        "vof_bubble_eddy": False,
        "vof_home_fill_empty": False,
        "vof_home_wall_eq": False,
        "vof_home_moment_quant": False,
        "vof_home_cuda_graph": False,
        "vof_wall_wetting": 0.0,
    }


def generic_home_vof_flags() -> dict[str, Any]:
    """Alias of :func:`generic_home_flags` (back-compat name)."""
    return generic_home_flags()


def make_home_model(
    *,
    fluid_grid_res: tuple[int, int, int],
    fluid_grid_cell_size: float,
    phase_mode: str = "none",
    tau: float = 0.51,
    gravity_x: float = 0.0,
    gravity_y: float = 0.0,
    gravity_z: float = 0.0,
    vof_gamma: float = 0.0,
    lattice: str = "D3Q27",
    initial_density: float = 1.0,
    lambda_trt: float = 0.0,
    **overrides: Any,
) -> LbmModel:
    """Build ``home_fp32`` on the shared HOME base.

    - ``phase_mode='none'``: full-liquid single-phase (free-surface flags off).
    - ``phase_mode='vof_sharp'``: HOME-FREE free-surface branch.
    """
    mode = str(phase_mode).lower()
    if mode not in ("none", "vof_sharp"):
        raise ValueError(
            f"make_home_model phase_mode must be 'none' or 'vof_sharp', got {phase_mode!r}"
        )
    kwargs: dict[str, Any] = {
        "fluid_grid_res": fluid_grid_res,
        "fluid_grid_cell_size": float(fluid_grid_cell_size),
        "lattice": lattice,
        "tau": float(tau),
        "G": 0.0,
        "phase_mode": mode,
        "lbm_backend": "home_fp32",
        "vof_rho_gas": 1.0,
        "vof_epsilon": 1.0e-4,
        "vof_gamma": float(vof_gamma) if mode == "vof_sharp" else 0.0,
        "vof_kappa_smooth": 2,
        "vof_seal_fg": mode == "vof_sharp",
        "lambda_trt": float(lambda_trt),
        "initial_density": float(initial_density),
        "gravity_x": float(gravity_x),
        "gravity_y": float(gravity_y),
        "gravity_z": float(gravity_z),
    }
    kwargs.update(generic_home_flags())
    if mode == "none":
        kwargs["vof_seal_fg"] = False
        kwargs["vof_gamma"] = 0.0
    kwargs.update(overrides)
    return LbmModel(**kwargs)


def make_home_vof_model(
    *,
    fluid_grid_res: tuple[int, int, int],
    fluid_grid_cell_size: float,
    tau: float = 0.51,
    gravity_z: float = -0.002,
    vof_gamma: float = 1.5e-3,
    lattice: str = "D3Q27",
    initial_density: float = 1.0,
    lambda_trt: float = 0.015,
    **overrides: Any,
) -> LbmModel:
    """Build an ``LbmModel`` for reusable HOME-FREE VOF (heuristics off).

    Thin wrapper around :func:`make_home_model` with ``phase_mode='vof_sharp'``.
    """
    return make_home_model(
        fluid_grid_res=fluid_grid_res,
        fluid_grid_cell_size=fluid_grid_cell_size,
        phase_mode="vof_sharp",
        tau=tau,
        gravity_z=gravity_z,
        vof_gamma=vof_gamma,
        lattice=lattice,
        initial_density=initial_density,
        lambda_trt=lambda_trt,
        **overrides,
    )
