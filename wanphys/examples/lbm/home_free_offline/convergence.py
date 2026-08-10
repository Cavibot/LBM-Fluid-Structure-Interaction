# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Grid-consistent HOME-Free convergence configurations."""

from __future__ import annotations

from dataclasses import replace

from .config import (
    HomeFreeOfflineConfig,
    OfflineSceneLevel,
    OfflineSceneName,
    make_scene_config,
)


_LEVEL_SCALE = {
    OfflineSceneLevel.L0: 1.0,
    OfflineSceneLevel.L1: 0.5,
    OfflineSceneLevel.L2: 0.25,
}

_END_TIME = {
    OfflineSceneName.DAMBREAK: 400.0,
    OfflineSceneName.DAMBREAK_SPHERE: 80.0,
    OfflineSceneName.SPHERE_ENTRY: 40.0,
}


def make_convergence_config(
    scene: OfflineSceneName | str,
    level: OfflineSceneLevel | str,
    *,
    device: str = "cuda:0",
) -> HomeFreeOfflineConfig:
    """Refine space and time while preserving each scene's physical problem."""

    scene = OfflineSceneName(scene)
    level = OfflineSceneLevel(level)
    scale = _LEVEL_SCALE[level]
    base = make_scene_config(scene, device=device)
    base_dx = base.fluid.cell_size
    dx = base_dx * scale
    dt = (
        0.5 * scale * scale
        if scene is OfflineSceneName.DAMBREAK
        else base.fluid.time_step * scale
    )
    physical_extent = tuple(
        cells * base_dx for cells in base.fluid.resolution
    )
    resolution = tuple(int(round(length / dx)) for length in physical_extent)
    fluid = replace(
        base.fluid,
        resolution=resolution,
        cell_size=dx,
        time_step=dt,
    )
    initial_height = base.initial_liquid_height
    initial_x_range = base.initial_liquid_x_range
    if scene is OfflineSceneName.DAMBREAK:
        fluid = replace(
            fluid,
            # The workflow baseline keeps tau=0.65 at every level under
            # diffusive scaling. Lower-viscosity L2 stress cases are tracked
            # separately because their interface kinetic mode is unstable.
            kinematic_viscosity=0.10,
            body_acceleration=(0.0, 0.0, -5.625e-4),
            periodic=(False, True, False),
            max_lattice_speed=0.12,
        )
        # Keep the physical cuboid identical across levels while avoiding an
        # exactly face-aligned zero-thickness interface on any target grid.
        initial_height = 8.125
        initial_x_range = (0.0, 8.125)
    end_time = _END_TIME[scene]
    steps = int(round(end_time / dt))
    output_every = max(1, int(round(5.0 / dt)))
    return replace(
        base,
        level=level,
        steps=steps,
        output_every_steps=output_every,
        fluid=fluid,
        initial_liquid_height=initial_height,
        initial_liquid_x_range=initial_x_range,
    )
