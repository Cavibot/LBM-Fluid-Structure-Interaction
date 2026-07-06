# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Utility functions to reduce boilerplate in WanPhys examples.

This module provides helper functions for common setup patterns:
- Asset loading from wanphys/assets folder
- Warp initialization and device selection
- Simulation parameter configuration
- Viewer setup and camera configuration
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import warp as wp

import newton
import newton.examples

# =============================================================================
# Asset Loading
# =============================================================================

# Cache for wanphys assets directory
_WANPHYS_ASSETS_DIR: Optional[Path] = None
def get_assets_dir() -> Path:
    """Get the path to the wanphys/assets directory.

    Returns:
        Path to the assets directory.
    """
    global _WANPHYS_ASSETS_DIR
    if _WANPHYS_ASSETS_DIR is None:
        # Go up from utils.py -> examples -> wanphys -> assets
        _WANPHYS_ASSETS_DIR = Path(__file__).parent.parent / "assets"
    return _WANPHYS_ASSETS_DIR


def get_asset(name: str) -> str:
    """Get the full path to a WanPhys asset file.

    Args:
        name: Name of the asset file (e.g., "dragon.usda").

    Returns:
        Full path to the asset file as a string.

    Raises:
        FileNotFoundError: If the asset doesn't exist.

    Example:
        >>> dragon_path = get_asset("dragon.usda")
        >>> print(dragon_path)
        /path/to/wanphys/assets/dragon.usda
    """
    asset_path = get_assets_dir() / name
    if not asset_path.exists():
        raise FileNotFoundError(
            f"WanPhys asset '{name}' not found at {asset_path}. "
            f"Available assets: {list(get_assets_dir().glob('*'))}"
        )
    return str(asset_path)


# =============================================================================
# Asset Loading
# =============================================================================

# Cache for wanphys assets directory
_WANPHYS_ASSETS_DIR: Optional[Path] = None


def get_assets_dir() -> Path:
    """Get the path to the wanphys/assets directory.

    Returns:
        Path to the assets directory.
    """
    global _WANPHYS_ASSETS_DIR
    if _WANPHYS_ASSETS_DIR is None:
        # Go up from utils.py -> examples -> wanphys -> assets
        _WANPHYS_ASSETS_DIR = Path(__file__).parent.parent / "assets"
    return _WANPHYS_ASSETS_DIR


def get_asset(name: str) -> str:
    """Get the full path to a WanPhys asset file.

    Args:
        name: Name of the asset file (e.g., "dragon.usda").

    Returns:
        Full path to the asset file as a string.

    Raises:
        FileNotFoundError: If the asset doesn't exist.

    Example:
        >>> dragon_path = get_asset("dragon.usda")
        >>> print(dragon_path)
        /path/to/wanphys/assets/dragon.usda
    """
    asset_path = get_assets_dir() / name
    if not asset_path.exists():
        raise FileNotFoundError(
            f"WanPhys asset '{name}' not found at {asset_path}. "
            f"Available assets: {list(get_assets_dir().glob('*'))}"
        )
    return str(asset_path)


@dataclass
class SimulationParams:
    """Simulation timing parameters.

    Attributes:
        fps: Target frames per second for rendering/output.
        frame_dt: Time step per frame (1/fps).
        sim_substeps: Number of physics substeps per frame.
        sim_dt: Physics timestep (frame_dt / substeps).
    """

    fps: int
    frame_dt: float
    sim_substeps: int
    sim_dt: float


def init_warp(device: Optional[str] = None) -> str:
    """Initialize Warp and select compute device.

    Args:
        device: Device string ("cuda:0", "cpu", etc.). If None, auto-selects
                "cuda:0" if CUDA is available, otherwise "cpu".

    Returns:
        Selected device string.

    Example:
        >>> device = init_warp()
        Using device: cuda:0
    """
    wp.init()

    if device is None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"

    print(f"Using device: {device}")
    return device


def init_simulation_params(fps: int = 60, substeps: int = 4) -> SimulationParams:
    """Create simulation timing parameters.

    Args:
        fps: Target frames per second (default: 60).
        substeps: Physics substeps per frame (default: 4).
                  Higher values = more stable but slower.

    Returns:
        SimulationParams with computed timesteps.

    Example:
        >>> params = init_simulation_params(fps=60, substeps=4)
        >>> params.sim_dt
        0.004166666666666667  # 1/240
    """
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / substeps

    return SimulationParams(
        fps=fps,
        frame_dt=frame_dt,
        sim_substeps=substeps,
        sim_dt=sim_dt,
    )


def setup_viewer(
    viewer,
    model,
    state,
    camera_pos: Optional[tuple] = None,
    camera_pitch: Optional[float] = None,
    camera_yaw: Optional[float] = None,
    camera_target: Optional[tuple] = None,
    collision_pipeline=None,
    collision_args=None,
):
    """Configure Newton viewer for WanPhys simulation.

    Sets up viewer with model, optionally positions camera, and creates collision pipeline.

    Args:
        viewer: Newton viewer instance (from newton.examples.init()).
        model: WanPhys model with viewer bridge support.
        state: WanPhys state with viewer bridge support.
        camera_pos: Camera position as (x, y, z) tuple (optional).
        camera_pitch: Camera pitch angle in degrees (up/down rotation, optional).
        camera_yaw: Camera yaw angle in degrees (left/right rotation, optional).
        camera_target: Camera target position as (x, y, z) tuple (alternative to pitch/yaw).
        collision_pipeline: Pre-created collision pipeline (optional).
        collision_args: Arguments for collision pipeline creation (optional).

    Returns:
        Tuple of (contacts, collision_pipeline) for rendering.

    Example:
        >>> contacts, pipeline = setup_viewer(
        ...     viewer, model, state,
        ...     camera_pos=(0, 5, 12),
        ...     camera_pitch=-10,
        ...     camera_yaw=-180
        ... )
    """
    # Set model through the WanPhys viewer bridge.
    if hasattr(model, "setup_viewer"):
        model.setup_viewer(viewer)
        newton_model = model.as_newton_model()
    else:
        newton_model = model._newton_backend if hasattr(model, "_newton_backend") else model
        viewer.set_model(newton_model)

    # Set camera position
    if camera_pos is not None:
        if camera_target is not None:
            # Use target-based camera
            viewer.set_camera_target(
                pos=wp.vec3(*camera_pos),
                target=wp.vec3(*camera_target),
            )
        elif camera_pitch is not None or camera_yaw is not None:
            # Use angle-based camera
            viewer.set_camera(
                pos=wp.vec3(*camera_pos),
                pitch=camera_pitch or 0.0,
                yaw=camera_yaw or 0.0,
            )
        else:
            # Just position, use defaults for angles
            viewer.set_camera(pos=wp.vec3(*camera_pos))

    # Create collision pipeline if needed
    if collision_pipeline is None:
        collision_pipeline = newton.examples.create_collision_pipeline(newton_model, collision_args)

    # Compute initial contacts through the viewer collision bridge.
    contacts = newton_model.collide(state.as_newton_state(), collision_pipeline=collision_pipeline)

    return contacts, collision_pipeline
