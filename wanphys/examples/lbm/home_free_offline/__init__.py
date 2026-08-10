# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Versioned, headless scenes for final HOME-Free offline rendering."""

from .acceptance import (
    OfflineSceneAcceptance,
    evaluate_scene_trace,
    run_scene_acceptance,
)
from .config import (
    CameraHint,
    CouplingConfig,
    FluidConfig,
    HomeFreeOfflineConfig,
    OfflineSceneLevel,
    OfflineSceneName,
    SphereConfig,
    make_scene_config,
)
from .factory import (
    build_dambreak_fill,
    build_offline_scene,
    build_rectangular_liquid_fill,
)
from .scene import HomeFreeOfflineScene, OfflineSceneMetrics

__all__ = [
    "CameraHint",
    "CouplingConfig",
    "FluidConfig",
    "HomeFreeOfflineConfig",
    "HomeFreeOfflineScene",
    "OfflineSceneAcceptance",
    "OfflineSceneLevel",
    "OfflineSceneMetrics",
    "OfflineSceneName",
    "SphereConfig",
    "build_dambreak_fill",
    "build_offline_scene",
    "build_rectangular_liquid_fill",
    "evaluate_scene_trace",
    "make_scene_config",
    "run_scene_acceptance",
]
