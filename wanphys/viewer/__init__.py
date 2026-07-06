# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""
Viewer interface for Newton physics simulations.

This module provides a high-level, renderer-agnostic interface for interactive
visualization of Newton models and simulation states.

Example usage:
    ```python
    import newton
    from newton.viewer import ViewerGL

    # Create viewer with OpenGL backend
    viewer = ViewerGL(model)

    # Render simulation
    while viewer.is_running():
        viewer.begin_frame(time)
        viewer.log_state(state)
        viewer.log_points(particle_positions)
        viewer.end_frame()

    viewer.close()
    ```
"""

from newton.viewer import ViewerFile
from newton.viewer import ViewerGL
from newton.viewer import ViewerNull
from newton.viewer import ViewerRerun
from newton.viewer import ViewerUSD
from .render_instance import log_points, log_lines, log_mesh

__all__ = [
    "ViewerFile",
    "ViewerGL",
    "ViewerNull",
    "ViewerRerun",
    "ViewerUSD",

    "log_points",
    "log_lines",
    "log_mesh"
]