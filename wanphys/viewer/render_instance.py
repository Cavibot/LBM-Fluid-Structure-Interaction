# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

def log_points(viewer, name, points, radii, colors, hidden=False):
    """
    Log points in viewer for rendering.

    Args:
        viewer: Current viewer for rendering.
        name (str): Unique name for the point batch.
        points (wp.array): Array of point positions.
        radii (wp.array): Array of point radius values.
        colors (wp.array): Array of point colors.
        hidden (bool): Whether the points are hidden.
    """
    viewer.log_points(name=name, points=points, radii=radii, colors=colors, hidden=hidden)

def log_lines(viewer, name, starts, ends, colors, width: float = 0.01, hidden=False):
    """
    log lines in viewer for rendering.

    Args:
        viewer: Current viewer for rendering.
        name (str): Unique identifier for the line batch.
        starts (wp.array): Array of line start positions (shape: [N, 3]) or None for empty.
        ends (wp.array): Array of line end positions (shape: [N, 3]) or None for empty.
        colors (wp.array): Array of line colors (shape: [N, 3]) or tuple/list of RGB or None for empty.
        width (float): The width of the lines
        hidden (bool): Whether the lines are initially hidden.
    """
    viewer.log_lines(name=name, starts=starts, ends=ends, colors=colors, width=width, hidden=hidden)

def log_mesh(viewer, name, points, indices, normals, uvs, hidden=False, backface_culling=True):
    """
    Log a mesh in viewer for rendering.

    Args:
        viewer: Current viewer for rendering.
        name (str): Unique name for the mesh.
        points (wp.array): Vertex positions.
        indices (wp.array): Triangle indices.
        normals (wp.array, optional): Vertex normals.
        uvs (wp.array, optional): Vertex UVs.
        hidden (bool): Whether the mesh is hidden.
        backface_culling (bool): Enable backface culling.
    """
    viewer.log_mesh(name=name, points=points, indices=indices, normals=normals, uvs=uvs, hidden=hidden, backface_culling=backface_culling)
