"""Mesh file loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from newton._src.geometry.types import Mesh


def _convert_tristrip_to_triangles(strip: list[int]) -> np.ndarray:
    """Convert a triangle strip to individual triangles.

    Triangle strip winding: for strip [v0, v1, v2, v3, v4...]:
    - Triangle 0: (v0, v1, v2)
    - Triangle 1: (v1, v3, v2) - winding reversed
    - Triangle 2: (v2, v3, v4)
    - Triangle 3: (v3, v5, v4) - winding reversed

    Supports primitive restart with -1 indices. When -1 is encountered,
    the strip restarts and winding resets to even (CCW).

    Args:
        strip: List of vertex indices in triangle strip format

    Returns:
        Array of triangle indices, shape (n_triangles, 3)
    """
    if len(strip) < 3:
        return np.array([], dtype=np.int32).reshape(0, 3)

    triangles = []
    restart_index = -1

    # Split strip by restart markers
    i = 0
    while i < len(strip):
        # Find next segment (until restart marker or end)
        segment_start = i
        while i < len(strip) and strip[i] != restart_index:
            i += 1

        segment = strip[segment_start:i]

        # Convert this segment to triangles
        for j in range(len(segment) - 2):
            v0, v1, v2 = segment[j], segment[j+1], segment[j+2]

            # Alternate winding order within this segment
            if j % 2 == 0:
                # Even triangles: normal order (CCW)
                triangles.append([v0, v1, v2])
            else:
                # Odd triangles: reverse last two vertices to maintain CCW
                triangles.append([v0, v2, v1])

        # Skip the restart marker
        i += 1

    return np.array(triangles, dtype=np.int32) if triangles else np.array([], dtype=np.int32).reshape(0, 3)


def _load_ply_with_tristrips(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PLY file with triangle strips using plyfile library.

    Args:
        filepath: Path to PLY file

    Returns:
        Tuple of (vertices, faces) as numpy arrays
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError(
            "plyfile is required for triangle strip support. "
            "Install with: pip install plyfile"
        )

    plydata = PlyData.read(str(filepath))

    # Extract vertices
    vertex_data = plydata['vertex']
    vertices = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ]).astype(np.float32)

    # Check if tristrips element exists
    if 'tristrips' in plydata:
        # Extract tristrip data
        tristrips = plydata['tristrips']
        all_triangles = []

        for strip_data in tristrips['vertex_indices']:
            # Convert each strip to triangles
            triangles = _convert_tristrip_to_triangles(strip_data)
            all_triangles.append(triangles)

        # Combine all triangles
        faces = np.vstack(all_triangles) if all_triangles else np.array([], dtype=np.int32).reshape(0, 3)
    elif 'face' in plydata:
        # Regular triangle faces
        face_data = plydata['face']
        faces = np.vstack(face_data['vertex_indices']).astype(np.int32)
    else:
        faces = np.array([], dtype=np.int32).reshape(0, 3)

    return vertices, faces


def load_mesh_file(
    filepath: str | Path,
    scale: float = 1.0,
    compute_normals: bool = True,
    compute_inertia: bool = False,
    maxhullvert: int | None = None,
    color: tuple[float, float, float] | None = None,
) -> "Mesh":
    """Load mesh from file (PLY, OBJ, STL, GLTF, etc.) using trimesh.

    Loads mesh file using trimesh library (returns trimesh.Trimesh),
    extracts geometry data, and converts to newton.Mesh format.

    Automatically detects format based on file extension. Supports:
    - PLY (ASCII and binary) - including triangle strips
    - OBJ (Wavefront)
    - STL (binary and ASCII)
    - GLTF/GLB
    - COLLADA (DAE)
    - And many others via trimesh

    Args:
        filepath: Path to mesh file
        scale: Uniform scale factor applied to vertices
        compute_normals: Auto-compute vertex normals if not in file
        compute_inertia: Compute mass, COM, inertia tensor
        maxhullvert: Max vertices for convex hull (None = no limit)
        color: RGB color tuple (0-1 range), None = auto

    Returns:
        newton.Mesh object (finalized, ready to add to ModelBuilder)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If mesh loading fails or format unsupported

    Example:
        >>> from wanphys.utils import load_mesh_file
        >>> mesh = load_mesh_file("bunny.ply", scale=0.01)
        >>>
        >>> import newton
        >>> builder = newton.ModelBuilder()
        >>> builder.add_shape_mesh(pos=(0, 1, 0), mesh=mesh)
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for mesh loading. "
            "Install with: pip install newton-sim[importers]"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    # Load mesh using trimesh (auto-detects format)
    # Note: Using load_mesh() instead of deprecated load()
    # Returns: trimesh.Trimesh object
    try:
        tm = trimesh.load_mesh(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {filepath}: {e}")

    # Check if trimesh failed to load faces (happens with tristrips)
    # If so, try loading with plyfile for tristrip support
    if len(tm.vertices) == 0 or len(tm.faces) == 0:
        if filepath.suffix.lower() == '.ply':
            print(f"Trimesh failed to load faces, trying plyfile for tristrip support...")
            vertices, faces = _load_ply_with_tristrips(filepath)
            vertices = vertices * scale
            faces = faces.flatten()

            # Create a new trimesh object for normal computation
            if compute_normals and len(faces) > 0:
                tm = trimesh.Trimesh(vertices=vertices, faces=faces.reshape(-1, 3))
                normals = np.array(tm.vertex_normals, dtype=np.float32)
            else:
                normals = None
            uvs = None
        else:
            raise ValueError(f"Loaded mesh has no faces. File may be a point cloud or corrupt.")
    else:
        # Extract geometry from trimesh.Trimesh object
        vertices = np.array(tm.vertices, dtype=np.float32) * scale
        faces = np.array(tm.faces.flatten(), dtype=np.int32)

        # Normals
        normals = None
        if compute_normals or hasattr(tm, "vertex_normals"):
            if hasattr(tm, "vertex_normals"):
                normals = np.array(tm.vertex_normals, dtype=np.float32)
            else:
                tm.compute_vertex_normals()
                normals = np.array(tm.vertex_normals, dtype=np.float32)

        # UVs (if available)
        uvs = None
        if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
            uvs = np.array(tm.visual.uv, dtype=np.float32)

        # Normals
        normals = None
        if compute_normals or hasattr(tm, "vertex_normals"):
            if hasattr(tm, "vertex_normals"):
                normals = np.array(tm.vertex_normals, dtype=np.float32)
            else:
                tm.compute_vertex_normals()
                normals = np.array(tm.vertex_normals, dtype=np.float32)

        # UVs (if available)
        uvs = None
        if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
            uvs = np.array(tm.visual.uv, dtype=np.float32)

    # Create Newton mesh from extracted data
    # This converts trimesh.Trimesh -> newton.Mesh
    from newton._src.geometry.types import Mesh

    mesh = Mesh(
        vertices=vertices,
        indices=faces,
        normals=normals,
        uvs=uvs,
        color=color,
        compute_inertia=compute_inertia,
        maxhullvert=maxhullvert,
    )

    # Finalize (converts newton.Mesh to wp.Mesh internally)
    mesh.finalize()

    return mesh  # Returns newton.Mesh, not trimesh.Trimesh


def load_point_cloud(
    filepath: str | Path,
    scale: float = 1.0,
) -> np.ndarray:
    """Load point cloud from file (PLY, XYZ, PCD, etc.).

    Supports PLY files with vertex data but no faces, as well as
    dedicated point cloud formats.

    Args:
        filepath: Path to point cloud file
        scale: Uniform scale factor

    Returns:
        points: (N, 3) float32 array - point positions

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains no vertices

    Example:
        >>> from wanphys.utils import load_point_cloud
        >>> points = load_point_cloud("scan.ply", scale=0.001)
        >>> print(f"Loaded {len(points)} points")
        >>>
        >>> # Use for convex hull collision
        >>> import newton
        >>> mesh = newton.Mesh(points, [], compute_inertia=True)
        >>> hull = mesh.compute_convex_hull()
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required. Install with: pip install newton-sim[importers]"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    # Load using trimesh.load() which auto-detects format and returns appropriate type
    # - For point clouds (no faces): returns trimesh.PointCloud
    # - For meshes: returns trimesh.Trimesh
    try:
        geom = trimesh.load(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to load point cloud from {filepath}: {e}")

    # Extract vertices/points - check multiple attributes for compatibility
    points = None
    if hasattr(geom, "vertices") and len(geom.vertices) > 0:
        points = np.array(geom.vertices, dtype=np.float32) * scale
    elif hasattr(geom, "points") and len(geom.points) > 0:
        # PointCloud object
        points = np.array(geom.points, dtype=np.float32) * scale

    if points is None or len(points) == 0:
        raise ValueError(f"No vertices found in {filepath}")

    return points
