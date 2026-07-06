# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
"""
mesh_to_points_api.py
---------------------

Callable APIs extracted from `mesh_to_points_warp.py`.

Supported:
1) OBJ -> (optional) PLY / TXT outputs
2) In-memory mesh (V,F) -> sampled points (P,N,face_id,layer_id)

About "layer":
- `mesh_to_points_warp.py` is explicitly "layers=0 fast path", and its sampler only produces
  surface samples (P, normals, face_id). See file header comment. 
- Therefore, multi-layer ("surface + inward layers") sampling is implemented at the API level by:
    * Running the surface sampler for each layer independently (different seeds)
    * Offsetting points along normal direction by k * layer_gap (k = 0..layers-1)
    * Assigning layer_id = k

This provides practical multi-layer point clouds for downstream pipelines.
If you need Poisson constraint **across** layers (3D volume blue-noise), you need a different sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mesh_to_points_warp import (
    load_obj,
    normalize_mesh,
    triangle_info,
    vertex_normals_area_weighted,
    poisson_sample_surface_warp,
    save_ply_ascii_points,
    save_txt_points,
)


@dataclass
class MeshToPointsResult:
    """Return container for mesh -> points conversion (possibly multi-layer)."""
    points: np.ndarray      # (N,3) float32
    normals: np.ndarray     # (N,3) float32
    face_id: np.ndarray     # (N,)  int32
    layer_id: np.ndarray    # (N,)  int32, 0..layers-1 (or custom if you post-process)
    radius_used: float      # sampling radius actually used (after optional normalize scaling)
    layer_gap_used: float   # layer gap actually used (after optional normalize scaling)
    layers: int             # number of layers generated
    scale: float            # normalize scale factor (1.0 when normalize=False)
    normalized: bool        # whether normalization was applied


def mesh_to_points(
    V: np.ndarray,
    F: np.ndarray,
    radius: float,
    point_count: int,
    *,
    device: str = "cuda:0",
    seed: int = 0,
    normalize: bool = True,
    # multilayer controls
    layers: int = 1,
    layer_gap: float = 0.0,
    inward: bool = True,
    # sampler knobs
    max_rounds: int = 200,
    batch_factor: int = 8,
    batch_min: int = 65536,
    batch_max: int = 1_048_576,
) -> MeshToPointsResult:
    """
    In-memory mesh -> points conversion (supports multi-layer by normal-offset).

    Parameters
    ----------
    V : (nV, 3) array-like
        Vertex positions.
    F : (nF, 3) array-like
        Triangle indices.
    radius : float
        Poisson-like minimum distance for **each layer** (surface sampling radius).
    point_count : int
        Number of points **per layer**.

    Keyword-only parameters
    -----------------------
    device : str
        "cuda:0" or "cpu" etc.
    seed : int
        RNG seed base used by Warp; each layer uses seed + 10007*k.
    normalize : bool
        If True, follow the original CLI behavior:
            Vn, scale = normalize_mesh(V)
            radius_used = radius * scale
            layer_gap_used = layer_gap * scale
        If False:
            Vn = V
            radius_used = radius
            layer_gap_used = layer_gap

    Multi-layer parameters
    ----------------------
    layers : int
        Number of layers to generate. layer_id will be 0..layers-1.
    layer_gap : float
        Offset distance between consecutive layers along normal direction.
    inward : bool
        If True: offset along -normal (typically "into" the object).
        If False: offset along +normal.

    Notes
    -----
    - This implementation enforces Poisson radius **within each layer** independently.
      It does NOT enforce min-distance between points of different layers.
    - The offset is along the sampled normal; for complex geometry, points may leave the solid or self-intersect.

    Returns
    -------
    MeshToPointsResult
    """
    V = np.asarray(V)
    F = np.asarray(F)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be shaped (n,3), got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must be shaped (m,3), got {F.shape}")

    if layers <= 0:
        raise ValueError(f"layers must be >= 1, got {layers}")

    V = V.astype(np.float32, copy=False)
    F = F.astype(np.int32, copy=False)

    if normalize:
        Vn, scale, _, _ = normalize_mesh(V)
        radius_used = float(radius) * float(scale)
        layer_gap_used = float(layer_gap) * float(scale)
    else:
        Vn = V
        scale = 1.0
        radius_used = float(radius)
        layer_gap_used = float(layer_gap)

    # Precompute normals once on normalized mesh
    tri_area, tri_n = triangle_info(Vn, F)
    VN = vertex_normals_area_weighted(Vn, F, tri_area, tri_n)

    sign = -1.0 if inward else 1.0

    Ps: list[np.ndarray] = []
    Ns: list[np.ndarray] = []
    Fs: list[np.ndarray] = []
    Ls: list[np.ndarray] = []

    for k in range(int(layers)):
        P, Nrm, FaceID = poisson_sample_surface_warp(
            Vn,
            F,
            VN,
            float(radius_used),
            int(point_count),
            device=device,
            seed=int(seed + 10007 * k),
            max_rounds=max_rounds,
            batch_factor=batch_factor,
            batch_min=batch_min,
            batch_max=batch_max,
        )

        if layer_gap_used != 0.0 and k != 0:
            offset = (sign * float(layer_gap_used) * float(k))
            P = P + Nrm * offset

        Ps.append(P.astype(np.float32, copy=False))
        Ns.append(Nrm.astype(np.float32, copy=False))
        Fs.append(FaceID.astype(np.int32, copy=False))
        Ls.append(np.full((P.shape[0],), k, dtype=np.int32))

    P_all = np.concatenate(Ps, axis=0) if len(Ps) > 1 else Ps[0]
    N_all = np.concatenate(Ns, axis=0) if len(Ns) > 1 else Ns[0]
    F_all = np.concatenate(Fs, axis=0) if len(Fs) > 1 else Fs[0]
    L_all = np.concatenate(Ls, axis=0) if len(Ls) > 1 else Ls[0]

    return MeshToPointsResult(
        points=P_all,
        normals=N_all,
        face_id=F_all,
        layer_id=L_all,
        radius_used=float(radius_used),
        layer_gap_used=float(layer_gap_used),
        layers=int(layers),
        scale=float(scale),
        normalized=bool(normalize),
    )


def obj_to_outputs(
    obj_path: str,
    *,
    ply_path: Optional[str] = None,
    txt_path: Optional[str] = None,
    radius: float,
    point_count: int,
    device: str = "cuda:0",
    seed: int = 0,
    normalize: bool = True,
    # multilayer controls
    layers: int = 1,
    layer_gap: float = 0.0,
    inward: bool = True,
    # sampler knobs
    max_rounds: int = 200,
    batch_factor: int = 8,
    batch_min: int = 65536,
    batch_max: int = 1_048_576,
) -> MeshToPointsResult:
    """
    OBJ -> points, and optionally export to PLY and/or TXT.

    Output rules:
      - If ply_path is not None/"" -> write ASCII PLY: x y z nx ny nz layer_id face_id
      - If txt_path is not None/"" -> write TXT: header + x y z layer_id face_id
      - If both are given -> write both
      - If neither is given -> no file is written, but the in-memory result is still returned

    Multi-layer behavior:
      - Generate `layers` layers, each has `point_count` points.
      - layer k is offset by k*layer_gap along normal direction (inward by default).
    """
    V, F = load_obj(obj_path)

    res = mesh_to_points(
        V, F, radius, point_count,
        device=device,
        seed=seed,
        normalize=normalize,
        layers=layers,
        layer_gap=layer_gap,
        inward=inward,
        max_rounds=max_rounds,
        batch_factor=batch_factor,
        batch_min=batch_min,
        batch_max=batch_max,
    )

    if ply_path:
        save_ply_ascii_points(
            ply_path,
            res.points,
            res.normals,
            res.layer_id,
            res.face_id,
            res.radius_used,
        )

    if txt_path:
        save_txt_points(
            txt_path,
            res.points,
            res.layer_id,
            res.face_id,
            res.radius_used,
        )

    return res
