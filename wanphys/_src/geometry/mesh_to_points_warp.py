# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""
mesh_to_points_warp.py
----------------------
Warp-accelerated surface Poisson-like sampling for OBJ triangle meshes (layers=0 fast path).

What is accelerated:
- The expensive neighbor checks (min-distance enforcement) are done on GPU using wp.HashGrid,
  using wp.hash_grid_query() + wp.hash_grid_query_next() as shown in NVIDIA's Warp examples.   (see citations below)

Algorithm (batched, GPU-friendly):
1) Precompute triangle-area alias table on CPU (O(nF)).
2) Iterate in rounds until we accept `target_count` points:
   a) Generate M candidate points on triangles (GPU, parallel) with face_id + normal.
   b) Build a candidate HashGrid over all candidates.
   c) If we already accepted some points, build an accepted HashGrid over accepted points.
   d) Prefilter candidates that are farther than radius from accepted points.
   e) Mutual-cull candidates within the batch: keep only the lowest-index candidate in any radius-neighborhood.
   f) Append survivors to the accepted arrays using an atomic counter.

This produces a blue-noise / Poisson-like distribution close to classic dart-throwing,
but not bit-identical to the serial CPU rejection sampler.

Outputs:
- ASCII PLY: x y z nx ny nz layer_id face_id
- TXT: header + x y z layer_id face_id

Requirements:
- warp-lang (NVIDIA Warp)
  pip install warp-lang

References for HashGrid build + query API:
- NVIDIA Warp blog example uses:
    grid = wp.HashGrid(dim_x=..., dim_y=..., dim_z=..., device="cuda")
    grid.build(points=p, radius=r)
    query = wp.hash_grid_query(grid, p, radius)
    while(wp.hash_grid_query_next(query, index)): ...
  https://developer.nvidia.com/blog/creating-differentiable-graphics-and-physics-simulation-in-python-with-nvidia-warp/   (lines 171-206)

NOTE:
- Warp HashGrid tiles periodically by its dimensions, so choose sufficiently large dims to avoid wrap-around
  interference. We pick dims as next power-of-two above bbox_extent/cell_size.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------
# Minimal OBJ loader (v, f)
# ----------------------------
def _parse_index(tok: str, vcount: int) -> int:
    slash = tok.find("/")
    a = tok if slash < 0 else tok[:slash]
    if not a:
        return -1
    idx = int(a)
    if idx < 0:
        idx = vcount + idx
    else:
        idx -= 1
    return idx


def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    V: List[List[float]] = []
    F: List[List[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    continue
                idx: List[int] = []
                for t in parts:
                    vi = _parse_index(t, len(V))
                    if vi >= 0:
                        idx.append(vi)
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    F.append([idx[0], idx[i], idx[i + 1]])

    if len(V) == 0 or len(F) == 0:
        raise RuntimeError("OBJ has no vertices or faces.")
    return np.asarray(V, dtype=np.float32), np.asarray(F, dtype=np.int32)


# ----------------------------
# Geometry helpers
# ----------------------------
def safe_normalize(v: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def triangle_info(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    nn = np.cross(b - a, c - a)
    lens = np.linalg.norm(nn, axis=1)
    area = 0.5 * lens
    n = np.zeros_like(nn, dtype=np.float32)
    good = lens > 1e-20
    n[good] = nn[good] / lens[good, None]
    n[~good] = np.array([0, 0, 1], dtype=np.float32)
    return area.astype(np.float32), n.astype(np.float32)


def vertex_normals_area_weighted(V: np.ndarray, F: np.ndarray, tri_area: np.ndarray, tri_n: np.ndarray) -> np.ndarray:
    VN = np.zeros_like(V, dtype=np.float32)
    w = tri_area.astype(np.float32)
    for k in range(3):
        np.add.at(VN, F[:, k], tri_n * w[:, None])
    VN = safe_normalize(VN)
    return VN.astype(np.float32)


def normalize_mesh(V: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    bmin = V.min(axis=0)
    bmax = V.max(axis=0)
    center = 0.5 * (bmin + bmax)
    diag = float(np.linalg.norm(bmax - bmin))
    scale = (2.0 / diag) if diag > 1e-8 else 1.0
    Vn = (V - center) * scale
    bminN = Vn.min(axis=0)
    bmaxN = Vn.max(axis=0)
    return Vn.astype(np.float32), float(scale), bminN.astype(np.float32), bmaxN.astype(np.float32)


# ----------------------------
# Alias table for triangle sampling (CPU)
# ----------------------------
def build_alias_table(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vose alias method.
    Returns:
      prob: (K,) float32 in [0,1]
      alias:(K,) int32
    """
    w = np.asarray(weights, dtype=np.float64)
    K = w.size
    if K == 0:
        raise ValueError("Empty weights.")
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Invalid weights sum.")
    w = w * (K / w_sum)
    prob = np.zeros(K, dtype=np.float64)
    alias = np.zeros(K, dtype=np.int32)
    small = []
    large = []
    for i, wi in enumerate(w):
        (small if wi < 1.0 else large).append(i)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = w[s]
        alias[s] = l
        w[l] = (w[l] + w[s]) - 1.0
        if w[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in large + small:
        prob[i] = 1.0
        alias[i] = i

    return prob.astype(np.float32), alias.astype(np.int32)


def next_pow2(x: int) -> int:
    x = max(1, int(x))
    return 1 << (x - 1).bit_length()


# ----------------------------
# Warp sampling
# ----------------------------
def _require_warp():
    try:
        import warp as wp  # noqa: F401
    except Exception as e:
        raise RuntimeError("warp is not available. Install with: pip install warp-lang\n" + str(e))


def poisson_sample_surface_warp(V: np.ndarray,
                                F: np.ndarray,
                                VN: np.ndarray,
                                radius: float,
                                target_count: int,
                                *,
                                device: str = "cuda:0",
                                seed: int = 0,
                                max_rounds: int = 200,
                                batch_factor: int = 8,
                                batch_min: int = 65536,
                                batch_max: int = 1_048_576) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      P: (N,3) float32
      Nrm:(N,3) float32
      FaceID:(N,) int32
    """
    _require_warp()
    import warp as wp

    wp.init()

    # Filter degenerate triangles for sampling pool
    tri_area, tri_n = triangle_info(V, F)
    area_total = float(np.sum(tri_area[tri_area > 1e-12]))
    # Rough upper bound for Poisson-disk on surface:
    # treat each point as center of a disk of radius r/2; best-case packing density ~ 0.9069.
    # => max points per unit area ~ 0.9069 / (pi*(r/2)^2) = 1.1547 / r^2
    if radius > 0.0 and area_total > 0.0:
        nmax_est = 1.154700538 * area_total / (radius * radius)
        if target_count > 5.0 * nmax_est:
            print(f"[WARN] Requested {target_count} points with radius={radius:.6g} on area~{area_total:.6g}. "
                  f"Estimated max ~{int(nmax_est)}. This request is likely infeasible; sampling may stall. "
                  f"Try a much smaller radius or lower point_count.", file=sys.stderr)

    valid = np.where(tri_area > 1e-12)[0].astype(np.int32)
    if valid.size == 0:
        raise RuntimeError("All triangles are degenerate.")

    weights = tri_area[valid].astype(np.float32)
    prob, alias = build_alias_table(weights)

    # Upload geometry
    Vw = wp.array(V.astype(np.float32), dtype=wp.vec3, device=device)
    Fw = wp.array(F.astype(np.int32), dtype=wp.vec3i, device=device)
    VNW = wp.array(VN.astype(np.float32), dtype=wp.vec3, device=device)

    valid_w = wp.array(valid.astype(np.int32), dtype=wp.int32, device=device)
    prob_w = wp.array(prob, dtype=wp.float32, device=device)
    alias_w = wp.array(alias, dtype=wp.int32, device=device)
    K = int(valid.size)

    # Accepted output buffers (device)
    accP = wp.empty(shape=(target_count,), dtype=wp.vec3, device=device)
    accN = wp.empty(shape=(target_count,), dtype=wp.vec3, device=device)
    accF = wp.empty(shape=(target_count,), dtype=wp.int32, device=device)
    acc_count = wp.zeros(shape=(1,), dtype=wp.int32, device=device)

    # Candidate buffers (device) - allocated per round based on M
    # Pre-compile kernels (define inside function to avoid import-time warp requirement)
    @wp.kernel
    def gen_candidates(V: wp.array(dtype=wp.vec3),
                       F: wp.array(dtype=wp.vec3i),
                       VN: wp.array(dtype=wp.vec3),
                       valid_ids: wp.array(dtype=wp.int32),
                       prob: wp.array(dtype=wp.float32),
                       alias: wp.array(dtype=wp.int32),
                       K: int,
                       seed: int,
                       outP: wp.array(dtype=wp.vec3),
                       outN: wp.array(dtype=wp.vec3),
                       outF: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        state = wp.rand_init(seed, tid)

        u = wp.randf(state)
        col = int(u * float(K))
        if col < 0:
            col = 0
        if col >= K:
            col = K - 1

        u2 = wp.randf(state)
        j = col
        if u2 >= prob[col]:
            j = alias[col]

        tri = valid_ids[j]  # original face id
        f = F[tri]
        i0 = int(f[0])
        i1 = int(f[1])
        i2 = int(f[2])

        # uniform barycentric
        r1 = wp.randf(state)
        r2 = wp.randf(state)
        sr1 = wp.sqrt(r1)
        w0 = 1.0 - sr1
        w1 = sr1 * (1.0 - r2)
        w2 = sr1 * r2

        p = V[i0] * w0 + V[i1] * w1 + V[i2] * w2
        n = VN[i0] * w0 + VN[i1] * w1 + VN[i2] * w2
        n = wp.normalize(n)

        outP[tid] = p
        outN[tid] = n
        outF[tid] = tri

    @wp.kernel
    def prefilter_vs_accepted(grid: wp.uint64,
                              accepted: wp.array(dtype=wp.vec3),
                              candP: wp.array(dtype=wp.vec3),
                              radius: float,
                              pre_ok: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        p = candP[tid]

        # if any neighbor within radius => reject
        query = wp.hash_grid_query(grid, p, radius)
        index = int(0)
        while wp.hash_grid_query_next(query, index):
            q = accepted[index]
            if wp.length(p - q) < radius:
                pre_ok[tid] = 0
                return
        pre_ok[tid] = 1

    @wp.kernel
    def mutual_cull_in_batch(cgrid: wp.uint64,
                             candP: wp.array(dtype=wp.vec3),
                             pre_ok: wp.array(dtype=wp.int32),
                             radius: float,
                             final_ok: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        if pre_ok[tid] == 0:
            final_ok[tid] = 0
            return

        p = candP[tid]
        # keep only if no lower-index pre_ok neighbor within radius
        query = wp.hash_grid_query(cgrid, p, radius)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if j < tid and pre_ok[j] == 1:
                q = candP[j]
                if wp.length(p - q) < radius:
                    final_ok[tid] = 0
                    return
        final_ok[tid] = 1

    @wp.kernel
    def append_survivors(candP: wp.array(dtype=wp.vec3),
                         candN: wp.array(dtype=wp.vec3),
                         candF: wp.array(dtype=wp.int32),
                         ok: wp.array(dtype=wp.int32),
                         accP: wp.array(dtype=wp.vec3),
                         accN: wp.array(dtype=wp.vec3),
                         accF: wp.array(dtype=wp.int32),
                         acc_count: wp.array(dtype=wp.int32),
                         target: int):
        tid = wp.tid()
        if ok[tid] == 0:
            return

        # atomic append
        idx = wp.atomic_add(acc_count, 0, 1)
        if idx < target:
            accP[idx] = candP[tid]
            accN[idx] = candN[tid]
            accF[idx] = candF[tid]

    # Choose hashgrid dimensions to avoid periodic wrap-around
    # Warp HashGrid uses periodic tiling in its dimensions; choose large enough. (see discussions/issues)
    bmin = V.min(axis=0)
    bmax = V.max(axis=0)
    extent = (bmax - bmin)
    cell = float(radius)
    cells = int(math.ceil(float(np.max(extent)) / max(cell, 1e-8))) + 4
    dim = next_pow2(cells)
    dim = int(min(max(dim, 64), 2048))  # cap to avoid absurd memory use
    if dim < cells:
        print(f"[WARN] HashGrid dim={dim} may be too small for cell count ~{cells}; you may see wrap artifacts. "
              f"Consider increasing cap or scaling your mesh.", file=sys.stderr)

    grid_acc = None

    # Round loop
    for r in range(max_rounds):
        n_acc = int(acc_count.numpy()[0])
        if n_acc >= target_count:
            break

        remaining = target_count - n_acc
        M = int(min(max(batch_min, remaining * batch_factor), batch_max))

        candP = wp.empty(shape=(M,), dtype=wp.vec3, device=device)
        candN = wp.empty(shape=(M,), dtype=wp.vec3, device=device)
        candF = wp.empty(shape=(M,), dtype=wp.int32, device=device)

        # Generate candidates (seed offset each round)
        wp.launch(gen_candidates, dim=M,
                  inputs=[Vw, Fw, VNW, valid_w, prob_w, alias_w, K, int(seed + 1337 * r),
                          candP, candN, candF],
                  device=device)

        # Candidate grid for mutual cull
        cgrid = wp.HashGrid(dim_x=dim, dim_y=dim, dim_z=dim, device=device)
        try:
            cgrid.reserve(M)
        except Exception:
            pass
        cgrid.build(points=candP, radius=float(radius))

        pre_ok = wp.zeros(shape=(M,), dtype=wp.int32, device=device)

        # Build accepted grid if needed
        if n_acc > 0:
            acc_active = wp.empty(shape=(n_acc,), dtype=wp.vec3, device=device)

            @wp.kernel
            def _copy_prefix(src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                dst[tid] = src[tid]

            wp.launch(_copy_prefix, dim=n_acc, inputs=[accP, acc_active], device=device)

            grid_acc = wp.HashGrid(dim_x=dim, dim_y=dim, dim_z=dim, device=device)
            try:
                grid_acc.reserve(n_acc)
            except Exception:
                pass
            grid_acc.build(points=acc_active, radius=float(radius))

            wp.launch(prefilter_vs_accepted, dim=M, inputs=[grid_acc.id, acc_active, candP, float(radius), pre_ok], device=device)
        else:
            # no accepted yet => all pre_ok = 1
            @wp.kernel
            def _fill_ones(a: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                a[tid] = 1
            wp.launch(_fill_ones, dim=M, inputs=[pre_ok], device=device)

        final_ok = wp.zeros(shape=(M,), dtype=wp.int32, device=device)
        wp.launch(mutual_cull_in_batch, dim=M, inputs=[cgrid.id, candP, pre_ok, float(radius), final_ok], device=device)

        # Append survivors
        wp.launch(append_survivors, dim=M,
                  inputs=[candP, candN, candF, final_ok, accP, accN, accF, acc_count, int(target_count)],
                  device=device)

        new_n = int(acc_count.numpy()[0])
        added = new_n - n_acc
        print(f"[INFO] round {r:03d}: accepted {new_n}/{target_count} (+{added}), batch={M}", file=sys.stderr)

        if added == 0 and r > 5:
            # if not making progress, likely radius too large for requested count
            print("[WARN] No progress in this round. Try smaller radius or smaller target_count.", file=sys.stderr)
            break

    n_final = int(min(int(acc_count.numpy()[0]), target_count))
    if n_final == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.int32)

    P = accP.numpy()[:n_final].astype(np.float32)
    Nrm = accN.numpy()[:n_final].astype(np.float32)
    FaceID = accF.numpy()[:n_final].astype(np.int32)
    return P, Nrm, FaceID


# ----------------------------
# Export
# ----------------------------
def save_ply_ascii_points(path: str, P: np.ndarray, N: np.ndarray, layer_id: np.ndarray, face_id: np.ndarray, radius: float) -> None:
    with open(path, "w", encoding="utf-8") as out:
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        out.write(f"comment poisson_radius {radius}\n")
        out.write(f"element vertex {P.shape[0]}\n")
        out.write("property float x\nproperty float y\nproperty float z\n")
        out.write("property float nx\nproperty float ny\nproperty float nz\n")
        out.write("property uint layer_id\nproperty uint face_id\n")
        out.write("end_header\n")
        for i in range(P.shape[0]):
            p = P[i]
            n = N[i]
            out.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {n[0]:.9f} {n[1]:.9f} {n[2]:.9f} {int(layer_id[i])} {int(face_id[i])}\n")


def save_txt_points(path: str, P: np.ndarray, layer_id: np.ndarray, face_id: np.ndarray, radius: float) -> None:
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# count={P.shape[0]} radius={radius}\n")
        out.write("# x y z layer_id face_id\n")
        for i in range(P.shape[0]):
            p = P[i]
            out.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {int(layer_id[i])} {int(face_id[i])}\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj")
    parser.add_argument("radius", type=float)
    parser.add_argument("point_count", type=int)
    parser.add_argument("--device", default="cuda:0", help='e.g. "cuda:0" or "cpu"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=200)
    parser.add_argument("--batch_factor", type=int, default=8)
    parser.add_argument("--batch_min", type=int, default=65536)
    parser.add_argument("--batch_max", type=int, default=1048576)
    parser.add_argument("out_ply", nargs="?", default="points.ply")
    parser.add_argument("out_txt", nargs="?", default="points.txt")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    V, F = load_obj(args.input_obj)
    Vn, scale, bminN, bmaxN = normalize_mesh(V)

    radius = float(args.radius) * scale
    tri_area, tri_n = triangle_info(Vn, F)
    VN = vertex_normals_area_weighted(Vn, F, tri_area, tri_n)

    print(f"[INFO] V={Vn.shape[0]} F={F.shape[0]}", file=sys.stderr)
    print(f"[INFO] radius(after normalize)={radius} target surface N={args.point_count}", file=sys.stderr)
    print(f"[INFO] device={args.device}", file=sys.stderr)

    P, Nrm, FaceID = poisson_sample_surface_warp(
        Vn, F, VN, radius, int(args.point_count),
        device=args.device, seed=args.seed,
        max_rounds=args.max_rounds,
        batch_factor=args.batch_factor,
        batch_min=args.batch_min, batch_max=args.batch_max
    )

    layer_id = np.zeros((P.shape[0],), dtype=np.int32)
    save_ply_ascii_points(args.out_ply, P, Nrm, layer_id, FaceID, radius)
    save_txt_points(args.out_txt, P, layer_id, FaceID, radius)

    print(f"[INFO] wrote: {args.out_ply}", file=sys.stderr)
    print(f"[INFO] wrote: {args.out_txt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
