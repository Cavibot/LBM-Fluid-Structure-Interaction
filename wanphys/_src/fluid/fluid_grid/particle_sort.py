from __future__ import annotations

import warp as wp
from warp.utils import radix_sort_pairs


@wp.kernel
def build_particle_linear_keys(
    particle_q: wp.array(dtype=wp.vec3),
    keys: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]

    i = wp.clamp(int(wp.floor(p[0] / dh)), 0, nx - 1)
    j = wp.clamp(int(wp.floor(p[1] / dh)), 0, ny - 1)
    k = wp.clamp(int(wp.floor(p[2] / dh)), 0, nz - 1)

    keys[tid] = i + nx * (j + ny * k)
    indices[tid] = tid


@wp.func
def morton_part1by2_10bit(x: int) -> int:
    x = x & 1023
    result = 0
    for bit in range(10):
        result = result | (((x >> bit) & 1) << (3 * bit))
    return result


@wp.func
def morton3_10bit(x: int, y: int, z: int) -> int:
    return morton_part1by2_10bit(x) | (morton_part1by2_10bit(y) << 1) | (morton_part1by2_10bit(z) << 2)


@wp.kernel
def build_particle_morton_keys(
    particle_q: wp.array(dtype=wp.vec3),
    keys: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    dh: float,
):
    tid = wp.tid()
    p = particle_q[tid]

    i = wp.clamp(int(wp.floor(p[0] / dh)), 0, nx - 1)
    j = wp.clamp(int(wp.floor(p[1] / dh)), 0, ny - 1)
    k = wp.clamp(int(wp.floor(p[2] / dh)), 0, nz - 1)

    keys[tid] = morton3_10bit(i, j, k)
    indices[tid] = tid


@wp.kernel
def gather_sorted_vec3(
    src: wp.array(dtype=wp.vec3),
    sorted_indices: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dst[tid] = src[sorted_indices[tid]]


@wp.kernel
def gather_sorted_mat33(
    src: wp.array(dtype=wp.mat33),
    sorted_indices: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    dst[tid] = src[sorted_indices[tid]]


class ParticleCellSorter:
    """Stable per-step particle reordering by owning MAC cell."""

    def __init__(self, particle_count: int, device, include_affine: bool = False) -> None:
        self.count = max(0, int(particle_count))
        self.device = device

        pair_capacity = max(2 * self.count, 2)
        vec_capacity = max(self.count, 1)

        self.keys = wp.zeros(pair_capacity, dtype=wp.int32, device=device)
        self.indices = wp.zeros(pair_capacity, dtype=wp.int32, device=device)
        self.q_tmp = wp.zeros(vec_capacity, dtype=wp.vec3, device=device)
        self.v_tmp = wp.zeros(vec_capacity, dtype=wp.vec3, device=device)
        self.c_tmp = wp.zeros(vec_capacity, dtype=wp.mat33, device=device) if include_affine else None

    def reorder_qv(
        self,
        particle_q,
        particle_v,
        key_mode: str,
        nx: int,
        ny: int,
        nz: int,
        dh: float,
    ) -> None:
        if self.count <= 1:
            return

        self._sort_indices(particle_q, key_mode, nx, ny, nz, dh)

        wp.launch(gather_sorted_vec3, dim=self.count, inputs=[particle_q, self.indices, self.q_tmp])
        wp.launch(gather_sorted_vec3, dim=self.count, inputs=[particle_v, self.indices, self.v_tmp])

        wp.copy(particle_q, self.q_tmp, count=self.count)
        wp.copy(particle_v, self.v_tmp, count=self.count)

    def reorder_qvc(
        self,
        particle_q,
        particle_v,
        particle_c,
        key_mode: str,
        nx: int,
        ny: int,
        nz: int,
        dh: float,
    ) -> None:
        if self.count <= 1:
            return
        if self.c_tmp is None:
            raise RuntimeError("ParticleCellSorter was created without affine storage")

        self._sort_indices(particle_q, key_mode, nx, ny, nz, dh)

        wp.launch(gather_sorted_vec3, dim=self.count, inputs=[particle_q, self.indices, self.q_tmp])
        wp.launch(gather_sorted_vec3, dim=self.count, inputs=[particle_v, self.indices, self.v_tmp])
        wp.launch(gather_sorted_mat33, dim=self.count, inputs=[particle_c, self.indices, self.c_tmp])

        wp.copy(particle_q, self.q_tmp, count=self.count)
        wp.copy(particle_v, self.v_tmp, count=self.count)
        wp.copy(particle_c, self.c_tmp, count=self.count)

    def _sort_indices(self, particle_q, key_mode: str, nx: int, ny: int, nz: int, dh: float) -> None:
        normalized_mode = str(key_mode).strip().lower()
        if normalized_mode == "linear":
            key_kernel = build_particle_linear_keys
        elif normalized_mode == "morton":
            if nx > 1024 or ny > 1024 or nz > 1024:
                raise ValueError("Morton particle sorting with int32 keys supports grid dimensions up to 1024 per axis")
            key_kernel = build_particle_morton_keys
        else:
            raise ValueError(f"Unsupported particle sort key mode: {key_mode!r}")

        wp.launch(
            key_kernel,
            dim=self.count,
            inputs=[particle_q, self.keys, self.indices, nx, ny, nz, dh],
        )
        radix_sort_pairs(self.keys, self.indices, self.count)
