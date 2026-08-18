# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM bubble tracking: YACCLAB CCL + merge/split + gas-law updates.

Truth source (reference CUDA):
- ``docs/Home-FSLBM/inc/3D/gpu/tDCCL.cu`` — YACCLAB 3D CCL
- ``docs/Home-FSLBM/inc/3D/gpu/mrLbmSolverGpu3D.cu`` — bubble kernels / coupling

Flat CCL indexing matches the reference: ``flat = x + nx*(y + ny*z)`` (x-fastest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from . import constants as C

if TYPE_CHECKING:
    from .state import HomeFslbmState

# ---------------------------------------------------------------------------
# Flat-index helpers (x-fastest, matching tDCCL / mrLbmSolverGpu3D)
# ---------------------------------------------------------------------------


@wp.func
def _flat_idx(x: int, y: int, z: int, nx: int, ny: int) -> int:
    return x + nx * (y + ny * z)


@wp.func
def _flat_to_x(n: int, nx: int) -> int:
    return n % nx


@wp.func
def _flat_to_y(n: int, nx: int, ny: int) -> int:
    return (n // nx) % ny


@wp.func
def _flat_to_z(n: int, nx: int, ny: int) -> int:
    return n // (nx * ny)


@wp.func
def _ccl_find(labels: wp.array(dtype=wp.int32), n: int) -> int:
    """UF find without compression (YACCLAB ``Find``)."""
    while labels[n] != n:
        n = labels[n]
    return n


@wp.func
def _ccl_find_compress(labels: wp.array(dtype=wp.int32), n: int) -> int:
    """UF find with path compression (YACCLAB ``FindAndCompress``)."""
    id_n = n
    while labels[n] != n:
        n = labels[n]
        labels[id_n] = n
    return n


@wp.func
def _ccl_union(labels: wp.array(dtype=wp.int32), a: int, b: int) -> None:
    """UF union via atomicMin toward smaller root (YACCLAB ``Union``)."""
    done = int(0)
    while done == 0:
        a = _ccl_find(labels, a)
        b = _ccl_find(labels, b)
        if a < b:
            old = wp.atomic_min(labels, b, a)
            if old == b:
                done = 1
            b = old
        elif b < a:
            old = wp.atomic_min(labels, a, b)
            if old == a:
                done = 1
            a = old
        else:
            done = 1


# ---------------------------------------------------------------------------
# YACCLAB CCL kernels (tDCCL.cu) — block-structured 2×2×2
# ---------------------------------------------------------------------------


@wp.kernel
def ccl_init_labeling_kernel(
    labels: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """InitLabeling: each even-corner voxel gets provisional label = flat index."""
    bx, by, bz = wp.tid()
    x = bx * 2
    y = by * 2
    z = bz * 2
    if x < nx and y < ny and z < nz:
        idx = _flat_idx(x, y, z, nx, ny)
        labels[idx] = idx


@wp.kernel
def ccl_merge_kernel(
    img: wp.array(dtype=wp.uint8),
    labels: wp.array(dtype=wp.int32),
    last_cube_fg: wp.array(dtype=wp.uint8),
    nx: int,
    ny: int,
    nz: int,
):
    """Merge: 2×2×2 block foreground mask + neighbour-block Union.

    Faithful to ``tDCCL.cu`` Merge connectivity (26-neighbour voxel adjacency
    across 2×2×2 blocks), expressed with explicit 3-D indexing instead of
    pitched ``reinterpret_cast`` loads.
    """
    bx, by, bz = wp.tid()
    x = bx * 2
    y = by * 2
    z = bz * 2
    if x >= nx or y >= ny or z >= nz:
        return

    labels_index = _flat_idx(x, y, z, nx, ny)

    # ---- Build 8-bit foreground mask for the 2×2×2 cube ----
    foreground = wp.uint8(0)
    if x < nx and y < ny and z < nz and int(img[_flat_idx(x, y, z, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(1)
    if x + 1 < nx and y < ny and z < nz and int(img[_flat_idx(x + 1, y, z, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(2)
    if x < nx and y + 1 < ny and z < nz and int(img[_flat_idx(x, y + 1, z, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(4)
    if x + 1 < nx and y + 1 < ny and z < nz and int(img[_flat_idx(x + 1, y + 1, z, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(8)
    if x < nx and y < ny and z + 1 < nz and int(img[_flat_idx(x, y, z + 1, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(16)
    if x + 1 < nx and y < ny and z + 1 < nz and int(img[_flat_idx(x + 1, y, z + 1, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(32)
    if x < nx and y + 1 < ny and z + 1 < nz and int(img[_flat_idx(x, y + 1, z + 1, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(64)
    if x + 1 < nx and y + 1 < ny and z + 1 < nz and int(img[_flat_idx(x + 1, y + 1, z + 1, nx, ny)]) != 0:
        foreground = foreground | wp.uint8(128)

    # Store foreground bitmask (same spill locations as reference)
    if x + 1 < nx:
        labels[_flat_idx(x + 1, y, z, nx, ny)] = int(foreground)
    elif y + 1 < ny:
        labels[_flat_idx(x, y + 1, z, nx, ny)] = int(foreground)
    elif z + 1 < nz:
        labels[_flat_idx(x, y, z + 1, nx, ny)] = int(foreground)
    else:
        last_cube_fg[0] = foreground

    if int(foreground) == 0:
        return

    # ---- Union with neighbouring blocks for each FG voxel's 26-neighbours ----
    # Offsets within the 2×2×2 cube (bit order matches foreground mask).
    for bit in range(8):
        if ((int(foreground) >> bit) & 1) == 0:
            continue
        lx = x + (bit & 1)
        ly = y + ((bit >> 1) & 1)
        lz = z + ((bit >> 2) & 1)
        if lx >= nx or ly >= ny or lz >= nz:
            continue

        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx_ = lx + dx
                    ny_ = ly + dy
                    nz_ = lz + dz
                    if nx_ < 0 or nx_ >= nx or ny_ < 0 or ny_ >= ny or nz_ < 0 or nz_ >= nz:
                        continue
                    if int(img[_flat_idx(nx_, ny_, nz_, nx, ny)]) == 0:
                        continue
                    # Neighbour block even-corner
                    bx2 = nx_ - (nx_ % 2)
                    by2 = ny_ - (ny_ % 2)
                    bz2 = nz_ - (nz_ % 2)
                    other = _flat_idx(bx2, by2, bz2, nx, ny)
                    if other != labels_index:
                        _ccl_union(labels, labels_index, other)


@wp.kernel
def ccl_path_compression_kernel(
    labels: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """PathCompression on each 2×2×2 even corner."""
    bx, by, bz = wp.tid()
    x = bx * 2
    y = by * 2
    z = bz * 2
    if x < nx and y < ny and z < nz:
        idx = _flat_idx(x, y, z, nx, ny)
        _ccl_find_compress(labels, idx)


@wp.kernel
def ccl_final_labeling_kernel(
    img: wp.array(dtype=wp.uint8),
    labels: wp.array(dtype=wp.int32),
    last_cube_fg: wp.array(dtype=wp.uint8),
    nx: int,
    ny: int,
    nz: int,
):
    """FinalLabeling: write root+1 to FG voxels in each 2×2×2 cube."""
    bx, by, bz = wp.tid()
    x = bx * 2
    y = by * 2
    z = bz * 2
    if x >= nx or y >= ny or z >= nz:
        return

    labels_index = _flat_idx(x, y, z, nx, ny)
    root = labels[labels_index]
    label = root + 1

    # Recover foreground bitmask
    foreground = wp.uint8(0)
    if x + 1 < nx:
        foreground = wp.uint8(labels[_flat_idx(x + 1, y, z, nx, ny)])
    elif y + 1 < ny:
        foreground = wp.uint8(labels[_flat_idx(x, y + 1, z, nx, ny)])
    elif z + 1 < nz:
        foreground = wp.uint8(labels[_flat_idx(x, y, z + 1, nx, ny)])
    else:
        foreground = last_cube_fg[0]

    for bit in range(8):
        lx = x + (bit & 1)
        ly = y + ((bit >> 1) & 1)
        lz = z + ((bit >> 2) & 1)
        if lx >= nx or ly >= ny or lz >= nz:
            continue
        idx = _flat_idx(lx, ly, lz, nx, ny)
        if ((int(foreground) >> bit) & 1) != 0:
            labels[idx] = label
        else:
            labels[idx] = 0


@wp.kernel
def ccl_apply_renumber_kernel(
    labels: wp.array(dtype=wp.int32),
    remap: wp.array(dtype=wp.int32),
    n: int,
):
    """Apply dense remapping: label → remap[label] (remap[0]=0)."""
    tid = wp.tid()
    if tid >= n:
        return
    lab = labels[tid]
    if lab > 0:
        labels[tid] = remap[lab]


@wp.kernel
def ccl_pack_u8_3d_to_flat_kernel(
    src: wp.array3d(dtype=wp.uint8),
    dst: wp.array(dtype=wp.uint8),
    nx: int,
    ny: int,
    nz: int,
):
    """Pack ``array3d[x,y,z]`` → flat x-fastest ``uint8`` (device-side)."""
    x, y, z = wp.tid()
    if x >= nx or y >= ny or z >= nz:
        return
    dst[_flat_idx(x, y, z, nx, ny)] = src[x, y, z]


@wp.kernel
def ccl_unpack_i32_flat_to_3d_kernel(
    src: wp.array(dtype=wp.int32),
    dst: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """Unpack flat x-fastest ``int32`` → ``array3d[x,y,z]`` (device-side)."""
    x, y, z = wp.tid()
    if x >= nx or y >= ny or z >= nz:
        return
    dst[x, y, z] = src[_flat_idx(x, y, z, nx, ny)]


@wp.kernel
def ccl_mark_labels_present_kernel(
    labels: wp.array(dtype=wp.int32),
    present: wp.array(dtype=wp.int32),
    n: int,
):
    """Mark which provisional label ids appear (``present[lab]=1``)."""
    tid = wp.tid()
    if tid >= n:
        return
    lab = labels[tid]
    if lab > 0:
        present[lab] = 1


@wp.kernel
def ccl_build_dense_remap_kernel(
    present: wp.array(dtype=wp.int32),
    scanned: wp.array(dtype=wp.int32),
    remap: wp.array(dtype=wp.int32),
    n_slots: int,
):
    """Build dense remap: ``new_id = exclusive_scan[lab] + 1`` if present."""
    lab = wp.tid()
    if lab >= n_slots:
        return
    if present[lab] != 0:
        remap[lab] = scanned[lab] + 1
    else:
        remap[lab] = 0


@wp.kernel
def ccl_count_from_scan_kernel(
    present: wp.array(dtype=wp.int32),
    scanned: wp.array(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    last_idx: int,
):
    """``label_num = exclusive_scan[last] + present[last]``."""
    if wp.tid() != 0:
        return
    out_count[0] = scanned[last_idx] + present[last_idx]


# ---------------------------------------------------------------------------
# Device CCL driver (no full-field host round-trips)
# ---------------------------------------------------------------------------


def connected_component_labeling(
    input_matrix: wp.array3d,
    label_matrix: wp.array3d,
) -> int:
    """Run YACCLAB-style CCL and write dense labels 1..N into ``label_matrix``.

    Layout conversion and dense renumber run entirely on device (Warp kernels
    + ``wp.utils.array_scan``), avoiding full-field ``numpy()`` D↔H copies.

    Parameters
    ----------
    input_matrix:
        Binary image (255 = foreground / gas+interface, 0 = background).
    label_matrix:
        Output labels (0 = background, 1..N = components).

    Returns
    -------
    int
        Number of connected components (``label_num`` / bubble count).
    """
    nx, ny, nz = int(input_matrix.shape[0]), int(input_matrix.shape[1]), int(input_matrix.shape[2])
    n = nx * ny * nz
    device = input_matrix.device
    dim3 = (nx, ny, nz)

    img = wp.empty(n, dtype=wp.uint8, device=device)
    labels = wp.zeros(n, dtype=wp.int32, device=device)
    last_cube_fg = wp.zeros(1, dtype=wp.uint8, device=device)

    wp.launch(
        ccl_pack_u8_3d_to_flat_kernel,
        dim=dim3,
        inputs=[input_matrix, img, nx, ny, nz],
        device=device,
    )

    grid = ((nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2)

    wp.launch(ccl_init_labeling_kernel, dim=grid, inputs=[labels, nx, ny, nz], device=device)
    wp.launch(
        ccl_merge_kernel,
        dim=grid,
        inputs=[img, labels, last_cube_fg, nx, ny, nz],
        device=device,
    )
    wp.launch(ccl_path_compression_kernel, dim=grid, inputs=[labels, nx, ny, nz], device=device)
    wp.launch(
        ccl_final_labeling_kernel,
        dim=grid,
        inputs=[img, labels, last_cube_fg, nx, ny, nz],
        device=device,
    )

    # Dense renumber on device (≡ thrust::sort + unique + renumber_*).
    # Provisional labels are in 1..n (root+1); slot 0 is background.
    n_slots = n + 1
    present = wp.zeros(n_slots, dtype=wp.int32, device=device)
    wp.launch(
        ccl_mark_labels_present_kernel,
        dim=n,
        inputs=[labels, present, n],
        device=device,
    )
    scanned = wp.empty(n_slots, dtype=wp.int32, device=device)
    wp.utils.array_scan(present, scanned, inclusive=False)

    remap = wp.zeros(n_slots, dtype=wp.int32, device=device)
    wp.launch(
        ccl_build_dense_remap_kernel,
        dim=n_slots,
        inputs=[present, scanned, remap, n_slots],
        device=device,
    )
    label_num_gpu = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        ccl_count_from_scan_kernel,
        dim=1,
        inputs=[present, scanned, label_num_gpu, n],
        device=device,
    )

    wp.launch(
        ccl_apply_renumber_kernel,
        dim=n,
        inputs=[labels, remap, n],
        device=device,
    )
    wp.launch(
        ccl_unpack_i32_flat_to_3d_kernel,
        dim=dim3,
        inputs=[labels, label_matrix, nx, ny, nz],
        device=device,
    )

    return int(label_num_gpu.numpy()[0])


# ---------------------------------------------------------------------------
# Bubble lifecycle / tagging kernels (mrLbmSolverGpu3D.cu)
# ---------------------------------------------------------------------------


@wp.kernel
def clear_detector_kernel(
    merge_detector: wp.array3d(dtype=wp.int32),
    merge_flag_gpu: wp.array(dtype=wp.int32),
    split_flag_gpu: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """clear_detector — ``mrLbmSolverGpu3D.cu:64-83``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    merge_detector[i, j, k] = 0
    flat = _flat_idx(i, j, k, nx, ny)
    if flat == 1:
        merge_flag_gpu[0] = 0
        split_flag_gpu[0] = 0


@wp.kernel
def clear_inlet_kernel(
    islet: wp.array3d(dtype=wp.int32),
    flag: wp.array3d(dtype=wp.uint8),
    phi: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    massex: wp.array3d(dtype=float),
    f_mom: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    """clear_inlet — ``mrLbmSolverGpu3D.cu:110-142``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    if islet[i, j, k] == 1:
        islet[i, j, k] = 0
        flag[i, j, k] = wp.uint8(C.CellFlag.TYPE_G)
        phi[i, j, k] = 0.0
        mass[i, j, k] = 0.0
        massex[i, j, k] = 0.0
        base = _flat_idx(i, j, k, nx, ny)
        f_mom[base + 0 * stride] = 1.0
        f_mom[base + 1 * stride] = 0.0
        f_mom[base + 2 * stride] = 0.0
        f_mom[base + 3 * stride] = 0.0
        f_mom[base + 4 * stride] = 0.0
        f_mom[base + 5 * stride] = 0.0
        f_mom[base + 6 * stride] = 0.0
        f_mom[base + 7 * stride] = 0.0
        f_mom[base + 8 * stride] = 0.0
        f_mom[base + 9 * stride] = 0.0


@wp.kernel
def init_tag_kernel(
    flag: wp.array3d(dtype=wp.uint8),
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_tag: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """InitTag — ``mrLbmSolverGpu3D.cu:1089-1112``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    flagsn = int(flag[i, j, k])
    flagsn_bo = flagsn & C.CellFlag.TYPE_BO
    if flagsn_bo == C.CellFlag.TYPE_S:
        tag_matrix[i, j, k] = -1
    if flagsn == C.CellFlag.TYPE_F:
        tag_matrix[i, j, k] = -1
    previous_tag[i, j, k] = -1


@wp.kernel
def convert_flag_to_input_kernel(
    flag: wp.array3d(dtype=wp.uint8),
    input_matrix: wp.array3d(dtype=wp.uint8),
    nx: int,
    ny: int,
    nz: int,
):
    """convertIntToUnsignedChar — ``mrLbmSolverGpu3D.cu:1115-1135``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    flagsn = int(flag[i, j, k])
    flagsn_sus = flagsn & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
    flagsn_bo = flagsn & C.CellFlag.TYPE_BO
    if (flagsn_sus == C.CellFlag.TYPE_G or flagsn_sus == C.CellFlag.TYPE_I) and (
        flagsn_bo != C.CellFlag.TYPE_S
    ):
        input_matrix[i, j, k] = wp.uint8(255)
    else:
        input_matrix[i, j, k] = wp.uint8(0)


@wp.kernel
def parse_label_kernel(
    label_matrix: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    label_volume: wp.array(dtype=wp.float64),
    label_num_gpu: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """parse_label — ``mrLbmSolverGpu3D.cu:1066-1086``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    lab = label_matrix[i, j, k]
    if lab > 0:
        wp.atomic_max(label_num_gpu, 0, lab)
        wp.atomic_add(
            label_volume,
            lab - 1,
            wp.float64(1.0 - float(phi[i, j, k])),
        )


@wp.kernel
def create_bubble_label_kernel(
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    label_volume: wp.array(dtype=wp.float64),
    label_num_gpu: wp.array(dtype=wp.int32),
    bubble_count_gpu: wp.array(dtype=wp.int32),
):
    """create_bubble_label — ``mrLbmSolverGpu3D.cu:1139-1148``."""
    if wp.tid() != 0:
        return
    n = label_num_gpu[0]
    bubble_count_gpu[0] = n
    for i in range(n):
        bubble_volume[i] = label_volume[i]
        bubble_init_volume[i] = label_volume[i]
        bubble_rho[i] = wp.float64(1.0)


@wp.kernel
def update_init_tag_kernel(
    label_matrix: wp.array3d(dtype=wp.int32),
    tag_matrix: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """update_init_tag — ``mrLbmSolverGpu3D.cu:1150-1169``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    lab = label_matrix[i, j, k]
    if lab > 0:
        tag_matrix[i, j, k] = lab


@wp.kernel
def reset_label_volume_kernel(
    label_volume: wp.array(dtype=wp.float64),
    label_init_volume: wp.array(dtype=wp.float64),
    max_bubbles: int,
):
    """reset_label_volume — ``mrLbmSolverGpu3D.cu:1505-1516``."""
    tid = wp.tid()
    if tid >= max_bubbles:
        return
    label_volume[tid] = wp.float64(0.0)
    label_init_volume[tid] = wp.float64(0.0)


@wp.kernel
def reset_label_num_kernel(label_num_gpu: wp.array(dtype=wp.int32)):
    """Zero label_num before reduce (``ResetLabelVolume`` sets label_num=0)."""
    if wp.tid() == 0:
        label_num_gpu[0] = 0


@wp.kernel
def reduce_label_rho_kernel(
    label_matrix: wp.array3d(dtype=wp.int32),
    tag_matrix: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    bubble_rho: wp.array(dtype=wp.float64),
    label_volume: wp.array(dtype=wp.float64),
    label_init_volume: wp.array(dtype=wp.float64),
    label_num_gpu: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """reduce_label_rho — ``mrLbmSolverGpu3D.cu:1519-1555``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    lab = label_matrix[i, j, k]
    if lab > 0:
        tag = tag_matrix[i, j, k] - 1
        vol = wp.float64(1.0 - float(phi[i, j, k]))
        wp.atomic_max(label_num_gpu, 0, lab)
        wp.atomic_add(label_volume, lab - 1, vol)
        if tag >= 0:
            wp.atomic_add(label_init_volume, lab - 1, vol * bubble_rho[tag])
        else:
            wp.atomic_add(label_init_volume, lab - 1, vol * wp.float64(1.0))
        tag_matrix[i, j, k] = lab
    else:
        tag_matrix[i, j, k] = -1


@wp.kernel
def num_rho_update_kernel(
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    label_num_gpu: wp.array(dtype=wp.int32),
    bubble_count_gpu: wp.array(dtype=wp.int32),
):
    """num_rho_update_kernel — ``mrLbmSolverGpu3D.cu:1564-1573``."""
    if wp.tid() != 0:
        return
    n = label_num_gpu[0]
    if n > 0:
        bubble_count_gpu[0] = n
    label_num_gpu[0] = 0
    bc = bubble_count_gpu[0]
    for i in range(bc):
        vol = bubble_volume[i]
        bubble_rho[i] = bubble_init_volume[i] / vol


@wp.kernel
def bubble_volume_update_kernel(
    delta_phi: wp.array3d(dtype=float),
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_tag: wp.array3d(dtype=wp.int32),
    bubble_volume: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
):
    """bubble_volume_update_kernel — ``mrLbmSolverGpu3D.cu:1301-1326``.

    Updates bubble volume from ``delta_phi`` (not raw 1-φ).
    """
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    dphi = delta_phi[i, j, k]
    if dphi != 0.0:
        tag = tag_matrix[i, j, k]
        if tag <= 0:
            tag = previous_tag[i, j, k]
            previous_tag[i, j, k] = -1
        if tag > 0:
            wp.atomic_add(bubble_volume, tag - 1, wp.float64(-dphi))
        delta_phi[i, j, k] = 0.0


@wp.kernel
def bubble_rho_update_kernel(
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    bubble_count_gpu: wp.array(dtype=wp.int32),
):
    """bubble_rho_update_kernel — ``mrLbmSolverGpu3D.cu:1356-1363``.

    ``ρ = V_init / V`` (init_volume stores ρ·V mass-like quantity after merges).
    """
    b = wp.tid()
    if b >= bubble_count_gpu[0]:
        return
    vol = bubble_volume[b]
    bubble_rho[b] = bubble_init_volume[b] / vol


@wp.kernel
def get_tag_kernel(
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_merge_tag: wp.array3d(dtype=wp.int32),
    merge_detector: wp.array3d(dtype=wp.int32),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """get_tag_kernel — ``mrLbmSolverGpu3D.cu:1375-1421``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    if merge_detector[i, j, k] != 0 and tag_matrix[i, j, k] == -1:
        this_cell_id = tag_matrix[i, j, k]
        for d in range(1, 27):
            x1 = i - int(cx[d])
            y1 = j - int(cy[d])
            z1 = k - int(cz[d])
            if x1 >= 0 and x1 < nx and y1 >= 0 and y1 < ny and z1 >= 0 and z1 < nz:
                if tag_matrix[x1, y1, z1] > -1:
                    this_cell_id = tag_matrix[x1, y1, z1]
        previous_merge_tag[i, j, k] = this_cell_id if this_cell_id > 0 else -1
    else:
        merge_detector[i, j, k] = 0


@wp.kernel
def assign_tag_kernel(
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_merge_tag: wp.array3d(dtype=wp.int32),
    merge_detector: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """assign_tag_kernel — ``mrLbmSolverGpu3D.cu:1424-1449``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    if previous_merge_tag[i, j, k] > 0 and merge_detector[i, j, k] != 0:
        tag_matrix[i, j, k] = previous_merge_tag[i, j, k]
        previous_merge_tag[i, j, k] = -1
    else:
        previous_merge_tag[i, j, k] = -1


@wp.kernel
def recheck_merge_kernel(
    tag_matrix: wp.array3d(dtype=wp.int32),
    merge_detector: wp.array3d(dtype=wp.int32),
    merge_flag_gpu: wp.array(dtype=wp.int32),
    cx: wp.array(dtype=wp.int32),
    cy: wp.array(dtype=wp.int32),
    cz: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    """recheck_merge_kernel — ``mrLbmSolverGpu3D.cu:1452-1502``."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    if merge_detector[i, j, k] == 0:
        return
    merge_detector[i, j, k] = 0
    this_cell_id = int(-1)
    for d in range(1, 27):
        x1 = i - int(cx[d])
        y1 = j - int(cy[d])
        z1 = k - int(cz[d])
        if x1 >= 0 and x1 < nx and y1 >= 0 and y1 < ny and z1 >= 0 and z1 < nz:
            ntag = tag_matrix[x1, y1, z1]
            if ntag > -1:
                if this_cell_id < 0:
                    this_cell_id = ntag
                elif this_cell_id != ntag:
                    merge_flag_gpu[0] = 1


@wp.kernel
def merge_split_detector_kernel(
    merge_flag_gpu: wp.array(dtype=wp.int32),
    split_flag_gpu: wp.array(dtype=wp.int32),
    out_merge: wp.array(dtype=wp.int32),
    out_split: wp.array(dtype=wp.int32),
):
    """MergeSplitDetectorKernel — ``mrLbmSolverGpu3D.cu:1366-1371``."""
    if wp.tid() == 0:
        out_merge[0] = merge_flag_gpu[0]
        out_split[0] = split_flag_gpu[0]


# ---------------------------------------------------------------------------
# Host orchestration: InitBubble / handle_merge_split
# ---------------------------------------------------------------------------


def _sync_host_counts(
    state: HomeFslbmState,
    label_num_gpu: wp.array,
    bubble_count_gpu: wp.array,
) -> None:
    state.label_num = int(label_num_gpu.numpy()[0])
    state.bubble_count = int(bubble_count_gpu.numpy()[0])


def init_bubbles(
    state: HomeFslbmState,
    label_num_gpu: wp.array,
    bubble_count_gpu: wp.array,
    merge_flag_gpu: wp.array,
    split_flag_gpu: wp.array,
) -> None:
    """InitBubble — ``mrLbmSolverGpu3D.cu:1196-1248``.

    ``InitTag → convert → CCL → parse_label → create_bubble_label →
    update_init_tag → ClearDetector``.
    """
    nx, ny, nz = int(state.res[0]), int(state.res[1]), int(state.res[2])
    dim = (nx, ny, nz)
    device = state.device

    label_num_gpu.zero_()
    bubble_count_gpu.zero_()

    wp.launch(
        init_tag_kernel,
        dim=dim,
        inputs=[state.flag, state.tag_matrix, state.previous_tag, nx, ny, nz],
        device=device,
    )
    wp.launch(
        convert_flag_to_input_kernel,
        dim=dim,
        inputs=[state.flag, state.input_matrix, nx, ny, nz],
        device=device,
    )
    label_num = connected_component_labeling(state.input_matrix, state.label_matrix)
    # Seed label_num before parse (CCL already dense 1..N)
    label_num_gpu.fill_(label_num)

    state.bubble_label_volume.zero_()
    wp.launch(
        parse_label_kernel,
        dim=dim,
        inputs=[
            state.label_matrix,
            state.phi,
            state.bubble_label_volume,
            label_num_gpu,
            nx,
            ny,
            nz,
        ],
        device=device,
    )
    wp.launch(
        create_bubble_label_kernel,
        dim=1,
        inputs=[
            state.bubble_volume,
            state.bubble_init_volume,
            state.bubble_rho,
            state.bubble_label_volume,
            label_num_gpu,
            bubble_count_gpu,
        ],
        device=device,
    )
    wp.launch(
        update_init_tag_kernel,
        dim=dim,
        inputs=[state.label_matrix, state.tag_matrix, nx, ny, nz],
        device=device,
    )
    wp.launch(
        clear_detector_kernel,
        dim=dim,
        inputs=[
            state.merge_detector,
            merge_flag_gpu,
            split_flag_gpu,
            nx,
            ny,
            nz,
        ],
        device=device,
    )
    _sync_host_counts(state, label_num_gpu, bubble_count_gpu)
    state.merge_flag = 0
    state.split_flag = 0


def handle_merge_split(
    state: HomeFslbmState,
    label_num_gpu: wp.array,
    bubble_count_gpu: wp.array,
) -> None:
    """handle_merge_spilt — ``mrLbmSolverGpu3D.cu:1576-1611``.

    ``convert → CCL → reset_label_volume → reduce_label_rho →
    bubble_list_swap → num_rho_update``.
    """
    nx, ny, nz = int(state.res[0]), int(state.res[1]), int(state.res[2])
    dim = (nx, ny, nz)
    device = state.device
    max_b = int(state.bubble_volume.shape[0])

    wp.launch(
        convert_flag_to_input_kernel,
        dim=dim,
        inputs=[state.flag, state.input_matrix, nx, ny, nz],
        device=device,
    )
    connected_component_labeling(state.input_matrix, state.label_matrix)

    wp.launch(
        reset_label_volume_kernel,
        dim=max_b,
        inputs=[state.bubble_label_volume, state.bubble_label_init_volume, max_b],
        device=device,
    )
    wp.launch(reset_label_num_kernel, dim=1, inputs=[label_num_gpu], device=device)

    wp.launch(
        reduce_label_rho_kernel,
        dim=dim,
        inputs=[
            state.label_matrix,
            state.tag_matrix,
            state.phi,
            state.bubble_rho,
            state.bubble_label_volume,
            state.bubble_label_init_volume,
            label_num_gpu,
            nx,
            ny,
            nz,
        ],
        device=device,
    )

    # bubble_list_swap ≡ MomSwap pointer exchange
    state.bubble_volume, state.bubble_label_volume = (
        state.bubble_label_volume,
        state.bubble_volume,
    )
    state.bubble_init_volume, state.bubble_label_init_volume = (
        state.bubble_label_init_volume,
        state.bubble_init_volume,
    )

    wp.launch(
        num_rho_update_kernel,
        dim=1,
        inputs=[
            state.bubble_volume,
            state.bubble_init_volume,
            state.bubble_rho,
            label_num_gpu,
            bubble_count_gpu,
        ],
        device=device,
    )
    _sync_host_counts(state, label_num_gpu, bubble_count_gpu)
