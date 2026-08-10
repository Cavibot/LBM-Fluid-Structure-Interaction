# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side range auditing for optional HOME moment quantization."""

import warp as wp


@wp.kernel
def audit_moment_quantization_ranges_kernel(
    moments: wp.array(dtype=float),
    offsets: wp.array(dtype=float),
    bounds: wp.array(dtype=float),
    saturation_counts: wp.array(dtype=wp.int32),
    invalid_counts: wp.array(dtype=wp.int32),
    flags: wp.array3d(dtype=wp.int32),
    use_active_flags: int,
    ny: int,
    nz: int,
    cell_count: int,
):
    cell = wp.tid()
    if use_active_flags != 0:
        i = cell // (ny * nz)
        remainder = cell - i * ny * nz
        j = remainder // nz
        k = remainder - j * nz
        flag = flags[i, j, k]
        if flag != 1 and flag != 2:
            return
    for component in range(10):
        value = moments[component * cell_count + cell]
        centered = value - offsets[component]
        if not wp.isfinite(value):
            wp.atomic_add(invalid_counts, component, 1)
        elif wp.abs(centered) > bounds[component]:
            wp.atomic_add(saturation_counts, component, 1)
