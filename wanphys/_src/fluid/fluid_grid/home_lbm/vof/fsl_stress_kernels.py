# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for Bogner bulk momentum-strain reconstruction."""

from __future__ import annotations

import warp as wp


Vec4f = wp.types.vector(4, wp.float32)
Mat44f = wp.types.matrix((4, 4), wp.float32)


@wp.func
def _neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + 4 * size) % size
        return -1
    return value


@wp.kernel
def reconstruct_fsl_bulk_strain_kernel(
    moments: wp.array(dtype=float),
    active: wp.array3d(dtype=wp.int32),
    bulk_strain: wp.array(dtype=float),
    donor_count: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    maximum_fit_condition: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    donor_count[i, j, k] = 0
    for component in range(6):
        bulk_strain[component * stride + cell] = 0.0
    if active[i, j, k] == 0:
        return

    gram = Mat44f()
    rhs_x = Vec4f()
    rhs_y = Vec4f()
    rhs_z = Vec4f()
    donors = int(0)
    valid_donors = int(1)
    uniform_x = int(1)
    uniform_y = int(1)
    uniform_z = int(1)
    first_x = float(0.0)
    first_y = float(0.0)
    first_z = float(0.0)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-2, 3):
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0 or active[ni, nj, nk] == 0:
                    continue
                donor = ni * ny * nz + nj * nz + nk
                rho = moments[donor]
                jx = moments[stride + donor]
                jy = moments[2 * stride + donor]
                jz = moments[3 * stride + donor]
                valid = (
                    wp.isfinite(rho)
                    and rho > 0.0
                    and wp.isfinite(jx)
                    and wp.isfinite(jy)
                    and wp.isfinite(jz)
                )
                if not valid:
                    valid_donors = 0
                    continue
                if donors == 0:
                    first_x = jx
                    first_y = jy
                    first_z = jz
                else:
                    if jx != first_x:
                        uniform_x = 0
                    if jy != first_y:
                        uniform_y = 0
                    if jz != first_z:
                        uniform_z = 0
                dx = float(di)
                dy = float(dj)
                dz = float(dk)
                row = Vec4f(1.0, dx, dy, dz)
                weight = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz)
                for row_index in range(4):
                    rhs_x[row_index] += weight * row[row_index] * jx
                    rhs_y[row_index] += weight * row[row_index] * jy
                    rhs_z[row_index] += weight * row[row_index] * jz
                    for column_index in range(4):
                        gram[row_index, column_index] += (
                            weight * row[row_index] * row[column_index]
                        )
                donors += 1
    donor_count[i, j, k] = donors
    if valid_donors == 0:
        wp.atomic_add(counts, 0, 1)
        return
    if donors < 4:
        wp.atomic_add(counts, 1, 1)
        return
    largest_entry = float(0.0)
    for row_index in range(4):
        for column_index in range(4):
            largest_entry = wp.max(
                largest_entry, wp.abs(gram[row_index, column_index])
            )
    largest_pivot = float(0.0)
    smallest_pivot = float(1.0e30)
    full_rank = int(1)
    for column_index in range(4):
        pivot_row = column_index
        pivot_magnitude = wp.abs(gram[column_index, column_index])
        for candidate_row in range(column_index + 1, 4):
            candidate = wp.abs(gram[candidate_row, column_index])
            if candidate > pivot_magnitude:
                pivot_magnitude = candidate
                pivot_row = candidate_row
        if pivot_magnitude <= 1.0e-7 * largest_entry:
            full_rank = 0
        if full_rank != 0:
            if pivot_row != column_index:
                for swap_column in range(4):
                    temporary = gram[column_index, swap_column]
                    gram[column_index, swap_column] = gram[pivot_row, swap_column]
                    gram[pivot_row, swap_column] = temporary
                temporary = rhs_x[column_index]
                rhs_x[column_index] = rhs_x[pivot_row]
                rhs_x[pivot_row] = temporary
                temporary = rhs_y[column_index]
                rhs_y[column_index] = rhs_y[pivot_row]
                rhs_y[pivot_row] = temporary
                temporary = rhs_z[column_index]
                rhs_z[column_index] = rhs_z[pivot_row]
                rhs_z[pivot_row] = temporary
            pivot = gram[column_index, column_index]
            pivot_absolute = wp.abs(pivot)
            largest_pivot = wp.max(largest_pivot, pivot_absolute)
            smallest_pivot = wp.min(smallest_pivot, pivot_absolute)
            for eliminate_row in range(column_index + 1, 4):
                factor = gram[eliminate_row, column_index] / pivot
                gram[eliminate_row, column_index] = 0.0
                for eliminate_column in range(column_index + 1, 4):
                    gram[eliminate_row, eliminate_column] -= (
                        factor * gram[column_index, eliminate_column]
                    )
                rhs_x[eliminate_row] -= factor * rhs_x[column_index]
                rhs_y[eliminate_row] -= factor * rhs_y[column_index]
                rhs_z[eliminate_row] -= factor * rhs_z[column_index]
    if (
        full_rank == 0
        or smallest_pivot * maximum_fit_condition * maximum_fit_condition
        < largest_pivot
    ):
        wp.atomic_add(counts, 2, 1)
        return
    if uniform_x != 0 and uniform_y != 0 and uniform_z != 0:
        wp.atomic_add(counts, 3, 1)
        return

    coefficient_x = Vec4f()
    coefficient_y = Vec4f()
    coefficient_z = Vec4f()
    for reverse_index in range(4):
        row_index = 3 - reverse_index
        value_x = rhs_x[row_index]
        value_y = rhs_y[row_index]
        value_z = rhs_z[row_index]
        for column_index in range(row_index + 1, 4):
            value_x -= gram[row_index, column_index] * coefficient_x[column_index]
            value_y -= gram[row_index, column_index] * coefficient_y[column_index]
            value_z -= gram[row_index, column_index] * coefficient_z[column_index]
        inverse = 1.0 / gram[row_index, row_index]
        coefficient_x[row_index] = value_x * inverse
        coefficient_y[row_index] = value_y * inverse
        coefficient_z[row_index] = value_z * inverse

    sxx = coefficient_x[1]
    syy = coefficient_y[2]
    szz = coefficient_z[3]
    sxy = 0.5 * (coefficient_x[2] + coefficient_y[1])
    sxz = 0.5 * (coefficient_x[3] + coefficient_z[1])
    syz = 0.5 * (coefficient_y[3] + coefficient_z[2])
    valid_result = (
        wp.isfinite(sxx)
        and wp.isfinite(syy)
        and wp.isfinite(szz)
        and wp.isfinite(sxy)
        and wp.isfinite(sxz)
        and wp.isfinite(syz)
    )
    if not valid_result:
        wp.atomic_add(counts, 0, 1)
        return
    bulk_strain[cell] = sxx
    bulk_strain[stride + cell] = syy
    bulk_strain[2 * stride + cell] = szz
    bulk_strain[3 * stride + cell] = sxy
    bulk_strain[4 * stride + cell] = sxz
    bulk_strain[5 * stride + cell] = syz
    wp.atomic_add(counts, 3, 1)


@wp.kernel
def build_fsl_normal_strain_kernel(
    moments: wp.array(dtype=float),
    active: wp.array3d(dtype=wp.int32),
    link_status: wp.array(dtype=wp.int32),
    plane_owner: wp.array(dtype=wp.int32),
    fraction: wp.array(dtype=float),
    gas_density: wp.array3d(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    normal_strain: wp.array(dtype=float),
    counts: wp.array(dtype=wp.int32),
    no_support_policy: int,
    lattice_viscosity: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    link = wp.tid()
    normal_strain[link] = 0.0
    status = link_status[link]
    if status == -7:
        wp.atomic_add(counts, 1, 1)
        if no_support_policy == 0:
            wp.atomic_add(counts, 0, 1)
        return
    if status != 1 and status != 2:
        return
    q = link % 27
    cell = link // 27
    k = cell % nz
    remainder = cell // nz
    j = remainder % ny
    i = remainder // ny
    c = directions[q]
    support_i = _neighbor(i + int(c[0]), nx, periodic_x)
    support_j = _neighbor(j + int(c[1]), ny, periodic_y)
    support_k = _neighbor(k + int(c[2]), nz, periodic_z)
    if (
        support_i < 0
        or support_j < 0
        or support_k < 0
        or active[support_i, support_j, support_k] == 0
    ):
        wp.atomic_add(counts, 0, 1)
        return
    support = support_i * ny * nz + support_j * nz + support_k
    owner_i = i
    owner_j = j
    owner_k = k
    owner = plane_owner[link]
    if owner == 2:
        owner_i = _neighbor(i - int(c[0]), nx, periodic_x)
        owner_j = _neighbor(j - int(c[1]), ny, periodic_y)
        owner_k = _neighbor(k - int(c[2]), nz, periodic_z)
    elif owner != 1:
        wp.atomic_add(counts, 0, 1)
        return
    if owner_i < 0 or owner_j < 0 or owner_k < 0:
        wp.atomic_add(counts, 0, 1)
        return
    local_rho = moments[cell]
    support_rho = moments[support]
    rho_gas = gas_density[owner_i, owner_j, owner_k]
    delta = fraction[link]
    rho_liquid_boundary = (1.0 + delta) * local_rho - delta * support_rho
    target = (rho_liquid_boundary - rho_gas) / (6.0 * lattice_viscosity)
    if (
        not wp.isfinite(local_rho)
        or local_rho <= 0.0
        or not wp.isfinite(support_rho)
        or support_rho <= 0.0
        or not wp.isfinite(rho_gas)
        or rho_gas <= 0.0
        or not wp.isfinite(delta)
        or delta < 0.0
        or delta > 1.0
        or not wp.isfinite(target)
    ):
        wp.atomic_add(counts, 0, 1)
        return
    normal_strain[link] = target
