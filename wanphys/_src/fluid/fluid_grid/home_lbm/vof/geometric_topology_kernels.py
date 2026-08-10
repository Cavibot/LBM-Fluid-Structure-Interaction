# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for transactional geometric PLIC topology resolution."""

from __future__ import annotations

import warp as wp


GAS = wp.constant(0)
INTERFACE = wp.constant(1)
LIQUID = wp.constant(2)
SOLID = wp.constant(3)


@wp.func
def _neighbor(value: int, size: int, periodic: int) -> int:
    if value < 0 or value >= size:
        if periodic != 0:
            return (value + size) % size
        return -1
    return value


@wp.func
def _valid_moments(moments: wp.array(dtype=float), cell: int, stride: int) -> bool:
    rho = moments[cell]
    valid = wp.isfinite(rho) and rho > 0.0
    for component in range(1, 10):
        valid = valid and wp.isfinite(moments[component * stride + cell])
    return valid


@wp.kernel
def classify_geometric_topology_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    transported_fill: wp.array3d(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    donor_count: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    snapped_volume_delta: wp.array(dtype=float),
    transition_tolerance: float,
):
    i, j, k = wp.tid()
    source = source_flags[i, j, k]
    fill = transported_fill[i, j, k]
    donor_count[i, j, k] = 0
    if source < GAS or source > SOLID or not wp.isfinite(fill) or fill < 0.0 or fill > 1.0:
        wp.atomic_add(counts, 5, 1)
        resolved_flags[i, j, k] = GAS
        normalized_fill[i, j, k] = 0.0
        return

    target = INTERFACE
    normalized = fill
    if source == SOLID:
        target = SOLID
        normalized = 0.0
        if fill > transition_tolerance:
            wp.atomic_add(counts, 6, 1)
    elif fill <= transition_tolerance:
        target = GAS
        normalized = 0.0
    elif fill >= 1.0 - transition_tolerance:
        target = LIQUID
        normalized = 1.0

    if normalized != fill:
        wp.atomic_add(counts, 4, 1)
        wp.atomic_add(snapped_volume_delta, 0, normalized - fill)
    if source == GAS and target == LIQUID:
        wp.atomic_add(counts, 7, 1)
    elif source == LIQUID and target == GAS:
        wp.atomic_add(counts, 7, 1)
    elif source == GAS and target == INTERFACE:
        wp.atomic_add(counts, 0, 1)
    elif source == LIQUID and target == INTERFACE:
        wp.atomic_add(counts, 1, 1)
    elif source == INTERFACE and target == GAS:
        wp.atomic_add(counts, 2, 1)
    elif source == INTERFACE and target == LIQUID:
        wp.atomic_add(counts, 3, 1)
    resolved_flags[i, j, k] = target
    normalized_fill[i, j, k] = normalized


@wp.kernel
def close_geometric_topology_separation_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    transported_fill: wp.array3d(dtype=float),
    classified_flags: wp.array3d(dtype=wp.int32),
    classified_fill: wp.array3d(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_kind: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    snapped_volume_delta: wp.array(dtype=float),
    endpoint_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    closure_kind[i, j, k] = 0
    source = source_flags[i, j, k]
    target = classified_flags[i, j, k]
    transported = transported_fill[i, j, k]
    normalized = classified_fill[i, j, k]
    lower_endpoint = target == GAS and source != SOLID
    upper_endpoint = target == LIQUID
    needs_closure = bool(False)
    if lower_endpoint or upper_endpoint:
        forbidden = GAS
        if lower_endpoint:
            forbidden = LIQUID
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = _neighbor(i + di, nx, periodic_x)
                    nj = _neighbor(j + dj, ny, periodic_y)
                    nk = _neighbor(k + dk, nz, periodic_z)
                    if (
                        ni >= 0
                        and nj >= 0
                        and nk >= 0
                        and classified_flags[ni, nj, nk] == forbidden
                    ):
                        neighbor_source = source_flags[ni, nj, nk]
                        if source == INTERFACE:
                            if lower_endpoint and transported > 0.0:
                                needs_closure = True
                            elif upper_endpoint and transported < 1.0:
                                needs_closure = True
                        elif (
                            lower_endpoint
                            and source == GAS
                            and neighbor_source == INTERFACE
                            and transported_fill[ni, nj, nk] == 1.0
                        ):
                            needs_closure = True
                        elif (
                            upper_endpoint
                            and source == LIQUID
                            and neighbor_source == INTERFACE
                            and transported_fill[ni, nj, nk] == 0.0
                        ):
                            needs_closure = True

    if needs_closure:
        target = INTERFACE
        if lower_endpoint:
            normalized = wp.max(transported, endpoint_tolerance)
            if source == INTERFACE:
                wp.atomic_add(counts, 2, -1)
            else:
                wp.atomic_add(counts, 0, 1)
        else:
            normalized = wp.min(transported, 1.0 - endpoint_tolerance)
            if source == INTERFACE:
                wp.atomic_add(counts, 3, -1)
            else:
                wp.atomic_add(counts, 1, 1)
        if normalized == transported and classified_fill[i, j, k] != transported:
            wp.atomic_add(counts, 4, -1)
        elif normalized != transported and classified_fill[i, j, k] == transported:
            wp.atomic_add(counts, 4, 1)
        wp.atomic_add(counts, 11, 1)
        if lower_endpoint:
            closure_kind[i, j, k] = 1
        else:
            closure_kind[i, j, k] = -1
        wp.atomic_add(
            snapped_volume_delta,
            0,
            normalized - classified_fill[i, j, k],
        )
    elif target == INTERFACE and transported < endpoint_tolerance:
        normalized = endpoint_tolerance
        closure_kind[i, j, k] = 3
        wp.atomic_add(counts, 4, 1)
        wp.atomic_add(
            snapped_volume_delta,
            0,
            normalized - classified_fill[i, j, k],
        )
    elif target == INTERFACE and transported > 1.0 - endpoint_tolerance:
        normalized = 1.0 - endpoint_tolerance
        closure_kind[i, j, k] = -3
        wp.atomic_add(counts, 4, 1)
        wp.atomic_add(
            snapped_volume_delta,
            0,
            normalized - classified_fill[i, j, k],
        )
    resolved_flags[i, j, k] = target
    normalized_fill[i, j, k] = normalized


@wp.kernel
def initialize_conservative_geometric_closure_kernel(
    transported_fill: wp.array3d(dtype=float),
    classified_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_base_fill: wp.array3d(dtype=float),
    closure_kind: wp.array3d(dtype=wp.int32),
    closure_partner: wp.array3d(dtype=wp.int32),
    closure_proposal: wp.array3d(dtype=wp.int32),
    closure_reserved: wp.array(dtype=wp.int32),
    closure_partner_volume_delta: wp.array3d(dtype=float),
    closure_self_mass_delta: wp.array3d(dtype=float),
    closure_partner_mass_delta: wp.array3d(dtype=float),
    counts: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    kind = closure_kind[i, j, k]
    target = classified_flags[i, j, k]
    fill = transported_fill[i, j, k]
    if kind == 0:
        if target == GAS and fill > 0.0:
            kind = 2
        elif target == LIQUID and fill < 1.0:
            kind = -2
    closure_kind[i, j, k] = kind
    closure_base_fill[i, j, k] = normalized_fill[i, j, k]
    closure_partner[i, j, k] = -1
    closure_proposal[i, j, k] = -1
    closure_reserved[cell] = 0
    closure_partner_volume_delta[i, j, k] = 0.0
    closure_self_mass_delta[i, j, k] = 0.0
    closure_partner_mass_delta[i, j, k] = 0.0
    if kind != 0:
        delta = wp.abs(normalized_fill[i, j, k] - fill)
        if kind == 1:
            delta = normalized_fill[i, j, k]
        elif kind == -1:
            delta = 1.0 - normalized_fill[i, j, k]
        elif kind == 2:
            delta = fill
        elif kind == -2:
            delta = 1.0 - fill
        if not wp.isfinite(delta) or delta <= 0.0:
            wp.atomic_add(counts, 13, 1)


@wp.kernel
def prepare_regular_conservative_geometric_closure_kernel(
    transported_fill: wp.array3d(dtype=float),
    transported_mass: wp.array3d(dtype=float),
    classified_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_kind: wp.array3d(dtype=wp.int32),
    closure_partner: wp.array3d(dtype=wp.int32),
    closure_partner_volume_delta: wp.array3d(dtype=float),
    closure_self_mass_delta: wp.array3d(dtype=float),
    closure_partner_mass_delta: wp.array3d(dtype=float),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    kind = closure_kind[i, j, k]
    if kind == 0 or kind == 3 or kind == -3:
        return
    fill = transported_fill[i, j, k]
    delta = fill
    if kind == 1:
        delta = normalized_fill[i, j, k]
    elif kind == -1:
        delta = 1.0 - normalized_fill[i, j, k]
    elif kind == -2:
        delta = 1.0 - fill
    if not wp.isfinite(delta) or delta <= 0.0:
        return

    partner_flag = int(INTERFACE)
    if kind == 1:
        partner_flag = int(LIQUID)
    elif kind == -1:
        partner_flag = int(GAS)
    partner = int(-1)
    partner_i = int(-1)
    partner_j = int(-1)
    partner_k = int(-1)
    selected_distance = int(4)
    selected_squared_distance = int(16)
    selected_offset_rank = int(27)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                candidate_flag = classified_flags[ni, nj, nk]
                if candidate_flag != partner_flag:
                    continue
                candidate_fill = transported_fill[ni, nj, nk]
                candidate_mass = transported_mass[ni, nj, nk]
                valid = bool(True)
                if kind == 2:
                    valid = candidate_fill + fill < 1.0
                elif kind == -2:
                    valid = candidate_fill > 1.0 - fill and candidate_mass > 0.0
                linear = ni * ny * nz + nj * nz + nk
                distance = wp.max(wp.abs(di), wp.max(wp.abs(dj), wp.abs(dk)))
                squared_distance = di * di + dj * dj + dk * dk
                offset_rank = (di + 1) * 9 + (dj + 1) * 3 + dk + 1
                better = (
                    partner < 0
                    or distance < selected_distance
                    or (
                        distance == selected_distance
                        and squared_distance < selected_squared_distance
                    )
                    or (
                        distance == selected_distance
                        and squared_distance == selected_squared_distance
                        and offset_rank < selected_offset_rank
                    )
                )
                if valid and better:
                    partner = linear
                    partner_i = ni
                    partner_j = nj
                    partner_k = nk
                    selected_distance = distance
                    selected_squared_distance = squared_distance
                    selected_offset_rank = offset_rank
    if partner < 0:
        wp.atomic_add(counts, 13, 1)
        return

    concentration = float(0.0)
    if kind == 1 or kind == -2:
        donor_fill = transported_fill[partner_i, partner_j, partner_k]
        donor_mass = transported_mass[partner_i, partner_j, partner_k]
        if (
            wp.isfinite(donor_fill)
            and donor_fill > 0.0
            and wp.isfinite(donor_mass)
            and donor_mass > 0.0
        ):
            concentration = donor_mass / donor_fill
    else:
        donor_fill = fill
        donor_mass = transported_mass[i, j, k]
        if (
            wp.isfinite(donor_fill)
            and donor_fill > 0.0
            and wp.isfinite(donor_mass)
            and donor_mass >= 0.0
        ):
            concentration = donor_mass / donor_fill
    transfer = concentration * delta
    valid_transfer = wp.isfinite(transfer) and transfer > 0.0
    if kind == 2:
        transfer = transported_mass[i, j, k]
        valid_transfer = wp.isfinite(transfer) and transfer >= 0.0
    if not valid_transfer:
        wp.atomic_add(counts, 13, 1)
        return

    closure_partner[i, j, k] = partner
    if kind == 1:
        closure_partner_volume_delta[i, j, k] = -delta
        closure_self_mass_delta[i, j, k] = transfer
        closure_partner_mass_delta[i, j, k] = -transfer
    elif kind == -1:
        closure_partner_volume_delta[i, j, k] = delta
        closure_self_mass_delta[i, j, k] = -transfer
        closure_partner_mass_delta[i, j, k] = transfer
    elif kind == 2:
        closure_partner_volume_delta[i, j, k] = delta
        closure_self_mass_delta[i, j, k] = -transfer
        closure_partner_mass_delta[i, j, k] = transfer
    else:
        closure_partner_volume_delta[i, j, k] = -delta
        closure_self_mass_delta[i, j, k] = transfer
        closure_partner_mass_delta[i, j, k] = -transfer


@wp.kernel
def gather_regular_conservative_geometric_closure_kernel(
    closure_kind: wp.array3d(dtype=wp.int32),
    closure_partner: wp.array3d(dtype=wp.int32),
    closure_partner_volume_delta: wp.array3d(dtype=float),
    closure_partner_mass_delta: wp.array3d(dtype=float),
    closure_incoming_volume: wp.array3d(dtype=float),
    closure_incoming_mass: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    volume = float(0.0)
    mass = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                kind = closure_kind[ni, nj, nk]
                if (
                    kind != 3
                    and kind != -3
                    and closure_partner[ni, nj, nk] == cell
                ):
                    volume += closure_partner_volume_delta[ni, nj, nk]
                    mass += closure_partner_mass_delta[ni, nj, nk]
    closure_incoming_volume[i, j, k] = volume
    closure_incoming_mass[i, j, k] = mass


@wp.kernel
def reset_endpoint_closure_proposals_kernel(
    closure_owner: wp.array(dtype=wp.int32),
    closure_proposal: wp.array3d(dtype=wp.int32),
    closure_reserved: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    closure_proposal[i, j, k] = -1
    if closure_reserved[cell] != 0:
        closure_owner[cell] = -1
    else:
        closure_owner[cell] = 2147483647


@wp.kernel
def propose_endpoint_conservative_geometric_closure_kernel(
    transported_fill: wp.array3d(dtype=float),
    transported_mass: wp.array3d(dtype=float),
    classified_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_kind: wp.array3d(dtype=wp.int32),
    closure_partner: wp.array3d(dtype=wp.int32),
    closure_proposal: wp.array3d(dtype=wp.int32),
    closure_owner: wp.array(dtype=wp.int32),
    closure_reserved: wp.array(dtype=wp.int32),
    closure_incoming_volume: wp.array3d(dtype=float),
    closure_incoming_mass: wp.array3d(dtype=float),
    closure_base_fill: wp.array3d(dtype=float),
    closure_partner_volume_delta: wp.array3d(dtype=float),
    closure_self_mass_delta: wp.array3d(dtype=float),
    closure_partner_mass_delta: wp.array3d(dtype=float),
    endpoint_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    kind = closure_kind[i, j, k]
    if (kind != 3 and kind != -3) or closure_partner[i, j, k] >= 0:
        return
    source_fill = transported_fill[i, j, k]
    delta = normalized_fill[i, j, k] - source_fill
    if kind == -3:
        delta = source_fill - normalized_fill[i, j, k]
    if not wp.isfinite(delta) or delta <= 0.0:
        return

    selected_partner = int(-1)
    selected_i = int(-1)
    selected_j = int(-1)
    selected_k = int(-1)
    selected_delta = float(0.0)
    selected_distance = int(8)
    selected_squared_distance = int(128)
    selected_phase_rank = int(2)
    selected_capacity = float(-1.0)
    selected_offset_rank = int(343)
    for di in range(-3, 4):
        for dj in range(-3, 4):
            for dk in range(-3, 4):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                linear = ni * ny * nz + nj * nz + nk
                if closure_reserved[linear] != 0 or closure_kind[ni, nj, nk] != 0:
                    continue
                candidate_flag = classified_flags[ni, nj, nk]
                phase_matches = (
                    (kind == 3 and (candidate_flag == INTERFACE or candidate_flag == LIQUID))
                    or (kind == -3 and (candidate_flag == INTERFACE or candidate_flag == GAS))
                )
                if not phase_matches:
                    continue

                candidate_before = (
                    normalized_fill[ni, nj, nk]
                    + closure_incoming_volume[ni, nj, nk]
                )
                candidate_mass_before = (
                    transported_mass[ni, nj, nk]
                    + closure_incoming_mass[ni, nj, nk]
                )
                candidate_delta = delta
                if (kind == 3 and candidate_flag == LIQUID) or (
                    kind == -3 and candidate_flag == GAS
                ):
                    candidate_delta = wp.max(candidate_delta, endpoint_tolerance)
                candidate_after = candidate_before - candidate_delta
                if kind == -3:
                    candidate_after = candidate_before + candidate_delta
                for _attempt in range(24):
                    partner_actual = candidate_before - candidate_after
                    source_after = source_fill + partner_actual
                    source_actual = source_after - source_fill
                    if kind == -3:
                        partner_actual = candidate_after - candidate_before
                        source_after = source_fill - partner_actual
                        source_actual = source_fill - source_after
                    if partner_actual >= delta and source_actual == partner_actual:
                        break
                    next_delta = wp.max(delta, wp.max(partner_actual, source_actual))
                    if next_delta <= candidate_delta:
                        next_delta = candidate_delta * 2.0
                    candidate_delta = next_delta
                    candidate_after = candidate_before - candidate_delta
                    if kind == -3:
                        candidate_after = candidate_before + candidate_delta
                partner_actual = candidate_before - candidate_after
                source_after = source_fill + partner_actual
                source_actual = source_after - source_fill
                capacity = candidate_before - endpoint_tolerance
                if kind == -3:
                    partner_actual = candidate_after - candidate_before
                    source_after = source_fill - partner_actual
                    source_actual = source_fill - source_after
                    capacity = 1.0 - endpoint_tolerance - candidate_before
                valid = (
                    wp.isfinite(partner_actual)
                    and partner_actual > 0.0
                    and source_actual == partner_actual
                    and wp.isfinite(candidate_after)
                    and candidate_after >= endpoint_tolerance
                    and candidate_after <= 1.0 - endpoint_tolerance
                    and wp.isfinite(source_after)
                    and source_after >= endpoint_tolerance
                    and source_after <= 1.0 - endpoint_tolerance
                    and capacity >= partner_actual
                )
                if kind == 3:
                    valid = valid and candidate_mass_before > 0.0
                if not valid:
                    continue

                distance = wp.max(wp.abs(di), wp.max(wp.abs(dj), wp.abs(dk)))
                squared_distance = di * di + dj * dj + dk * dk
                phase_rank = int(0)
                if candidate_flag != INTERFACE:
                    phase_rank = 1
                offset_rank = (di + 3) * 49 + (dj + 3) * 7 + dk + 3
                better = (
                    selected_partner < 0
                    or distance < selected_distance
                    or (distance == selected_distance and squared_distance < selected_squared_distance)
                    or (
                        distance == selected_distance
                        and squared_distance == selected_squared_distance
                        and phase_rank < selected_phase_rank
                    )
                    or (
                        distance == selected_distance
                        and squared_distance == selected_squared_distance
                        and phase_rank == selected_phase_rank
                        and capacity > selected_capacity
                    )
                    or (
                        distance == selected_distance
                        and squared_distance == selected_squared_distance
                        and phase_rank == selected_phase_rank
                        and capacity == selected_capacity
                        and offset_rank < selected_offset_rank
                    )
                )
                if better:
                    selected_partner = linear
                    selected_i = ni
                    selected_j = nj
                    selected_k = nk
                    selected_delta = partner_actual
                    selected_distance = distance
                    selected_squared_distance = squared_distance
                    selected_phase_rank = phase_rank
                    selected_capacity = capacity
                    selected_offset_rank = offset_rank
    if selected_partner < 0:
        return

    concentration = float(0.0)
    if kind == 3:
        donor_fill = (
            normalized_fill[selected_i, selected_j, selected_k]
            + closure_incoming_volume[selected_i, selected_j, selected_k]
        )
        donor_mass = (
            transported_mass[selected_i, selected_j, selected_k]
            + closure_incoming_mass[selected_i, selected_j, selected_k]
        )
        if donor_fill > 0.0 and donor_mass > 0.0:
            concentration = donor_mass / donor_fill
    else:
        donor_mass = transported_mass[i, j, k]
        if source_fill > 0.0 and donor_mass > 0.0:
            concentration = donor_mass / source_fill
    transfer = concentration * selected_delta
    if not wp.isfinite(transfer) or transfer <= 0.0:
        return

    closure_proposal[i, j, k] = selected_partner
    if kind == 3:
        closure_base_fill[i, j, k] = source_fill + selected_delta
        closure_partner_volume_delta[i, j, k] = -selected_delta
        closure_self_mass_delta[i, j, k] = transfer
        closure_partner_mass_delta[i, j, k] = -transfer
    else:
        closure_base_fill[i, j, k] = source_fill - selected_delta
        closure_partner_volume_delta[i, j, k] = selected_delta
        closure_self_mass_delta[i, j, k] = -transfer
        closure_partner_mass_delta[i, j, k] = transfer
    source = i * ny * nz + j * nz + k
    wp.atomic_min(closure_owner, selected_partner, source)


@wp.kernel
def accept_endpoint_conservative_geometric_closure_kernel(
    closure_kind: wp.array3d(dtype=wp.int32),
    closure_proposal: wp.array3d(dtype=wp.int32),
    closure_owner: wp.array(dtype=wp.int32),
    closure_reserved: wp.array(dtype=wp.int32),
    closure_partner: wp.array3d(dtype=wp.int32),
    closure_partner_volume_delta: wp.array3d(dtype=float),
    closure_partner_mass_delta: wp.array3d(dtype=float),
    closure_incoming_volume: wp.array3d(dtype=float),
    closure_incoming_mass: wp.array3d(dtype=float),
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    kind = closure_kind[i, j, k]
    if kind != 3 and kind != -3:
        return
    proposal = closure_proposal[i, j, k]
    if proposal < 0:
        return
    source = i * ny * nz + j * nz + k
    if closure_owner[proposal] != source:
        return
    partner_i = proposal // (ny * nz)
    remainder = proposal - partner_i * ny * nz
    partner_j = remainder // nz
    partner_k = remainder - partner_j * nz
    closure_reserved[proposal] = 1
    closure_partner[i, j, k] = proposal
    closure_incoming_volume[partner_i, partner_j, partner_k] = (
        closure_incoming_volume[partner_i, partner_j, partner_k]
        + closure_partner_volume_delta[i, j, k]
    )
    closure_incoming_mass[partner_i, partner_j, partner_k] = (
        closure_incoming_mass[partner_i, partner_j, partner_k]
        + closure_partner_mass_delta[i, j, k]
    )


@wp.kernel
def count_unmatched_endpoint_closures_kernel(
    closure_kind: wp.array3d(dtype=wp.int32),
    closure_partner: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    kind = closure_kind[i, j, k]
    if (kind == 3 or kind == -3) and closure_partner[i, j, k] < 0:
        wp.atomic_add(counts, 13, 1)


@wp.kernel
def apply_conservative_geometric_closure_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    transported_fill: wp.array3d(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_base_fill: wp.array3d(dtype=float),
    closure_self_mass_delta: wp.array3d(dtype=float),
    closure_incoming_volume: wp.array3d(dtype=float),
    closure_incoming_mass: wp.array3d(dtype=float),
    closure_mass_delta: wp.array3d(dtype=float),
    counts: wp.array(dtype=wp.int32),
    snapped_volume_delta: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    delta = closure_incoming_volume[i, j, k]
    mass_delta = (
        closure_self_mass_delta[i, j, k] + closure_incoming_mass[i, j, k]
    )
    closure_mass_delta[i, j, k] = mass_delta
    original_before = normalized_fill[i, j, k]
    before = closure_base_fill[i, j, k]
    base_delta = before - original_before
    if delta == 0.0 and base_delta == 0.0:
        return
    after = before + delta
    if not wp.isfinite(after) or after < 0.0 or after > 1.0:
        wp.atomic_add(counts, 13, 1)
        return
    source = source_flags[i, j, k]
    previous_target = resolved_flags[i, j, k]
    target = int(INTERFACE)
    if after == 0.0:
        target = int(GAS)
    elif after == 1.0:
        target = int(LIQUID)

    if source == GAS and previous_target == INTERFACE:
        wp.atomic_add(counts, 0, -1)
    elif source == LIQUID and previous_target == INTERFACE:
        wp.atomic_add(counts, 1, -1)
    elif source == INTERFACE and previous_target == GAS:
        wp.atomic_add(counts, 2, -1)
    elif source == INTERFACE and previous_target == LIQUID:
        wp.atomic_add(counts, 3, -1)
    elif (source == GAS and previous_target == LIQUID) or (
        source == LIQUID and previous_target == GAS
    ):
        wp.atomic_add(counts, 7, -1)

    if source == GAS and target == INTERFACE:
        wp.atomic_add(counts, 0, 1)
    elif source == LIQUID and target == INTERFACE:
        wp.atomic_add(counts, 1, 1)
    elif source == INTERFACE and target == GAS:
        wp.atomic_add(counts, 2, 1)
    elif source == INTERFACE and target == LIQUID:
        wp.atomic_add(counts, 3, 1)
    elif (source == GAS and target == LIQUID) or (
        source == LIQUID and target == GAS
    ):
        wp.atomic_add(counts, 7, 1)
    was_snapped = original_before != transported_fill[i, j, k]
    is_snapped = after != transported_fill[i, j, k]
    if was_snapped and not is_snapped:
        wp.atomic_add(counts, 4, -1)
    elif not was_snapped and is_snapped:
        wp.atomic_add(counts, 4, 1)
    resolved_flags[i, j, k] = target
    normalized_fill[i, j, k] = after
    wp.atomic_add(snapped_volume_delta, 0, base_delta + delta)


@wp.kernel
def prepare_direct_separation_repair_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    transported_mass: wp.array3d(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_mass_delta: wp.array3d(dtype=float),
    repair_partner: wp.array3d(dtype=wp.int32),
    repair_self_volume: wp.array3d(dtype=float),
    repair_self_mass: wp.array3d(dtype=float),
    repair_partner_volume: wp.array3d(dtype=float),
    repair_partner_mass: wp.array3d(dtype=float),
    endpoint_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    """Pair collapsed interface gas cells with deterministic liquid donors."""

    i, j, k = wp.tid()
    repair_partner[i, j, k] = -1
    repair_self_volume[i, j, k] = 0.0
    repair_self_mass[i, j, k] = 0.0
    repair_partner_volume[i, j, k] = 0.0
    repair_partner_mass[i, j, k] = 0.0
    if resolved_flags[i, j, k] != GAS:
        return

    partner = int(-1)
    partner_i = int(-1)
    partner_j = int(-1)
    partner_k = int(-1)
    selected_distance = int(4)
    selected_squared_distance = int(16)
    selected_offset_rank = int(27)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                if resolved_flags[ni, nj, nk] != LIQUID:
                    continue
                repairable = source_flags[i, j, k] == INTERFACE
                repairable = repairable or source_flags[ni, nj, nk] == INTERFACE
                linear = ni * ny * nz + nj * nz + nk
                distance = wp.max(wp.abs(di), wp.max(wp.abs(dj), wp.abs(dk)))
                squared_distance = di * di + dj * dj + dk * dk
                offset_rank = (di + 1) * 9 + (dj + 1) * 3 + dk + 1
                better = (
                    partner < 0
                    or distance < selected_distance
                    or (
                        distance == selected_distance
                        and squared_distance < selected_squared_distance
                    )
                    or (
                        distance == selected_distance
                        and squared_distance == selected_squared_distance
                        and offset_rank < selected_offset_rank
                    )
                )
                if repairable and better:
                    partner = linear
                    partner_i = ni
                    partner_j = nj
                    partner_k = nk
                    selected_distance = distance
                    selected_squared_distance = squared_distance
                    selected_offset_rank = offset_rank
    if partner < 0:
        return

    donor_fill = normalized_fill[partner_i, partner_j, partner_k]
    donor_mass = (
        transported_mass[partner_i, partner_j, partner_k]
        + closure_mass_delta[partner_i, partner_j, partner_k]
    )
    if (
        not wp.isfinite(donor_fill)
        or donor_fill <= 0.0
        or not wp.isfinite(donor_mass)
        or donor_mass <= 0.0
    ):
        return
    transfer = endpoint_tolerance * donor_mass / donor_fill
    if not wp.isfinite(transfer) or transfer <= 0.0:
        return
    repair_partner[i, j, k] = partner
    repair_self_volume[i, j, k] = endpoint_tolerance
    repair_self_mass[i, j, k] = transfer
    repair_partner_volume[i, j, k] = -endpoint_tolerance
    repair_partner_mass[i, j, k] = -transfer


@wp.kernel
def gather_direct_separation_repair_kernel(
    repair_partner: wp.array3d(dtype=wp.int32),
    repair_partner_volume: wp.array3d(dtype=float),
    repair_partner_mass: wp.array3d(dtype=float),
    repair_incoming_volume: wp.array3d(dtype=float),
    repair_incoming_mass: wp.array3d(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    cell = i * ny * nz + j * nz + k
    volume = float(0.0)
    mass = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    if repair_partner[ni, nj, nk] == cell:
                        volume += repair_partner_volume[ni, nj, nk]
                        mass += repair_partner_mass[ni, nj, nk]
    repair_incoming_volume[i, j, k] = volume
    repair_incoming_mass[i, j, k] = mass


@wp.kernel
def apply_direct_separation_repair_kernel(
    source_flags: wp.array3d(dtype=wp.int32),
    transported_fill: wp.array3d(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    closure_mass_delta: wp.array3d(dtype=float),
    repair_self_volume: wp.array3d(dtype=float),
    repair_self_mass: wp.array3d(dtype=float),
    repair_incoming_volume: wp.array3d(dtype=float),
    repair_incoming_mass: wp.array3d(dtype=float),
    counts: wp.array(dtype=wp.int32),
    snapped_volume_delta: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    volume_delta = (
        repair_self_volume[i, j, k] + repair_incoming_volume[i, j, k]
    )
    mass_delta = repair_self_mass[i, j, k] + repair_incoming_mass[i, j, k]
    if volume_delta == 0.0 and mass_delta == 0.0:
        return
    before = normalized_fill[i, j, k]
    after = before + volume_delta
    if (
        not wp.isfinite(after)
        or after <= 0.0
        or after >= 1.0
        or not wp.isfinite(mass_delta)
    ):
        wp.atomic_add(counts, 13, 1)
        return
    source = source_flags[i, j, k]
    target = resolved_flags[i, j, k]
    if target == GAS:
        if source == GAS:
            wp.atomic_add(counts, 0, 1)
        elif source == INTERFACE:
            wp.atomic_add(counts, 2, -1)
    elif target == LIQUID:
        if source == LIQUID:
            wp.atomic_add(counts, 1, 1)
        elif source == INTERFACE:
            wp.atomic_add(counts, 3, -1)
    else:
        wp.atomic_add(counts, 13, 1)
        return
    was_snapped = before != transported_fill[i, j, k]
    is_snapped = after != transported_fill[i, j, k]
    if was_snapped and not is_snapped:
        wp.atomic_add(counts, 4, -1)
    elif not was_snapped and is_snapped:
        wp.atomic_add(counts, 4, 1)
    resolved_flags[i, j, k] = INTERFACE
    normalized_fill[i, j, k] = after
    closure_mass_delta[i, j, k] += mass_delta
    wp.atomic_add(counts, 11, 1)
    wp.atomic_add(snapped_volume_delta, 0, volume_delta)


@wp.kernel
def count_geometric_topology_donors_kernel(
    moments: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    resolved_flags: wp.array3d(dtype=wp.int32),
    donor_count: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    source = source_flags[i, j, k]
    target = resolved_flags[i, j, k]
    source_active = source == INTERFACE or source == LIQUID
    target_active = target == INTERFACE or target == LIQUID
    if source_active and target_active:
        cell = i * ny * nz + j * nz + k
        if not _valid_moments(moments, cell, stride):
            wp.atomic_add(counts, 8, 1)
    if source != GAS or target != INTERFACE:
        return

    donors = int(0)
    valid = bool(True)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                donor_source = source_flags[ni, nj, nk]
                donor_target = resolved_flags[ni, nj, nk]
                if (
                    (donor_source == INTERFACE or donor_source == LIQUID)
                    and (donor_target == INTERFACE or donor_target == LIQUID)
                ):
                    donor = ni * ny * nz + nj * nz + nk
                    donors += 1
                    valid = valid and _valid_moments(moments, donor, stride)
    donor_count[i, j, k] = donors
    if donors == 0 or not valid:
        wp.atomic_add(counts, 8, 1)


@wp.kernel
def initialize_fresh_geometric_interface_kernel(
    moments_before: wp.array(dtype=float),
    source_flags: wp.array3d(dtype=wp.int32),
    resolved_flags: wp.array3d(dtype=wp.int32),
    donor_count: wp.array3d(dtype=wp.int32),
    moments_after: wp.array(dtype=float),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    if source_flags[i, j, k] != GAS or resolved_flags[i, j, k] != INTERFACE:
        return
    donors = donor_count[i, j, k]
    rho_sum = float(0.0)
    velocity_sum = wp.vec3(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni < 0 or nj < 0 or nk < 0:
                    continue
                donor_source = source_flags[ni, nj, nk]
                donor_target = resolved_flags[ni, nj, nk]
                if (
                    (donor_source == INTERFACE or donor_source == LIQUID)
                    and (donor_target == INTERFACE or donor_target == LIQUID)
                ):
                    donor = ni * ny * nz + nj * nz + nk
                    rho = moments_before[donor]
                    rho_sum += rho
                    velocity_sum += wp.vec3(
                        moments_before[stride + donor] / rho,
                        moments_before[2 * stride + donor] / rho,
                        moments_before[3 * stride + donor] / rho,
                    )
    inverse = 1.0 / float(donors)
    rho = rho_sum * inverse
    velocity = velocity_sum * inverse
    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]
    cell = i * ny * nz + j * nz + k
    moments_after[cell] = rho
    moments_after[stride + cell] = rho * ux
    moments_after[2 * stride + cell] = rho * uy
    moments_after[3 * stride + cell] = rho * uz
    moments_after[4 * stride + cell] = rho * ux * ux
    moments_after[5 * stride + cell] = rho * uy * uy
    moments_after[6 * stride + cell] = rho * uz * uz
    moments_after[7 * stride + cell] = rho * ux * uy
    moments_after[8 * stride + cell] = rho * ux * uz
    moments_after[9 * stride + cell] = rho * uy * uz


@wp.kernel
def validate_geometric_topology_kernel(
    moments: wp.array(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    validation_error: wp.array3d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
):
    i, j, k = wp.tid()
    flag = resolved_flags[i, j, k]
    error = int(0)
    if flag == INTERFACE or flag == LIQUID:
        cell = i * ny * nz + j * nz + k
        if not _valid_moments(moments, cell, stride):
            error = error | 1
            wp.atomic_add(counts, 9, 1)
    if flag != GAS and flag != LIQUID:
        validation_error[i, j, k] = error
        return
    forbidden = LIQUID
    if flag == LIQUID:
        forbidden = GAS
    invalid_link = bool(False)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni = _neighbor(i + di, nx, periodic_x)
                nj = _neighbor(j + dj, ny, periodic_y)
                nk = _neighbor(k + dk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    invalid_link = invalid_link or resolved_flags[ni, nj, nk] == forbidden
    if invalid_link:
        error = error | 2
        wp.atomic_add(counts, 10, 1)
    validation_error[i, j, k] = error


@wp.kernel
def build_geometric_topology_mass_kernel(
    moments: wp.array(dtype=float),
    transported_mass: wp.array3d(dtype=float),
    closure_mass_delta: wp.array3d(dtype=float),
    resolved_flags: wp.array3d(dtype=wp.int32),
    normalized_fill: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    mass_validation_error: wp.array3d(dtype=wp.int32),
    invalid_mass_count: wp.array(dtype=wp.int32),
    use_transported_mass: int,
    empty_mass_tolerance: float,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    mass_validation_error[i, j, k] = 0
    flag = resolved_flags[i, j, k]
    if flag == INTERFACE or flag == LIQUID:
        cell = i * ny * nz + j * nz + k
        value = moments[cell] * normalized_fill[i, j, k]
        if use_transported_mass != 0:
            value = transported_mass[i, j, k] + closure_mass_delta[i, j, k]
        if not wp.isfinite(value) or value <= 0.0:
            mass_validation_error[i, j, k] = 1
            wp.atomic_add(invalid_mass_count, 12, 1)
            return
        mass[i, j, k] = value
    else:
        if (
            use_transported_mass != 0
            and (
                not wp.isfinite(transported_mass[i, j, k])
                or wp.abs(
                    transported_mass[i, j, k] + closure_mass_delta[i, j, k]
                ) > empty_mass_tolerance
            )
        ):
            mass_validation_error[i, j, k] = 2
            wp.atomic_add(invalid_mass_count, 12, 1)
            return
        mass[i, j, k] = 0.0
