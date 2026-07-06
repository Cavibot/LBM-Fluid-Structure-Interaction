# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Helpers for merging split collision outputs into one Contacts object."""

from __future__ import annotations

from typing import Iterable

import warp as wp
from newton import Contacts


_RIGID_FIELDS = (
    "rigid_contact_point_id",
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_tids",
    "rigid_contact_force",
)

_RIGID_OPTIONAL_FIELDS = (
    "rigid_contact_stiffness",
    "rigid_contact_damping",
    "rigid_contact_friction",
)

_SOFT_FIELDS = (
    "soft_contact_particle",
    "soft_contact_shape",
    "soft_contact_body_pos",
    "soft_contact_body_vel",
    "soft_contact_normal",
    "soft_contact_tids",
)


def build_merged_contacts(
    *,
    rigid_src: Contacts | None = None,
    soft_src: Contacts | None = None,
) -> Contacts:
    """Allocate and return a unified Contacts object from split collision outputs."""

    template = rigid_src if rigid_src is not None else soft_src

    rigid_count = int(rigid_src.rigid_contact_count.numpy()[0]) if rigid_src is not None else 0
    soft_count = int(soft_src.soft_contact_count.numpy()[0]) if soft_src is not None else 0
    requested_attributes = _merge_requested_attributes(rigid_src, soft_src)

    dst = Contacts(
        rigid_contact_max=rigid_count,
        soft_contact_max=soft_count,
        requires_grad=bool(getattr(template, "requires_grad", False)) if template is not None else False,
        device=getattr(template, "device", None),
        per_contact_shape_properties=bool(getattr(rigid_src, "per_contact_shape_properties", False)),
        requested_attributes=requested_attributes,
    )

    if rigid_src is not None and rigid_src.per_contact_shape_properties and not dst.per_contact_shape_properties:
        raise ValueError("Destination contacts must enable per_contact_shape_properties for rigid source data")

    if rigid_src is not None and rigid_count > 0:
        _copy_fields(dst, rigid_src, _RIGID_FIELDS, rigid_count)
        _copy_optional_fields(dst, rigid_src, _RIGID_OPTIONAL_FIELDS, rigid_count)

    if soft_src is not None and soft_count > 0:
        _copy_fields(dst, soft_src, _SOFT_FIELDS, soft_count)

    dst.rigid_contact_count.assign([rigid_count])
    dst.soft_contact_count.assign([soft_count])
    return dst


def _copy_fields(dst: Contacts, src: Contacts, field_names: Iterable[str], count: int) -> None:
    for field_name in field_names:
        src_array = getattr(src, field_name, None)
        dst_array = getattr(dst, field_name, None)
        if src_array is None or dst_array is None:
            continue
        wp.copy(dst_array[:count], src_array[:count])


def _copy_optional_fields(dst: Contacts, src: Contacts, field_names: Iterable[str], count: int) -> None:
    for field_name in field_names:
        src_array = getattr(src, field_name, None)
        if src_array is None:
            continue
        dst_array = getattr(dst, field_name, None)
        if dst_array is None:
            raise ValueError(f"Destination contacts missing required field {field_name!r}")
        wp.copy(dst_array[:count], src_array[:count])


def _merge_requested_attributes(*sources: Contacts | None) -> set[str] | None:
    requested_attributes: set[str] = set()

    for src in sources:
        if src is None:
            continue

        attrs = getattr(src, "requested_attributes", None)
        if attrs:
            requested_attributes.update(str(attr) for attr in attrs)

        # Backward-compatible fallback for sources that expose force but not
        # requested_attributes metadata.
        if getattr(src, "force", None) is not None:
            requested_attributes.add("force")

    return requested_attributes or None
