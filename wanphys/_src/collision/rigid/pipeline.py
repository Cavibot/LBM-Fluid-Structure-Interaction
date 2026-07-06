# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys-owned rigid collision pipeline.

This module ports Newton's rigid collision orchestration into WanPhys while
using WanPhys ``RigidModel`` and ``RigidState`` as the primary data source.
Newton is still reused for contact storage and imported narrow-phase kernels.
"""

from __future__ import annotations

from typing import Any, Literal

import warp as wp
import numpy as np

from newton import Contacts
from newton import Model as NewtonModel
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF
from newton._src.geometry.types import GeoType

from .broad_phase_bvh import BroadPhaseBVH
from .broad_phase_hash import BroadPhaseHash
from .broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from .broad_phase_sap import BroadPhaseSAP
from .kernels import ContactWriterData, compute_shape_aabbs, prepare_geom_data_kernel, write_contact
from .narrow_phase import NarrowPhase


_BROAD_PHASE_MODES = ("nxn", "sap", "explicit", "bvh", "hash")


def _normalize_broad_phase_mode(mode: str) -> str:
    mode_str = str(mode).lower()
    if mode_str not in _BROAD_PHASE_MODES:
        raise ValueError(f"Unsupported broad phase mode: {mode!r}")
    return mode_str


def _infer_broad_phase_mode_from_instance(
    broad_phase: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit,
) -> str:
    if isinstance(broad_phase, BroadPhaseAllPairs):
        return "nxn"
    if isinstance(broad_phase, BroadPhaseSAP):
        return "sap"
    if isinstance(broad_phase, BroadPhaseExplicit):
        return "explicit"
    raise TypeError(
        "broad_phase must be a BroadPhaseAllPairs, BroadPhaseSAP, or BroadPhaseExplicit instance "
        f"(got {type(broad_phase)!r})"
    )


class RigidCollisionPipeline:
    """Rigid-only collision pipeline"""

    def __init__(
        self,
        model: Any,
        *,
        reduce_contacts: bool = True,
        rigid_contact_max: int | None = None,
        max_triangle_pairs: int = 1000000,
        max_heightfield_cell_pairs: int = 1000000,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        requires_grad: bool | None = None,
        broad_phase: Literal["nxn", "sap", "explicit", "bvh", "hash"]
        | BroadPhaseAllPairs
        | BroadPhaseSAP
        | BroadPhaseExplicit
        | None = None,
        narrow_phase: NarrowPhase | None = None,
        sdf_hydroelastic_config: HydroelasticSDF.Config | None = None,
    ):
        mode_from_broad_phase: str | None = None
        broad_phase_instance: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit | None = None
        if broad_phase is not None:
            if isinstance(broad_phase, str):
                mode_from_broad_phase = _normalize_broad_phase_mode(broad_phase)
            else:
                broad_phase_instance = broad_phase

        shape_count = model.shape_count
        device = model.device

        if rigid_contact_max is None:
            model_rigid_contact_max = int(getattr(model, "rigid_contact_max", 0) or 0)
            rigid_contact_max = (
                model_rigid_contact_max if model_rigid_contact_max > 0 else _estimate_rigid_contact_max(model)
            )
        if max_triangle_pairs <= 0:
            raise ValueError("max_triangle_pairs must be > 0")
        if max_heightfield_cell_pairs <= 0:
            raise ValueError("max_heightfield_cell_pairs must be > 0")
        if requires_grad is None:
            requires_grad = getattr(model, "requires_grad", False)

        self._rigid_contact_max = rigid_contact_max
        model.rigid_contact_max = rigid_contact_max

        shape_world = getattr(model, "shape_world", None)
        shape_flags = getattr(model, "shape_flags", None)

        with wp.ScopedDevice(device):
            shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

        self.model = model
        self.shape_count = shape_count
        self.device = device
        self.reduce_contacts = reduce_contacts
        self.requires_grad = requires_grad

        using_expert_components = broad_phase_instance is not None or narrow_phase is not None
        if using_expert_components:
            if broad_phase_instance is None or narrow_phase is None:
                raise ValueError("Provide both broad_phase and narrow_phase for expert component construction")
            if sdf_hydroelastic_config is not None:
                raise ValueError("sdf_hydroelastic_config cannot be used when narrow_phase is provided")

            inferred_mode = _infer_broad_phase_mode_from_instance(broad_phase_instance)
            self.broad_phase_mode = inferred_mode
            self.broad_phase = broad_phase_instance

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError(
                        "shape_pairs_filtered must be provided for explicit broad phase "
                        "(or set model.shape_contact_pairs)"
                    )
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
            else:
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )

            if narrow_phase.max_candidate_pairs < self.shape_pairs_max:
                raise ValueError(
                    "Provided narrow_phase.max_candidate_pairs is too small for this model and broad phase mode "
                    f"(required at least {self.shape_pairs_max}, got {narrow_phase.max_candidate_pairs})"
                )
            self.narrow_phase = narrow_phase
            self.hydroelastic_sdf = self.narrow_phase.hydroelastic_sdf
        else:
            self.broad_phase_mode = mode_from_broad_phase if mode_from_broad_phase is not None else "explicit"

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError(
                        "shape_pairs_filtered must be provided for broad_phase=EXPLICIT "
                        "(or set model.shape_contact_pairs)"
                    )
                self.broad_phase = BroadPhaseExplicit()
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
            elif self.broad_phase_mode == "nxn":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=NXN")
                self.broad_phase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "sap":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=SAP")
                self.broad_phase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "bvh":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=BVH")
                self.broad_phase = BroadPhaseBVH(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "hash":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=HASH")
                self.broad_phase = BroadPhaseHash(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            else:
                raise ValueError(f"Unsupported broad phase mode: {self.broad_phase_mode}")

            hydroelastic_sdf = HydroelasticSDF._from_model(
                model,
                config=sdf_hydroelastic_config,
                writer_func=write_contact,
            )

            has_meshes = False
            has_heightfields = False
            if hasattr(model, "shape_type") and model.shape_type is not None:
                shape_types = model.shape_type.numpy()
                has_heightfields = bool((shape_types == int(GeoType.HFIELD)).any())
                has_meshes = bool((shape_types == int(GeoType.MESH)).any())

            self.narrow_phase = NarrowPhase(
                max_candidate_pairs=self.shape_pairs_max,
                max_triangle_pairs=max_triangle_pairs,
                max_heightfield_cell_pairs=max_heightfield_cell_pairs,
                reduce_contacts=self.reduce_contacts,
                device=device,
                shape_aabb_lower=shape_aabb_lower,
                shape_aabb_upper=shape_aabb_upper,
                contact_writer_warp_func=write_contact,
                shape_voxel_resolution=model._shape_voxel_resolution,
                hydroelastic_sdf=hydroelastic_sdf,
                has_meshes=has_meshes,
                has_heightfields=has_heightfields,
            )
            self.hydroelastic_sdf = self.narrow_phase.hydroelastic_sdf

        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)

        if getattr(self.narrow_phase, "shape_aabb_lower", None) is None or getattr(
            self.narrow_phase, "shape_aabb_upper", None
        ) is None:
            raise ValueError("narrow_phase must expose shape_aabb_lower and shape_aabb_upper arrays")
        if self.narrow_phase.shape_aabb_lower.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_lower must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_lower.shape[0]})"
            )
        if self.narrow_phase.shape_aabb_upper.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_upper must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_upper.shape[0]})"
            )

    @property
    def rigid_contact_max(self) -> int:
        """Maximum rigid contact capacity used by this pipeline."""

        return self._rigid_contact_max

    @staticmethod
    def _build_excluded_pairs(model) -> wp.array(dtype=wp.vec2i) | None:
        filters = getattr(model, "shape_collision_filter_pairs", None)
        if not filters:
            return None
        sorted_pairs = sorted(filters)
        return wp.array(
            np.array(sorted_pairs),
            dtype=wp.vec2i,
            device=model.device,
        )

    def contacts(self) -> Contacts:
        """Allocate a Newton-compatible contact buffer for rigid contacts."""

        contacts = Contacts(
            self.rigid_contact_max,
            0,
            requires_grad=self.requires_grad,
            device=self.model.device,
            per_contact_shape_properties=self.narrow_phase.hydroelastic_sdf is not None,
            requested_attributes=(
                self.model.get_requested_contact_attributes() if hasattr(self.model, "get_requested_contact_attributes") else None
            ),
        )
        if hasattr(self.model, "_add_custom_attributes"):
            self.model._add_custom_attributes(
                contacts,
                NewtonModel.AttributeAssignment.CONTACT,
                requires_grad=self.requires_grad,
            )
        return contacts

    def collide(self, state: Any, contacts: Contacts | None = None) -> Contacts:
        """Populate rigid contacts for the provided rigid state."""

        if contacts is None:
            contacts = self.contacts()

        contacts.clear()
        self.broad_phase_pair_count.zero_()

        model = self.model

        if not self.requires_grad:
            wp.launch(
                kernel=compute_shape_aabbs,
                dim=model.shape_count,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_collision_radius,
                    model.shape_source_ptr,
                    model.shape_margin,
                    model.shape_gap,
                ],
                outputs=[
                    self.narrow_phase.shape_aabb_lower,
                    self.narrow_phase.shape_aabb_upper,
                ],
                device=self.device,
            )

            if isinstance(
                self.broad_phase,
                BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseBVH | BroadPhaseHash,
            ):
                self.broad_phase.launch(
                    self.narrow_phase.shape_aabb_lower,
                    self.narrow_phase.shape_aabb_upper,
                    None,
                    model.shape_collision_group,
                    model.shape_world,
                    model.shape_count,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self.device,
                    filter_pairs=self.shape_pairs_excluded,
                    num_filter_pairs=self.shape_pairs_excluded_count,
                )
            elif isinstance(self.broad_phase, BroadPhaseExplicit):
                self.broad_phase.launch(
                    self.narrow_phase.shape_aabb_lower,
                    self.narrow_phase.shape_aabb_upper,
                    None,
                    self.shape_pairs_filtered,
                    len(self.shape_pairs_filtered),
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self.device,
                )
            else:
                raise TypeError(f"Unsupported broad phase type: {type(self.broad_phase)!r}")

            wp.launch(
                kernel=prepare_geom_data_kernel,
                dim=model.shape_count,
                inputs=[
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_margin,
                    state.body_q,
                ],
                outputs=[
                    self.geom_data,
                    self.geom_transform,
                ],
                device=self.device,
            )

            writer_data = ContactWriterData()
            writer_data.contact_max = contacts.rigid_contact_max
            writer_data.body_q = state.body_q
            writer_data.shape_body = model.shape_body
            writer_data.shape_gap = model.shape_gap
            writer_data.contact_count = contacts.rigid_contact_count
            writer_data.out_shape0 = contacts.rigid_contact_shape0
            writer_data.out_shape1 = contacts.rigid_contact_shape1
            writer_data.out_point0 = contacts.rigid_contact_point0
            writer_data.out_point1 = contacts.rigid_contact_point1
            writer_data.out_offset0 = contacts.rigid_contact_offset0
            writer_data.out_offset1 = contacts.rigid_contact_offset1
            writer_data.out_normal = contacts.rigid_contact_normal
            writer_data.out_margin0 = contacts.rigid_contact_margin0
            writer_data.out_margin1 = contacts.rigid_contact_margin1
            writer_data.out_tids = contacts.rigid_contact_tids
            writer_data.out_stiffness = contacts.rigid_contact_stiffness
            writer_data.out_damping = contacts.rigid_contact_damping
            writer_data.out_friction = contacts.rigid_contact_friction

            self.narrow_phase.launch_custom_write(
                candidate_pair=self.broad_phase_shape_pairs,
                candidate_pair_count=self.broad_phase_pair_count,
                shape_types=model.shape_type,
                shape_data=self.geom_data,
                shape_transform=self.geom_transform,
                shape_source=model.shape_source_ptr,
                sdf_data=model.sdf_data,
                shape_sdf_index=model.shape_sdf_index,
                shape_gap=model.shape_gap,
                shape_collision_radius=model.shape_collision_radius,
                shape_flags=model.shape_flags,
                shape_collision_aabb_lower=model.shape_collision_aabb_lower,
                shape_collision_aabb_upper=model.shape_collision_aabb_upper,
                shape_voxel_resolution=self.narrow_phase.shape_voxel_resolution,
                shape_heightfield_data=model.shape_heightfield_data,
                heightfield_elevation_data=model.heightfield_elevation_data,
                writer_data=writer_data,
                device=self.device,
            )

        return contacts


def _estimate_rigid_contact_max(model) -> int:
    """Estimate rigid contact buffer capacity from model shape metadata."""

    if not hasattr(model, "shape_type") or model.shape_type is None:
        return 1000

    shape_types = model.shape_type.numpy()

    primitive_cpp = 5
    mesh_cpp = 40
    max_neighbors_per_shape = 20

    mesh_mask = (shape_types == int(GeoType.MESH)) | (shape_types == int(GeoType.HFIELD))
    plane_mask = shape_types == int(GeoType.PLANE)
    non_plane_mask = ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    non_plane_contacts = (
        num_primitives * max_neighbors_per_shape * primitive_cpp
        + num_meshes * max_neighbors_per_shape * mesh_cpp
    ) // 2

    avg_cpp = (
        (num_primitives * primitive_cpp + num_meshes * mesh_cpp) // max(num_non_planes, 1) if num_non_planes > 0 else 0
    )

    plane_contacts = 0
    if num_planes > 0 and num_non_planes > 0:
        has_world_info = (
            hasattr(model, "shape_world")
            and model.shape_world is not None
            and hasattr(model, "world_count")
            and model.world_count > 0
        )
        shape_world = model.shape_world.numpy() if has_world_info else None

        if shape_world is not None and len(shape_world) == len(shape_types):
            n_worlds = model.world_count

            global_mask = shape_world == -1
            local_mask = ~global_mask

            global_planes = int(np.count_nonzero(global_mask & plane_mask))
            global_non_planes = int(np.count_nonzero(global_mask & non_plane_mask))

            local_plane_counts = np.bincount(shape_world[local_mask & plane_mask], minlength=n_worlds)[:n_worlds]
            local_non_plane_counts = np.bincount(shape_world[local_mask & non_plane_mask], minlength=n_worlds)[:n_worlds]

            per_world_planes = local_plane_counts + global_planes
            per_world_non_planes = local_non_plane_counts + global_non_planes

            plane_pair_count = int(np.sum(per_world_planes * per_world_non_planes))
            if n_worlds > 1:
                plane_pair_count -= (n_worlds - 1) * global_planes * global_non_planes
            plane_contacts = plane_pair_count * avg_cpp
        else:
            plane_contacts = num_planes * (num_primitives * primitive_cpp + num_meshes * mesh_cpp)

    total_contacts = non_plane_contacts + plane_contacts

    if hasattr(model, "shape_contact_pair_count") and model.shape_contact_pair_count > 0:
        weighted_cpp = max(avg_cpp, primitive_cpp)
        pair_contacts = int(model.shape_contact_pair_count) * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)

    return max(1000, total_contacts)
