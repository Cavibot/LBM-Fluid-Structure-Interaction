# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from wanphys.core import DomainModel, DomainState
from wanphys.rigid import RigidDomain, RigidModel
from newton._src.geometry.flags import ShapeFlags as NewtonShapeFlags
from ..utils.selection import match_labels


@wp.kernel
def compute_shape_transforms_kernel(
    shapes: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    # output
    world_transform: wp.array(dtype=wp.transform),
):
    """Compute world transforms for a list of shape indices.

    Args:
        shape_indices: Array of shape indices
        shape_body: Model's shape_body array (body parent of each shape)
        shape_transform: Model's shape_transform array (local transforms)
        body_q: State's body_q array (body world transforms)
        world_transforms: Output array for computed world transforms
    """

    tid = wp.tid()
    shape_idx = shapes[tid]

    body_idx = shape_body[shape_idx]
    if body_idx >= 0:
        # Shape attached to a body
        # body to world
        X_wb = body_q[body_idx]
        # shape to body
        X_bs = shape_transform[shape_idx]
        world_transform[shape_idx] = wp.transform_multiply(X_wb, X_bs)
    else :
        # Static shape in world frame
        world_transform[shape_idx] = shape_transform[shape_idx]


@wp.kernel
def compute_relative_transforms_kernel(
    all_shape_transform: wp.array(dtype=wp.transform),
    target_shapes: wp.array(dtype=int),
    reference_sites: wp.array(dtype=int),
    # output
    relative_transforms: wp.array(dtype=wp.transform),
):
    """Compute relative transforms expressing object poses in reference frame coordinates.

    Args:
        all_shape_transforms: Array of world transforms for all shapes (indexed by shape index)
        target_shape_indices: Indices of target shapes
        reference_indices: Indices of reference sites
        relative_transforms: Output array of relative transforms

    Computes X_ro = X_wr^{-1} * X_wo for each pair, where:
    - X_wo is the world transform of the object shape (object to world)
    - X_wr is the world transform of the reference site (reference to world)
    - X_ro is the transform from object to reference (expresses object pose in reference frame)
    """
    tid = wp.tid()
    shape_idx = target_shapes[tid]
    ref_idx = reference_sites[tid]

    X_ws = all_shape_transform[shape_idx]
    X_wr = all_shape_transform[ref_idx]

    # Compute relative transform: express object pose in reference frame coordinates
    relative_transforms[tid] = wp.transform_multiply(wp.transform_inverse(X_wr), X_ws)


class SensorFrameTransform:
    """Sensor that measures transforms of shapes/sites relative to reference sites.

    This sensor computes the relative transform from each reference site to each
    target shape: ``X_ro = inverse(X_wr) * X_wo``, where *X_wo* is the world
    transform of the target, *X_wr* is the world transform of the reference site,
    and *X_ro* expresses the target's pose in the reference frame.

    **Objects** (``shapes``) can be any shape index, including both regular shapes
    and sites. **Reference frames** (``reference_sites``) must be sites (validated
    at initialization). A single reference site broadcasts to all shapes;
    otherwise the counts must match 1:1.

    Attributes:
        transforms: Relative transforms [m, unitless quaternion], shape
            ``(N,)`` (updated after each call to :meth:`update`).

    The ``shapes`` and ``reference_sites`` parameters accept label patterns -- see :ref:`label-matching`.

    Example:
        Measure a shape's pose relative to a reference site:

        .. testcode::

            from wanphys.rigid import RigidDomain, RigidModelBuilder
            from wanphys.sensors import SensorFrameTransform

            builder = RigidModelBuilder()
            # Add a target shape labeled "box" and a reference site labeled "ref".
            model = builder.finalize()
            domain = RigidDomain(model)
            domain.create_state()

            sensor = SensorFrameTransform(domain, shapes="box", reference_sites="ref")
            sensor.update()
            transforms = sensor.transforms.numpy()  # shape: (N, 7)
    """

    def __init__(
        self,
        domain: RigidDomain,
        shapes: str | list[str] | list[int],
        reference_sites: str | list[str] | list[int],
        *,
        verbose: bool | None = None,
    ):
        """Initialize the SensorFrameTransform.

        Args:
            domain: The rigid domain to measure.
            shapes: List of shape indices to measure.
            reference_sites: List of reference site indices (shapes with SITE flag).
                Must match 1:1 with shape_indices, or be a single site for all shapes.
            verbose: If True, print details. If None, uses ``wp.config.verbose``.

        Raises:
            ValueError: If arguments are invalid.
        """
        self.domain = domain
        model: RigidModel = domain.model
        self.model = model
        self.verbose = verbose if verbose is not None else wp.config.verbose

        # resolve label patterns to indices
        original_shapes = shapes
        target_shapes: list[int] = match_labels(model.shape_label, shapes)
        original_reference_sites = reference_sites
        reference_site_indices: list[int] = match_labels(model.shape_label, reference_sites)

        # Validate shape indices
        if not target_shapes:
            if isinstance(original_shapes, list) and len(original_shapes) == 0:
                raise ValueError("'shapes' must not be empty")
            raise ValueError(f"No shapes matched the given pattern {original_shapes!r}")
        if any(idx < 0 or idx >= model.shape_count for idx in reference_site_indices):
            raise ValueError(f"Invalid reference site indices. Must be in range [0, {model.shape_count}]")
        
        #verify that reference indices are actually sites
        shape_flags: np.ndarray = model.shape_flags.numpy()
        for idx in reference_site_indices:
            if not (shape_flags[idx] & NewtonShapeFlags.SITE):
                raise ValueError(f"Reference index {idx} (label: {model.shape_label[idx]}) is not a site")
            
        # Handle reference site match
        if len(reference_site_indices) == 1:
            # Single reference site for all shapes
            reference_sites_matched: list[int] = reference_site_indices * len(target_shapes)
        elif len(reference_site_indices) == len(target_shapes):
            reference_sites_matched: list[int] = list(reference_site_indices)
        else :
            raise ValueError(
                f"Number of refernce sites ({len(reference_site_indices)}) must match "
                f"number of shapes ({len(target_shapes)}) or be 1"
            )
        
        # Build list of unique shape indices that need transforms computed
        all_indices: set[int] = set(target_shapes) | set(reference_sites_matched)
        self._unique_shape_indices = sorted(all_indices)

        # Allocate transform array for all shapes (indexed by shape index)
        # Only the shapes we care about will be computed, rest stay zero
        self._all_shape_transforms = wp.zeros(
            model.shape_count,
            dtype=wp.transform,
            device=model.device,
        )

        # Allocate output array
        self.transforms = wp.zeros(
            len(target_shapes),
            dtype=wp.transform,
            device=model.device,
        )

        # Convert indices to warp arrays (done once at init)
        self._unique_indices_arr: wp.array = wp.array(self._unique_shape_indices, dtype=int, device=model.device)
        self._shape_indices_arr: wp.array = wp.array(target_shapes, dtype=int, device=model.device)
        self._reference_indices_arr: wp.array = wp.array(reference_sites_matched, dtype=int, device=model.device)

        if self.verbose:
            print("SensorFrameTransform initialized:")
            print(f"  Target Shapes: {len(target_shapes)}")
            print(f"  Reference sites: {len(set(reference_sites_matched))} unique")
            print(
                f"  Unique shapes to compute: {len(self._unique_shape_indices)} (optimized from {len(target_shapes) + len(reference_sites_matched)})"
            )

    def update(self):
        """Update sensor measurements based on current state.

        This should be called after eval_fk to compute transforms.

        Args:
            model: The model (must match the one used in __init__)
            state: The current state with body_q populated by eval_fk
        """
        state = self.domain.state
        # Compute world transforms for all unique shapes directly into the all_shape_transforms array
        wp.launch(
            compute_shape_transforms_kernel,
            dim=len(self._unique_shape_indices),
            inputs=[self._unique_indices_arr, self.model.shape_body, self.model.shape_transform, state.body_q],
            outputs=[self._all_shape_transforms],
            device=self.model.device,
        )

        # Compute relative transforms by indexing directly into all_shape_transforms
        wp.launch(
            compute_relative_transforms_kernel,
            dim=len(self._shape_indices_arr),
            inputs=[self._all_shape_transforms, self._shape_indices_arr, self._reference_indices_arr],
            outputs=[self.transforms],
            device=self.model.device,
        )
        
