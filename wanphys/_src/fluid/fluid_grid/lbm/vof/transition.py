# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P4 host dispatch, scratch ownership, and transition validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..constants import CX, CY, CZ
from .contracts import VofCellType
from .initialization import validate_initial_topology
from .transition_kernels import (
    finalize_bounded_vof_kernel,
    prepare_vof_redistribution_kernel,
    propose_vof_type_kernel,
    resolve_vof_topology_kernel,
)

if TYPE_CHECKING:
    from .state import VofGridState


@dataclass(frozen=True)
class VofTransitionResult:
    """Solver-owned P4 arrays, valid until the next transition call."""

    proposed_type: wp.array
    final_type: wp.array
    mass_base: wp.array
    excess: wp.array
    share: wp.array
    receiver_count: wp.array
    unresolved_excess: wp.array
    mass_final: wp.array
    phi_final: wp.array
    new_interface: wp.array
    retired_active: wp.array
    changed: wp.array


def validate_p4_transition_result(
    mass_pre: np.ndarray,
    density_out: np.ndarray,
    cell_type_n: np.ndarray,
    result: VofTransitionResult,
    *,
    periodic: tuple[bool, bool, bool],
    mass_tolerance: float,
) -> None:
    """Validate P4 scratch, topology, masks, and global mass closure."""

    mass_pre = np.asarray(mass_pre)
    density_out = np.asarray(density_out)
    cell_type_n = np.asarray(cell_type_n)
    final_type = np.asarray(result.final_type.numpy())
    mass_final = np.asarray(result.mass_final.numpy())
    phi_final = np.asarray(result.phi_final.numpy())
    mass_base = np.asarray(result.mass_base.numpy())
    excess = np.asarray(result.excess.numpy())
    share = np.asarray(result.share.numpy())
    receiver_count = np.asarray(result.receiver_count.numpy())
    unresolved = np.asarray(result.unresolved_excess.numpy())

    if not np.all(np.isfinite(mass_pre)):
        raise ValueError("P4 provisional mass must be finite")
    if not np.all(np.isfinite(density_out)) or np.any(density_out <= 0.0):
        raise ValueError("P4 density_out must be finite and positive")
    legal = np.isin(cell_type_n, np.array([0, 1, 2], dtype=np.uint8))
    if not np.all(legal):
        raise ValueError("P4 input cell_type contains an unknown value")
    legal = np.isin(final_type, np.array([0, 1, 2], dtype=np.uint8))
    if not np.all(legal):
        raise ValueError("P4 final_type contains an unknown value")
    if not np.all(np.isfinite(mass_final)) or not np.all(np.isfinite(phi_final)):
        raise ValueError("P4 final mass and phi must be finite")
    if not np.all(np.isfinite(excess)) or not np.all(np.isfinite(share)):
        raise ValueError("P4 pending excess and share must be finite")
    if np.any(receiver_count > 18):
        raise ValueError("P4 receiver count exceeds D3Q19 degree")
    expected_unresolved = np.where(receiver_count == 0, excess, 0.0)
    if not np.allclose(
        unresolved,
        expected_unresolved,
        atol=mass_tolerance,
        rtol=mass_tolerance,
    ):
        raise ValueError("P4 retained zero-receiver excess is inconsistent")

    validate_initial_topology(final_type, periodic=periodic)
    gas = final_type == int(VofCellType.GAS)
    interface = final_type == int(VofCellType.INTERFACE)
    liquid = final_type == int(VofCellType.LIQUID)
    if np.any(phi_final < 0.0) or np.any(phi_final > 1.0):
        raise ValueError("P4 committed phi must lie in [0, 1]")
    if not np.array_equal(mass_final, mass_base):
        raise ValueError("P4 final resident mass must equal bounded mass_base")
    if np.any(mass_final[gas] != 0.0) or np.any(phi_final[gas] != 0.0):
        raise ValueError("P4 final GAS cells must have mass=0 and phi=0")
    became_liquid = (
        (cell_type_n == int(VofCellType.INTERFACE))
        & liquid
    )
    if not np.allclose(
        mass_final[became_liquid],
        density_out[became_liquid],
        atol=mass_tolerance,
        rtol=mass_tolerance,
    ) or not np.allclose(
        phi_final[liquid],
        1.0,
        atol=mass_tolerance,
        rtol=mass_tolerance,
    ):
        raise ValueError(
            "P4 new LIQUID cells must have mass=density and all LIQUID phi=1"
        )
    if not np.allclose(
        mass_final[interface],
        density_out[interface] * phi_final[interface],
        atol=mass_tolerance,
        rtol=mass_tolerance,
    ):
        raise ValueError("P4 final INTERFACE must satisfy mass=density*phi")
    expected_share = np.zeros_like(share)
    has_receiver = receiver_count > 0
    expected_share[has_receiver] = (
        excess[has_receiver]
        / receiver_count[has_receiver].astype(np.float32)
    )
    if not np.allclose(
        share,
        expected_share,
        atol=mass_tolerance,
        rtol=mass_tolerance,
    ):
        raise ValueError("P4 pending excess share/count closure failed")
    shape = tuple(int(value) for value in final_type.shape)
    for raw_cell in np.argwhere(np.abs(excess) > mass_tolerance):
        cell = tuple(int(value) for value in raw_cell)
        actual_count = 0
        for q in range(1, 19):
            neighbor = [
                cell[0] - CX[q],
                cell[1] - CY[q],
                cell[2] - CZ[q],
            ]
            outside = False
            for axis in range(3):
                if 0 <= neighbor[axis] < shape[axis]:
                    continue
                if not periodic[axis]:
                    outside = True
                    break
                neighbor[axis] %= shape[axis]
            if not outside and final_type[tuple(neighbor)] == int(
                VofCellType.INTERFACE
            ):
                actual_count += 1
        if actual_count != int(receiver_count[cell]):
            raise ValueError(
                "P4 receiver count does not match final topology"
            )

    expected_new = (
        (cell_type_n == int(VofCellType.GAS))
        & (final_type == int(VofCellType.INTERFACE))
    ).astype(np.uint8)
    expected_retired = (
        (cell_type_n != int(VofCellType.GAS))
        & (final_type == int(VofCellType.GAS))
    ).astype(np.uint8)
    expected_changed = (cell_type_n != final_type).astype(np.uint8)
    for name, expected in (
        ("new_interface", expected_new),
        ("retired_active", expected_retired),
        ("changed", expected_changed),
    ):
        actual = np.asarray(getattr(result, name).numpy())
        if not np.array_equal(actual, expected):
            raise ValueError(f"P4 {name} mask does not match old/final types")

    scale = max(1.0, float(np.sqrt(mass_pre.size)))
    if not np.isclose(
        np.sum(mass_final, dtype=np.float64)
        + np.sum(excess, dtype=np.float64),
        np.sum(mass_pre, dtype=np.float64),
        atol=mass_tolerance * scale,
        rtol=mass_tolerance,
    ):
        raise ValueError("P4 resident plus pending mass is not conservative")


class VofTopologyTransition:
    """Resolve P4 type changes and redistribute mass without state mutation."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
        epsilon: float,
        *,
        mass_tolerance: float = 2.0e-6,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)
        self.epsilon = float(epsilon)
        self.mass_tolerance = float(mass_tolerance)

        def floats() -> wp.array:
            return wp.zeros(self.shape, dtype=float, device=device)

        def bytes_() -> wp.array:
            return wp.zeros(self.shape, dtype=wp.uint8, device=device)

        self._proposed_type = bytes_()
        self._final_type = bytes_()
        self._mass_base = floats()
        self._excess = floats()
        self._share = floats()
        self._receiver_count = bytes_()
        self._unresolved_excess = floats()
        self._mass_final = floats()
        self._phi_final = floats()
        self._new_interface = bytes_()
        self._retired_active = bytes_()
        self._changed = bytes_()
        self.result = VofTransitionResult(
            proposed_type=self._proposed_type,
            final_type=self._final_type,
            mass_base=self._mass_base,
            excess=self._excess,
            share=self._share,
            receiver_count=self._receiver_count,
            unresolved_excess=self._unresolved_excess,
            mass_final=self._mass_final,
            phi_final=self._phi_final,
            new_interface=self._new_interface,
            retired_active=self._retired_active,
            changed=self._changed,
        )

    def compute(
        self,
        mass_pre: wp.array,
        density_out: wp.array,
        cell_type_n: wp.array,
        *,
        validate: bool = True,
    ) -> VofTransitionResult:
        """Return deterministic P4 transition scratch without mutating inputs."""

        expected_shape = self.shape
        for name, array in (
            ("mass_pre", mass_pre),
            ("density_out", density_out),
            ("cell_type_n", cell_type_n),
        ):
            if tuple(int(value) for value in array.shape) != expected_shape:
                raise ValueError(
                    f"P4 {name} shape {array.shape} does not match {expected_shape}"
                )

        periodic = tuple(bool(value) for value in self.periodic)
        if validate:
            input_mass = np.asarray(mass_pre.numpy())
            input_density = np.asarray(density_out.numpy())
            input_type = np.asarray(cell_type_n.numpy())
            if not np.all(np.isfinite(input_mass)):
                raise ValueError("P4 provisional mass must be finite")
            if not np.all(np.isfinite(input_density)) or np.any(
                input_density <= 0.0
            ):
                raise ValueError("P4 density_out must be finite and positive")
            legal = np.isin(
                input_type,
                np.array([0, 1, 2], dtype=np.uint8),
            )
            if not np.all(legal):
                raise ValueError("P4 input cell_type contains an unknown value")
            validate_initial_topology(input_type, periodic=periodic)

        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            propose_vof_type_kernel,
            dim=self.shape,
            inputs=[
                mass_pre,
                density_out,
                cell_type_n,
                self._proposed_type,
                self.epsilon,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        wp.launch(
            resolve_vof_topology_kernel,
            dim=self.shape,
            inputs=[
                self._proposed_type,
                cell_type_n,
                self._final_type,
                self._new_interface,
                self._retired_active,
                self._changed,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        wp.launch(
            prepare_vof_redistribution_kernel,
            dim=self.shape,
            inputs=[
                mass_pre,
                density_out,
                cell_type_n,
                self._final_type,
                self._mass_base,
                self._excess,
                self._share,
                self._receiver_count,
                self._unresolved_excess,
                self.mass_tolerance,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        wp.launch(
            finalize_bounded_vof_kernel,
            dim=self.shape,
            inputs=[
                density_out,
                self._final_type,
                self._mass_base,
                self._mass_final,
                self._phi_final,
            ],
            device=self.device,
        )

        if validate:
            validate_p4_transition_result(
                np.asarray(mass_pre.numpy()),
                np.asarray(density_out.numpy()),
                np.asarray(cell_type_n.numpy()),
                self.result,
                periodic=periodic,
                mass_tolerance=self.mass_tolerance,
            )
        return self.result

    def commit(self, target: VofGridState, *, source_epoch: int) -> None:
        """Copy the latest validated P4 result into a persistent VOF state."""

        if tuple(int(value) for value in target.shape) != self.shape:
            raise ValueError(
                f"P4 target shape {target.shape} does not match {self.shape}"
            )
        wp.copy(target.mass, self._mass_final)
        wp.copy(target.phi, self._phi_final)
        wp.copy(target.pending_excess, self._excess)
        wp.copy(target.pending_receiver_count, self._receiver_count)
        wp.copy(target.cell_type, self._final_type)
        target.epoch = int(source_epoch) + 1
        target.geometry_epoch = -1


__all__ = [
    "VofTopologyTransition",
    "VofTransitionResult",
    "validate_p4_transition_result",
]
