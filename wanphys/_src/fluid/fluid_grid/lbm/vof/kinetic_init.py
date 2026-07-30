# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P5 host dispatch and validation for new-interface FullF initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..constants import CX, CY, CZ, W
from .kinetic_init_kernels import (
    initialize_new_interface_fullf_kernel,
    initialize_new_interface_home_kernel,
    prepare_new_interface_donors_kernel,
)

if TYPE_CHECKING:
    from ..state import FullFLbmState, HomeLbmState, LbmStateBase
    from .transition import VofTransitionResult


@dataclass(frozen=True)
class VofKineticInitializationResult:
    """Solver-owned P5 donor averages, valid until the next prepare call."""

    donor_count: wp.array
    rho_init: wp.array
    ux_init: wp.array
    uy_init: wp.array
    uz_init: wp.array


def _equilibrium_numpy(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> np.ndarray:
    values = np.empty((19, rho.size), dtype=np.float64)
    flat_rho = rho.reshape(-1).astype(np.float64)
    flat_ux = ux.reshape(-1).astype(np.float64)
    flat_uy = uy.reshape(-1).astype(np.float64)
    flat_uz = uz.reshape(-1).astype(np.float64)
    u2 = flat_ux * flat_ux + flat_uy * flat_uy + flat_uz * flat_uz
    for q in range(19):
        cu = CX[q] * flat_ux + CY[q] * flat_uy + CZ[q] * flat_uz
        values[q] = W[q] * flat_rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
        )
    return values


def validate_p5_prepared_kinetic(
    new_interface: np.ndarray,
    result: VofKineticInitializationResult,
    *,
    max_lattice_speed: float,
    enforce_population_positivity: bool,
    population_floor: float,
    atol: float = 2.0e-6,
) -> None:
    """Reject missing donors or inadmissible equilibrium before kinetic writes."""

    mask = np.asarray(new_interface, dtype=bool)
    count = np.asarray(result.donor_count.numpy())
    rho = np.asarray(result.rho_init.numpy())
    ux = np.asarray(result.ux_init.numpy())
    uy = np.asarray(result.uy_init.numpy())
    uz = np.asarray(result.uz_init.numpy())
    if np.any(count[mask] == 0):
        first = tuple(int(value) for value in np.argwhere(mask & (count == 0))[0])
        raise ValueError(
            "P5 new INTERFACE has no old-active/final-active donor; "
            f"first cell={first}"
        )
    if not np.all(np.isfinite(rho[mask])) or np.any(rho[mask] <= 0.0):
        raise ValueError("P5 new-interface density must be finite and positive")
    for name, values in (("ux", ux), ("uy", uy), ("uz", uz)):
        if not np.all(np.isfinite(values[mask])):
            raise ValueError(f"P5 new-interface {name} must be finite")
    speed = np.sqrt(ux[mask] ** 2 + uy[mask] ** 2 + uz[mask] ** 2)
    if max_lattice_speed > 0.0 and np.any(
        speed > max_lattice_speed + atol
    ):
        raise ValueError("P5 donor-average velocity exceeds max_lattice_speed")

    populations = _equilibrium_numpy(
        rho[mask],
        ux[mask],
        uy[mask],
        uz[mask],
    )
    if not np.all(np.isfinite(populations)):
        raise ValueError("P5 equilibrium populations must be finite")
    if enforce_population_positivity and np.any(
        populations < population_floor - atol
    ):
        raise ValueError(
            "P5 equilibrium populations violate population_floor"
        )

    outside = ~mask
    if (
        np.any(count[outside] != 0)
        or np.any(rho[outside] != 0.0)
        or np.any(ux[outside] != 0.0)
        or np.any(uy[outside] != 0.0)
        or np.any(uz[outside] != 0.0)
    ):
        raise ValueError("P5 prepare scratch must be zero outside new interfaces")


def validate_p5_initialized_state(
    state: LbmStateBase,
    transition: VofTransitionResult,
    result: VofKineticInitializationResult,
    *,
    gravity: tuple[float, float, float],
    atol: float = 3.0e-6,
    rtol: float = 3.0e-6,
) -> None:
    """Validate new FullF moments, macros, force, and mass/phi closure."""

    mask = np.asarray(transition.new_interface.numpy(), dtype=bool)
    if not np.any(mask):
        return
    from ..state import FullFLbmState, HomeLbmState

    shape = tuple(int(value) for value in state.res)
    rho = np.asarray(state.density.numpy())
    ux = np.asarray(state.velocity_x.numpy())
    uy = np.asarray(state.velocity_y.numpy())
    uz = np.asarray(state.velocity_z.numpy())
    expected_rho = np.asarray(result.rho_init.numpy())
    expected_ux = np.asarray(result.ux_init.numpy())
    expected_uy = np.asarray(result.uy_init.numpy())
    expected_uz = np.asarray(result.uz_init.numpy())

    if isinstance(state, FullFLbmState):
        f_post = np.asarray(state.f_post.numpy()).reshape((19, *shape))
        expected_f = _equilibrium_numpy(
            expected_rho[mask],
            expected_ux[mask],
            expected_uy[mask],
            expected_uz[mask],
        )
        if not np.allclose(
            f_post[:, mask],
            expected_f,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "P5 initialized populations do not match equilibrium projection"
            )
    elif isinstance(state, HomeLbmState):
        expected_fields = (
            expected_rho,
            expected_rho * expected_ux,
            expected_rho * expected_uy,
            expected_rho * expected_uz,
            expected_rho * expected_ux * expected_ux,
            expected_rho * expected_uy * expected_uy,
            expected_rho * expected_uz * expected_uz,
            expected_rho * expected_ux * expected_uy,
            expected_rho * expected_ux * expected_uz,
            expected_rho * expected_uy * expected_uz,
        )
        for actual_field, expected_field in zip(
            state.kinetic_fields, expected_fields, strict=True
        ):
            if not np.allclose(
                np.asarray(actual_field.numpy())[mask],
                expected_field[mask],
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "P7 initialized HOME moments do not match equilibrium"
                )
    else:
        raise TypeError("P5/P7 validator requires FullF or HOME state")
    for actual, expected, name in (
        (rho, expected_rho, "density"),
        (ux, expected_ux, "velocity_x"),
        (uy, expected_uy, "velocity_y"),
        (uz, expected_uz, "velocity_z"),
    ):
        if not np.allclose(
            actual[mask],
            expected[mask],
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(f"P5 initialized {name} does not match donor mean")

    for actual, component, name in (
        (np.asarray(state.force_x.numpy()), gravity[0], "force_x"),
        (np.asarray(state.force_y.numpy()), gravity[1], "force_y"),
        (np.asarray(state.force_z.numpy()), gravity[2], "force_z"),
    ):
        if not np.allclose(
            actual[mask],
            expected_rho[mask] * component,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(f"P5 initialized {name} is inconsistent")

    mass = np.asarray(transition.mass_final.numpy())
    phi = np.asarray(transition.phi_final.numpy())
    if not np.allclose(
        mass[mask],
        rho[mask] * phi[mask],
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError("P5 new-interface mass/phi/density closure failed")


class VofKineticInitializer:
    """Prepare and initialize P5 FullF GAS-to-INTERFACE cells."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)

        def floats() -> wp.array:
            return wp.zeros(self.shape, dtype=float, device=device)

        self._donor_count = wp.zeros(
            self.shape,
            dtype=wp.uint8,
            device=device,
        )
        self._rho_init = floats()
        self._ux_init = floats()
        self._uy_init = floats()
        self._uz_init = floats()
        self.result = VofKineticInitializationResult(
            donor_count=self._donor_count,
            rho_init=self._rho_init,
            ux_init=self._ux_init,
            uy_init=self._uy_init,
            uz_init=self._uz_init,
        )

    def prepare(
        self,
        state_out: LbmStateBase,
        cell_type_n: wp.array,
        transition: VofTransitionResult,
        *,
        max_lattice_speed: float,
        enforce_population_positivity: bool,
        population_floor: float,
        validate: bool = True,
    ) -> VofKineticInitializationResult:
        """Compute and validate donor averages without changing kinetic state."""

        if tuple(int(value) for value in state_out.res) != self.shape:
            raise ValueError(
                f"P5 state shape {state_out.res} does not match {self.shape}"
            )
        for name, array in (
            ("cell_type_n", cell_type_n),
            ("final_type", transition.final_type),
            ("new_interface", transition.new_interface),
            ("mass_final", transition.mass_final),
            ("phi_final", transition.phi_final),
        ):
            if tuple(int(value) for value in array.shape) != self.shape:
                raise ValueError(
                    f"P5 {name} shape {array.shape} does not match {self.shape}"
                )
        new_interface_host = None
        if validate:
            old_type_host = np.asarray(cell_type_n.numpy())
            final_type_host = np.asarray(transition.final_type.numpy())
            new_interface_host = np.asarray(transition.new_interface.numpy())
            expected_new = (
                (old_type_host == 0) & (final_type_host == 1)
            ).astype(np.uint8)
            if not np.array_equal(new_interface_host, expected_new):
                raise ValueError(
                    "P5 new_interface mask must equal old GAS -> final INTERFACE"
                )
        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            prepare_new_interface_donors_kernel,
            dim=self.shape,
            inputs=[
                state_out.density,
                state_out.velocity_x,
                state_out.velocity_y,
                state_out.velocity_z,
                cell_type_n,
                transition.final_type,
                transition.new_interface,
                self._donor_count,
                self._rho_init,
                self._ux_init,
                self._uy_init,
                self._uz_init,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            ],
            device=self.device,
        )
        if validate:
            assert new_interface_host is not None
            validate_p5_prepared_kinetic(
                new_interface_host,
                self.result,
                max_lattice_speed=float(max_lattice_speed),
                enforce_population_positivity=bool(
                    enforce_population_positivity
                ),
                population_floor=float(population_floor),
            )
        return self.result

    def initialize_fullf(
        self,
        state_out: FullFLbmState,
        cell_type_n: wp.array,
        transition: VofTransitionResult,
        *,
        gravity: tuple[float, float, float],
        max_lattice_speed: float,
        enforce_population_positivity: bool,
        population_floor: float,
        validate: bool = True,
    ) -> VofKineticInitializationResult:
        """Prepare, write, and validate new-interface FullF state."""

        result = self.prepare(
            state_out,
            cell_type_n,
            transition,
            max_lattice_speed=max_lattice_speed,
            enforce_population_positivity=enforce_population_positivity,
            population_floor=population_floor,
            validate=validate,
        )
        gx, gy, gz = (float(value) for value in gravity)
        nx, ny, nz = self.shape
        wp.launch(
            initialize_new_interface_fullf_kernel,
            dim=self.shape,
            inputs=[
                transition.new_interface,
                transition.mass_final,
                transition.phi_final,
                self._rho_init,
                self._ux_init,
                self._uy_init,
                self._uz_init,
                state_out.f_post,
                state_out.density,
                state_out.velocity_x,
                state_out.velocity_y,
                state_out.velocity_z,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                gx,
                gy,
                gz,
                ny,
                nz,
                nx * ny * nz,
            ],
            device=self.device,
        )
        if validate:
            validate_p5_initialized_state(
                state_out,
                transition,
                result,
                gravity=(gx, gy, gz),
            )
        return result

    def initialize_home(
        self,
        state_out: HomeLbmState,
        cell_type_n: wp.array,
        transition: VofTransitionResult,
        *,
        gravity: tuple[float, float, float],
        max_lattice_speed: float,
        enforce_population_positivity: bool,
        population_floor: float,
        validate: bool = True,
    ) -> VofKineticInitializationResult:
        """Prepare, write, and validate new-interface HOME equilibrium moments."""

        result = self.prepare(
            state_out,
            cell_type_n,
            transition,
            max_lattice_speed=max_lattice_speed,
            enforce_population_positivity=enforce_population_positivity,
            population_floor=population_floor,
            validate=validate,
        )
        gx, gy, gz = (float(value) for value in gravity)
        wp.launch(
            initialize_new_interface_home_kernel,
            dim=self.shape,
            inputs=[
                transition.new_interface,
                transition.mass_final,
                transition.phi_final,
                self._rho_init,
                self._ux_init,
                self._uy_init,
                self._uz_init,
                *state_out.kinetic_fields,
                state_out.density,
                state_out.velocity_x,
                state_out.velocity_y,
                state_out.velocity_z,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                gx,
                gy,
                gz,
            ],
            device=self.device,
        )
        if validate:
            validate_p5_initialized_state(
                state_out,
                transition,
                result,
                gravity=(gx, gy, gz),
            )
        return result


__all__ = [
    "VofKineticInitializationResult",
    "VofKineticInitializer",
    "validate_p5_initialized_state",
    "validate_p5_prepared_kinetic",
]
