# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-pass periodic HOME-LBM solver."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from . import kernels
from .constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from .cut_link import CutLinkBuffer
from .diagnostics import HomeLbmDiagnostics
from .model import HomeLbmModel
from .state import HomeLbmState


class HomeLbmSolver:
    """Advance ten persistent moments without storing D3Q27 populations."""

    def __init__(self, model: HomeLbmModel) -> None:
        self.model = model
        self.nx = int(model.nx)
        self.ny = int(model.ny)
        self.nz = int(model.nz)
        self.stride = self.nx * self.ny * self.nz
        self.device = model._device
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device)
        self._opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=self.device)
        self._diag_invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct_liquid_gas_link_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_gas_density_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._gas_boundary_impulse = wp.zeros(
            3, dtype=wp.float64, device=self.device
        )
        self._cut_link_frame_correction = wp.zeros(
            3, dtype=wp.float64, device=self.device
        )
        self._missing_cut_link_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._unsupported_interpolation_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._wall_confined_interpolation_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._wall_confined_lubrication_closure = False
        self._pressure_reference_scale = 1.0
        self._pressure_reference_log_ratio = (0.0, 0.0, 0.0)
        self._diag_min_density = wp.zeros(1, dtype=float, device=self.device)
        self._diag_max_density = wp.zeros(1, dtype=float, device=self.device)
        self._diag_max_speed_squared = wp.zeros(1, dtype=float, device=self.device)
        self._diag_max_stress_squared = wp.zeros(1, dtype=float, device=self.device)
        self._diag_packed_counts = wp.zeros(6, dtype=wp.int32, device=self.device)
        self._diag_packed_values = wp.zeros(4, dtype=float, device=self.device)
        self._diagnostics_state: HomeLbmState | None = None
        self._diagnostics_free_surface_flags: wp.array | None = None
        self._diagnostics_gas_density_field: wp.array | None = None
        self._cut_links: CutLinkBuffer | None = None
        self._empty_cut_counts = wp.zeros(self.stride, dtype=wp.int32, device=self.device)
        self._empty_cut_offsets = wp.zeros(self.stride, dtype=wp.int32, device=self.device)
        self._empty_cut_directions = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._empty_cut_fraction = wp.zeros(1, dtype=float, device=self.device)
        self._empty_cut_wall_velocity = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._empty_cut_impulse = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._empty_free_surface_flags = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=wp.int32, device=self.device
        )
        self._empty_gas_density_field = wp.ones(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )
        self._empty_force_density = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=wp.vec3, device=self.device
        )

    @property
    def cut_links(self) -> CutLinkBuffer | None:
        return self._cut_links

    def set_cut_links(self, cut_links: CutLinkBuffer | None) -> None:
        if cut_links is not None:
            if cut_links.cell_count != self.stride:
                raise ValueError(
                    f"cut-link grid has {cut_links.cell_count} cells, expected {self.stride}"
                )
            if cut_links.device != self.device:
                raise ValueError("cut-link and HOME solver devices must match")
        self._cut_links = cut_links

    def enable_wall_confined_lubrication_closure(self) -> None:
        """Permit local reflection only where sphere-wall lubrication owns the gap."""

        self._wall_confined_lubrication_closure = True

    def initialize_uniform_lattice(
        self,
        state: HomeLbmState,
        rho: float = 1.0,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
        if rho <= 0.0:
            raise ValueError(f"rho must be positive, got {rho}")
        if speed > self.model.max_lattice_speed:
            raise ValueError(
                f"initial lattice speed {speed} exceeds configured limit {self.model.max_lattice_speed}"
            )
        wp.launch(
            kernels.initialize_uniform_kernel,
            dim=self.stride,
            inputs=[state.moments, float(rho), wp.vec3(*velocity), self.stride],
            device=self.device,
        )
        self._pressure_reference_scale = float(rho)
        self._pressure_reference_log_ratio = (0.0, 0.0, 0.0)
        self._diagnostics_state = None
        self._diagnostics_free_surface_flags = None
        self._diagnostics_gas_density_field = None

    def initialize_hydrostatic_lattice(
        self,
        state: HomeLbmState,
        mean_rho: float = 1.0,
        acceleration: tuple[float, float, float] | None = None,
    ) -> None:
        """Initialize the exact D3Q27 rest-state density recurrence.

        The persistent state is post-collision, so zero physical velocity has
        raw momentum ``rho*a/2``.  A nonzero body force on a periodic axis has
        no hydrostatic equilibrium and is rejected.
        """

        if not np.isfinite(mean_rho) or mean_rho <= 0.0:
            raise ValueError("mean_rho must be finite and positive")
        lattice_acceleration, log_ratios = self._hydrostatic_profile(acceleration)
        shape = np.asarray((self.nx, self.ny, self.nz), dtype=np.int64)
        mean_factors = np.ones(3, dtype=np.float64)
        for axis in range(3):
            coordinates = np.arange(shape[axis], dtype=np.float64) + 0.5 - 0.5 * shape[axis]
            mean_factors[axis] = float(
                np.mean(np.exp(log_ratios[axis] * coordinates))
            )

        density_scale = float(mean_rho / np.prod(mean_factors))
        self._launch_hydrostatic_initialization(
            state, lattice_acceleration, log_ratios, density_scale
        )

    def initialize_anchored_hydrostatic_lattice(
        self,
        state: HomeLbmState,
        reference_coordinate: tuple[float, float, float],
        reference_density: float,
        acceleration: tuple[float, float, float] | None = None,
    ) -> None:
        """Initialize hydrostatics anchored at a continuous lattice coordinate.

        Cell centers have coordinates ``(i+0.5, j+0.5, k+0.5)``. This lets a
        free-surface pressure be anchored at its effective half-link boundary
        instead of renormalizing a closed domain by mean density.
        """

        coordinate = np.asarray(reference_coordinate, dtype=np.float64)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ValueError("reference_coordinate must contain three finite values")
        if not np.isfinite(reference_density) or reference_density <= 0.0:
            raise ValueError("reference_density must be finite and positive")
        lattice_acceleration, log_ratios = self._hydrostatic_profile(acceleration)
        centered_coordinate = coordinate - 0.5 * np.asarray(
            (self.nx, self.ny, self.nz), dtype=np.float64
        )
        density_scale = float(
            reference_density / np.exp(np.dot(log_ratios, centered_coordinate))
        )
        self._launch_hydrostatic_initialization(
            state, lattice_acceleration, log_ratios, density_scale
        )

    def _hydrostatic_profile(
        self, acceleration: tuple[float, float, float] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        lattice_acceleration = np.asarray(
            self.model.lattice_acceleration if acceleration is None else acceleration,
            dtype=np.float64,
        )
        if lattice_acceleration.shape != (3,) or not np.isfinite(lattice_acceleration).all():
            raise ValueError("acceleration must contain three finite lattice components")
        ratios = np.ones(3, dtype=np.float64)
        for axis, value in enumerate(lattice_acceleration):
            if value != 0.0 and self.model.periodic[axis]:
                raise ValueError(
                    f"axis {axis} is periodic and cannot support hydrostatic acceleration"
                )
            numerator = 1.0 + 1.5 * value
            denominator = 1.0 - 1.5 * value
            if numerator <= 0.0 or denominator <= 0.0:
                raise ValueError(
                    f"lattice acceleration {value} on axis {axis} is too large for a positive hydrostatic state"
                )
            ratios[axis] = numerator / denominator
        return lattice_acceleration, np.log(ratios)

    def _launch_hydrostatic_initialization(
        self,
        state: HomeLbmState,
        lattice_acceleration: np.ndarray,
        log_ratios: np.ndarray,
        density_scale: float,
    ) -> None:
        wp.launch(
            kernels.initialize_hydrostatic_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state.moments,
                wp.vec3(*lattice_acceleration),
                wp.vec3(*log_ratios),
                density_scale,
                self.nx,
                self.ny,
                self.nz,
                self.stride,
            ],
            device=self.device,
        )
        self._pressure_reference_scale = density_scale
        self._pressure_reference_log_ratio = tuple(float(value) for value in log_ratios)
        self._diagnostics_state = None
        self._diagnostics_free_surface_flags = None
        self._diagnostics_gas_density_field = None

    def _reset_diagnostics(self) -> None:
        self._diag_invalid.zero_()
        self._diag_min_density.fill_(float("inf"))
        self._diag_max_density.fill_(float("-inf"))
        self._diag_max_speed_squared.zero_()
        self._diag_max_stress_squared.zero_()

    def _validate_free_surface_flags(self, flags: wp.array | None) -> None:
        if flags is None:
            return
        if tuple(flags.shape) != (self.nx, self.ny, self.nz):
            raise ValueError("free_surface_flags shape must match the HOME grid")
        if flags.device != self.device:
            raise ValueError("free_surface_flags and HOME solver devices must match")
        if flags.dtype != wp.int32:
            raise TypeError("free_surface_flags must use wp.int32")

    def _validate_gas_density_field(self, field: wp.array | None) -> None:
        if field is None:
            return
        if tuple(field.shape) != (self.nx, self.ny, self.nz):
            raise ValueError("gas_density_field shape must match the HOME grid")
        if field.device != self.device:
            raise ValueError("gas_density_field and HOME solver devices must match")
        if field.dtype != wp.float32:
            raise TypeError("gas_density_field must use wp.float32")

    def _validate_force_density(self, field: wp.array | None) -> None:
        if field is None:
            return
        if tuple(field.shape) != (self.nx, self.ny, self.nz):
            raise ValueError("force_density shape must match the HOME grid")
        if field.device != self.device:
            raise ValueError("force_density and HOME solver devices must match")
        if field.dtype != wp.vec3:
            raise TypeError("force_density must use wp.vec3")

    def step(
        self,
        state_in: HomeLbmState,
        state_out: HomeLbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
        *,
        free_surface_flags: wp.array | None = None,
        gas_density: float = 1.0,
        gas_density_field: wp.array | None = None,
        force_density: wp.array | None = None,
    ) -> None:
        del contacts, control
        self.model.validate_step(dt)
        self._validate_free_surface_flags(free_surface_flags)
        self._validate_gas_density_field(gas_density_field)
        self._validate_force_density(force_density)
        if gas_density_field is not None and free_surface_flags is None:
            raise ValueError("gas_density_field requires free_surface_flags")
        if free_surface_flags is not None:
            if not np.isfinite(gas_density) or gas_density <= 0.0:
                raise ValueError("gas_density must be finite and positive")
        cut_links = self._cut_links
        if cut_links is not None:
            cut_links.impulse.zero_()
        self._missing_cut_link_count.zero_()
        self._unsupported_interpolation_count.zero_()
        self._wall_confined_interpolation_count.zero_()
        self._direct_liquid_gas_link_count.zero_()
        self._invalid_gas_density_count.zero_()
        self._gas_boundary_impulse.zero_()
        self._cut_link_frame_correction.zero_()
        self._reset_diagnostics()
        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.solid_body_id, state_in.solid_body_id)
        wp.launch(
            kernels.stream_collide_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_in.moments,
                state_in.solid_phi,
                free_surface_flags
                if free_surface_flags is not None
                else self._empty_free_surface_flags,
                gas_density_field
                if gas_density_field is not None
                else self._empty_gas_density_field,
                force_density
                if force_density is not None
                else self._empty_force_density,
                self._directions,
                self._weights,
                self._opposites,
                cut_links.counts if cut_links is not None else self._empty_cut_counts,
                cut_links.offsets if cut_links is not None else self._empty_cut_offsets,
                cut_links.direction if cut_links is not None else self._empty_cut_directions,
                cut_links.fraction if cut_links is not None else self._empty_cut_fraction,
                cut_links.wall_velocity if cut_links is not None else self._empty_cut_wall_velocity,
                cut_links.impulse if cut_links is not None else self._empty_cut_impulse,
                self._missing_cut_link_count,
                self._unsupported_interpolation_count,
                self._wall_confined_interpolation_count,
                self._direct_liquid_gas_link_count,
                self._invalid_gas_density_count,
                self._gas_boundary_impulse,
                self._cut_link_frame_correction,
                self._diag_invalid,
                self._diag_min_density,
                self._diag_max_density,
                self._diag_max_speed_squared,
                self._diag_max_stress_squared,
                int(self._wall_confined_lubrication_closure),
                self._pressure_reference_scale,
                wp.vec3(*self._pressure_reference_log_ratio),
                int(cut_links is not None),
                int(free_surface_flags is not None),
                int(gas_density_field is not None),
                int(force_density is not None),
                float(gas_density),
                state_out.moments,
                float(self.model.shear_omega),
                wp.vec3(*self.model.lattice_acceleration),
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                self.nx,
                self.ny,
                self.nz,
                self.stride,
            ],
            device=self.device,
        )
        self._diagnostics_state = state_out
        self._diagnostics_free_surface_flags = free_surface_flags
        self._diagnostics_gas_density_field = gas_density_field

    def collect_diagnostics(
        self,
        state: HomeLbmState,
        *,
        free_surface_flags: wp.array | None = None,
        gas_density_field: wp.array | None = None,
        force_recompute: bool = False,
    ) -> HomeLbmDiagnostics:
        """Read per-step diagnostics, scanning only non-step or explicitly modified states."""

        self._validate_free_surface_flags(free_surface_flags)
        self._validate_gas_density_field(gas_density_field)

        if (
            force_recompute
            or state is not self._diagnostics_state
            or free_surface_flags is not self._diagnostics_free_surface_flags
            or gas_density_field is not self._diagnostics_gas_density_field
        ):
            self._reset_diagnostics()
            self._direct_liquid_gas_link_count.zero_()
            self._invalid_gas_density_count.zero_()
            wp.launch(
                kernels.diagnose_moments_kernel,
                dim=self.stride,
                inputs=[
                    state.moments,
                    state.solid_phi,
                    free_surface_flags
                    if free_surface_flags is not None
                    else self._empty_free_surface_flags,
                    gas_density_field
                    if gas_density_field is not None
                    else self._empty_gas_density_field,
                    int(free_surface_flags is not None),
                    int(gas_density_field is not None),
                    self._diag_invalid,
                    self._diag_min_density,
                    self._diag_max_density,
                    self._diag_max_speed_squared,
                    self._diag_max_stress_squared,
                    self._direct_liquid_gas_link_count,
                    self._invalid_gas_density_count,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    self.nx,
                    self.stride,
                    self.ny,
                    self.nz,
                ],
                device=self.device,
            )
            self._diagnostics_state = state
            self._diagnostics_free_surface_flags = free_surface_flags
            self._diagnostics_gas_density_field = gas_density_field
        wp.launch(
            kernels.pack_diagnostics_kernel,
            dim=1,
            inputs=[
                self._diag_invalid,
                self._direct_liquid_gas_link_count,
                self._invalid_gas_density_count,
                self._missing_cut_link_count,
                self._unsupported_interpolation_count,
                self._wall_confined_interpolation_count,
                self._diag_min_density,
                self._diag_max_density,
                self._diag_max_speed_squared,
                self._diag_max_stress_squared,
                self._diag_packed_counts,
                self._diag_packed_values,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._diag_packed_counts.numpy()
        values = self._diag_packed_values.numpy()
        gas_boundary_impulse = self._gas_boundary_impulse.numpy()
        cut_link_frame_correction = self._cut_link_frame_correction.numpy()
        return HomeLbmDiagnostics(
            invalid_cell_count=int(counts[0]),
            direct_liquid_gas_link_count=int(counts[1]),
            invalid_gas_density_count=int(counts[2]),
            missing_cut_link_count=int(counts[3]),
            unsupported_interpolation_count=int(counts[4]),
            wall_confined_interpolation_count=int(counts[5]),
            min_density=float(values[0]),
            max_density=float(values[1]),
            max_speed=float(np.sqrt(values[2])),
            max_nonequilibrium_stress=float(np.sqrt(values[3])),
            gas_boundary_impulse_lattice=tuple(
                float(value) for value in gas_boundary_impulse
            ),
            cut_link_frame_correction_lattice=tuple(
                float(value) for value in cut_link_frame_correction
            ),
        )

    def validate_state(
        self,
        state: HomeLbmState,
        *,
        free_surface_flags: wp.array | None = None,
        gas_density_field: wp.array | None = None,
    ) -> HomeLbmDiagnostics:
        diagnostics = self.collect_diagnostics(
            state,
            free_surface_flags=free_surface_flags,
            gas_density_field=gas_density_field,
        )
        if diagnostics.invalid_cell_count:
            raise FloatingPointError(
                f"HOME-LBM state contains {diagnostics.invalid_cell_count} invalid cells"
            )
        if diagnostics.direct_liquid_gas_link_count:
            raise RuntimeError(
                "HOME-FREE pressure boundary found "
                f"{diagnostics.direct_liquid_gas_link_count} direct liquid-gas links; "
                "interface topology is invalid"
            )
        if diagnostics.invalid_gas_density_count:
            raise FloatingPointError(
                "HOME-FREE pressure boundary found "
                f"{diagnostics.invalid_gas_density_count} interface cells with "
                "nonpositive or non-finite gas density"
            )
        if diagnostics.missing_cut_link_count:
            raise RuntimeError(
                f"HOME-LBM encountered {diagnostics.missing_cut_link_count} solid links "
                "without matching CutLinkBuffer entries"
            )
        if diagnostics.unsupported_interpolation_count:
            raise RuntimeError(
                "HOME-LBM encountered "
                f"{diagnostics.unsupported_interpolation_count} Bouzidi links with q < 0.5 "
                "but no second fluid support node"
            )
        if (
            diagnostics.wall_confined_interpolation_count
            and not self._wall_confined_lubrication_closure
        ):
            raise RuntimeError(
                "HOME-LBM encountered "
                f"{diagnostics.wall_confined_interpolation_count} wall-confined Bouzidi "
                "links with q < 0.5; enable the explicit sphere-wall lubrication closure"
            )
        if diagnostics.max_speed > self.model.max_lattice_speed:
            raise FloatingPointError(
                f"HOME-LBM maximum lattice speed {diagnostics.max_speed} exceeds "
                f"configured limit {self.model.max_lattice_speed}"
            )
        return diagnostics
