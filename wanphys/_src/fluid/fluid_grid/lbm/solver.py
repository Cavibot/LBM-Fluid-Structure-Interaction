# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 FullF/HOME stream-physics-collide-encode pipeline."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import warp as wp

from ..base import FluidGridSolverBase
from . import collisions, encoding, forcing, kernels, moments, streaming
from .boundaries import normalize_boundary_type, resolve_boundary_faces
from .constants import BC_OUTFLOW
from .contracts import CollisionContext, CollisionSpace, ForceModel, collision_contract
from .model import LbmModel
from .state import FullFLbmState, HomeLbmState, LbmStateBase


class LbmSolver(FluidGridSolverBase):
    """D3Q19 LBM solver with pluggable persistent encoding and collision.

    The solver owns temporary arrays for macroscopic moments that are
    reused across steps.  Distribution functions and visualisation fields
    live on :class:`LbmStateBase`.

    Parameters
    ----------
    model:
        Static LBM configuration (grid size, *τ*, body force, …).
    """

    def __init__(self, model: LbmModel) -> None:
        self.model: LbmModel = model
        self.nx: int = int(model.nx)
        self.ny: int = int(model.ny)
        self.nz: int = int(model.nz)
        self.device: wp.Device = model._device

        # Stride for flat-array indexing: nx * ny * nz
        self._stride: int = self.nx * self.ny * self.nz

        # SC force stride: recompute only every N steps, reuse cached force
        # between updates.  Configured by LbmModel.sc_force_stride.
        self._step_count: int = 0
        self._sc_stride: int = int(model.sc_force_stride)

        # ---- Boundary condition arrays (synced from model) ------------------
        self._bc_types = wp.zeros(6, dtype=wp.int32, device=self.device)
        self._bc_vel_x = wp.zeros(6, dtype=float, device=self.device)
        self._bc_vel_y = wp.zeros(6, dtype=float, device=self.device)
        self._bc_vel_z = wp.zeros(6, dtype=float, device=self.device)
        self._bc_density = wp.zeros(6, dtype=float, device=self.device)
        self._bc_convective_speed = wp.zeros(6, dtype=float, device=self.device)
        self._sync_bc_from_model()
        self._boundary_resolution = resolve_boundary_faces(
            tuple(int(value) for value in model.bc_types),
            (self.nx, self.ny, self.nz),
        )
        if self._boundary_resolution.conflict is not None:
            warnings.warn(
                self._boundary_resolution.conflict.warning_message,
                RuntimeWarning,
                stacklevel=2,
            )
        self._boundary_history = None
        if any(value == BC_OUTFLOW for value in model.bc_types):
            self._boundary_history = wp.zeros(
                19 * self._stride, dtype=float, device=self.device
            )
        self._boundary_history_ready = False

        # ---- Solver-owned temporary macroscopic fields --------------------
        shape = (self.nx, self.ny, self.nz)
        self._moments: tuple[wp.array3d, ...] = tuple(
            wp.zeros(shape, dtype=float, device=self.device) for _ in range(10)
        )
        self._rho, self._jx, self._jy, self._jz = self._moments[:4]
        self._ux: wp.array3d = wp.zeros(
            shape, dtype=float, device=self.device
        )
        self._uy: wp.array3d = wp.zeros(
            shape, dtype=float, device=self.device
        )
        self._uz: wp.array3d = wp.zeros(
            shape, dtype=float, device=self.device
        )

        # ---- Solver-owned temporary force arrays (Shan-Chen interaction) ---
        self._fx: wp.array3d = wp.zeros(
            shape, dtype=float, device=self.device
        )
        self._fy: wp.array3d = wp.zeros(
            shape, dtype=float, device=self.device
        )
        self._fz: wp.array3d = wp.zeros(
            shape, dtype=float, device=self.device
        )
        self._sc_fx = wp.zeros(shape, dtype=float, device=self.device)
        self._sc_fy = wp.zeros(shape, dtype=float, device=self.device)
        self._sc_fz = wp.zeros(shape, dtype=float, device=self.device)
        self._force_provider = forcing.ForceProvider(ForceModel(model.force_model))

        self._collision_name = model.resolved_collision
        self._collision_contract = collision_contract(model.encoding, self._collision_name)
        self._collision_space = self._collision_contract.space
        if self._collision_name == "srt":
            self._collision_backend = collisions.SrtCollision()
        elif self._collision_name == "trt":
            self._collision_backend = collisions.TrtCollision()
        elif self._collision_name == "raw_mrt":
            self._collision_backend = collisions.RawMrtCollision()
        elif model.encoding == "home":
            self._collision_backend = collisions.HomeNocmMrtCollision()
        else:
            self._collision_backend = collisions.FullNocmMrtCollision()

        # Boundary completion is population-based for every collision path.
        self._f_star = wp.zeros(19 * self._stride, dtype=float, device=self.device)
        self._collision_context = CollisionContext(
            populations=self._f_star,
            moments=self._moments,
            rho=self._rho,
            momentum=(self._jx, self._jy, self._jz),
            velocity=(self._ux, self._uy, self._uz),
            force=(self._fx, self._fy, self._fz),
        )
        self._f_post = None
        if model.encoding == "home" and self._collision_name in ("srt", "trt"):
            self._f_post = wp.zeros(19 * self._stride, dtype=float, device=self.device)

        self._raw_moments = None
        self._raw_source_moments = None
        self._post_raw_moments = None
        self._post_central_moments = None
        self._inverse_moment_transform = None
        self._mrt_rates = None
        if self._collision_space in (CollisionSpace.RAW_MOMENT, CollisionSpace.NOCM_MOMENT):
            self._raw_moments = wp.zeros(19 * self._stride, dtype=float, device=self.device)
            self._raw_source_moments = wp.zeros(
                19 * self._stride, dtype=float, device=self.device
            )
            self._post_raw_moments = wp.zeros(19 * self._stride, dtype=float, device=self.device)
            if self._collision_space is CollisionSpace.NOCM_MOMENT:
                self._post_central_moments = wp.zeros(
                    19 * self._stride, dtype=float, device=self.device
                )
            _, inverse = moments.host_moment_matrices()
            rates = moments.host_relaxation_rates(
                float(model.omega),
                float(model.mrt_third_omega),
                float(model.mrt_fourth_omega),
            )
            self._inverse_moment_transform = wp.array(
                inverse.reshape(-1), dtype=float, device=self.device
            )
            self._mrt_rates = wp.array(rates, dtype=float, device=self.device)


    def create_state(self, requires_grad: bool = False) -> LbmStateBase:
        """Create the concrete persistent state selected by the model."""
        if self.model.encoding == "home":
            return HomeLbmState(self.model, requires_grad=requires_grad)
        return FullFLbmState(self.model, requires_grad=requires_grad)

    # ------------------------------------------------------------------
    # Boundary condition helpers
    # ------------------------------------------------------------------

    def _sync_bc_from_model(self) -> None:
        """Copy BC parameters from the model to device arrays."""
        wp.copy(
            self._bc_types,
            wp.array(np.array(self.model.bc_types, dtype=np.int32), dtype=wp.int32, device=self.device),
        )
        wp.copy(
            self._bc_vel_x,
            wp.array(np.array([v[0] for v in self.model.bc_velocity], dtype=np.float32), dtype=float, device=self.device),
        )
        wp.copy(
            self._bc_vel_y,
            wp.array(np.array([v[1] for v in self.model.bc_velocity], dtype=np.float32), dtype=float, device=self.device),
        )
        wp.copy(
            self._bc_vel_z,
            wp.array(np.array([v[2] for v in self.model.bc_velocity], dtype=np.float32), dtype=float, device=self.device),
        )
        wp.copy(
            self._bc_density,
            wp.array(
                np.array(self.model.bc_density, dtype=np.float32),
                dtype=float,
                device=self.device,
            ),
        )
        wp.copy(
            self._bc_convective_speed,
            wp.array(
                np.array(self.model.bc_convective_speed, dtype=np.float32),
                dtype=float,
                device=self.device,
            ),
        )

    def set_boundary_condition(
        self,
        face: int,
        bc_type: int | str,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        convective_speed: float | None = None,
        density: float | None = None,
    ) -> None:
        """Set the boundary condition on one face of the domain.

        Parameters
        ----------
        face:
            Face index: 0=xmin, 1=xmax, 2=ymin, 3=ymax, 4=zmin, 5=zmax.
        bc_type:
            Integer compatibility code or a public boundary name such as
            ``"zou_he"``, ``"pressure"`` or ``"convective"``.
        velocity:
            Prescribed velocity ``(ux, uy, uz)`` in lattice units.  Only
            used when *bc_type* == 1 (Zou-He).
        convective_speed:
            First-order outlet Courant speed in ``[0, 1]``.
        density:
            Prescribed density for a pressure boundary.
        """
        if face < 0 or face >= 6:
            raise ValueError(f"LBM boundary face must be in [0, 5], got {face}")
        resolved_bc_type, boundary_model = normalize_boundary_type(bc_type)
        bc_type = resolved_bc_type
        if boundary_model.value == "moving_wall":
            self.model.has_moving_walls = True
        elif boundary_model.value == "cut_link":
            self.model.use_cut_link = True
        axis = face // 2
        if self.model.bc_periodic[axis] and bc_type in (1, 2, 4):
            raise ValueError("An open boundary cannot be placed on a periodic axis")
        if convective_speed is not None and not (0.0 <= convective_speed <= 1.0):
            raise ValueError(
                f"convective_speed must be in [0, 1], got {convective_speed}"
            )
        if density is not None and density <= 0.0:
            raise ValueError(f"boundary density must be > 0, got {density}")
        types = list(self.model.bc_types)
        boundary_models = list(self.model.boundary_models or ())
        vels = list(self.model.bc_velocity)
        densities = list(self.model.bc_density)
        speeds = list(self.model.bc_convective_speed)
        types[face] = bc_type
        boundary_models[face] = boundary_model.value
        if bc_type == 3:
            types[face ^ 1] = 3
            boundary_models[face ^ 1] = "periodic"
        elif types[face ^ 1] == 3:
            raise ValueError(
                "Clear or replace both faces of a periodic axis together"
            )
        vels[face] = tuple(velocity)
        if convective_speed is not None:
            speeds[face] = float(convective_speed)
        if density is not None:
            densities[face] = float(density)
        self.model.bc_types = tuple(types)  # type: ignore[assignment]
        self.model.boundary_models = tuple(boundary_models)  # type: ignore[assignment]
        self.model.bc_velocity = tuple(vels)  # type: ignore[assignment]
        self.model.bc_density = tuple(densities)  # type: ignore[assignment]
        self.model.bc_convective_speed = tuple(speeds)  # type: ignore[assignment]
        self._boundary_resolution = resolve_boundary_faces(
            tuple(int(value) for value in self.model.bc_types),
            (self.nx, self.ny, self.nz),
        )
        if self._boundary_resolution.conflict is not None:
            warnings.warn(
                self._boundary_resolution.conflict.warning_message,
                RuntimeWarning,
                stacklevel=2,
            )
        self._boundary_history_ready = False
        if any(value == BC_OUTFLOW for value in self.model.bc_types) and self._boundary_history is None:
            self._boundary_history = wp.zeros(
                19 * self._stride, dtype=float, device=self.device
            )
        self._sync_bc_from_model()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: LbmStateBase,
        state_out: LbmStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance ``post-collision -> stream -> collide -> post-collision``."""
        del contacts, control, dt
        self._copy_boundary_fields(state_in, state_out)
        px, py, pz = self.model._periodic_ints

        # 1. StateProvider + StreamingEngine always form logical populations.
        self._stream_to_populations(state_in, px, py, pz)
        if self.model.use_cut_link:
            if isinstance(state_in, FullFLbmState):
                cut_link_kernel = streaming.apply_fullf_cut_link_transport_kernel
                cut_link_inputs = [
                    state_in.f_post,
                    state_in.solid_phi,
                    self._f_star,
                    self._bc_types,
                    px,
                    py,
                    pz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ]
            else:
                assert isinstance(state_in, HomeLbmState)
                cut_link_kernel = streaming.apply_home_cut_link_transport_kernel
                cut_link_inputs = [
                    *state_in.kinetic_fields,
                    state_in.solid_phi,
                    self._f_star,
                    self._bc_types,
                    px,
                    py,
                    pz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ]
            wp.launch(
                cut_link_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=cut_link_inputs,
                device=self.device,
            )
        if self.model.has_moving_walls:
            wp.launch(
                kernels.apply_moving_wall_transport_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._f_star,
                    state_in.density,
                    state_in.solid_phi,
                    state_in.vel_solid_u,
                    state_in.vel_solid_v,
                    state_in.vel_solid_w,
                    self._bc_types,
                    int(self.model.use_cut_link),
                    px,
                    py,
                    pz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
                device=self.device,
            )

        # 2. Part 3 completes all open-boundary populations before collection.
        self._complete_open_boundaries(self._f_star)

        # 3. The collector inventories the same completed f*[Q] for all paths.
        wp.launch(
            encoding.populations_to_home_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._f_star,
                *self._moments,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )

        # 4. Part 2 computes one physical force density and one hydro closure.
        self._compute_force_density(state_out)
        wp.launch(
            forcing.hydro_closure_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._rho,
                self._jx,
                self._jy,
                self._jz,
                self._fx,
                self._fy,
                self._fz,
                self._ux,
                self._uy,
                self._uz,
            ],
            device=self.device,
        )

        if self.model.use_regularization and self.model.omega_reg > 0.0:
            wp.launch(
                kernels.reg_trt_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._f_star,
                    self._rho,
                    self._ux,
                    self._uy,
                    self._uz,
                    float(self.model.omega_reg),
                    px,
                    py,
                    pz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
                device=self.device,
            )

        # 5. Part 1 dispatches by collision space; each path translates F.
        if isinstance(state_out, FullFLbmState):
            output_f = state_out.f_post
        else:
            output_f = self._f_post

        if self._collision_space in (
            CollisionSpace.POPULATION,
            CollisionSpace.EVEN_ODD_POPULATION,
        ):
            assert output_f is not None
            omega_even, omega_odd = self._collision_backend.relaxation_rates(self.model)
            wp.launch(
                collisions.epc_forced_collision_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._f_star,
                    self._rho,
                    self._ux,
                    self._uy,
                    self._uz,
                    self._fx,
                    self._fy,
                    self._fz,
                    output_f,
                    omega_even,
                    omega_odd,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
                device=self.device,
            )
            if isinstance(state_out, HomeLbmState):
                self._encode_populations_to_home(output_f, state_out)
        elif self._collision_space is CollisionSpace.RAW_MOMENT:
            assert isinstance(state_out, FullFLbmState)
            self._collide_raw_mrt(state_out.f_post)
        elif self._collision_space is CollisionSpace.NOCM_MOMENT:
            assert isinstance(state_out, FullFLbmState)
            self._collide_fullf_nocm(state_out.f_post)
        else:
            assert isinstance(state_out, HomeLbmState)
            wp.launch(
                collisions.home_nocm_mrt_collision_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    *self._moments,
                    self._ux,
                    self._uy,
                    self._uz,
                    self._fx,
                    self._fy,
                    self._fz,
                    *state_out.kinetic_fields,
                    float(self.model.omega),
                ],
                device=self.device,
            )

        self._write_observables(state_out)

    def _copy_boundary_fields(self, state_in: LbmStateBase, state_out: LbmStateBase) -> None:
        for name in ("solid_phi", "solid_body_id", "vel_solid_u", "vel_solid_v", "vel_solid_w"):
            wp.copy(getattr(state_out, name), getattr(state_in, name))

    def _home_inputs(self, state: HomeLbmState) -> list[wp.array]:
        return list(state.kinetic_fields)

    def _stream_to_populations(self, state: LbmStateBase, px: int, py: int, pz: int) -> None:
        assert self._f_star is not None
        common = [state.solid_phi, self._f_star, px, py, pz, self.nx, self.ny, self.nz, self._stride]
        if isinstance(state, FullFLbmState):
            kernel = streaming.stream_fullf_to_populations_kernel
            inputs = [state.f_post, *common]
        elif isinstance(state, HomeLbmState):
            kernel = streaming.stream_home_to_populations_kernel
            inputs = [*self._home_inputs(state), *common]
        else:
            raise TypeError(f"Unsupported LBM state type: {type(state).__name__}")
        wp.launch(kernel, dim=(self.nx, self.ny, self.nz), inputs=inputs, device=self.device)

    def _complete_open_boundaries(self, populations: wp.array) -> None:
        if not self._boundary_resolution.has_open_boundaries:
            return
        has_convective = any(value == BC_OUTFLOW for value in self.model.bc_types)
        if has_convective and self._boundary_history is None:
            self._boundary_history = wp.zeros(
                19 * self._stride, dtype=float, device=self.device
            )
        if has_convective and not self._boundary_history_ready:
            assert self._boundary_history is not None
            wp.copy(self._boundary_history, populations)
            self._boundary_history_ready = True
        wp.launch(
            kernels.apply_boundary_conditions_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                populations,
                self._boundary_history if self._boundary_history is not None else populations,
                self._bc_types,
                self._bc_vel_x,
                self._bc_vel_y,
                self._bc_vel_z,
                self._bc_density,
                self._bc_convective_speed,
                self.nx,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )
        if has_convective:
            assert self._boundary_history is not None
            wp.copy(self._boundary_history, populations)

    def _compute_force_density(self, state_out: LbmStateBase) -> None:
        if self._force_provider.includes_shan_chen:
            self._step_count += 1
            if (self._step_count - 1) % self._sc_stride == 0:
                px, py, pz = self.model._periodic_ints
                wp.launch(
                    kernels.compute_shan_chen_force_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self._rho,
                        state_out.solid_phi,
                        self._sc_fx,
                        self._sc_fy,
                        self._sc_fz,
                        float(self.model.G),
                        int(self.model.psi_type),
                        float(self.model.psi_ref),
                        float(self.model.sc_solid_psi_scale),
                        float(self.model.sc_boundary_psi),
                        float(self.model.cs_a),
                        float(self.model.cs_b),
                        float(self.model.cs_T),
                        int(self.model.sc_homogeneous_early_out),
                        float(self.model.sc_homogeneous_rel_tol),
                        px,
                        py,
                        pz,
                        self.nx,
                        self.ny,
                        self.nz,
                    ],
                    device=self.device,
                )
        gx = float(self.model.gravity_x) if self._force_provider.includes_gravity else 0.0
        gy = float(self.model.gravity_y) if self._force_provider.includes_gravity else 0.0
        gz = float(self.model.gravity_z) if self._force_provider.includes_gravity else 0.0
        wp.launch(
            forcing.compose_force_density_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._rho,
                self._sc_fx,
                self._sc_fy,
                self._sc_fz,
                self._fx,
                self._fy,
                self._fz,
                gx,
                gy,
                gz,
                int(self._force_provider.includes_shan_chen),
            ],
            device=self.device,
        )

    def _collide_raw_mrt(self, output_f: wp.array) -> None:
        assert self._raw_moments is not None
        assert self._raw_source_moments is not None
        assert self._post_raw_moments is not None
        assert self._inverse_moment_transform is not None
        assert self._mrt_rates is not None
        wp.launch(
            moments.populations_to_raw_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self._f_star, self._raw_moments, self.ny, self.nz, self._stride],
            device=self.device,
        )
        self._translate_force_to_raw_moments()
        wp.launch(
            moments.raw_mrt_collision_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._raw_moments,
                self._raw_source_moments,
                self._post_raw_moments,
                self._rho,
                self._ux,
                self._uy,
                self._uz,
                self._fx,
                self._fy,
                self._fz,
                self._mrt_rates,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )
        wp.launch(
            moments.raw_moments_to_populations_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._post_raw_moments,
                self._inverse_moment_transform,
                output_f,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )

    def _collide_fullf_nocm(self, output_f: wp.array) -> None:
        assert self._raw_moments is not None
        assert self._raw_source_moments is not None
        assert self._post_raw_moments is not None
        assert self._post_central_moments is not None
        assert self._inverse_moment_transform is not None
        assert self._mrt_rates is not None
        wp.launch(
            moments.populations_to_raw_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self._f_star, self._raw_moments, self.ny, self.nz, self._stride],
            device=self.device,
        )
        self._translate_force_to_raw_moments()
        wp.launch(
            moments.nocm_collision_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._raw_moments,
                self._raw_source_moments,
                self._post_central_moments,
                self._rho,
                self._ux,
                self._uy,
                self._uz,
                self._fx,
                self._fy,
                self._fz,
                self._mrt_rates,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )
        wp.launch(
            moments.central_to_raw_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._post_central_moments,
                self._post_raw_moments,
                self._ux,
                self._uy,
                self._uz,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )
        wp.launch(
            moments.raw_moments_to_populations_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._post_raw_moments,
                self._inverse_moment_transform,
                output_f,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )

    def _translate_force_to_raw_moments(self) -> None:
        assert self._raw_source_moments is not None
        wp.launch(
            moments.guo_source_to_raw_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._ux,
                self._uy,
                self._uz,
                self._fx,
                self._fy,
                self._fz,
                self._raw_source_moments,
                self.ny,
                self.nz,
                self._stride,
            ],
            device=self.device,
        )

    def _encode_populations_to_home(self, f_post: wp.array, state: HomeLbmState) -> None:
        wp.launch(
            encoding.populations_to_home_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[f_post, *state.kinetic_fields, self.ny, self.nz, self._stride],
            device=self.device,
        )

    def _write_observables(self, state_out: LbmStateBase) -> None:
        wp.copy(state_out.density, self._rho)
        wp.copy(state_out.velocity_x, self._ux)
        wp.copy(state_out.velocity_y, self._uy)
        wp.copy(state_out.velocity_z, self._uz)
        wp.copy(state_out.force_x, self._fx)
        wp.copy(state_out.force_y, self._fy)
        wp.copy(state_out.force_z, self._fz)
        wp.launch(kernels.moments_to_mac_u_kernel, dim=(self.nx + 1, self.ny, self.nz), inputs=[self._ux, state_out.vel_u, self.nx], device=self.device)
        wp.launch(kernels.moments_to_mac_v_kernel, dim=(self.nx, self.ny + 1, self.nz), inputs=[self._uy, state_out.vel_v, self.ny], device=self.device)
        wp.launch(kernels.moments_to_mac_w_kernel, dim=(self.nx, self.ny, self.nz + 1), inputs=[self._uz, state_out.vel_w, self.nz], device=self.device)


    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_equilibrium(
        self,
        state: LbmStateBase,
        rho0: float = 1.0,
        u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Initialise *state.f* to the equilibrium for uniform (ρ₀, u₀).

        Also sets the macroscopic fields on *state* so that
        ``state.density`` / ``state.velocity_*`` are consistent from the
        first step.

        Parameters
        ----------
        state:
            LBM state whose ``f`` array will be overwritten.
        rho0:
            Uniform initial density (default 1.0).
        u0:
            Uniform initial velocity ``(ux, uy, uz)`` in lattice units.
        """
        u0x, u0y, u0z = u0
        self._boundary_history_ready = False
        if self._boundary_history is not None:
            self._boundary_history.zero_()
        self._step_count = 0
        self._sc_fx.zero_()
        self._sc_fy.zero_()
        self._sc_fz.zero_()

        if isinstance(state, FullFLbmState):
            wp.launch(
                kernels.initialize_equilibrium_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state.f_post, rho0, u0x, u0y, u0z,
                    self.nx, self.ny, self.nz, self._stride,
                ],
                device=self.device,
            )
        elif isinstance(state, HomeLbmState):
            wp.launch(
                encoding.initialize_home_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[*state.kinetic_fields, rho0, u0x, u0y, u0z],
                device=self.device,
            )
        else:
            raise TypeError(f"Unsupported LBM state type: {type(state).__name__}")

        # Populate macroscopic fields for consistency
        state.density.fill_(rho0)
        state.velocity_x.fill_(u0x)
        state.velocity_y.fill_(u0y)
        state.velocity_z.fill_(u0z)
