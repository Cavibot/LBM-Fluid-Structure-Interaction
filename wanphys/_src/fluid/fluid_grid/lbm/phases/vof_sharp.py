# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""FSLBM-style sharp free-surface VOF phase (HOME-FREE Eq. 9–12)."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.lattice import CX, CY, CZ, OPPOSITE, W
from ..model import LbmModel
from ..state import LbmState
from . import vof_kernels
from . import vof_plic
from .shan_chen import MacroscopicBuffers


class VofSharpPhase:
    """Sharp free-surface VOF on top of the distribution D3Q19 LBM.

    - Skip gas collide/stream
    - Interface missing DFs via Eq. (11) with ``ρ_g = ρ_atm - 6 γ κ``
    - PLIC / Parker–Youngs curvature κ (Lehmann 2019) when ``vof_gamma > 0``
    - φ update via Körner mass exchange (Eq. 9–10)
    - L/I/G reclassification with ``ε_φ`` and closed interface layer
    """

    def __init__(self, model: LbmModel) -> None:
        self.model: LbmModel = model
        device = model._device
        self._cx = wp.array(np.asarray(CX, dtype=np.int32), dtype=wp.int32, device=device)
        self._cy = wp.array(np.asarray(CY, dtype=np.int32), dtype=wp.int32, device=device)
        self._cz = wp.array(np.asarray(CZ, dtype=np.int32), dtype=wp.int32, device=device)
        self._w = wp.array(np.asarray(W, dtype=np.float32), dtype=float, device=device)
        self._opp = wp.array(
            np.asarray(OPPOSITE, dtype=np.int32), dtype=wp.int32, device=device
        )

        nx, ny, nz = int(model.nx), int(model.ny), int(model.nz)
        self._fill_flag = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
        self._empty_flag = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
        self._cell_type_tmp = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
        self._phi_tmp = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self._excess_phi = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self._kappa = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self._kappa_tmp = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self._vol_liquid = wp.zeros(1, dtype=float, device=device)
        self._vol_interface = wp.zeros(1, dtype=float, device=device)
        self._n_interface = wp.zeros(1, dtype=wp.int32, device=device)
        self._target_volume: float | None = None
        self._visual = wp.zeros((nx, ny, nz), dtype=float, device=device)

    @property
    def enabled(self) -> bool:
        return str(self.model.phase_mode).lower() == "vof_sharp"

    @property
    def visual_field(self) -> wp.array3d:
        """Liquid=1 / interface=φ field for SSFR rendering."""
        return self._visual

    def reset(self) -> None:
        self._fill_flag.zero_()
        self._empty_flag.zero_()
        self._excess_phi.zero_()
        self._kappa.zero_()
        self._kappa_tmp.zero_()
        self._target_volume = None

    def compute_moments(
        self,
        state: LbmState,
        buffers: MacroscopicBuffers,
        nx: int,
        ny: int,
        nz: int,
        stride: int,
    ) -> None:
        wp.launch(
            vof_kernels.compute_moments_vof_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state.f,
                state.cell_type,
                stride,
                nx,
                ny,
                nz,
                buffers.rho,
                buffers.ux,
                buffers.uy,
                buffers.uz,
            ],
        )

    def collide_stream(
        self,
        state_in: LbmState,
        state_out: LbmState,
        buffers: MacroscopicBuffers,
        nx: int,
        ny: int,
        nz: int,
        stride: int,
    ) -> None:
        px, py, pz = self.model._periodic_ints
        gamma = float(self.model.vof_gamma)
        if gamma != 0.0:
            wp.launch(
                vof_plic.vof_compute_kappa_kernel,
                dim=(nx, ny, nz),
                inputs=[
                    state_in.phi,
                    state_in.cell_type,
                    state_out.solid_phi,
                    self._kappa,
                    int(px),
                    int(py),
                    int(pz),
                    nx,
                    ny,
                    nz,
                ],
            )
            n_smooth = int(self.model.vof_kappa_smooth)
            for _ in range(n_smooth):
                wp.launch(
                    vof_plic.vof_smooth_kappa_kernel,
                    dim=(nx, ny, nz),
                    inputs=[
                        self._kappa,
                        state_in.cell_type,
                        state_out.solid_phi,
                        self._kappa_tmp,
                        int(px),
                        int(py),
                        int(pz),
                        nx,
                        ny,
                        nz,
                    ],
                )
                self._kappa, self._kappa_tmp = self._kappa_tmp, self._kappa
        else:
            self._kappa.zero_()
        wp.launch(
            vof_kernels.vof_collide_stream_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_in.f,
                buffers.rho,
                buffers.ux,
                buffers.uy,
                buffers.uz,
                state_in.cell_type,
                state_out.solid_phi,
                state_out.vel_solid_u,
                state_out.vel_solid_v,
                state_out.vel_solid_w,
                state_out.f,
                self._cx,
                self._cy,
                self._cz,
                self._w,
                self._opp,
                float(self.model.omega_plus),
                float(self.model.omega_minus),
                float(self.model.vof_rho_gas),
                gamma,
                self._kappa,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
                stride,
            ],
        )
        wp.launch(
            vof_kernels.vof_sanitize_distributions_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.f,
                state_in.cell_type,
                state_in.phi,
                self._w,
                float(self.model.initial_density),
                nx,
                ny,
                nz,
                stride,
            ],
        )

    def update_interface(
        self,
        state_in: LbmState,
        state_out: LbmState,
        buffers: MacroscopicBuffers,
        nx: int,
        ny: int,
        nz: int,
        stride: int,
    ) -> None:
        """Mass exchange, reclassification, and new-interface DF init."""
        px, py, pz = self.model._periodic_ints
        eps = float(self.model.vof_epsilon)

        wp.launch(
            vof_kernels.vof_update_phi_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_in.f,
                state_in.phi,
                state_in.cell_type,
                buffers.rho,
                state_out.solid_phi,
                self._phi_tmp,
                self._cx,
                self._cy,
                self._cz,
                self._opp,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
                stride,
            ],
        )
        wp.copy(state_out.phi, self._phi_tmp)

        self._excess_phi.zero_()
        wp.launch(
            vof_kernels.vof_reclassify_flags_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.phi,
                state_in.cell_type,
                state_out.solid_phi,
                self._cell_type_tmp,
                self._fill_flag,
                self._empty_flag,
                self._excess_phi,
                eps,
                nx,
                ny,
                nz,
            ],
        )
        wp.copy(state_out.cell_type, self._cell_type_tmp)

        wp.launch(
            vof_kernels.vof_convert_neighbors_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.cell_type,
                state_out.phi,
                state_out.solid_phi,
                self._fill_flag,
                self._empty_flag,
                self._cx,
                self._cy,
                self._cz,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
            ],
        )

        wp.launch(
            vof_kernels.vof_mild_topology_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.cell_type,
                state_out.phi,
                state_out.solid_phi,
                self._excess_phi,
                self._cx,
                self._cy,
                self._cz,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
            ],
        )

        # Redistribute truncated φ from fill/empty (no aggressive orphan wipe).
        wp.launch(
            vof_kernels.vof_redistribute_excess_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.phi,
                state_out.cell_type,
                state_out.solid_phi,
                self._excess_phi,
                self._cx,
                self._cy,
                self._cz,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
            ],
        )
        self._excess_phi.zero_()

        wp.launch(
            vof_kernels.vof_clamp_phi_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.phi,
                state_out.cell_type,
                state_out.solid_phi,
                nx,
                ny,
                nz,
            ],
        )

        wp.launch(
            vof_kernels.vof_init_new_interface_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.f,
                state_out.cell_type,
                state_in.cell_type,
                buffers.rho,
                buffers.ux,
                buffers.uy,
                buffers.uz,
                state_out.solid_phi,
                self._cx,
                self._cy,
                self._cz,
                self._w,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
                stride,
                float(self.model.initial_density),
            ],
        )

        wp.launch(
            vof_kernels.vof_zero_gas_distributions_kernel,
            dim=(nx, ny, nz),
            inputs=[state_out.f, state_out.cell_type, nx, ny, nz, stride],
        )

        # Sanitize again after interface init / type changes.
        wp.launch(
            vof_kernels.vof_sanitize_distributions_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_out.f,
                state_out.cell_type,
                state_out.phi,
                self._w,
                float(self.model.initial_density),
                nx,
                ny,
                nz,
                stride,
            ],
        )

        self._project_volume(state_out, nx, ny, nz)
        self.update_visual_field(state_out, nx, ny, nz)

    def update_visual_field(self, state: LbmState, nx: int, ny: int, nz: int) -> None:
        wp.launch(
            vof_kernels.vof_build_visual_field_kernel,
            dim=(nx, ny, nz),
            inputs=[state.phi, state.cell_type, self._visual],
        )

    def _project_volume(self, state: LbmState, nx: int, ny: int, nz: int) -> None:
        """Keep Σφ near the seeded volume by rescaling interface cells only."""
        if self._target_volume is None:
            return

        self._vol_liquid.zero_()
        self._vol_interface.zero_()
        self._n_interface.zero_()
        wp.launch(
            vof_kernels.vof_accumulate_volume_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state.phi,
                state.cell_type,
                state.solid_phi,
                self._vol_liquid,
                self._vol_interface,
                self._n_interface,
            ],
        )
        wp.synchronize_device(self.model._device)
        vol_l = float(self._vol_liquid.numpy()[0])
        vol_i = float(self._vol_interface.numpy()[0])
        n_i = int(self._n_interface.numpy()[0])
        if n_i <= 0:
            return
        if not (vol_l == vol_l and vol_i == vol_i):
            return  # NaN guard
        target_i = float(self._target_volume) - vol_l
        if target_i < 0.0:
            # Too much liquid tagged; skip rather than wiping interfaces.
            return
        if vol_i < 1.0e-8:
            return
        scale = target_i / vol_i
        if scale != scale:
            return
        # Avoid extreme corrections from transient topology glitches.
        if scale < 0.85:
            scale = 0.85
        if scale > 1.15:
            scale = 1.15
        wp.launch(
            vof_kernels.vof_project_interface_volume_kernel,
            dim=(nx, ny, nz),
            inputs=[state.phi, state.cell_type, state.solid_phi, scale, nx, ny, nz],
        )

    def apply_gravity_shift(
        self,
        buffers: MacroscopicBuffers,
        cell_type: wp.array3d,
        gx: float,
        gy: float,
        gz: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        """Pre-collision gravity via velocity shift (preferred for VOF)."""
        if gx == 0.0 and gy == 0.0 and gz == 0.0:
            return
        wp.launch(
            vof_kernels.apply_gravity_velocity_shift_vof_kernel,
            dim=(nx, ny, nz),
            inputs=[
                buffers.ux,
                buffers.uy,
                buffers.uz,
                cell_type,
                gx,
                gy,
                gz,
                float(self.model.tau),
                nx,
                ny,
                nz,
            ],
        )

    def apply_body_force(
        self,
        state: LbmState,
        cell_type: wp.array3d,
        gx: float,
        gy: float,
        gz: float,
        nx: int,
        ny: int,
        nz: int,
        stride: int,
    ) -> None:
        if gx == 0.0 and gy == 0.0 and gz == 0.0:
            return
        wp.launch(
            vof_kernels.apply_guo_force_vof_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state.f,
                cell_type,
                gx,
                gy,
                gz,
                float(self.model.omega),
                nx,
                ny,
                nz,
                stride,
            ],
        )

    def seed_dam_break_column(
        self,
        state: LbmState,
        dam_x: int,
        fill_z: int,
        rho_liquid: float | None = None,
    ) -> None:
        """Initialise a dam-break liquid column and mark the free surface."""
        nx, ny, nz = int(self.model.nx), int(self.model.ny), int(self.model.nz)
        stride = nx * ny * nz
        rho0 = float(self.model.initial_density if rho_liquid is None else rho_liquid)
        px, py, pz = self.model._periodic_ints

        wp.launch(
            vof_kernels.vof_seed_column_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state.f,
                state.density,
                state.phi,
                state.cell_type,
                int(dam_x),
                int(fill_z),
                rho0,
                nx,
                ny,
                nz,
                stride,
            ],
        )
        wp.launch(
            vof_kernels.vof_mark_initial_interface_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state.phi,
                state.cell_type,
                state.solid_phi,
                self._cx,
                self._cy,
                self._cz,
                int(px),
                int(py),
                int(pz),
                nx,
                ny,
                nz,
            ],
        )
        state.velocity_x.zero_()
        state.velocity_y.zero_()
        state.velocity_z.zero_()
        state.pressure.fill_(rho0 / 3.0)
        wp.synchronize_device(self.model._device)
        self._target_volume = float(state.phi.numpy().sum())
