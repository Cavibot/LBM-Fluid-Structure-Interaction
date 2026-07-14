# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bridge: moment HOME-FREE VOF ↔ distribution ``LbmState`` (H5/H6).

Owns GPU-resident HOME-FREE buffers (H6 Warp). Each step syncs ρ,u,φ,cell_type
onto ``LbmState`` for visualisation. Distribution ``f`` is unused.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
    HomeFaceBC,
    HomeFaceKind,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
    HomeVofGpuBuffers,
    alloc_home_vof_gpu,
    seed_home_vof_gpu,
    set_face_bc_gpu,
    step_home_vof_gpu,
    upload_home_vof_state,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_step import (
    HomeVofState,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.step import (
    HomeMomentArrays,
)
from wanphys._src.fluid.fluid_grid.lbm.constants import (
    BC_OUTFLOW,
    BC_PERIODIC,
    BC_VELOCITY_INLET,
)
from wanphys._src.fluid.fluid_grid.lbm.model import LbmModel
from wanphys._src.fluid.fluid_grid.lbm.state import LbmState


def home_domain_bc_from_model(model: LbmModel) -> HomeDomainBC:
    """Map six-face ``LbmModel`` BC types to :class:`HomeDomainBC`."""
    px, py, pz = model._periodic_ints
    periodic_axis = (bool(px), bool(py), bool(pz))

    def _face(idx: int, axis: int) -> HomeFaceBC:
        if periodic_axis[axis] or int(model.bc_types[idx]) == BC_PERIODIC:
            return HomeFaceBC(kind=HomeFaceKind.PERIODIC)
        vx, vy, vz = model.bc_velocity[idx]
        kind_i = int(model.bc_types[idx])
        if kind_i == BC_VELOCITY_INLET:
            return HomeFaceBC(
                kind=HomeFaceKind.ZOU_HE, ux=float(vx), uy=float(vy), uz=float(vz),
            )
        if kind_i == BC_OUTFLOW:
            return HomeFaceBC(kind=HomeFaceKind.ZOU_HE, ux=0.0, uy=0.0, uz=0.0)
        return HomeFaceBC(
            kind=HomeFaceKind.WALL, ux=float(vx), uy=float(vy), uz=float(vz),
        )

    return HomeDomainBC(
        xmin=_face(0, 0),
        xmax=_face(1, 0),
        ymin=_face(2, 1),
        ymax=_face(3, 1),
        zmin=_face(4, 2),
        zmax=_face(5, 2),
    )


class HomeFp32VofBridge:
    """GPU HOME-FREE VOF integrator synced to ``LbmState``."""

    def __init__(self, model: LbmModel) -> None:
        self.model = model
        self._gpu: HomeVofGpuBuffers | None = None
        self._domain_bc = home_domain_bc_from_model(model)

    @property
    def enabled(self) -> bool:
        return str(self.model.lbm_backend).lower() == "home_fp32"

    @property
    def vof_state(self) -> HomeVofState | None:
        """Host snapshot (slow); prefer GPU buffers for stepping."""
        if self._gpu is None:
            return None
        g = self._gpu
        return HomeVofState(
            moments=HomeMomentArrays(
                rho=g.rho.numpy().astype(np.float64),
                ux=g.ux.numpy().astype(np.float64),
                uy=g.uy.numpy().astype(np.float64),
                uz=g.uz.numpy().astype(np.float64),
                sxx=g.sxx.numpy().astype(np.float64),
                syy=g.syy.numpy().astype(np.float64),
                szz=g.szz.numpy().astype(np.float64),
                sxy=g.sxy.numpy().astype(np.float64),
                sxz=g.sxz.numpy().astype(np.float64),
                syz=g.syz.numpy().astype(np.float64),
            ),
            phi=g.phi.numpy().astype(np.float64),
            cell_type=g.cell_type.numpy().astype(np.int32),
        )

    def reset(self) -> None:
        self._gpu = None
        self._domain_bc = home_domain_bc_from_model(self.model)

    def _ensure_gpu(self) -> HomeVofGpuBuffers:
        if self._gpu is None:
            shape = (int(self.model.nx), int(self.model.ny), int(self.model.nz))
            device = str(self.model._device)
            self._domain_bc = home_domain_bc_from_model(self.model)
            self._gpu = alloc_home_vof_gpu(
                shape,
                self.model.lattice_spec,
                device=device,
                domain_bc=self._domain_bc,
            )
        return self._gpu

    def seed_dam_break(
        self,
        state: LbmState,
        dam_x: int,
        fill_z: int,
        rho_liquid: float | None = None,
    ) -> None:
        """Seed column on GPU and push macros onto ``state``."""
        rho0 = float(
            self.model.initial_density if rho_liquid is None else rho_liquid
        )
        buf = self._ensure_gpu()
        seed_home_vof_gpu(buf, dam_x=dam_x, fill_z=fill_z, rho_liquid=rho0)
        self.sync_to_state(state)
        state.f.zero_()

    def ensure_from_state(self, state: LbmState) -> None:
        """If no GPU state yet, upload macros from ``LbmState``."""
        if self._gpu is not None:
            return
        phi = state.phi.numpy().astype(np.float64)
        ctype = state.cell_type.numpy().astype(np.int32)
        rho = state.density.numpy().astype(np.float64)
        ux = state.velocity_x.numpy().astype(np.float64)
        uy = state.velocity_y.numpy().astype(np.float64)
        uz = state.velocity_z.numpy().astype(np.float64)
        gas = ctype == 0
        rho[gas] = 0.0
        ux[gas] = 0.0
        uy[gas] = 0.0
        uz[gas] = 0.0
        host = HomeVofState(
            moments=HomeMomentArrays(
                rho=rho, ux=ux, uy=uy, uz=uz,
                sxx=ux * ux, syy=uy * uy, szz=uz * uz,
                sxy=ux * uy, sxz=ux * uz, syz=uy * uz,
            ),
            phi=phi,
            cell_type=ctype,
        )
        buf = self._ensure_gpu()
        upload_home_vof_state(buf, host)

    def sync_to_state(self, state: LbmState) -> None:
        """Copy GPU HOME-FREE fields onto ``LbmState`` macros."""
        if self._gpu is None:
            return
        g = self._gpu
        wp.copy(state.density, g.rho)
        wp.copy(state.velocity_x, g.ux)
        wp.copy(state.velocity_y, g.uy)
        wp.copy(state.velocity_z, g.uz)
        from wanphys._src.fluid.fluid_grid.lbm import kernels as lbm_kernels

        nx, ny, nz = g.shape
        wp.launch(
            lbm_kernels.compute_pressure_kernel,
            dim=(nx, ny, nz),
            inputs=[g.rho, state.pressure],
            device=g.device,
        )
        wp.copy(state.phi, g.phi)
        wp.copy(state.cell_type, g.cell_type)

    def step(self, state_out: LbmState) -> None:
        """Advance one lattice step on GPU; write macros into ``state_out``."""
        buf = self._ensure_gpu()
        self._domain_bc = home_domain_bc_from_model(self.model)
        set_face_bc_gpu(buf, self._domain_bc)
        step_home_vof_gpu(
            buf,
            tau=float(self.model.tau),
            fx=float(self.model.gravity_x),
            fy=float(self.model.gravity_y),
            fz=float(self.model.gravity_z),
            rho_g0=float(self.model.vof_rho_gas),
            gamma=float(self.model.vof_gamma),
            eps_phi=float(self.model.vof_epsilon),
            rho_liquid=float(self.model.initial_density),
            kappa_smooth=int(self.model.vof_kappa_smooth),
            wall_wetting=float(self.model.vof_wall_wetting),
            wall_film_drain=bool(self.model.vof_wall_film_drain),
            wall_film_phi_max=float(self.model.vof_wall_film_phi_max),
            wall_film_u_max=float(self.model.vof_wall_film_u_max),
            wall_film_edge_only=bool(self.model.vof_wall_film_edge_only),
        )
        self.sync_to_state(state_out)
        state_out.f.zero_()

    def copy_kappa_to(self, kappa_dst: wp.array) -> None:
        """Copy PLIC κ onto solver visual/metrics buffer (may be zero if γ=0)."""
        if self._gpu is None:
            kappa_dst.zero_()
            return
        wp.copy(kappa_dst, self._gpu.kappa)
