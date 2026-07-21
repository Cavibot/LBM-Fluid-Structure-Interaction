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
    flatten_top_interface_layer,
    level_surface_high_to_low,
    reabsorb_orphan_liquid,
    seed_home_vof_gpu,
    set_face_bc_gpu,
    step_home_vof_gpu,
    sync_solids_from_lbm_state,
    topup_surface_with_budget,
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
        self._late_pool = None
        self._height_eq_counter = 0
        self._last_height_eq_stats: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return str(self.model.lbm_backend).lower() == "home_fp32"

    @property
    def late_pool(self):
        """Lazy late-pool surface controller (quiet level / orphan / topup)."""
        if self._late_pool is None:
            from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.surface_policy import (
                HomeVofLatePoolController,
            )

            self._late_pool = HomeVofLatePoolController(self)
        return self._late_pool

    def configure_late_pool(self, *, home_faithful: bool = False, **kwargs):
        """Create/configure the late-pool controller (call once after construction)."""
        from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.surface_policy import (
            HomeVofLatePoolController,
        )

        self._late_pool = HomeVofLatePoolController(
            self, home_faithful=home_faithful, **kwargs
        )
        return self._late_pool

    def reset(self) -> None:
        self._gpu = None
        self._domain_bc = home_domain_bc_from_model(self.model)
        if self._late_pool is not None:
            self._late_pool.reset()

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
                moment_quant=bool(
                    getattr(self.model, "vof_home_moment_quant", False)
                ),
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

    def step(self, state_out: LbmState, state_in: LbmState | None = None) -> None:
        """Advance one lattice step on GPU; write macros into ``state_out``.

        When ``state_in`` is provided, rigid ``solid_phi`` / wall velocities are
        synced from the LBM state before the HOME-FREE step (FSI path).
        """
        buf = self._ensure_gpu()
        if state_in is not None:
            sync_solids_from_lbm_state(buf, state_in)
        # Domain BCs are static for this model — build GPU face tables once.
        if getattr(self, "_face_bc_ready", False) is False:
            self._domain_bc = home_domain_bc_from_model(self.model)
            set_face_bc_gpu(buf, self._domain_bc)
            self._face_bc_ready = True
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
            home_fill_empty=bool(self.model.vof_home_fill_empty),
            home_wall_eq=bool(self.model.vof_home_wall_eq),
            seal_fg=bool(self.model.vof_seal_fg),
            bubble_pressure=bool(self.model.vof_bubble_pressure),
            bubble_atm_volume=float(self.model.vof_bubble_atm_volume),
            bubble_update_every=int(self.model.vof_bubble_update_every),
            bubble_disjoint=bool(self.model.vof_bubble_disjoint),
            bubble_disjoint_factor=float(self.model.vof_bubble_disjoint_factor),
            bubble_small_sigma=bool(self.model.vof_bubble_small_sigma),
            bubble_small_vol=float(self.model.vof_bubble_small_vol),
            bubble_small_six_sigma=float(self.model.vof_bubble_small_six_sigma),
            bubble_eddy=bool(self.model.vof_bubble_eddy),
            bubble_eddy_atm_vol=float(self.model.vof_bubble_eddy_atm_vol),
            moment_quant=bool(getattr(self.model, "vof_home_moment_quant", False)),
            moment_quant_dither=bool(
                getattr(self.model, "vof_home_moment_quant_dither", True)
            ),
        )
        # Opt-in free-surface leveling (IF-only, gradual). Runs in the solver.
        if bool(getattr(self.model, "vof_height_eq", False)):
            every = max(1, int(getattr(self.model, "vof_height_eq_every", 8)))
            self._height_eq_counter += 1
            if self._height_eq_counter % every == 0:
                # Pull tiny stats only occasionally (status logs); skip D2H most calls.
                self._height_eq_stat_pulls = getattr(self, "_height_eq_stat_pulls", 0) + 1
                sync_stats = self._height_eq_stat_pulls % 4 == 1
                self._last_height_eq_stats = self.apply_height_equation(
                    state_out=None, sync_stats=sync_stats
                )
        self.sync_to_state(state_out)
        # HOME-FREE does not use distribution ``f``; skip the D3Q27×N³ zero.
        # Keep solid fields on the output buffer for feedback / visualisation.
        if state_in is not None:
            wp.copy(state_out.solid_phi, state_in.solid_phi)
            wp.copy(state_out.solid_body_id, state_in.solid_body_id)
            wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
            wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
            wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

    def apply_height_equation(
        self,
        state_out: LbmState | None = None,
        *,
        sync_stats: bool = False,
    ) -> dict[str, float]:
        """Gradual free-surface leveling: local IF Laplacian + airborne drop."""
        from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.height_eq import (
            apply_vof_height_equation,
        )

        buf = self._ensure_gpu()
        stats = apply_vof_height_equation(
            buf,
            rate=float(self.model.vof_height_eq_rate),
            u_max=float(self.model.vof_height_eq_u_max),
            dh_cap=float(self.model.vof_height_eq_dh_cap),
            n_sweeps=int(getattr(self.model, "vof_height_eq_sweeps", 1)),
            sync_stats=bool(sync_stats),
        )
        if sync_stats or not self._last_height_eq_stats:
            self._last_height_eq_stats = dict(stats)
        elif stats:
            # Keep prior log fields when this call skipped D2H.
            merged = dict(self._last_height_eq_stats)
            merged.update(stats)
            self._last_height_eq_stats = merged
        if state_out is not None:
            self.sync_to_state(state_out)
        return dict(self._last_height_eq_stats)

    def level_high_to_low(self, state_out: LbmState | None = None) -> float:
        """Once-per-frame: move mass from high free-surface tops to lower columns.

        Returns inventory delta of ``Σmass`` (≈0 if conservative).
        """
        buf = self._ensure_gpu()
        dmass = level_surface_high_to_low(
            buf,
            rate=float(self.model.vof_quiet_fill_rate),
            dz_min=2,
            wet_phi=0.5,
        )
        if state_out is not None:
            self.sync_to_state(state_out)
        return float(dmass)

    def flatten_top_interface(self, state_out: LbmState | None = None) -> dict[str, float]:
        """Sub-cell late-pool pass: one interface layer, shared continuous height."""
        buf = self._ensure_gpu()
        stats = flatten_top_interface_layer(buf)
        if state_out is not None:
            self.sync_to_state(state_out)
        return dict(stats)

    def topup_with_budget(
        self,
        budget: float,
        state_out: LbmState | None = None,
        target_z: int | None = None,
    ) -> float:
        """Invent ≤``budget`` mass to raise low columns (``inf`` = fill all)."""
        buf = self._ensure_gpu()
        invented = topup_surface_with_budget(
            buf,
            budget=float(budget) if np.isfinite(budget) else 1.0e30,
            wet_phi=0.5,
            target_z=target_z,
        )
        if state_out is not None:
            self.sync_to_state(state_out)
        return float(invented)

    def reabsorb_orphans(self, state_out: LbmState | None = None) -> tuple[float, int]:
        """Fold airborne orphan liquid blobs into the main pool (conservative)."""
        buf = self._ensure_gpu()
        moved, n_orphans = reabsorb_orphan_liquid(
            buf,
            max_cells=int(self.model.vof_orphan_max_cells),
            height_margin=int(self.model.vof_orphan_height_margin),
        )
        if state_out is not None:
            self.sync_to_state(state_out)
        return float(moved), int(n_orphans)

    def last_bubble_stats(self) -> dict[str, float | int]:
        """Diagnostics from the last host bubble CCL update (empty if disabled)."""
        if self._gpu is None:
            return {"n_bubbles": 0, "n_trapped": 0, "rho_max_bubble": 1.0}
        return dict(getattr(self._gpu, "_last_bubble_stats", {}))

    def copy_kappa_to(self, kappa_dst: wp.array) -> None:
        """Copy PLIC κ onto solver visual/metrics buffer (may be zero if γ=0)."""
        if self._gpu is None:
            kappa_dst.zero_()
            return
        wp.copy(kappa_dst, self._gpu.kappa)
