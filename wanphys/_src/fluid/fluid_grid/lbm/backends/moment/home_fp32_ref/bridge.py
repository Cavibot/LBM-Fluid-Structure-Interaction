# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bridge: moment HOME ↔ distribution ``LbmState`` (shared base + FS branch).

Owns GPU-resident HOME buffers. ``phase_mode='none'`` runs the same fused
operators with free-surface policy off (full liquid); ``vof_sharp`` enables
mass/φ / fill-empty / optional surface cosmetics.
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
    set_face_bc_gpu,
    step_home_vof_gpu,
    sync_solids_from_lbm_state,
    topup_surface_with_budget,
    upload_home_vof_state,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.ic import (
    seed_dam_break_column,
    seed_droplet,
    seed_full_liquid,
    seed_pool,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_step import (
    CELL_GAS,
    CELL_LIQUID,
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


class HomeFp32Bridge:
    """GPU HOME integrator synced to ``LbmState`` (base + optional free-surface)."""

    def __init__(self, model: LbmModel) -> None:
        self.model = model
        self._gpu: HomeVofGpuBuffers | None = None
        self._domain_bc = home_domain_bc_from_model(model)
        self._late_pool = None
        self._height_eq_counter = 0
        self._last_height_eq_stats: dict[str, float] = {}
        self._fused_me: dict | None = None

    def prepare_fused_link_me(
        self,
        *,
        enabled: bool,
        solid_body_id: wp.array | None = None,
        body_q: wp.array | None = None,
        body_com: wp.array | None = None,
        body_f: wp.array | None = None,
        dh: float = 1.0,
        force_scale: float = 1.0,
    ) -> None:
        """Arm stream-time ME for the next :meth:`step` (cleared after the step)."""
        if not enabled:
            self._fused_me = None
            return
        self._fused_me = {
            "solid_body_id": solid_body_id,
            "body_q": body_q,
            "body_com": body_com,
            "body_f": body_f,
            "dh": float(dh),
            "force_scale": float(force_scale),
        }

    @property
    def fused_link_me_armed(self) -> bool:
        return self._fused_me is not None

    @property
    def enabled(self) -> bool:
        return str(self.model.lbm_backend).lower() == "home_fp32"

    @property
    def free_surface(self) -> bool:
        """True when the VOF / HOME-FREE free-surface branch is active."""
        return str(self.model.phase_mode).lower() == "vof_sharp"

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
        self._face_bc_ready = False
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

    def seed_full_liquid(
        self,
        state: LbmState,
        rho_liquid: float | None = None,
        *,
        u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Seed full-domain liquid (HOME base / ``phase_mode=none``)."""
        rho0 = float(
            self.model.initial_density if rho_liquid is None else rho_liquid
        )
        host = seed_full_liquid(
            self._ensure_gpu().shape,
            rho0,
            ux0=float(u0[0]),
            uy0=float(u0[1]),
            uz0=float(u0[2]),
        )
        self.seed_host_state(state, host)

    def seed_dam_break(
        self,
        state: LbmState,
        dam_x: int,
        fill_z: int,
        rho_liquid: float | None = None,
    ) -> None:
        """Seed dam-break column on GPU and push macros onto ``state``."""
        rho0 = float(
            self.model.initial_density if rho_liquid is None else rho_liquid
        )
        host = seed_dam_break_column(self._ensure_gpu().shape, dam_x, fill_z, rho0)
        self.seed_host_state(state, host)

    def seed_pool(
        self,
        state: LbmState,
        fill_z: int,
        rho_liquid: float | None = None,
    ) -> None:
        """Seed a still pool (``z < fill_z``) onto ``state``."""
        rho0 = float(
            self.model.initial_density if rho_liquid is None else rho_liquid
        )
        host = seed_pool(self._ensure_gpu().shape, fill_z, rho0)
        self.seed_host_state(state, host)

    def seed_droplet(
        self,
        state: LbmState,
        center: tuple[float, float, float],
        radius: float,
        rho_liquid: float | None = None,
    ) -> None:
        """Seed a spherical droplet (lattice cell-center coords) onto ``state``."""
        rho0 = float(
            self.model.initial_density if rho_liquid is None else rho_liquid
        )
        host = seed_droplet(self._ensure_gpu().shape, center, radius, rho0)
        self.seed_host_state(state, host)

    def seed_host_state(self, state: LbmState, host: HomeVofState) -> None:
        """Upload an arbitrary ``HomeVofState`` IC and sync macros to ``state``."""
        buf = self._ensure_gpu()
        upload_home_vof_state(buf, host)
        self.sync_to_state(state)
        state.f.zero_()

    def refresh_domain_bc(self) -> None:
        """Re-read ``LbmModel`` face BC into GPU buffers (after ``apply_home_domain_bc_to_model``)."""
        self._domain_bc = home_domain_bc_from_model(self.model)
        if self._gpu is not None:
            set_face_bc_gpu(self._gpu, self._domain_bc)

    def _host_from_macros(
        self,
        *,
        rho: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        phi: np.ndarray,
        cell_type: np.ndarray,
    ) -> HomeVofState:
        """Build a host ``HomeVofState`` from macroscopic arrays."""
        return HomeVofState(
            moments=HomeMomentArrays(
                rho=rho,
                ux=ux,
                uy=uy,
                uz=uz,
                sxx=ux * ux,
                syy=uy * uy,
                szz=uz * uz,
                sxy=ux * uy,
                sxz=ux * uz,
                syz=uy * uz,
            ),
            phi=phi,
            cell_type=cell_type,
        )

    def ensure_from_state(self, state: LbmState) -> None:
        """If no GPU state yet, upload macros from ``LbmState``."""
        if self._gpu is not None:
            return
        shape = (int(self.model.nx), int(self.model.ny), int(self.model.nz))
        rho = state.density.numpy().astype(np.float64)
        ux = state.velocity_x.numpy().astype(np.float64)
        uy = state.velocity_y.numpy().astype(np.float64)
        uz = state.velocity_z.numpy().astype(np.float64)

        if not self.free_surface:
            rho0 = float(self.model.initial_density)
            if float(np.max(np.abs(rho))) <= 1.0e-12:
                host = seed_full_liquid(shape, rho0)
            else:
                host = self._host_from_macros(
                    rho=rho,
                    ux=ux,
                    uy=uy,
                    uz=uz,
                    phi=np.ones(shape, dtype=np.float64),
                    cell_type=np.full(shape, CELL_LIQUID, dtype=np.int32),
                )
            upload_home_vof_state(self._ensure_gpu(), host)
            self.sync_to_state(state)
            return

        phi = state.phi.numpy().astype(np.float64)
        ctype = state.cell_type.numpy().astype(np.int32)
        gas = ctype == CELL_GAS
        rho = rho.copy()
        ux = ux.copy()
        uy = uy.copy()
        uz = uz.copy()
        rho[gas] = 0.0
        ux[gas] = 0.0
        uy[gas] = 0.0
        uz[gas] = 0.0
        host = self._host_from_macros(
            rho=rho, ux=ux, uy=uy, uz=uz, phi=phi, cell_type=ctype
        )
        upload_home_vof_state(self._ensure_gpu(), host)

    def sync_to_state(self, state: LbmState) -> None:
        """Copy GPU HOME fields onto ``LbmState`` macros."""
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

    def _home_step_kwargs(self) -> dict:
        """Keyword args for ``step_home_vof_gpu`` (FS policy gated by branch)."""
        fs = self.free_surface
        m = self.model
        kwargs = {
            "tau": float(m.tau),
            "fx": float(m.gravity_x),
            "fy": float(m.gravity_y),
            "fz": float(m.gravity_z),
            "rho_g0": float(m.vof_rho_gas),
            "gamma": float(m.vof_gamma) if fs else 0.0,
            "eps_phi": float(m.vof_epsilon),
            "rho_liquid": float(m.initial_density),
            "kappa_smooth": int(m.vof_kappa_smooth),
            "wall_wetting": float(m.vof_wall_wetting) if fs else 0.0,
            "wall_film_drain": bool(m.vof_wall_film_drain) if fs else False,
            "wall_film_phi_max": float(m.vof_wall_film_phi_max),
            "wall_film_u_max": float(m.vof_wall_film_u_max),
            "wall_film_edge_only": bool(m.vof_wall_film_edge_only),
            "home_fill_empty": bool(m.vof_home_fill_empty) if fs else False,
            "home_wall_eq": bool(m.vof_home_wall_eq),
            "seal_fg": bool(m.vof_seal_fg) if fs else False,
            "bubble_pressure": bool(m.vof_bubble_pressure) if fs else False,
            "bubble_atm_volume": float(m.vof_bubble_atm_volume),
            "bubble_update_every": int(m.vof_bubble_update_every),
            "bubble_disjoint": bool(m.vof_bubble_disjoint) if fs else False,
            "bubble_disjoint_factor": float(m.vof_bubble_disjoint_factor),
            "bubble_small_sigma": bool(m.vof_bubble_small_sigma) if fs else False,
            "bubble_small_vol": float(m.vof_bubble_small_vol),
            "bubble_small_six_sigma": float(m.vof_bubble_small_six_sigma),
            "bubble_eddy": bool(m.vof_bubble_eddy) if fs else False,
            "bubble_eddy_atm_vol": float(m.vof_bubble_eddy_atm_vol),
            "moment_quant": bool(getattr(m, "vof_home_moment_quant", False)),
            "moment_quant_dither": bool(
                getattr(m, "vof_home_moment_quant_dither", True)
            ),
            "use_cuda_graph": bool(getattr(m, "vof_home_cuda_graph", False)),
            "me_enable": False,
        }
        me = self._fused_me
        if me is not None and bool(getattr(m, "vof_home_me_in_fused", False)):
            kwargs["me_enable"] = True
            kwargs["solid_body_id"] = me["solid_body_id"]
            kwargs["body_q"] = me["body_q"]
            kwargs["body_com"] = me["body_com"]
            kwargs["body_f"] = me["body_f"]
            kwargs["me_dh"] = float(me["dh"])
            kwargs["me_force_scale"] = float(me["force_scale"])
        return kwargs

    def step(self, state_out: LbmState, state_in: LbmState | None = None) -> None:
        """Advance one lattice step on GPU; write macros into ``state_out``.

        Same fused HOME operators for base and free-surface; FS policy flags
        are forced off when ``phase_mode='none'``.
        """
        buf = self._ensure_gpu()
        if state_in is not None:
            sync_solids_from_lbm_state(buf, state_in)
        if getattr(self, "_face_bc_ready", False) is False:
            self._domain_bc = home_domain_bc_from_model(self.model)
            set_face_bc_gpu(buf, self._domain_bc)
            self._face_bc_ready = True

        step_home_vof_gpu(buf, **self._home_step_kwargs())
        self._fused_me = None
        if self.free_surface and bool(getattr(self.model, "vof_height_eq", False)):
            every = max(1, int(getattr(self.model, "vof_height_eq_every", 8)))
            self._height_eq_counter += 1
            if self._height_eq_counter % every == 0:
                self._height_eq_stat_pulls = getattr(self, "_height_eq_stat_pulls", 0) + 1
                sync_stats = self._height_eq_stat_pulls % 4 == 1
                self._last_height_eq_stats = self.apply_height_equation(
                    state_out=None, sync_stats=sync_stats
                )
        self.sync_to_state(state_out)
        if state_in is not None:
            wp.copy(state_out.solid_phi, state_in.solid_phi)
            wp.copy(state_out.solid_body_id, state_in.solid_body_id)
            wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
            wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
            wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

    def accumulate_reconstructed_link_me(
        self,
        *,
        solid_body_id: wp.array,
        body_q: wp.array,
        body_com: wp.array,
        body_f: wp.array,
        dh: float,
        force_scale: float = 1.0,
    ) -> None:
        """Opt-in Ladd-style ME from reconstructed wall populations (no live ``f``)."""
        from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.link_me_warp import (
            launch_home_reconstructed_link_me,
        )

        buf = self._ensure_gpu()
        launch_home_reconstructed_link_me(
            buf=buf,
            solid_body_id=solid_body_id,
            body_q=body_q,
            body_com=body_com,
            body_f=body_f,
            dh=float(dh),
            force_scale=float(force_scale),
            home_wall_eq=bool(self.model.vof_home_wall_eq),
        )

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

# Back-compat alias (prefer ``HomeFp32Bridge``).
HomeFp32VofBridge = HomeFp32Bridge

