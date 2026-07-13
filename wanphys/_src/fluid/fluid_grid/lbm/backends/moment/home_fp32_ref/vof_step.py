# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Moment-encoded HOME-FREE VOF stepper (H4, fp32, no quant).

Stores only ``(ρ,u,S)`` + ``φ`` + cell flags. One step mirrors HOME-FREE Alg. 1
fluid core (without bubbles / cut-cell / foam)::

  1. Körner mass exchange (Eq. 9–10) using Hermite-reconstructed ``f``
  2. Reconstruct-stream with gas→interface Eq. 11 using ``\\bar f_ī`` (Eq. 16)
  3. HOME moment collide (Eq. 18–21)
  4. Fill/empty reclassification + closed interface layer

Not wired into ``LbmSolver`` yet — distribution VOF remains the default path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
    HomeFaceBC,
    HomeFaceKind,
    reconstruct_solid_f_i_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.step import (
    HomeMomentArrays,
    _empty_like_field,
    _resolve_pull_neighbor,
)
from wanphys._src.fluid.fluid_grid.lbm.core.hermite import (
    CS2,
    reconstruct_f_i_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.lattice import LatticeSpec, get_lattice_spec
from wanphys._src.fluid.fluid_grid.lbm.core.moments import collide_moments_numpy

CELL_GAS: int = 0
CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2


@dataclass
class HomeVofState:
    """Moment-encoded free-surface field."""

    moments: HomeMomentArrays
    phi: np.ndarray
    cell_type: np.ndarray  # int32, CELL_*

    @property
    def shape(self) -> tuple[int, ...]:
        return self.phi.shape

    def copy(self) -> HomeVofState:
        return HomeVofState(
            moments=self.moments.copy(),
            phi=self.phi.copy(),
            cell_type=self.cell_type.copy(),
        )


def _feq(w: float, rho: float, ux: float, uy: float, uz: float,
         cx: float, cy: float, cz: float) -> float:
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux * ux + uy * uy + uz * uz
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def _fs_bar_f(
    rho_g: float,
    rho_c: float,
    ux: float,
    uy: float,
    uz: float,
    sxx: float,
    syy: float,
    szz: float,
    sxy: float,
    sxz: float,
    syz: float,
    w: float,
    cx: float,
    cy: float,
    cz: float,
) -> float:
    """HOME-FREE Eq. 11 with filtered ``\\bar f_ī``."""
    u2 = ux * ux + uy * uy + uz * uz
    if u2 > 0.01:
        s = 0.1 / np.sqrt(u2)
        ux, uy, uz = ux * s, uy * s, uz * s
    feq = _feq(w, rho_g, ux, uy, uz, cx, cy, cz)
    feq_opp = _feq(w, rho_g, ux, uy, uz, -cx, -cy, -cz)
    bar_opp = reconstruct_f_i_numpy(
        rho_c, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz,
        -cx, -cy, -cz, w,
    )
    val = feq + feq_opp - bar_opp
    return max(val, 0.0)


def seed_dam_break_column(
    shape: tuple[int, int, int],
    dam_x: int,
    fill_z: int,
    rho_liquid: float = 1.0,
) -> HomeVofState:
    """Liquid column for x < dam_x and z < fill_z; mark free-surface interface."""
    nx, ny, nz = shape
    moments = HomeMomentArrays(
        rho=np.zeros(shape, dtype=np.float64),
        ux=np.zeros(shape, dtype=np.float64),
        uy=np.zeros(shape, dtype=np.float64),
        uz=np.zeros(shape, dtype=np.float64),
        sxx=np.zeros(shape, dtype=np.float64),
        syy=np.zeros(shape, dtype=np.float64),
        szz=np.zeros(shape, dtype=np.float64),
        sxy=np.zeros(shape, dtype=np.float64),
        sxz=np.zeros(shape, dtype=np.float64),
        syz=np.zeros(shape, dtype=np.float64),
    )
    phi = np.zeros(shape, dtype=np.float64)
    cell_type = np.zeros(shape, dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i < dam_x and k < fill_z:
                    moments.rho[i, j, k] = rho_liquid
                    phi[i, j, k] = 1.0
                    cell_type[i, j, k] = CELL_LIQUID
    # Mark liquid cells that touch gas as interface
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if cell_type[i, j, k] != CELL_LIQUID:
                    continue
                for di, dj, dk in (
                    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                ):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz:
                        continue  # walls are not free surface
                    if cell_type[ni, nj, nk] == CELL_GAS:
                        cell_type[i, j, k] = CELL_INTERFACE
                        break
    # Interface fill level starts at 0.5 (mass = φρ); φ=1 would immediately trip IF.
    phi[cell_type == CELL_INTERFACE] = 0.5
    return HomeVofState(moments=moments, phi=phi, cell_type=cell_type)


def _update_phi_korner(
    state: HomeVofState,
    spec: LatticeSpec,
    domain_bc: HomeDomainBC,
) -> np.ndarray:
    """Eq. 9–10 using Hermite-reconstructed populations from moments."""
    del domain_bc  # walls skip via OOB continue; periodic faces not used here
    cx = np.asarray(spec.cx, dtype=np.int32)
    cy = np.asarray(spec.cy, dtype=np.int32)
    cz = np.asarray(spec.cz, dtype=np.int32)
    w = np.asarray(spec.weights, dtype=np.float64)
    opp = np.asarray(spec.opposite, dtype=np.int32)
    q = spec.num_dirs
    nx, ny, nz = state.shape
    m = state.moments
    phi_out = state.phi.copy()

    def recon(ii: int, jj: int, kk: int, d: int) -> float:
        return reconstruct_f_i_numpy(
            float(m.rho[ii, jj, kk]),
            float(m.ux[ii, jj, kk]),
            float(m.uy[ii, jj, kk]),
            float(m.uz[ii, jj, kk]),
            float(m.sxx[ii, jj, kk]),
            float(m.syy[ii, jj, kk]),
            float(m.szz[ii, jj, kk]),
            float(m.sxy[ii, jj, kk]),
            float(m.sxz[ii, jj, kk]),
            float(m.syz[ii, jj, kk]),
            float(cx[d]), float(cy[d]), float(cz[d]), float(w[d]),
        )

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if int(state.cell_type[i, j, k]) != CELL_INTERFACE:
                    continue
                rho_c = float(m.rho[i, j, k])
                if rho_c < 0.1:
                    continue
                phi_c = float(state.phi[i, j, k])
                dm = 0.0
                for d in range(1, q):
                    ni = i + int(cx[d])
                    nj = j + int(cy[d])
                    nk = k + int(cz[d])
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                        continue
                    ntype = int(state.cell_type[ni, nj, nk])
                    if ntype == CELL_GAS:
                        continue
                    phi_n = 1.0 if ntype == CELL_LIQUID else float(state.phi[ni, nj, nk])
                    theta = 0.5 * (phi_c + phi_n)
                    od = int(opp[d])
                    f_opp_nb = recon(ni, nj, nk, od)
                    f_self = recon(i, j, k, d)
                    dm += theta * (f_opp_nb - f_self)
                phi_out[i, j, k] = phi_c + dm / rho_c
    return phi_out


def _reclassify(
    phi: np.ndarray,
    cell_type: np.ndarray,
    moments: HomeMomentArrays,
    eps: float,
    rho_liquid: float,
) -> tuple[np.ndarray, np.ndarray, HomeMomentArrays]:
    """Fill/empty + keep a closed interface layer (simplified Lehmann)."""
    nx, ny, nz = phi.shape
    phi2 = phi.copy()
    ct = cell_type.copy()
    m = moments.copy()

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if int(ct[i, j, k]) != CELL_INTERFACE:
                    continue
                p = float(phi2[i, j, k])
                if p >= 1.0 + eps:
                    phi2[i, j, k] = 1.0
                    ct[i, j, k] = CELL_LIQUID
                elif p <= 0.0 - eps:
                    phi2[i, j, k] = 0.0
                    ct[i, j, k] = CELL_GAS
                    m.rho[i, j, k] = 0.0
                    m.ux[i, j, k] = 0.0
                    m.uy[i, j, k] = 0.0
                    m.uz[i, j, k] = 0.0
                    m.sxx[i, j, k] = 0.0
                    m.syy[i, j, k] = 0.0
                    m.szz[i, j, k] = 0.0
                    m.sxy[i, j, k] = 0.0
                    m.sxz[i, j, k] = 0.0
                    m.syz[i, j, k] = 0.0
                else:
                    phi2[i, j, k] = float(np.clip(p, 0.0, 1.0))

    # Liquid touching gas → interface; gas touching liquid → interface
    ct_new = ct.copy()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                t = int(ct[i, j, k])
                if t == CELL_GAS:
                    for di, dj, dk in (
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                    ):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            if int(ct[ni, nj, nk]) == CELL_LIQUID:
                                ct_new[i, j, k] = CELL_INTERFACE
                                if phi2[i, j, k] <= 0.0:
                                    phi2[i, j, k] = eps
                                if m.rho[i, j, k] <= 0.0:
                                    m.rho[i, j, k] = rho_liquid
                                break
                elif t == CELL_LIQUID:
                    for di, dj, dk in (
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                    ):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            if int(ct[ni, nj, nk]) == CELL_GAS:
                                ct_new[i, j, k] = CELL_INTERFACE
                                break
                        else:
                            ct_new[i, j, k] = CELL_INTERFACE
                            break
    return phi2, ct_new, m


def _stream_collide_vof(
    state: HomeVofState,
    spec: LatticeSpec,
    tau: float,
    fx: float,
    fy: float,
    fz: float,
    domain_bc: HomeDomainBC,
    rho_g0: float,
    gamma: float,
    kappa: np.ndarray | None,
) -> HomeMomentArrays:
    """Pull reconstruct-stream + FS bar_f + moment collide for L/I cells."""
    cx = np.asarray(spec.cx, dtype=np.int32)
    cy = np.asarray(spec.cy, dtype=np.int32)
    cz = np.asarray(spec.cz, dtype=np.int32)
    w = np.asarray(spec.weights, dtype=np.float64)
    q = spec.num_dirs
    nx, ny, nz = state.shape
    field = state.moments
    out = _empty_like_field(field)
    kap = kappa if kappa is not None else np.zeros(state.shape, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ctype = int(state.cell_type[i, j, k])
                if ctype == CELL_GAS:
                    out.rho[i, j, k] = 0.0
                    out.ux[i, j, k] = 0.0
                    out.uy[i, j, k] = 0.0
                    out.uz[i, j, k] = 0.0
                    out.sxx[i, j, k] = 0.0
                    out.syy[i, j, k] = 0.0
                    out.szz[i, j, k] = 0.0
                    out.sxy[i, j, k] = 0.0
                    out.sxz[i, j, k] = 0.0
                    out.syz[i, j, k] = 0.0
                    continue

                rho_c = float(field.rho[i, j, k])
                if rho_c < 0.2:
                    rho_c = 0.2
                ux = float(field.ux[i, j, k])
                uy = float(field.uy[i, j, k])
                uz = float(field.uz[i, j, k])
                sxx = float(field.sxx[i, j, k])
                syy = float(field.syy[i, j, k])
                szz = float(field.szz[i, j, k])
                sxy = float(field.sxy[i, j, k])
                sxz = float(field.sxz[i, j, k])
                syz = float(field.syz[i, j, k])

                rho_g = rho_g0 - 6.0 * gamma * float(kap[i, j, k])
                rho_g = float(np.clip(rho_g, 0.2, 1.8))

                f_pop = np.zeros(q, dtype=np.float64)
                for d in range(q):
                    cxi, cyi, czi = int(cx[d]), int(cy[d]), int(cz[d])
                    wf = float(w[d])
                    kind, payload = _resolve_pull_neighbor(
                        i, j, k, cxi, cyi, czi, nx, ny, nz, domain_bc, None,
                    )
                    if kind == "wall":
                        face = payload
                        assert isinstance(face, HomeFaceBC)
                        f_pop[d] = reconstruct_solid_f_i_numpy(
                            rho_c, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz,
                            face.ux, face.uy, face.uz,
                            float(cxi), float(cyi), float(czi), wf,
                        )
                    elif kind == "zh_unknown":
                        f_pop[d] = reconstruct_solid_f_i_numpy(
                            rho_c, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz,
                            0.0, 0.0, 0.0,
                            float(cxi), float(cyi), float(czi), wf,
                        )
                    else:
                        ni, nj, nk = payload  # type: ignore[misc]
                        ntype = int(state.cell_type[ni, nj, nk])
                        if ntype == CELL_GAS:
                            f_pop[d] = _fs_bar_f(
                                rho_g, rho_c, ux, uy, uz,
                                sxx, syy, szz, sxy, sxz, syz,
                                wf, float(cxi), float(cyi), float(czi),
                            )
                        else:
                            f_pop[d] = reconstruct_f_i_numpy(
                                float(field.rho[ni, nj, nk]),
                                float(field.ux[ni, nj, nk]),
                                float(field.uy[ni, nj, nk]),
                                float(field.uz[ni, nj, nk]),
                                float(field.sxx[ni, nj, nk]),
                                float(field.syy[ni, nj, nk]),
                                float(field.szz[ni, nj, nk]),
                                float(field.sxy[ni, nj, nk]),
                                float(field.sxz[ni, nj, nk]),
                                float(field.syz[ni, nj, nk]),
                                float(cxi), float(cyi), float(czi), wf,
                            )

                r = float(np.sum(f_pop))
                if r <= 1.0e-12:
                    continue
                cxf = cx.astype(np.float64)
                cyf = cy.astype(np.float64)
                czf = cz.astype(np.float64)
                inv = 1.0 / r
                mom = collide_moments_numpy(
                    r,
                    float(np.dot(cxf, f_pop)) * inv,
                    float(np.dot(cyf, f_pop)) * inv,
                    float(np.dot(czf, f_pop)) * inv,
                    float(np.dot(cxf * cxf - CS2, f_pop)) * inv,
                    float(np.dot(cyf * cyf - CS2, f_pop)) * inv,
                    float(np.dot(czf * czf - CS2, f_pop)) * inv,
                    float(np.dot(cxf * cyf, f_pop)) * inv,
                    float(np.dot(cxf * czf, f_pop)) * inv,
                    float(np.dot(cyf * czf, f_pop)) * inv,
                    tau, fx, fy, fz,
                )
                out.rho[i, j, k] = mom.rho
                out.ux[i, j, k] = mom.ux
                out.uy[i, j, k] = mom.uy
                out.uz[i, j, k] = mom.uz
                out.sxx[i, j, k] = mom.sxx
                out.syy[i, j, k] = mom.syy
                out.szz[i, j, k] = mom.szz
                out.sxy[i, j, k] = mom.sxy
                out.sxz[i, j, k] = mom.sxz
                out.syz[i, j, k] = mom.syz
    return out


def step_home_vof_numpy(
    state: HomeVofState,
    lattice: LatticeSpec | str = "D3Q27",
    tau: float = 0.6,
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
    domain_bc: HomeDomainBC | None = None,
    rho_g0: float = 1.0,
    gamma: float = 0.0,
    kappa: np.ndarray | None = None,
    eps_phi: float = 1.0e-4,
    rho_liquid: float = 1.0,
) -> HomeVofState:
    """One HOME-FREE VOF step (moments only, numpy reference)."""
    spec = get_lattice_spec(lattice) if isinstance(lattice, str) else lattice
    bc = domain_bc if domain_bc is not None else HomeDomainBC.all_walls()

    # 1) Mass exchange from reconstructed f (pre-stream moments)
    phi_new = _update_phi_korner(state, spec, bc)

    # 2) Stream + FS + collide
    moments_new = _stream_collide_vof(
        state, spec, tau, fx, fy, fz, bc, rho_g0, gamma, kappa,
    )

    # 3) Reclassify on updated φ (with post-collide moments)
    phi2, ct2, m2 = _reclassify(phi_new, state.cell_type, moments_new, eps_phi, rho_liquid)
    return HomeVofState(moments=m2, phi=phi2, cell_type=ct2)
