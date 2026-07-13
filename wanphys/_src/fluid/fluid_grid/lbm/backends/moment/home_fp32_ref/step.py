# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""fp32 HOME reconstruct-stream + collide.

HOME-FREE loop (§4.1): pull ``f_i`` by third-order Hermite reconstruction from
the **upstream** neighbor (Eq. 16), form temporary moments (HOME Eq. 7), then
collide (HOME-FREE Eq. 18–21).

H1: fully periodic ``step_periodic_*``.
H2: ``step_domain_numpy`` with HOME Eq. 24 walls, solid mask, Zou–He faces.

Numpy reference is the source of truth for unit tests; Warp kernel covers the
periodic path. Not wired into ``LbmSolver`` yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
    HomeFaceBC,
    HomeFaceKind,
    face_normal_inward,
    reconstruct_solid_f_i_numpy,
    zou_he_velocity_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.hermite import (
    CS2,
    home_reconstruct_f_i,
    reconstruct_f_i_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.lattice import LatticeSpec, get_lattice_spec
from wanphys._src.fluid.fluid_grid.lbm.core.moments import (
    collide_moments_numpy,
    home_collide_moments,
)


@dataclass
class HomeMomentArrays:
    """SoA moment buffers for one HOME field (host or device-ready numpy)."""

    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    sxx: np.ndarray
    syy: np.ndarray
    szz: np.ndarray
    sxy: np.ndarray
    sxz: np.ndarray
    syz: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rho.shape

    def copy(self) -> HomeMomentArrays:
        return HomeMomentArrays(
            rho=self.rho.copy(),
            ux=self.ux.copy(),
            uy=self.uy.copy(),
            uz=self.uz.copy(),
            sxx=self.sxx.copy(),
            syy=self.syy.copy(),
            szz=self.szz.copy(),
            sxy=self.sxy.copy(),
            sxz=self.sxz.copy(),
            syz=self.syz.copy(),
        )


def _empty_like_field(field: HomeMomentArrays) -> HomeMomentArrays:
    return HomeMomentArrays(
        rho=np.empty_like(field.rho),
        ux=np.empty_like(field.ux),
        uy=np.empty_like(field.uy),
        uz=np.empty_like(field.uz),
        sxx=np.empty_like(field.sxx),
        syy=np.empty_like(field.syy),
        szz=np.empty_like(field.szz),
        sxy=np.empty_like(field.sxy),
        sxz=np.empty_like(field.sxz),
        syz=np.empty_like(field.syz),
    )


def make_uniform_equilibrium(
    shape: tuple[int, int, int],
    rho0: float = 1.0,
    ux0: float = 0.0,
    uy0: float = 0.0,
    uz0: float = 0.0,
) -> HomeMomentArrays:
    """Fill a grid with equilibrium moments ``S = u ⊗ u``."""
    return HomeMomentArrays(
        rho=np.full(shape, rho0, dtype=np.float64),
        ux=np.full(shape, ux0, dtype=np.float64),
        uy=np.full(shape, uy0, dtype=np.float64),
        uz=np.full(shape, uz0, dtype=np.float64),
        sxx=np.full(shape, ux0 * ux0, dtype=np.float64),
        syy=np.full(shape, uy0 * uy0, dtype=np.float64),
        szz=np.full(shape, uz0 * uz0, dtype=np.float64),
        sxy=np.full(shape, ux0 * uy0, dtype=np.float64),
        sxz=np.full(shape, ux0 * uz0, dtype=np.float64),
        syz=np.full(shape, uy0 * uz0, dtype=np.float64),
    )


def step_periodic_numpy(
    field: HomeMomentArrays,
    lattice: LatticeSpec | str = "D3Q27",
    tau: float = 0.6,
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
    out: HomeMomentArrays | None = None,
) -> HomeMomentArrays:
    """One HOME step on a fully periodic box (numpy reference)."""
    return step_domain_numpy(
        field,
        lattice=lattice,
        tau=tau,
        fx=fx,
        fy=fy,
        fz=fz,
        domain_bc=HomeDomainBC(),
        out=out,
    )


def _resolve_pull_neighbor(
    i: int,
    j: int,
    k: int,
    cxi: int,
    cyi: int,
    czi: int,
    nx: int,
    ny: int,
    nz: int,
    domain_bc: HomeDomainBC,
    solid: np.ndarray | None,
) -> tuple[str, object]:
    """Classify upstream sample: ('fluid', (ni,nj,nk)) | ('wall', FaceBC) | ('zh_unknown', FaceBC)."""
    ni = i - cxi
    nj = j - cyi
    nk = k - czi

    # Axis-aligned OOB: first hit decides the face (HOME voxel walls).
    if ni < 0:
        face = domain_bc.xmin
        if face.kind == HomeFaceKind.PERIODIC:
            ni = nx - 1
        elif face.kind == HomeFaceKind.WALL:
            return ("wall", face)
        else:
            return ("zh_unknown", face)
    elif ni >= nx:
        face = domain_bc.xmax
        if face.kind == HomeFaceKind.PERIODIC:
            ni = 0
        elif face.kind == HomeFaceKind.WALL:
            return ("wall", face)
        else:
            return ("zh_unknown", face)

    if nj < 0:
        face = domain_bc.ymin
        if face.kind == HomeFaceKind.PERIODIC:
            nj = ny - 1
        elif face.kind == HomeFaceKind.WALL:
            return ("wall", face)
        else:
            return ("zh_unknown", face)
    elif nj >= ny:
        face = domain_bc.ymax
        if face.kind == HomeFaceKind.PERIODIC:
            nj = 0
        elif face.kind == HomeFaceKind.WALL:
            return ("wall", face)
        else:
            return ("zh_unknown", face)

    if nk < 0:
        face = domain_bc.zmin
        if face.kind == HomeFaceKind.PERIODIC:
            nk = nz - 1
        elif face.kind == HomeFaceKind.WALL:
            return ("wall", face)
        else:
            return ("zh_unknown", face)
    elif nk >= nz:
        face = domain_bc.zmax
        if face.kind == HomeFaceKind.PERIODIC:
            nk = 0
        elif face.kind == HomeFaceKind.WALL:
            return ("wall", face)
        else:
            return ("zh_unknown", face)

    if solid is not None and bool(solid[ni, nj, nk]):
        return ("wall", HomeFaceBC(kind=HomeFaceKind.WALL))
    return ("fluid", (ni, nj, nk))


def _zh_face_for_cell(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    domain_bc: HomeDomainBC,
) -> tuple[str, HomeFaceBC] | None:
    """Primary Zou–He face for a boundary cell (skip edges/corners)."""
    on_xmin = i == 0 and domain_bc.xmin.kind == HomeFaceKind.ZOU_HE
    on_xmax = i == nx - 1 and domain_bc.xmax.kind == HomeFaceKind.ZOU_HE
    on_ymin = j == 0 and domain_bc.ymin.kind == HomeFaceKind.ZOU_HE
    on_ymax = j == ny - 1 and domain_bc.ymax.kind == HomeFaceKind.ZOU_HE
    on_zmin = k == 0 and domain_bc.zmin.kind == HomeFaceKind.ZOU_HE
    on_zmax = k == nz - 1 and domain_bc.zmax.kind == HomeFaceKind.ZOU_HE
    flags = (on_xmin, on_xmax, on_ymin, on_ymax, on_zmin, on_zmax)
    if sum(flags) != 1:
        return None
    if on_xmin:
        return ("xmin", domain_bc.xmin)
    if on_xmax:
        return ("xmax", domain_bc.xmax)
    if on_ymin:
        return ("ymin", domain_bc.ymin)
    if on_ymax:
        return ("ymax", domain_bc.ymax)
    if on_zmin:
        return ("zmin", domain_bc.zmin)
    return ("zmax", domain_bc.zmax)


def step_domain_numpy(
    field: HomeMomentArrays,
    lattice: LatticeSpec | str = "D3Q27",
    tau: float = 0.6,
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
    domain_bc: HomeDomainBC | None = None,
    solid: np.ndarray | None = None,
    out: HomeMomentArrays | None = None,
) -> HomeMomentArrays:
    """One HOME step with domain faces + optional interior solid mask.

    Walls use HOME Eq. 24 reconstruct on cut links. Zou–He faces fill unknown
    populations after the pull, then moments + collide.
    """
    spec = get_lattice_spec(lattice) if isinstance(lattice, str) else lattice
    bc = domain_bc if domain_bc is not None else HomeDomainBC()
    cx = np.asarray(spec.cx, dtype=np.int32)
    cy = np.asarray(spec.cy, dtype=np.int32)
    cz = np.asarray(spec.cz, dtype=np.int32)
    w = np.asarray(spec.weights, dtype=np.float64)
    q = spec.num_dirs
    nx, ny, nz = field.shape
    if out is None:
        out = _empty_like_field(field)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if solid is not None and bool(solid[i, j, k]):
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

                f_pop = np.zeros(q, dtype=np.float64)
                known = np.ones(q, dtype=bool)
                need_zh = False
                for d in range(q):
                    kind, payload = _resolve_pull_neighbor(
                        i, j, k,
                        int(cx[d]), int(cy[d]), int(cz[d]),
                        nx, ny, nz, bc, solid,
                    )
                    cxf, cyf, czf = float(cx[d]), float(cy[d]), float(cz[d])
                    wf = float(w[d])
                    if kind == "fluid":
                        ni, nj, nk = payload  # type: ignore[misc]
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
                            cxf, cyf, czf, wf,
                        )
                    elif kind == "wall":
                        face = payload  # type: ignore[assignment]
                        assert isinstance(face, HomeFaceBC)
                        f_pop[d] = reconstruct_solid_f_i_numpy(
                            float(field.rho[i, j, k]),
                            float(field.ux[i, j, k]),
                            float(field.uy[i, j, k]),
                            float(field.uz[i, j, k]),
                            float(field.sxx[i, j, k]),
                            float(field.syy[i, j, k]),
                            float(field.szz[i, j, k]),
                            float(field.sxy[i, j, k]),
                            float(field.sxz[i, j, k]),
                            float(field.syz[i, j, k]),
                            face.ux, face.uy, face.uz,
                            cxf, cyf, czf, wf,
                        )
                    else:
                        need_zh = True
                        known[d] = False
                        f_pop[d] = 0.0

                if need_zh:
                    zh = _zh_face_for_cell(i, j, k, nx, ny, nz, bc)
                    if zh is not None:
                        face_name, face_bc = zh
                        nin = face_normal_inward(face_name)
                        f_pop = zou_he_velocity_numpy(
                            f_pop, spec, nin[0], nin[1], nin[2],
                            face_bc.ux, face_bc.uy, face_bc.uz,
                        )
                    else:
                        for d in range(q):
                            if not known[d]:
                                f_pop[d] = reconstruct_solid_f_i_numpy(
                                    float(field.rho[i, j, k]),
                                    float(field.ux[i, j, k]),
                                    float(field.uy[i, j, k]),
                                    float(field.uz[i, j, k]),
                                    float(field.sxx[i, j, k]),
                                    float(field.syy[i, j, k]),
                                    float(field.szz[i, j, k]),
                                    float(field.sxy[i, j, k]),
                                    float(field.sxz[i, j, k]),
                                    float(field.syz[i, j, k]),
                                    0.0, 0.0, 0.0,
                                    float(cx[d]), float(cy[d]), float(cz[d]),
                                    float(w[d]),
                                )

                r = float(np.sum(f_pop))
                if r <= 1.0e-12:
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

                mx = float(np.dot(cx.astype(np.float64), f_pop))
                my = float(np.dot(cy.astype(np.float64), f_pop))
                mz = float(np.dot(cz.astype(np.float64), f_pop))
                cxf = cx.astype(np.float64)
                cyf = cy.astype(np.float64)
                czf = cz.astype(np.float64)
                axx = float(np.dot(cxf * cxf - CS2, f_pop))
                ayy = float(np.dot(cyf * cyf - CS2, f_pop))
                azz = float(np.dot(czf * czf - CS2, f_pop))
                axy = float(np.dot(cxf * cyf, f_pop))
                axz = float(np.dot(cxf * czf, f_pop))
                ayz = float(np.dot(cyf * czf, f_pop))
                inv = 1.0 / r
                m = collide_moments_numpy(
                    r, mx * inv, my * inv, mz * inv,
                    axx * inv, ayy * inv, azz * inv,
                    axy * inv, axz * inv, ayz * inv,
                    tau, fx, fy, fz,
                )
                out.rho[i, j, k] = m.rho
                out.ux[i, j, k] = m.ux
                out.uy[i, j, k] = m.uy
                out.uz[i, j, k] = m.uz
                out.sxx[i, j, k] = m.sxx
                out.syy[i, j, k] = m.syy
                out.szz[i, j, k] = m.szz
                out.sxy[i, j, k] = m.sxy
                out.sxz[i, j, k] = m.sxz
                out.syz[i, j, k] = m.syz
    return out


@wp.func
def _wrap_periodic(i: int, n: int) -> int:
    if i < 0:
        return i + n
    if i >= n:
        return i - n
    return i


@wp.kernel
def home_stream_collide_periodic_kernel(
    rho_in: wp.array3d(dtype=float),
    ux_in: wp.array3d(dtype=float),
    uy_in: wp.array3d(dtype=float),
    uz_in: wp.array3d(dtype=float),
    sxx_in: wp.array3d(dtype=float),
    syy_in: wp.array3d(dtype=float),
    szz_in: wp.array3d(dtype=float),
    sxy_in: wp.array3d(dtype=float),
    sxz_in: wp.array3d(dtype=float),
    syz_in: wp.array3d(dtype=float),
    rho_out: wp.array3d(dtype=float),
    ux_out: wp.array3d(dtype=float),
    uy_out: wp.array3d(dtype=float),
    uz_out: wp.array3d(dtype=float),
    sxx_out: wp.array3d(dtype=float),
    syy_out: wp.array3d(dtype=float),
    szz_out: wp.array3d(dtype=float),
    sxy_out: wp.array3d(dtype=float),
    sxz_out: wp.array3d(dtype=float),
    syz_out: wp.array3d(dtype=float),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=float),
    num_dirs: int,
    tau: float,
    fx: float,
    fy: float,
    fz: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Periodic HOME: pull reconstruct-stream then moment collide."""
    i, j, k = wp.tid()
    cs2 = 1.0 / 3.0
    r = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)
    axx = float(0.0)
    ayy = float(0.0)
    azz = float(0.0)
    axy = float(0.0)
    axz = float(0.0)
    ayz = float(0.0)

    for d in range(num_dirs):
        cxi = cx_arr[d]
        cyi = cy_arr[d]
        czi = cz_arr[d]
        ni = _wrap_periodic(i - cxi, nx)
        nj = _wrap_periodic(j - cyi, ny)
        nk = _wrap_periodic(k - czi, nz)
        fi = home_reconstruct_f_i(
            rho_in[ni, nj, nk],
            ux_in[ni, nj, nk],
            uy_in[ni, nj, nk],
            uz_in[ni, nj, nk],
            sxx_in[ni, nj, nk],
            syy_in[ni, nj, nk],
            szz_in[ni, nj, nk],
            sxy_in[ni, nj, nk],
            sxz_in[ni, nj, nk],
            syz_in[ni, nj, nk],
            cxi,
            cyi,
            czi,
            w_arr[d],
        )
        fcx = float(cxi)
        fcy = float(cyi)
        fcz = float(czi)
        r = r + fi
        mx = mx + fcx * fi
        my = my + fcy * fi
        mz = mz + fcz * fi
        axx = axx + (fcx * fcx - cs2) * fi
        ayy = ayy + (fcy * fcy - cs2) * fi
        azz = azz + (fcz * fcz - cs2) * fi
        axy = axy + fcx * fcy * fi
        axz = axz + fcx * fcz * fi
        ayz = ayz + fcy * fcz * fi

    inv = 1.0 / r
    rho, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz = home_collide_moments(
        r,
        mx * inv,
        my * inv,
        mz * inv,
        axx * inv,
        ayy * inv,
        azz * inv,
        axy * inv,
        axz * inv,
        ayz * inv,
        tau,
        fx,
        fy,
        fz,
    )
    rho_out[i, j, k] = rho
    ux_out[i, j, k] = ux
    uy_out[i, j, k] = uy
    uz_out[i, j, k] = uz
    sxx_out[i, j, k] = sxx
    syy_out[i, j, k] = syy
    szz_out[i, j, k] = szz
    sxy_out[i, j, k] = sxy
    sxz_out[i, j, k] = sxz
    syz_out[i, j, k] = syz


def step_periodic_warp(
    field: HomeMomentArrays,
    lattice: LatticeSpec | str = "D3Q27",
    tau: float = 0.6,
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
    device: str | None = None,
) -> HomeMomentArrays:
    """One HOME periodic step on GPU/CPU via Warp (returns host numpy arrays)."""
    spec = get_lattice_spec(lattice) if isinstance(lattice, str) else lattice
    nx, ny, nz = field.shape
    dev = device or "cuda:0"

    def _to_wp(a: np.ndarray) -> wp.array:
        return wp.array(a.astype(np.float32), dtype=float, device=dev)

    rho_in = _to_wp(field.rho)
    ux_in = _to_wp(field.ux)
    uy_in = _to_wp(field.uy)
    uz_in = _to_wp(field.uz)
    sxx_in = _to_wp(field.sxx)
    syy_in = _to_wp(field.syy)
    szz_in = _to_wp(field.szz)
    sxy_in = _to_wp(field.sxy)
    sxz_in = _to_wp(field.sxz)
    syz_in = _to_wp(field.syz)

    rho_out = wp.zeros_like(rho_in)
    ux_out = wp.zeros_like(ux_in)
    uy_out = wp.zeros_like(uy_in)
    uz_out = wp.zeros_like(uz_in)
    sxx_out = wp.zeros_like(sxx_in)
    syy_out = wp.zeros_like(syy_in)
    szz_out = wp.zeros_like(szz_in)
    sxy_out = wp.zeros_like(sxy_in)
    sxz_out = wp.zeros_like(sxz_in)
    syz_out = wp.zeros_like(syz_in)

    cx_arr = wp.array(np.asarray(spec.cx, dtype=np.int32), dtype=wp.int32, device=dev)
    cy_arr = wp.array(np.asarray(spec.cy, dtype=np.int32), dtype=wp.int32, device=dev)
    cz_arr = wp.array(np.asarray(spec.cz, dtype=np.int32), dtype=wp.int32, device=dev)
    w_arr = wp.array(np.asarray(spec.weights, dtype=np.float32), dtype=float, device=dev)

    wp.launch(
        home_stream_collide_periodic_kernel,
        dim=(nx, ny, nz),
        inputs=[
            rho_in, ux_in, uy_in, uz_in,
            sxx_in, syy_in, szz_in, sxy_in, sxz_in, syz_in,
            rho_out, ux_out, uy_out, uz_out,
            sxx_out, syy_out, szz_out, sxy_out, sxz_out, syz_out,
            cx_arr, cy_arr, cz_arr, w_arr,
            spec.num_dirs, float(tau), float(fx), float(fy), float(fz),
            nx, ny, nz,
        ],
        device=dev,
    )
    wp.synchronize()

    return HomeMomentArrays(
        rho=rho_out.numpy().astype(np.float64),
        ux=ux_out.numpy().astype(np.float64),
        uy=uy_out.numpy().astype(np.float64),
        uz=uz_out.numpy().astype(np.float64),
        sxx=sxx_out.numpy().astype(np.float64),
        syy=syy_out.numpy().astype(np.float64),
        szz=szz_out.numpy().astype(np.float64),
        sxy=sxy_out.numpy().astype(np.float64),
        sxz=sxz_out.numpy().astype(np.float64),
        syz=syz_out.numpy().astype(np.float64),
    )
