# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME domain / solid boundary helpers (H2).

Solid links follow HOME-LBM §4.4 (Eq. 24)::

    ρ^p = ρ^x
    u^p = u_wall
    S^p_αβ = u^p_α u^p_β + (S^x_αβ − u^x_α u^x_β)

then ``f_i^p`` via third-order Hermite (Eq. 17).

Open faces use NEBB Zou–He on the reconstructed populations (same density /
unknown formula as ``kernels_q._zou_he_velocity_face``), then moments are
re-extracted before collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.core.hermite import reconstruct_f_i_numpy
from wanphys._src.fluid.fluid_grid.lbm.core.lattice import LatticeSpec


class HomeFaceKind(IntEnum):
    PERIODIC = 0
    WALL = 1  # HOME solid reconstruct (Eq. 24), includes moving lid
    ZOU_HE = 2  # NEBB velocity inlet / prescribed-u face


@dataclass(frozen=True)
class HomeFaceBC:
    """Per-face boundary condition for the HOME fp32 reference stepper."""

    kind: HomeFaceKind = HomeFaceKind.PERIODIC
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0


@dataclass(frozen=True)
class HomeDomainBC:
    """Six-face BC set. Default: fully periodic."""

    xmin: HomeFaceBC = HomeFaceBC()
    xmax: HomeFaceBC = HomeFaceBC()
    ymin: HomeFaceBC = HomeFaceBC()
    ymax: HomeFaceBC = HomeFaceBC()
    zmin: HomeFaceBC = HomeFaceBC()
    zmax: HomeFaceBC = HomeFaceBC()

    @staticmethod
    def all_walls(
        *,
        lid_uy: float = 0.0,
        lid_face: str = "ymax",
        lid_ux: float = 0.0,
        lid_uz: float = 0.0,
    ) -> HomeDomainBC:
        """Closed box; optional moving lid on one face (HOME Eq. 24)."""
        wall = HomeFaceBC(kind=HomeFaceKind.WALL)
        faces = {
            "xmin": wall,
            "xmax": wall,
            "ymin": wall,
            "ymax": wall,
            "zmin": wall,
            "zmax": wall,
        }
        faces[lid_face] = HomeFaceBC(
            kind=HomeFaceKind.WALL, ux=lid_ux, uy=lid_uy, uz=lid_uz,
        )
        return HomeDomainBC(**faces)

    @staticmethod
    def channel_x(
        u_in: float,
        *,
        periodic_y: bool = True,
        periodic_z: bool = True,
    ) -> HomeDomainBC:
        """Zou–He inlet at xmin, Zou–He (same u) or wall at xmax; side walls."""
        side = (
            HomeFaceBC(kind=HomeFaceKind.PERIODIC)
            if periodic_y
            else HomeFaceBC(kind=HomeFaceKind.WALL)
        )
        side_z = (
            HomeFaceBC(kind=HomeFaceKind.PERIODIC)
            if periodic_z
            else HomeFaceBC(kind=HomeFaceKind.WALL)
        )
        inlet = HomeFaceBC(kind=HomeFaceKind.ZOU_HE, ux=u_in)
        # Outlet: convective not yet; use Zou–He with same bulk u as soft outflow
        outlet = HomeFaceBC(kind=HomeFaceKind.ZOU_HE, ux=u_in)
        return HomeDomainBC(
            xmin=inlet,
            xmax=outlet,
            ymin=side,
            ymax=side,
            zmin=side_z,
            zmax=side_z,
        )


def solid_moments_eq24(
    rho_x: float,
    ux_x: float,
    uy_x: float,
    uz_x: float,
    sxx_x: float,
    syy_x: float,
    szz_x: float,
    sxy_x: float,
    sxz_x: float,
    syz_x: float,
    ux_p: float,
    uy_p: float,
    uz_p: float,
) -> tuple[float, ...]:
    """HOME Eq. 24: (ρ^p, u^p, S^p) from fluid cell + wall velocity."""
    sxx_p = ux_p * ux_p + (sxx_x - ux_x * ux_x)
    syy_p = uy_p * uy_p + (syy_x - uy_x * uy_x)
    szz_p = uz_p * uz_p + (szz_x - uz_x * uz_x)
    sxy_p = ux_p * uy_p + (sxy_x - ux_x * uy_x)
    sxz_p = ux_p * uz_p + (sxz_x - ux_x * uz_x)
    syz_p = uy_p * uz_p + (syz_x - uy_x * uz_x)
    return (
        rho_x, ux_p, uy_p, uz_p,
        sxx_p, syy_p, szz_p, sxy_p, sxz_p, syz_p,
    )


def reconstruct_solid_f_i_numpy(
    rho_x: float,
    ux_x: float,
    uy_x: float,
    uz_x: float,
    sxx_x: float,
    syy_x: float,
    szz_x: float,
    sxy_x: float,
    sxz_x: float,
    syz_x: float,
    ux_p: float,
    uy_p: float,
    uz_p: float,
    cx: float,
    cy: float,
    cz: float,
    w: float,
) -> float:
    """Reconstruct f_i at solid intersection (HOME Eq. 24 + Eq. 17)."""
    m = solid_moments_eq24(
        rho_x, ux_x, uy_x, uz_x,
        sxx_x, syy_x, szz_x, sxy_x, sxz_x, syz_x,
        ux_p, uy_p, uz_p,
    )
    return reconstruct_f_i_numpy(*m, cx, cy, cz, w)


def zou_he_velocity_numpy(
    f: np.ndarray,
    spec: LatticeSpec,
    nx_in: int,
    ny_in: int,
    nz_in: int,
    vx: float,
    vy: float,
    vz: float,
) -> np.ndarray:
    """NEBB Zou–He velocity BC on a length-Q population vector (in-place copy).

    Inward lattice normal ``(nx_in, ny_in, nz_in)`` points into the fluid
    (e.g. xmin face → +x = (1,0,0)).
    """
    out = np.array(f, dtype=np.float64, copy=True)
    cx = np.asarray(spec.cx, dtype=np.int32)
    cy = np.asarray(spec.cy, dtype=np.int32)
    cz = np.asarray(spec.cz, dtype=np.int32)
    w = np.asarray(spec.weights, dtype=np.float64)
    opp = np.asarray(spec.opposite, dtype=np.int32)
    q = spec.num_dirs

    un = nx_in * vx + ny_in * vy + nz_in * vz
    denom = 1.0 - un
    if denom <= 1.0e-8:
        return out

    known = 0.0
    for d in range(q):
        cn = int(cx[d]) * nx_in + int(cy[d]) * ny_in + int(cz[d]) * nz_in
        if cn == 0:
            known += out[d]
        elif cn < 0:
            known += 2.0 * out[d]

    rho_w = known / denom
    if rho_w < 0.05 or not np.isfinite(rho_w):
        return out

    # Equilibrium helper (2nd-order; ZH NEBB uses feq difference)
    def feq(d: int, rho: float, ux: float, uy: float, uz: float) -> float:
        cxi, cyi, czi = float(cx[d]), float(cy[d]), float(cz[d])
        cu = cxi * ux + cyi * uy + czi * uz
        u2 = ux * ux + uy * uy + uz * uz
        return rho * w[d] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

    for d in range(q):
        cn = int(cx[d]) * nx_in + int(cy[d]) * ny_in + int(cz[d]) * nz_in
        if cn > 0:
            od = int(opp[d])
            out[d] = (
                feq(d, rho_w, vx, vy, vz)
                + out[od]
                - feq(od, rho_w, vx, vy, vz)
            )
    return out


def face_normal_inward(face: str) -> tuple[int, int, int]:
    """Inward unit normal for a named domain face."""
    return {
        "xmin": (1, 0, 0),
        "xmax": (-1, 0, 0),
        "ymin": (0, 1, 0),
        "ymax": (0, -1, 0),
        "zmin": (0, 0, 1),
        "zmax": (0, 0, -1),
    }[face]
