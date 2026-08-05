"""Shan-Chen interaction force kernels."""

from __future__ import annotations

import warp as wp


@wp.func
def _psi(rho: float, psi_type: int, ref_rho: float, G: float, cs_a: float, cs_b: float, cs_T: float) -> float:
    """Shan-Chen pseudopotential (effective mass).

    ``PSI_RHO`` (0): ψ = ρ.
    ``PSI_EXP`` (1): ψ = 1 - exp(-ρ / ρ_ref).
    ``PSI_CS``  (2): ψ = sqrt(2·(P_CS − ρ·c_s²) / (G·c_s²))
        where P_CS is the Carnahan-Starling equation of state.
    """
    if psi_type == 1:
        return 1.0 - wp.exp(-rho / ref_rho)
    if psi_type == 2:
        cs2 = 1.0 / 3.0
        eta = cs_b * rho / 4.0
        eta2 = eta * eta
        eta3 = eta2 * eta
        denom = (1.0 - eta) * (1.0 - eta) * (1.0 - eta)
        p_cs = rho * cs_T * (1.0 + eta + eta2 - eta3) / denom - cs_a * rho * rho
        arg = 2.0 * (p_cs - rho * cs2) / (G * cs2)
        return wp.sqrt(wp.max(arg, 0.0))
    return rho


@wp.func
def _resolve_boundary_psi(
    ni: int,
    nj: int,
    nk: int,
    rho: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    psi_c: float,
    psi_type: int,
    psi_ref: float,
    solid_psi_scale: float,
    boundary_psi: float,
    G: float,
    cs_a: float,
    cs_b: float,
    cs_T: float,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """Return the pseudopotential ψ of a D3Q19 neighbour at (ni, nj, nk).

    Virtual-density method: fluid neighbours use their actual ψ(ρ); solid
    bodies and out-of-bounds domain faces each contribute a constant virtual
    ψ so that the unified SC force ``F = -G ψ(x) Σ w_i ψ_n e_i`` captures
    both fluid–fluid interaction and fluid–wall adsorption in a single sum.

    When a periodic axis flag (*px*, *py*, *pz*) is set, out-of-bounds
    neighbours on that axis wrap to the opposite face and contribute their
    actual fluid ψ instead of a virtual wall ψ.
    """
    # ---- periodic wrap -------------------------------------------------------
    if ni < 0 and px:
        ni = nx - 1
    if ni >= nx and px:
        ni = 0
    if nj < 0 and py:
        nj = ny - 1
    if nj >= ny and py:
        nj = 0
    if nk < 0 and pz:
        nk = nz - 1
    if nk >= nz and pz:
        nk = 0

    # ---- classify wrapped neighbour ------------------------------------------
    in_bounds = (
        (ni >= 0) and (ni < nx)
        and (nj >= 0) and (nj < ny)
        and (nk >= 0) and (nk < nz)
    )
    if in_bounds and solid_phi[ni, nj, nk] >= 0.0:
        return _psi(rho[ni, nj, nk], psi_type, psi_ref, G, cs_a, cs_b, cs_T)   # fluid

    if in_bounds:
        return psi_c * solid_psi_scale                     # solid body

    # out-of-bounds (domain wall on non-periodic axis)
    if boundary_psi >= 0.0:
        return boundary_psi                                 # fixed wall ψ
    return psi_c * solid_psi_scale                          # legacy mirror


@wp.kernel
def compute_shan_chen_force_kernel(
    rho: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    G: float,
    psi_type: int,
    psi_ref: float,
    solid_psi_scale: float,
    boundary_psi: float,
    cs_a: float,
    cs_b: float,
    cs_T: float,
    homogeneous_early_out: int,
    homogeneous_rel_tol: float,
    px: int,
    py: int,
    pz: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Unified Shan-Chen force via the virtual-density method.

    ``F(x) = -G ψ(x) Σ_i w_i ψ_n e_i``

    Every D3Q19 neighbour contributes to the same sum:
    - **fluid**: actual ψ(ρ) from the density field.
    - **solid body** (``solid_phi < 0``): virtual ψ = ``ψ_c × solid_psi_scale``.
    - **domain wall** (out-of-bounds on non-periodic axis): virtual ψ
      = *boundary_psi* when ≥ 0, else the legacy mirror closure
      ``ψ_c × solid_psi_scale``.
    - **periodic neighbour**: wraps to the opposite face via *px*/*py*/*pz*.

    Fluid–fluid interaction and fluid–wall adsorption are both controlled
    by the single interaction strength *G*; wetting is tuned via the
    virtual ψ of the wall / solid.
    """
    i, j, k = wp.tid()

    if solid_phi[i, j, k] < 0.0:
        fx[i, j, k] = 0.0
        fy[i, j, k] = 0.0
        fz[i, j, k] = 0.0
        return

    # ---- current-cell pseudopotential --------------------------------------
    rho_c = rho[i, j, k]
    psi_c = _psi(rho_c, psi_type, psi_ref, G, cs_a, cs_b, cs_T)

    # ---- homogeneous-region early-out ---------------------------------------
    # In bulk fluid (all neighbours have similar ρ, none is solid/wall),
    # the symmetric sum Σ w_i ψ_n e_i ≈ 0.  We check 6 face neighbours;
    # if any has deviant ρ, is solid, or is out-of-bounds on a non-periodic
    # axis, we fall through to the full 19-dir computation.
    # Periodic OOB neighbours wrap to the opposite face for the density check.
    lo = rho_c * (1.0 - homogeneous_rel_tol)
    hi = rho_c * (1.0 + homogeneous_rel_tol)
    homogeneous = 1
    if homogeneous_early_out == 0:
        homogeneous = 0
    # +x
    if homogeneous == 1:
        ni = i + 1
        if ni < nx:
            pass  # in-bounds
        elif px:
            ni = 0  # wrap xmax -> xmin
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[ni, j, k] < 0.0 or rho[ni, j, k] < lo or rho[ni, j, k] > hi:
                homogeneous = 0
    # -x
    if homogeneous == 1:
        ni = i - 1
        if ni >= 0:
            pass  # in-bounds
        elif px:
            ni = nx - 1  # wrap xmin -> xmax
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[ni, j, k] < 0.0 or rho[ni, j, k] < lo or rho[ni, j, k] > hi:
                homogeneous = 0
    # +y
    if homogeneous == 1:
        nj = j + 1
        if nj < ny:
            pass  # in-bounds
        elif py:
            nj = 0  # wrap ymax -> ymin
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, nj, k] < 0.0 or rho[i, nj, k] < lo or rho[i, nj, k] > hi:
                homogeneous = 0
    # -y
    if homogeneous == 1:
        nj = j - 1
        if nj >= 0:
            pass  # in-bounds
        elif py:
            nj = ny - 1  # wrap ymin -> ymax
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, nj, k] < 0.0 or rho[i, nj, k] < lo or rho[i, nj, k] > hi:
                homogeneous = 0
    # +z
    if homogeneous == 1:
        nk = k + 1
        if nk < nz:
            pass  # in-bounds
        elif pz:
            nk = 0  # wrap zmax -> zmin
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, j, nk] < 0.0 or rho[i, j, nk] < lo or rho[i, j, nk] > hi:
                homogeneous = 0
    # -z
    if homogeneous == 1:
        nk = k - 1
        if nk >= 0:
            pass  # in-bounds
        elif pz:
            nk = nz - 1  # wrap zmin -> zmax
        else:
            homogeneous = 0
        if homogeneous == 1:
            if solid_phi[i, j, nk] < 0.0 or rho[i, j, nk] < lo or rho[i, j, nk] > hi:
                homogeneous = 0
    if homogeneous == 1:
        fx[i, j, k] = 0.0
        fy[i, j, k] = 0.0
        fz[i, j, k] = 0.0
        return

    # ---- accumulate Σ w_i ψ(x+e_i) e_i ------------------------------------
    sx = 0.0
    sy = 0.0
    sz = 0.0

    # ========================================================================
    # Face directions (w = 1/18)
    # ========================================================================

    # d=1: +x  (1,0,0)
    ni = i + 1; nj = j; nk = k
    sx += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz) * 1.0

    # d=2: -x  (-1,0,0)
    ni = i - 1; nj = j; nk = k
    sx += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz) * (-1.0)

    # d=3: +y  (0,1,0)
    ni = i; nj = j + 1; nk = k
    sy += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz) * 1.0

    # d=4: -y  (0,-1,0)
    ni = i; nj = j - 1; nk = k
    sy += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz) * (-1.0)

    # d=5: +z  (0,0,1)
    ni = i; nj = j; nk = k + 1
    sz += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz) * 1.0

    # d=6: -z  (0,0,-1)
    ni = i; nj = j; nk = k - 1
    sz += (1.0 / 18.0) * _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz) * (-1.0)

    # ========================================================================
    # Edge directions (w = 1/36)
    # ========================================================================

    # d=7: +x+y  (1,1,0)
    ni = i + 1; nj = j + 1; nk = k
    psi_d7 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d7 * 1.0
    sy += (1.0 / 36.0) * psi_d7 * 1.0

    # d=8: -x+y  (-1,1,0)
    ni = i - 1; nj = j + 1; nk = k
    psi_d8 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d8 * (-1.0)
    sy += (1.0 / 36.0) * psi_d8 * 1.0

    # d=9: +x-y  (1,-1,0)
    ni = i + 1; nj = j - 1; nk = k
    psi_d9 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d9 * 1.0
    sy += (1.0 / 36.0) * psi_d9 * (-1.0)

    # d=10: -x-y  (-1,-1,0)
    ni = i - 1; nj = j - 1; nk = k
    psi_d10 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d10 * (-1.0)
    sy += (1.0 / 36.0) * psi_d10 * (-1.0)

    # d=11: +x+z  (1,0,1)
    ni = i + 1; nj = j; nk = k + 1
    psi_d11 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d11 * 1.0
    sz += (1.0 / 36.0) * psi_d11 * 1.0

    # d=12: -x+z  (-1,0,1)
    ni = i - 1; nj = j; nk = k + 1
    psi_d12 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d12 * (-1.0)
    sz += (1.0 / 36.0) * psi_d12 * 1.0

    # d=13: +x-z  (1,0,-1)
    ni = i + 1; nj = j; nk = k - 1
    psi_d13 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d13 * 1.0
    sz += (1.0 / 36.0) * psi_d13 * (-1.0)

    # d=14: -x-z  (-1,0,-1)
    ni = i - 1; nj = j; nk = k - 1
    psi_d14 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sx += (1.0 / 36.0) * psi_d14 * (-1.0)
    sz += (1.0 / 36.0) * psi_d14 * (-1.0)

    # d=15: +y+z  (0,1,1)
    ni = i; nj = j + 1; nk = k + 1
    psi_d15 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d15 * 1.0
    sz += (1.0 / 36.0) * psi_d15 * 1.0

    # d=16: -y+z  (0,-1,1)
    ni = i; nj = j - 1; nk = k + 1
    psi_d16 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d16 * (-1.0)
    sz += (1.0 / 36.0) * psi_d16 * 1.0

    # d=17: +y-z  (0,1,-1)
    ni = i; nj = j + 1; nk = k - 1
    psi_d17 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d17 * 1.0
    sz += (1.0 / 36.0) * psi_d17 * (-1.0)

    # d=18: -y-z  (0,-1,-1)
    ni = i; nj = j - 1; nk = k - 1
    psi_d18 = _resolve_boundary_psi(ni, nj, nk, rho, solid_phi, psi_c, psi_type, psi_ref, solid_psi_scale, boundary_psi, G, cs_a, cs_b, cs_T, px, py, pz, nx, ny, nz)
    sy += (1.0 / 36.0) * psi_d18 * (-1.0)
    sz += (1.0 / 36.0) * psi_d18 * (-1.0)

    # ---- unified force: F = -G ψ_c Σ w_i ψ_n e_i -------------------------
    coeff = -G * psi_c
    fx[i, j, k] = coeff * sx
    fy[i, j, k] = coeff * sy
    fz[i, j, k] = coeff * sz
