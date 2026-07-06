# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Elastic force kernels for deformable elements.

Four element types are supported:

* **Spring-dashpot** – pairwise damped springs (Hooke + dashpot).
* **Membrane (triangle) FEM** – thin-shell Neo-Hookean with area preservation
  and optional aerodynamic lift/drag.
* **Hinge bending** – dihedral-angle bending resistance across shared edges.
* **Solid (tetrahedral) FEM** – 3-D Neo-Hookean with volume preservation.

Design
------
Each kernel is decomposed into small ``@wp.func`` helpers that map directly
to the underlying continuum-mechanics concepts.  The kernel itself is then a
thin orchestration layer: read element data → call physics funcs → scatter
forces with ``wp.atomic_add / wp.atomic_sub``.

All force contributions are *accumulated* into a shared buffer so kernels can
run in any order within a time-step.
"""

from __future__ import annotations

import warp as wp

from .material import pk1_stress_2d, pk1_stress_3d


# =====================================================================
# 1. Spring-dashpot helpers + kernel
# =====================================================================


@wp.func
def spring_force(
    pi: wp.vec3,
    pj: wp.vec3,
    vi: wp.vec3,
    vj: wp.vec3,
    rest_len: float,
    stiffness: float,
    damping: float,
) -> wp.vec3:
    """Damped spring (Hooke + dashpot) force acting on particle i.

    Hooke's law:
        F_spring = k_e * (|x_ij| - L0) * x_hat

    Dashpot (velocity-proportional damping along spring axis):
        F_damp = k_d * (v_ij · x_hat) * x_hat

    The returned force should be *subtracted* from i and *added* to j
    to satisfy Newton's third law.
    """
    x_ij = pi - pj
    length = wp.length(x_ij)
    x_hat = x_ij / length                          # unit vector i → j

    extension = length - rest_len                  # positive = stretched
    rel_vel_axial = wp.dot(vi - vj, x_hat)         # rate of extension

    return x_hat * (stiffness * extension + damping * rel_vel_axial)


@wp.kernel
def spring_dashpot_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    conn: wp.array(dtype=int),
    rest_len: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    frc: wp.array(dtype=wp.vec3),
):
    """Evaluate damped-spring forces for each spring element."""
    idx = wp.tid()

    i = conn[idx * 2]
    j = conn[idx * 2 + 1]
    if i == -1 or j == -1:
        return

    f = spring_force(
        pos[i], pos[j],
        vel[i], vel[j],
        rest_len[idx], stiffness[idx], damping[idx],
    )

    wp.atomic_sub(frc, i, f)
    wp.atomic_add(frc, j, f)


# =====================================================================
# 2. Membrane FEM helpers + kernel
# =====================================================================


@wp.func
def deformation_gradient_2d(
    e1: wp.vec3,
    e2: wp.vec3,
    Bm: wp.mat22,
) -> tuple[wp.vec3, wp.vec3]:
    """Compute the 2-D deformation gradient columns from edge vectors.

    The deformation gradient F maps reference coordinates to world space:

        F = X_s * Bm,   X_s = [e1 | e2]  (world edge matrix)

    Returns the two columns (g1, g2) of F.  The same function is used
    for the rate dF/dt by passing velocity differences instead of positions.
    """
    g1 = e1 * Bm[0, 0] + e2 * Bm[1, 0]
    g2 = e1 * Bm[0, 1] + e2 * Bm[1, 1]
    return g1, g2


@wp.func
def area_preservation_force(
    e1: wp.vec3,
    e2: wp.vec3,
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    inv_rest_area: float,
    mu: float,
    lam: float,
    damp: float,
    activation: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Area-preservation (volumetric) constraint force for a triangle.

    Constraint function:
        c = A / A0 - alpha + act,   alpha = 1 + mu/lambda

    Force on nodes 1 and 2 (node 0 gets -(f1 + f2)):
        f_i = (lambda * c + k_d * dc/dt) * dc/dx_i
    """
    normal = wp.cross(e1, e2)
    area   = wp.length(normal) * 0.5
    normal = wp.normalize(normal)

    alpha = 1.0 + mu / lam
    c     = area * inv_rest_area - alpha + activation

    # gradient of constraint w.r.t. node positions
    dc_dp1 = wp.cross(e2, normal) * inv_rest_area * 0.5
    dc_dp2 = wp.cross(normal, e1) * inv_rest_area * 0.5

    # constraint velocity (time derivative of c)
    dcdt = wp.dot(dc_dp1, u1) + wp.dot(dc_dp2, u2) - wp.dot(dc_dp1 + dc_dp2, u0)

    scale = lam * c + damp * dcdt
    return dc_dp1 * scale, dc_dp2 * scale


@wp.func
def aero_force(
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    normal: wp.vec3,
    area: float,
    k_drag: float,
    k_lift: float,
) -> wp.vec3:
    """Aerodynamic drag + lift on a triangle element.

    Drag (opposes motion, proportional to normal flux):
        F_drag = k_d * A * |n · v_mid| * v_mid

    Lift (perpendicular to motion, angle-of-attack model):
        F_lift = k_l * A * (pi/2 - acos(n · v_dir)) * |v_mid|^2 * n
    """
    v_mid = (u0 + u1 + u2) * 0.3333
    v_dir = wp.normalize(v_mid)

    f_drag = v_mid * (k_drag * area * wp.abs(wp.dot(normal, v_mid)))
    f_lift = normal * (k_lift * area * (wp.HALF_PI - wp.acos(wp.dot(normal, v_dir)))) * wp.dot(v_mid, v_mid)
    return f_drag + f_lift


@wp.kernel
def membrane_stress_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    elem: wp.array2d(dtype=int),
    ref_inv: wp.array(dtype=wp.mat22),
    activation: wp.array(dtype=float),
    mat: wp.array2d(dtype=float),
    law: int,
    frc: wp.array(dtype=wp.vec3),
):
    """Triangle membrane FEM: deviatoric PK1 + area preservation + aero.

    ``law < 0`` uses the built-in linear Neo-Hookean deviatoric term
    (P = mu * F + k_d * dF/dt).  Any non-negative value is forwarded to
    :func:`pk1_stress_2d` for a pluggable material law.
    """
    idx = wp.tid()

    mu   = mat[idx, 0]
    lam  = mat[idx, 1]
    damp = mat[idx, 2]
    drag = mat[idx, 3]
    lift = mat[idx, 4]

    n0 = elem[idx, 0]
    n1 = elem[idx, 1]
    n2 = elem[idx, 2]

    p0 = pos[n0];  p1 = pos[n1];  p2 = pos[n2]
    u0 = vel[n0];  u1 = vel[n1];  u2 = vel[n2]

    e1 = p1 - p0
    e2 = p2 - p0
    de1 = u1 - u0
    de2 = u2 - u0

    Bm = ref_inv[idx]

    inv_rest_area = wp.determinant(Bm) * 2.0
    rest_area     = 1.0 / inv_rest_area

    # scale material params by rest area (standard FEM area weighting)
    mu   = mu   * rest_area
    lam  = lam  * rest_area
    damp = damp * rest_area

    # deformation gradient and its rate
    g1, g2   = deformation_gradient_2d(e1, e2, Bm)
    dg1, dg2 = deformation_gradient_2d(de1, de2, Bm)

    # deviatoric PK1 stress
    if law < 0:
        # built-in: linear Neo-Hookean deviatoric term
        s1 = g1 * mu + dg1 * damp
        s2 = g2 * mu + dg2 * damp
    else:
        s1, s2 = pk1_stress_2d(law, g1, g2, dg1, dg2, mu, damp)

    # nodal forces from PK1:  nf = P * Bm^T
    nf1 = s1 * Bm[0, 0] + s2 * Bm[0, 1]
    nf2 = s1 * Bm[1, 0] + s2 * Bm[1, 1]

    # area-preservation volumetric term
    if lam > 0.0:
        a1, a2 = area_preservation_force(e1, e2, u0, u1, u2,
                                         inv_rest_area, mu, lam, damp,
                                         activation[idx])
        nf1 = nf1 + a1
        nf2 = nf2 + a2

    nf0 = nf1 + nf2

    # aerodynamic contribution
    normal = wp.normalize(wp.cross(e1, e2))
    area   = wp.length(wp.cross(e1, e2)) * 0.5
    aero   = aero_force(u0, u1, u2, normal, area, drag, lift)

    nf0 = nf0 - aero
    nf1 = nf1 + aero
    nf2 = nf2 + aero

    wp.atomic_add(frc, n0,  nf0)
    wp.atomic_sub(frc, n1,  nf1)
    wp.atomic_sub(frc, n2,  nf2)


# =====================================================================
# 3. Hinge bending helpers + kernel
# =====================================================================


@wp.func
def dihedral_angle(
    n1_hat: wp.vec3,
    n2_hat: wp.vec3,
    e_hat: wp.vec3,
) -> float:
    """Signed dihedral angle between two triangle faces sharing an edge.

    Uses atan2 for a full [-pi, pi] range:
        cos(theta) = n1 · n2
        sin(theta) = (n1 × n2) · e_hat
    """
    cos_th = wp.dot(n1_hat, n2_hat)
    sin_th = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    return wp.atan2(sin_th, cos_th)


@wp.func
def dihedral_gradients(
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    p4: wp.vec3,
    n1_hat: wp.vec3,
    n2_hat: wp.vec3,
    e_hat: wp.vec3,
    e_len: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    """Gradient of the dihedral angle w.r.t. each of the four vertices.

    Follows Bridson et al. "Simulation of Clothing with Folds and Wrinkles".
    p1, p2 are the 'wing' vertices (one per face, not on the shared edge).
    p3, p4 are the shared edge endpoints.
    """
    d1 = -n1_hat * e_len
    d2 = -n2_hat * e_len
    d3 = -n1_hat * wp.dot(p1 - p4, e_hat) - n2_hat * wp.dot(p2 - p4, e_hat)
    d4 = -n1_hat * wp.dot(p3 - p1, e_hat) - n2_hat * wp.dot(p3 - p2, e_hat)
    return d1, d2, d3, d4


@wp.kernel
def hinge_bending_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    edge_nodes: wp.array2d(dtype=int),
    rest_angle: wp.array(dtype=float),
    bend_props: wp.array2d(dtype=float),
    frc: wp.array(dtype=wp.vec3),
):
    """Evaluate dihedral bending forces across triangle-pair edges."""
    idx = wp.tid()
    eps = 1.0e-6

    k_e = bend_props[idx, 0]
    k_d = bend_props[idx, 1]

    i = edge_nodes[idx, 0]
    j = edge_nodes[idx, 1]
    k = edge_nodes[idx, 2]
    l = edge_nodes[idx, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    p1 = pos[i];  p2 = pos[j];  p3 = pos[k];  p4 = pos[l]
    u1 = vel[i];  u2 = vel[j];  u3 = vel[k];  u4 = vel[l]

    # face normals (unnormalised)
    n1 = wp.cross(p3 - p1, p4 - p1)
    n2 = wp.cross(p4 - p2, p3 - p2)
    edge = p4 - p3

    len_n1 = wp.length(n1)
    len_n2 = wp.length(n2)
    e_len  = wp.length(edge)

    if len_n1 < eps or len_n2 < eps or e_len < eps:
        return

    n1_hat = n1 / len_n1
    n2_hat = n2 / len_n2
    e_hat  = edge / e_len

    theta = dihedral_angle(n1_hat, n2_hat, e_hat)
    d1, d2, d3, d4 = dihedral_gradients(p1, p2, p3, p4, n1_hat, n2_hat, e_hat, e_len)

    # elastic restoring torque + velocity damping
    f_elastic = k_e * (theta - rest_angle[idx])
    f_damp    = k_d * (wp.dot(d1, u1) + wp.dot(d2, u2) + wp.dot(d3, u3) + wp.dot(d4, u4))
    scale     = -e_len * (f_elastic + f_damp)

    wp.atomic_add(frc, i, d1 * scale)
    wp.atomic_add(frc, j, d2 * scale)
    wp.atomic_add(frc, k, d3 * scale)
    wp.atomic_add(frc, l, d4 * scale)


# =====================================================================
# 4. Solid FEM helpers + kernel
# =====================================================================


@wp.func
def deformation_gradient_3d(
    e1: wp.vec3,
    e2: wp.vec3,
    e3: wp.vec3,
    Bm: wp.mat33,
) -> wp.mat33:
    """Compute the 3-D deformation gradient F and its rate dF/dt.

    F = Xs * Bm,   Xs = [e1 | e2 | e3]  (world edge matrix)

    Pass velocity differences for (e1, e2, e3) to obtain dF/dt.
    Returns (F, dFdt) — call twice or pass vel differences for the rate.
    """
    Xs = wp.matrix_from_cols(e1, e2, e3)
    return Xs * Bm


@wp.func
def volume_preservation_force(
    e1: wp.vec3,
    e2: wp.vec3,
    e3: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    u3: wp.vec3,
    inv_rest_vol: float,
    J: float,
    alpha: float,
    lam: float,
    damp: float,
    activation: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Volume-preservation (hydrostatic) constraint force for a tetrahedron.

    Constraint function:
        c = J - alpha + act,   J = det(F)

    Gradient of J w.r.t. node positions (cofactor formula):
        dJ/dp1 = (e2 × e3) * s,   s = 1 / (6 * V0)

    Force on nodes 1, 2, 3 (node 0 gets -(f1+f2+f3)):
        f_i = (lambda * c + k_d * dJ/dt) * dJ/dp_i
    """
    s = inv_rest_vol / 6.0
    dJ_dp1 = wp.cross(e2, e3) * s
    dJ_dp2 = wp.cross(e3, e1) * s
    dJ_dp3 = wp.cross(e1, e2) * s

    # constraint velocity
    dJdt = wp.dot(dJ_dp1, u1) + wp.dot(dJ_dp2, u2) + wp.dot(dJ_dp3, u3)

    scale = (J - alpha + activation) * lam + dJdt * damp

    return dJ_dp1 * scale, dJ_dp2 * scale, dJ_dp3 * scale


@wp.kernel
def solid_stress_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    elem: wp.array2d(dtype=int),
    ref_inv: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    mat: wp.array2d(dtype=float),
    law: int,
    frc: wp.array(dtype=wp.vec3),
):
    """Tetrahedral FEM: deviatoric PK1 + volume preservation.

    ``law < 0`` uses the built-in rest-stability Neo-Hookean:
        P = mu * (1 - 1/(Ic+1)) * F + k_d * dF/dt

    Any non-negative value is forwarded to :func:`pk1_stress_3d`.
    """
    idx = wp.tid()

    n0 = elem[idx, 0];  n1 = elem[idx, 1]
    n2 = elem[idx, 2];  n3 = elem[idx, 3]

    mu   = mat[idx, 0]
    lam  = mat[idx, 1]
    damp = mat[idx, 2]
    act  = activation[idx]

    p0 = pos[n0];  p1 = pos[n1];  p2 = pos[n2];  p3 = pos[n3]
    u0 = vel[n0];  u1 = vel[n1];  u2 = vel[n2];  u3 = vel[n3]

    e1 = p1 - p0;  e2 = p2 - p0;  e3 = p3 - p0
    de1 = u1 - u0; de2 = u2 - u0; de3 = u3 - u0

    Bm = ref_inv[idx]

    inv_rest_vol = wp.determinant(Bm) * 6.0
    rest_vol     = 1.0 / inv_rest_vol

    # neo-Hookean stability parameter (Smith et al. 2018)
    alpha = 1.0 + mu / lam - mu / (4.0 * lam)

    # scale material params by rest volume
    mu   = mu   * rest_vol
    lam  = lam  * rest_vol
    damp = damp * rest_vol

    # deformation gradient and its rate
    F    = deformation_gradient_3d(e1,  e2,  e3,  Bm)
    dFdt = deformation_gradient_3d(de1, de2, de3, Bm)

    # deviatoric PK1 stress
    if law < 0:
        # built-in: rest-stability Neo-Hookean (Smith et al. 2018)
        c0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
        c1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
        c2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
        Ic = wp.dot(c0, c0) + wp.dot(c1, c1) + wp.dot(c2, c2)   # first invariant
        P  = F * (mu * (1.0 - 1.0 / (Ic + 1.0))) + dFdt * damp
    else:
        P = pk1_stress_3d(law, F, dFdt, mu, damp)

    # nodal forces from PK1:  H = P * Bm^T,  nf_i = col_i(H)
    H   = P * wp.transpose(Bm)
    nf1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    nf2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    nf3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    # volume-preservation hydrostatic term
    J = wp.determinant(F)
    vf1, vf2, vf3 = volume_preservation_force(
        e1, e2, e3, u1, u2, u3,
        inv_rest_vol, J, alpha, lam, damp, act,
    )

    nf1 = nf1 + vf1
    nf2 = nf2 + vf2
    nf3 = nf3 + vf3
    nf0 = -(nf1 + nf2 + nf3)

    wp.atomic_sub(frc, n0, nf0)
    wp.atomic_sub(frc, n1, nf1)
    wp.atomic_sub(frc, n2, nf2)
    wp.atomic_sub(frc, n3, nf3)


# =====================================================================
# Python-side launchers
# =====================================================================


def apply_spring_dashpot(model, state, pforce: wp.array):
    """Launch spring-dashpot kernel if springs exist."""
    if model.spring_count:
        wp.launch(
            kernel=spring_dashpot_kernel,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
            ],
            outputs=[pforce],
            device=model.device,
        )


def apply_membrane_stress(model, state, control, pforce: wp.array, material_law: int | None = None):
    """Launch membrane FEM kernel for triangle elements.

    ``material_law=None`` uses the built-in Neo-Hookean deviatoric term.
    Pass a :class:`~.material.MaterialLaw` value to select an alternative.
    """
    if model.tri_count:
        wp.launch(
            kernel=membrane_stress_kernel,
            dim=model.tri_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.tri_indices,
                model.tri_poses,
                control.tri_activations,
                model.tri_materials,
                material_law if material_law is not None else -1,
            ],
            outputs=[pforce],
            device=model.device,
        )


def apply_hinge_bending(model, state, pforce: wp.array):
    """Launch hinge bending kernel for edge elements."""
    if model.edge_count:
        wp.launch(
            kernel=hinge_bending_kernel,
            dim=model.edge_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_bending_properties,
            ],
            outputs=[pforce],
            device=model.device,
        )


def apply_solid_stress(model, state, control, pforce: wp.array, material_law: int | None = None):
    """Launch solid (tet) FEM kernel.

    ``material_law=None`` uses the built-in rest-stability Neo-Hookean.
    Pass a :class:`~.material.MaterialLaw` value to select an alternative.
    """
    if model.tet_count:
        wp.launch(
            kernel=solid_stress_kernel,
            dim=model.tet_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.tet_indices,
                model.tet_poses,
                control.tet_activations,
                model.tet_materials,
                material_law if material_law is not None else -1,
            ],
            outputs=[pforce],
            device=model.device,
        )
