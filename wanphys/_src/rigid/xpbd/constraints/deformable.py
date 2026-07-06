# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deformable-body constraint groups: springs, bending, and tetrahedra."""

from __future__ import annotations

import warp as wp

from ..constraint_base import ConstraintGroup
from ..context import ConstraintPhase, XPBDContext

# ────────────────────────────────────────────────────────────────────
# Warp kernels
# ────────────────────────────────────────────────────────────────────


@wp.kernel
def solve_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    lambdas[tid] = lambdas[tid] * 0.4

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)

    if l == 0.0:
        return

    n = xij / l

    c = l - rest
    grad_c_xi = n
    grad_c_xj = -1.0 * n

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj

    # Note strict inequality for damping -- 0 damping is ok
    if denom <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_c_dot_v = dt * wp.dot(grad_c_xi, vij)  # Note: dt because from the paper we want x_i - x^n, not v...
    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

    dxi = wi * dlambda * grad_c_xi
    dxj = wj * dlambda * grad_c_xj

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)


@wp.kernel
def bending_constraint(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    lambdas[tid] = lambdas[tid] * 0.4

    eps = 1.0e-6
    # print(lambdas[tid])
    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    w1 = invmass[i]
    w2 = invmass[j]
    w3 = invmass[k]
    w4 = invmass[l]

    n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    # Check for degenerate cases
    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1_hat, n2_hat)
    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    c = theta - rest_angle

    grad_x1 = -n1_hat * e_length
    grad_x2 = -n2_hat * e_length
    grad_x3 = -n1_hat * wp.dot(x1 - x4, e_hat) - n2_hat * wp.dot(x2 - x4, e_hat)
    grad_x4 = -n1_hat * wp.dot(x3 - x1, e_hat) - n2_hat * wp.dot(x3 - x2, e_hat)

    denominator = (
        w1 * wp.length_sq(grad_x1)
        + w2 * wp.length_sq(grad_x2)
        + w3 * wp.length_sq(grad_x3)
        + w4 * wp.length_sq(grad_x4)
    )

    # Note strict inequality for damping -- 0 damping is ok
    if denominator <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_dot_v = dt * (wp.dot(grad_x1, v1) + wp.dot(grad_x2, v2) + wp.dot(grad_x3, v3) + wp.dot(grad_x4, v4))

    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

    delta0 = w1 * dlambda * grad_x1
    delta1 = w2 * dlambda * grad_x2
    delta2 = w3 * dlambda * grad_x3
    delta3 = w4 * dlambda * grad_x4

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, delta0)
    wp.atomic_add(delta, j, delta1)
    wp.atomic_add(delta, k, delta2)
    wp.atomic_add(delta, l, delta3)


@wp.kernel
def solve_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    rest_matrix: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    # k_mu = materials[tid, 0]
    # k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)

    stretching_compliance = relaxation
    volume_compliance = relaxation

    num_terms = 2
    for term in range(0, num_terms):
        if term == 0:
            # deviatoric, stable
            C = tr - 3.0
            dC = F * 2.0
            compliance = stretching_compliance
        elif term == 1:
            # volume conservation
            C = wp.determinant(F) - 1.0
            dC = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = volume_compliance

        if C != 0.0:
            dP = dC * inv_QT
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -grad1 - grad2 - grad3

            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance / dt / dt
                if inv_rest_volume > 0.0:
                    alpha *= inv_rest_volume
                dlambda = -C / (w + alpha)

                wp.atomic_add(delta, i, w0 * dlambda * grad0)
                wp.atomic_add(delta, j, w1 * dlambda * grad1)
                wp.atomic_add(delta, k, w2 * dlambda * grad2)
                wp.atomic_add(delta, l, w3 * dlambda * grad3)

@wp.kernel
def solve_tetrahedra3(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    rest_matrix: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # print(tid)

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0
    if inv_rest_volume <= 0.0:
        return

    rest_volume = 1.0 / inv_rest_volume

    F = Ds * Dm
    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    for term in range(2):
        C = float(0.0)
        dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        alpha_tilde = float(0.0)

        if term == 0:
            C = tr - 3.0
            dC = F * 2.0
            if k_mu > 0.0:
                alpha_tilde = 1.0 / (k_mu * rest_volume * dt * dt)
            else:
                alpha_tilde = relaxation * inv_rest_volume / (dt * dt)
        else:
            C = wp.determinant(F) - 1.0
            dC = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            if k_lambda > 0.0:
                alpha_tilde = 1.0 / (k_lambda * rest_volume * dt * dt)
            else:
                alpha_tilde = relaxation * inv_rest_volume / (dt * dt)

        dP = dC * inv_QT
        grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
        grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
        grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
        grad0 = -grad1 - grad2 - grad3

        w = (
            wp.dot(grad0, grad0) * w0
            + wp.dot(grad1, grad1) * w1
            + wp.dot(grad2, grad2) * w2
            + wp.dot(grad3, grad3) * w3
        )

        denom = w + alpha_tilde
        if denom > 0.0:
            lambda_idx = tid * 2 + term
            lambda_prev = lambdas[lambda_idx]
            dlambda = -(C + alpha_tilde * lambda_prev) / denom

            lambdas[lambda_idx] = lambda_prev + dlambda

            wp.atomic_add(delta, i, w0 * dlambda * grad0)
            wp.atomic_add(delta, j, w1 * dlambda * grad1)
            wp.atomic_add(delta, k, w2 * dlambda * grad2)
            wp.atomic_add(delta, l, w3 * dlambda * grad3)



# ────────────────────────────────────────────────────────────────────
# Constraint group implementations
# ────────────────────────────────────────────────────────────────────


class SpringConstraint(ConstraintGroup):
    """Distance (spring) constraint with stiffness and damping.

    Each spring connects two particles and enforces a rest length via the
    XPBD compliant constraint formulation with Lagrange multiplier
    accumulation.
    """

    phase = ConstraintPhase.PARTICLE

    def __init__(self, warm_start: bool = True) -> None:
        self._lambdas: wp.array | None = None
        self._prev_lambdas: wp.array | None = None
        self.warm_start = warm_start

    def is_active(self, model, contacts) -> bool:
        # return model.spring_count > 0
        return getattr(model, "spring_count", 0) > 0

    def initialize(self, ctx: XPBDContext) -> None:
        # self._lambdas = wp.empty_like(ctx.model.spring_rest_length)
        self._lambdas = wp.zeros_like(ctx.model.spring_rest_length)

    def reset_iteration(self, ctx: XPBDContext, iteration: int) -> None:
        # if (
        #     iteration == 0
        #     and self.warm_start
        #     and self._prev_lambdas is not None
        #     and self._prev_lambdas.shape == self._lambdas.shape
        # ):
        #     wp.copy(self._lambdas, self._prev_lambdas)
        # else:
        #     self._lambdas.zero_()

        if iteration == 0 or self.warm_start == False:
            self._lambdas.zero_()


    def stash_lambdas(self, ctx: XPBDContext) -> None:
        # if self.warm_start and self._lambdas is not None:
        #     if self._prev_lambdas is None or self._prev_lambdas.shape != self._lambdas.shape:
        #         self._prev_lambdas = wp.clone(self._lambdas)
        #     else:
        #         wp.copy(self._prev_lambdas, self._lambdas)
        pass

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        wp.launch(
            kernel=solve_springs,
            dim=model.spring_count,
            inputs=[
                ctx.particle_q,
                ctx.particle_qd,
                model.particle_inv_mass,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                ctx.dt,
                self._lambdas,
            ],
            outputs=[ctx.particle_deltas],
            device=model.device,
        )


class BendingConstraint(ConstraintGroup):
    """Dihedral-angle bending constraint for cloth / thin-shell simulation.

    Operates on edge quads (4 particles sharing an edge) and enforces the
    rest dihedral angle using the XPBD compliant formulation.
    """

    phase = ConstraintPhase.PARTICLE

    def __init__(self, warm_start: bool = True) -> None:
        self._lambdas: wp.array | None = None
        self._prev_lambdas: wp.array | None = None
        self.warm_start = warm_start

    def is_active(self, model, contacts) -> bool:
        # return model.edge_count > 0
        return getattr(model, "edge_count", 0) > 0

    def initialize(self, ctx: XPBDContext) -> None:
        # self._lambdas = wp.empty_like(ctx.model.edge_rest_angle)
        self._lambdas = wp.zeros_like(ctx.model.edge_rest_angle)

    def reset_iteration(self, ctx: XPBDContext, iteration: int) -> None:
        # if (
        #     iteration == 0
        #     and self.warm_start
        #     and self._prev_lambdas is not None
        #     and self._prev_lambdas.shape == self._lambdas.shape
        # ):
        #     wp.copy(self._lambdas, self._prev_lambdas)
        # else:
        #     self._lambdas.zero_()
        if iteration == 0 or self.warm_start == False:
            self._lambdas.zero_()

    def stash_lambdas(self, ctx: XPBDContext) -> None:
        # if self.warm_start and self._lambdas is not None:
        #     if self._prev_lambdas is None or self._prev_lambdas.shape != self._lambdas.shape:
        #         self._prev_lambdas = wp.clone(self._lambdas)
        #     else:
        #         wp.copy(self._prev_lambdas, self._lambdas)
        pass

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        wp.launch(
            kernel=bending_constraint,
            dim=model.edge_count,
            inputs=[
                ctx.particle_q,
                ctx.particle_qd,
                model.particle_inv_mass,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_bending_properties,
                ctx.dt,
                self._lambdas,
            ],
            outputs=[ctx.particle_deltas],
            device=model.device,
        )


class TetrahedraConstraint(ConstraintGroup):
    """Tetrahedral FEM constraint (deviatoric + volume conservation).

    Enforces a stable Neo-Hookean-like energy on each tetrahedron using
    two constraint terms: a deviatoric (stretch) term and a volume
    preservation term.
    """

    phase = ConstraintPhase.PARTICLE

    def __init__(self, relaxation: float = 0.9) -> None:
        self.tet_constraint_lambdas = None
        self.relaxation = relaxation

    def is_active(self, model, contacts) -> bool:
        # return model.tet_count > 0
        return getattr(model, "tet_count", 0) > 0

    def initialize(self, ctx: XPBDContext) -> None:
        # self._lambdas = wp.empty_like(ctx.model.edge_rest_angle)
        model = ctx.model
        n = model.tet_count

        if model.tet_count:
            self.tet_constraint_lambdas = wp.zeros(
                n * 2, dtype=float, device=model.device
            )
        # if self._lambda_curr is None or self._lambda_curr.shape[0] != n:
        #     self._lambda_curr = wp.zeros(n, dtype=float, device=model.device)
        #     self._lambda_prev = wp.zeros(n, dtype=float, device=model.device)

    def reset_iteration(self, ctx: XPBDContext, iteration: int) -> None:
        # self._lambdas.zero_()
        # 只在 step 第一次 iteration 做 warmstart
        if iteration != 0:
            return
        model = ctx.model
        self.tet_constraint_lambdas = wp.zeros(
            model.tet_count * 2, dtype=float, device=model.device
        )

    def project(self, ctx: XPBDContext, iteration: int) -> None:
        model = ctx.model
        wp.launch(
            kernel=solve_tetrahedra3,
            dim=model.tet_count,
            inputs=[
                ctx.particle_q,
                ctx.particle_qd,
                model.particle_inv_mass,
                model.tet_indices,
                model.tet_poses,
                model.tet_activations,
                model.tet_materials,
                ctx.dt,
                self.relaxation,
                self.tet_constraint_lambdas,  # new para
            ],
            outputs=[ctx.particle_deltas],
            device=model.device,
        )
