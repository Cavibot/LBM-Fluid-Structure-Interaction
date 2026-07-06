# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import warp as wp
from .bvh import BvhEdge, BvhTri
from .collision_backend import CollisionBackend
from newton._src.sim import Contacts, Model, State
from newton._src.core.types import Devicelike
from .narrow_phase_vf import (
    handle_vertex_triangle_contacts_geometry_kernel,
    handle_vertex_triangle_contacts_force_kernel,
)
from .narrow_phase_ee import (
    handle_edge_edge_contacts_geometry_kernel,
    handle_edge_edge_contacts_force_kernel,
)
from .narrow_phase_ef import (
    handle_edge_face_contacts_geometry_kernel,
    handle_edge_face_contacts_force_kernel,
)
from .utils.geometry_utils import eval_body_contact_kernel


########################################################################################################################
###########################################    CollisionTriMeshStyle3D    ##############################################
########################################################################################################################
class CollisionTriMeshStyle3D(CollisionBackend):
    name: str = "collision_trimesh_style3d"

    def __init__(self, model: Model):
        """
        Initialize the collision handler, including BVHs and buffers.

        Args:
            model: The simulation model containing particle and geometry data.
        """
        self.model = model
        self.radius = 3e-3  # Contact radius
        self.stiff_vf = 0.5  # Stiffness coefficient for vertex-face (VF) collision constraints
        self.stiff_ee = 0.1  # Stiffness coefficient for edge-edge (EE) collision constraints
        self.stiff_ef = 1.0  # Stiffness coefficient for edge-face (EF) collision constraints
        self.friction_epsilon = 1e-2
        self.max_contacts_count = 32
        self.integrate_with_external_rigid_solver = True
        self.tri_bvh = BvhTri(model.tri_count, self.model.device)
        self.edge_bvh = BvhEdge(model.edge_count, self.model.device)
        self.body_contact_max = model.shape_count * model.particle_count
        self.broad_phase_ee = wp.array(shape=(self.max_contacts_count, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_ef = wp.array(shape=(self.max_contacts_count, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_vf = wp.array(shape=(self.max_contacts_count, model.particle_count), dtype=int, device=self.model.device)

        self.vf_contacts = ContactVertexTriangle.allocate(
            model.particle_count,
            self.max_contacts_count,
            self.model.device,
        )

        self.ee_contacts = ContactEdgeEdge.allocate(
            model.edge_count,
            self.max_contacts_count,
            self.model.device,
        )

        self.ef_contacts = ContactEdgeFace.allocate(
            model.edge_count,
            self.max_contacts_count,
            self.model.device,
        )


        self.Hx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.contact_hessian_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.model.device)

        self.edge_bvh.build(model.particle_q, self.model.edge_indices, self.radius)
        self.tri_bvh.build(model.particle_q, self.model.tri_indices, self.radius)

    def build(self, model: Model, device=None):
        """Backend interface. CollisionTriMeshStyle3D builds in __init__, so this is a light reinit."""
        self.rebuild_bvh(model.particle_q)

    def refit(self, state: State):
        """Backend interface. Update BVHs using the current state."""
        self.refit_bvh(state.particle_q)

    def generate_candidates(self, state: State, params, dt: float, out_pairs, out_pair_count) -> None:
        """Backend interface. Uses internal broad-phase buffers."""
        self.frame_begin(state.particle_q, state.particle_qd, dt)

    def narrow_phase(self, state: State, params, dt: float, pairs, pair_count, out_hits, mode: str) -> None:
        """Backend interface. Builds contact geometry using internal candidates."""
        thickness = 2.0 * self.radius
        if self.stiff_vf > 0:
            self.build_vertex_triangle_contacts(thickness, state.particle_q)
        if self.stiff_ee > 0:
            self.build_edge_edge_contacts(thickness, state.particle_q)
        if self.stiff_ef > 0:
            self.build_edge_face_contacts(state.particle_q)

    def build_vertex_triangle_contacts(self, thickness: float,  particle_q: wp.array):
        wp.launch(
            handle_vertex_triangle_contacts_geometry_kernel,
            dim=len(particle_q),
            inputs=[
                thickness,
                particle_q,
                self.model.tri_indices,
                self.broad_phase_vf,
                self.vf_contacts.max_contacts,
            ],
            outputs=[
                self.vf_contacts.contact_count,
                self.vf_contacts.contact_fid,
                self.vf_contacts.contact_normal,
                self.vf_contacts.contact_dist,
                self.vf_contacts.contact_bary,
                self.vf_contacts.contact_point,
                self.vf_contacts.contact_penetration,
            ],
            device=self.model.device,
        )

    def solve_vertex_triangle_contacts(
        self,
        thickness: float,
        particle_stiff: wp.array,
        particle_forces: wp.array,
    ) -> None:
        wp.launch(
            handle_vertex_triangle_contacts_force_kernel,
            dim=len(particle_forces),
            inputs=[
                thickness,
                self.stiff_vf,
                self.model.tri_indices,
                particle_stiff,
                self.vf_contacts.max_contacts,
                self.vf_contacts.contact_count,
                self.vf_contacts.contact_fid,
                self.vf_contacts.contact_normal,
                self.vf_contacts.contact_dist,
                self.vf_contacts.contact_bary,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

    def build_edge_edge_contacts(self, thickness: float, particle_q: wp.array):
        wp.launch(
            handle_edge_edge_contacts_geometry_kernel,
            dim=self.model.edge_indices.shape[0],
            inputs=[
                thickness,
                particle_q,
                self.model.edge_indices,
                self.broad_phase_ee,
                self.max_contacts_count,
            ],
            outputs=[
                self.ee_contacts.contact_count,
                self.ee_contacts.contact_eid,
                self.ee_contacts.contact_s,
                self.ee_contacts.contact_t,
                self.ee_contacts.contact_dir,
                self.ee_contacts.contact_dist,
                self.ee_contacts.contact_limit,
                self.ee_contacts.contact_point,
                self.ee_contacts.contact_penetration,
            ],
            device=self.model.device,
        )

    def solve_edge_edge_contacts(self, particle_stiff: wp.array, particle_forces: wp.array):
        wp.launch(
            handle_edge_edge_contacts_force_kernel,
            dim=self.model.edge_indices.shape[0],
            inputs=[
                self.stiff_ee,
                self.model.edge_indices,
                particle_stiff,
                self.max_contacts_count,
                self.ee_contacts.contact_count,
                self.ee_contacts.contact_eid,
                self.ee_contacts.contact_s,
                self.ee_contacts.contact_t,
                self.ee_contacts.contact_dir,
                self.ee_contacts.contact_dist,
                self.ee_contacts.contact_limit,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

    def build_edge_face_contacts(self, particle_q: wp.array):
        wp.launch(
            handle_edge_face_contacts_geometry_kernel,
            dim=self.model.edge_indices.shape[0],
            inputs=[
                particle_q,
                self.model.tri_indices,
                self.model.edge_indices,
                self.broad_phase_ef,
                self.ef_contacts.max_contacts,
            ],
            outputs=[
                self.ef_contacts.contact_count,
                self.ef_contacts.contact_fid,
                self.ef_contacts.contact_dir,
                self.ef_contacts.contact_bary,
                self.ef_contacts.contact_edge_bary,
                self.ef_contacts.contact_point,
            ],
            device=self.model.device,
        )

    def solve_edge_face_contacts(self, thickness: float, particle_stiff: wp.array, particle_forces: wp.array):
        wp.launch(
            handle_edge_face_contacts_force_kernel,
            dim=self.model.edge_indices.shape[0],
            inputs=[
                thickness,
                self.stiff_ef,
                self.model.tri_indices,
                self.model.edge_indices,
                particle_stiff,
                self.ef_contacts.max_contacts,
                self.ef_contacts.contact_count,
                self.ef_contacts.contact_fid,
                self.ef_contacts.contact_dir,
                self.ef_contacts.contact_bary,
                self.ef_contacts.contact_edge_bary,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

    def rebuild_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """
        Rebuild triangle and edge BVHs.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.rebuild(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.rebuild(pos, self.model.edge_indices, self.radius)

    def refit_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """
        Refit (update) triangle and edge BVHs based on new positions without changing topology.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.refit(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.refit(pos, self.model.edge_indices, self.radius)

    def frame_begin(self, particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3), dt: float):
        """
        Perform broad-phase collision detection using BVHs.

        Args:
            particle_q: Array of vertex positions.
            particle_qd: Array of vertex velocities.
            dt: simulation time step.
        """
        max_dist = self.radius * 3.0
        query_radius = self.radius

        self.refit_bvh(particle_q)

        # Vertex-face collision candidates
        if self.stiff_vf > 0.0:
            self.tri_bvh.triangle_vs_point(
                particle_q,
                particle_q,
                self.model.tri_indices,
                self.broad_phase_vf,
                True,
                max_dist,
                query_radius,
            )

        # Edge-edge collision candidates
        if self.stiff_ee > 0.0:
            self.edge_bvh.edge_vs_edge(
                particle_q,
                self.model.edge_indices,
                particle_q,
                self.model.edge_indices,
                self.broad_phase_ee,
                True,
                max_dist,
                query_radius,
            )

        # Face-edge collision candidates
        if self.stiff_ef > 0.0:
            self.tri_bvh.aabb_vs_aabb(
                self.edge_bvh.lower_bounds,
                self.edge_bvh.upper_bounds,
                self.broad_phase_ef,
                query_radius,
                False,
            )

    def accumulate_contact_force(
        self,
        dt: float,
        _iter: int,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        particle_forces: wp.array(dtype=wp.vec3),
        particle_q_prev: wp.array(dtype=wp.vec3),
        particle_stiff: wp.array(dtype=wp.vec3) = None,
    ):
        """
        Evaluates contact forces and the diagonal of the Hessian for implicit time integration.

        This method launches kernels to compute contact forces and Hessian contributions
        based on broad-phase collision candidates computed in frame_begin().

        Args:
            dt (float): Time step.
            state_in (GeometryState): Current simulation state (input).
            state_out (GeometryState): Next simulation state (output).
            contacts (Contacts): Contact data structure containing contact information.
            particle_forces (wp.array): Output array for computed contact forces.
            particle_q_prev (wp.array): Previous positions (optional, for velocity-based damping).
            particle_stiff (wp.array): Optional stiffness array for particles.
        """
        thickness = 2.0 * self.radius
        self.contact_hessian_diags.zero_()

        if self.stiff_vf > 0:
            self.build_vertex_triangle_contacts(thickness, state_in.particle_q)
            self.solve_vertex_triangle_contacts(thickness, particle_stiff, particle_forces)

        if self.stiff_ee > 0:
            self.build_edge_edge_contacts(thickness, state_in.particle_q)
            self.solve_edge_edge_contacts(particle_stiff, particle_forces)

        if self.stiff_ef > 0:
            self.build_edge_face_contacts(state_in.particle_q)
            self.solve_edge_face_contacts(thickness, particle_stiff, particle_forces)

        wp.launch(
            kernel=eval_body_contact_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                particle_q_prev,
                state_in.particle_q,
                # body-particle contact
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.model.shape_material_mu,
                self.model.shape_body,
                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                state_in.body_q if self.integrate_with_external_rigid_solver else None,
                self.model.body_qd,
                self.model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

    def contact_hessian_diagonal(self):
        """Return diagonal of contact Hessian for preconditioning.
        Note:
            Should be called after `accumulate_contact_force()`.
        """
        return self.contact_hessian_diags

    def hessian_multiply(self, x: wp.array(dtype=wp.vec3)):
        """Computes the Hessian-vector product for implicit integration."""

        @wp.kernel
        def hessian_multiply_kernel(
            hessian_diags: wp.array(dtype=wp.mat33),
            x: wp.array(dtype=wp.vec3),
            # outputs
            Hx: wp.array(dtype=wp.vec3),
        ):
            tid = wp.tid()
            Hx[tid] = hessian_diags[tid] * x[tid]

        wp.launch(
            hessian_multiply_kernel,
            dim=self.model.particle_count,
            inputs=[self.contact_hessian_diags, x],
            outputs=[self.Hx],
            device=self.model.device,
        )
        return self.Hx

    def linear_iteration_end(self, dx: wp.array(dtype=wp.vec3)):
        """Displacement constraints"""
        pass

    def frame_end(self, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
        """Apply post-processing"""
        pass


@dataclass
class ContactVertexTriangle:
    max_contacts: int
    contact_count: wp.array
    contact_fid: wp.array
    contact_normal: wp.array
    contact_dist: wp.array
    contact_bary: wp.array
    contact_point: wp.array
    contact_penetration: wp.array

    @classmethod
    def allocate(cls, particle_count: int, max_contacts: int, device: Devicelike) -> "ContactVertexTriangle":
        return cls(
            max_contacts=max_contacts,
            contact_count=wp.array(shape=(particle_count), dtype=int, device=device),
            contact_fid=wp.array(shape=(max_contacts, particle_count), dtype=int, device=device),
            contact_normal=wp.array(shape=(max_contacts, particle_count), dtype=wp.vec3, device=device),
            contact_dist=wp.array(shape=(max_contacts, particle_count), dtype=float, device=device),
            contact_bary=wp.array(shape=(max_contacts, particle_count), dtype=wp.vec3, device=device),
            contact_point=wp.array(shape=(max_contacts, particle_count), dtype=wp.vec3, device=device),
            contact_penetration=wp.array(shape=(max_contacts, particle_count), dtype=float, device=device),
        )

@dataclass
class ContactEdgeEdge:
    max_contacts: int
    contact_count: wp.array
    contact_eid: wp.array
    contact_s: wp.array
    contact_t: wp.array
    contact_dir: wp.array
    contact_dist: wp.array
    contact_limit: wp.array
    contact_point: wp.array
    contact_penetration: wp.array

    @classmethod
    def allocate(cls, edge_count: int, max_contacts: int, device: Devicelike) -> "ContactEdgeEdge":
        return cls(
            max_contacts=max_contacts,
            contact_count=wp.array(shape=(edge_count), dtype=int, device=device),
            contact_eid=wp.array(shape=(max_contacts, edge_count), dtype=int, device=device),
            contact_s=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_t=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_dir=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_dist=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_limit=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_point=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_penetration=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
        )

@dataclass
class ContactEdgeFace:
    max_contacts: int
    contact_count: wp.array
    contact_fid: wp.array
    contact_dir: wp.array
    contact_bary: wp.array
    contact_edge_bary: wp.array
    contact_point: wp.array

    @classmethod
    def allocate(cls, edge_count: int, max_contacts: int, device: Devicelike) -> "ContactEdgeFace":
        return cls(
            max_contacts=max_contacts,
            contact_count=wp.array(shape=(edge_count), dtype=int, device=device),
            contact_fid=wp.array(shape=(max_contacts, edge_count), dtype=int, device=device),
            contact_dir=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_bary=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_edge_bary=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec2, device=device),
            contact_point=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
        )
