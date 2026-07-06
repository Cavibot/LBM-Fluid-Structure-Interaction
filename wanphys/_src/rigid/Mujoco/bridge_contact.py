from __future__ import annotations

import warp as wp

from .kernels import (
    _create_inverse_shape_mapping_kernel,
    convert_newton_contacts_to_mjwarp_kernel,
    convert_mjw_contact_to_warp_kernel,
)


class MujocoContactBridgeMixin:
    def _create_inverse_shape_mapping(self):
        """Build the reverse shape lookup used by externally supplied contacts.

        The compiler gives us a MuJoCo-geom -> source-shape map. When contacts
        come from WanPhys instead of MuJoCo's own collision path, we also need
        the inverse lookup so a source shape pair can be written into MuJoCo's
        contact buffers.
        """
        if self.mjc_geom_to_source_shape is None:
            return

        nworld = self.mjc_geom_to_source_shape.shape[0]
        ngeom = self.mjc_geom_to_source_shape.shape[1]

        self.source_shape_to_mjc_geom = wp.full(
            self._source_model.shape_count,
            -1,
            dtype=wp.int32,
            device=self.device,
        )

        wp.launch(
            _create_inverse_shape_mapping_kernel,
            dim=(nworld, ngeom),
            inputs=[
                self.mjc_geom_to_source_shape,
            ],
            outputs=[
                self.source_shape_to_mjc_geom,
            ],
            device=self.device,
        )

    def _convert_contacts_to_mjwarp(self, source_state, contacts):
        """Write externally generated contacts into MuJoCo warp contact buffers.

        This path is only used when the solver is configured to bypass MuJoCo's
        own collision detection and consume contacts prepared by WanPhys.
        """
        if self.mjw_model is None or self.mjw_data is None:
            return

        if contacts is None:
            return

        if self.source_shape_to_mjc_geom is None:
            self._create_inverse_shape_mapping()

        model = self._source_model
        data = self.mjw_data
        mjw_model = self.mjw_model

        bodies_per_world = model.body_count // model.world_count

        wp.launch(
                convert_newton_contacts_to_mjwarp_kernel,
                dim=(contacts.rigid_contact_max,),
                inputs=[
                    source_state.body_q,
                    model.shape_body,
                mjw_model.geom_condim,
                mjw_model.geom_priority,
                mjw_model.geom_solmix,
                mjw_model.geom_solref,
                mjw_model.geom_solimp,
                mjw_model.geom_friction,
                mjw_model.geom_margin,
                mjw_model.geom_gap,
                # External contact arrays provided by the upstream rigid pipeline.
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                model.shape_margin,
                bodies_per_world,
                self.source_shape_to_mjc_geom,
                # MuJoCo warp contact storage to be filled for the current step.
                data.naconmax,
                data.nacon,
                data.contact.dist,
                data.contact.pos,
                data.contact.frame,
                data.contact.includemargin,
                data.contact.friction,
                data.contact.solref,
                data.contact.solreffriction,
                data.contact.solimp,
                data.contact.dim,
                data.contact.geom,
                data.contact.worldid,
                # Auxiliary counters that must stay in sync with the rewritten contacts.
                data.nworld,
                data.ncollision,
            ],
            device=self.device,
        )

    def update_contacts(self, contacts):
        """Export MuJoCo-generated contacts into WanPhys-facing contact arrays."""
        if self.mjw_data is None or self.mjw_model is None:
            return

        if contacts is None:
            return

        data = self.mjw_data
        mj_contact = data.contact
        naconmax = data.naconmax

        if naconmax > contacts.rigid_contact_max:
            raise ValueError(
                f"MuJoCo naconmax ({naconmax}) exceeds contacts.rigid_contact_max "
                f"({contacts.rigid_contact_max}). Create Contacts with at least "
                f"rigid_contact_max={naconmax}."
            )

        wp.launch(
            convert_mjw_contact_to_warp_kernel,
            dim=naconmax,
            inputs=[
                self.mjc_geom_to_source_shape,
                self.mjw_model.opt.cone == int(self._mujoco.mjtCone.mjCONE_PYRAMIDAL),
                data.nacon,
                mj_contact.frame,
                mj_contact.dim,
                mj_contact.geom,
                mj_contact.efc_address,
                mj_contact.worldid,
                data.efc.force,
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.force,
            ],
            device=self.device,
        )

        contacts.n_contacts = data.nacon
