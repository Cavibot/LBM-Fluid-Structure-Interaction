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

"""XPBD solver shared context and constraint execution phases."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import warp as wp

from ..model import RigidModel
from ..state import RigidState

class ConstraintPhase(IntEnum):
    """Constraint execution phase within each XPBD iteration.

    The XPBD iteration loop applies deltas in three ordered phases.
    Each :class:`ConstraintGroup` declares which phase it belongs to so the
    solver can orchestrate the correct zero/accumulate/apply sequence.

    Phase ordering within one iteration::

        PARTICLE      →  accumulate particle_deltas (+ body_deltas from
                         particle-shape interactions), then apply_particle_deltas
        BODY_JOINT    →  accumulate body_deltas (on top of Phase 0 residuals),
                         then apply_body_deltas
        RIGID_CONTACT →  zero body_deltas, accumulate from rigid contacts,
                         then apply_body_deltas with inverse-weight
    """

    PARTICLE = 0
    BODY_JOINT = 1
    RIGID_CONTACT = 2


class XPBDContext:
    """Per-step shared working state for the XPBD solver.

    An :class:`XPBDContext` is created at the beginning of every
    :py:meth:`SolverXPBD.step` call and passed to each active
    :class:`ConstraintGroup`.  It holds references to the model, states,
    control, contacts, time-step size, and all working buffers so that
    constraint implementations do not need bloated argument lists.

    The solver owns the lifecycle of all ``wp.array`` buffers stored here;
    constraint groups should only *read from* or *atomically add to* them.
    """

    def __init__(
        self,
        model: RigidModel,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        self.model: RigidModel = model
        self.state_in: RigidState = state_in
        self.state_out: RigidState = state_out
        self.control: Any = control
        self.contacts: Any = contacts
        self.dt: float = dt
        self.requires_grad: bool = state_in.requires_grad

        # ── Current q / qd (swapped by the solver after apply_deltas) ──
        self.particle_q: wp.array | None = None
        self.particle_qd: wp.array | None = None
        self.body_q: wp.array | None = None
        self.body_qd: wp.array | None = None

        # ── Delta accumulators (managed by the solver) ──
        self.particle_deltas: wp.array | None = None
        self.body_deltas: wp.array | None = None

        # ── Rigid-contact specific ──
        self.rigid_contact_inv_weight: wp.array | None = None
        self.rigid_contact_inv_weight_init: wp.array | None = None

        # ── Initial-state snapshots (for restitution) ──
        self.particle_q_init: wp.array | None = None
        self.particle_qd_init: wp.array | None = None
        self.body_q_init: wp.array | None = None
        self.body_qd_init: wp.array | None = None
