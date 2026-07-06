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

"""Abstract base class for XPBD constraint groups."""

from __future__ import annotations

from abc import ABC, abstractmethod

# from ...sim import Contacts, Model
from .context import ConstraintPhase, XPBDContext


class ConstraintGroup(ABC):
    """Base class for an XPBD constraint group.

    Every distinct physical constraint type (e.g. springs, rigid-body joints,
    particle–shape contacts …) is implemented as a concrete subclass.  The
    solver discovers active constraints via :meth:`is_active`, then drives
    their lifecycle through:

    1. :meth:`initialize`       – once per step (allocate / cache state)
    2. :meth:`reset_iteration`  – once per iteration (zero lambda accumulators)
    3. :meth:`project`          – once per iteration (accumulate deltas)
    4. :meth:`apply_restitution` – once after all iterations (optional)

    Subclasses **must** set the :attr:`phase` class attribute to one of the
    :class:`ConstraintPhase` values so the solver can schedule delta zeroing
    and application correctly.

    Example
    -------

    .. code-block:: python

        class SpringConstraint(ConstraintGroup):
            phase = ConstraintPhase.PARTICLE

            def __init__(self, relaxation: float = 0.9):
                self.relaxation = relaxation
                self._lambdas = None

            def is_active(self, model, contacts):
                return model.spring_count > 0

            def initialize(self, ctx):
                self._lambdas = wp.zeros_like(ctx.model.spring_rest_length)

            def reset_iteration(self, ctx, iteration):
                self._lambdas.zero_()

            def project(self, ctx, iteration):
                wp.launch(
                    kernel=solve_springs,
                    dim=ctx.model.spring_count,
                    inputs=[...],
                    outputs=[ctx.particle_deltas],
                    device=ctx.model.device,
                )
    """

    # ── Subclass must override ──────────────────────────────────────────

    phase: ConstraintPhase = ConstraintPhase.PARTICLE
    """Execution phase that determines when deltas are zeroed / applied."""

    # ── Lifecycle methods ───────────────────────────────────────────────

    @abstractmethod
    def is_active(self, model, contacts) -> bool:
        """Return whether this constraint group participates in the current step.

        Called once at the beginning of :py:meth:`SolverXPBD.step`.  If it
        returns ``False``, :meth:`initialize`, :meth:`project`, and
        :meth:`apply_restitution` will all be skipped for this step.

        Args:
            model: The simulation model.
            contacts: Contact information (may be ``None``).
        """
        ...

    def initialize(self, ctx: XPBDContext) -> None:
        """Per-step initialization (allocate buffers, snapshot state, etc.).

        Called once before the iteration loop.  Use this to create or resize
        internal accumulators (e.g. Lagrange multiplier arrays) and to perform
        any pre-integration work (e.g. applying joint forces to ``body_f``).

        The default implementation does nothing.

        Args:
            ctx: The shared XPBD context for the current step.
        """

    def reset_iteration(self, ctx: XPBDContext, iteration: int) -> None:
        """Per-iteration reset (e.g. zero out lambda accumulators).

        Called at the start of every Gauss–Seidel iteration, *before*
        :meth:`project`.  The default implementation does nothing.

        Args:
            ctx: The shared XPBD context.
            iteration: Zero-based iteration index.
        """

    @abstractmethod
    def project(self, ctx: XPBDContext, iteration: int) -> None:
        """Project this constraint — accumulate correction deltas.

        This is the core method invoked once per iteration.  Implementations
        should launch the appropriate Warp kernel(s) that write into
        ``ctx.particle_deltas`` and / or ``ctx.body_deltas`` via atomic adds.

        Args:
            ctx: The shared XPBD context (read ``particle_q`` / ``body_q``,
                 write ``particle_deltas`` / ``body_deltas``).
            iteration: Zero-based iteration index.
        """
        ...

    def stash_lambdas(self, ctx: XPBDContext) -> None:
        """End-of-step: save accumulated lambdas for warm-starting the next step.

        Called once after all iterations have finished.  Implementations should
        copy the current Lagrange multiplier buffer to a persistent store so
        that :meth:`reset_iteration` can use it as the initial guess on the
        next time step.

        The default implementation does nothing.

        Args:
            ctx: The shared XPBD context.
        """

    def apply_restitution(self, ctx: XPBDContext) -> None:
        """Post-iteration restitution pass (optional).

        Called once after all iterations have finished, only when restitution
        is enabled in the solver.  The default implementation does nothing.

        Args:
            ctx: The shared XPBD context.
        """
