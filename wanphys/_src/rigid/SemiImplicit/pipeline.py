# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Composable force evaluation pipeline.

The :class:`ForcePipeline` collects registered force modules and evaluates
them in order during each time-step.  This decouples the solver loop from
knowledge of which force contributions are active, making it easy to add,
remove, or reorder modules without touching the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class _ForceSlot:
    """Internal record for a registered force module."""

    name: str
    fn: Callable[..., None]
    enabled: bool = True


class ForcePipeline:
    """Ordered collection of force modules evaluated each step.

    Usage::

        pipe = ForcePipeline()
        pipe.register("spring_dashpot", apply_spring_dashpot)
        pipe.register("membrane_fem", apply_membrane_stress)
        ...
        pipe.evaluate(ctx)

    The *ctx* dict is forwarded to every module.  Each module function
    should accept ``**ctx`` or pick only the keys it needs.
    """

    def __init__(self):
        self._slots: List[_ForceSlot] = []
        self._index: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, fn: Callable[..., None], *, enabled: bool = True) -> None:
        """Append a force module to the end of the pipeline.

        Args:
            name: Unique identifier used for enable/disable.
            fn: Callable ``fn(ctx: dict) -> None`` that launches GPU kernels.
            enabled: Whether the module is active (default ``True``).
        """
        if name in self._index:
            raise ValueError(f"Force module '{name}' already registered")
        self._index[name] = len(self._slots)
        self._slots.append(_ForceSlot(name=name, fn=fn, enabled=enabled))

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        self._slots[self._index[name]].enabled = True

    def disable(self, name: str) -> None:
        self._slots[self._index[name]].enabled = False

    def is_enabled(self, name: str) -> bool:
        return self._slots[self._index[name]].enabled

    @property
    def module_names(self) -> List[str]:
        return [s.name for s in self._slots]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, ctx: Dict[str, Any]) -> None:
        """Run every enabled force module in registration order.

        Args:
            ctx: Dictionary forwarded to each module function.  Typically
                contains ``model``, ``state``, ``control``, ``contacts``,
                ``pforce``, ``body_wrench``, and solver parameters.
        """
        for slot in self._slots:
            if slot.enabled:
                slot.fn(ctx)
