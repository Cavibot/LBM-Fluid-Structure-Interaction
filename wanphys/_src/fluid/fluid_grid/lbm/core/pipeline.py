# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Step pipeline types for LBM solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepStats:
    """Wall-clock timing breakdown for one LBM step (milliseconds)."""

    ms_total: float = 0.0
    ms_moments: float = 0.0
    ms_phase: float = 0.0
    ms_regularization: float = 0.0
    ms_collision: float = 0.0
    ms_bc: float = 0.0
    ms_body_force: float = 0.0
    ms_export: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize timings for JSON benchmark reports."""
        return {
            "ms_total": self.ms_total,
            "ms_moments": self.ms_moments,
            "ms_phase": self.ms_phase,
            "ms_regularization": self.ms_regularization,
            "ms_collision": self.ms_collision,
            "ms_bc": self.ms_bc,
            "ms_body_force": self.ms_body_force,
            "ms_export": self.ms_export,
        }

    @property
    def mlups(self) -> float:
        """Mega lattice updates per second (requires ``num_cells`` to be set)."""
        cells = float(getattr(self, "_num_cells", 0))
        if cells <= 0.0 or self.ms_total <= 0.0:
            return 0.0
        return cells / (self.ms_total * 1.0e-3) / 1.0e6

    def with_num_cells(self, num_cells: int) -> StepStats:
        """Attach grid size so :meth:`mlups` can be computed."""
        self._num_cells = int(num_cells)
        return self


@dataclass
class LbmStepControl:
    """Optional per-step control passed to :meth:`LbmSolver.step`.

    Parameters
    ----------
    collect_stats:
        When ``True`` (default), the solver returns a populated
        :class:`StepStats` instance.
    stats_out:
        If provided, the solver writes the latest stats into this object
        in addition to returning them.
    """

    collect_stats: bool = True
    stats_out: StepStats | None = None
    extra: dict[str, Any] = field(default_factory=dict)
