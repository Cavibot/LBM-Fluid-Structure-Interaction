# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod

class CollisionBackend(ABC):
    """Base class for collision backends (candidate generation + narrowphase)."""

    name: str = "base"

    @abstractmethod
    def build(self, model, device):
        """Initialize backend-specific data structures."""

    @abstractmethod
    def refit(self, state):
        """Update acceleration structures with new state."""

    @abstractmethod
    def generate_candidates(self, state, params, dt: float, out_pairs, out_pair_count) -> None:
        """Generate candidate pairs for narrowphase."""

    @abstractmethod
    def narrow_phase(self, state, params, dt: float, pairs, pair_count, out_hits, mode: str) -> None:
        """Compute contact geometry for candidate pairs."""