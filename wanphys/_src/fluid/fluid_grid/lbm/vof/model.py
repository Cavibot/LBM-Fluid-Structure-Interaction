"""Static VOF physical and topology configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import VofMassScheme


@dataclass(frozen=True)
class VofModel:
    """Minimal authoritative VOF configuration extracted from LBM config."""

    mass_scheme: VofMassScheme = VofMassScheme.FSLBM_NEIGHBOR
    atmosphere_pressure: float = 1.0
    surface_tension: float = 0.0
    transition_epsilon: float = 1.0e-6


__all__ = ["VofModel"]
