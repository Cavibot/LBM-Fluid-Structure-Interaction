"""Diagnostics policy configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import VofRuntimeProfile


@dataclass(frozen=True)
class VofDiagnosticsConfig:
    runtime_profile: VofRuntimeProfile = VofRuntimeProfile.SAMPLED
    validation_interval: int = 1


__all__ = ["VofDiagnosticsConfig", "VofRuntimeProfile"]
