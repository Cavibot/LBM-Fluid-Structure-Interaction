# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Martin & Moyce (1952) dam-break surge-front reference data.

Source
------
J. C. Martin & W. J. Moyce, *Part IV. An experimental study of the collapse of
liquid columns on a rigid horizontal plane*, Phil. Trans. R. Soc. Lond. A
244(882):312–324 (1952). doi:10.1098/rsta.1952.0006

Digitized (τ, δ) pairs below match the commonly used n²=2 rectangular-column
set as published in the Lethe multiphase dam-break example post-processor
(Apache-2.0 / LLVM-exception), which cites Martin & Moyce 1952.

Nondimensionalisation (Martin corner / plane-symmetry notation)
---------------------------------------------------------------
- ``a``  — column base width (reservoir width against the wall)
- ``H₀ = n² a`` — initial column height
- ``Z = z / a`` — surge-front distance from the wall, in units of ``a``
- ``T = n t √(g / a)`` — nondimensional time (``n = √(n²)``)

For **n² = 2** (``H₀ / a = 2``)::

    T = t √(2 g / a),   Z = z / a

This matches the usual HOME-FREE dam-break seed
``dam_x / fill_z ≈ 1/2`` (width:height = 1:2).

Expect **qualitative** agreement only: LBM-VOF ≠ experiment (viscosity,
surface tension, 3D sidewall, gate release). Early ``T`` is especially
sensitive; Martin normalised records at a finite ``Z``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# (T, Z) for rectangular n²=2 — Lethe digitization of Martin & Moyce 1952.
# T = t√(2g/a), Z = z/a.  Initial front at Z=1.
MARTIN_MOYCE_N2_2: tuple[tuple[float, float], ...] = (
    (0.00, 1.00),
    (0.41, 1.11),
    (0.84, 1.23),
    (1.19, 1.44),
    (1.43, 1.67),
    (1.63, 1.89),
    (1.82, 2.11),
    (1.97, 2.33),
    (2.20, 2.56),
    (2.32, 2.78),
    (2.50, 3.00),
    (2.64, 3.22),
    (2.82, 3.44),
    (2.96, 3.67),
)

# Table-1 style n²=1 means (T, Z) reconstructed from Martin OCR means
# (a≈2.25 in branch; T normalised at Z=1.44). Sparse / coarser than n²=2.
MARTIN_MOYCE_N2_1: tuple[tuple[float, float], ...] = (
    (0.80, 1.44),
    (0.97, 1.67),
    (1.14, 1.89),
    (1.29, 2.11),
    (1.45, 2.33),
    (1.61, 2.56),
    (1.76, 2.78),
    (1.94, 3.00),
    (2.07, 3.22),
    (2.24, 3.44),
    (2.40, 3.67),
    (2.54, 3.89),
    (2.71, 4.11),
    (2.87, 4.33),
    (3.04, 4.56),
)


@dataclass(frozen=True)
class MartinMoyceScale:
    """Lattice → Martin nondimensional map for a rectangular column."""

    a_cells: float
    height_cells: float
    g_lattice: float
    tau: float = 0.51

    @property
    def n2(self) -> float:
        return float(self.height_cells) / max(float(self.a_cells), 1.0e-12)

    @property
    def n(self) -> float:
        return math.sqrt(max(self.n2, 0.0))

    @property
    def nu_lattice(self) -> float:
        """BGK kinematic viscosity ``ν = (τ − 1/2) / 3`` (D3Q27 / D3Q19 cs²=1/3)."""
        return (float(self.tau) - 0.5) / 3.0

    @property
    def Re_column(self) -> float:
        """Rough column Reynolds ``√(g H₀³) / ν`` (lattice units)."""
        h = max(float(self.height_cells), 1.0e-12)
        g = max(float(self.g_lattice), 0.0)
        nu = max(self.nu_lattice, 1.0e-12)
        return math.sqrt(g * h * h * h) / nu

    @property
    def Fr_column(self) -> float:
        """Froude based on ``√(g H₀)`` scale (always 1 by construction of U=√(gH))."""
        return 1.0

    def T_from_steps(self, steps: int) -> float:
        """Nondimensional time after ``steps`` lattice timesteps (Δt=1)."""
        a = max(float(self.a_cells), 1.0e-12)
        g = max(float(self.g_lattice), 0.0)
        return self.n * float(steps) * math.sqrt(g / a)

    def Z_from_front_cell(self, front_i: float) -> float:
        """Front cell index (0-based, wall at 0) → Z = z/a."""
        return float(front_i) / max(float(self.a_cells), 1.0e-12)

    def diagnostics(self) -> dict[str, float]:
        return {
            "n2": self.n2,
            "a_cells": float(self.a_cells),
            "H_cells": float(self.height_cells),
            "g": float(self.g_lattice),
            "tau": float(self.tau),
            "nu": self.nu_lattice,
            "Re_column": self.Re_column,
        }


def reference_arrays(n2: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(T, Z)`` float arrays for the nearest tabulated aspect."""
    if abs(float(n2) - 1.0) < 0.25:
        pairs = MARTIN_MOYCE_N2_1
    else:
        pairs = MARTIN_MOYCE_N2_2
    t = np.asarray([p[0] for p in pairs], dtype=np.float64)
    z = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return t, z


def interpolate_Z_at_T(
    t_sim: np.ndarray,
    z_sim: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """Linear interpolate simulated Z(T) onto query times (clip outside)."""
    if t_sim.size < 2:
        return np.full_like(t_query, np.nan, dtype=np.float64)
    order = np.argsort(t_sim)
    return np.interp(
        t_query.astype(np.float64),
        t_sim[order],
        z_sim[order],
        left=np.nan,
        right=np.nan,
    )


def align_T_to_anchor(
    t_sim: np.ndarray,
    z_sim: np.ndarray,
    *,
    z_anchor: float = 1.44,
    t_anchor: float = 1.19,
) -> np.ndarray:
    """Shift simulated time so the front crosses ``z_anchor`` at ``t_anchor``.

    Martin & Moyce normalised cine records at a finite spread (often near
    ``Z≈1.44`` for n²=2 digitizations). Matching that gate-release offset is
    standard for CFD comparisons (cf. Lethe ``time_correction``).
    """
    t = np.asarray(t_sim, dtype=np.float64)
    z = np.asarray(z_sim, dtype=np.float64)
    if t.size < 2:
        return t.copy()
    order = np.argsort(t)
    t_o = t[order]
    z_o = z[order]
    # First crossing of z_anchor (linear within segment).
    t_cross = float("nan")
    for i in range(len(z_o) - 1):
        z0, z1 = float(z_o[i]), float(z_o[i + 1])
        if z0 <= float(z_anchor) <= z1 or z1 <= float(z_anchor) <= z0:
            if abs(z1 - z0) < 1.0e-12:
                t_cross = float(t_o[i])
            else:
                alpha = (float(z_anchor) - z0) / (z1 - z0)
                t_cross = float(t_o[i]) + alpha * (float(t_o[i + 1]) - float(t_o[i]))
            break
    if not math.isfinite(t_cross):
        return t.copy()
    return t + (float(t_anchor) - t_cross)


def compare_front_to_martin(
    t_sim: np.ndarray,
    z_sim: np.ndarray,
    *,
    n2: float = 2.0,
    t_min: float = 0.8,
    t_max: float = 2.8,
    align_gate: bool = True,
) -> dict[str, float]:
    """Compare simulated front to Martin–Moyce on overlapping mid-range times.

    Skips the earliest gate-release window (``t < t_min``) where experiments
    were time-normalised and numerics differ most.  When ``align_gate`` is
    True, shifts ``T`` so ``Z=1.44`` matches the tabulated anchor time.
    """
    t_use = np.asarray(t_sim, dtype=np.float64)
    z_use = np.asarray(z_sim, dtype=np.float64)
    if align_gate:
        t_ref0, z_ref0 = reference_arrays(n2)
        # Prefer tabulated (T,Z)=(1.19,1.44) when present.
        t_anchor = 1.19
        z_anchor = 1.44
        for ti, zi in zip(t_ref0.tolist(), z_ref0.tolist(), strict=True):
            if abs(zi - 1.44) < 1.0e-6:
                t_anchor = float(ti)
                z_anchor = float(zi)
                break
        t_use = align_T_to_anchor(
            t_use, z_use, z_anchor=z_anchor, t_anchor=t_anchor
        )

    t_ref, z_ref = reference_arrays(n2)
    mask = (t_ref >= float(t_min)) & (t_ref <= float(t_max))
    t_q = t_ref[mask]
    z_q = z_ref[mask]
    z_s = interpolate_Z_at_T(t_use, z_use, t_q)
    valid = np.isfinite(z_s)
    if not np.any(valid):
        return {
            "n_points": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_abs": float("nan"),
            "mean_rel": float("nan"),
            "aligned": 1.0 if align_gate else 0.0,
        }
    err = z_s[valid] - z_q[valid]
    rel = np.abs(err) / np.maximum(np.abs(z_q[valid]), 1.0e-6)
    return {
        "n_points": float(valid.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs": float(np.max(np.abs(err))),
        "mean_rel": float(np.mean(rel)),
        "aligned": 1.0 if align_gate else 0.0,
    }
