# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark variant registry for LBM comparison harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class VariantSpec:
    """Description of one benchmarkable LBM configuration."""

    variant_id: str
    name: str
    lattice: str
    backend: str
    phase: str
    description: str = ""
    default_config: dict[str, Any] = field(default_factory=dict)


_VARIANTS: dict[str, VariantSpec] = {}


def register_variant(spec: VariantSpec, *, overwrite: bool = False) -> None:
    """Register a benchmark variant."""
    key = spec.variant_id.upper()
    if key in _VARIANTS and not overwrite:
        raise ValueError(f"Variant {key!r} is already registered")
    _VARIANTS[key] = spec


def get_variant(variant_id: str) -> VariantSpec:
    """Look up a registered variant by id (case-insensitive)."""
    key = str(variant_id).upper()
    try:
        return _VARIANTS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_VARIANTS))
        raise KeyError(
            f"Unknown benchmark variant {variant_id!r}.  Known: {known}"
        ) from exc


def list_variants() -> list[VariantSpec]:
    """Return all registered variants sorted by id."""
    return [_VARIANTS[k] for k in sorted(_VARIANTS)]


def _register_builtin_variants() -> None:
    register_variant(
        VariantSpec(
            variant_id="V0",
            name="dist_d3q19_sc",
            lattice="D3Q19",
            backend="dist",
            phase="shan_chen",
            description="D3Q19 distribution LBM with TRT/SC (G1 baseline).",
            default_config={
                "tau": 0.51,
                "lambda_trt": 0.001,
                "G": -5.0,
                "psi_type": 1,
                "use_regularization": True,
                "omega_reg": 0.01,
            },
        ),
        overwrite=True,
    )
    register_variant(
        VariantSpec(
            variant_id="V1",
            name="dist_d3q27_sc",
            lattice="D3Q27",
            backend="dist",
            phase="shan_chen",
            description="D3Q27 distribution LBM with Shan-Chen (G2).",
        ),
        overwrite=True,
    )
    register_variant(
        VariantSpec(
            variant_id="V2",
            name="home_quant_d3q27",
            lattice="D3Q27",
            backend="home_quant",
            phase="none",
            description="Stability-guided quantized HOME-LBM (G3).",
        ),
        overwrite=True,
    )
    register_variant(
        VariantSpec(
            variant_id="V3",
            name="home_free_quant_d3q27",
            lattice="D3Q27",
            backend="home_quant",
            phase="vof_sharp",
            description="Quantized HOME-FREE sharp-interface LBM (G4).",
        ),
        overwrite=True,
    )


_register_builtin_variants()

# Type alias for future runners that construct domains from a variant spec.
VariantFactory = Callable[[VariantSpec], Any]
