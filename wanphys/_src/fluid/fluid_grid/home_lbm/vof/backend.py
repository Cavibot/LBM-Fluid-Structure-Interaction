# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit production and compatibility backend selection for HOME-Free."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..model import HomeLbmModel
from .domain import HomeFreeDomain as HomeFreeLegacyDomain
from .geometric_domain import HomeFreeGeometricDomain


class HomeFreeBackend(str, Enum):
    GEOMETRIC = "geometric"
    LEGACY = "legacy"


def create_home_free_domain(
    model: HomeLbmModel,
    *,
    backend: HomeFreeBackend | str = HomeFreeBackend.GEOMETRIC,
    **kwargs: Any,
) -> HomeFreeGeometricDomain | HomeFreeLegacyDomain:
    """Construct an explicit HOME-Free backend; geometric OM is the default."""

    try:
        selected = HomeFreeBackend(backend)
    except ValueError as error:
        choices = ", ".join(item.value for item in HomeFreeBackend)
        raise ValueError(
            f"unknown HOME-Free backend {backend!r}; choose one of {choices}"
        ) from error
    if selected is HomeFreeBackend.GEOMETRIC:
        return HomeFreeGeometricDomain(model, **kwargs)
    return HomeFreeLegacyDomain(model, **kwargs)
