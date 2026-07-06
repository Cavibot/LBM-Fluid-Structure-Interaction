# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any, Callable, Dict, Optional, Type

import warp as wp
import os
from pathlib import Path
import re
import yaml

from .sensor_contact import SensorContact
from .sensor_imu import SensorIMU
from .sensor_frame_transform import SensorFrameTransform
from .sensor_tiled_camera import SensorTiledCamera
from newton import Model as NewtonModel

from wanphys.core import DomainModel, DomainState, Domain
from wanphys.rigid import RigidDomain

def _resolve_match_fn(spec) -> Optional[Callable]:
    """
    spec:
      - None / "none" / "" -> None
      - "re_match" -> lambda s, pat: re.match(pat, s)
      - "re_fullmatch" -> lambda s, pat: re.fullmatch(pat, s)
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in ("", "none", "null"):
            return None
        if key in ("re_match", "regex", "re"):
            return lambda string, pat: re.match(pat, string) is not None
        if key in ("re_fullmatch", "fullmatch"):
            return lambda string, pat: re.fullmatch(pat, string) is not None
    raise ValueError(f"Unsupported match_fn spec in yaml: {spec!r}")

@dataclass(frozen=True)
class _SensorEntry:
    cls: Type[Any]
    preprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

def _preprocess_match_func(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    if "match_fn" in params:
        params["match_fn"] = _resolve_match_fn(params["match_fn"])
    return params

SENSOR_REGISTRY: Dict[str, _SensorEntry] = {
    "contact": _SensorEntry(SensorContact, _preprocess_match_func),
    "sensorcontact": _SensorEntry(SensorContact, _preprocess_match_func),
    "sensor_contact": _SensorEntry(SensorContact, _preprocess_match_func),
    "imu": _SensorEntry(SensorIMU),
    "sensorimu": _SensorEntry(SensorIMU),
    "sensor_imu": _SensorEntry(SensorIMU),
    "frame_transform": _SensorEntry(SensorFrameTransform),
    "sensorframetransform": _SensorEntry(SensorFrameTransform),
    "sensor_frame_transform": _SensorEntry(SensorFrameTransform),
    "tiled_camera": _SensorEntry(SensorTiledCamera),
    "sensortiledcamera": _SensorEntry(SensorTiledCamera),
    "sensor_tiled_camera": _SensorEntry(SensorTiledCamera),
}


class Sensor:
    """
    A thin wrapper that loads a yaml config and instantiates the real sensor into self.sensor.
    """

    def __init__(self, domain: RigidDomain, file_path: str, extra_params: dict | None = None):
        self.file_path = Path(file_path)

        cfg = self._load_yaml(self.file_path)
        sensor_type = str(cfg.get("type", "")).strip()
        if not sensor_type:
            raise ValueError(f"Sensor yaml missing 'type': {self.file_path}")

        entry = SENSOR_REGISTRY.get(sensor_type.lower())
        if entry is None:
            supported = ", ".join(sorted(SENSOR_REGISTRY.keys()))
            raise ValueError(f"Unknown sensor type {sensor_type!r}. Supported: {supported}")

        params = cfg.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be a dict in {self.file_path}, got {type(params)}")

        if entry.preprocess is not None:
            params = entry.preprocess(params)

        if extra_params is not None:
            if not isinstance(extra_params, dict):
                raise TypeError(f"'extra_params' must be a dict, got {type(extra_params)}")
            params = {**params, **extra_params}

        self.sensor = entry.cls(domain, **params)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Top-level yaml must be a dict in {path}, got {type(data)}")
        return data
