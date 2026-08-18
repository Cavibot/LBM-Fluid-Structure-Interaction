# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase-5 stability probes: bubble volume / rho / COM / disjoin snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StabilitySnapshot:
    """One-step diagnostics for volume conservation / foam stability."""

    step: int
    bubble_count: int
    volumes: np.ndarray
    init_volumes: np.ndarray
    rhos: np.ndarray
    com_distance: float
    sum_abs_disjoin: float
    merge_flag: int


@dataclass
class StabilitySeries:
    """Time series of :class:`StabilitySnapshot` with summary helpers."""

    snaps: list[StabilitySnapshot] = field(default_factory=list)

    def append(self, snap: StabilitySnapshot) -> None:
        self.snaps.append(snap)

    @property
    def volume_matrix(self) -> np.ndarray:
        """Shape ``(T, B_max)`` padded with nan."""
        if not self.snaps:
            return np.zeros((0, 0))
        bmax = max(int(s.bubble_count) for s in self.snaps)
        bmax = max(bmax, 1)
        out = np.full((len(self.snaps), bmax), np.nan, dtype=np.float64)
        for t, s in enumerate(self.snaps):
            n = min(int(s.bubble_count), bmax)
            if n > 0:
                out[t, :n] = np.asarray(s.volumes[:n], dtype=np.float64)
        return out

    def relative_volume_drift(self, bubble_index: int = 0) -> float:
        """``|V_T - V_0| / max(|V_0|, eps)`` for one bubble id (0-based)."""
        mat = self.volume_matrix
        if mat.shape[0] < 2 or bubble_index >= mat.shape[1]:
            return float("nan")
        v0 = mat[0, bubble_index]
        vt = mat[-1, bubble_index]
        if not np.isfinite(v0) or abs(v0) < 1e-30:
            return float("nan")
        return float(abs(vt - v0) / abs(v0))

    def volume_amplitude(self, bubble_index: int = 0) -> float:
        """``max(V) - min(V)`` over the series for one bubble."""
        mat = self.volume_matrix
        if mat.size == 0 or bubble_index >= mat.shape[1]:
            return float("nan")
        col = mat[:, bubble_index]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return float("nan")
        return float(np.max(col) - np.min(col))

    def max_com_distance(self) -> float:
        vals = [s.com_distance for s in self.snaps if s.com_distance >= 0.0]
        return float(max(vals)) if vals else float("nan")


def _tag_coms(tag: np.ndarray) -> list[np.ndarray]:
    ids = sorted(int(x) for x in np.unique(tag) if x > 0)
    return [np.argwhere(tag == tid).mean(axis=0) for tid in ids]


def capture_snapshot(state, step: int, *, sum_abs_disjoin: float | None = None) -> StabilitySnapshot:
    """Read host-side bubble diagnostics from a :class:`HomeFslbmState`."""
    bc = int(state.bubble_count)
    vol = state.bubble_volume.numpy()
    init = state.bubble_init_volume.numpy()
    rho = state.bubble_rho.numpy()
    volumes = vol[:bc].copy() if bc > 0 else np.zeros(0, dtype=np.float64)
    inits = init[:bc].copy() if bc > 0 else np.zeros(0, dtype=np.float64)
    rhos = rho[:bc].copy() if bc > 0 else np.zeros(0, dtype=np.float64)

    tag = state.tag_matrix.numpy()
    coms = _tag_coms(tag)
    if len(coms) >= 2:
        dist = float(np.linalg.norm(coms[0] - coms[1]))
    else:
        dist = -1.0

    if sum_abs_disjoin is None:
        sum_abs_disjoin = float(np.abs(state.disjoin_force.numpy()).sum())

    return StabilitySnapshot(
        step=step,
        bubble_count=bc,
        volumes=volumes,
        init_volumes=inits,
        rhos=rhos,
        com_distance=dist,
        sum_abs_disjoin=float(sum_abs_disjoin),
        merge_flag=int(state.merge_flag),
    )


def paint_soft_bubble(flag, phi, mass, tag, nx, ny, nz, cx, cy, cz, radius, tid: int = 1):
    """Soft gas core + interface shell into numpy arrays (Warp i-fastest 3d)."""
    from . import constants as C

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                d = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                if d < radius - 0.5:
                    flag[i, j, k] = C.TYPE_G
                    phi[i, j, k] = 0.0
                    mass[i, j, k] = 0.0
                    tag[i, j, k] = tid
                elif d <= radius + 0.5:
                    flag[i, j, k] = C.TYPE_I
                    phi[i, j, k] = 0.5
                    mass[i, j, k] = 0.5
                    tag[i, j, k] = tid


def set_solid_walls(flag, phi, mass, nx, ny, nz):
    from . import constants as C

    for a, v in ((flag, C.TYPE_S), (phi, 0.0), (mass, 0.0)):
        a[0, :, :] = v
        a[-1, :, :] = v
        a[:, 0, :] = v
        a[:, -1, :] = v
        a[:, :, 0] = v
        a[:, :, -1] = v


def sync_double_buffer(domain, names: tuple[str, ...] | None = None) -> None:
    """Copy state_in → state_out for listed fields after IC setup."""
    import warp as wp

    src = domain.state
    dst = domain._state_out
    assert dst is not None
    if src.shares_buffers_with(dst):
        return
    if names is None:
        names = (
            "f_mom", "f_mom_post", "flag", "mass", "massex", "phi",
            "tag_matrix", "previous_tag", "previous_merge_tag",
            "bubble_volume", "bubble_init_volume", "bubble_rho",
            "g_mom", "g_mom_post", "c_value", "src", "delta_g",
            "disjoin_force", "label_matrix", "input_matrix", "merge_detector",
            "force_x", "force_y", "force_z", "islet",
        )
    for name in names:
        wp.copy(getattr(dst, name), getattr(src, name))
    dst.bubble_count = src.bubble_count
    dst.label_num = src.label_num
