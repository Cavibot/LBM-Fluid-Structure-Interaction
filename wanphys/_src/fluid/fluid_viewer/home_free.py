# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only HOME-Free state adapter for the OpenGL fluid viewer."""

from __future__ import annotations

import warp as wp


@wp.kernel
def _pack_home_free_fill(
    fill_level: wp.array3d(dtype=float),
    flags: wp.array3d(dtype=wp.int32),
    gas_flag: int,
    source_up_axis: int,
    density: wp.array3d(dtype=float),
) -> None:
    x, y, z = wp.tid()
    value = 0.0
    if flags[x, y, z] != gas_flag:
        value = wp.clamp(fill_level[x, y, z], 0.0, 1.0)

    if source_up_axis == 1:
        # HOME rebuild scenes use y-up; ViewerGL uses z-up.
        density[x, z, y] = value
    else:
        # The original HOME-LBM/FSI domains already use z-up.
        density[x, y, z] = value


class HomeFreeRenderField:
    """Build a z-up density copy without mutating HOME-Free state."""

    def __init__(
        self,
        source_shape: tuple[int, int, int],
        *,
        device: str | wp.Device,
        gas_flag: int = 0,
        source_up_axis: int = 1,
    ) -> None:
        if len(source_shape) != 3 or any(int(value) < 1 for value in source_shape):
            raise ValueError("source_shape must contain three positive dimensions")
        self.source_shape = tuple(int(value) for value in source_shape)
        if source_up_axis not in (1, 2):
            raise ValueError("source_up_axis must be 1 (y-up) or 2 (z-up)")
        self.source_up_axis = int(source_up_axis)
        self.render_shape = (
            (
                self.source_shape[0],
                self.source_shape[2],
                self.source_shape[1],
            )
            if self.source_up_axis == 1
            else self.source_shape
        )
        self.device = wp.get_device(device)
        self.gas_flag = int(gas_flag)
        self.density = wp.zeros(
            self.render_shape, dtype=float, device=self.device
        )

    def update(
        self,
        fill_level: wp.array,
        flags: wp.array,
    ) -> wp.array:
        """Update and return the render-only density field."""
        if tuple(fill_level.shape) != self.source_shape:
            raise ValueError("fill_level shape does not match the render adapter")
        if tuple(flags.shape) != self.source_shape:
            raise ValueError("flags shape does not match the render adapter")
        if fill_level.device != self.device or flags.device != self.device:
            raise ValueError("HOME-Free state and render adapter must share a device")

        wp.launch(
            _pack_home_free_fill,
            dim=self.source_shape,
            inputs=[fill_level, flags, self.gas_flag, self.source_up_axis],
            outputs=[self.density],
            device=self.device,
            record_tape=False,
        )
        return self.density
