"""Open and prescribed population boundary kernels."""

from __future__ import annotations

import warp as wp

from .common import equilibrium_f


@wp.kernel
def apply_boundary_conditions_kernel(
    f: wp.array(dtype=float),
    boundary_history: wp.array(dtype=float),
    bc_types: wp.array(dtype=wp.int32),
    bc_vel_x: wp.array(dtype=float),
    bc_vel_y: wp.array(dtype=float),
    bc_vel_z: wp.array(dtype=float),
    bc_density: wp.array(dtype=float),
    bc_convective_speed: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Complete Zou-He or history-based convective boundary populations.

    Launched over the entire grid.  Interior cells are early-return no-ops.
    Bounce-back faces (bc_type == 0) are no-ops because StreamingEngine and
    its transport link laws have already supplied their incoming values.
    """
    i, j, k = wp.tid()

    # Only operate on boundary cells
    on_xmin = (i == 0)
    on_xmax = (i == nx - 1)
    on_ymin = (j == 0)
    on_ymax = (j == ny - 1)
    on_zmin = (k == 0)
    on_zmax = (k == nz - 1)

    if not (on_xmin or on_xmax or on_ymin or on_ymax or on_zmin or on_zmax):
        return

    # Corner/edge cells: skip; BoundaryResolver reports them and transport
    # has already supplied the required static bounce-back fallback.
    face_count = 0
    if on_xmin:
        face_count += 1
    if on_xmax:
        face_count += 1
    if on_ymin:
        face_count += 1
    if on_ymax:
        face_count += 1
    if on_zmin:
        face_count += 1
    if on_zmax:
        face_count += 1
    if face_count != 1:
        return

    idx = i * ny * nz + j * nz + k

    # ===================================================================
    # Face 0: x-min  (i == 0)
    # ===================================================================
    if on_xmin:
        bc = bc_types[0]
        # --- Zou-He velocity inlet --------------------------------
        if bc == 1 or bc == 4:
            vx = bc_vel_x[0]
            vy = bc_vel_y[0]
            vz = bc_vel_z[0]
            denom = 1.0 - vx
            if denom <= 0.0:
                return
            # Known distribution sum for rho calculation
            known = (
                f[0 * stride + idx]
                + f[3 * stride + idx] + f[4 * stride + idx]
                + f[5 * stride + idx] + f[6 * stride + idx]
                + f[15 * stride + idx] + f[16 * stride + idx]
                + f[17 * stride + idx] + f[18 * stride + idx]
                + 2.0 * (
                    f[2 * stride + idx] + f[8 * stride + idx]
                    + f[10 * stride + idx] + f[12 * stride + idx]
                    + f[14 * stride + idx]
                )
            )
            rho_w = known / denom
            if bc == 4:
                rho_w = bc_density[0]
                vx = 1.0 - known / rho_w

            # Incoming: f1 (+x), f7 (+x+y), f9 (+x-y), f11 (+x+z), f13 (+x-z)
            # Bounce-back of non-equilibrium from opposite directions
            f[1 * stride + idx] = (
                equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 1, 0, 0)
                + (f[2 * stride + idx] - equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, -1, 0, 0))
            )
            f[7 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0)
                + (f[10 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0))
            )
            f[9 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0)
                + (f[8 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0))
            )
            f[11 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1)
                + (f[14 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1))
            )
            f[13 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1)
                + (f[12 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1))
            )
        # --- Convective outflow ---------------------------------
        elif bc == 2:
            src_idx = 1 * ny * nz + j * nz + k  # i=1
            courant = bc_convective_speed[0]
            for d in range(19):
                previous = boundary_history[d * stride + idx]
                interior = f[d * stride + src_idx]
                f[d * stride + idx] = previous + courant * (interior - previous)

    # ===================================================================
    # Face 1: x-max  (i == nx - 1)
    # ===================================================================
    if on_xmax:
        bc = bc_types[1]
        if bc == 1 or bc == 4:
            vx = bc_vel_x[1]
            vy = bc_vel_y[1]
            vz = bc_vel_z[1]
            denom = 1.0 + vx
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx]
                + f[3 * stride + idx] + f[4 * stride + idx]
                + f[5 * stride + idx] + f[6 * stride + idx]
                + f[15 * stride + idx] + f[16 * stride + idx]
                + f[17 * stride + idx] + f[18 * stride + idx]
                + 2.0 * (
                    f[1 * stride + idx] + f[7 * stride + idx]
                    + f[9 * stride + idx] + f[11 * stride + idx]
                    + f[13 * stride + idx]
                )
            )
            rho_w = known / denom
            if bc == 4:
                rho_w = bc_density[1]
                vx = known / rho_w - 1.0
            # Incoming: f2 (-x), f8 (-x+y), f10 (-x-y), f12 (-x+z), f14 (-x-z)
            f[2 * stride + idx] = (
                equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, -1, 0, 0)
                + (f[1 * stride + idx] - equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 1, 0, 0))
            )
            f[8 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0)
                + (f[9 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0))
            )
            f[10 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0)
                + (f[7 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0))
            )
            f[12 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1)
                + (f[13 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1))
            )
            f[14 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1)
                + (f[11 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1))
            )
        elif bc == 2:
            src_idx = (nx - 2) * ny * nz + j * nz + k
            courant = bc_convective_speed[1]
            for d in range(19):
                previous = boundary_history[d * stride + idx]
                interior = f[d * stride + src_idx]
                f[d * stride + idx] = previous + courant * (interior - previous)

    # ===================================================================
    # Face 2: y-min  (j == 0)
    # ===================================================================
    if on_ymin:
        bc = bc_types[2]
        if bc == 1 or bc == 4:
            vx = bc_vel_x[2]
            vy = bc_vel_y[2]
            vz = bc_vel_z[2]
            denom = 1.0 - vy
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx]
                + f[1 * stride + idx] + f[2 * stride + idx]
                + f[5 * stride + idx] + f[6 * stride + idx]
                + f[11 * stride + idx] + f[12 * stride + idx]
                + f[13 * stride + idx] + f[14 * stride + idx]
                + 2.0 * (
                    f[4 * stride + idx] + f[9 * stride + idx]
                    + f[10 * stride + idx] + f[16 * stride + idx]
                    + f[18 * stride + idx]
                )
            )
            rho_w = known / denom
            if bc == 4:
                rho_w = bc_density[2]
                vy = 1.0 - known / rho_w
            # Incoming: f3 (+y), f7 (+x+y), f8 (-x+y), f15 (+y+z), f17 (+y-z)
            f[3 * stride + idx] = (
                equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, 1, 0)
                + (f[4 * stride + idx] - equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, -1, 0))
            )
            f[7 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0)
                + (f[10 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0))
            )
            f[8 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0)
                + (f[9 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0))
            )
            f[15 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1)
                + (f[18 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1))
            )
            f[17 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1)
                + (f[16 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + 1 * nz + k
            courant = bc_convective_speed[2]
            for d in range(19):
                previous = boundary_history[d * stride + idx]
                interior = f[d * stride + src_idx]
                f[d * stride + idx] = previous + courant * (interior - previous)

    # ===================================================================
    # Face 3: y-max  (j == ny - 1)
    # ===================================================================
    if on_ymax:
        bc = bc_types[3]
        if bc == 1 or bc == 4:
            vx = bc_vel_x[3]
            vy = bc_vel_y[3]
            vz = bc_vel_z[3]
            denom = 1.0 + vy
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx]
                + f[1 * stride + idx] + f[2 * stride + idx]
                + f[5 * stride + idx] + f[6 * stride + idx]
                + f[11 * stride + idx] + f[12 * stride + idx]
                + f[13 * stride + idx] + f[14 * stride + idx]
                + 2.0 * (
                    f[3 * stride + idx] + f[7 * stride + idx]
                    + f[8 * stride + idx] + f[15 * stride + idx]
                    + f[17 * stride + idx]
                )
            )
            rho_w = known / denom
            if bc == 4:
                rho_w = bc_density[3]
                vy = known / rho_w - 1.0
            # Incoming: f4 (-y), f9 (+x-y), f10 (-x-y), f16 (-y+z), f18 (-y-z)
            f[4 * stride + idx] = (
                equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, -1, 0)
                + (f[3 * stride + idx] - equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, 1, 0))
            )
            f[9 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, -1, 0)
                + (f[8 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 1, 0))
            )
            f[10 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, -1, 0)
                + (f[7 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 1, 0))
            )
            f[16 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1)
                + (f[17 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1))
            )
            f[18 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1)
                + (f[15 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + (ny - 2) * nz + k
            courant = bc_convective_speed[3]
            for d in range(19):
                previous = boundary_history[d * stride + idx]
                interior = f[d * stride + src_idx]
                f[d * stride + idx] = previous + courant * (interior - previous)

    # ===================================================================
    # Face 4: z-min  (k == 0)
    # ===================================================================
    if on_zmin:
        bc = bc_types[4]
        if bc == 1 or bc == 4:
            vx = bc_vel_x[4]
            vy = bc_vel_y[4]
            vz = bc_vel_z[4]
            denom = 1.0 - vz
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx]
                + f[1 * stride + idx] + f[2 * stride + idx]
                + f[3 * stride + idx] + f[4 * stride + idx]
                + f[7 * stride + idx] + f[8 * stride + idx]
                + f[9 * stride + idx] + f[10 * stride + idx]
                + 2.0 * (
                    f[6 * stride + idx] + f[13 * stride + idx]
                    + f[14 * stride + idx] + f[17 * stride + idx]
                    + f[18 * stride + idx]
                )
            )
            rho_w = known / denom
            if bc == 4:
                rho_w = bc_density[4]
                vz = 1.0 - known / rho_w
            # Incoming: f5 (+z), f11 (+x+z), f12 (-x+z), f15 (+y+z), f16 (-y+z)
            f[5 * stride + idx] = (
                equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, 1)
                + (f[6 * stride + idx] - equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, -1))
            )
            f[11 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1)
                + (f[14 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1))
            )
            f[12 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1)
                + (f[13 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1))
            )
            f[15 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1)
                + (f[18 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1))
            )
            f[16 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1)
                + (f[17 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + j * nz + 1
            courant = bc_convective_speed[4]
            for d in range(19):
                previous = boundary_history[d * stride + idx]
                interior = f[d * stride + src_idx]
                f[d * stride + idx] = previous + courant * (interior - previous)

    # ===================================================================
    # Face 5: z-max  (k == nz - 1)
    # ===================================================================
    if on_zmax:
        bc = bc_types[5]
        if bc == 1 or bc == 4:
            vx = bc_vel_x[5]
            vy = bc_vel_y[5]
            vz = bc_vel_z[5]
            denom = 1.0 + vz
            if denom <= 0.0:
                return
            known = (
                f[0 * stride + idx]
                + f[1 * stride + idx] + f[2 * stride + idx]
                + f[3 * stride + idx] + f[4 * stride + idx]
                + f[7 * stride + idx] + f[8 * stride + idx]
                + f[9 * stride + idx] + f[10 * stride + idx]
                + 2.0 * (
                    f[5 * stride + idx] + f[11 * stride + idx]
                    + f[12 * stride + idx] + f[15 * stride + idx]
                    + f[16 * stride + idx]
                )
            )
            rho_w = known / denom
            if bc == 4:
                rho_w = bc_density[5]
                vz = known / rho_w - 1.0
            # Incoming: f6 (-z), f13 (+x-z), f14 (-x-z), f17 (+y-z), f18 (-y-z)
            f[6 * stride + idx] = (
                equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, -1)
                + (f[5 * stride + idx] - equilibrium_f(1.0 / 18.0, rho_w, vx, vy, vz, 0, 0, 1))
            )
            f[13 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, -1)
                + (f[12 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, 1))
            )
            f[14 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, -1, 0, -1)
                + (f[11 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 1, 0, 1))
            )
            f[17 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, -1)
                + (f[16 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, 1))
            )
            f[18 * stride + idx] = (
                equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, -1, -1)
                + (f[15 * stride + idx] - equilibrium_f(1.0 / 36.0, rho_w, vx, vy, vz, 0, 1, 1))
            )
        elif bc == 2:
            src_idx = i * ny * nz + j * nz + (nz - 2)
            courant = bc_convective_speed[5]
            for d in range(19):
                previous = boundary_history[d * stride + idx]
                interior = f[d * stride + src_idx]
                f[d * stride + idx] = previous + courant * (interior - previous)
