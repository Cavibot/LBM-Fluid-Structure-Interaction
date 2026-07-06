# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
import warp as wp

@wp.struct
class ContactHit:
    obj_a: int
    obj_b: int
    prim_type_a: int
    prim_type_b: int
    prim_id_a: int
    prim_id_b: int

    point_a: wp.vec3
    point_b: wp.vec3
    normal_a_to_b: wp.vec3
    distance: float          # signed distance（穿透为负）

    toi: float               # [0,1]，离散常用 1.0, 
    flags: int               # HitFlags（例如 IS_CCD/HAS_TOI）
    aux4: wp.vec4            # VF: (u,v,w,0); EE: (s,t,0,0); V/SDF: 约定
    pair_key: wp.uint64      # matching / 去重 / warmstart


@wp.struct
class ContactHitWriterData:
    hit_max: int
    hit_count: wp.array(dtype=int)

    obj_a: wp.array(dtype=int); obj_b: wp.array(dtype=int)
    prim_type_a: wp.array(dtype=int); prim_type_b: wp.array(dtype=int)
    prim_id_a: wp.array(dtype=int); prim_id_b: wp.array(dtype=int)

    point_a: wp.array(dtype=wp.vec3)
    point_b: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    distance: wp.array(dtype=float)

    toi: wp.array(dtype=float)
    flags: wp.array(dtype=int)
    aux4: wp.array(dtype=wp.vec4)
    pair_key: wp.array(dtype=wp.uint64)

@wp.func
def write_hit(hit: ContactHit, out: ContactHitWriterData, output_index: int, margin: float):
    # output_index < 0 -> atomic allocate
    # output_index >=0 -> deterministic slot
    pass
