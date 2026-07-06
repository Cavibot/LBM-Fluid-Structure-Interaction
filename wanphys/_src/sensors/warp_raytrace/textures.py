# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry import GeoType


@wp.func
def sample_texture_2d(
    uv: wp.vec2f, width: wp.int32, height: wp.int32, texture_offsets: wp.int32, texture_data: wp.array(dtype=wp.uint32)
) -> wp.vec3f:
    ix = wp.min(width - 1, wp.int32(uv[0] * wp.float32(width)))
    iy = wp.min(height - 1, wp.int32(uv[1] * wp.float32(height)))
    linear_idx = texture_offsets + (iy * width + ix)
    packed_rgba = texture_data[linear_idx]
    r = wp.float32((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0
    g = wp.float32((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0
    b = wp.float32(packed_rgba & wp.uint32(0xFF)) / 255.0
    return wp.vec3f(r, g, b)


@wp.func
def sample_texture_plane(
    hit_point: wp.vec3f,
    shape_transforms: wp.transformf,
    material_texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(shape_transforms)
    local = wp.transform_point(inv_transform, hit_point)
    u = local[0] * material_texture_repeat[0]
    v = local[1] * material_texture_repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(wp.vec2f(u, v), texture_width, texture_height, texture_offsets, texture_data)


@wp.func
def sample_texture_mesh(
    bary_u: wp.float32,
    bary_v: wp.float32,
    uv_baseadr: wp.int32,
    v_idx: wp.vec3i,
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    material_texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
) -> wp.vec3f:
    bw = 1.0 - bary_u - bary_v
    uv0 = mesh_texcoord[uv_baseadr + v_idx.x]
    uv1 = mesh_texcoord[uv_baseadr + v_idx.y]
    uv2 = mesh_texcoord[uv_baseadr + v_idx.z]
    uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
    u = uv[0] * material_texture_repeat[0]
    v = uv[1] * material_texture_repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(
        wp.vec2f(u, v),
        texture_width,
        texture_height,
        texture_offsets,
        texture_data,
    )


@wp.func
def sample_texture(
    shape_type: wp.int32,
    shape_transform: wp.transformf,
    material_index: wp.int32,
    texture_index: wp.int32,
    material_texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
    mesh_face_offsets: wp.array(dtype=wp.int32),
    mesh_face_vertices: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    mesh_texcoord_offsets: wp.array(dtype=wp.int32),
    hit_point: wp.vec3f,
    u: wp.float32,
    v: wp.float32,
    f: wp.int32,
    mesh_id: wp.int32,
) -> wp.vec3f:
    tex_color = wp.vec3f(1.0, 1.0, 1.0)

    if material_index == -1 or texture_index == -1:
        return tex_color
    
    if shape_type == GeoType.PLANE:
        tex_color = sample_texture_plane(
            hit_point,
            shape_transform,
            material_texture_repeat,
            texture_offsets,
            texture_data,
            texture_height,
            texture_width,
        )

    if shape_type == GeoType.MESH:
        if f < 0 or mesh_id < 0 or not mesh_texcoord_offsets.shape[0]:
            return tex_color
        
        uv_base = mesh_texcoord_offsets[mesh_id]

        if mesh_texcoord.shape[0] <= uv_base:
            return tex_color
        
        tex_color = sample_texture_mesh(
            u,
            v,
            uv_base,
            wp.vec3i(f * 3 + 2, f * 3 + 0, f * 3 + 1),
            mesh_texcoord,
            material_texture_repeat,
            texture_offsets,
            texture_data,
            texture_height,
            texture_width,
        )

    return tex_color
