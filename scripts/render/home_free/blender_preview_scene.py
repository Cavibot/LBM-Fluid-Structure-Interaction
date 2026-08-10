#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Build and render one HOME-Free preview frame inside Blender."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--tank-extent", type=float, nargs=3, default=(24.0, 12.0, 10.0))
    parser.add_argument("--camera-view", choices=("side", "diagonal"), default="side")
    parser.add_argument(
        "--render-style",
        choices=("diagnostic", "cinematic"),
        default="diagnostic",
    )
    parser.add_argument("--common-comparison", action="store_true")
    return parser.parse_args(argv)


def _material(name: str, color: tuple[float, float, float, float], roughness: float):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    return material


def _glass_material(name: str):
    material = _material(name, (0.62, 0.78, 0.88, 1.0), 0.04)
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Transmission Weight"].default_value = 0.92
    principled.inputs["IOR"].default_value = 1.45
    return material


def _cinematic_water_material(name: str):
    material = _material(name, (0.16, 0.62, 0.82, 1.0), 0.035)
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Transmission Weight"].default_value = 0.72
    principled.inputs["IOR"].default_value = 1.333
    principled.inputs["Metallic"].default_value = 0.0

    absorption = material.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = (0.18, 0.62, 0.78, 1.0)
    absorption.inputs["Density"].default_value = 0.008
    material.node_tree.links.new(
        absorption.outputs["Volume"],
        material.node_tree.nodes["Material Output"].inputs["Volume"],
    )
    return material


def _cinematic_glass_material(name: str):
    material = _material(name, (0.92, 0.97, 1.0, 1.0), 0.035)
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Transmission Weight"].default_value = 1.0
    principled.inputs["IOR"].default_value = 1.46
    return material


def _add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material,
    *,
    bevel: float = 0.0,
):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel > 0.0:
        modifier = obj.modifiers.new(name="Edge bevel", type="BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def _set_world(scene, *, cinematic: bool, common_comparison: bool = False) -> None:
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if common_comparison:
        background.inputs["Color"].default_value = (0.12, 0.15, 0.18, 1.0)
        background.inputs["Strength"].default_value = 0.7
    elif cinematic:
        background.inputs["Color"].default_value = (0.055, 0.075, 0.095, 1.0)
        background.inputs["Strength"].default_value = 0.28
    else:
        background.inputs["Color"].default_value = (0.008, 0.01, 0.014, 1.0)
        background.inputs["Strength"].default_value = 1.0


def _add_area_light(
    name: str,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    energy: float,
    size: float,
    color: tuple[float, float, float],
    specular: float = 1.0,
) -> None:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    light.data.color = color
    light.data.specular_factor = specular
    _point_at(light, target)


def _point_at(obj, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _world_bounds(obj) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = [min(corner[axis] for corner in corners) for axis in range(3)]
    upper = [max(corner[axis] for corner in corners) for axis in range(3)]
    return lower, upper


def _combined_world_bounds(objects) -> tuple[list[float], list[float]]:
    bounds = [_world_bounds(obj) for obj in objects]
    lower = [min(bound[0][axis] for bound in bounds) for axis in range(3)]
    upper = [max(bound[1][axis] for bound in bounds) for axis in range(3)]
    return lower, upper


def _configure_optix(scene) -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.refresh_devices()
    enabled = []
    for device in preferences.devices:
        use = device.type == "OPTIX" and "5090" in device.name
        device.use = use
        if use:
            enabled.append(device.name)
    if not enabled:
        raise RuntimeError("RTX 5090 OptiX device was not found")
    scene.cycles.device = "GPU"
    return enabled


def main() -> None:
    args = _arguments()
    cinematic = args.render_style == "cinematic"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    enabled_devices = _configure_optix(scene)
    scene.render.resolution_x = 1280 if cinematic else 960
    scene.render.resolution_y = 720 if cinematic else 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(args.output.resolve())
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    _set_world(
        scene,
        cinematic=cinematic,
        common_comparison=args.common_comparison,
    )

    # Preview OBJ vertices are already in the solver's Z-up world frame.
    # Blender's importer otherwise assumes the conventional OBJ Y-up frame
    # and rotates the liquid out of alignment with the procedurally built tank.
    bpy.ops.wm.obj_import(
        filepath=str(args.mesh.resolve()),
        forward_axis="Y",
        up_axis="Z",
    )
    water = bpy.context.selected_objects[0]
    water.name = "HOME-Free preview water"
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    if args.common_comparison:
        water_material = _material(
            "Water common comparison", (0.025, 0.32, 0.62, 1.0), 0.16
        )
        principled = water_material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Transmission Weight"].default_value = 0.12
        principled.inputs["IOR"].default_value = 1.333
        principled.inputs["Coat Weight"].default_value = 0.24
        principled.inputs["Coat Roughness"].default_value = 0.08
    elif cinematic:
        water_material = _cinematic_water_material("Water cinematic")
    else:
        water_material = _material("Water preview", (0.04, 0.36, 0.68, 1.0), 0.08)
        principled = water_material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Metallic"].default_value = 0.05
        principled.inputs["Transmission Weight"].default_value = 0.55
        principled.inputs["IOR"].default_value = 1.333
    water.data.materials.append(water_material)

    floor_material = _material(
        "Tank floor",
        (0.38, 0.42, 0.45, 1.0) if cinematic else (0.18, 0.21, 0.25, 1.0),
        0.2 if cinematic else 0.28,
    )
    backplate_material = _material(
        "Tank backdrop",
        (
            (0.18, 0.21, 0.24, 1.0)
            if args.common_comparison
            else (
                (0.36, 0.42, 0.46, 1.0)
                if cinematic
                else (0.07, 0.09, 0.11, 1.0)
            )
        ),
        0.58 if cinematic else 0.35,
    )
    glass_material = (
        _cinematic_glass_material("Tank glass cinematic")
        if cinematic
        else _glass_material("Tank glass")
    )
    frame_material = _material(
        "Tank frame",
        (0.42, 0.46, 0.49, 1.0) if cinematic else (0.72, 0.78, 0.82, 1.0),
        0.2 if cinematic else 0.16,
    )
    frame_principled = frame_material.node_tree.nodes.get("Principled BSDF")
    frame_principled.inputs["Metallic"].default_value = 0.65
    frame_principled.inputs["Emission Color"].default_value = (0.16, 0.2, 0.23, 1.0)
    frame_principled.inputs["Emission Strength"].default_value = 0.35
    tank_length, tank_depth, tank_height = args.tank_extent
    panel = 0.06
    rail = 0.055
    frame_offset = 0.16
    _add_box(
        "Floor",
        (tank_length / 2.0, tank_depth / 2.0, -0.12),
        (tank_length / 2.0 + 0.12, tank_depth / 2.0 + 0.12, 0.12),
        floor_material,
        bevel=0.025 if cinematic else 0.0,
    )
    _add_box(
        "Studio backplate" if cinematic else "Diagnostic backplate",
        (
            tank_length / 2.0,
            tank_depth + (0.65 if cinematic else panel),
            tank_height / 2.0,
        ),
        (tank_length / 2.0, panel, tank_height / 2.0),
        backplate_material,
    )
    if cinematic:
        _add_box(
            "Back glass",
            (tank_length / 2.0, tank_depth + panel, tank_height / 2.0),
            (tank_length / 2.0, panel, tank_height / 2.0),
            glass_material,
            bevel=0.015,
        )
        if not args.common_comparison:
            _add_box(
                "Front glass",
                (tank_length / 2.0, -panel, tank_height / 2.0),
                (tank_length / 2.0, panel, tank_height / 2.0),
                glass_material,
                bevel=0.015,
            )
    _add_box(
        "Left glass",
        (-panel, tank_depth / 2.0, tank_height / 2.0),
        (panel, tank_depth / 2.0, tank_height / 2.0),
        glass_material,
        bevel=0.015 if cinematic else 0.0,
    )
    _add_box(
        "Right glass",
        (tank_length + panel, tank_depth / 2.0, tank_height / 2.0),
        (panel, tank_depth / 2.0, tank_height / 2.0),
        glass_material,
        bevel=0.015 if cinematic else 0.0,
    )
    for x in (-frame_offset, tank_length + frame_offset):
        for y in (-frame_offset, tank_depth + frame_offset):
            _add_box(
                f"Vertical rail {x:g} {y:g}",
                (x, y, tank_height / 2.0),
                (rail, rail, tank_height / 2.0 + frame_offset),
                frame_material,
                bevel=0.025 if cinematic else 0.0,
            )
    for y in (-frame_offset, tank_depth + frame_offset):
        for z in (-frame_offset, tank_height + frame_offset):
            _add_box(
                f"Length rail {y:g} {z:g}",
                (tank_length / 2.0, y, z),
                (tank_length / 2.0 + frame_offset, rail, rail),
                frame_material,
                bevel=0.025 if cinematic else 0.0,
            )
    for x in (-frame_offset, tank_length + frame_offset):
        for z in (-frame_offset, tank_height + frame_offset):
            _add_box(
                f"Depth rail {x:g} {z:g}",
                (x, tank_depth / 2.0, z),
                (rail, tank_depth / 2.0 + frame_offset, rail),
                frame_material,
                bevel=0.025 if cinematic else 0.0,
            )
    # The front outline is deliberately independent of the glass shader. It
    # sits camera-side of both glass and liquid, so wet walls remain legible.
    front_y = -0.5
    front_rail = 0.14
    _add_box(
        "Diagnostic front left boundary",
        (-front_rail, front_y, tank_height / 2.0),
        (front_rail, front_rail, tank_height / 2.0 + front_rail),
        frame_material,
        bevel=0.025 if cinematic else 0.0,
    )
    _add_box(
        "Diagnostic front right boundary",
        (tank_length + front_rail, front_y, tank_height / 2.0),
        (front_rail, front_rail, tank_height / 2.0 + front_rail),
        frame_material,
        bevel=0.025 if cinematic else 0.0,
    )
    for z in (-front_rail, tank_height + front_rail):
        _add_box(
            f"Diagnostic front horizontal boundary {z:g}",
            (tank_length / 2.0, front_y, z),
            (tank_length / 2.0 + front_rail, front_rail, front_rail),
            frame_material,
            bevel=0.025 if cinematic else 0.0,
        )

    if cinematic:
        _add_area_light(
            "Large softbox",
            (0.18 * tank_length, -1.8 * tank_depth, 1.35 * tank_height),
            (0.46 * tank_length, 0.45 * tank_depth, 0.35 * tank_height),
            energy=2100.0,
            size=max(10.0, 0.7 * tank_length),
            color=(0.88, 0.95, 1.0),
        )
        _add_area_light(
            "Warm rim",
            (0.92 * tank_length, 1.5 * tank_depth, 0.95 * tank_height),
            (0.55 * tank_length, 0.5 * tank_depth, 0.42 * tank_height),
            energy=1500.0,
            size=max(7.0, 0.45 * tank_length),
            color=(1.0, 0.82, 0.66),
        )
        _add_area_light(
            "Top reflection strip",
            (0.52 * tank_length, 0.3 * tank_depth, 1.7 * tank_height),
            (0.52 * tank_length, 0.4 * tank_depth, 0.3 * tank_height),
            energy=1100.0,
            size=max(8.0, 0.55 * tank_length),
            color=(0.78, 0.9, 1.0),
        )
        _add_area_light(
            "Camera fill",
            (0.55 * tank_length, -2.8 * tank_depth, 0.55 * tank_height),
            (0.5 * tank_length, 0.45 * tank_depth, 0.32 * tank_height),
            energy=1250.0,
            size=max(9.0, 0.5 * tank_length),
            color=(0.72, 0.88, 1.0),
            specular=0.0,
        )
    else:
        diagnostic_key_location = (
            (0.22 * tank_length, -0.45 * tank_length, 1.35 * tank_height)
            if args.common_comparison
            else (6.0, -3.0, 18.0)
        )
        diagnostic_key_target = (
            (0.45 * tank_length, 0.5 * tank_depth, 0.35 * tank_height)
            if args.common_comparison
            else (10.0, 6.0, 3.0)
        )
        _add_area_light(
            "Diagnostic key",
            diagnostic_key_location,
            diagnostic_key_target,
            energy=1500.0,
            size=max(9.0, 0.45 * tank_length),
            color=(1.0, 1.0, 1.0),
            specular=0.0 if args.common_comparison else 1.0,
        )
        diagnostic_fill_location = (
            (0.85 * tank_length, 1.5 * tank_depth, 1.1 * tank_height)
            if args.common_comparison
            else (22.0, 9.0, 12.0)
        )
        diagnostic_fill_target = (
            (0.55 * tank_length, 0.5 * tank_depth, 0.3 * tank_height)
            if args.common_comparison
            else (13.0, 6.0, 3.0)
        )
        _add_area_light(
            "Diagnostic fill",
            diagnostic_fill_location,
            diagnostic_fill_target,
            energy=900.0,
            size=max(7.0, 0.35 * tank_length),
            color=(1.0, 1.0, 1.0),
            specular=0.0 if args.common_comparison else 1.0,
        )

    # Keep the side projection as the authoritative diagnostic view and offer
    # a fixed three-quarter view for reading tank depth and surface shape.
    if args.common_comparison and args.camera_view == "diagonal":
        camera_location = (
            1.25 * tank_length,
            -0.55 * tank_length,
            1.05 * tank_height,
        )
    elif args.camera_view == "side":
        camera_location = (tank_length / 2.0, -50.0, tank_height / 2.0)
    else:
        camera_location = (
            1.3 * tank_length,
            -8.0 * max(tank_depth, 1.0),
            1.2 * tank_height,
        )
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    if args.common_comparison:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = max(
            1.8 * tank_height,
            1.1 * tank_length,
            2.5 * tank_depth,
        )
    elif cinematic:
        camera.data.type = "PERSP"
        camera.data.lens = 52.0
        camera.data.sensor_width = 36.0
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = (
            Vector(camera_location)
            - Vector((0.5 * tank_length, 0.5 * tank_depth, 0.42 * tank_height))
        ).length
        camera.data.dof.aperture_fstop = 8.0
    else:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = max(27.0, 1.55 * tank_height)
    if args.camera_view == "side" and not cinematic:
        camera.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    else:
        _point_at(
            camera,
            (0.5 * tank_length, 0.5 * tank_depth, 0.42 * tank_height),
        )
    scene.camera = camera

    camera_forward = camera.rotation_euler.to_matrix() @ Vector((0.0, 0.0, -1.0))
    camera_up = camera.rotation_euler.to_matrix() @ Vector((0.0, 1.0, 0.0))
    print(f"HOME_FREE_WATER_BOUNDS={_world_bounds(water)}")
    print(f"HOME_FREE_CAMERA_FORWARD={tuple(camera_forward)}")
    print(f"HOME_FREE_CAMERA_UP={tuple(camera_up)}")
    print(
        "HOME_FREE_TANK_INTERIOR_BOUNDS="
        f"{([0.0, 0.0, 0.0], [tank_length, tank_depth, tank_height])}"
    )
    frame_objects = [
        obj
        for obj in bpy.context.scene.objects
        if "rail" in obj.name.lower() or obj.name.startswith("Diagnostic front")
    ]
    print(f"HOME_FREE_TANK_FRAME_BOUNDS={_combined_world_bounds(frame_objects)}")
    print(
        "HOME_FREE_CAMERA_ORTHO_SCALE="
        f"{camera.data.ortho_scale if camera.data.type == 'ORTHO' else 0.0}"
    )
    print(f"HOME_FREE_CAMERA_VIEW={args.camera_view!r}")
    print(f"HOME_FREE_RENDER_STYLE={args.render_style!r}")
    print(f"HOME_FREE_COMMON_COMPARISON={args.common_comparison!r}")
    print(
        "HOME_FREE_TANK_FRONT_BOUNDS="
        f"{_world_bounds(bpy.data.objects['Diagnostic front left boundary'])}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"HOME_FREE_OPTIX_DEVICES={enabled_devices}")
    print(f"HOME_FREE_PREVIEW_RENDER={args.output.resolve()}")


if __name__ == "__main__":
    main()
