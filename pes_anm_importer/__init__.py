# SPDX-License-Identifier: GPL-3.0-or-later
bl_info = {
    "name": "PES ANM Importer",
    "author": "marqisspes6",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "File > Import > PES ANM (.anm)",
    "description": "Import PES 2013 ANM animation onto a PES KTMDL armature",
    "warning": "ANM format is reverse engineered; unknown fields are preserved as metadata",
    "doc_url": "",
    "category": "Import-Export",
}

import bisect
import json
import math
import os

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector

from . import anm

AXIS_PES_TO_BLENDER = Matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


def _safe_name(value, fallback):
    if not value:
        return fallback
    return str(value).replace("/", "_").replace("\\", "_")


def _set_prop(owner, key, value):
    try:
        if isinstance(value, bool):
            owner[key] = bool(value)
        elif isinstance(value, int):
            owner[key] = int(value)
        elif isinstance(value, float):
            owner[key] = float(value)
        elif isinstance(value, str):
            owner[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float, bool)) for v in value
        ):
            owner[key] = list(value)
        elif value is None:
            owner[key] = "null"
        else:
            owner[key] = json.dumps(value, separators=(",", ":"), sort_keys=True)
    except Exception:
        try:
            owner[key] = str(value)
        except Exception:
            pass


def _bone_node_index(bone):
    try:
        return int(bone.get("ktmdl_node_index"))
    except Exception:
        return None


def _has_ktmdl_bones(obj):
    if obj is None or obj.type != "ARMATURE":
        return False
    for bone in obj.data.bones:
        if _bone_node_index(bone) is not None:
            return True
    return False


def _find_target_armature(context):
    active = context.view_layer.objects.active
    if _has_ktmdl_bones(active):
        return active

    candidates = [obj for obj in context.selected_objects if _has_ktmdl_bones(obj)]
    if len(candidates) == 1:
        return candidates[0]

    scene_candidates = [obj for obj in context.scene.objects if _has_ktmdl_bones(obj)]
    if len(scene_candidates) == 1:
        return scene_candidates[0]

    if len(candidates) > 1:
        raise anm.ANMError("Select exactly one KTMDL armature before importing ANM")
    if len(scene_candidates) > 1:
        raise anm.ANMError("Multiple KTMDL armatures exist; select the target armature")
    raise anm.ANMError(
        "No KTMDL armature found. Import a KTMDL with Create Armature enabled first"
    )


def _bone_maps(arm_obj):
    bones = {}
    pose_bones = {}
    for bone in arm_obj.data.bones:
        node_index = _bone_node_index(bone)
        if node_index is None:
            continue
        bones[node_index] = bone
        pbone = arm_obj.pose.bones.get(bone.name)
        if pbone is not None:
            pose_bones[node_index] = pbone
    return bones, pose_bones


def _validate_hierarchy(model, bones, strict=True):
    problems = []
    anm_nodes = model["hierarchy"]["nodes"]
    for entry in anm_nodes:
        idx = int(entry["nodeIndex"])
        parent_idx = int(entry["parentIndex"])
        bone = bones.get(idx)
        if bone is None:
            problems.append("ANM node %d is missing from KTMDL armature" % idx)
            continue

        model_parent = None
        try:
            model_parent = int(bone.get("ktmdl_parent_index"))
        except Exception:
            if bone.parent is not None:
                model_parent = _bone_node_index(bone.parent)
            else:
                model_parent = -1
        if model_parent is not None and model_parent != parent_idx:
            problems.append(
                "node %d parent mismatch: ANM=%d KTMDL=%d"
                % (idx, parent_idx, model_parent)
            )

    if strict and problems:
        raise anm.ANMError("ANM/KTMDL hierarchy mismatch: " + "; ".join(problems[:8]))
    return problems


def _matrix_from_flat_file(value):
    try:
        flat = [float(v) for v in value]
    except Exception:
        return None
    if len(flat) != 16:
        return None
    rows = [flat[i : i + 4] for i in range(0, 16, 4)]
    # Same convention as the KTMDL importer: file matrices are row-vector form.
    return Matrix(rows).transposed()


def _quat_distance(a, b):
    qa = a.normalized()
    qb = b.normalized()
    dot = abs(qa.dot(qb))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _candidate_basis(mode):
    return (
        AXIS_PES_TO_BLENDER.copy() if mode == "PES_TO_BLENDER" else Matrix.Identity(3)
    )


def _infer_coordinate_mode_and_scale(arm_obj, bones):
    # If a future/updated KTMDL importer stores these explicitly, use them.
    explicit_mode = arm_obj.get("ktmdl_coordinate_mode")
    explicit_scale = arm_obj.get("ktmdl_import_scale")
    if explicit_mode in ("PES_TO_BLENDER", "RAW"):
        try:
            return (
                explicit_mode,
                float(explicit_scale if explicit_scale is not None else 1.0),
                "armature metadata",
            )
        except Exception:
            return explicit_mode, 1.0, "armature metadata"

    samples = []
    for idx, bone in bones.items():
        raw = _matrix_from_flat_file(bone.get("ktmdl_matrixA"))
        if raw is not None:
            samples.append((idx, bone, raw))
    if not samples:
        return "PES_TO_BLENDER", 1.0, "default (no KTMDL matrix metadata)"

    scores = {}
    for mode in ("PES_TO_BLENDER", "RAW"):
        basis = _candidate_basis(mode)
        c4 = basis.to_4x4()
        ci4 = c4.inverted()
        errors = []
        for _idx, bone, raw in samples[:64]:
            candidate = c4 @ raw @ ci4
            try:
                errors.append(
                    _quat_distance(
                        candidate.to_quaternion(), bone.matrix_local.to_quaternion()
                    )
                )
            except Exception:
                pass
        scores[mode] = sum(errors) / float(len(errors)) if errors else 1.0e9

    mode = min(scores, key=scores.get)
    basis = _candidate_basis(mode)
    c4 = basis.to_4x4()
    ci4 = c4.inverted()
    ratios = []
    for _idx, bone, raw in samples:
        candidate = c4 @ raw @ ci4
        raw_len = candidate.translation.length
        blender_len = bone.matrix_local.translation.length
        if raw_len > 1.0e-8 and blender_len > 1.0e-8:
            ratios.append(blender_len / raw_len)
    if ratios:
        ratios.sort()
        mid = len(ratios) // 2
        scale = (
            ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
        )
    else:
        scale = 1.0
    return mode, float(scale), "inferred from KTMDL matrices"


def _rest_local_matrix(bone):
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted_safe() @ bone.matrix_local


def _track_map(model):
    rotation = {}
    translation = {}
    for track in model["animation"]["tracks"]:
        node = int(track["nodeIndex"])
        ctype = int(track["curveType"])
        if ctype == anm.ANM_CURVE_ROTATION:
            rotation[node] = track
        elif ctype in (
            anm.ANM_CURVE_TRANSLATION_FLOAT,
            anm.ANM_CURVE_TRANSLATION_COMPRESSED,
        ):
            translation[node] = track
    return rotation, translation


def _quat_xyzw(value):
    return Quaternion(
        (float(value[3]), float(value[0]), float(value[1]), float(value[2]))
    )


def _lerp_vec(a, b, t):
    return Vector(a).lerp(Vector(b), t)


def _track_value(track, sample_index, is_rotation):
    values = track.get("values", [])
    if not values:
        return None

    key_frames = track.get("keyFrames", [])
    if not key_frames:
        # Dense/per-sample. Clamp defensively for malformed or partial tracks.
        index = int(max(0, min(int(sample_index), len(values) - 1)))
        return _quat_xyzw(values[index]) if is_rotation else Vector(values[index])

    # Sparse/keyed track.
    if sample_index <= key_frames[0]:
        return _quat_xyzw(values[0]) if is_rotation else Vector(values[0])
    if sample_index >= key_frames[-1]:
        return _quat_xyzw(values[-1]) if is_rotation else Vector(values[-1])

    right = bisect.bisect_right(key_frames, sample_index)
    left = right - 1
    f0 = float(key_frames[left])
    f1 = float(key_frames[right])
    t = 0.0 if f1 == f0 else (float(sample_index) - f0) / (f1 - f0)
    if is_rotation:
        q0 = _quat_xyzw(values[left])
        q1 = _quat_xyzw(values[right])
        if q0.dot(q1) < 0.0:
            q1.negate()
        return q0.slerp(q1, t)
    return _lerp_vec(values[left], values[right], t)


def _convert_rotation(file_quaternion, basis):
    r = file_quaternion.to_matrix()
    converted = basis @ r @ basis.inverted()
    return converted.to_quaternion().normalized()


def _convert_translation(file_translation, basis, scale):
    return (basis @ Vector(file_translation)) * scale


def _local_target_matrix(
    node_index, sample_index, bone, rotation_tracks, translation_tracks, basis, scale
):
    rest_local = _rest_local_matrix(bone)
    rest_translation = rest_local.translation.copy()
    rest_rotation = rest_local.to_quaternion().normalized()
    rest_scale = rest_local.to_scale()

    rot_track = rotation_tracks.get(node_index)
    if rot_track is not None:
        file_q = _track_value(rot_track, sample_index, True)
        rotation = (
            _convert_rotation(file_q, basis) if file_q is not None else rest_rotation
        )
    else:
        rotation = rest_rotation

    trans_track = translation_tracks.get(node_index)
    if trans_track is not None:
        file_t = _track_value(trans_track, sample_index, False)
        translation = (
            _convert_translation(file_t, basis, scale)
            if file_t is not None
            else rest_translation
        )
    else:
        translation = rest_translation

    scale_matrix = Matrix.Diagonal((rest_scale.x, rest_scale.y, rest_scale.z, 1.0))
    return (
        Matrix.Translation(translation) @ rotation.to_matrix().to_4x4() @ scale_matrix
    )


def _hierarchy_maps(model):
    parent = {}
    order = []
    for entry in model["hierarchy"]["nodes"]:
        idx = int(entry["nodeIndex"])
        parent[idx] = int(entry["parentIndex"])
        order.append(idx)
    return parent, order


def _global_matrix(node_index, local_mats, parent_map, cache, visiting):
    if node_index in cache:
        return cache[node_index]
    if node_index in visiting:
        raise anm.ANMError("Cycle detected in ANM hierarchy at node %d" % node_index)
    visiting.add(node_index)
    local = local_mats[node_index]
    parent = parent_map.get(node_index, -1)
    if parent < 0 or parent not in local_mats:
        result = local
    else:
        result = _global_matrix(parent, local_mats, parent_map, cache, visiting) @ local
    visiting.remove(node_index)
    cache[node_index] = result
    return result


def _matrix_basis_from_pose(bone, pose_matrix, parent_pose):
    # Bone.convert_local_to_pose has existed since Blender 2.80 and remains the
    # correct way to account for rest transforms/inherit settings.
    if bone.parent is None:
        return bone.convert_local_to_pose(pose_matrix, bone.matrix_local, invert=True)
    return bone.convert_local_to_pose(
        pose_matrix,
        bone.matrix_local,
        parent_matrix=parent_pose,
        parent_matrix_local=bone.parent.matrix_local,
        invert=True,
    )


def _ensure_action_fcurve(action, arm_obj, data_path, index, group_name):
    # Blender 5.0 removed Action.fcurves/groups in favor of layered Actions.
    # fcurve_ensure_for_datablock is the cross-over API for current Blender.
    ensure = getattr(action, "fcurve_ensure_for_datablock", None)
    if ensure is not None:
        try:
            return ensure(arm_obj, data_path, index=index, group_name=group_name)
        except TypeError:
            try:
                return ensure(arm_obj, data_path, index=index)
            except TypeError:
                return ensure(arm_obj, data_path, index)

    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        raise anm.ANMError(
            "This Blender build exposes neither Action.fcurves nor fcurve_ensure_for_datablock"
        )
    return fcurves.new(data_path=data_path, index=index, action_group=group_name)


def _write_curve(action, arm_obj, data_path, array_index, group_name, frames, values):
    curve = _ensure_action_fcurve(action, arm_obj, data_path, array_index, group_name)
    points = curve.keyframe_points
    points.add(len(frames))
    for i in range(len(frames)):
        point = points[i]
        point.co = (float(frames[i]), float(values[i]))
        try:
            point.interpolation = "LINEAR"
        except Exception:
            pass
    try:
        curve.extrapolation = "CONSTANT"
    except Exception:
        pass
    curve.update()
    return curve


def _create_metadata_text(model, stem):
    name = stem + ".anm_metadata.json"
    text = bpy.data.texts.get(name)
    if text is None:
        text = bpy.data.texts.new(name)
    else:
        text.clear()
    text.write(json.dumps(model, indent=2, sort_keys=True))
    return text


def import_anm(
    context,
    filepath,
    coordinate_mode="AUTO",
    scale=0.0,
    fallback_fps=30.0,
    set_scene_fps=True,
    set_frame_range=True,
    frame_start=1,
    strict_hierarchy=True,
    store_metadata=True,
):
    model = anm.parse_file(filepath)
    arm_obj = _find_target_armature(context)
    bones, pose_bones = _bone_maps(arm_obj)
    if not bones:
        raise anm.ANMError("Target armature contains no ktmdl_node_index bone metadata")

    hierarchy_warnings = _validate_hierarchy(model, bones, strict=strict_hierarchy)

    inferred_mode, inferred_scale, inference_source = _infer_coordinate_mode_and_scale(
        arm_obj, bones
    )
    actual_mode = inferred_mode if coordinate_mode == "AUTO" else coordinate_mode
    actual_scale = inferred_scale if scale <= 0.0 else float(scale)
    basis = _candidate_basis(actual_mode)

    rotation_tracks, translation_tracks = _track_map(model)
    parent_map, node_order = _hierarchy_maps(model)
    last_sample = int(model["header"]["lastSampleIndex"])
    sample_count = last_sample + 1

    source_fps = float(fallback_fps)
    sequence = model.get("sequence")
    if sequence is not None and float(sequence.get("fps", 0)) > 0.0:
        source_fps = float(sequence["fps"])
    if source_fps <= 0.0:
        source_fps = 30.0

    if set_scene_fps:
        # Preserve integer FPS exactly when possible.
        rounded = int(round(source_fps))
        context.scene.render.fps = max(1, rounded)
        try:
            context.scene.render.fps_base = float(rounded) / source_fps
        except Exception:
            pass

    scene_fps = float(context.scene.render.fps)
    try:
        scene_fps = scene_fps / float(context.scene.render.fps_base)
    except Exception:
        pass
    frame_step = 1.0 if set_scene_fps else (scene_fps / source_fps)
    frames = [float(frame_start) + i * frame_step for i in range(sample_count)]

    stem = _safe_name(os.path.splitext(os.path.basename(filepath))[0], "ANM")
    action = bpy.data.actions.new(stem)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    _set_prop(action, "pes_anm_source", os.path.basename(filepath))
    _set_prop(action, "pes_anm_core_offset", model["coreOffset"])
    _set_prop(action, "pes_anm_last_sample_index", last_sample)
    _set_prop(action, "pes_anm_source_fps", source_fps)
    _set_prop(action, "pes_anm_coordinate_mode", actual_mode)
    _set_prop(action, "pes_anm_import_scale", actual_scale)
    _set_prop(action, "pes_anm_transform_detection", inference_source)
    _set_prop(action, "pes_anm_hierarchy_warnings", hierarchy_warnings)

    # Collect channels per bone, then emit F-curves in bulk. This is much faster
    # than calling keyframe_insert for every bone at every sample.
    channel_data = {}
    previous_quat = {}
    for node_index in node_order:
        if node_index not in bones or node_index not in pose_bones:
            continue
        channel_data[node_index] = {
            "loc": [[], [], []],
            "rot": [[], [], [], []],  # Blender quaternion order w,x,y,z
            "scale": [[], [], []],
        }
        pose_bones[node_index].rotation_mode = "QUATERNION"

    for sample_index in range(sample_count):
        local_mats = {}
        for node_index in node_order:
            bone = bones.get(node_index)
            if bone is None:
                continue
            local_mats[node_index] = _local_target_matrix(
                node_index,
                sample_index,
                bone,
                rotation_tracks,
                translation_tracks,
                basis,
                actual_scale,
            )

        global_cache = {}
        for node_index in node_order:
            if node_index in local_mats:
                _global_matrix(node_index, local_mats, parent_map, global_cache, set())

        for node_index in node_order:
            bone = bones.get(node_index)
            pbone = pose_bones.get(node_index)
            channels = channel_data.get(node_index)
            if bone is None or pbone is None or channels is None:
                continue
            pose_matrix = global_cache[node_index]
            parent_index = parent_map.get(node_index, -1)
            parent_pose = global_cache.get(parent_index)
            basis_matrix = _matrix_basis_from_pose(bone, pose_matrix, parent_pose)
            loc, rot, sca = basis_matrix.decompose()
            rot.normalize()
            prev = previous_quat.get(node_index)
            if prev is not None and prev.dot(rot) < 0.0:
                rot.negate()
            previous_quat[node_index] = rot.copy()

            channels["loc"][0].append(loc.x)
            channels["loc"][1].append(loc.y)
            channels["loc"][2].append(loc.z)
            channels["rot"][0].append(rot.w)
            channels["rot"][1].append(rot.x)
            channels["rot"][2].append(rot.y)
            channels["rot"][3].append(rot.z)
            channels["scale"][0].append(sca.x)
            channels["scale"][1].append(sca.y)
            channels["scale"][2].append(sca.z)

    curve_count = 0
    for node_index in node_order:
        pbone = pose_bones.get(node_index)
        channels = channel_data.get(node_index)
        if pbone is None or channels is None:
            continue
        group = pbone.name
        loc_path = pbone.path_from_id("location")
        rot_path = pbone.path_from_id("rotation_quaternion")
        scale_path = pbone.path_from_id("scale")
        for axis in range(3):
            _write_curve(
                action, arm_obj, loc_path, axis, group, frames, channels["loc"][axis]
            )
            curve_count += 1
        for axis in range(4):
            _write_curve(
                action, arm_obj, rot_path, axis, group, frames, channels["rot"][axis]
            )
            curve_count += 1
        for axis in range(3):
            _write_curve(
                action,
                arm_obj,
                scale_path,
                axis,
                group,
                frames,
                channels["scale"][axis],
            )
            curve_count += 1

    if set_frame_range and frames:
        context.scene.frame_start = int(math.floor(frames[0]))
        context.scene.frame_end = int(math.ceil(frames[-1]))
        try:
            action.frame_range = (frames[0], frames[-1])
        except Exception:
            try:
                action.use_frame_range = True
                action.frame_start = frames[0]
                action.frame_end = frames[-1]
            except Exception:
                pass

    metadata_text = _create_metadata_text(model, stem) if store_metadata else None
    if metadata_text is not None:
        _set_prop(action, "pes_anm_metadata_text", metadata_text.name)
        _set_prop(arm_obj, "pes_anm_last_metadata_text", metadata_text.name)

    _set_prop(arm_obj, "pes_anm_last_action", action.name)
    _set_prop(arm_obj, "pes_anm_last_source", os.path.basename(filepath))

    # Leave the armature selected/active and show first frame.
    for obj in context.selected_objects:
        try:
            obj.select_set(False)
        except Exception:
            pass
    arm_obj.select_set(True)
    context.view_layer.objects.active = arm_obj
    context.scene.frame_set(int(round(frames[0])) if frames else frame_start)

    return {
        "model": model,
        "armature": arm_obj,
        "action": action,
        "curveCount": curve_count,
        "sampleCount": sample_count,
        "sourceFps": source_fps,
        "coordinateMode": actual_mode,
        "scale": actual_scale,
        "hierarchyWarnings": hierarchy_warnings,
    }


class IMPORT_OT_pes_anm(bpy.types.Operator, ImportHelper):
    bl_idname = "import_anim.pes_anm"
    bl_label = "Import PES ANM"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".anm"
    filter_glob: StringProperty(default="*.anm", options={"HIDDEN"})

    coordinate_mode: EnumProperty(
        name="Coordinates",
        description="Match the coordinate conversion used when importing the KTMDL",
        items=(
            (
                "AUTO",
                "Auto from KTMDL",
                "Infer Raw/PES-to-Blender from KTMDL bone matrices",
            ),
            ("PES_TO_BLENDER", "PES to Blender", "Convert (X,Y,Z) to (X,-Z,Y)"),
            ("RAW", "Raw", "Keep ANM coordinate axes unchanged"),
        ),
        default="AUTO",
    )

    scale: FloatProperty(
        name="Scale Override",
        description="0 = infer the KTMDL import scale; otherwise use this uniform scale for ANM translations",
        default=0.0,
        min=0.0,
        soft_max=100.0,
    )

    fallback_fps: FloatProperty(
        name="Fallback FPS",
        description="Used when the ANM has no validated sequence header carrying FPS",
        default=30.0,
        min=1.0,
        max=240.0,
    )

    set_scene_fps: BoolProperty(
        name="Set Scene FPS",
        description="Set Blender scene timing to the ANM sequence FPS (or fallback FPS) so one ANM sample equals one Blender frame",
        default=True,
    )

    set_frame_range: BoolProperty(
        name="Set Frame Range",
        description="Set the scene playback range to the imported animation",
        default=True,
    )

    frame_start: IntProperty(
        name="Start Frame",
        description="Blender frame corresponding to ANM sample 0",
        default=1,
    )

    strict_hierarchy: BoolProperty(
        name="Require Matching KTMDL Hierarchy",
        description="Cancel import if ANM node/parent indices do not match the selected KTMDL armature",
        default=True,
    )

    store_metadata: BoolProperty(
        name="Store Full ANM Metadata",
        description="Store headers, hierarchy, native tracks, packed rotations, sequence events and unknown bytes in a Blender Text datablock",
        default=True,
    )

    def execute(self, context):
        try:
            result = import_anm(
                context,
                self.filepath,
                coordinate_mode=self.coordinate_mode,
                scale=self.scale,
                fallback_fps=self.fallback_fps,
                set_scene_fps=self.set_scene_fps,
                set_frame_range=self.set_frame_range,
                frame_start=self.frame_start,
                strict_hierarchy=self.strict_hierarchy,
                store_metadata=self.store_metadata,
            )
        except anm.ANMError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, "ANM import failed: %s" % exc)
            raise

        self.report(
            {"INFO"},
            "Imported %d ANM samples at %.3f FPS onto %s (%d F-curves)"
            % (
                result["sampleCount"],
                result["sourceFps"],
                result["armature"].name,
                result["curveCount"],
            ),
        )
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_pes_anm.bl_idname, text="PES ANM (.anm)")


classes = (IMPORT_OT_pes_anm,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    try:
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    except Exception:
        pass
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
