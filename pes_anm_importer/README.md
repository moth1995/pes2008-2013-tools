# PES ANM Importer for Blender 2.80–5.x

Imports PES/Konami `.anm` animation files onto an armature created by the companion **PES KTMDL Importer**.

## Workflow

1. Install both add-ons.
2. Import the matching standalone `.ktmdl` with **Create Armature** enabled.
3. Select the generated KTMDL armature.
4. Use **File → Import → PES ANM (.anm)**.
5. The importer creates a Blender Action and assigns it to the armature.

The link between both importers is the `ktmdl_node_index` and hierarchy metadata stored on KTMDL bones. ANM node indices are therefore matched by their actual game indices, not by Blender bone names.

## Blender compatibility

The add-on is written for Blender **2.80 through 5.x**. Blender 5 removed the legacy `Action.fcurves` collection, so the importer feature-detects and uses `Action.fcurve_ensure_for_datablock()` on newer versions while retaining the legacy path for Blender 2.8–4.x.

## Decoded ANM data

The parser currently decodes everything established by `PES_ANM_updated.bt` and preserves unknown data in metadata:

- FF010001 core header.
- FF010003 hierarchy and parent links.
- FF010002 animation chunk.
- Special node-0 rotation stored immediately after the track-offset table.
- Referenced track-offset table and zero terminator(s).
- Curve `0x1C`: 48-bit packed quaternion rotation.
- Curve `0x1D`: raw FLOAT3 translation.
- Curve `0x1F`: base FLOAT3 + half-float delta translation.
- Dense storage mode `3`.
- Sparse/keyed storage mode `0`.
- Optional pre-core `anmSeqHeader_t`, FPS, sequence tracks and raw 16-byte event records.
- Unknown prelude, padding, per-track raw bytes and undecoded fields.

Packed quaternions use the confirmed 1/15/15/15/2 layout and are normalized after decoding.

## Transform behavior

ANM rotation/translation values are treated as **absolute local bone transforms**.

When a bone has no ANM translation curve, its KTMDL rest-local translation is retained. This is important for player-body animation files, where child bones normally have rotations but no translation tracks.

The importer reconstructs absolute animated bone matrices, then converts them to Blender `matrix_basis` using `Bone.convert_local_to_pose()`. This accounts for Blender's armature rest transforms instead of incorrectly writing the game quaternion straight into `PoseBone.rotation_quaternion`.

Every integer ANM sample is baked into the Action. This preserves every dense sample exactly at sample times. Sparse tracks are evaluated between their native keys using quaternion slerp for rotations and linear interpolation for translations; the native sparse keys and raw bytes remain in metadata. The supplied sparse tracks are constant rotations, so no unproven interpolation affects the validated samples.

## Coordinate mode and scale

`Auto from KTMDL` compares the raw KTMDL matrices stored on bones with the actual Blender rest matrices to infer whether the model was imported using:

- `PES to Blender`: `(X,Y,Z) → (X,-Z,Y)`
- `Raw`

It also estimates the uniform scale used by the KTMDL importer. Both can be overridden in the ANM import options.

## FPS

If a validated `anmSeqHeader_t` is present, its FPS is used. The supplied player-body sample contains `30 FPS`.

If no sequence header is present (such as the supplied ribbon sample), the **Fallback FPS** option is used; this defaults to 30 because the core itself does not prove the FPS.

## Full metadata

When **Store Full ANM Metadata** is enabled, a Blender Text datablock named approximately:

`animation_name.anm_metadata.json`

contains the parsed headers, hierarchy, sequence metadata/events, native track values, packed quaternion information, offsets, unknown fields and raw hexadecimal regions.

## Known limits

- Only curve types proven by the supplied samples (`0x1C`, `0x1D`, `0x1F`) are converted into Blender animation channels.
- Unknown curve types are preserved in the metadata but cannot be animated until their payload format is identified.
- No scale curve type has been observed in the supplied PES ANMs.
- The exact meaning of the 16-byte sequence event fields remains unknown; all eight uint16 values are preserved.
- Runtime interpolation rules for hypothetical non-constant sparse tracks are not proven. The importer uses slerp/linear when baking such tracks and preserves the original keys so this can be changed later without losing information.
