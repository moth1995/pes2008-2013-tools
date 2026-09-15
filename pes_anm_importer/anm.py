# SPDX-License-Identifier: GPL-3.0-or-later
"""PES/Konami ANM parser used by the Blender importer.

Based only on structures confirmed by the supplied PES 2013 ANM samples and
PES_ANM_updated.bt. The parser is intentionally conservative: known curve
payloads are decoded, while unknown data is preserved as hexadecimal metadata.
"""

from __future__ import division

import math
import os
import struct

ANM_MAGIC_FILE = 0xFF010001
ANM_MAGIC_HIERARCHY = 0xFF010003
ANM_MAGIC_ANIMATION = 0xFF010002

ANM_CURVE_ROTATION = 0x001C
ANM_CURVE_TRANSLATION_FLOAT = 0x001D
ANM_CURVE_TRANSLATION_COMPRESSED = 0x001F

ANM_MODE_SPARSE = 0
ANM_MODE_DENSE = 3


class ANMError(Exception):
    pass


class Reader(object):
    def __init__(self, data):
        self.data = data
        self.size = len(data)

    def check(self, offset, size=1):
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ANMError(
                "Read outside ANM file: offset=0x%X size=0x%X file=0x%X"
                % (offset, size, self.size)
            )

    def bytes(self, offset, size):
        self.check(offset, size)
        return self.data[offset : offset + size]

    def u8(self, offset):
        self.check(offset, 1)
        return self.data[offset]

    def i8(self, offset):
        self.check(offset, 1)
        return struct.unpack_from("<b", self.data, offset)[0]

    def u16(self, offset):
        self.check(offset, 2)
        return struct.unpack_from("<H", self.data, offset)[0]

    def u32(self, offset):
        self.check(offset, 4)
        return struct.unpack_from("<I", self.data, offset)[0]

    def f32(self, offset):
        self.check(offset, 4)
        return struct.unpack_from("<f", self.data, offset)[0]

    def half(self, offset):
        self.check(offset, 2)
        try:
            return struct.unpack_from("<e", self.data, offset)[0]
        except (struct.error, ValueError):
            # Fallback IEEE-754 binary16 decoder for unusual Python builds.
            raw = self.u16(offset)
            sign = -1.0 if (raw & 0x8000) else 1.0
            exp = (raw >> 10) & 0x1F
            frac = raw & 0x03FF
            if exp == 0:
                if frac == 0:
                    return math.copysign(0.0, sign)
                return sign * (frac / 1024.0) * (2.0**-14)
            if exp == 31:
                if frac == 0:
                    return sign * float("inf")
                return float("nan")
            return sign * (1.0 + frac / 1024.0) * (2.0 ** (exp - 15))


def _hex(reader, start, end):
    if end <= start:
        return ""
    return reader.bytes(start, end - start).hex()


def _align16(value):
    return (value + 15) & ~15


def _find_core(reader):
    signature = struct.pack("<I", ANM_MAGIC_FILE)
    pos = 0
    while True:
        pos = reader.data.find(signature, pos)
        if pos < 0:
            return -1
        if pos + 0x1C <= reader.size:
            hierarchy_offset = reader.u32(pos + 0x10)
            animation_offset = reader.u32(pos + 0x14)
            hierarchy_pos = pos + hierarchy_offset
            animation_pos = pos + animation_offset
            if (
                hierarchy_pos + 4 <= reader.size
                and animation_pos + 4 <= reader.size
                and reader.u32(hierarchy_pos) == ANM_MAGIC_HIERARCHY
                and reader.u32(animation_pos) == ANM_MAGIC_ANIMATION
            ):
                return pos
        pos += 1


def _find_sequence_header(reader, core_base):
    # Mirrors the structural validation in PES_ANM_updated.bt.
    p = 0
    while p + 14 <= core_base:
        fps = reader.u16(p + 8)
        num_tracks = reader.u16(p + 10)
        if fps <= 0 or fps > 240 or num_tracks <= 0 or num_tracks > 32:
            p += 2
            continue

        header_size = 12 + num_tracks * 2
        if p + header_size > core_base:
            p += 2
            continue
        if reader.u16(p + 12) != header_size:
            p += 2
            continue

        prev = 0
        valid = True
        for i in range(num_tracks):
            cur = reader.u16(p + 12 + i * 2)
            if cur < header_size or (i > 0 and cur <= prev) or p + cur + 6 > core_base:
                valid = False
                break
            prev = cur
        if valid:
            return p
        p += 2
    return -1


def _parse_sequence(reader, seq_base, core_base):
    version = reader.u32(seq_base)
    bundle_offset = reader.u32(seq_base + 4)
    fps = reader.u16(seq_base + 8)
    num_tracks = reader.u16(seq_base + 10)
    offsets = [reader.u16(seq_base + 12 + i * 2) for i in range(num_tracks)]
    header_size = 12 + num_tracks * 2

    tracks = []
    max_used = seq_base + header_size
    for i, rel in enumerate(offsets):
        track_base = seq_base + rel
        next_base = seq_base + offsets[i + 1] if i + 1 < len(offsets) else core_base
        reader.check(track_base, 4)
        event_offset = reader.u16(track_base)
        num_channels = reader.u16(track_base + 2)
        reader.check(track_base + 4, num_channels * 2)
        counts = [reader.u16(track_base + 4 + j * 2) for j in range(num_channels)]
        total_events = sum(counts)
        events_base = track_base + event_offset
        reader.check(events_base, total_events * 16)

        events = []
        for event_index in range(total_events):
            ep = events_base + event_index * 16
            fields = list(struct.unpack_from("<8H", reader.data, ep))
            events.append(
                {
                    "index": event_index,
                    "fields": fields,
                    "rawHex": reader.bytes(ep, 16).hex(),
                }
            )

        used_end = max(
            track_base + 4 + num_channels * 2, events_base + total_events * 16
        )
        max_used = max(max_used, used_end)
        span_end = max(track_base, min(next_base, core_base))
        tracks.append(
            {
                "index": i,
                "offset": rel,
                "absoluteOffset": track_base,
                "eventOffset": event_offset,
                "numChannels": num_channels,
                "eventCounts": counts,
                "totalEvents": total_events,
                "events": events,
                "rawHex": _hex(reader, track_base, span_end),
            }
        )

    return {
        "offset": seq_base,
        "version": version,
        "bundleOffset": bundle_offset,
        "fps": fps,
        "numTracks": num_tracks,
        "trackOffsets": offsets,
        "headerSize": header_size,
        "tracks": tracks,
        "tailHex": _hex(reader, max_used, core_base),
    }


def _decode_packed_quaternion(reader, offset):
    raw = reader.bytes(offset, 6)
    bits = int.from_bytes(raw, byteorder="little", signed=False)
    missing_index = int(bits & 3)
    raw_components = [
        int((bits >> 32) & 0x7FFF),
        int((bits >> 17) & 0x7FFF),
        int((bits >> 2) & 0x7FFF),
    ]
    stored = [(v - 16383.5) / 23169.767578125 for v in raw_components]
    missing = math.sqrt(max(0.0, 1.0 - sum(v * v for v in stored)))

    values = []
    stored_index = 0
    for component_index in range(4):
        if component_index == missing_index:
            values.append(missing)
        else:
            values.append(stored[stored_index])
            stored_index += 1

    # Normalize away tiny quantization drift while retaining the decoded sign.
    length = math.sqrt(sum(v * v for v in values))
    if length > 1.0e-20:
        values = [v / length for v in values]

    return {
        "value": values,  # x, y, z, w
        "missingIndex": missing_index,
        "raw15": raw_components,
        "reservedBit47": int((bits >> 47) & 1),
        "packedHex": raw.hex(),
        "packedValue": bits,
    }


def _parse_track(reader, track_base, span_end, animation_base, is_root=False):
    reader.check(track_base, 0x10)
    curve_type, storage_mode, sample_count, node_index, key_off, value_off = (
        struct.unpack_from("<HHHHII", reader.data, track_base)
    )
    if sample_count > 0x100000:
        raise ANMError(
            "Unreasonable ANM sample count %d at 0x%X" % (sample_count, track_base)
        )

    key_frames = []
    if key_off:
        key_base = track_base + key_off
        reader.check(key_base, sample_count * 2)
        key_frames = (
            list(struct.unpack_from("<%dH" % sample_count, reader.data, key_base))
            if sample_count
            else []
        )

    value_base = track_base + value_off
    reader.check(value_base, 0)
    values = []
    packed_rotations = []
    base_translation = None
    deltas = []
    decoded_end = value_base

    if curve_type == ANM_CURVE_ROTATION:
        reader.check(value_base, sample_count * 6)
        for i in range(sample_count):
            decoded = _decode_packed_quaternion(reader, value_base + i * 6)
            values.append(decoded["value"])
            packed_rotations.append(
                {
                    "missingIndex": decoded["missingIndex"],
                    "raw15": decoded["raw15"],
                    "reservedBit47": decoded["reservedBit47"],
                    "packedHex": decoded["packedHex"],
                    "packedValue": decoded["packedValue"],
                }
            )
        decoded_end = value_base + sample_count * 6
        value_kind = "ROTATION_QUATERNION_48"
    elif curve_type == ANM_CURVE_TRANSLATION_FLOAT:
        reader.check(value_base, sample_count * 12)
        for i in range(sample_count):
            v = list(struct.unpack_from("<3f", reader.data, value_base + i * 12))
            values.append(v)
        decoded_end = value_base + sample_count * 12
        value_kind = "TRANSLATION_FLOAT3"
    elif curve_type == ANM_CURVE_TRANSLATION_COMPRESSED:
        reader.check(value_base, 12 + sample_count * 6)
        base = list(struct.unpack_from("<3f", reader.data, value_base))
        base_translation = base
        for i in range(sample_count):
            p = value_base + 12 + i * 6
            delta = [reader.half(p), reader.half(p + 2), reader.half(p + 4)]
            deltas.append(delta)
            values.append([base[j] + delta[j] for j in range(3)])
        decoded_end = value_base + 12 + sample_count * 6
        value_kind = "TRANSLATION_BASE_PLUS_HALF3"
    else:
        value_kind = "UNKNOWN"

    if span_end < track_base:
        span_end = track_base
    span_end = min(span_end, reader.size)

    return {
        "isRootRotation": bool(is_root),
        "absoluteOffset": track_base,
        "offset": track_base - animation_base,
        "curveType": curve_type,
        "storageMode": storage_mode,
        "sampleCount": sample_count,
        "nodeIndex": node_index,
        "keyFrameOffset": key_off,
        "valueOffset": value_off,
        "keyFrames": key_frames,
        "valueKind": value_kind,
        "values": values,
        "packedRotations": packed_rotations,
        "baseTranslation": base_translation,
        "deltas": deltas,
        "decodedEndOffset": decoded_end,
        "alignedDecodedEndOffset": _align16(decoded_end),
        "rawHex": _hex(reader, track_base, span_end),
    }


def parse_bytes(data, source_name="<memory>"):
    reader = Reader(data)
    core_base = _find_core(reader)
    if core_base < 0:
        raise ANMError("Could not find a validated FF010001/FF010003/FF010002 ANM core")

    seq_base = _find_sequence_header(reader, core_base) if core_base > 0 else -1
    if seq_base >= 0:
        sequence = _parse_sequence(reader, seq_base, core_base)
        unknown_prefix_hex = _hex(reader, 0, seq_base)
    else:
        sequence = None
        unknown_prefix_hex = _hex(reader, 0, core_base)

    # Main file header.
    reader.check(core_base, 0x1C)
    magic = reader.u32(core_base)
    if magic != ANM_MAGIC_FILE:
        raise ANMError("Unexpected ANM core magic at 0x%X" % core_base)
    last_sample_index = reader.u16(core_base + 4)
    unknown06 = reader.u16(core_base + 6)
    unknown08 = reader.u32(core_base + 8)
    unknown0C = reader.u32(core_base + 0x0C)
    hierarchy_offset = reader.u32(core_base + 0x10)
    animation_offset = reader.u32(core_base + 0x14)
    unknown18 = reader.u32(core_base + 0x18)

    # Hierarchy chunk.
    hierarchy_base = core_base + hierarchy_offset
    reader.check(hierarchy_base, 0x10)
    hierarchy_magic = reader.u32(hierarchy_base)
    if hierarchy_magic != ANM_MAGIC_HIERARCHY:
        raise ANMError("Unexpected hierarchy magic at 0x%X" % hierarchy_base)
    node_count = reader.u32(hierarchy_base + 4)
    node_data_offset = reader.u32(hierarchy_base + 8)
    chunk_data_end_offset = reader.u32(hierarchy_base + 0x0C)
    if node_count > 0x10000:
        raise ANMError("Unreasonable ANM hierarchy node count %d" % node_count)
    nodes_base = hierarchy_base + node_data_offset
    reader.check(nodes_base, node_count * 2)
    nodes = []
    for i in range(node_count):
        p = nodes_base + i * 2
        nodes.append(
            {
                "entryIndex": i,
                "nodeIndex": reader.u8(p),
                "parentIndex": reader.i8(p + 1),
                "rawHex": reader.bytes(p, 2).hex(),
            }
        )

    hierarchy_logical_end = hierarchy_base + chunk_data_end_offset
    animation_base = core_base + animation_offset
    if hierarchy_logical_end > animation_base:
        hierarchy_padding_hex = ""
    else:
        hierarchy_padding_hex = _hex(reader, hierarchy_logical_end, animation_base)

    # Animation chunk and offset table.
    reader.check(animation_base, 0x0C)
    animation_magic = reader.u32(animation_base)
    if animation_magic != ANM_MAGIC_ANIMATION:
        raise ANMError("Unexpected animation magic at 0x%X" % animation_base)
    animation_unknown04 = reader.u32(animation_base + 4)
    table_end = reader.u32(animation_base + 8)
    if table_end < 0x0C or (table_end - 0x0C) % 4:
        raise ANMError("Invalid ANM track offset table end 0x%X" % table_end)
    table_count = (table_end - 0x0C) // 4
    reader.check(animation_base + 0x0C, table_count * 4)
    offsets = (
        list(
            struct.unpack_from("<%dI" % table_count, reader.data, animation_base + 0x0C)
        )
        if table_count
        else []
    )

    nonzero_offsets = [o for o in offsets if o]
    track_starts = [table_end] + nonzero_offsets
    unique_sorted = sorted(set(track_starts))
    next_by_offset = {}
    for i, rel in enumerate(unique_sorted):
        next_rel = (
            unique_sorted[i + 1]
            if i + 1 < len(unique_sorted)
            else reader.size - animation_base
        )
        next_by_offset[rel] = next_rel

    tracks = []
    root_rotation = None
    root_base = animation_base + table_end
    if root_base + 2 <= reader.size and reader.u16(root_base) == ANM_CURVE_ROTATION:
        root_rotation = _parse_track(
            reader,
            root_base,
            animation_base
            + next_by_offset.get(table_end, reader.size - animation_base),
            animation_base,
            is_root=True,
        )
        tracks.append(root_rotation)

    referenced_tracks = []
    for table_index, rel in enumerate(offsets):
        if rel == 0:
            continue
        track_base = animation_base + rel
        if track_base >= reader.size:
            raise ANMError("Track offset 0x%X points outside file" % rel)
        span_end = animation_base + next_by_offset.get(
            rel, reader.size - animation_base
        )
        track = _parse_track(
            reader, track_base, span_end, animation_base, is_root=False
        )
        track["tableIndex"] = table_index
        referenced_tracks.append(track)
        tracks.append(track)

    return {
        "sourceName": os.path.basename(source_name),
        "fileSize": reader.size,
        "coreOffset": core_base,
        "unknownBeforeSequenceOrCoreHex": unknown_prefix_hex,
        "sequence": sequence,
        "header": {
            "magic": magic,
            "lastSampleIndex": last_sample_index,
            "unknown06": unknown06,
            "unknown08": unknown08,
            "unknown0C": unknown0C,
            "hierarchyOffset": hierarchy_offset,
            "animationOffset": animation_offset,
            "unknown18": unknown18,
        },
        "hierarchy": {
            "absoluteOffset": hierarchy_base,
            "magic": hierarchy_magic,
            "nodeCount": node_count,
            "nodeDataOffset": node_data_offset,
            "chunkDataEndOffset": chunk_data_end_offset,
            "nodes": nodes,
            "paddingHex": hierarchy_padding_hex,
        },
        "animation": {
            "absoluteOffset": animation_base,
            "magic": animation_magic,
            "unknown04": animation_unknown04,
            "trackOffsetTableEnd": table_end,
            "trackOffsetCount": table_count,
            "trackOffsets": offsets,
            "rootRotation": root_rotation,
            "referencedTracks": referenced_tracks,
            "tracks": tracks,
        },
    }


def parse_file(filepath):
    with open(filepath, "rb") as handle:
        data = handle.read()
    return parse_bytes(data, source_name=filepath)
