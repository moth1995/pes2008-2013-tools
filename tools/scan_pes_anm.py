"""
scan_pes_anm.py

Structural scanner for PES/Konami ANM data.

Supported containers:
  * PES WESYS-family indexed animation bank observed in unnamed_1.bin
  * classic CRI AFS (AFS\\0)
  * nested AFS/WESYS containers recursively
  * arbitrary/raw binary blobs

Detection is structural, not extension-based. A candidate FF010001 core must link
correctly to FF010003 bone-info and FF010002 curve-data chunks and its curves
must satisfy the layouts observed in the PES corpus.

Known serialized value formats:
  0x1C  packed 48-bit smallest-three quaternion
  0x1D  raw float3 vector
  0x1E  unknown 16-byte value; only one nKeys=1 sample observed
  0x1F  float3 base + half3 delta vector

Known curve types:
  0     explicit-time/keyed (uint16 times[])
  3     implicit-time (usually one value per frame, but nKeys=1 constants exist)

Examples:
  python scan_pes_anm.py unnamed_1.bin
  python scan_pes_anm.py unnamed_1.bin --verbose
  python scan_pes_anm.py unnamed_1.bin --extract-dir extracted_anm
  python scan_pes_anm.py archive.afs --extract-dir out --max-depth 8
  python scan_pes_anm.py archive.afs --extract-dir out --extract-mode cores
  python scan_pes_anm.py file.bin --raw
  python scan_pes_anm.py unnamed_1.bin --json report.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

ANM_MAGIC_FILE = 0xFF010001
ANM_MAGIC_BONE_INFO = 0xFF010003
ANM_MAGIC_CURVE_DATA = 0xFF010002

ANM_VALUE_QUAT48 = 0x001C
ANM_VALUE_FLOAT3 = 0x001D
ANM_VALUE_UNKNOWN_1E = 0x001E
ANM_VALUE_BASE_HALF3 = 0x001F

CURVE_EXPLICIT_TIME = 0
CURVE_IMPLICIT_TIME = 3

KNOWN_VALUE_FORMATS = {
    ANM_VALUE_QUAT48,
    ANM_VALUE_FLOAT3,
    ANM_VALUE_UNKNOWN_1E,
    ANM_VALUE_BASE_HALF3,
}
KNOWN_CURVE_TYPES = {CURVE_EXPLICIT_TIME, CURVE_IMPLICIT_TIME}

AFS_MAGIC = b"AFS\x00"


def u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def s8(data: bytes, off: int) -> int:
    return struct.unpack_from("<b", data, off)[0]


def u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def align16(value: int) -> int:
    return (value + 15) & ~15


def in_range(size: int, off: int, length: int = 1) -> bool:
    return 0 <= off <= size and 0 <= length and off + length <= size


@dataclass
class CurveInfo:
    offset: int
    value_format: int
    curve_type: int
    curve_channel: int
    n_keys: int
    target_no: int
    times_offset: int
    values_offset: int
    record_end: int
    inline: bool = False
    table_slot: Optional[int] = None


@dataclass
class CoreInfo:
    offset: int
    total_frame: int
    loop: int
    reserved07: int
    reserved08: int
    reserved0C: int
    reserved18: int
    bone_info_offset: int
    curve_data_offset: int
    bone_count: int
    connect_offset: int
    bone_chunk_size: int
    curve_table_end: int
    curve_table_slots: int
    curves: List[CurveInfo] = field(default_factory=list)
    core_end: int = 0


@dataclass
class EntryInfo:
    # Full logical path through nested containers. Example:
    #   [12]          -> top-level entry 12
    #   [12, 0]       -> child 0 inside top-level entry 12
    #   [12, 0, 3]    -> child 3 one level deeper
    path: List[int]
    index: int
    archive_offset: int       # absolute offset in the original input file
    size: int
    flag: int
    source_container: str
    main_core_offset: Optional[int]
    core_count: int
    bundled_eye: bool
    sequence_version: Optional[int]
    fps: Optional[int]
    cores: List[CoreInfo] = field(default_factory=list)


@dataclass
class ContainerEntry:
    index: int
    offset: int
    size: int
    flag: int = 0


@dataclass
class ContainerInfo:
    kind: str
    entries: List[ContainerEntry]
    notes: dict = field(default_factory=dict)


def value_payload_size(curve: CurveInfo) -> Optional[int]:
    if curve.value_format == ANM_VALUE_QUAT48:
        return curve.n_keys * 6
    if curve.value_format == ANM_VALUE_FLOAT3:
        return curve.n_keys * 12
    if curve.value_format == ANM_VALUE_BASE_HALF3:
        return 12 + curve.n_keys * 6
    if curve.value_format == ANM_VALUE_UNKNOWN_1E:
        # Corpus: exactly one sample, nKeys=1, physical value payload=0x10.
        # Do not extrapolate an unproven multi-key layout.
        return 16 if curve.n_keys == 1 else None
    return None


def parse_curve_header(data: bytes, curve_off: int, limit: int, strict: bool) -> Optional[CurveInfo]:
    if not in_range(limit, curve_off, 0x10):
        return None

    value_format = u16(data, curve_off + 0x00)
    curve_type = data[curve_off + 0x02]
    curve_channel = data[curve_off + 0x03]
    n_keys = u16(data, curve_off + 0x04)
    target_no = u16(data, curve_off + 0x06)
    times_offset = u32(data, curve_off + 0x08)
    values_offset = u32(data, curve_off + 0x0C)

    if value_format not in KNOWN_VALUE_FORMATS:
        return None
    if curve_type not in KNOWN_CURVE_TYPES:
        return None
    if n_keys == 0:
        return None
    if values_offset < 0x10:
        return None

    if curve_type == CURVE_EXPLICIT_TIME:
        if times_offset < 0x10:
            return None
        if not in_range(limit, curve_off + times_offset, n_keys * 2):
            return None
        expected_values = align16(times_offset + n_keys * 2)
        if strict and values_offset != expected_values:
            return None
    else:
        if strict and times_offset != 0:
            return None
        if strict and values_offset != 0x10:
            return None

    info = CurveInfo(
        offset=curve_off,
        value_format=value_format,
        curve_type=curve_type,
        curve_channel=curve_channel,
        n_keys=n_keys,
        target_no=target_no,
        times_offset=times_offset,
        values_offset=values_offset,
        record_end=0,
    )

    payload = value_payload_size(info)
    if payload is None:
        return None

    value_end_rel = values_offset + payload
    record_size = align16(value_end_rel)
    record_end = curve_off + record_size

    if not in_range(limit, curve_off + values_offset, payload):
        return None
    if record_end > limit:
        return None

    info.record_end = record_end

    if strict and curve_channel != 0:
        # All 8,147 curves in the supplied corpus use channel 0.
        return None

    return info


def parse_core(data: bytes, core_off: int, strict: bool = True) -> Optional[CoreInfo]:
    size = len(data)
    if not in_range(size, core_off, 0x1C):
        return None
    if u32(data, core_off) != ANM_MAGIC_FILE:
        return None

    total_frame = u16(data, core_off + 0x04)
    loop = data[core_off + 0x06]
    reserved07 = data[core_off + 0x07]
    reserved08 = u32(data, core_off + 0x08)
    reserved0C = u32(data, core_off + 0x0C)
    bone_info_rel = u32(data, core_off + 0x10)
    curve_data_rel = u32(data, core_off + 0x14)
    reserved18 = u32(data, core_off + 0x18)

    bone_off = core_off + bone_info_rel
    curve_data_off = core_off + curve_data_rel

    if bone_info_rel < 0x1C or curve_data_rel < 0x1C:
        return None
    if not in_range(size, bone_off, 0x10):
        return None
    if not in_range(size, curve_data_off, 0x0C):
        return None
    if u32(data, bone_off) != ANM_MAGIC_BONE_INFO:
        return None
    if u32(data, curve_data_off) != ANM_MAGIC_CURVE_DATA:
        return None

    bone_count = u32(data, bone_off + 0x04)
    connect_rel = u32(data, bone_off + 0x08)
    bone_chunk_size = u32(data, bone_off + 0x0C)

    if bone_count == 0 or bone_count > 4096:
        return None
    if connect_rel < 0x10:
        return None
    connect_off = bone_off + connect_rel
    if not in_range(size, connect_off, bone_count * 2):
        return None
    if bone_chunk_size < connect_rel + bone_count * 2:
        return None

    bone_numbers = []
    roots = 0
    for i in range(bone_count):
        p = connect_off + i * 2
        bone_no = data[p]
        parent_no = s8(data, p + 1)
        if bone_no >= bone_count:
            return None
        if parent_no < -1 or parent_no >= bone_count:
            return None
        roots += parent_no == -1
        bone_numbers.append(bone_no)

    if len(set(bone_numbers)) != bone_count:
        return None
    if strict and roots == 0:
        return None

    if strict:
        if reserved07 != 0 or reserved08 != 0 or reserved0C != 0 or reserved18 != 0:
            return None
        if u32(data, curve_data_off + 0x04) != 0:
            return None

    table_end = u32(data, curve_data_off + 0x08)
    curves: List[CurveInfo] = []

    # Static/no-curve auxiliary cores are valid in the corpus.
    if table_end == 0:
        core_end = core_off + align16(curve_data_rel + 0x0C)
        if core_end > size:
            return None
        return CoreInfo(
            offset=core_off,
            total_frame=total_frame,
            loop=loop,
            reserved07=reserved07,
            reserved08=reserved08,
            reserved0C=reserved0C,
            reserved18=reserved18,
            bone_info_offset=bone_info_rel,
            curve_data_offset=curve_data_rel,
            bone_count=bone_count,
            connect_offset=connect_rel,
            bone_chunk_size=bone_chunk_size,
            curve_table_end=table_end,
            curve_table_slots=0,
            curves=[],
            core_end=core_end,
        )

    if table_end < 0x0C or (table_end - 0x0C) % 4 != 0:
        return None
    if not in_range(size, curve_data_off, table_end):
        return None

    slot_count = (table_end - 0x0C) // 4
    if slot_count <= 0 or slot_count > 65536:
        return None

    offsets = [u32(data, curve_data_off + 0x0C + i * 4) for i in range(slot_count)]

    if strict:
        # Every non-empty curve table in the 512-core corpus has a final zero
        # terminator and no nonzero entries after the first zero.
        if offsets[-1] != 0:
            return None
        zero_seen = False
        for rel in offsets:
            if rel == 0:
                zero_seen = True
            elif zero_seen:
                return None

    # Inline/unreferenced curve begins immediately after the offset table in all
    # non-static supplied cores. Its semantic role depends on the core:
    # player main = root rotation; EYE auxiliary = vector curve.
    inline_off = curve_data_off + table_end
    inline = parse_curve_header(data, inline_off, size, strict)
    if inline is not None:
        inline.inline = True
        curves.append(inline)
    elif strict:
        return None

    seen = set()
    for slot, rel in enumerate(offsets):
        if rel == 0:
            continue
        curve_off = curve_data_off + rel
        if curve_off in seen:
            return None
        seen.add(curve_off)
        curve = parse_curve_header(data, curve_off, size, strict)
        if curve is None:
            return None
        curve.table_slot = slot
        curves.append(curve)

    if not curves:
        return None

    if strict:
        if any(c.target_no >= bone_count for c in curves):
            return None

        # Implicit-time curves are not always dense: nKeys=1 constants exist.
        # For multi-key implicit curves, all corpus examples are per-frame.
        for c in curves:
            if c.curve_type == CURVE_IMPLICIT_TIME and c.n_keys > 1:
                if c.n_keys != total_frame + 1:
                    return None

        # Explicit time arrays are sorted and bounded by totalFrame.
        for c in curves:
            if c.curve_type == CURVE_EXPLICIT_TIME:
                times = [u16(data, c.offset + c.times_offset + 2 * k) for k in range(c.n_keys)]
                if times != sorted(times):
                    return None
                if times and times[-1] > total_frame:
                    return None

    core_end = max(c.record_end for c in curves)

    return CoreInfo(
        offset=core_off,
        total_frame=total_frame,
        loop=loop,
        reserved07=reserved07,
        reserved08=reserved08,
        reserved0C=reserved0C,
        reserved18=reserved18,
        bone_info_offset=bone_info_rel,
        curve_data_offset=curve_data_rel,
        bone_count=bone_count,
        connect_offset=connect_rel,
        bone_chunk_size=bone_chunk_size,
        curve_table_end=table_end,
        curve_table_slots=slot_count,
        curves=curves,
        core_end=core_end,
    )


def find_cores(data: bytes, strict: bool = True) -> List[CoreInfo]:
    magic = struct.pack("<I", ANM_MAGIC_FILE)
    result = []
    pos = 0
    while True:
        pos = data.find(magic, pos)
        if pos < 0:
            break
        core = parse_core(data, pos, strict)
        if core is not None:
            result.append(core)
        pos += 1
    return result


def parse_wesys_bank(data: bytes) -> Optional[ContainerInfo]:
    # Observed PES animation-bank variant:
    #   +0x00  00 01 00 'WESYS'
    #   +0x08  uint32 payloadSize (= fileSize-0x10 here)
    #   +0x0C  uint32 0
    #   +0x10  uint32 logical slot count
    #   +0x14  uint32 observed 8 (exact meaning unknown)
    #   +0x18  slot[count] {uint32 relOffset,uint32 size,uint32 flag}
    # relOffset is relative to file +0x10.
    if len(data) < 0x18:
        return None
    if data[3:8] != b"WESYS":
        return None

    count = u32(data, 0x10)
    unknown14 = u32(data, 0x14)
    if count == 0 or count > 1_000_000:
        return None
    if 0x18 + count * 12 > len(data):
        return None

    entries = []
    for i in range(count):
        rel, size, flag = struct.unpack_from("<III", data, 0x18 + i * 12)
        if size == 0:
            entries.append(ContainerEntry(i, 0, 0, flag))
            continue
        off = 0x10 + rel
        if not in_range(len(data), off, size):
            return None
        entries.append(ContainerEntry(i, off, size, flag))

    return ContainerInfo(
        kind="WESYS_INDEXED_BANK",
        entries=entries,
        notes={
            "payload_size_field": u32(data, 0x08),
            "unknown_0x14": unknown14,
            "offset_base": 0x10,
            "table_offset": 0x18,
            "table_record_size": 12,
        },
    )


def parse_afs(data: bytes) -> Optional[ContainerInfo]:
    if len(data) < 8 or data[:4] != AFS_MAGIC:
        return None
    count = u32(data, 4)
    if count > 1_000_000 or 8 + count * 8 > len(data):
        return None

    entries = []
    for i in range(count):
        off, size = struct.unpack_from("<II", data, 8 + i * 8)
        if off == 0 and size == 0:
            entries.append(ContainerEntry(i, 0, 0, 0))
            continue
        if not in_range(len(data), off, size):
            return None
        entries.append(ContainerEntry(i, off, size, 0))
    return ContainerInfo(kind="CRI_AFS", entries=entries)


def player_info2_main_core_offset(blob: bytes) -> Optional[int]:
    # anmPlInfo2::PL_ANIME_INFO.offset_anime = +0x34
    if len(blob) < 0x40:
        return None
    off = u16(blob, 0x34)
    if off == 0:
        return None
    return off if parse_core(blob, off, strict=True) is not None else None


def parse_sequence_summary(blob: bytes) -> Tuple[Optional[int], Optional[int], Optional[int], bool]:
    # anmPlInfo2::PL_ANIME_INFO.offset_clani = +0x32
    if len(blob) < 0x40:
        return None, None, None, False
    seq = u16(blob, 0x32)
    if seq == 0 or seq + 12 > len(blob):
        return None, None, None, False
    version = u32(blob, seq)
    bundle_rel = u32(blob, seq + 4)
    fps = u16(blob, seq + 8)
    bundled_eye = False
    bundle_abs = None
    if bundle_rel:
        bundle_abs = seq + bundle_rel
        if bundle_abs + 12 <= len(blob):
            n = u32(blob, bundle_abs)
            first = u32(blob, bundle_abs + 4) if n else 0
            if n == 1 and first == 0x0C and blob[bundle_abs + 8:bundle_abs + 12] == b"EYE\x00":
                bundled_eye = parse_core(blob, bundle_abs + first, strict=True) is not None
    return version, fps, bundle_abs, bundled_eye


def classify_blob(
    blob: bytes,
    *,
    path: List[int],
    archive_offset: int,
    flag: int,
    source_container: str,
    strict: bool,
) -> Optional[EntryInfo]:
    cores = find_cores(blob, strict=strict)
    if not cores:
        return None

    main_off = player_info2_main_core_offset(blob)
    seq_version, fps, bundle_abs, bundled_eye = parse_sequence_summary(blob)

    return EntryInfo(
        path=list(path),
        index=path[-1] if path else 0,
        archive_offset=archive_offset,
        size=len(blob),
        flag=flag,
        source_container=source_container,
        main_core_offset=main_off,
        core_count=len(cores),
        bundled_eye=bundled_eye,
        sequence_version=seq_version,
        fps=fps,
        cores=cores,
    )


def classify_entry(
    entry: ContainerEntry,
    archive: bytes,
    strict: bool,
    *,
    base_offset: int = 0,
    path: Optional[List[int]] = None,
    source_container: str = "",
) -> Optional[EntryInfo]:
    if entry.size == 0:
        return None
    blob = archive[entry.offset:entry.offset + entry.size]
    if path is None:
        path = [entry.index]
    return classify_blob(
        blob,
        path=path,
        archive_offset=base_offset + entry.offset,
        flag=entry.flag,
        source_container=source_container,
        strict=strict,
    )


def detect_container(data: bytes) -> Optional[ContainerInfo]:
    return parse_wesys_bank(data) or parse_afs(data)


def scan_container_tree(
    data: bytes,
    *,
    strict: bool,
    max_depth: int,
    base_offset: int = 0,
    path_prefix: Optional[List[int]] = None,
    depth: int = 0,
    force_raw: bool = False,
    stats: Optional[dict] = None,
) -> Tuple[ContainerInfo, List[EntryInfo]]:
    """
    Recursively descend recognized AFS/WESYS containers.

    The returned EntryInfo objects always represent LEAF logical blobs that
    actually contain structurally valid ANM cores. Their archive_offset is
    absolute in the original input, and path records every containing entry.
    """
    if path_prefix is None:
        path_prefix = []
    if stats is None:
        stats = {"nested_containers": 0, "max_depth_seen": depth}

    container = None if force_raw else detect_container(data)
    if container is None:
        raw = ContainerInfo("RAW", [ContainerEntry(0, 0, len(data), 0)])
        hit = classify_blob(
            data,
            path=path_prefix or [0],
            archive_offset=base_offset,
            flag=0,
            source_container="RAW",
            strict=strict,
        )
        return raw, ([hit] if hit else [])

    stats["max_depth_seen"] = max(stats.get("max_depth_seen", 0), depth)
    hits: List[EntryInfo] = []

    for entry in container.entries:
        if entry.size == 0:
            continue

        child_path = path_prefix + [entry.index]
        child_blob = data[entry.offset:entry.offset + entry.size]
        child_abs = base_offset + entry.offset
        nested = detect_container(child_blob)

        if nested is not None and depth < max_depth:
            stats["nested_containers"] = stats.get("nested_containers", 0) + 1
            _, child_hits = scan_container_tree(
                child_blob,
                strict=strict,
                max_depth=max_depth,
                base_offset=child_abs,
                path_prefix=child_path,
                depth=depth + 1,
                force_raw=False,
                stats=stats,
            )
            hits.extend(child_hits)
            continue

        # Either this is a normal leaf, or maximum recursion depth was reached.
        hit = classify_blob(
            child_blob,
            path=child_path,
            archive_offset=child_abs,
            flag=entry.flag,
            source_container=container.kind,
            strict=strict,
        )
        if hit is not None:
            hits.append(hit)

    return container, hits


def fmt_value(v: int) -> str:
    return {
        ANM_VALUE_QUAT48: "QUAT48",
        ANM_VALUE_FLOAT3: "FLOAT3",
        ANM_VALUE_UNKNOWN_1E: "UNKNOWN_1E",
        ANM_VALUE_BASE_HALF3: "BASE+HALF3",
    }.get(v, "0x%04X" % v)


def fmt_curve_type(v: int) -> str:
    return {
        CURVE_EXPLICIT_TIME: "explicit-time",
        CURVE_IMPLICIT_TIME: "implicit-time",
    }.get(v, str(v))


def print_core(prefix: str, c: CoreInfo, verbose: bool) -> None:
    print(
        f"{prefix}core=+0x{c.offset:X} frames=0..{c.total_frame} "
        f"loop={c.loop} bones={c.bone_count} curves={len(c.curves)} "
        f"tableSlots={c.curve_table_slots} end=+0x{c.core_end:X}"
    )
    if verbose:
        for cv in c.curves:
            where = "inline" if cv.inline else f"slot={cv.table_slot}"
            print(
                f"{prefix}  {where:8s} target={cv.target_no:3d} "
                f"fmt={fmt_value(cv.value_format):12s} "
                f"type={fmt_curve_type(cv.curve_type):13s} "
                f"ch={cv.curve_channel} keys={cv.n_keys:4d} "
                f"times=0x{cv.times_offset:X} values=0x{cv.values_offset:X}"
            )


def format_entry_path(path: List[int], root_width: int = 5, depth_width: int = 3) -> str:
    if not path:
        return "entry_00000"
    parts = [f"{path[0]:0{root_width}d}"]
    parts.extend(f"{v:0{depth_width}d}" for v in path[1:])
    return "entry_" + "_".join(parts)


def safe_write(path: Path, payload: bytes, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        # Never overwrite an existing extraction from an earlier run.
        stem = path.stem
        suffix = path.suffix
        n = 1
        while True:
            candidate = path.with_name(f"{stem}_dup{n:03d}{suffix}")
            if not candidate.exists():
                path = candidate
                break
            n += 1
    path.write_bytes(payload)
    return path


def extract_hit(
    out_dir: Path,
    hit: EntryInfo,
    archive: bytes,
    *,
    mode: str,
    root_width: int,
    depth_width: int,
    overwrite: bool,
) -> List[Path]:
    """
    Extract one logical ANM leaf.

    mode=entry:
        Preserve the complete logical file/wrapper exactly.

    mode=cores:
        Extract every FF010001 core separately. If there is more than one,
        append a new depth component: _000, _001, ...

    mode=both:
        Write the complete logical file plus a 'cores' subdirectory containing
        the individual cores. This avoids any filename collision between the
        wrapper and core 0.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = format_entry_path(hit.path, root_width, depth_width)
    blob = archive[hit.archive_offset:hit.archive_offset + hit.size]
    written: List[Path] = []

    if mode in ("entry", "both"):
        out = out_dir / f"{base}.anm"
        written.append(safe_write(out, blob, overwrite))

    if mode in ("cores", "both"):
        core_dir = out_dir / "cores" if mode == "both" else out_dir
        multi = len(hit.cores) > 1
        for i, core in enumerate(hit.cores):
            # Always append a depth component in core mode when multiple cores
            # are present, e.g. entry_00012_000.anm, entry_00012_001.anm.
            # For a single core, keep the logical path unchanged.
            core_path = list(hit.path)
            if multi:
                core_path.append(i)
            core_name = format_entry_path(core_path, root_width, depth_width)
            payload = blob[core.offset:core.core_end]
            out = core_dir / f"{core_name}.anm"
            written.append(safe_write(out, payload, overwrite))

    return written


def report_summary(container: ContainerInfo, hits: List[EntryInfo], tree_stats: Optional[dict] = None) -> dict:
    value_formats = {}
    curve_types = {}
    total_cores = 0
    eye_bundles = 0
    static_cores = 0

    for hit in hits:
        total_cores += len(hit.cores)
        eye_bundles += int(hit.bundled_eye)
        for core in hit.cores:
            static_cores += int(len(core.curves) == 0)
            for cv in core.curves:
                key = f"0x{cv.value_format:02X}"
                value_formats[key] = value_formats.get(key, 0) + 1
                key = str(cv.curve_type)
                curve_types[key] = curve_types.get(key, 0) + 1

    nonempty = sum(e.size > 0 for e in container.entries)
    return {
        "container_kind": container.kind,
        "logical_entries": len(container.entries),
        "nonempty_entries": nonempty,
        "recognized_anm_entries": len(hits),
        "valid_anm_cores": total_cores,
        "bundled_eye_entries": eye_bundles,
        "static_no_curve_cores": static_cores,
        "value_formats": value_formats,
        "curve_types": curve_types,
        "container_notes": container.notes,
        "nested_containers": (tree_stats or {}).get("nested_containers", 0),
        "max_depth_seen": (tree_stats or {}).get("max_depth_seen", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Structurally identify PES/Konami ANM data")
    ap.add_argument("input", help="WESYS bank, CRI AFS, ANM entry, or arbitrary binary")
    ap.add_argument("--raw", action="store_true", help="Force raw structural scan")
    ap.add_argument("--lenient", action="store_true", help="Relax corpus consistency checks")
    ap.add_argument("--verbose", action="store_true", help="Print every core/curve")
    ap.add_argument("--extract-dir", help="Extract recognized ANM logical files/cores")
    ap.add_argument(
        "--extract-mode",
        choices=("entry", "cores", "both"),
        default="entry",
        help=(
            "entry=complete logical ANM file (default); "
            "cores=each FF010001 core separately; "
            "both=logical files plus cores/ subdirectory"
        ),
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum nested AFS/WESYS container depth to descend (default: 8)",
    )
    ap.add_argument(
        "--depth-width",
        type=int,
        default=3,
        help="Digits for nested path components: 3 gives _000, _001, ...",
    )
    ap.add_argument(
        "--root-width",
        type=int,
        default=5,
        help="Digits for top-level entry number (default: 5)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing extracted files; default creates _dupNNN instead",
    )
    ap.add_argument("--json", dest="json_path", help="Write machine-readable report")
    args = ap.parse_args()

    path = Path(args.input)
    data = path.read_bytes()
    strict = not args.lenient

    tree_stats = {"nested_containers": 0, "max_depth_seen": 0}
    if args.raw:
        container, hits = scan_container_tree(
            data,
            strict=strict,
            max_depth=args.max_depth,
            base_offset=0,
            path_prefix=[],
            force_raw=True,
            stats=tree_stats,
        )
    else:
        container, hits = scan_container_tree(
            data,
            strict=strict,
            max_depth=args.max_depth,
            base_offset=0,
            path_prefix=[],
            force_raw=False,
            stats=tree_stats,
        )

    summary = report_summary(container, hits, tree_stats)

    print("Container:", summary["container_kind"])
    print("Logical entries:", summary["logical_entries"])
    print("Non-empty entries:", summary["nonempty_entries"])
    print("Recognized ANM entries:", summary["recognized_anm_entries"])
    print("Valid FF010001 cores:", summary["valid_anm_cores"])
    print("Bundled EYE entries:", summary["bundled_eye_entries"])
    print("Static/no-curve cores:", summary["static_no_curve_cores"])
    print("Value formats:", summary["value_formats"])
    print("Curve types:", summary["curve_types"])
    print("Nested containers descended:", summary["nested_containers"])
    print("Maximum nested depth seen:", summary["max_depth_seen"])

    if args.verbose:
        for hit in hits:
            path_text = format_entry_path(hit.path, args.root_width, args.depth_width)
            print(
                f"[{path_text}] off=0x{hit.archive_offset:X} size=0x{hit.size:X} "
                f"src={hit.source_container} flag=0x{hit.flag:08X} cores={hit.core_count} "
                f"main={('0x%X' % hit.main_core_offset) if hit.main_core_offset is not None else '-'} "
                f"seq={('0x%08X' % hit.sequence_version) if hit.sequence_version is not None else '-'} "
                f"fps={hit.fps if hit.fps is not None else '-'} eye={'yes' if hit.bundled_eye else 'no'}"
            )
            for c in hit.cores:
                print_core("        ", c, True)

    if args.extract_dir:
        out_dir = Path(args.extract_dir)
        extracted_count = 0
        for hit in hits:
            written = extract_hit(
                out_dir,
                hit,
                data,
                mode=args.extract_mode,
                root_width=args.root_width,
                depth_width=args.depth_width,
                overwrite=args.overwrite,
            )
            extracted_count += len(written)
            if args.verbose:
                for out in written:
                    print("extracted", out)
        print("Extracted files:", extracted_count)

    if args.json_path:
        payload = {
            "input": str(path),
            "summary": summary,
            "entries": [asdict(h) for h in hits],
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
