#!/usr/bin/env python3
"""
Print texture names and IDs from a standalone PES KTMDL.

Usage:
    python list_ktmdl_textures.py model.ktmdl

The script does not require Blender or the KTMDL add-on.
"""

import argparse
import os
import struct
import sys


MAGIC = b"KTMDL\x00\x00\x00"
HEADER_SIZE = 0xC0


def detect_endian(data):
    if len(data) < HEADER_SIZE:
        raise ValueError("File is smaller than the KTMDL 0xC0-byte header")

    if data[:8] != MAGIC:
        raise ValueError("Not a standalone KTMDL file")

    marker = data[0x10:0x12]
    if marker == b"\x01\x00":
        return "<"
    if marker == b"\x00\x01":
        return ">"

    # Fallback: compare the header-reported file size with the real size.
    le_size = struct.unpack_from("<I", data, 0x90)[0]
    be_size = struct.unpack_from(">I", data, 0x90)[0]

    if be_size == len(data) and le_size != len(data):
        return ">"
    return "<"


def check_range(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(
            "Read outside file: offset=0x%X size=0x%X file=0x%X"
            % (offset, size, len(data))
        )


def u32(data, offset, endian):
    check_range(data, offset, 4)
    return struct.unpack_from(endian + "I", data, offset)[0]


def i32(data, offset, endian):
    check_range(data, offset, 4)
    return struct.unpack_from(endian + "i", data, offset)[0]


def u64(data, offset, endian):
    check_range(data, offset, 8)
    return struct.unpack_from(endian + "Q", data, offset)[0]


def cstring(data, offset):
    check_range(data, offset, 1)
    end = data.find(b"\x00", offset)
    if end < 0:
        end = len(data)

    raw = data[offset:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", "replace")


def parse_texture_names(data, endian, texture_count, debug_info_offset):
    names = []

    if texture_count == 0:
        return names

    if debug_info_offset <= 0:
        return ["texture_%03d" % i for i in range(texture_count)]

    check_range(data, debug_info_offset, 0x0C)

    # ktModelDataDebugInfo:
    # +0x00 uint flag
    # +0x04 int  textureNameOffset
    # +0x08 int  shaderNameOffset
    texture_name_offset = i32(data, debug_info_offset + 0x04, endian)

    if texture_name_offset <= 0:
        return ["texture_%03d" % i for i in range(texture_count)]

    table = debug_info_offset + texture_name_offset
    check_range(data, table, texture_count * 4)

    for i in range(texture_count):
        relative_string_offset = u32(data, table + i * 4, endian)
        names.append(cstring(data, table + relative_string_offset))

    return names


def id_filename_candidates(hi, lo):
    """These are the same ID basename forms accepted by importer v1.2.0."""
    candidates = [
        "%016x%016x" % (hi, lo),
        "%016x_%016x" % (hi, lo),
        "%016x-%016x" % (hi, lo),
        "%016x" % hi,
        "%016x" % lo,
    ]
    return [c for c in candidates if set(c) != {"0"}]


def main():
    parser = argparse.ArgumentParser(
        description="Print texture names and IDs stored in a PES KTMDL"
    )
    parser.add_argument("ktmdl", help="Path to the standalone KTMDL file")
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Also print the ID-based basenames accepted by importer v1.2.0",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.ktmdl)

    with open(path, "rb") as f:
        data = f.read()

    endian = detect_endian(data)

    texture_count = u32(data, 0x7C, endian)
    texture_id_offset = i32(data, 0x80, endian)
    debug_info_offset = i32(data, 0x60, endian)

    if texture_count and texture_id_offset <= 0:
        raise ValueError(
            "KTMDL reports %d textures but textureNameIdOffset is invalid"
            % texture_count
        )

    names = parse_texture_names(
        data,
        endian,
        texture_count,
        debug_info_offset,
    )

    print("KTMDL :", path)
    print("Endian:", "little" if endian == "<" else "big")
    print("Textures:", texture_count)
    print()

    for i in range(texture_count):
        entry = texture_id_offset + i * 0x10
        hi = u64(data, entry + 0x00, endian)
        lo = u64(data, entry + 0x08, endian)

        name = names[i] if i < len(names) else "texture_%03d" % i

        print("[%03d]" % i)
        print("  Name : %s" % name)
        print("  HI   : %016x" % hi)
        print("  LO   : %016x" % lo)
        print("  ID   : %016x%016x" % (hi, lo))

        if args.candidates:
            print("  Importer ID basenames:")
            for candidate in id_filename_candidates(hi, lo):
                print("    %s" % candidate)

        print()


if __name__ == "__main__":
    try:
        main()
        input("Press a key to exit")
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
