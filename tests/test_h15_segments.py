#!/usr/bin/env python3
"""Host checks for ane/h15/h15-segments.py. No device, no real ADT dump: the ADT is built here
from the record format the script's docstring states, two 32 byte little-endian records
{phys, iova offset, remap iova, size | flags << 32}. The T6031 boards name the node
/arm-io/ane0, not /arm-io/ane, so the lookup must try ane0 first and still find ane.

Run: python3 tests/test_h15_segments.py (exit 0 required)."""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ane" / "h15" / "h15-segments.py"

TEXT = (0x10000AC4000, 0x0, 0x10000000000, 0xB0000 | (1 << 32))
DATA = (0x10001544000, 0xB0000, 0x100000B0000, 0x43C000)
RECORD_0 = "record 0: phys=0x10000ac4000 iova_offset=0x0 remap=0x10000000000 size=0xb0000 flags=0x1"
RECORD_1 = "record 1: phys=0x10001544000 iova_offset=0xb0000 remap=0x100000b0000 size=0x43c000 flags=0x0"
FW_SEG = "fw_seg=10000ac4000:b0000,10001544000:43c000"


def prop(name, value):
    if isinstance(value, str):
        value = value.encode()
    field = (name.encode() + b"\0" * 32)[:32]
    pad = (4 - len(value) % 4) % 4
    return field + struct.pack("<I", len(value)) + value + b"\0" * pad


def node(props, kids):
    return struct.pack("<II", len(props), len(kids)) + b"".join(props) + b"".join(kids)


def adt(ane_name):
    ane = node([prop("name", ane_name),
                prop("segment-ranges",
                     struct.pack("<QQQQ", *TEXT) + struct.pack("<QQQQ", *DATA))], [])
    arm_io = node([prop("name", "arm-io")], [ane])
    return node([prop("name", "device-tree")], [arm_io])


def run_script(blob):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "adt.bin"
        p.write_bytes(blob)
        return subprocess.run([sys.executable, str(SCRIPT), str(p)],
                              capture_output=True, text=True)


def ane_node_lookup(ane_name):
    r = run_script(adt(ane_name))
    assert r.returncode == 0, r.stdout + r.stderr
    assert RECORD_0 in r.stdout, r.stdout
    assert RECORD_1 in r.stdout, r.stdout
    assert FW_SEG in r.stdout, r.stdout


def no_ane_node_refuses():
    arm_io = node([prop("name", "arm-io")], [])
    r = run_script(node([prop("name", "device-tree")], [arm_io]))
    assert r.returncode != 0, r.stdout + r.stderr
    assert "no /arm-io/ane node in this ADT" in r.stdout + r.stderr, r.stdout + r.stderr


if __name__ == "__main__":
    bad = 0
    for name in ("ane0", "ane"):
        try:
            ane_node_lookup(name)
            print(f"PASS node {name}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL node {name}: {e}")
    try:
        no_ane_node_refuses()
        print("PASS no ane node refuses")
    except AssertionError as e:
        bad += 1
        print(f"FAIL no ane node refuses: {e}")
    sys.exit(1 if bad else 0)
