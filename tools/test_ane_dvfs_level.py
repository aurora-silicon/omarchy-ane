#!/usr/bin/env python3
"""Source checks for the ANE clock level in ane/src/ane_dvfs.c (T6001).
There is no device and no kernel build here: the test reads the C file and checks what the 30 minute level 5 soak and the opt-down rule need.
Level 5 is the default and the highest level the driver writes; the parameter dvfs_level opts down to 4 .. 1 or 0; level 0 touches no register;
the release path undoes the level that was requested, not a constant; perf_state names the level in effect."""
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
repo = Path(__file__).resolve().parents[1]
src = (repo / "ane/src/ane_dvfs.c").read_text()
fails = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        fails.append(what)


def define(name):
    m = re.search(r"^#define\s+" + name + r"\s+(\d+)\s*$", src, re.M)
    return int(m.group(1)) if m else None


check(define("ANE_DVFS_LEVEL_DEFAULT") == 5, "the default level is 5")
check(define("ANE_DVFS_LEVEL_MAX") == 5, "the highest level the driver writes is 5")
check(re.search(r"module_param\(dvfs_level,\s*uint,\s*0644\)", src) is not None, "module parameter dvfs_level (uint, 0644)")
desc = re.search(r'MODULE_PARM_DESC\(dvfs_level,\s*"([^"]*)"', src)
check(bool(desc) and all(w in desc.group(1) for w in ("0", "1", "5")), "the parameter description names levels 0, 1 and 5")
check(re.search(r"\bANE_DVFS_LEVEL\b", src) is None, "no fixed level constant is left (ANE_DVFS_LEVEL)")

up = src[src.index("void ane_dvfs_up("):src.index("void ane_dvfs_down(")]
first_hw = min(up.index(t) for t in ("ptd_read(", "readl(", "writel(", "ptd_write("))
zero = re.search(r"if\s*\(\s*!level\s*\)", up)
check(bool(zero) and zero.start() < first_hw, "level 0 returns before any register is read or written")
check("level = min(" in up or "ANE_DVFS_LEVEL_MAX" in up, "a value above the maximum is clamped")
check(up.count("level") >= 6 and "d->level = level" in up, "the level that was requested is remembered")
check(all('"level-%d"' % n in src for n in range(6)), "perf_state names level-0 to level-5")
rel = src[src.index("static void ane_dvfs_release("):src.index("/* Called at the end of runtime resume")]
check("dvfs_cmd(ane, d->level, 0)" in rel, "the release undoes the level that was requested")
check("level-0-by-parameter" in src, "perf_state says when level 0 is the parameter's choice")

sys.exit(1 if fails else 0)
