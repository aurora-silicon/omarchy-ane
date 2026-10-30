#!/usr/bin/env python3
"""Score one ane_h15 stage 5 part 2 output directory into outcome A to E, or UNSCORED.

usage: score_h15_part2.py OUTDIR      (the folder the part 2 wrapper wrote; reads files only, runs nothing)

Reads wrapper.log, ane_h15-dmesg.txt, data-before.txt, data-after.txt and, when present, compare.txt. The page comparison
is recomputed from the two hash files and must agree with compare.txt. Anything missing, inconsistent or unexpected is
UNSCORED with the reason; it is never mapped to a letter by guessing.

A  the module logged a mailbox message                       B  no message, hashed data pages changed
C  no message, data unchanged, the gate opened/forced/was open   D  the power gate never opened
E  the job never finished (machine went down)
"""
import re
import sys
from pathlib import Path

NEXT = {
    "A": "Stage 6, one run, no other change. It writes the two mailbox send registers for the first time on an H15 part: say so in the ask.",
    "B": "The firmware probably executed (a net change in the hashed range, nothing more). The mailbox transport or the interrupt wiring is the next suspect. Rerun stage 5 on a build that logs the post-wait status line; do not widen the read guard.",
    "C": "The power gate is not the missing step on this chip, or the firmware stalls earlier. Try pwgate_t6031=2 (always write 0 first) before any other offset; the others are hypotheses only.",
    "D": "The core never started and no reboot is needed. Log the window word, read it once more after a long wait, then check the power domain status registers before any new write.",
    "E": "Stop. Do not retry. Ask for the last line of the per-line log; a crash stops the run for a fix.",
}
RESULT = re.compile(r"ane_h15 RESULT stage=5 soc=\S+ verdict=(\S+) reason=(.*)")


def unscored(why, **extra):
    return {"branch": "UNSCORED", "why": why, "next": "Not a branch: resolve the reason first, then rescore.", **extra}


def pages(path):
    return dict(line.split() for line in path.read_text().splitlines() if line.strip())


def log_facts(dmesg, reason):
    """What the module logged, without judging it."""
    gate = re.search(r"pwgate=(opened|forced|already-open)", dmesg)
    info = {"pwgate": gate.group(1) if gate else None, "reboot_required": "core-not-stopped" in reason}
    post = re.search(r"post-wait cpu_status=(\S+) a2i_ctrl=(\S+) i2a_ctrl=(\S+)", dmesg)
    if post:
        info["post_wait"] = dict(zip(("cpu_status", "a2i_ctrl", "i2a_ctrl"), post.groups()))
    counts = [len(re.findall(r"ane_h15 msg\d+ raw0=", dmesg))] + [int(m) for m in re.findall(r"\bmessages=(\d+)", dmesg)]
    info["messages"] = max(counts)
    return info


def compare_disagrees(out, count):
    """The wrapper's own compare.txt line, when present, must agree with the hashes; returns the count it states or None."""
    compare = out / "compare.txt"
    listed = re.search(r"differing (\d+)", compare.read_text()) if compare.exists() else None
    return int(listed.group(1)) if listed and int(listed.group(1)) != count else None


def classify_data(out, info):
    """B or C from the two page hashes, or UNSCORED when they cannot be trusted."""
    try:
        before, after = pages(out / "data-before.txt"), pages(out / "data-after.txt")
    except (OSError, ValueError) as error:
        return unscored(f"unreadable page hashes: {error}", **info)
    if not before or before.keys() != after.keys():
        return unscored(f"the two hashes cover different pages ({len(before)} before, {len(after)} after)", **info)
    diff = sorted(int(k) for k in before if before[k] != after[k])
    stated = compare_disagrees(out, len(diff))
    if stated is not None:
        return unscored(f"compare.txt says {stated} differing pages, the hashes say {len(diff)}", **info)
    info["differing_pages"] = diff
    if diff:
        return {"branch": "B", "why": f"{len(diff)} hashed page(s) changed, no message", "next": NEXT["B"], **info}
    if not info["pwgate"]:
        return unscored("data unchanged but the log has no pwgate line to say whether the gate opened", **info)
    return {"branch": "C", "why": f"data unchanged, no message, pwgate {info['pwgate']}", "next": NEXT["C"], **info}


def no_result(dmesg):
    """The module logged no RESULT line: say whether it ever ran at all."""
    if [line for line in dmesg.splitlines() if "ane_h15" in line and "taints kernel" not in line]:
        return unscored("no RESULT line from the module")
    return unscored("the module loaded but never ran: its only kernel line is the taint notice, so probe() was never "
                    "called and no device-tree node matched; unchanged data says nothing about the firmware", never_probed=True,
                    next="Put the ANE device-tree node on the boot (the opt-in overlay; the DART variant for map_fw=1), make the job "
                         "refuse before loading when the node is absent, then run part 2 again.")


def score(out):
    out = Path(out)
    try:
        wrapper = (out / "wrapper.log").read_text()
        dmesg = (out / "ane_h15-dmesg.txt").read_text()
    except OSError as error:
        return unscored(f"missing input: {error.filename}")
    if not re.search(r"h15 part2 wrapper end .* insmod_rc=\d+ second_hash_rc=\d+", wrapper):
        return {"branch": "E", "why": "the wrapper never reached its end line", "next": NEXT["E"]}
    results = RESULT.findall(dmesg)
    if not results:
        return no_result(dmesg)
    verdict, reason = results[-1]
    reason = reason.strip()
    if verdict == "REFUSED":
        return unscored(f"the module refused the run: {reason}")
    info = log_facts(dmesg, reason)
    if "pwgate-not-open" in reason:
        return {"branch": "D", "why": reason, "next": NEXT["D"], **info}
    if info["messages"]:
        return {"branch": "A", "why": f"{info['messages']} mailbox message(s) logged", "next": NEXT["A"], **info}
    return classify_data(out, info)


def main(argv):
    if len(argv) != 2:
        print((__doc__ or "").split("\n\n")[0], file=sys.stderr)
        return 2
    result = score(argv[1])
    print(f"branch {result['branch']}: {result['why']}")
    for key in ("pwgate", "messages", "differing_pages", "post_wait", "reboot_required"):
        if key in result:
            print(f"{key}: {result[key]}")
    print(f"next: {result['next']}")
    return 0 if result["branch"] != "UNSCORED" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
