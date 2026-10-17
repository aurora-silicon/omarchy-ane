# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren

"""Host tests for qwen_fold_layout.py and the judge and report functions of qwen_fold_set_ab.py. No device, no session."""
import json
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_fold_layout as layout  # noqa: E402

LANES = ("q", "k", "v")


def ports(rows):
    return [{"port": p, "kind": k, "lane": lane, "shape": s} for p, k, lane, s in rows]


def control_manifest():
    """P0 input block, P1 state block, P2 readout plus next input block, P3 a plain program."""
    p0 = {"srcs": ports([("t1", "lane", "x", [1, 8]), ("t5", "state_in", "state0", [6, 3])]),
          "dsts": ports([("t16", "lane", "q", [4]), ("t20", "lane", "k", [4]), ("t22", "lane", "v", [4]),
                         ("t26", "lane", "beta", [4, 1, 1]), ("t35", "lane", "gt", [4, 1, 1]), ("t37", "lane", "z", [4])]),
          "states": [{"in_port": "t5", "in_shape": [6, 3], "out_port": "t38", "out_shape": [6, 3]}], "group_start": True, "group_end": False}
    p1 = {"srcs": ports([("t14", "lane", "q", [4]), ("t4", "lane", "k", [4]), ("t7", "lane", "v", [4]), ("t0", "lane", "beta", [4, 1, 1]),
                         ("t1", "lane", "gt", [4, 1, 1]), ("t2", "state_in", "state0", [4, 4, 4])]),
          "dsts": ports([("t17", "lane", "o", [4])]),
          "states": [{"in_port": "t2", "in_shape": [4, 4, 4], "out_port": "t13", "out_shape": [4, 4, 4]}], "group_start": False, "group_end": False}
    p2 = {"srcs": ports([("t7", "lane", "x", [1, 8]), ("t2", "lane", "o", [4]), ("t0", "lane", "z", [4]), ("t76", "state_in", "state0", [6, 3])]),
          "dsts": ports([("t71", "lane", "x", [1, 8]), ("t87", "lane", "q", [4]), ("t91", "lane", "k", [4]), ("t93", "lane", "v", [4]),
                         ("t97", "lane", "beta", [4, 1, 1]), ("t106", "lane", "gt", [4, 1, 1]), ("t108", "lane", "z", [4])]),
          "states": [{"in_port": "t76", "in_shape": [6, 3], "out_port": "t109", "out_shape": [6, 3]}], "group_start": False, "group_end": False}
    p3 = {"srcs": ports([("t3", "lane", "x", [1, 8])]), "dsts": ports([("t9", "lane", "x", [1, 8])]), "states": [],
          "group_start": False, "group_end": True}
    return {"max_len": 50, "n_programs": 4, "programs": [p0, p1, p2, p3], "host_head": {}, "aneforge_tip": "x"}


def pm_of(entry):
    pm = {"x": {}, "h": {}, "states": {}}
    for s in entry["srcs"]:
        if s["kind"] == "lane":
            pm["x"][s["lane"]] = s["port"]
        else:
            pm["states"][s["port"]] = s["shape"]
    for d in entry["dsts"]:
        pm["h"][d["lane"]] = d["port"]
    if entry["group_start"]:
        pm["group_start"] = True
    if entry["group_end"]:
        pm["group_end"] = True
    return pm


def mil_of(entry, tag):
    shape = lambda s: ", ".join(str(x) for x in s)  # noqa: E731
    args = ", ".join(f"tensor<fp16, [{shape(s['shape'])}]> {s['port']}" for s in entry["srcs"])
    body = "".join(f"        tensor<fp16, [{shape(d['shape'])}]> {d['port']} = identity(x = {entry['srcs'][0]['port']})[name = string(\"{d['port']}\")];\n"
                   for d in entry["dsts"])
    outs = [d["port"] for d in entry["dsts"]] + [s["out_port"] for s in entry["states"]]
    body += "".join(f"        tensor<fp16, [{shape(s['out_shape'])}]> {s['out_port']} = identity(x = {entry['srcs'][0]['port']})[name = string(\"{s['out_port']}\")];\n"
                    for s in entry["states"])
    return f"program(1.3)\n{{\n    func main<ios18>({args}) {{\n{body}    }} -> ({', '.join(outs)});\n}}\n", tag


def write_record(directory, manifest, mil_tags=None):
    """probe.out with PORTMAP lines and mil/NN-<tag>.mil for each program of a manifest."""
    (directory / "mil").mkdir(parents=True)
    lines = []
    for i, e in enumerate(manifest["programs"]):
        text, tag = mil_of(e, (mil_tags or {}).get(i, f"{i:012x}"))
        (directory / "mil" / f"{i:02d}-{tag}.mil").write_text(text)
        lines.append(f"PORTMAP {i} {json.dumps(pm_of(e))}")
    (directory / "probe.out").write_text("\n".join(lines) + "\n")


def folded_manifest(control):
    """P1 and P2 joined: the state block's output o stays inside, its state and the next program's state are both inputs."""
    p1, p2 = control["programs"][1], control["programs"][2]
    f = {"srcs": ports([("a24", "lane", "x", [1, 8]), ("a0", "lane", "z", [4]), ("a16", "lane", "q", [4]), ("a6", "lane", "k", [4]), ("a9", "lane", "v", [4]),
                        ("a2", "lane", "beta", [4, 1, 1]), ("a3", "lane", "gt", [4, 1, 1]), ("a4", "state_in", "state0", [4, 4, 4]),
                        ("a37", "state_in", "state1", [6, 3])]),
         "dsts": json.loads(json.dumps(p2["dsts"])),
         "states": [{"in_port": "a4", "in_shape": [4, 4, 4], "out_port": "a15", "out_shape": [4, 4, 4]},
                    {"in_port": "a37", "in_shape": [6, 3], "out_port": "a70", "out_shape": [6, 3]}], "group_start": False, "group_end": False}
    assert len(f["states"]) == len(p1["states"]) + len(p2["states"])
    m = json.loads(json.dumps(control))
    m["programs"] = [control["programs"][0], f, control["programs"][3]]
    m["n_programs"] = 3
    return m


@pytest.fixture()
def record(tmp_path):
    control = control_manifest()
    folded = folded_manifest(control)
    write_record(tmp_path / "control", control)
    tags = {0: "000000000000", 2: "000000000003"}
    write_record(tmp_path / "F00", folded, {**tags, 1: "00000000ffff"})
    return tmp_path, control, folded


def test_self_check_rebuilds_every_control_entry(record):
    root, control, _ = record
    assert layout.self_check(control, root / "control") == []


def test_fold_builds_and_maps(record):
    root, control, folded = record
    manifest, mapping, problems = layout.build(control, root / "control", root / "F00")
    assert problems == []
    assert mapping == [[0], [1, 2], [3]]
    assert manifest["n_programs"] == 3 and manifest["fold_mapping"] == mapping
    assert layout.norm(manifest["programs"][1]) == layout.norm(folded["programs"][1])
    assert manifest["programs"][0] == control["programs"][0] and manifest["programs"][2] == control["programs"][3]


def test_unfolded_program_with_other_mil_is_refused(record):
    root, control, _ = record
    (root / "F00" / "mil" / "00-000000000000.mil").rename(root / "F00" / "mil" / "00-0000000000aa.mil")
    _, _, problems = layout.build(control, root / "control", root / "F00")
    assert any("is not control 0's" in p for p in problems)


def test_folded_output_lane_with_wrong_shape_is_refused(record):
    root, control, folded = record
    folded["programs"][1]["dsts"][1]["shape"] = [5]
    write_record(root / "F01", folded, {0: "000000000000", 1: "00000000ffff", 2: "000000000003"})
    _, _, problems = layout.build(control, root / "control", root / "F01")
    assert any("output lane" in p for p in problems)


def test_swapped_state_pairs_are_caught_by_shape(record):
    root, control, folded = record
    folded["programs"][1]["states"][0]["out_port"], folded["programs"][1]["states"][1]["out_port"] = "a70", "a15"
    write_record(root / "F02", folded, {0: "000000000000", 1: "00000000ffff", 2: "000000000003"})
    manifest, _, problems = layout.build(control, root / "control", root / "F02")
    # the MIL declares the outputs in its own order; the builder pairs by shape, so both pairs stay shape-consistent
    assert problems == []
    assert {(s["in_shape"] == s["out_shape"]) for s in manifest["programs"][1]["states"]} == {True}


def test_fold_that_drops_a_state_block_input_is_refused(record):
    root, control, folded = record
    folded["programs"][1]["srcs"] = [s for s in folded["programs"][1]["srcs"] if s["lane"] != "beta"]
    write_record(root / "F03", folded, {0: "000000000000", 1: "00000000ffff", 2: "000000000003"})
    _, _, problems = layout.build(control, root / "control", root / "F03")
    assert any("state block" in p and "beta" in p for p in problems)


def test_a_layout_that_leaves_control_programs_over_is_refused(record):
    root, control, _ = record
    (root / "F00" / "probe.out").write_text("\n".join(line for line in (root / "F00" / "probe.out").read_text().splitlines() if not line.startswith("PORTMAP 2")) + "\n")
    (root / "F00" / "mil" / "02-000000000003.mil").rename(root / "F00" / "mil" / "xx-000000000003.mil.bak")
    _, _, problems = layout.build(control, root / "control", root / "F00")
    assert any("consumes" in p for p in problems)


np = pytest.importorskip("numpy")
ab = pytest.importorskip("qwen_fold_set_ab")


def arm_record(arm, programs, step_p50, exec_ms, steps=4, tokens=None):
    return {"arm": arm, "tokens": tokens or list(range(steps)), "steps": steps, "logits_sha256": "a" * 64, "logits_bytes": 4 * steps * 3,
            "step_p10_s": step_p50, "step_p50_s": step_p50, "step_p90_s": step_p50, "walls_s": [step_p50] * steps,
            "exec_ms_per_program_median": {f"prog_{i:03d}": exec_ms.get(i, 1.0) for i in range(programs)}, "kernel_stop": [],
            "boot_id_start": "b", "boot_id_end": "b", "bo_total_before": 7, "bo_total_after": 7, "exch_new": 0}


LAYOUTS = {"A": {"n": 4, "mapping": [[0], [1], [2], [3]]}, "F": {"n": 3, "mapping": [[0], [1, 2], [3]]}}
RUN = {"boot_id_pre": "b", "bo_loaded": 7, "kernel_stop": [], "exch_new": 0}


def run_order(order, p50, exec_f=0.8):
    recs = []
    for arm in order:
        n = 4 if arm == "A" else 3
        ex = {1: 1.0, 2: 1.0} if arm == "A" else {1: exec_f}
        recs.append(arm_record(arm, n, p50[arm].pop(0), ex))
    return recs


def test_saving_inside_the_control_spread_is_not_counted():
    recs = run_order(["A", "F", "A", "F"], {"A": [0.170, 0.175], "F": [0.169, 0.172]})
    failures, report = ab.judge_timing(recs, RUN, LAYOUTS, ["A", "F", "A", "F"], 4, anchor=None)
    assert failures == []
    assert report["layouts"]["F"]["outside_control_spread"] is False
    assert report["layouts"]["F"]["exec_saved_per_fold"][0]["saved_ms"] == pytest.approx(1.2)


def test_saving_beyond_the_control_spread_is_counted():
    recs = run_order(["A", "F", "A", "F"], {"A": [0.170, 0.171], "F": [0.150, 0.151]})
    failures, report = ab.judge_timing(recs, RUN, LAYOUTS, ["A", "F", "A", "F"], 4, anchor=None)
    assert failures == [] and report["layouts"]["F"]["outside_control_spread"] is True
    assert report["layouts"]["F"]["median_ratio"] < 0.9


def test_boot_change_bo_growth_and_missing_program_fail():
    order = ["A", "F"]
    recs = run_order(order, {"A": [0.17], "F": [0.17]})
    recs[1]["boot_id_end"] = "other"
    assert any("boot ids" in f for f in ab.judge_timing(recs, RUN, LAYOUTS, order, 4, anchor=None)[0])
    recs = run_order(order, {"A": [0.17], "F": [0.17]})
    recs[1]["bo_total_after"] = 9
    assert any("zero growth" in f for f in ab.judge_timing(recs, RUN, LAYOUTS, order, 4, anchor=None)[0])
    recs = run_order(order, {"A": [0.17], "F": [0.17]})
    del recs[1]["exec_ms_per_program_median"]["prog_002"]
    assert any("exec medians cover" in f for f in ab.judge_timing(recs, RUN, LAYOUTS, order, 4, anchor=None)[0])


def test_identity_needs_every_arm_equal_to_the_control():
    def rec(tokens):
        return {"tokens": tokens, "kernel_stop": [], "exch_new": 0}
    good = [{"prompt": f"p{i}", "A": rec(list(range(3))), "F": rec(list(range(3)))} for i in range(2)]
    bad, summary = ab.judge_identity(good, ["A", "F"], anchor=None, prompts=2, tokens=3)
    assert bad == [] and summary["prompts_with_ids_equal_to_control"] == {"F": 2}
    good[1]["F"] = rec([0, 1, 9])
    bad, _ = ab.judge_identity(good, ["A", "F"], anchor=None, prompts=2, tokens=3)
    assert any("differ" in b for b in bad)


def test_anchor_ids_gate_the_control():
    def rec(tokens):
        return {"tokens": tokens, "kernel_stop": [], "exch_new": 0}
    ident = [{"prompt": "p001", "A": rec([5, 6, 7]), "F": rec([5, 6, 7])}]
    assert ab.judge_identity(ident, ["A", "F"], anchor=[5, 6], prompts=1, tokens=3)[0] == []
    bad, _ = ab.judge_identity(ident, ["A", "F"], anchor=[5, 9], prompts=1, tokens=3)
    assert any("anchor" in b for b in bad)


def test_default_order_alternates_the_layouts_between_rounds():
    assert ab.default_order(["S", "F"], 3) == ["A", "S", "F", "A", "F", "S", "A", "S", "F"]


def test_build_sets_reads_the_control_files_for_unfolded_programs(tmp_path):
    control = {"programs": [{}, {}, {}, {}]}
    layouts = {"F": {"programs": [{}, {}, {}], "fold_mapping": [[0], [1, 2], [3]]}}
    sets = ab.build_sets(control, layouts, tmp_path / "anec", tmp_path / "ports", tmp_path / "fold")
    files = sets["F"]["files"]
    assert files[0] == sets["A"]["files"][0] and files[2] == sets["A"]["files"][3]
    assert files[1][0] == tmp_path / "fold" / "F" / "prog01" / "program-0.anec" and files[1][2] == "prog_001_fold"
