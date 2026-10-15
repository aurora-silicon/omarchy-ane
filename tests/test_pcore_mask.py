#!/usr/bin/env python3
"""Host checks for tools/pcore_mask.py. No device. Fake sysfs trees stand in for the chips; the pin tests change the
affinity of the test thread and restore it."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pcore_mask
from pcore_mask import OPT_OUT, p_core_cpus, pin_self, pin_tasks

NEEDS_TWO_CPUS = pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two allowed CPUs")


def fake_sysfs(tmp_path, capacities):
    for cpu, capacity in enumerate(capacities):
        (tmp_path / f"cpu{cpu}").mkdir()
        (tmp_path / f"cpu{cpu}" / "cpu_capacity").write_text(f"{capacity}\n")
    (tmp_path / "cpufreq").mkdir()            # a non-cpu entry the scan must ignore
    return tmp_path


@pytest.mark.parametrize("capacities, expected", [
    ([561] * 4 + [1024] * 8, [4, 5, 6, 7, 8, 9, 10, 11]),            # M2 Max
    ([485] * 2 + [1024] * 8, [2, 3, 4, 5, 6, 7, 8, 9]),              # M1 Max
    ([493] * 4 + [1024] * 4, [4, 5, 6, 7]),                          # M1
    ([1024] * 4 + [561] * 2, [0, 1, 2, 3]),                          # efficiency cores numbered last
    ([1024] * 8, None),                                              # no efficiency cores: leave alone
])
def test_p_cores_come_from_capacity(tmp_path, capacities, expected):
    assert p_core_cpus(fake_sysfs(tmp_path, capacities)) == expected


def test_unreadable_or_missing_capacity_is_not_guessed(tmp_path):
    root = fake_sysfs(tmp_path, [561, 1024, 1024])
    (root / "cpu1" / "cpu_capacity").unlink()
    assert p_core_cpus(root) is None
    assert p_core_cpus(tmp_path / "does-not-exist") is None


def test_command_line_prints_a_taskset_list(tmp_path, capsys):
    assert pcore_mask.main(fake_sysfs(tmp_path, [561] * 4 + [1024] * 8), {}) == 0
    assert capsys.readouterr().out == "4-11\n"
    assert pcore_mask.main(fake_sysfs(tmp_path / "split", [1024, 561, 1024, 1024]) if (tmp_path / "split").mkdir() is None else tmp_path, {}) == 0
    assert capsys.readouterr().out == "0,2-3\n"
    assert pcore_mask.main(tmp_path, {OPT_OUT: "1"}) == 0
    assert capsys.readouterr().out == ""
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "pcore_mask.py")], capture_output=True, text=True,
                         env={**os.environ, OPT_OUT: "1"}, check=True)
    assert run.stdout == ""


@NEEDS_TWO_CPUS
def test_pin_self_sets_the_calling_thread_and_is_inherited(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    root = fake_sysfs(tmp_path, [561] + [1024] * (max(allowed) + 1))   # cpu 0 is an E core in the fake chip
    p_cores = [cpu for cpu in allowed if cpu != 0]
    if not p_cores or set(p_cores) == set(allowed):
        pytest.skip("the allowed set does not contain cpu 0 plus another CPU")
    before = os.sched_getaffinity(0)
    seen = {}
    try:
        assert pin_self(root, env={}) == p_cores
        assert sorted(os.sched_getaffinity(0)) == p_cores
        worker = threading.Thread(target=lambda: seen.update(mask=sorted(os.sched_getaffinity(0))))
        worker.start()
        worker.join()
        assert seen["mask"] == p_cores                                  # a thread started after the pin inherits it
    finally:
        os.sched_setaffinity(0, before)


@NEEDS_TWO_CPUS
def test_opt_out_and_nothing_to_pin_change_nothing(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    root = fake_sysfs(tmp_path, [561] + [1024] * (max(allowed) + 1))
    before = os.sched_getaffinity(0)
    assert pin_self(root, env={OPT_OUT: "1"}) is None
    assert os.sched_getaffinity(0) == before
    (tmp_path / "flat").mkdir()
    assert pin_self(fake_sysfs(tmp_path / "flat", [1024] * (max(allowed) + 1)), env={}) is None
    assert os.sched_getaffinity(0) == before
    assert pcore_mask.enabled({}) and not pcore_mask.enabled({OPT_OUT: "1"}) and pcore_mask.enabled({OPT_OUT: "0"})


@NEEDS_TWO_CPUS
def test_an_existing_narrower_mask_is_respected(tmp_path):
    """A caller already limited to cpus that are all P cores keeps its mask: the target equals the allowed set, so nothing is set."""
    allowed = sorted(os.sched_getaffinity(0))
    root = fake_sysfs(tmp_path, [561] + [1024] * (max(allowed) + 1))
    narrow = {c for c in allowed if c != 0}
    if not narrow or 0 not in allowed and len(allowed) < 2:
        pytest.skip("needs cpu 0 plus another allowed CPU")
    before = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, narrow)
        assert pin_self(root, env={}) is None
        assert os.sched_getaffinity(0) == narrow
    finally:
        os.sched_setaffinity(0, before)


@NEEDS_TWO_CPUS
def test_pin_tasks_sets_every_task_of_a_process_and_survives_exits(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    if 0 not in allowed or len(allowed) < 2:
        pytest.skip("needs cpu 0 plus another allowed CPU")
    root = fake_sysfs(tmp_path, [561] + [1024] * (max(allowed) + 1))
    p_cores = {c for c in allowed if c != 0}
    stop = threading.Event()
    threads = [threading.Thread(target=stop.wait) for _ in range(3)]
    for thread in threads:
        thread.start()
    try:
        count = pin_tasks(os.getpid(), root, env={})
        assert count >= 4                                              # the main thread and the three workers
        for name in os.listdir(f"/proc/{os.getpid()}/task"):
            assert set(os.sched_getaffinity(int(name))) == p_cores, name
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        for name in os.listdir(f"/proc/{os.getpid()}/task"):
            os.sched_setaffinity(int(name), set(allowed))
