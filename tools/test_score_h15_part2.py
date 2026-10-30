"""Tests for score_h15_part2.py: fixtures follow the exact files and lines the part 2 wrapper and ane_h15 write."""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
import score_h15_part2 as scorer  # noqa: E402

PAGES = 232
PREFIX = "[  412.100000] ane_h15 fixture-dev: "


def page_hash(i, salt=""):
    return hashlib.sha256(f"{i}{salt}".encode()).hexdigest()[:16]


class Part2Fixture:
    """One part 2 output directory, written the way the wrapper writes it."""

    def __init__(self, root, changed=(), messages=0, result="PASS core-started-and-stopped", pwgate="pwgate=opened word=0x1",
                 ended=True, insmod_rc=0, compare_count=None, after_pages=None, result_line=True):
        self.dir = Path(root)
        before = {i: page_hash(i) for i in range(PAGES)}
        after = {i: page_hash(i, "x" if i in changed else "") for i in range(PAGES)}
        if after_pages is not None:
            after = {i: after[i] for i in after_pages}
        (self.dir / "data-before.txt").write_text("".join(f"{i} {h}\n" for i, h in before.items()))
        (self.dir / "data-after.txt").write_text("".join(f"{i} {h}\n" for i, h in after.items()))
        shown = len(changed) if compare_count is None else compare_count
        (self.dir / "compare.txt").write_text(
            f"pages before {len(before)}, after {len(after)}, differing {shown}\n"
            f"differing page indices: {sorted(changed)}\n")
        verdict, _, reason = result.partition(" ")
        lines = []
        if pwgate:
            lines.append(f"{PREFIX}ane_h15 {pwgate}")
        for n in range(messages):
            lines.append(f"{PREFIX}ane_h15 msg{n} raw0=0x1{n} raw1=0x0")
        lines.append(f"{PREFIX}ane_h15 run started_seen=1 i2a_posted={int(bool(messages))} messages={messages} "
                     "final cpu_status=0x22 i2a_ctrl=0x20001 stopped=1")
        if result_line:
            tail = f" {pwgate}" if pwgate and verdict == "PASS" else ""
            lines.append(f"{PREFIX}ane_h15 RESULT stage=5 soc=t6031 verdict={verdict} reason={reason}{tail}")
        (self.dir / "ane_h15-dmesg.txt").write_text("\n".join(lines) + "\n")
        wrapper = ["h15 part2 wrapper start 2026-10-31T01:00:00Z boot_id=aaaaaaaa uname=7.2.11-lab2f725546-99b0",
                   f"INSMOD rc={insmod_rc}"]
        if ended:
            wrapper.append(f"h15 part2 wrapper end 2026-10-31T01:01:00Z insmod_rc={insmod_rc} second_hash_rc=0")
        (self.dir / "wrapper.log").write_text("\n".join(wrapper) + "\n")


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def score(self, **kw):
        Part2Fixture(self.tmp.name, **kw)
        return scorer.score(Path(self.tmp.name))

    def test_a_mailbox_message_wins_even_when_data_changed(self):
        r = self.score(messages=2, changed=(3, 9))
        self.assertEqual(r["branch"], "A")
        self.assertEqual(r["messages"], 2)

    def test_b_data_changed_without_a_message(self):
        r = self.score(changed=(7, 8, 200))
        self.assertEqual(r["branch"], "B")
        self.assertEqual(r["differing_pages"], [7, 8, 200])

    def test_c_data_unchanged_after_the_gate_was_opened_or_forced(self):
        for gate in ("pwgate=opened word=0x1", "pwgate=forced word=0x0"):
            with self.subTest(gate=gate):
                r = self.score(pwgate=gate)
                self.assertEqual(r["branch"], "C")
                self.assertEqual(r["pwgate"], gate.split()[0].split("=")[1])

    def test_c_when_the_gate_was_already_open_says_so(self):
        r = self.score(pwgate="pwgate=already-open")
        self.assertEqual(r["branch"], "C")
        self.assertEqual(r["pwgate"], "already-open")

    def test_d_gate_never_opened(self):
        r = self.score(result="FAIL pwgate-not-open", pwgate=None)
        self.assertEqual(r["branch"], "D")

    def test_e_job_that_never_ended_is_a_machine_down(self):
        r = self.score(ended=False)
        self.assertEqual(r["branch"], "E")
        self.assertIn("do not retry", r["next"].lower())

    def test_refused_insmod_is_a_result_not_a_branch(self):
        r = self.score(insmod_rc=1, result="REFUSED pwgate-window-refused", pwgate=None)
        self.assertEqual(r["branch"], "UNSCORED")
        self.assertIn("refused", r["why"].lower())

    def test_no_result_line_is_unscored_not_guessed(self):
        r = self.score(result_line=False)
        self.assertEqual(r["branch"], "UNSCORED")

    def test_compare_file_disagreeing_with_the_hashes_is_unscored(self):
        r = self.score(changed=(1,), compare_count=0)
        self.assertEqual(r["branch"], "UNSCORED")
        self.assertIn("compare", r["why"].lower())

    def test_incomplete_second_hash_is_unscored(self):
        r = self.score(after_pages=range(100))
        self.assertEqual(r["branch"], "UNSCORED")
        self.assertIn("pages", r["why"].lower())

    def test_core_not_stopped_is_flagged_for_reboot(self):
        r = self.score(result="FAIL core-not-stopped", changed=(5,))
        self.assertTrue(r["reboot_required"])

    def test_post_wait_line_is_reported_when_the_module_has_it(self):
        fx = Part2Fixture(self.tmp.name, pwgate="pwgate=opened word=0x1")
        log = fx.dir / "ane_h15-dmesg.txt"
        log.write_text(log.read_text() + f"{PREFIX}ane_h15 post-wait cpu_status=0x22 a2i_ctrl=0x20001 i2a_ctrl=0x20001\n")
        r = scorer.score(fx.dir)
        self.assertEqual(r["post_wait"], {"cpu_status": "0x22", "a2i_ctrl": "0x20001", "i2a_ctrl": "0x20001"})

    def test_every_branch_names_the_next_step_and_never_claims_compute(self):
        for kw in ({"messages": 1}, {"changed": (2,)}, {}, {"result": "FAIL pwgate-not-open", "pwgate": None}, {"ended": False}):
            with self.subTest(kw=kw):
                r = self.score(**kw)
                self.assertTrue(r["next"])
                self.assertNotIn("compute", r["next"].lower().replace("no compute", ""))

    def test_a_missing_second_hash_file_is_unscored(self):
        fx = Part2Fixture(self.tmp.name)
        (fx.dir / "data-after.txt").unlink()
        r = scorer.score(fx.dir)
        self.assertEqual(r["branch"], "UNSCORED")
        self.assertIn("unreadable", r["why"])

    def test_a_malformed_hash_row_is_unscored(self):
        fx = Part2Fixture(self.tmp.name)
        (fx.dir / "data-after.txt").write_text("0 aa bb\n")
        r = scorer.score(fx.dir)
        self.assertEqual(r["branch"], "UNSCORED")
        self.assertIn("unreadable", r["why"])

    def test_unchanged_data_with_no_pwgate_line_is_unscored_not_c(self):
        r = self.score(pwgate=None)
        self.assertEqual(r["branch"], "UNSCORED")
        self.assertIn("pwgate", r["why"])

    def test_missing_directory_inputs_are_unscored(self):
        r = scorer.score(Path(self.tmp.name))
        self.assertEqual(r["branch"], "UNSCORED")

    def test_the_last_result_line_wins_when_the_module_logged_two(self):
        fx = Part2Fixture(self.tmp.name, changed=(5,))
        log = fx.dir / "ane_h15-dmesg.txt"
        first = f"{PREFIX}ane_h15 RESULT stage=5 soc=t6031 verdict=FAIL reason=pwgate-not-open\n"
        log.write_text(first + log.read_text())
        self.assertEqual(scorer.score(fx.dir)["branch"], "B")

    def test_a_compare_mismatch_names_both_counts(self):
        r = self.score(changed=(1,), compare_count=0)
        self.assertIn("says 0", r["why"])
        self.assertIn("say 1", r["why"])

    def test_main_prints_branch_evidence_and_next_in_process(self):
        import contextlib
        import io
        Part2Fixture(self.tmp.name, changed=(4, 6))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = scorer.main(["score", self.tmp.name])
        lines = out.getvalue().splitlines()
        self.assertEqual(rc, 0)
        self.assertTrue(lines[0].startswith("branch B"))
        self.assertIn("differing_pages: [4, 6]", lines)
        self.assertTrue(lines[-1].startswith("next: "))

    def test_main_with_no_folder_prints_usage_and_exits_two(self):
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(scorer.main(["score"]), 2)
        self.assertIn("score one", err.getvalue().lower())

    def test_main_returns_two_for_an_unscored_folder(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scorer.main(["score", self.tmp.name]), 2)

    def test_cli_prints_the_branch_first_and_exits_zero_only_when_scored(self):
        Part2Fixture(self.tmp.name, changed=(4,))
        ok = subprocess.run([sys.executable, str(HERE / "score_h15_part2.py"), self.tmp.name], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue(ok.stdout.splitlines()[0].startswith("branch B"))
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        bad = subprocess.run([sys.executable, str(HERE / "score_h15_part2.py"), empty.name], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertTrue(bad.stdout.splitlines()[0].startswith("branch UNSCORED"))


if __name__ == "__main__":
    unittest.main()
