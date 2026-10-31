# SPDX-License-Identifier: MIT
from types import SimpleNamespace
import pytest
from aurora_camera_ane.native import H13Frame


def owner(directory, failed):
    native = H13Frame.__new__(H13Frame)
    native.directory, native.failed = directory, failed
    native.records = {"test": object()}
    native.runner = SimpleNamespace(_close_all=lambda programs, primary: None)
    return native


def test_success_removes_private_pixel_scratch(tmp_path):
    directory = tmp_path/"native"
    directory.mkdir(mode=0o700)
    (directory/"x.fp16").write_bytes(b"private frame")
    native = owner(directory, False)
    native.close()
    assert not directory.exists()
    assert not native.records


def test_failure_preserves_private_evidence(tmp_path):
    directory = tmp_path/"native"
    directory.mkdir(mode=0o700)
    (directory/"failure.txt").write_text("native failure")
    native = owner(directory, True)
    native.close()
    assert (directory/"failure.txt").read_text() == "native failure"


def test_cleanup_failure_does_not_replace_inference_failure(tmp_path):
    directory = tmp_path/"native"
    directory.mkdir(mode=0o700)
    (directory/"failure.txt").write_text("inference failure")
    native = owner(directory, True)
    def fail(programs, primary):
        raise OSError("cleanup failure")
    native.runner._close_all = fail
    with pytest.raises(OSError):
        native.close()
    assert (directory/"failure.txt").read_text() == "inference failure"
    assert "cleanup failure" in (directory/"cleanup-failure.txt").read_text()
