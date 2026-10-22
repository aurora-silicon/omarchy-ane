#!/usr/bin/env python3
"""Offline source-selection checks; stop before any compiler or DTB build."""
import os
from pathlib import Path
import subprocess
import tempfile

BUILDER = Path(__file__).resolve().with_name("asahi-dtbs")


def run(*args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, timeout=30, **kwargs)


with tempfile.TemporaryDirectory(prefix="asahi-dtbs-test-") as tmp:
    root = Path(tmp)
    source, out = root / "source", root / "out"
    env = {**os.environ, "GIT_AUTHOR_NAME": "Joshua Warren",
           "GIT_ALLOW_PROTOCOL": "file",
           "GIT_AUTHOR_EMAIL": "816217+joshuaswarren@users.noreply.github.com",
           "GIT_COMMITTER_NAME": "Joshua Warren",
           "GIT_COMMITTER_EMAIL": "816217+joshuaswarren@users.noreply.github.com",
           "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
           "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00"}

    def git(*args):
        return run("git", "-C", str(source), *args, env=env, check=True).stdout.strip()

    run("git", "-c", "init.defaultBranch=aurora-wip", "init", "-q", str(source), check=True)
    version = source / "scripts/dtc/version.txt"
    version.parent.mkdir(parents=True)
    version.write_text("first\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "first")
    first = git("rev-parse", "HEAD")
    git("tag", "fixture-tag")
    version.write_text("second\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "second")
    second = git("rev-parse", "HEAD")

    # A deliberate failure at the first build command proves that source
    # selection finished, without faking a successful compile or requiring flex.
    tools = root / "bin"
    tools.mkdir()
    sentinel = tools / "flex"
    sentinel.write_text("#!/bin/sh\necho source-ready >&2\nexit 75\n")
    sentinel.chmod(0o755)
    env["PATH"] = f"{tools}:{os.environ['PATH']}"

    def prepare(ref, expected, dest=out, url=source):
        result = run(str(BUILDER), str(dest), str(url), ref, env=env)
        assert result.returncode == 75 and "source-ready" in result.stderr, result
        actual = run("git", "-C", str(dest / "linux"), "rev-parse", "HEAD", check=True).stdout.strip()
        assert actual == expected, (ref, expected, actual)
        assert (dest / "linux/scripts/dtc/version.txt").read_text() == (
            "first\n" if expected == first else "second\n")

    # The branch has advanced since first was resolved for the cache key.
    prepare("aurora-wip", second)
    prepare(first, first, root / "cold-sha")
    prepare(first, first)  # warm cache must not silently retain second
    env["GIT_ALLOW_PROTOCOL"] = ""
    prepare(first, first)  # exact cache hits need no network or fetch
    env["GIT_ALLOW_PROTOCOL"] = "file"
    prepare("aurora-wip", second)  # named refs must refresh, not reuse HEAD
    prepare("fixture-tag", first)  # existing tag callers remain supported

    # Never compile a modified cache or silently accept a different repository.
    cached = out / "linux/scripts/dtc/version.txt"
    cached.write_text("dirty\n")
    result = run(str(BUILDER), str(out), str(source), first, env=env)
    assert result.returncode not in (0, 75) and "source-ready" not in result.stderr, result
    assert cached.read_text() == "dirty\n", "dirty cache must not be discarded"
    cached.write_text("first\n")
    result = run(str(BUILDER), str(out), str(root / "wrong-source"), first, env=env)
    assert result.returncode not in (0, 75) and "source-ready" not in result.stderr, result

print("test_asahi_dtbs: ok (cold/warm SHA, cache hit, branch, tag, dirty and foreign cache)")
