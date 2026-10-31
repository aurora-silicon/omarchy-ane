# SPDX-License-Identifier: MIT
"""Delegate graph validation, layouts and scheduling to the compiler's runner.

No register encoding, ioctl, graph scheduler or device recovery is duplicated.
This adapter is dormant when the compiler refuses the model. H14/H16 are
explicitly refused until their existing graph owners support this contract.
"""
import ctypes as C
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np


class H13Frame:
    def __init__(self, root, compiler, library):
        self.root = Path(root)
        self.package = self.root/"compiled/H13/full-model"
        self.records = {}
        self.directory = None
        if not (self.package/"manifest.json").is_file():
            raise OSError("tiny-luma-unet-v1 has no compiled H13 package")
        metadata = json.loads((self.root/"model.json").read_text())
        if "H13" not in metadata.get("native_targets", []):
            raise OSError("full-model H13 compilation did not succeed")
        # Reject a stale or hand-copied native package before device access.
        for file in self.package.rglob("*"):
            if file.is_file() and file.suffix not in (".log",):
                relative = str(file.relative_to(self.root))
                if metadata["artifacts"].get(relative) != hashlib.sha256(file.read_bytes()).hexdigest():
                    raise ValueError(f"unidentified native package file: {relative}")
        lib = C.CDLL(str(library))
        lib.pyane_query_caps.argtypes = [C.c_int, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]
        lib.pyane_query_caps.restype = C.c_int
        abi, chip = C.c_uint32(), C.c_uint32()
        rc = lib.pyane_query_caps(0, C.byref(abi), C.byref(chip))
        if rc or (abi.value, chip.value) != (1, 13):
            raise OSError(f"H13 graph owner unavailable: rc={rc}, ABI={abi.value}, chip={chip.value}")
        sys.path.insert(0, str(Path(compiler)/"tools"))
        sys.path.insert(0, str(compiler))
        self.runner = importlib.import_module("h13_run_linux")
        self.adapter = self.runner.LibANEAdapter(library)
        # Reuse PreparedProgram and the unchanged run_package scheduler. Cache
        # program handles here rather than reloading the same tile per frame.
        self.adapter.execute = self.execute
        self.directory = Path(tempfile.mkdtemp(prefix="camera-ane-native-"))
        # The existing runner writes parity-failure outputs into its package
        # directory. Stage the verified open package in private scratch so no
        # camera data can ever be written beside installed public model data.
        try:
            private_package = self.directory/"package"
            shutil.copytree(self.package, private_package, ignore=shutil.ignore_patterns("*.log"))
            self.package = private_package
        except BaseException:
            shutil.rmtree(self.directory)
            raise
        self.failed = False

    def execute(self, anec, inputs, sizes):
        key = hashlib.sha256(anec).digest()
        if key not in self.records:
            self.records[key] = self.runner.PreparedProgram(self.adapter, anec, inputs, sizes)
        program = self.records[key]
        program.refill(inputs)
        return program.run_once()

    def __call__(self, frame):
        height, width = frame.shape
        padded = np.pad(frame, ((12, 12+(-height) % 40), (12, 12+(-width) % 40)), mode="edge")
        result = np.empty_like(frame, dtype=np.float16)
        deadline = time.monotonic()+2
        try:
            for y in range(0, height, 40):
                for x in range(0, width, 40):
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("native frame deadline exceeded between tiles")
                    input_path = self.directory/"x.fp16"
                    input_path.write_bytes(padded[y:y+64, x:x+64].astype("<f2").tobytes())
                    # The existing owner checks a CPU fp16 reference on each
                    # tile. This is a qualification path, not a speed claim.
                    _, actual, _ = self.runner.run_package(
                        self.package, self.root/"model.mil", self.root,
                        {"x": input_path}, {"denoised": self.directory/"denoised.fp16"},
                        self.adapter, deadline_seconds=remaining)
                    tile = np.frombuffer(actual["denoised"], dtype="<f2").reshape(64, 64)
                    hh, ww = min(40, height-y), min(40, width-x)
                    result[y:y+hh, x:x+ww] = tile[12:12+hh, 12:12+ww]
            if not np.isfinite(result).all():
                raise ValueError("nonfinite native output")
            return result
        except Exception as error:
            self.failed = True
            (self.directory/"failure.txt").write_text(str(error)+"\n")
            # Fallback retires this native callback. Keep handles and evidence
            # with this owner until fini; never reset/reload a failed device.
            raise

    def close(self):
        try:
            self.runner._close_all(list(self.records.values()), None)
        except Exception as error:
            self.failed = True
            (self.directory/"cleanup-failure.txt").write_text(str(error)+"\n")
            raise
        finally:
            self.records.clear()
            if not self.failed and self.directory is not None:
                shutil.rmtree(self.directory)
