# SPDX-License-Identifier: MIT
"""Minimal model backend. The filter owns policy, frame scheduling and chroma.

This module only touches the supplied luma plane. Calls belong on the filter's
worker thread: neither Python nor ONNX Runtime is an RT processing callback.
"""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT.parents[1]
sys.path.insert(0, str(RUNTIME_ROOT/"bindings/python/python"))
from ane import Fallback

MODEL = ROOT/"models/tiny-luma-v1"
LIBANE = (RUNTIME_ROOT/"bindings/python/dylib/libane_python.so"
          if (RUNTIME_ROOT/"bindings/python/dylib/libane_python.so").is_file()
          else Path("/usr/lib/libane_python.so"))
COMPILER = Path("/usr/lib/mil-hwx-compiler")


def verify(root):
    root = Path(root).resolve()
    metadata = json.loads((root/"model.json").read_text())
    if (metadata.get("model"), metadata.get("tile"), metadata.get("halo")) != (
            "tiny-luma-unet-v1", 64, 12):
        raise ValueError("unsupported model contract")
    required = {"weights.npz", "weights.bin", "model.onnx", "model.mil", "training.json"}
    if not required.issubset(metadata["artifacts"]):
        raise ValueError("incomplete model identities")
    for name, digest in metadata["artifacts"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"model identity mismatch: {name}")
    return metadata


class FrameCPU:
    """The same fp16-rounded graph, with fp32 convolution accumulation."""
    def __init__(self, root):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(root/"model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def tensor(self, value):
        value = np.ascontiguousarray(value, dtype=np.float16).astype(np.float32)
        result, = self.session.run(["denoised"], {"x": value})
        result = result.astype(np.float16)
        if result.shape != value.shape or not np.isfinite(result).all():
            raise ValueError("invalid model output")
        return result

    def __call__(self, frame):
        height, width = frame.shape
        padded = np.pad(frame, ((12, 12+height % 2), (12, 12+width % 2)), mode="edge")
        return self.tensor(padded[None, None])[0, 0, 12:12+height, 12:12+width]


class Backend:
    """init(width,height,format), process(in,prev_state,strength,out), fini().

    GRAY8 is an 8-bit full-range luma plane, including a luma plane borrowed
    from a caller-owned NV12/I420 frame. Other planes are the caller's property.
    prev_state is reserved and untouched: this first model is spatial.
    """
    def __init__(self, width, height, format="GRAY8", *, model_root=MODEL,
                 library=LIBANE, native_factory=None):
        if (type(width) is not int or type(height) is not int or
                not 1 <= width <= 4096 or not 1 <= height <= 2160 or format != "GRAY8"):
            raise ValueError("expected a GRAY8 plane up to 4096x2160")
        self.width, self.height = width, height
        self.root = Path(model_root).resolve()
        self.metadata = verify(self.root)
        self.cpu = FrameCPU(self.root)
        self._native = None
        self._closed = False
        self._lock = threading.Lock()

        def load_native():
            from .native import H13Frame
            self._native = (native_factory() if native_factory else
                            H13Frame(self.root, COMPILER, library))
            return self._native
        self.runtime = Fallback(self.cpu, library=library, native_factory=load_native)

    def _plane(self, value, writable=False):
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or
                value.shape != (self.height, self.width) or value.strides[1] != 1 or
                value.strides[0] < self.width or (writable and not value.flags.writeable)):
            raise ValueError("frame must be a row-strided uint8 luma plane of the initialized size")

    def process(self, in_frame, prev_state, strength, out_frame):
        if not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and in [0,1]")
        with self._lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            self._plane(in_frame)
            self._plane(out_frame, writable=True)
            if strength == 0:
                np.copyto(out_frame, in_frame)
                return None
            # Preserve originals across a partially completed native inference.
            original = in_frame.astype(np.float32) / 255
            denoised, call = self.runtime.run(original)
            if denoised.shape != original.shape or not np.isfinite(denoised).all():
                raise ValueError("backend returned an invalid frame")
            result = original + strength*(denoised.astype(np.float32)-original)
            # Commit only after a complete successful inference. In-place works.
            out_frame[:] = np.rint(np.clip(result, 0, 1)*255).astype(np.uint8)
            return call

    def process_bytes(self, source, strength):
        """Small copying bridge for C ABI callers. Rows arrive tightly packed."""
        original = np.frombuffer(source, np.uint8).reshape(self.height, self.width)
        output = np.empty_like(original)
        self.process(original, None, strength, output)
        return output.tobytes()

    def fini(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._native is not None:
                    close = getattr(self._native, "close", None)
                    if close is not None:
                        close()


def init(width, height, format="GRAY8"):
    return Backend(width, height, format)
