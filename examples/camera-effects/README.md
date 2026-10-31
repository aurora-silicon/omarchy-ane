# Camera effects backend

Opt-in background blur, user-image replacement and face-driven framing for the
NV12 PipeWire filter in `aurora-silicon/camera-denoise`. Each filter output owns
its controls, queues, model instances and framing history. No effect is enabled
by installing this backend.

The CPU path uses the official Apache-2.0 MediaPipe selfie segmentation weights
at 256×256 and MIT YuNet weights at 320×320. Both are exported to ONNX and typed
MIL from the same weights. YuNet weights are rounded to fp16 before either
export; MediaPipe's released weights already use fp16 storage. CPU activations
accumulate in fp32; MIL activations are fp16. These are the same learned model,
not a claim of bit-identical activation arithmetic.

`ane.Fallback` from the shared runtime owns selection and failure retirement.
The native adapter delegates residency, graph scheduling, layouts and device
cleanup to the existing compiler runner. It requires an identified, complete,
independently qualified H13 package before querying a device. Released models
advertise no qualified native targets. H14/H16 require their graph owners and
compiler coverage; H17 is not an owned Linux accelerator. See `VALIDATION.md`.

## Processing

Only model-sized RGB images are made from NV12, using negotiated BT.601/709 and
limited/full range. Unknown matrix/range use BT.601 limited, the existing camera
convention; other matrices are refused. Background composition operates on
native Y and interleaved UV planes, preserving source colorimetry and foreground
pixels. The mask uses fast guided upsampling: local coefficients are fit at
256×256, then evaluated against full-resolution luma. Blur is a separable
Gaussian evaluated at reduced size for wide kernels; replacement uses a
user-supplied image loaded once per stream.

YuNet sees aspect-preserving, padded BGR. The face union determines a crop with
head/shoulder margin. A 0.4-second exponential smoother, a dead zone and a
one-second face-loss hold prevent rapid jumps; loss then returns toward the
center/default zoom. NV12 crop coordinates are even and the crop is resized to
the negotiated output size. This is digital framing; no lens dewarping or Apple
algorithm/model is included. Board data supplies stock zoom bounds/defaults;
validated user configuration can explicitly enable framing and select its range
on any board. Effects remain off until selected separately per stream.

All Python, model creation and inference run on the filter's worker. Four-slot
owned queues bound memory and select the newest queued input under load. Only
complete output frames are published, with original timestamps; results older than 200 ms are
dropped. No old mask is placed on a new image. Effects-off bypasses the worker.
Backend initialization/inference failure passes the original frame and logs a
diagnostic. Blur/replacement are visual effects, not a privacy boundary.

## Build and installation

Build through the controller's buildbox slot. In a source checkout with the
shared runtime library and Python binding installed:

```sh
python3 -m venv /path/to/effects-venv
/path/to/effects-venv/bin/pip install -r examples/camera-effects/requirements-runtime.txt
# Install bindings/python/python with the same interpreter if not packaged.
make -C bindings/python/dylib libane_python
make -C examples/camera-effects PYTHON=/path/to/effects-venv/bin/python
make -C examples/camera-effects install PYTHON=/path/to/effects-venv/bin/python
```

Use the target distro's libdir when different, e.g. `LIBDIR=/usr/lib64`.
`libane_python.so` uses the shared runtime's `/usr/lib` location. Build with the
final interpreter path; the C bridge embeds that interpreter and its environment.
Install the shared `ane` Python package in that environment; do not copy another
runtime implementation into the plugin. The bridge promotes the already-linked CPython C API for extension imports; it
does not load a second interpreter. Keep libpython/extensions resident until
process exit, and stop/join stream workers before destroying their instances.

`camera_effects.h` is ABI v1. The filter carries the same header. It passes tightly
packed NV12 through the optional `libaurora-camera-effects.so` plugin. The default
library directory is `<libdir>/aurora-camera-denoise/`; there is no per-device
plugin path or hardware tuning environment variable.

## Reproduce exports and host checks

```sh
# Python 3.12, CPU PyTorch 2.7.1 + torchvision 0.22.1, then requirements-export.txt
python examples/camera-effects/export.py /fresh/export
python examples/camera-effects/probe_compiler.py /fresh/export /path/to/mil-hwxc --output /fresh/compiler-gaps.json
BUILD_OUT=/fresh/output bash examples/camera-effects/check-buildbox.sh
BUILD_OUT=/fresh/export-check bash examples/camera-effects/check-export-buildbox.sh
```

The exporter reads the committed, hash-identified source models; it does not
fetch a converted model or call Apple's compiler. Conversion dependencies are
build-only. Do not infer native support from successful MIL export or isolated
operator probes. See `LICENCE-AUDIT.md`, `VALIDATION.md` and `LEAD-TEST-PLAN.md`.
