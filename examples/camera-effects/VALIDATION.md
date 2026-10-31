# Camera effects validation

The released models currently select CPU through the shared runtime. No native
ANE execution, performance or real-camera quality result is claimed.

## Export identities

The official source model hashes and licences are in `LICENCE-AUDIT.md`. The
exporter reads those exact source files, converts the same weights to ONNX/MIL,
and records every derived artifact hash in `model.json`. It does not call an
Apple compiler. MIL identifiers are normalized to the compiler's identifier
syntax. Hundreds of isolated operator probes are stored in `probes.zip` and
read individually by the compiler probe tool, without extracting archive paths.

The selfie graph has convolutions/depthwise convolutions, explicit padding,
local/global pooling, hard-swish arithmetic, sigmoid, bilinear upsampling and a
final transpose convolution. YuNet has convolutions/depthwise convolutions,
max-pooling, nearest upsampling, adds, sigmoids, reshapes and transposes. Its
320×320 interface is fixed at export time. A successful MIL conversion does not
prove that any target can lower or run the full graph.

## Host tests

C tests cover the worker's owned strided planes, padding sentinels, timestamp and
sequence preservation, partial-output failure, bounded saturation and per-stream
isolation. The filter's existing CPU/trigger/config suites also remain enabled.
The optional host fixture creates an isolated PipeWire server, synthetic NV12
source/sink and test-only delayed plugin. It exercises Props round-trips,
independent outputs, rejected control transactions and recovery after a long
inference stall following initial output. It touches no camera hardware.

Python tests exercise exact off bypass, guided edges/constant masks, shared
runtime failure retirement, nonfinite native output fallback, crop smoothing,
face loss, independent framing state, NV12 chroma/colorimetry and model identity
refusal. The C bridge harness dynamically loads the plugin with `RTLD_LOCAL` from a
plain C executable, then tests real model processing and in-place off bypass.
This catches CPython extension-symbol visibility that direct linking would hide. Independent comparisons use MediaPipe's own CPU graph and OpenCV's
original YuNet graph, rather than another copy of the exporter.

Review caught and fixed output-buffer exhaustion when an inference result was
pending, and stale effect colorimetry after same-size format renegotiation.
Pending buffers now return to the free pool; matrix/range changes rebuild the
worker. Config reload is transactional and updates advertised board capability.

## Limits and lead work

Host measurements exclude camera DMA mapping, denoise, C packing/queue overhead,
portal/application latency and ANE work. They are not target power/latency
numbers. COCO's nearby-person subset still includes full-body/multiple-person
scenes outside the selfie model's main training domain. The default 50-image protocol
is a small open-label stress sample, not a temporal or demographic study.

The dormant native path reuses the existing H13 graph owner and requires an
identified whole-model package plus independent qualification. Native support
on H14/H16 needs their existing owners, compiler coverage and lead validation.
H17 stays CPU-only until ownership exists. Follow `LEAD-TEST-PLAN.md`; never
turn a compilation result into a native performance or qualification claim.
