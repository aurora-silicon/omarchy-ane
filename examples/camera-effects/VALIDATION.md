# Camera effects validation

The released models currently select CPU through the shared runtime. No native
ANE execution or performance is qualified. Mask quality below uses an open
still-image dataset; live-camera CPU figures are attributed to the lead.

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

## Recorded CPU and quality receipts

The [COCO receipt](receipts/coco50-quality.json) records 50 CC BY 2.0 images,
selected in ascending ID order with summed person area at least 10% and no crowd
person labels. The production BT.601 limited NV12 round-trip, model preprocessing
and guided upsampler are used; at most one final odd row/column is trimmed. At a
0.5 threshold, mean IoU is **0.6814**, global IoU **0.7156**, mean Dice **0.7688**
and boundary F1 **0.3913** at two-pixel (5x5 square) tolerance. The model bytes and mask path
are unchanged by the later reduced-size blur patch. This selected still-image
sample does not establish temporal, demographic or real-camera segmentation quality.

The [independent source comparison](receipts/source-reference.json) passed on
three synthetic inputs and five COCO images. Maximum absolute selfie output
error versus MediaPipe's own CPU graph was **6.99e-5**. YuNet's fp16-rounded
weights differed from the original OpenCV fp32 graph by at most **0.00934** in
raw outputs. These are conversion checks, not face-detection accuracy scores.
Fresh ONNX, MIL and weight files were byte-identical to both released models.

The [full-size-blur host baseline](receipts/host-full-size-blur-baseline.json)
uses a shared Intel i9-13900K Linux x86_64 build host, two ORT threads and one
OpenCV thread, three warm-up frames and 30 timed frames per case. CPU is process
time per frame; wall is median time, in milliseconds. These measurements precede
the reduced-size blur optimisation; current-source host timing is pending.

| Effect | 720p CPU | 720p wall | 1080p CPU | 1080p wall |
| --- | ---: | ---: | ---: | ---: |
| Off | 0.0012 | 0.0016 | 0.0011 | 0.0016 |
| Blur | 83.42 | 57.43 | 184.25 | 146.23 |
| Replace | 53.46 | 27.97 | 99.16 | 64.78 |
| Framing | 19.52 | 9.69 | 21.84 | 11.13 |
| Blur + framing | 119.86 | 63.08 | 208.00 | 160.06 |

The lead's [M2 live-camera receipt](https://github.com/aurora-silicon/omarchy-ane/pull/5)
records ORT CPU processing of real apple-isp 1080p frames. Reduced-size blur
changed backend median wall time from **104.9 to 42.0 ms**, process CPU from
**130 to 49 ms/frame**, and plane blur from 48.1 to 1.1 ms. Blur plus framing
changed from 121.9 to 66.1 ms median. These device measurements use a different
host/input/protocol from the x86 baseline and are recorded separately.

The build host passed ten Python tests, the actual RTLD_LOCAL bridge and four C
suites on the pre-lead snapshot. Its old PipeWire fixture failed to connect on
1.6; the lead's [fixture/newest-input fix](https://github.com/aurora-silicon/camera-denoise/pull/3)
passed four C suites and delivered 20 complete frames through a synthetic stall
on M2. The reduced-blur lead receipt reports ten Python tests. The merged
current-source build-host run, including consistent test core/socket naming and
per-user capability configuration, is still pending.

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
