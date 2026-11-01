# Whole-graph camera mode

`AURORA_CAMERA_ENGINE=ane-whole` selects one resident H13 program per model.
`try-camera-h13 --engine ane-whole` selects the same path. The existing
`auto` mode prefers a linked candidate when present. Missing, corrupt,
unqualified, slow or failing candidates route that model to its original
ONNX CPU reference through the shared fallback. A failed linked candidate
does not select the island engine. Blur and framing qualify independently.

The program is loaded once when a model opens. Every inference packs the
normalized input, refills that handle, executes once and reads the outputs.
Letterboxing, normalization and effect/decode/NMS processing stay outside
the program. No graph partitions or intermediate CPU operations run inside
this mode. The existing compiler reader and `PreparedProgram` own all device
buffers and cleanup.

## Current availability

YuNet has a source-built **one-program, 113-task** package in
[`../packages/`](../packages/README.md), with package-MIL parity passing on all
30 audited portraits. It uses convolution/relu pooling, native nearest resize,
exclusive producer placement, a phase-layout stem and compact head readback.
The CPU boundary performs image layout and score/head decode. No NN
intermediate is evaluated on the host. Those boundary costs are included in
device frame timing.

There is no ANE device timing or speedup result yet. The package contains a
CPU simulation receipt and cannot activate production inference until the
lead's device qualification succeeds. Selfie still lacks a linked package;
blur uses CPU. The original exports still refuse direct linked compilation
at padding or multiple results; see the compiler's
`docs/ane/camera-linked.md` for the operation census and remaining selfie work.

## Build and host parity

Use the audited models fetched by `fetch-models.py` and the exact portraits
selected by `h13/images.py`. Models, images and generated programs remain
outside Git. From the compiler checkout:

```sh
python3 tools/camera_linked.py "$CAMERA_MODELS/selfie" --output "$PACKAGES/selfie"
python3 tools/camera_linked.py "$CAMERA_MODELS/yunet" --output "$PACKAGES/yunet"
```

Those direct-export commands preserve their refusals. For the complete YuNet
lowering, use:

```sh
python3 tools/yunet_linked.py "$CAMERA_MODELS/yunet" --compiler build/mil-hwxc --output "$PACKAGES/yunet"
```

Its successful `package/` directory contains
`camera-linked.json`, the compiler manifest, ANEC, MIL and weights. Install
that directory as `models/MODEL/compiled/H13/linked`, retaining its hashes.

Run host parity from this camera-effects directory, using the compatible
compiler checkout (with its `tools/` and `research/` directories):

```sh
python linked/qualify.py --simulate --model yunet --models "$CAMERA_MODELS" --images "$PORTRAITS" --compiler-root "$COMPILER" --output "$NEW_RESULTS"
```

This validates the ANEC structure. Lowered YuNet simulation then uses decoded
DMA addresses/strides for physical input, scratch and output movement, and
package-MIL arithmetic with fp16 surface boundaries. It checks task output
slices and zero-copy concat semantics on the real linked stream. Other
packages use numerical MIL simulation. ONNX Runtime supplies the separate
unchanged CPU reference. Unsupported simulation operators are refused.
Internal hardware accumulator/LUT microarchitecture is
not emulated; host parity cannot authorize hardware or production loading.

## Lead's one-command device qualification

Run **inside a controller-approved execute request for `device=m1pro`**.
No direct SSH or script invocation substitutes for policy approval. Prepare
an external identity JSON from that approved request, containing:

```json
{
  "device": "m1pro",
  "request": "h-REQUEST_NUMBER",
  "library_sha256": "APPROVED_BINDING_SHA256",
  "reader_sha256": "APPROVED_H13_RUN_LINUX_PY_SHA256",
  "compiler_files": {
    "tools/h13_run_linux.py": "APPROVED_H13_RUN_LINUX_PY_SHA256",
    "research/inspect_anec.py": "APPROVED_INSPECTOR_SHA256",
    "research/h13_td.py": "APPROVED_DECODER_SHA256"
  },
  "portraits_sha256": "APPROVED_IMAGES_JSON_SHA256",
  "packages": {
    "selfie": "APPROVED_SELFIE_CAMERA_LINKED_JSON_SHA256",
    "yunet": "APPROVED_YUNET_CAMERA_LINKED_JSON_SHA256"
  }
}
```

Keep concrete controller records, device pins and paths outside public source.
The JSON binds source identities; the controller supplies policy approval.
The qualification script makes no controller request itself.

```sh
timeout --kill-after=20 580 python linked/qualify.py --qualify --model yunet --device m1pro --approval "$APPROVED_IDENTITIES" --models "$CAMERA_MODELS" --images "$PORTRAITS" --compiler-root "$COMPILER" --library "$LIBANE_BINDING" --frames 300 --output "$NEW_RESULTS"
```

Omit `--model yunet` to attempt both models independently. A missing selfie
package leaves its failed receipt and does not retire a passing YuNet package.

Thirty fresh program handles each execute a distinct portrait once, without
warmup. Then one resident handle executes 300 frames, alternating ANE/CPU
timing order. Timing includes input packing/conversion and dense output
readback, excluding image preprocessing, model LOAD and parity comparisons.
LOAD setup time is reported separately. The script reads `wedged` before,
during and after execution. It changes no system settings.

Expected successful device results (still unmeasured):

| Field | Required result |
| --- | --- |
| `status`, `mode` | `pass`, `libane-device` |
| `images`, `cold_passes`, records | 30, 30, all 30 pass |
| Selfie per portrait | IoU >= 0.98; mask disagreement and mean absolute error <= 0.005 |
| YuNet every output tensor | absolute error <= 0.03 + 0.02 * abs(reference), the t-4019 tensor gate including box and landmark heads |
| `timing.loads`, `timing.executes` | 1, 300 |
| `timing.ane_p50_ms` | below the measured `timing.cpu_p50_ms` |
| p50/p99 | finite measured milliseconds for both engines |
| `wedged_before`, `wedged_after` | both `0` |

Any failure returns nonzero and leaves a failed per-model receipt. Failed
timings are retained even when the speed gate fails. Existing device receipts
are renamed before a new run; atomic failed progress receipts prevent an
interrupted qualification from retaining stale authorization. A model's
passing receipt is stored beside its package; a failure in the other model
does not retire it. Host simulation never writes a device receipt.

Preserve the entire new results directory and prior receipts for the lead.
Rollback is selecting `--engine cpu`, or removing the candidate linked
directory after preserving it. No system installation or device state change
is required by this feature.

## Bisect a device/host numerical disagreement

The original YuNet DRAM candidate failed device parity despite passing host
simulation. Keep it unqualified. The compiler now provides
`tools/camera_exact_prefixes.py` (original task stream) and
`tools/camera_prefixes.py` (dependency-closed source prefixes and isolated
convolutions). Build their probes from the same extracted package.

Prepare an oracle without device access, using the exact audited portraits:

```sh
python linked/bisect_camera.py --prepare --probes "$PROBES" --source "$PACKAGE" \
  --images "$PORTRAITS" --compiler-root "$COMPILER" --output "$NEW_ORACLE"
```

The oracle contains derived intermediate tensors and stays with evaluation
artifacts; do not commit it. Review an external approval file binding
`device=m1pro`, the concrete controller `request` ID, `library_sha256`,
`compiler_files` (the three runtime files from qualification above),
`script_sha256` (this bisection script), `index_sha256` and
`oracle_metadata_sha256`. Hash the actual script/index/oracle metadata files.
Run only within that approved controller execute request:

```sh
timeout --kill-after=20 580 python linked/bisect_camera.py --run --device m1pro \
  --approval "$APPROVAL" --probes "$PROBES" --source "$PACKAGE" \
  --oracle "$ORACLE" --compiler-root "$COMPILER" --library "$LIBANE_BINDING" \
  --cold-runs 3 --output "$NEW_RESULTS"
```

Start with `--layer 0 --layer 1 --layer 2 --layer 9 --layer 10`, then narrow the
first divergent interval. `--mode exact-prefix`, `--mode prefix` and
`--mode isolated` select the ladder type. Every cold run opens a new handle;
readbacks, worst coordinates, relative L2 and error fractions are retained.
The diagnostic threshold is 0.03 + 0.02*abs(reference) on intermediates, and
is not a replacement for the unchanged final model gate.

Expected completion is `status=diagnostic-complete` with wedged zero before
and after. Inspect all numerical records: this status alone is no parity
claim. A submission/readback error records `status=device-error` and the
active original task interval, then aborts. Preserve the controller and
kernel logs and do not reuse results after a hang. The diagnostic never
writes `qualification.json`.

The resident runtime can also fill prepared input buffers directly and read
owned output buffers in place. YuNet writes phase axes into that input view,
clears padded channels on each frame, and converts only the live sampled
heads to float32. Returned camera outputs remain independent of later frames.
This removes host copies; its end-to-end performance still needs measurement.

The model qualification command also aborts immediately after a LOAD or
execution failure, even if the recovery driver currently reports `wedged=0`.
It retains the active portrait, cold/timing phase and program task count, and
checks wedged state between timed frames. It does not issue another LOAD after
a failed completion. Numeric parity failures still retain all 30 portrait
comparisons when execution remains healthy.
