# Lead: install, qualify, try

No device run was done by this worker. The bundles are unqualified until the
commands below pass on the M1 Pro. Use the controller's device identity and
policy approval for device execution. Do not copy a CPU simulation receipt into
`qualification.json`; the runtime rejects simulation receipts.

M1 Pro, Fedora 44 aarch64, from `examples/camera-effects` in this branch:

```sh
bash h13/setup.sh /path/to/mil-hwx-compiler/build/mil-hwxc
```

The recorded offline bundles use PR #5 revision `72e1d99c6f51eaf97ad61280936d42dbb0b6a1ae`. Use the [reader compatibility PR #9](https://github.com/aurora-silicon/mil-hwx-compiler/pull/9), branch `fu-camera-h13-reader` on PR #5 when running them: the original reader misclassifies channel-changing conv as elementwise. No new compiler op encoder is required by this branch. The build
command is `make -j6 build/mil-hwxc` on its supported build host. The worker's
macOS-built compiler emits platform-independent H13 packages; it is not a Linux
runtime binary. Build libane and its Python binding natively on Fedora:

```sh
make -C ../../libane -j6 libane && make -C ../../bindings/python/dylib libane_python
```

The existing graph owner must be available at `/usr/lib/mil-hwx-compiler`, or
pass its checkout via `--compiler-root` when qualifying and previewing. The C
consumer accepts that path via `AURORA_CAMERA_COMPILER_ROOT`; its default is the
existing system installation. libane remains the sole device owner.
No compiler internals or device ownership rules change.

Download evaluation portraits, then run CPU simulation on either build host:

```sh
.venv-h13/bin/python h13/images.py /tmp/camera-h13-portraits && for model in selfie yunet; do .venv-h13/bin/python h13/verify.py "models/$model" --package "models/$model/compiled/H13/hybrid" --images /tmp/camera-h13-portraits --simulate --output "/tmp/$model-simulation.json" || exit; done
```

M1 Pro only, explicit H13 qualification, one command:

```sh
for model in selfie yunet; do .venv-h13/bin/python h13/verify.py "models/$model" --package "models/$model/compiled/H13/hybrid" --images /tmp/camera-h13-portraits --qualify --library ../../bindings/python/dylib/libane_python.so --compiler-root /path/to/mil-hwx-compiler --output "/tmp/$model-h13.json" || exit; done
```

Qualification runs every candidate island once per portrait against its original
ONNX operation. A failed comparison or candidate exception refuses only that
island; later islands still see original-graph inputs and all 30 portraits.
The schema-2 receipt records `islands` (qualified/refused plus all 30 samples,
maximum absolute errors and exception text), `qualified_islands`,
`refused_islands`, and the plan hash. The final pass runs the selected mix:
qualified islands on the ANE, refused islands through the original ONNX operation.
It must pass **mask IoU >=0.98 on every image**, disagreement and MAE <=0.005;
YuNet retains its output tolerance. The engine prints, for example,
`engine: ane-hybrid (45 of 58 islands on the ANE)`.

If accumulated numerical error fails the final output gate, qualification can
conservatively move up to six additional islands to CPU in reverse graph order,
checking all 30 images after each change. Receipts distinguish these output-gate
fallbacks from island parity failures and retain every attempt. The thresholds
stay fixed. Execution errors and an exhausted fallback budget leave a failed
receipt. Use `--max-output-fallbacks 0` to disable this step for diagnosis.

The tool checks ABI 1/H13 before preparing programs; its owner validates program
surfaces and libane allocation sizes. Production prepares only qualified islands.
Legacy, incomplete, stale and simulation receipts require requalification.
A failed final gate replaces any old qualification with a failed receipt while
preserving island errors. Preserve both JSON outputs and failing packages/logs
for the compiler lane. The inter-island deadline cannot interrupt an in-flight
driver submission.

CPU-only fault injection (never writes `qualification.json`):

```sh
.venv-h13/bin/python h13/verify.py models/selfie --package models/selfie/compiled/H13/hybrid --images /tmp/camera-h13-portraits --simulate --inject-failure original-046.onnx --output /tmp/selfie-fault-simulation.json
PYTHONPATH=. .venv-h13/bin/python -m pytest tests/test_qualification.py -q
```

Inside a controller-approved ten-minute slot for an explicit M1 Pro device,
set `LIB_SHA256`, `SELFIE_PLAN_SHA256` and `YUNET_PLAN_SHA256` from the
controller-approved artifacts, then rerun qualification, three still-image
previews, and timing with:

```sh
COMPILER=/path/to/mil-hwx-compiler PORTRAITS=/tmp/camera-h13-portraits bash h13/slot.sh
```

The script archives previous results, preserves logs/receipts, snapshots package
and library hashes and device counters, and bounds work to 580 seconds plus
20 seconds for forced teardown. It depends only on the checkout, installed
binding/reader, portraits and `timeout`; it changes no system settings. An
interrupted qualification cannot publish a new passing receipt. To time alone:

```sh
.venv-h13/bin/python h13/timing.py /tmp/camera-h13-portraits/000000000872.jpg --engine both --framing --compiler-root /path/to/mil-hwx-compiler --output /tmp/camera-timing.json
```

Timing reports per-frame median/mean and individual samples at **256x256** and
**full preview** (aspect preserved, capped at 1280x720). It includes NV12/model
conversion, inference, guided mask, blur/composite and optional YuNet framing;
initialization, JPEG decoding, warm-up and file writes are excluded. Each engine
gets one warm-up and ten measured frames by default. A hybrid timing run fails
if either model falls back to whole-model CPU. Initialization and per-model
inference times are also retained in JSON; no hardware timings are claimed by
CPU simulation.

Then run the [TRY.md](TRY.md) preview command. `AURORA_CAMERA_ENGINE=cpu` forces
the existing C camera backend to CPU. Auto falls back for missing libane,
unqualified bundles, a different generation, or an execution/output failure.
Every startup/fallback prints the engine. No Vulkan provider is implemented in
this consumer, so its available fallback is ONNX Runtime CPU.

M2: use the same preview with `--engine auto`; expect CPU. H13 receipts must
never qualify an H14 device. M4 Pro/Neo likewise retain generation checks.

Off is the TRY.md `off` command. To retire qualification as well:

```sh
for model in selfie yunet; do if [ -f "models/$model/compiled/H13/hybrid/qualification.json" ]; then mv "models/$model/compiled/H13/hybrid/qualification.json" "models/$model/compiled/H13/hybrid/qualification.retired.json"; fi; done
```

The next auto startup uses CPU. Keep the retired receipts and compile logs;
no system configuration, boot state, camera default, audio route or protection
daemon needs rollback. No audio playback or microphone recording is involved.
