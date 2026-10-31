# Lead qualification plan

No device tests are part of the host receipt. All target commands below need
controller approval for explicit device **m1pro**, using this task id and the
existing qualified kernel/driver/source lease. No boot change, kernel module
replacement or hardware recovery is included here.

## 1. Install only the candidate model in a scratch checkout

The CPU/filter session owns distro packaging and filter selection. Its final
vtable can wrap `camera_ane_backend_v1`. Do not install a second PipeWire node
or replace that session's CPU denoiser. The C bridge must be built on the
buildbox for the target ABI using the distro's matching Python headers/libs;
the source-level Python check below needs no target build.

After the lead has installed t-3650 libane from the distro package:

```sh
git clone --branch feat/camera-denoise-backend https://github.com/aurora-silicon/omarchy-ane.git camera-model-check
cd camera-model-check
git rev-parse HEAD
python3.12 -m venv /tmp/camera-model-venv
. /tmp/camera-model-venv/bin/activate
pip install --only-binary=:all: -r examples/camera-denoise/requirements-runtime.txt
export PYTHONPATH="$PWD/examples/camera-denoise:$PWD/bindings/python/python"
python - <<'PY'
import numpy as np
from aurora_camera_ane.backend import Backend
model = Backend(1280, 720)
try:
    image = np.full((720,1280), 128, np.uint8)
    result = np.empty_like(image)
    call = model.process(image, None, 1., result)
    print('native refusal:', model.runtime.native_failure)
    print('backend:', call.backend, 'ms:', call.latency_ns/1e6)
    print('stats:', model.runtime.stats())
finally:
    model.fini()
PY
```

With the pinned compiler this should report **CPU (backend 0)** and a compiler
refusal. That is expected fallback evidence, never an ANE result. Record the
kernel, git revision, libane package version and model hashes. Do not replace
these numbers with the x86_64 host receipt.

## 2. Recompile after the compiler gaps are fixed

On the approved buildbox slot, use a fresh model directory and the reviewed
compiler revision. Copy only `weights.npz` and `training.json` into it, then:

```sh
python examples/camera-denoise/export.py "$BUILD_OUT/model" \
  --compiler /path/to/reviewed/mil-hwxc
cat "$BUILD_OUT/model/compiler-gaps.json"
```

Proceed only if full-model H13 compilation succeeds, the native source and
all emitted files have identities in `model.json`, the compiler's existing
package/reference tests pass, and the lead approves that exact source/package
for execution. Preserve failures and their compiler diagnostics; do not edit a
manifest or relax an envelope to manufacture a successful package. Stage the
open model directory and pinned compiler Python sources through the controller.
No Apple-compiled programs or captured payloads are inputs to this process.

## 3. One-tile H13 native parity, then backend validation

The compiler Python owner is pinned at
`45558fb76e08f4c12d9b7d3e333dbac7595f2e80` for the host checks. Use the reviewed
replacement revision when the missing model shapes/movement land. Suppose its
staged path is `/opt/mil-hwx-compiler` and the new model is `/tmp/camera-model`:

```sh
mkdir -m 700 /tmp/camera-model-parity
python - <<'PY'
from pathlib import Path
import numpy as np
x = np.random.default_rng(3652).uniform(0,1,(1,1,64,64)).astype('<f2')
Path('/tmp/camera-model-parity/x.fp16').write_bytes(x.tobytes())
PY
python /opt/mil-hwx-compiler/tools/h13_run_linux.py \
  /tmp/camera-model/compiled/H13/full-model \
  --mil /tmp/camera-model/model.mil --model-root /tmp/camera-model \
  --input x=/tmp/camera-model-parity/x.fp16 \
  --output denoised=/tmp/camera-model-parity/denoised.fp16 \
  --dry-run > /tmp/camera-model-parity/plan.json
```

Review the dry-run plan using the existing compiler/device qualification
ladder. The next command is a separate controller-approved hardware step:

```sh
python /opt/mil-hwx-compiler/tools/h13_run_linux.py \
  /tmp/camera-model/compiled/H13/full-model \
  --mil /tmp/camera-model/model.mil --model-root /tmp/camera-model \
  --input x=/tmp/camera-model-parity/x.fp16 \
  --output denoised=/tmp/camera-model-parity/denoised.fp16 \
  --libane-library /usr/lib/libane_python.so \
  --deadline-seconds 120 --warmup 3 --iterations 20 \
  --benchmark-json /tmp/camera-model-parity/native.json
```

The owner checks numerical parity on every iteration and owns all handles and
cleanup. Its deadline cannot interrupt an in-flight ioctl. Preserve a native
failure privately, retire that owner, and follow the existing lead-run recovery
procedure before retrying. Never reset/reload a failed module from the backend.

Next, stage the approved model under the candidate checkout's
`examples/camera-denoise/models/tiny-luma-v1` and the reviewed compiler sources
under `/usr/lib/mil-hwx-compiler`. Repeat step 1, then test a full frame. Check
`native_calls` as well as latency: CPU fallback must never be labelled ANE.
The frame wrapper still performs CPU reference checks per tile, so tile-native
timings and end-to-end frame latency are separate measurements. This path is
qualification-only until a performant resident graph API is available.

## 4. Camera A/B handoff and rollback

Use the CPU/filter session's installed filter and its documented capture
commands. That session owns gain/fps correlation, low/high-gain capture,
trigger/ramp acceptance and distro switching. Begin with the M1 Pro. Record
whether the selected model actually ran native or fell back; compare identical
luma range, frame size and exposure, and keep captures private. This backend
has no independent A/B camera pipeline or undocumented selection switch.

Rollback selects the already-installed CPU backend through the filter's
documented backend selection, stops/joins the model worker, calls `fini`, and
removes the scratch checkout/venv if desired. No kernel or boot rollback is
needed for this userspace-only candidate. The model scratch remains separate
from the packaged filter. Remove successful private scratch directories;
retain failed native receipts privately until the lead has reviewed them.

H14 requires its existing native graph owner; H16 requires that owner plus
driver qualification. H17 remains excluded until ownership exists. Nothing in
this plan enables the J616s/Neo ANE by overriding generation or ownership.
