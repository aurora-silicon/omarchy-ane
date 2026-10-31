# CPU fallback and the MobileNetV1 baseline

This change extends libane and its existing Python binding. It adds no parallel
runtime library, transport, graph scheduler or package format.

## What already existed

At omarchy-ane `ffc5c266bbb746ce8b198ff90da80d1f6ceff229`:

- `libane/ane.c`, `ane_m2.c` and `ane.h`: ANEC loading, fp16 transfer helpers,
  H13 SUBMIT, H14 program/process load and EXEC, buffer ownership and cleanup.
- `bindings/python/`: the C shared-library shim and `ane.model` Python binding.
- `tools/ane-run.c`: native execution, reference checks and timing output.
- `tools/ane-session.c`: resident program sessions, cache and state handling.
- `packaging/omarchy-ane-smoke` and `tools/omarchy-ane-probe`: smoke and probe tools.
- H15 bring-up code and branches. This change does not replace that work.

At [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx), commit
`033f6061e78dfb50c28f3fb12f9b473bcb373abf`:

- `ane_port_detail` collection in the existing collector: ANE port state and
  coverage evidence.
- `overlay/mlx/backend/omarchy/ane/runtime*`: validated bundles, worker-owned
  libane handles, resident buffers, ordered multi-program dispatch, deadlines and
  failure quarantine. Its documented runtime has no CPU fallback internally.
- Parakeet ASR work and encoder integration. Existing reported native encoder
  results do not license redistribution of Apple-compiled program bytes.

At [mil-hwx-compiler](https://github.com/joshuaswarren/mil-hwx-compiler), commit
`927013f5480768b07c8e62a866e1514c41aba778`:

- Open MIL-to-ANEC/HWX compilation, package validation and physical layout tools.
- `tools/h13_run_linux.py`: prepared resident programs, multi-program graph
  execution, per-call benchmark samples and numerical checks. `ane-bench`
  delegates native H13 graph execution to this runner unchanged.

## What this change adds

- Optional `ane_fallback_init`/`ane_fallback_run` in the existing libane archive.
  A caller supplies native and CPU callbacks using the same inputs and model
  weights. No new hardware execution code is introduced.
- The existing Python binding exports that helper as `ane.Fallback`; optional
  `ane.OnnxCPU` holds an ONNX Runtime CPU session. Existing `ane.model.predict`
  now raises native transfer/execution errors so they can trigger fallback.
- A small `ane_query_caps` accessor through libane's existing enumeration, so
  the H13 benchmark runner is selected only for ABI 1/chip 13. The added GET_CAPS
  definition matches the public Aurora Linux UAPI, where size is an output field.
  This is an execution compatibility check, not another port-survey subsystem.
- `tools/mobilenet_baseline.py`: independent MIL/ONNX export of the baseline
  architecture and identical fp16-rounded random weights, with isolated compiler
  probes for every operation.
- `tools/ane-bench`: CPU baseline measurement with fallback and an optional native
  path through the existing H13 compiler runner, plus tests and this run plan.

## C integration

```c
#include <ane.h>

/* Each callback returns zero or negative errno. Native can wrap ane_exec or
 * an existing graph worker. CPU uses the same open weights and original inputs. */
struct ane_fallback_state state;
struct ane_fallback_call call;
ane_fallback_init(&state, native_is_available);
int rc = ane_fallback_run(&state, native_callback, cpu_callback, context, &call);
```

Initialize once per stream/model and keep the existing native runner and CPU
session resident. Do not open programs per frame. `native_ready` becomes false
when a native callback fails; later calls use CPU without retrying a potentially
faulted context. The native owner retains cleanup, deadline and quarantine policy.
This helper never resets hardware, unloads modules or changes ownership gates.

Callbacks must use the same output contract and leave input tensors and weights unchanged. A CPU callback must
recompute the complete graph from those original inputs, rather than consume
partially written native intermediates. Outputs should be returned or committed
only after a callback succeeds. Camera/mic/STT adapters should wrap their existing
whole-graph call at this boundary. They must retain worker failure evidence and
perform any required external recovery before constructing another native owner.

One caller owns and serializes each C state. Separate states can run concurrently.
The Python wrapper serializes calls and statistics. `calls`, `native_calls` and
`cpu_calls` count successful calls; `failed_calls` counts CPU failures. Nanosecond
counters cover callbacks and fallback, excluding time waiting for a caller's
lock. Average calls/s is `calls * 1e9 / total_ns`. Multiply by logical batch size
for images/s. The per-call record reports backend, latency, native attempt time,
native errno, fallback reason and final status. There is no generation override.

## Python integration

```python
from ane import Fallback, OnnxCPU

cpu = OnnxCPU("open-model.onnx")
# native_graph is supplied by an existing libane/MLX graph owner, or None.
model = Fallback(cpu, native=native_graph, library="/usr/lib/libane_python.so")
outputs, call = model.run(inputs)  # dict of dense NumPy fp16 tensors
print(call.latency_ns, call.backend, model.stats())
```

Use `native_factory=lambda: existing_owner_load(...)` instead of `native` when
loading can fail. The factory runs once; a missing/busy/refused device records
`native_failure` and leaves the CPU path available. It must return the existing
owner's callable graph entry point. It introduces no second model loader.

`OnnxCPU` accepts static float32 ONNX interfaces but requires dense fp16 caller
inputs and returns fp16 outputs. The ONNX constants must be the same fp16-rounded
weights used for MIL. Insert fp16/fp32 casts between operations when activation
rounding matters. The MobileNet exporter builds both graphs from the same arrays
and records source/weight/model hashes. Hashes detect mismatches; they do not prove
equivalence of arbitrary separately authored models.

The ONNX adapter is optional. C applications can supply ONNX Runtime, another
licensed CPU engine or a plain reference callback. Installing the optional Python
adapter requires `pip install 'ane[fallback]'` from this binding's source package.

See [exact compiler gaps and shapes](MOBILENET-COMPILER-GAPS.md).

## Baseline model and timing

The exporter matches the baseline generator's structure and RNG order: alpha 1.0,
224x224 NCHW, batch 1 or 8, NumPy default_rng seed 1, standard-normal weights
scaled by .05, biases on every convolution/linear, 27 convolutions including 13
depthwise blocks, ReLU6, global spatial mean, 1024-to-1000 linear and softmax.
There is no batch normalization. Inputs repeat `(i % 251) / 251`, rounded to fp16.
No pretrained or Apple-compiled model is copied.

CPU operators accumulate in fp32 and round to fp16 at each MIL operation. This
matches graph/weight/precision intent, while accumulation and compiler fusions
can differ from CoreML. CPU records are ONNX Runtime measurements, not proof of
bitwise equivalence with macOS. Native records use the existing runner's MIL
reference criteria. Each CPU result is checked against an untimed CPU reference;
`numerical_parity_checked` identifies that check, not a macOS device comparison.

Twenty warmup calls precede measurement. CPU p50/p90/min and throughput use each
libane fallback call's latency, excluding output comparison time. `wall_s`
includes comparison overhead. Native timing is delegated to the runner, with its
setup and timing scope preserved in `native_report`; it uses an iteration count.
Never place CPU or failed native measurements in an ANE comparison row.

## Build and test

```sh
make -C libane -j8 libane
make -C bindings/python/dylib libane_python
make -C tools check-fallback
python3.12 -m venv /tmp/ane-bench-venv
. /tmp/ane-bench-venv/bin/activate
pip install --only-binary=:all: -r tools/requirements-bench.txt
python3 -m unittest discover -s tools -p 'test_fallback.py' -v
python3 -m unittest discover -s tools -p 'test_ane_bench_owner.py' -v
python3 -m unittest discover -s tools -p 'test_mobilenet_baseline.py' -v
```

A build host can run `scripts/check-fallback-bench.sh` for the same checks plus
batch 1/8 CPU smoke measurements. `scripts/check-fallback-compiler.sh` additionally
builds the pinned published compiler unchanged and captures full/per-op refusals.
Both scripts use private build directories, Python 3.12 wheels and at most eight
compiler jobs. No ANE is needed. The owner/factory policy tests also run with the Python
standard library alone. Tests cover missing/busy/failed native callbacks,
retirement, CPU errors, concurrent calls, same-weight CPU math and baseline shape,
RNG, depthwise count and weight-blob identity.

## M1 Pro lead commands

Run only after controller approval for explicit device `m1pro`, with the existing
qualified H13 kernel/driver and bring-up ladder. No module replacement, boot or
recovery operation is part of these commands. Build the native aarch64 libane
binding first; buildbox x86_64 binaries cannot execute on the M1 Pro.

```sh
git clone --branch feat/libane-cpu-fallback-bench https://github.com/aurora-silicon/omarchy-ane.git omarchy-ane-fallback-bench
cd omarchy-ane-fallback-bench
make -C libane -j8 libane
make -C bindings/python/dylib libane_python
python3.12 -m venv /tmp/ane-bench-venv
. /tmp/ane-bench-venv/bin/activate
pip install --only-binary=:all: -r tools/requirements-bench.txt
export MIL_HWXC=/opt/mil-hwx-compiler/build/mil-hwxc
python3 tools/mobilenet_baseline.py /tmp/mnv1-b1 --batch 1 --compiler "$MIL_HWXC"
python3 tools/mobilenet_baseline.py /tmp/mnv1-b8 --batch 8 --compiler "$MIL_HWXC"
for repeat in 1 2 3; do
    tools/ane-bench /tmp/mnv1-b1 8 | tee "/tmp/mnv1-b1-$repeat.json"
    tools/ane-bench /tmp/mnv1-b8 8 | tee "/tmp/mnv1-b8-$repeat.json"
done
sha256sum /tmp/mnv1-b1/model.onnx /tmp/mnv1-b1/model.mil /tmp/mnv1-b1/weights.bin
sha256sum /tmp/mnv1-b8/model.onnx /tmp/mnv1-b8/model.mil /tmp/mnv1-b8/weights.bin
```

Use Python 3.12 for the pinned wheels; the build scripts can provision it with
uv on build hosts whose system Python is newer. The compiler checkout must be pinned at the commit above, and model destination
paths must be absent on first generation. Preserve `compiler-gaps.json` for both
batches. The current compiler refuses the whole model, so these are CPU results.
After full H13 compiler support lands and device qualification is authorized:

```sh
tools/ane-bench /tmp/mnv1-b1 8 --native-runner /opt/mil-hwx-compiler --iterations 1000
```

The existing runner owns native preparation, validation, residency, dispatch and
cleanup. Native inputs, outputs and logs are preserved in a receipt directory;
the benchmark waits for the native owner to finish cleanup before propagating
wrapper cancellation and does not kill an in-flight runner. A failed native attempt records
its diagnostic and measures CPU as a
separate record; it never combines failed ANE timing with CPU throughput. Do not
retry on the device until the existing runner's recovery policy is satisfied.
The pinned runner's convolution reference also accepts only batch 1, so native
batch 8 needs a reference-runner update even after compiler coverage lands.
H14/H16 native MobileNet benchmarking remains unsupported by that H13 runner;
the runtime's existing H14 support is not changed by this benchmark addition.

Record kernel, libane/compiler commits, generation and model hashes with each
comparison. The M1 Pro is H13 while the macOS baseline is H16. Linux ANE MobileNet
cells remain TBD until the model compiles and native parity passes. Energy also
remains TBD without a separately measured idle/load power protocol.

## Licensing

New code and generated random weights: MIT under this repository's LICENSE.
libane and mil-hwxc: MIT. ONNX Runtime: MIT. ONNX: Apache-2.0. NumPy: BSD-3-Clause.
No Apple firmware, model or program bytes are added. External contributors' repos
are used read-only; this change is proposed only in the Aurora fork.
