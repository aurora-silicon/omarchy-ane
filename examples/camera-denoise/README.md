# Open camera model backend

This is the ANE/model half of camera denoising. It supplies a provisional C
backend vtable and a matching Python interface. The video filter owns gain/fps
policy, hysteresis, ramp length, configuration, temporal CPU denoising and
chroma. No kernel or PipeWire node is added here.

TinyLuma-UNet-v1 is an original 1-level, 8-channel additive-skip spatial UNet.
It processes full-range BT.709 luma with five 3x3 convolutions, one stride-2
downsample and nearest 2x upsampling. Both native MIL and CPU ONNX use the same
released fp16-rounded parameters. CPU accumulation is fp32, with explicit
fp16 activation casts at every MIL operation. It is a candidate model trained
on a very small open corpus, not a qualified camera-quality replacement.

## Integration

`camera_ane_backend.h` defines the temporary seam:

```c
void *context = camera_ane_backend_v1.init(width, height, CAMERA_ANE_GRAY8);
int rc = camera_ane_backend_v1.process(context, &input_luma, NULL, strength, &output_luma);
camera_ane_backend_v1.fini(context);
```

The filter can adapt this object to its final backend vtable. One context owns
one stream. Call from a worker thread; join all processing before `fini`.
The bridge embeds CPython once, never finalizes the process interpreter, and
uses ONNX Runtime and the existing libane Python binding. A host that already
embeds Python must arrange its normal Python module path and interpreter
lifetime; subinterpreters are unsupported. This is a prototype integration,
not an allocation-free or real-time-safe implementation.

Input/output are GRAY8 planes with explicit stride and byte capacity. Pass a
borrowed luma plane from NV12/I420; preserve chroma and padding in the filter.
This ABI expects full-range luma. The filter must convert limited-range input
or avoid selecting this model. The backend reserves `prev_state` without
reading or modifying it. Width/height may be odd, up to 4096x2160. Strength is
finite in [0,1]. Zero strength copies exactly and never runs inference. A
successful call commits a whole blended frame; errors leave output untouched.
The filter supplies the ramp strength, so this backend has no trigger policy.

The Python equivalent is `Backend(width,height,"GRAY8")`,
`process(in_frame,prev_state,strength,out_frame)`, and `fini()`.
`runtime.stats()` and `runtime.native_failure` expose fallback counters and
the native refusal. The C bridge prints Python failures to stderr and returns
negative errno. It introduces no gain, generation, device or environment knob.

## Runtime and native limits

The t-3650 `ane.Fallback` owns fallback and callback timing. A native exception
retires that callback for the context's lifetime; the full CPU graph recomputes
from the original input. A fresh model/backend context is only appropriate
after the existing native owner's recovery policy is satisfied.

Native execution delegates layout validation, fp16 reference checking and
graph scheduling to the pinned compiler's existing `h13_run_linux.py` owner.
Its existing `PreparedProgram` keeps tile programs resident. The wrapper adds
no ioctl, register encoder, graph scheduler or recovery stack. Fixed 64x64
tiles have a 12-pixel halo and a 40x40 useful center, aligned to the stride-2
network. A host test checks dense versus tiled output including seams/edges.

The owner evaluates a CPU reference per tile and uses private scratch files.
This native path is for qualification; it does not establish a performance
benefit. It checks a two-second frame deadline between tiles and delegates
between-program checks to the existing owner. It cannot interrupt an in-flight
kernel submission. Success removes scratch data at close. Failure preserves a
mode-0700 `camera-ane-native-*` directory and keeps the failed owner retired.
Those directories may contain camera pixels and are private failure evidence,
never material for public reports. No hardware reset or reload occurs here.

Only a successfully compiled, hash-identified full-model H13 package can
activate native execution, followed by ABI/chip capability checks. Missing or
refused packages use CPU. H14/H16 graph integration is refused explicitly;
their library support alone does not prove this graph. H17 ownership remains
outside this backend. Board policy and selection belong to the filter.

## Reproduce training and export

Use a buildbox controller slot, Python 3.12 and at most eight build jobs. These
are commands inside that approved slot, with a fresh `$BUILD_OUT`:

```sh
python3.12 -m venv "$BUILD_OUT/venv"
. "$BUILD_OUT/venv/bin/activate"
pip install --only-binary=:all: -r examples/camera-denoise/requirements.txt
pip install --only-binary=:all: torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python examples/camera-denoise/train.py "$BUILD_OUT/trained" --steps 3000
python examples/camera-denoise/export.py "$BUILD_OUT/trained" \
  --compiler /path/to/pinned/mil-hwxc
```

Train seed 3652, 16 random 64x64 patches/step, Adam, clipped AWGN sigma 5..40,
rotations/flips and a 40x40 center loss are fixed in source. Training images
are `astronaut` and `coffee`; `chelsea` and `hubble_deep_field` are held out.
The exporter consumes `weights.npz`, constructs a typed coremltools 9.0 MIL
program, validates it, and serializes that actual graph to `model.mil` plus
`weights.bin`. It also saves an uncompiled source `.mlpackage` and an ONNX
graph. It never calls Core ML's compiler. Generated source is open; there are
no Apple-compiled HWX bytes. Export refuses reused destinations to prevent
stale packages from becoming active.

`compiler-gaps.json` records the complete graph and every isolated operation
for H13/H14/H16G, with exact shapes, return codes and diagnostics. Failed
compilation never activates native execution. The pinned compiler revision is
`45558fb76e08f4c12d9b7d3e333dbac7595f2e80` from the Aurora fork, including the
accepted t-3651 changes. Native tiling/movement/resize and model geometries
must remain refused until the compiler proves them.

The model [licence audit](LICENCE-AUDIT.md) identifies all code/weights/data.
Released weights are MIT, with their own [notice](models/tiny-luma-v1/LICENSE).
Training needs PyTorch; inference does not.

[Host measurements and validation limits](VALIDATION.md) include the CPU
720p/1080p timings and held-out PSNR/SSIM. The CPU model is currently slower
than a 30 fps frame budget; this candidate must not replace the existing
NEON backend by default.

[Exact compiler coverage and handoff](COMPILER-GAPS.md) records the current
native blocker. The shipped metadata activates no native target.

## Build and host validation

After installing the Python dependencies and building libane:

```sh
make -C libane -j8 libane
make -C bindings/python/dylib libane_python
export PYTHONPATH="$PWD/examples/camera-denoise:$PWD/bindings/python/python"
python -m pytest -q examples/camera-denoise/tests
make -C examples/camera-denoise PYTHON="$(command -v python)" \
  PYTHON_DIR="$PWD/examples/camera-denoise" check
python examples/camera-denoise/measure.py --output /tmp/camera-model-measurements.json
```

The standalone C bridge is built against the selected Python executable and
its matching development headers/library. Build the plugin on the intended
architecture; buildbox x86_64 binaries are not ARM install artifacts. Default
installation uses `/usr/lib/aurora-camera-ane` for Python code/model data and
the normal libane Python package/library. `PREFIX`, `DESTDIR`, `PYTHON_DIR`
are installation paths, not inference policy controls.

`check-buildbox.sh` retrains the candidate and reproduces the pinned compiler, export, model identity,
seam/stride/fallback tests, warning-clean C bridge and host measurements in one
controller build slot. It records the training source hashes and retains all
compiler refusal diagnostics. Receipts document CPU model latency separately
from normalize/blend/quantize cost. Host CPU results do not imply ANE or ARM
performance. Quality is PSNR/SSIM on `open-luma-awgn-v1`, with decoded image
hashes and independent noise seeds. This tiny synthetic test does not cover
sensor noise, color quality, temporal artifacts or real camera motion.

See [LEAD-TEST-PLAN.md](LEAD-TEST-PLAN.md) for the exact native qualification
steps and how to return to the existing CPU backend.
