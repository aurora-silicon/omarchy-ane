# Lead: install, qualify, try

No device run was done by this worker. The bundles are unqualified until the
commands below pass on the M1 Pro. Use the controller's device identity and
policy approval for device execution. Do not copy a CPU simulation receipt into
`qualification.json`; the runtime rejects simulation receipts.

M1 Pro, Fedora 44 aarch64, from `examples/camera-effects` in this branch:

```sh
bash h13/setup.sh /path/to/mil-hwx-compiler/build/mil-hwxc
```

The recorded offline bundles use PR #5 revision `72e1d99c6f51eaf97ad61280936d42dbb0b6a1ae`. Use the reader compatibility branch `fu-camera-h13-reader` on PR #5 when running them: the original reader misclassifies channel-changing conv as elementwise. No new compiler op encoder is required by this branch. The build
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
for model in selfie yunet; do .venv-h13/bin/python h13/verify.py "models/$model" --package "models/$model/compiled/H13/hybrid" --images /tmp/camera-h13-portraits --qualify --library ../../bindings/python/dylib/libane_python.so --output "/tmp/$model-h13.json" || exit; done
```

Expect all island comparisons and 30-image criteria to pass. The tool checks
ABI 1/H13 before preparing programs; its owner validates program surfaces and
libane allocation sizes. A successful run writes a receipt bound to the plan
hash. A failed requalification replaces any old qualification with a failure.
Preserve both JSON outputs and failing packages/logs for the compiler lane.
The inter-island deadline cannot interrupt an in-flight driver submission.

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
