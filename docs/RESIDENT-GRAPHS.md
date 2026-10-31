# Guarded resident graph loading

The existing libane handles already keep their H13 BOs or H14 program/procedure
and I/O buffers resident until `ane_free`. The new entry points make that
lifetime usable by the matching compiler graph runner with explicit validation:

```c
/* target is 13 or 14, derived from a validated ANEC package manifest. */
struct ane_nn *nn = ane_init_target(path, device_index, target);
/* Check for NULL before use, and validate all manifest physical buffer sizes. */
int rc = ane_send_checked(nn, input, input_allocation_bytes, input_index);
if (!rc)
    rc = ane_exec(nn);
if (!rc)
    rc = ane_read_checked(nn, output, output_allocation_bytes, output_index);
/* Repeat on this handle; the owner closes it once after the final call. */
ane_free(nn);
```

The loader checks GET_CAPS on the same fd used by the load, before any buffer
allocation or firmware load. Expected target 13 requires ABI 1/family 13; target
14 requires ABI 2/family 14. A family mismatch, ABI mismatch, unknown family,
missing GET_CAPS or short capability result is refused. This expected target is
a package compatibility assertion, not a device generation override. The
legacy init entry points retain their behavior.

Checked transfers require a non-null pointer, a valid role and the exact
physical allocation size. They preserve ABI-2 transfer errors and return zero
or negative errno. `ane_exec` keeps its existing return convention: a negative
ioctl result can carry the error in `errno`. Callers normalize that convention
at their callback boundary. Every graph owner must stop using and close the
whole graph after any native failure, retaining its diagnostic and obeying the
existing device recovery procedure. Neither these APIs nor close recover a
hung engine or grant hardware ownership.

The matching `mil-hwx-compiler/tools/h13_run_linux.py` change exports a reusable
`ResidentGraph`. It validates all package metadata before load, checks physical
sizes/counts before transfer, prepares all programs once, composes fresh fp16
intermediates per call, and retires every handle on failure. H13 keeps its
existing v2 package validation. Initial H14 graph validation admits per-operation
`h14-oracle-parity` elementwise packages only. The libane H14 loader still uses
its existing section, BAR and buffer checks. See that compiler's
`docs/ane/resident-linux.md` for its exact supported scope and caller lifecycle.

Use the existing `ane_fallback_run` or Python `ane.Fallback` around the complete
graph callback. A CPU callback must use the same original inputs and weights,
and recompute the full graph after a native failure. It must never consume
partially written native intermediates. Keep the native owner's lifetime
explicit and close it on exit. No fallback policy, transport, model loader,
package format or graph scheduler is duplicated inside libane.

The software contracts extend `libane/ane.c`, `libane/ane_m2.c` and
`ane/src/uapi/drm/ane_accel.h` at base commit
`1adf2ee` in this repository. The matching compiler base is
`8f8fb25f58e05a27db4c503e8eea7d4f13be4219`. No descriptor encoding, channel
assignment, kernel interface, device resource gate or qualification gate changes.
These APIs do not qualify native device behavior or performance.

Host checks:

```sh
make -C libane -j8 libane
make -C bindings/python/dylib libane_python
make -C tools test_libane_ioctl check-fallback
tools/test_libane_ioctl fixtures/h14-anec
```

The fake-device ioctl harness checks both families, refuses mismatched or
missing capabilities before BO allocation, reuses handles without allocating
new BOs per call, rejects short/oversized transfers, and closes all owned BOs.
The paired compiler tests cover graph residency, validation, failure retirement
and existing CPU fallback. They require no ANE hardware.
