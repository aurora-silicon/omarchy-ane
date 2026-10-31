# Lead qualification and rollback

All target access belongs to the lead through controller requests naming the
actual device and satisfying source, ownership, resource and policy gates.
This change's worker performs host buildbox checks only. A compiler receipt or
host test is not device qualification.

## Install and opt-in CPU checks

1. Build/install the companion camera filter and this backend into a temporary,
   reviewable package/staging prefix, with runtime dependencies and open models.
   Record source commits and installed artifact hashes. Retain the prior package.
2. On the lead-selected device, confirm the existing camera's negotiated NV12
   format, matrix/range and frame metadata. Test M1 Pro's CPU path first; default
   effects off must produce byte-identical effect-stage pass-through and must
   not query an ANE. No device/power or usage override is needed.
3. Start two instances:

   ```sh
   aurora-camera-denoise --stream callA --mode off --verbose
   aurora-camera-denoise --stream callB --mode off --verbose
   ```

   Select each named camera explicitly in two clients at the same 720p format.
   Identify output node IDs with `pw-dump`. Never apply controls to the physical
   V4L2 source or the other call's node. Commands below use `CALL_A_NODE_ID` as a
   placeholder for the output ID, not a shell variable or guessed hardware ID.
4. Verify both outputs advertise background `off`, framing `false`, zoom `0`.
   Enable blur only on A:

   ```sh
   pw-cli set-param CALL_A_NODE_ID Props '{ params = [ "camera.background" "blur" ] }'
   ```

   A changes; B remains untouched. Repeat with a user-owned replacement image and
   with both effects off, and confirm instance teardown releases its worker and
   camera input. Installing/restarting the service must not enable effects.
5. Inspect nearby hair, hands, glasses, partial occlusion, backlighting and low
   light. Test 720p and 1080p, limited/full BT.601/709 where the source offers
   them, padded row strides, format changes, stop/start and source removal.
   Check foreground colour and chroma edges. Compare with raw/denoise-only output.
   Capture only lead-approved non-private inputs; do not publish camera pixels.

## Framing board contract

- J616s must advertise capability and min/default/max 1.0/1.5/6.0 from the board
  file. It starts off. Enable automatic framing with `camera.framing=true`,
  `camera.zoom=0.0`. Verify smooth panning, zoom-in/out, multiple-face union,
  one-second face-loss hold and return to default. This is an open digital crop,
  not a claim of macOS lens correction or exact framing composition.
- Exercise manual zoom 1.0, 1.5 and 6.0; reject values outside the board range or
  nonfinite values atomically. Even NV12 coordinates must stay in the sensor.
- J700/Neo and unknown boards have no framing block. A mixed transaction that
  requests framing must be rejected without partially enabling replacement or
  changing another control. Blur remains a separate supported user choice.

## Overload and failure acceptance

Use the host fixture first: it deliberately stalls inference after output begins.
Then, on the lead's target, observe a busy CPU/ANE and rapid control/format
changes. Queues stay bounded; pending buffers return to the free pool; output
recovers after a stall. No stale mask/frame is shown after stream restart or
changing replacement. Delivered frames retain their own original PTS/sequence,
never the timestamp of the newer input callback that delivered them. A source
format change at the same size/rate but different matrix/range rebuilds effects.

Missing plugin, failed model identity, invalid replacement or inference failure
must leave a usable pass-through camera and a diagnostic. Verify the selected
control transaction is visible with `pw-cli enum-params ... Props`; visual
effects are not a guarantee that the background cannot be exposed on failure.

Measure callback copy/queue overhead plus worker wall and process CPU ms/frame;
record negotiated size/rate, dropped frames, thermal/clock context and competing
work. Compare these with the host receipt, not with an assumed ANE speedup. Check
end-to-end call latency/audio sync and browser/portal camera selection.

## Native ANE gate

The shipped models advertise no native targets. Do not enable the dormant
adapter by editing a flag alone. Run `check-compiler-buildbox.sh` through a
controller build slot and inspect its `$BUILD_OUT/compiler-gaps.json`; resolve
any full-model lowering refusals before producing identified native packages
through the open compiler. Establish a complete fp16 reference for all
ops and output bindings. Record numerical parity and resident reuse on the
lead-selected generation before adding a qualified target to model metadata.

The current adapter delegates to the existing H13 graph owner only. H14 and H16
need their respective owners and qualification; J616s waits for that path. Neo
H17 remains CPU-only while Linux lacks accelerator ownership. Do not bypass a
missing generation/ownership gate. Native failure retires that instance through
shared runtime fallback and preserves private failure evidence; follow the
existing runtime's recovery procedure, never attempt application-local reset.

## Rollback

Set `camera.background=off` and `camera.framing=false`, then stop the named
instances. This restores direct effect-stage output and joins workers. Remove
only the optional effects plugin/models or reinstall the saved filter/backend
packages. The existing raw camera and CPU denoise remain selectable. Preserve
failed build/test receipts and private runtime failure directories for review;
do not include their camera data in a source commit or public report. No kernel,
boot, firmware, volume or device calibration change is part of this task.
