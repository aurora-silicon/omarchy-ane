#!/bin/bash
# SPDX-License-Identifier: MIT
# Run only inside a controller-approved execute slot for an explicit H13 device.
# No usage/presence checks; no camera/audio, boot or system configuration changes.
# COMPILER=/path/to/mil-hwx-compiler PORTRAITS=/path/to/portraits bash h13/slot.sh
set -euo pipefail
CE=${CE:-$(cd "$(dirname "$0")/.." && pwd)}
# Source identities come from the controller-approved artifacts, never a local
# machine pin committed upstream.
: "${COMPILER:?set COMPILER to the reader-compatible compiler checkout}"
: "${LIB_SHA256:?set the controller-approved binding SHA-256}"
: "${SELFIE_PLAN_SHA256:?set the controller-approved selfie plan SHA-256}"
: "${YUNET_PLAN_SHA256:?set the controller-approved YuNet plan SHA-256}"
PORTRAITS=${PORTRAITS:-/tmp/camera-h13-portraits}
RESULTS=${RESULTS:-$CE/slot-results}
LIB=$CE/../../bindings/python/dylib/libane_python.so
PY=${PY:-$CE/.venv-h13/bin/python}
DEV=/sys/class/accel/accel0/device
if [ "${1:-}" != --inner ]; then
    # Leave 20 seconds for forced teardown inside a ten-minute controller slot.
    echo 'slot: ten-minute budget (580s work + 20s teardown); preserve all results'
    exec timeout --kill-after=20 580 bash "$0" --inner
fi
START=$SECONDS
step() { printf '\n=== [%3ds] %s\n' "$((SECONDS-START))" "$*"; }
die() { echo "slot failed: $*" >&2; exit 1; }
[ -e /dev/accel/accel0 ] || die 'missing accelerator'
tr -d '\0' </proc/device-tree/compatible | grep -qE 'apple,t600[01]' || die 'requires M1 Pro/Max H13'
[ "$(cat "$DEV/wedged")" = 0 ] || die 'accelerator wedged'
[ "$(cat /sys/module/ane/parameters/boost_idle_ms)" = 100 ] || die 'expected boost_idle_ms=100'
[ -f "$LIB" ] || die 'missing libane binding'
[ "$(sha256sum "$LIB" | cut -d' ' -f1)" = "$LIB_SHA256" ] || die 'binding identity changed'
[ "$(sha256sum "$CE/models/selfie/compiled/H13/hybrid/plan.json" | cut -d' ' -f1)" = "$SELFIE_PLAN_SHA256" ] || die 'selfie plan identity changed'
[ "$(sha256sum "$CE/models/yunet/compiled/H13/hybrid/plan.json" | cut -d' ' -f1)" = "$YUNET_PLAN_SHA256" ] || die 'YuNet plan identity changed'
[ -f "$COMPILER/tools/h13_run_linux.py" ] || die 'missing compiler graph owner'
if [ -e "$RESULTS" ]; then
    old=$RESULTS.prev-$(date +%Y%m%d-%H%M%S)-$$
    mv "$RESULTS" "$old"
    echo "previous results preserved: $old"
fi
mkdir -p "$RESULTS/preview"
RESULTS=$(cd "$RESULTS" && pwd)
EPOCH0=$(date +%s)
final() {
    rc=$?
    trap - EXIT
    echo "slot finished: rc=$rc elapsed=$((SECONDS-START))s"
    cat "$DEV/wedged" > "$RESULTS/wedged.after" 2>&1 || true
    cat "$DEV/ane_stats" > "$RESULTS/ane_stats.after" 2>&1 || true
    journalctl -k --no-pager -S "@$EPOCH0" > "$RESULTS/kernel.log" 2>&1 || true
    echo "results: $RESULTS"
    exit "$rc"
}
trap final EXIT
trap 'exit 124' TERM
exec > >(tee -a "$RESULTS/slot.log") 2>&1
cd "$CE"
uname -r > "$RESULTS/kernel.txt"
sha256sum "$LIB" models/{selfie,yunet}/compiled/H13/hybrid/plan.json > "$RESULTS/identities.sha256"
cat "$DEV/ane_stats" > "$RESULTS/ane_stats.before"
for model in selfie yunet; do
    step "qualify $model: all islands, then final mixed graph"
    "$PY" h13/verify.py "models/$model" --package "models/$model/compiled/H13/hybrid" \
        --images "$PORTRAITS" --qualify --library "$LIB" --compiler-root "$COMPILER" \
        --output "$RESULTS/$model-h13.json"
    cp "models/$model/compiled/H13/hybrid/qualification.json" "$RESULTS/$model-qualification.json"
    [ "$(cat "$DEV/wedged")" = 0 ] || die "wedged after $model"
done
# Same three public evaluation portraits as the original lead slot.
paths=()
for image in 000000000872.jpg 000000012280.jpg 000000019924.jpg; do
    paths+=("$PORTRAITS/$image")
    step "blur/framing preview: $image"
    "$PY" try-camera-h13 preview --input "$PORTRAITS/$image" --engine auto --framing \
        --compiler-root "$COMPILER" 2>&1 | tee "$RESULTS/preview/${image%.jpg}.log"
    grep -q 'engine: ane-hybrid (' "$RESULTS/preview/${image%.jpg}.log" || die 'preview did not use hybrid'
    if grep -q 'engine=cpu' "$RESULTS/preview/${image%.jpg}.log"; then die 'whole-model CPU fallback'; fi
    cp "${XDG_CACHE_HOME:-$HOME/.cache}/aurora-camera-h13/preview.png" "$RESULTS/preview/${image%.jpg}.png"
done
step 'timing: hybrid vs CPU, 256x256 and full preview, 1 warm-up + 10 frames'
"$PY" h13/timing.py --engine both --framing --frames 10 --compiler-root "$COMPILER" \
    --output "$RESULTS/timing.json" "${paths[@]}"
step done
