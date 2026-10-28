#!/bin/sh
# test for ane-pmp-load.sh, with stand-in directories and stub insmod, dmesg and sleep:
#  1. the ready check passes from the bound driver directory alone, and insmod ran once;
#  2. a missing module file exits 4;
#  3. a failing insmod exits 4;
#  4. the ready line in the kernel log alone is enough;
#  5. no bound device and no ready line exits 4.
set -eu
HERE=$(dirname "$0")
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/bin" "$D/drv/apple_pmp/28e700000.pmp-on" "$D/empty"
: > "$D/x.ko"
# pmp_overlay is not in /proc/modules on the test host, so the script would insmod. Stub the calls.
# shellcheck disable=SC2016
printf '#!/bin/sh\necho x >> "%s/insmod.calls"\nexit "${INSMOD_RC:-0}"\n' "$D" > "$D/bin/insmod"
printf '#!/bin/sh\n[ -n "${DMESG_LINE:-}" ] && echo "$DMESG_LINE"\nexit 0\n' > "$D/bin/dmesg"
printf '#!/bin/sh\nexit 0\n' > "$D/bin/sleep"
chmod +x "$D/bin/insmod" "$D/bin/dmesg" "$D/bin/sleep"
run() { PATH="$D/bin:$PATH" sh "$HERE/ane-pmp-load.sh" "$@"; }

PMP_DRIVERS="$D/drv" run "$D/x.ko" | grep -q 'PMP ready'
[ "$(wc -l < "$D/insmod.calls")" = 1 ] || { echo "FAIL 1: insmod not called once"; exit 1; }

rc=0; PMP_DRIVERS="$D/drv" run "$D/missing.ko" 2> /dev/null || rc=$?
[ "$rc" = 4 ] || { echo "FAIL 2: missing file rc $rc"; exit 1; }

rm -f "$D/insmod.calls"
rc=0; INSMOD_RC=1 PMP_DRIVERS="$D/drv" run "$D/x.ko" 2> /dev/null || rc=$?
[ "$rc" = 4 ] || { echo "FAIL 3: insmod failure rc $rc"; exit 1; }
[ -s "$D/insmod.calls" ] || { echo "FAIL 3: insmod not called"; exit 1; }

DMESG_LINE='apple_pmp 28e700000.pmp-on: PMP firmware and endpoint ready' PMP_DRIVERS="$D/empty" run "$D/x.ko" | grep -q 'PMP ready' \
	|| { echo "FAIL 4: ready line alone"; exit 1; }

rc=0; PMP_DRIVERS="$D/empty" run "$D/x.ko" 2> /dev/null || rc=$?
[ "$rc" = 4 ] || { echo "FAIL 5: not ready rc $rc"; exit 1; }
echo "test-ane-pmp-load: ok"
