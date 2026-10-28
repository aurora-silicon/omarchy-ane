#!/bin/sh
# ane-pmp-load: start the PMP firmware for the ANE clock level on demand (16-inch M1 Max).
#
# usage: ane-pmp-load.sh /path/to/pmp_overlay.ko
#
# The level request in ane.ko needs the PMP firmware running. A module that
# applies the PMP device-tree node starts it. That module has no unload path:
# once it is loaded, the PMP stays on until the next reboot. Load it from the
# script that runs the ANE work, and leave it out of boot, if you also run GPU
# work on the machine and want the PMP off for that work.
#
# Exits 0 when the PMP is ready (already loaded or loaded now), 4 when the
# module is missing, fails to load, or the PMP is not ready after 90 s.
set -u
KO=${1:?usage: ane-pmp-load.sh /path/to/pmp_overlay.ko}
[ -f "$KO" ] || { echo "ane-pmp-load: $KO not found" >&2; exit 4; }
if ! grep -q '^pmp_overlay ' /proc/modules; then
	insmod "$KO" || { echo "ane-pmp-load: insmod failed" >&2; exit 4; }
fi
# Ready when the PMP device is bound to its driver (this stays true after the
# kernel log wraps) or the ready line is in the log (right after a load).
pmp_ready() {
	for d in "${PMP_DRIVERS:-/sys/bus/platform/drivers}"/apple_pmp/*.pmp* "${PMP_DRIVERS:-/sys/bus/platform/drivers}"/apple-pmp/*.pmp*; do
		[ -e "$d" ] && return 0
	done
	dmesg | grep -q 'PMP firmware and endpoint ready'
}
i=0
while [ "$i" -lt 90 ]; do
	pmp_ready && { echo "ane-pmp-load: PMP ready"; exit 0; }
	i=$((i + 1))
	sleep 1
done
echo "ane-pmp-load: PMP not ready after 90 s" >&2
exit 4
