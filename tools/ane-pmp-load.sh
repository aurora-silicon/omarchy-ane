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
# module is missing, fails to load, or the ready line does not show up in 90 s.
set -u
KO=${1:?usage: ane-pmp-load.sh /path/to/pmp_overlay.ko}
[ -f "$KO" ] || { echo "ane-pmp-load: $KO not found" >&2; exit 4; }
if ! grep -q '^pmp_overlay ' /proc/modules; then
	insmod "$KO" || { echo "ane-pmp-load: insmod failed" >&2; exit 4; }
fi
i=0
while [ "$i" -lt 90 ]; do
	dmesg | grep -q 'PMP firmware and endpoint ready' && { echo "ane-pmp-load: PMP ready"; exit 0; }
	i=$((i + 1))
	sleep 1
done
echo "ane-pmp-load: no ready line in 90 s" >&2
exit 4
