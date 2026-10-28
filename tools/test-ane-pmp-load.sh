#!/bin/sh
# test for ane-pmp-load.sh: the ready check passes from the bound driver directory alone (no kernel log needed).
set -eu
D=$(mktemp -d)
mkdir -p "$D/apple_pmp/28e700000.pmp-on"
: > "$D/x.ko"
# pmp_overlay is already in /proc/modules on a host that has it; here it is not, so insmod would run. Stub insmod.
mkdir -p "$D/bin"
printf '#!/bin/sh\nexit 0\n' > "$D/bin/insmod"
chmod +x "$D/bin/insmod"
PATH="$D/bin:$PATH" PMP_DRIVERS="$D" sh "$(dirname "$0")/ane-pmp-load.sh" "$D/x.ko" | grep -q 'PMP ready'
echo "test-ane-pmp-load: ok"
