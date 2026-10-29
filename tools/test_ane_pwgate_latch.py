#!/usr/bin/env python3
"""Source-order guard for the T8112 PWGATE write (ane/t6021/ane_t6021_rtclient_main.c).

The probe may write the gate word at most once per boot. After a poll TIMEOUT it returns an error, and the probe wrapper latches
"reboot required" only if *hardware_touched is set; without the latch a later probe in the same boot could write again. After a
write that WORKED the latch must stay clear, so later clean failures of the same probe (-EPROBE_DEFER, -EINVAL, -ENOMEM) remain
retryable. So the latch must be set once, in the timeout branch: after the writel and the poll, before the REFUSE plan.
This is a text check of that order, not a run of the driver. Exit 0 = ordered.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ane/t6021/ane_t6021_rtclient_main.c"
with open(path) as f:
    src = f.read()
block = re.search(r"if \(plan == ANE_PWGATE_WRITE\) \{(.*?)\n\t\t\}\n", src, re.DOTALL)
if not block:
    sys.exit("FAIL: cannot find the ANE_PWGATE_WRITE block")
body = block.group(1)
write = body.find("writel(0, set + ane->soc->pwgate_off)")
poll = body.find("readl_poll_timeout(")
latch = body.find("*hardware_touched = true;")
refuse = body.find("plan = ANE_PWGATE_REFUSE;")
if min(write, poll, refuse) < 0:
    sys.exit("FAIL: write, poll or refuse not found in the write block")
if body.count("writel(") != 1:
    sys.exit("FAIL: the write block has more than one writel")
if latch < 0:
    sys.exit("FAIL: *hardware_touched is never set in the write block")
if body.count("*hardware_touched = true;") != 1:
    sys.exit("FAIL: *hardware_touched is set more than once in the write block")
if not write < poll < latch < refuse:
    sys.exit("FAIL: the latch must sit in the poll-timeout branch (after the write and poll, before the REFUSE plan)")
print("ok: the latch is set only in the poll-timeout branch")
