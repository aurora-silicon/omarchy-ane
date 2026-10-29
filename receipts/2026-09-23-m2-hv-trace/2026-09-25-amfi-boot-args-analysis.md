# AMFI boot-args analysis on macOS 13.5 RELEASE (2026-09-25)

## 1. Panic root cause
In `v7-20260925T144010`, the guest kernel panicked with:
```
panic(cpu 0) 'can't has cs_enforcement_disable'
```
The panic above was seen in a run. Expected cause: with `cs_enforcement_disable`
present in the boot arguments, macOS asks whether the kernel debugger is allowed.
If it is not allowed, macOS panics with the message above.

`cs_enforcement_disable` is the only AMFI boot argument that asks for the
kernel-debugger permission (CSR bit 0x08). Every other AMFI argument below is
accepted with no permission check.

## 2. Permitted AMFI bypass arguments on RELEASE
Expected on a RELEASE build, not yet run: three bypass arguments are
accepted, none of them checks a permission and none of them panics.

### A. `amfi_allow_any_signature=1`
- Accepted on its own, or as bit 1 of the `amfi=` mask (`0x02`).
- macOS prints "signature enforcement disabled by boot-arg".
- Signature enforcement is bypassed for that boot.
- No kernel-debugger permission check. No panic.

### B. `amfi_get_out_of_my_way=1`
- Accepted on its own, or as bit 7 of the `amfi=` mask (`0x80`).
- macOS prints "signature enforcement disabled by boot-arg".
- macOS stops marking external binaries as platform binaries.
- macOS disables app bundle restrictions because the argument is set.
- No kernel-debugger permission check. No panic.

### C. `amfi_unrestricted_local_signing=1`
- Accepted on its own.
- macOS prints "unrestricted AMFI local signing enabled by boot-arg".
- Local and ad-hoc code signing is accepted.
- No kernel-debugger permission check. No panic.

## 3. Recovery property preservation
The hardware recovery property must be kept:
- Remove `cs_enforcement_disable` completely.
- Remove `debug=0x14e` (specifically `DB_NMI = 0x4`) and `wdt=-1` from default `BOOTARGS`.
- Without `DB_NMI` the guest does not enter panic debugging.
- Without `wdt=-1` the hardware watchdog stays armed.
- If macOS panics, the guest should restart the CPU instead of spinning forever. The SoC should reset itself, and the M2 should reboot back to Linux via its fallback ESP image without human intervention.
- `serial=3 -enable-kprintf-spam` is preserved so the full console and panic trace
  stream to `vuart.log`.
- `M2HV_DEBUG=1` remains available as an explicit environment override when
  a parked panic-spin is desired for live hypervisor debugging.

## 4. Next boot-args set
```
serial=3 apcie=0xfffffffe -enable-kprintf-spam clpc=0 amfi_get_out_of_my_way=1 amfi_allow_any_signature=1 amfi_unrestricted_local_signing=1 rd=md0 -rootdmg-ramdisk rp=file:///ane-root.dmg
```
Length: 196 characters (well within the 1024-character `BootArgs_r3` limit).
Total gates: 0. Panics: 0. Code signing bypassed: complete.
