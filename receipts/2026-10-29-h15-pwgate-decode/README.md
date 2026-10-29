# T6031 PWGATE step: where the register claims come from

This receipt gives the source for every register claim behind the `pwgate_t6031` module parameter of `ane/h15`. It is a static decode of Apple's kernel extension. Nothing here ran on an M3. The claims are decoded, not measured.

## Inputs, so anyone can repeat it

| Item | Value |
|---|---|
| Restore image | `UniversalMac_27.0.1_26A434_Restore.ipsw`, https://updates.cdn-apple.com/2026FallFCS/59241290-5d51-4ca8-9df4-31624b9a4eac/UniversalMac_27.0.1_26A434_Restore.ipsw, sha256 `2f016638293c3e641b8b25391a76fbc16563b3711915a5551cf8aa0f5598a5c1` |
| Kernelcache | `kernelcache.release.Mac15,6_7`, sha256 `1e95b8667e1afb3e52ef45fb7e66e67f89b2da084a65f46c49d3d0eb7db84d11` (`ipsw extract --kernel --remote --device Mac15,7 <url>`) |
| Kext | `com.apple.driver.AppleH11ANEInterface` 10.19.2, sha256 `75061e9ce3c5980a4d5d9ce38c5ec20eb45bbdd0d75485f061b0c6c9037b4f50` (`ipsw kernel extract <kernelcache> com.apple.driver.AppleH11ANEInterface`) |
| Device tree | `DeviceTree.j516cap.im4p` from the same image (`ipsw extract --remote --pattern 'DeviceTree\.j516cap\.im4p' <url>`), node `/arm-io/ane0`, ane-type 224 (0xe0) |

Addresses below are virtual addresses inside that kext. Disassemble with any arm64 disassembler; the kext uses pointer authentication, so skip `autda` and `xpacd` lines when reading.

## Claim 1: the PWGATE window is the fifth reg entry of the ANE node

The `ane0` node of the J516C device tree has eight reg entries. Each device tree address gets the 0x200000000 offset of the tree's ranges. In order (physical address, size): wrapper `0x3c9400000` `0x6c000`, RVBAR block `0x3c9050000` `0x4000`, engine `0x3c8000000` `0x2000000`, PMGR `0x292280000` `0x10000`, **`0x292290000` `0x4000`**, `0x2903c0000` `0x30000`, `0x404000000` `0x2000000`, `0x2a02dc000` `0x4000`.

The kext maps its windows by name. At `0xfffffe00095c9594` to `0xfffffe00095c95cc` it stores the "PS" window pointer at object offset `0x190` and the "PWGATE" window pointer at `0x198`. The strings are at `0xfffffe00074c1de1` ("PS") and `0xfffffe00074c1de4` ("PWGATE").

The PMGR window holds the ANE power words (`ANE_CPU` at `0x5a8`). So the next window in the list, `0x292290000`, is the one the kext calls PWGATE. That mapping of window index to name is the one inferred step. The store of the PWGATE pointer runs only when a config byte at `[cfg + 0x16b]` has bit 0 set (`0xfffffe00095c95a0`). The T6031 config sets it.

## Claim 2: the T6031 branch clears one word and polls it

Function `0xfffffe0009598d2c` (the kext's "EnableCPUClocksAndPower", named by its own log strings). For the T6031 type (`0xe0`, compare at `0xfffffe000959901c`) it runs, in order:

```
0xfffffe0009599024: mov w1, #0x5a8        ; PS window, ANE_CPU power word
0xfffffe0009599028: mov w2, #0xf          ; write32(PS+0x5a8, 0xf)
0xfffffe0009599034: mov w1, #0x5a8        ; poll: (word & 0xff) == 0xff
0xfffffe0009599038: mov w2, #0xff         ;   expected
0xfffffe000959903c: mov w3, #0xff         ;   mask
0xfffffe0009599040: mov w4, #0x1388       ;   tries
0xfffffe0009599048: bl 0xfffffe00095eb2d4 ; the poll helper
0xfffffe0009599050: ... ldrb [obj + 0x3ea0]; tbz bit 1 -> skip the next block
0xfffffe0009599060: ldr x0, [x20, #0x198] ; the PWGATE window
0xfffffe0009599078: mov w1, #0x199c       ; write32(PWGATE+0x199c, 0)
0xfffffe000959907c: mov w2, #0
0xfffffe0009599084: blraa x8, x16         ; vtable slot 0x18 = write32
0xfffffe000959908c: mov w1, #0x199c       ; poll: (word & 3) == 0
0xfffffe0009599090: mov w2, #0            ;   expected
0xfffffe0009599094: mov w3, #3            ;   mask
0xfffffe0009599098: mov w4, #0x1388       ;   tries
0xfffffe00095990a0: bl 0xfffffe00095eb2d4
```

The poll helper at `0xfffffe00095eb2d4` takes (object, offset, expected, mask, tries, log flag). It reads the 32 bit word through vtable slot `0x10` and masks it. It compares the result with the expected value. Between tries it waits 10 microseconds (`mov w0, #0xa`, then a delay call at `0xfffffe00095eb344`). Vtable slot `0x18` is write32 and slot `0x10` is read32. The same slots read and write the ANE SCRATCH words elsewhere in the kext.

So the kext writes 0 to PWGATE offset `0x199c`. It then waits until bits 1:0 read 0, for up to `0x1388` tries of about 10 microseconds. That is the whole claim behind offset `0x199c`, mask `3`, value `0` and the 5000 try poll in the module (about 50 ms).

## What is not decoded

- What sets bit 1 of the byte at `[obj + 0x3ea0]`. It gates this block and about 25 other sites. I read it as "this chip has a PWGATE window" but did not trace the store. If it is clear on a real T6031, macOS skips this step.
- What a closed gate means for the core. Nothing here shows that the gate is closed on a T6031 or that opening it makes the core run.
- The T6030 branch of the same function (type `0xc0`) does not touch the PWGATE window at all.
- Other PWGATE offsets (`0x2b8c`, `0x2e0`) appear in other chips' branches. The module never touches them.
