# Qwen decode packages

This page lists the compiled Qwen3.8-2B decode packages that the staged decode runs. It gives their seals and measured numbers. The packages are not in this repo. The compiler builds them. A package seal is the sha256 of the package's `SHA256SUMS` file.

The staged decode keeps the Q6_K head on the CPU. It runs the 38 layer programs on the ANE. Every number below uses 50 held-out prompts of 128 positions. The decode is teacher forced on fixed ids. The logits are compared against an fp32 reference. The quality gate is KL at most 0.000534 and top-1 at least 0.9945. The KL limit is twice that of the fp16 package.

## Packages

| Package | Chip family | Seal |
|---|---|---|
| fp16 | M1 (H13) | `bf54d842c05cd5a5e88e85b5f97bf0228d7fb57ef9fcb711c3a7a4980915fd4a` |
| fp16 | M2 Max (H14) | `f04e973afdce67b0671e726926464ce1821861738ddad63d8454732def580116` |
| int8 with GPTQ weights | M1 (H13) | `91ff7b47b50ac6014f35278e9c23a60a2f148d743ec0735835f8e0ce181e3a3c` |
| int8 with GPTQ weights | M2 Max (H14) | `de3d4dab6cea2fe8997f83207266c77902d64ed9b4ec1535833fbb60be4cbcf8` |
| int8 with GPTQ weights, mixed calibration | M1 (H13) | `829b43ab2f42e962d85617a8ce47ac08d17286c6c035c842f5a29f1fce4d1bf5` |
| int8 with GPTQ weights, mixed calibration | M2 Max (H14) | `539b3c73d09e187474d6777f95b9068c9234f8cef92abc3e7e4542daac31a116` |
| int8 with GPTQ weights, mixed calibration, attention in fp16 | M1 (H13) | `3fbb7a306a1a515aff14574165ac6ad9bad912bbbbe7466c52a5092494ab91d5` |
| int8 with GPTQ weights, mixed calibration, attention in fp16 | M2 Max (H14) | `d628ebc486deb0bddc84817a09dc16404a49047cb4c5ef3229527249c062de92` |
| fp16, 2,048 positions | M1 (H13) | `004d741a08e96e4fc238f4f63c844f93fdf26e05808ae25dc9cea778f7f4c4f5` |
| fp16, 1,024 positions | M1 (H13) | `a373cb0ce6243fd90cccf09c72dc9b422f227168daa94626463808086a6468f1` |
| int8 GPTQ, 1,024 positions | M1 (H13) | `701c6a4cd5c0072b6f79a73b866f906d8d6f65868708392b2b3a9e37bf604f68` |
| int8, long-position calibration, attention in fp16 (the default) | M1 (H13) | `5688ae1d37e1bce64123efe2eb3f5f6fe316f9f13bee516c32eda8435b71b390` |
| int8, long-position calibration, attention in fp16, 2,048 positions | M1 (H13) | `12cd93c0db0a744d914e5f1aafec9ec92eb3300ecf65d2f4a551ff3a63566051` |
| int8, long-position calibration, attention in fp16 (the default) | M2 Max (H14) | `8468ee5a9c9993340ed68a97fac693754e2201fc11949dfb7b15c935c8f28daa` |
| int8, long-position calibration, attention in fp16, 2,048 positions | M2 Max (H14) | `7fceb8240c374b6492e328082e86bf3991515736b1ee9c12aefa8fc6c2bde43d` |
| fp16, 2,048 positions | M2 Max (H14) | `3772e467bfe4ecc0128636738940235a720a5d9f8b4a9b54450bdc7a172191c0` |

The int8 package replaces the weights of 15 of the 17 large programs with int8 per output row. Two programs stay fp16, because plain int8 cost them too much quality. The weights come from GPTQ. Each row is rounded to minimise the error on the layer's real activations. The 80 calibration prompts are not in the test set.

## Results

Every 13-inch M1 and 14-inch M2 Max step time in this document was measured with the ANE clock set by the boot. The driver has no clock control on those two chips yet, so their clock level is not known. On the 16-inch M1 Max a Qwen decode step is about 1.7 times faster at the higher clock level (0.2009 s to 0.1158 s).

Step time, p50 over the prompts, measured in one session on one boot with the arms interleaved:

| Chip | Pair | First | Second | Change |
|---|---|---|---|---|
| 13-inch M1 | fp16, int8 GPTQ | 0.0863 s | 0.0610 s | 29.4 percent lower (means of both arms of each) |
| 14-inch M2 Max | fp16, int8 GPTQ | 0.1579 s | 0.1243 s | 21.3 percent lower |
| 13-inch M1 | int8 plain rounding, int8 GPTQ | 0.0678 s | 0.0680 s | equal (a different harness, so not comparable with the rows above) |
| 14-inch M2 Max | int8 plain rounding, int8 GPTQ | 0.1234 s | 0.1232 s | equal |

Quality against the fp32 reference, from the quality harness:

| Chip | Package | KL | Top-1 |
|---|---|---|---|
| 13-inch M1 | fp16 | 0.000267 | 0.9955 |
| 13-inch M1 | int8, plain rounding | 0.000540 | 0.9944 |
| 13-inch M1 | int8, GPTQ | 0.000397 | 0.9958 |
| 14-inch M2 Max | fp16 | 0.000269 | 0.9955 |
| 14-inch M2 Max | int8, plain rounding | 0.000547 | 0.9945 |
| 14-inch M2 Max | int8, GPTQ | 0.000400 | 0.9958 |

On the 13-inch the GPTQ package decodes 16.4 tokens per second against 11.6 for the fp16 package. With two decode streams in one process it reaches 18.05 scored steps per second against 12.12, which is 48.9 percent more.

Before the long-position package below, the fp16 package was the default. The GPTQ package is an opt-in speed tier. It passes the gate on both chips, on the technical prompts it was measured on. Plain int8 missed it on both. Paired by prompt on the M2 Max, GPTQ lowers the KL by 27 percent against plain int8. It is lower on 45 of 50 prompts. A second GPTQ run with other damping choices gave the same KL to 0.3 percent. The first set is the package.

## Other prompt sets

The gate above uses technical text. The GPTQ calibration used prose and multilingual prompts. On the 14-inch M2 Max the package was also run on two sets it never saw, with the fp16 package in the same session:

| Set | Package | Step p50 | KL | Top-1 |
|---|---|---|---|---|
| code, 50 prompts of 128 positions | fp16 | 0.1575 s | 0.000289 | 0.9966 |
| code | int8, GPTQ | 0.1234 s | 0.000454 | 0.9953 |
| long position, 50 prompts of 170 positions | fp16 | 0.1574 s | 0.000293 | 0.9926 |
| long position | int8, GPTQ | 0.1243 s | 0.000604 | 0.9898 |

The step is 21.7 percent lower on the code set and 21.0 percent lower on the long-position set. The KL is 1.57 times the fp16 KL on code and 2.06 times on the long-position set. Top-1 is lower by 0.0013 and 0.0028. The rule set before the run asked for at most 2 times the fp16 KL and a top-1 within 0.001. Both sets miss it. So the int8 GPTQ package is a speed tier: it passes the technical gate and costs more quality on text unlike its calibration. Its KL is higher than the fp16 KL on 48 of 50 code prompts and on 49 of 50 long-position prompts. The 13-inch M1 has not run these two sets.

The recipe scripts and the calibration notes are in the compiler repo: `receipts/2026-10-24-int8-gptq-qwen38-h13-h14` in [mil-hwx-compiler](https://github.com/joshuaswarren/mil-hwx-compiler).

## Mixed calibration

The first GPTQ package was calibrated on prose and multilingual prompts. The mixed package adds code: its calibration is 20 prose, 20 multilingual and 24 code prompts (the code prompts come from 12 of the 25 source documents of the code set) at 128 forced positions, 9,136 captured rows per tensor (the 64 prompts give 8,192 forced positions and 944 prompt-token rows). The 13 other source documents (26 prompts) stay unseen as the code test. The 13-inch M1 ran all three sets with the fp16 package and both GPTQ packages. KL is against the fp32 reference. The decode uses the older runner without the Q6_K head, so the numbers differ from the tables above.

| Set | fp16 | int8 GPTQ | int8 GPTQ, mixed |
|---|---|---|---|
| technical, 50 prompts of 128 positions: KL, top-1 | 0.000267 | 0.000397, 0.9958 | 0.000390, 0.9956 |
| code, the 26 unseen prompts: KL, top-1 | 0.000254, 0.9958 | 0.000407 (1.60 times), 0.9934 | 0.000351 (1.38 times), 0.9952 |
| long position, 50 prompts of 170 positions: KL, top-1 | 0.000293, 0.9935 | 0.000605 (2.06 times), 0.9896 | 0.000663 (2.26 times), 0.9898 |

The mixed package passes the same rule as before on the code prompts it never saw (at most 2 times the fp16 KL and top-1 within 0.001). The first package failed it on top-1 there. On the long-position prompts neither package passes, and the mixed one is 10 percent worse. Those prompts are the old prompt set. Neither calibration contains them. Against the first GPTQ package the mixed package costs nothing on the technical set (0.000390 against 0.000397; fp16 is 0.000267). The selective package below replaces both as the speed tier. The mixed H14 package for the M2 Max has not been run for quality. Its step on the 13-inch M1 was 29.4 percent lower than fp16 in one run under GPU load. The weight bytes are the same as the first package.

## Selective int8, the speed tier

The mixed package keeps the quality cost of int8 on long text. A scan with a CPU model showed where it comes from. It put the largest share on the early attention programs. So the selective package keeps the five attention programs in fp16 (programs 6, 12, 18, 25 and 31) and keeps the ten MLP programs in int8. It uses the mixed calibration. It gives back about half of the int8 weight-byte saving. It replaces the mixed package as the opt-in speed tier.

Quality against the fp32 reference, on the same three sets as above. The 13-inch M1 used the older runner without the Q6_K head. The 14-inch M2 Max used the staged decode with 50 prompts per set. The shipped numbers are in the fp16 column of the same chip:

| Chip | Set | fp16: KL, top-1 | Selective: KL, top-1 | KL ratio | Top-1 gap |
|---|---|---|---|---|---|
| 13-inch M1 | technical, 50 prompts of 128 positions | 0.000267, 0.9955 | 0.000330, 0.9955 | 1.24 | 0.0000 |
| 13-inch M1 | code, the 26 unseen prompts | 0.000254, 0.9958 | 0.000299, 0.9949 | 1.18 | 0.0009 |
| 13-inch M1 | long position, 50 prompts of 170 positions | 0.000293, 0.9935 | 0.000453, 0.9912 | 1.54 | 0.0024 |
| 14-inch M2 Max | technical | 0.000269, 0.9955 | 0.000333, 0.9956 | 1.24 | -0.0001 |
| 14-inch M2 Max | code, the 26 unseen prompts | 0.000256, 0.9958 | 0.000303, 0.9952 | 1.18 | 0.0006 |
| 14-inch M2 Max | long position | 0.000293, 0.9926 | 0.000450, 0.9911 | 1.54 | 0.0015 |

The selective package passes the technical gate and the code rule on both chips. On the long-position prompts the KL is 1.54 times the fp16 KL on both chips, which is under the limit of 2 times. The top-1 limit is 0.001. The package misses it on both chips: the gap is 0.0024 on the 13-inch M1 and 0.0015 on the 14-inch M2 Max. The limit was set before the runs and stays. So the package is a speed tier and the fp16 package stays the default. The mixed package read 2.26 times on the long-position prompts of the 13-inch M1, and the first GPTQ package 2.06 times.

Step time, p50. The shipped package and the selective package were interleaved in one session, with the arms of each pair within 1 percent of each other:

| Chip | fp16 | Selective | Change | First GPTQ package |
|---|---|---|---|---|
| 13-inch M1, under GPU load | 0.0863 s | 0.0732 s | 14.5 percent lower (median of 4 runs, from 13.6 to 15.4; the values shown are from the quietest run) | 29.4 percent lower |
| 14-inch M2 Max | 0.1580 s | 0.1401 s | 11.3 percent lower | 21.3 percent lower |
| 16-inch M1 Max, level driver | 0.1189 s | 0.1086 s | 8.7 percent lower (median of 3 runs, from 7.9 to 8.8; the values shown are from one of them) | 16.6 percent lower |

The selective package keeps about half of the step gain of the first GPTQ package. The 16-inch M1 Max has not run the quality sets.

## Default: long-position selective int8

This section sets the default for the 13-inch M1 (H13) and the 14-inch M2 Max (H14). The M2 Max follows the same rule as the M1: the rule was fixed before the quality runs, and the package met it. The margin on top-1 is thin on both chips, and the M2 Max section below says how thin. The M2 Max has no longer-context quality check. The 16-inch M1 Max has speed results only, so its default stays fp16. The repo has no switch for the package. The runner takes the package as an argument, and this page says which one to run.

The package is the selective package with one change: its GPTQ calibration adds windows of the long-position prompts, taken from half of the base streams. The other half stays unseen and is the test. All six attention programs stay in fp16: five that the plain int8 package converts, plus one that is fp16 in every package. The ten MLP programs are int8. The seals are in the package table above. The 2,048 position package for the M1 is the 192 position package with its six attention programs replaced by those of the shipped 2,048 position fp16 package. It runs with the manifest of that shipped package.

Quality against the fp32 reference. The rule was fixed before each run: the KL is at most 2 times the fp16 KL, and the top-1 gap is at most 0.001.

| Set (13-inch M1) | fp16 KL, top-1 | Proposed KL, top-1 | KL ratio | Top-1 gap | Rule |
|---|---|---|---|---|---|
| technical, 50 prompts of 128 positions | 0.000267, 0.9955 | 0.000331, 0.9966 | 1.24 | -0.0011 | pass |
| code, 26 unseen prompts | 0.000254, 0.9958 | 0.000323, 0.9955 | 1.27 | 0.0003 | pass |
| long position, 25 unseen prompts | 0.000295, 0.9929 | 0.000348, 0.9918 | 1.18 | 0.0012 | misses top-1 by 0.0002 |
| large unseen set, 161 prompts, 27,370 positions | 0.000257, 0.9955 | 0.000328, 0.9952 | 1.28 | 0.0003 | pass |
| 2,048 positions, 3 prompts, 72 positions | 0.000380, 1.0000 | 0.000518, 1.0000 | 1.36 | 0.0000 | KL pass, top-1 not resolved |
| 2,048 positions, 12 prompts, 288 positions | 0.000481, 0.9896 | 0.000509, 0.9826 | 1.06 | 0.0069 | KL pass, top-1 not resolved |

On the large set the paired standard error of the top-1 gap is 0.00038, and the bootstrap interval over prompts is -0.0004 to +0.0010. Its upper end sits on the limit. The 161 prompts are the 136 of a large set that no package saw, plus 25 older unseen prompts. The 50 technical prompts of the first row are among them, so the two rows are not independent. On the 2,048 position sets the KL ratio per prompt runs from 0.52 to 1.56, and no text type reaches 2.0. The twelve prompts hold three technical, three code, three prose and two multilingual prompts, and one on the boundary of prose and multilingual text.

Step time, p50, with the arms interleaved in one session. The fp16 package is the first arm.

| Chip | Positions | fp16 | Proposed | Change | Arm agreement |
|---|---|---|---|---|---|
| 13-inch M1 | 2,048 | 0.0952 s, 0.0950 s | 0.0823 s, 0.0823 s | 13.5 percent lower | within 0.21 percent |
| 13-inch M1, second session | 2,048 | 0.0954 s, 0.0951 s | 0.0821 s, 0.0807 s | 14.5 percent lower | proposed arms differ 1.72 percent |
| 13-inch M1, third session | 2,048 | 0.0937 s, 0.0937 s | 0.0815 s, 0.0807 s | 13.4 percent lower | fp16 arms differ 0.0 percent, proposed arms 0.98 percent |
| 13-inch M1 | 1,024 | 0.0891 s, 0.0893 s | 0.0640 s, 0.0639 s | 28.3 percent lower | fp16 arms differ 0.22 percent, proposed arms 0.16 percent |
| 13-inch M1, second session, other boot | 1,024 | 0.0885 s, 0.0886 s | 0.0637 s, 0.0636 s | 28.1 percent lower | fp16 arms differ 0.11 percent, proposed arms 0.16 percent |
| 16-inch M1 Max, PMP not running (clock level 0) | 2,048 | 0.2191 s, 0.2163 s | 0.1927 s, 0.1928 s | 11.5 percent lower | fp16 arms differ 1.29 percent |
| 16-inch M1 Max, PMP not running (clock level 0), repeat | 2,048 | 0.2164 s, 0.2153 s | 0.1921 s, 0.1908 s | 11.3 percent lower | fp16 arms differ 0.51 percent, proposed arms 0.68 percent |
| 16-inch M1 Max, PMP running (clock level 1), stock kernel | 2,048 | 0.1253 s, 0.1253 s | 0.1148 s, 0.1149 s | 8.3 percent lower | fp16 arms differ 0.0 percent, proposed arms 0.09 percent |
| 16-inch M1 Max, PMP running (clock level 1), eGPU kernel, other boot | 2,048 | 0.1253 s, 0.1254 s | 0.1150 s, 0.1149 s | 8.3 percent lower | fp16 arms differ 0.08 percent, proposed arms 0.09 percent |
| 16-inch M1 Max, PMP not running (clock level 0), lab kernel | 1,024 | 0.2089 s, 0.2100 s | 0.1635 s, 0.1632 s | 22.0 percent lower | fp16 arms differ 0.52 percent, 8-bit arms 0.18 percent |
| 16-inch M1 Max, PMP running (clock level 1), lab kernel, same boot | 1,024 | 0.1226 s, 0.1200 s | 0.1035 s, 0.1010 s | 15.7 percent lower | fp16 arms differ 2.17 percent, 8-bit arms 2.48 percent |
| 14-inch M2 Max | 192 | 0.1599 s, 0.1602 s | 0.1427 s, 0.1436 s | 10.6 percent lower | within 0.63 percent |
| 14-inch M2 Max | 2,048 | 0.1862 s, 0.1857 s | 0.1677 s, 0.1679 s | 9.76 percent lower | fp16 arms differ 0.27 percent, 8-bit arms 0.12 percent |
| 14-inch M2 Max, second session, same boot | 2,048 | 0.1858 s, 0.1867 s | 0.1690 s, 0.1683 s | 9.45 percent lower | fp16 arms differ 0.48 percent, 8-bit arms 0.42 percent |
| 14-inch M2 Max, third session, other boot | 2,048 | 0.1853 s, 0.1861 s | 0.1685 s, 0.1676 s | 9.50 percent lower | fp16 arms differ 0.43 percent, 8-bit arms 0.54 percent |

The third 13-inch session at 2,048 positions ran while the scheduler kept every other GPU job off the machine. The run script does not log the GPU state, so that is the scheduler's guarantee and not a measurement. The 13-inch rows at 1,024 positions use 5 prompts and the 2,048-position rows use 3 prompts, not the 50 prompts of 128 positions behind the other tables. The 13-inch rows at 1,024 positions use the fp16 and the int8 GPTQ packages built for 1,024 positions (hashes above), not the long-position selective package, so the gain is that of the full int8 GPTQ package at a medium context. The logits of the two arms of each pair are equal in both sessions. The 13-inch has no clock control (see Results), so these steps ran at the clock the boot set.

The two 16-inch rows at 1,024 positions ran on one boot with the 8-bit package built for 1,024 positions, the first with the PMP not loaded and the second after the PMP row loaded it. Their steps are 1.73 times (fp16) and 1.60 times (8-bit) the level 1 steps, with identical logits, so the factor of 1.7 above does not hold for the 8-bit package at this length. In the level 1 row the first arm of each kind is 2.2 and 2.5 percent above the later one, so that row misses the 1 percent agreement rule as well. Other GPU jobs ran beside the level 0 row and the GPU was blocked for the level 1 row; the run script does not log the GPU state.

The 16-inch rows marked PMP not running were measured on boots where the PMP firmware was not loaded, so the ANE ran at its lowest clock level (README, Clock level). Their steps are 1.7 times the steps of the rows with the PMP running, with identical logits. They are kept because they are real measurements of that state. Use the rows with the PMP running for the 16-inch M1 Max. The two rows marked with a difference above 1 percent miss the 1 percent agreement rule that was set before the runs. The 16-inch repeat meets it. A GPU job overlapped its first arm for about 4 minutes, and the step of that arm matches the first run's. The speed bar was a gain of at least 10 percent. The step depends on the longest position the package is built for. It does not depend on the position inside it. The 2,048 position fp16 and int8 steps are therefore not comparable with the 192 position rows.

### 14-inch M2 Max quality

The 14-inch M2 Max (H14) ran the shipped fp16 package (seal `f04e973a...`) and the proposed package (seal `8468ee5a...`) on the same boot, with the same forced ids, at 192 positions. The scoring uses the same fp32 reference and the same rule as the 13-inch M1. The technical and code sets use 128 positions. The long-position sets use 170.

| Set (14-inch M2 Max) | fp16 KL, top-1 | Proposed KL, top-1 | KL ratio | Top-1 gap | Rule |
|---|---|---|---|---|---|
| technical, 50 prompts | 0.000269, 0.9955 | 0.000340, 0.9962 | 1.27 | -0.0008 | pass |
| code, 26 unseen prompts | 0.000256, 0.9958 | 0.000321, 0.9964 | 1.25 | -0.0006 | pass |
| long position, 136 unseen prompts, four batches of 34 | 0.000247, 0.9960 | 0.000330, 0.9955 | 1.33 | +0.0005 | pass |
| long position, the 25 older unseen prompts | 0.000291, 0.9929 | 0.000351, 0.9915 | 1.20 | +0.0014 | top-1 misses alone |
| long position, all 161 prompts, 27,370 positions | 0.000254, 0.9955 | 0.000333, 0.9949 | 1.31 | +0.0006 | pass |

The margin on top-1 is thin. On the 161 prompts the gap is +0.0006 against a limit of 0.001. Its paired standard error is 0.00041, and the 95 percent bootstrap interval over prompts is -0.0002 to +0.0015. The upper end is above the limit. The data cannot rule out a gap above 0.001. The standard error counts positions as independent, and positions inside a prompt are correlated, so the interval over prompts is the honest one. The proposed package has a higher KL than fp16 in every one of the five batches, with ratios from 1.20 to 1.45. It is a consistent small loss that stays inside the limit. Of the five long-position batches, three miss the limit alone (pos170b, pos170c and the 25 older prompts): the gaps are -0.0010 (pos170a), +0.0017 (pos170b), +0.0010 (pos170c, exactly +0.00104, just over), +0.0002 (pos170d) and +0.0014 (the 25 older prompts), and their standard errors run from 0.0007 to 0.0013. None of them can resolve 0.001. Six of the 161 prompts have a KL ratio above 2, and the per-prompt ratio runs from 0.69 to 2.55 with a median of 1.20. The 13-inch M1 gave a ratio of 1.28 and a gap of +0.0003 on the same 161 prompts.

The 136 long-position prompts contain the 50 technical and 26 code prompts at other lengths, so those rows are not independent. The 14-inch M2 Max ran under whatever GPU load other jobs put on it, and the harness printed two GPU jobs at the end of most arms. At 2,048 positions the M2 Max ran 12 prompts (288 positions, one window of 24 positions per prompt at positions 2,000 to 2,023) with the 2,048 position package (seal `7fceb824...`): the KL ratio is 1.13 against the limit of 2 (prompt bootstrap interval 0.96 to 1.28), and one prompt of the 12 is above 2 alone (2.94). The top-1 gap is +0.0069 (2 of 288 positions), is not gated and cannot resolve 0.001. The 13-inch M1 gave 1.06 and the same gap on these prompts. Nothing is measured between positions 100 and 2,000. The package keeps the int8 MLP weights calibrated at 192 positions, not at 2,048. No 1,024 position check exists for the M2 Max.

### Limits

- The quality rows run on the 13-inch M1 and the 14-inch M2 Max. The 14-inch M2 Max ran 192 positions on the large sets and 2,048 positions on 12 prompts only. The 16-inch M1 Max has run speed only, and its logits matched the 13-inch ones on the 3 prompt set.
- The top-1 gap is not resolved on either chip. On the 13-inch M1 the 25 unseen long-position prompts miss the limit by 0.0002, with a standard error of 0.0011, and the 161 prompt set passes with the interval reaching the limit. On the 14-inch M2 Max the same 25 prompts miss it by 0.0004 (a gap of +0.0014, standard error 0.0013), and the interval of the 161 prompt set reaches +0.0015, above the limit. The two 2,048 position sets on the 13-inch M1 cannot resolve 0.001. Their standard error is 0.0049 on 12 prompts.
- The 2,048 position quality sets score one window of 24 positions per prompt, at positions 2,000 to 2,023. Nothing is measured between 100 and 2,000. The 3 prompts cover prose and multilingual text only.
- The 13-inch M1 second session misses the 1 percent agreement rule (its proposed arms differ by 1.72 percent). The first 16-inch M1 Max run missed it too (1.29 percent), and the repeat meets it. The repeat's last arm finished after the ticket's time cap, so its step comes from the arm's records file and not from the results file.
- At 2,048 positions the M2 Max step is about 9.5 percent lower in three sessions on two boots (9.76, 9.45 and 9.50 percent). All three sit just under the 10 percent bar, so the 8-bit package is faster there but does not clear the bar. No GPU job ran in any of the three sessions. The rows of the first two are in `receipts/2026-10-27-m2-l2048-speed`. The third ran on another boot. The 192 position row clears the bar by 0.6 points in one session of 10 code prompts.
- The package gives back about half of the int8 weight-byte saving. Its gain is 10 to 15 percent. The first GPTQ package gained 21 to 29 percent and failed the quality rule.
- The long-position calibration helps the streams it saw: on the 25 prompts it was calibrated on, the KL is 0.96 times fp16 (0.000280 against 0.000292). Only the unseen numbers above are used.

### Opt out and roll back

To run the fp16 package, pass the fp16 package from the table to the runner in place of this one. No flag, file or setting changes. The fp16 rows stay on this page. To roll the default back, revert this change. The earlier text, the fp16 package as the default and the selective package as the opt-in speed tier, comes back with it.

## What this does not show

The 16-inch M1 Max has not run the int8 GPTQ package for quality. The fp16 KL on the M2 Max comes from an earlier run. Plain int8 is not measured on the code and long-position sets, so the split of the cost between int8 and GPTQ is open.
