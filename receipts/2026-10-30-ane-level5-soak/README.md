# ANE level 5 soak and the level ladder, 16-inch M1 Max, 2026-10-30

Three runs on one 16-inch M1 Max, each with a driver build whose level maximum was raised to 5 and the
`dvfs_level` parameter set. The shipped module was restored after each run.

Soak (`soak.csv`): the Parakeet whole encoder at level 5 for 30 minutes, in blocks of 250 calls, 9,000 calls in all.
Each block compared the output with the gold output.

- Every one of the 36 blocks was bit-exact.
- Block medians ran from 198.971 to 199.052 ms, a spread of 0.04 percent. The first 5 blocks and the last 5 differ by 0.006 percent.
- The highest sensor reading was 49.0 C. The sensors are NAND, battery hotspot, charge regulator and WiFi/BT. None is the ANE.
- The kernel log shows no fault. The driver logged each level it asked for.
- The run stops above 70 C, on a block that is not bit-exact, or when the driver does not confirm the level.

Ladder (`levels.csv`): the same encoder for 5 minutes at each of levels 3, 4 and 5 (6 blocks each), all bit-exact:
216.93, 206.46 and 199.04 ms. Level 0, 1 and 2 on the same boot gave 438.75, 289.68 and 240.57 ms.

Qwen decode (`qwen-decode.csv`): the two-arm decode cell at 1,024 positions, 5 prompts of 24 forced tokens, at levels 2 and 5.
The step time (median over prompts) with the fp16 packages was 0.1189 s at level 2 and 0.1184 s at level 5, against 0.1200 s at level 1.
With the S1-GPTQ packages it was 0.0836 s and 0.0787 s, against 0.1010 s at level 1. The logits were identical to the level 1 run in every prompt.

The machine had no other work on it during these runs.
