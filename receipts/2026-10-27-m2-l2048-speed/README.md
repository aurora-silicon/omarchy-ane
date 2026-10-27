# 14-inch M2 Max: 2,048 position Qwen decode speed, fp16 against the 8-bit default

Two sessions on one boot of the 14-inch M2 Max. Each session ran four arms in the order fp16, 8-bit, 8-bit, fp16. The 8-bit arm is the long-position int8 package for 2,048 positions. The fp16 arm is the shipped fp16 package for 2,048 positions. Each arm ran 3 prompts of 2,000 prompt tokens plus 24 forced positions.

The metric is the median of the three per-prompt p50 step times of an arm. The change is the 8-bit step against the mean of the two fp16 steps. The bar set before the runs was a change of at least 10 percent, both fp16 arms within 1 percent of each other, both 8-bit arms within 1 percent of each other, and no GPU job at any arm start or end.

| Session | fp16 arms (s) | 8-bit arms (s) | Change | fp16 arms differ | 8-bit arms differ |
|---|---|---|---|---|---|
| 1 | 0.1862, 0.1857 | 0.1677, 0.1679 | 9.76 percent lower | 0.27 percent | 0.12 percent |
| 2 | 0.1858, 0.1867 | 0.1690, 0.1683 | 9.45 percent lower | 0.48 percent | 0.42 percent |

No GPU job ran at any arm start or end. Every arm ended with exit code 0 and no fault lines.

Both sessions pass the agreement and GPU gates and both land just under the 10 percent bar. The 8-bit package is about 9.5 percent faster at 2,048 positions on this chip. Nothing here supports a 10 percent claim. Both sessions share one boot, so a session on another boot would be the independent check.

Files: `session1` and `session2` hold the four per-arm result files and the output of the computation for each session.
