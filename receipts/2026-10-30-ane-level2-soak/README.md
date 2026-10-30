# ANE level 2 soak, 16-inch M1 Max, 2026-10-30

One 30 minute run of the Parakeet whole encoder at ANE clock level 2.
It ran in blocks of 250 calls, 7,500 calls in all.
Each block compared the output with the gold output. It recorded the median and the largest call time.
It also recorded the highest reading of the machine's temperature sensors.
The run stops above 85 C.

Results:

- Every block was bit-exact against the gold (the first 240,000 words).
- Block medians ran from 240.379 to 240.586 ms, a spread of 0.09 percent.
- The first 10 blocks averaged 240.476 ms and the last 10 blocks 240.526 ms.
- The slowest call took 241.805 ms.
- The highest sensor reading was 49.1 C. The sensors are NAND, battery hotspot, charge regulator and WiFi/BT. None is the ANE.
- The kernel log shows no fault.

Other GPU work ran beside blocks 4 to 12, about 8 minutes.
The median of those blocks was 240.516 ms.
The three blocks before them gave 240.483 ms, and the 18 blocks after them 240.532 ms.
There is no difference.

The level 2 module was built from the same source with the level constant set to 2.
The shipped module was restored afterwards.
On the same boot, levels 0, 1 and 2 gave 438.75, 289.68 and 240.57 ms.

`soak.csv` has one row per block.
Its columns are the block number, the start time, the median and largest call time in ms, whether the output matched,
the ANE busy time, and the highest sensor reading in millidegrees.
