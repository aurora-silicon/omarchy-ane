/* SPDX-License-Identifier: GPL-2.0-only OR MIT
 *
 * Offline test of ane_pwgate_plan() (ane/t6021/ane_pwgate.h): what the T8112
 * probe may do about a closed ANE power gate. The kernel probe and this test
 * share the header.
 *
 * Build and run:
 *   gcc -Wall -Wextra -O2 -I ane/t6021 -o test_ane_pwgate tools/test_ane_pwgate.c
 *   ./test_ane_pwgate
 * Exit 0 = every case behaved as asserted.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef uint32_t u32;

#include "ane_pwgate.h"

static int failures;

static void expect(const char *what, u32 gate, bool opt_in, enum ane_pwgate_plan want)
{
	enum ane_pwgate_plan got = ane_pwgate_plan(gate, opt_in);

	if (got != want) {
		printf("FAIL %s: gate 0x%08x opt_in %d: got %d want %d\n", what, gate, opt_in, got, want);
		failures++;
	}
}

int main(void)
{
	/* Open gate: never a write, with or without the parameter. */
	expect("open", 0x00000000, false, ANE_PWGATE_OPEN);
	expect("open, opt in", 0x00000000, true, ANE_PWGATE_OPEN);
	expect("other bits set, gate bits clear", 0x0fffffff, true, ANE_PWGATE_OPEN);

	/* The exact closed word the boot firmware leaves: write only on opt in. */
	expect("closed, no opt in", 0x30000000, false, ANE_PWGATE_REFUSE);
	expect("closed, opt in", 0x30000000, true, ANE_PWGATE_WRITE);

	/* Any other closed word refuses, even with the parameter. */
	expect("half closed 28", 0x10000000, true, ANE_PWGATE_REFUSE);
	expect("half closed 29", 0x20000000, true, ANE_PWGATE_REFUSE);
	expect("closed plus low bit", 0x30000001, true, ANE_PWGATE_REFUSE);
	expect("closed plus high bit", 0xb0000000, true, ANE_PWGATE_REFUSE);
	expect("unmapped window", 0xffffffff, true, ANE_PWGATE_REFUSE);
	expect("unmapped window, no opt in", 0xffffffff, false, ANE_PWGATE_REFUSE);

	if (failures) {
		printf("test_ane_pwgate: %d failures\n", failures);
		return 1;
	}
	printf("test_ane_pwgate: ok\n");
	return 0;
}
