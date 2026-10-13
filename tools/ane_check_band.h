/* SPDX-License-Identifier: MIT */
/* Copyright 2026 Joshua Warren */
/*
 * Per-lane acceptance for ane-run --check matvec and --check rms.
 *
 * A lane is in band when the device value is within 2 ulp of the rounded exact value, or within 4 condition units, where one
 * condition unit is 2^-11 * sum|terms|: the fp16 rounding error of one partial sum.
 *
 * A lane whose sum of absolute terms is 0 has an exact value of 0 and no scale to measure against. Dividing by it would turn every
 * error into 0 condition units and pass any output, including all-ones or NaN. Such a lane passes only on an exact match.
 */
#ifndef ANE_CHECK_BAND_H
#define ANE_CHECK_BAND_H

#include <math.h>

/* err = |device - exact| in value units; diff = |device - rounded exact| in ulp of the rounded exact value. */
static inline int ane_lane_in_band(double diff, double err, double sumabs)
{
	if (sumabs == 0.0) {
		return err == 0.0;
	}
	return diff <= 2.0 || err / (sumabs * 0x1p-11) <= 4.0;
}

/* The reported error of a lane in condition units; a lane with no scale and a nonzero or non-finite error is infinitely bad. */
static inline double ane_cond_units(double err, double sumabs)
{
	if (sumabs > 0.0) {
		return err / (sumabs * 0x1p-11);
	}
	return err == 0.0 ? 0.0 : INFINITY;
}

#endif
