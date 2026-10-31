// SPDX-License-Identifier: MIT
/* Optional inference fallback policy; execution remains in existing backends. */
#define _POSIX_C_SOURCE 200809L
#include "ane.h"
#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

void ane_fallback_init(struct ane_fallback_state *s, int native_ready)
{
	if (!s)
		return;
	memset(s, 0, sizeof(*s));
	s->native_ready = native_ready != 0;
	s->reason = native_ready ? ANE_FALLBACK_NONE : ANE_FALLBACK_UNAVAILABLE;
}

int ane_fallback_run(struct ane_fallback_state *s, ane_infer_fn native,
		     ane_infer_fn cpu, void *context,
		     struct ane_fallback_call *call)
{
	struct ane_fallback_call c = {0};
	uint64_t start;
	int rc;

	if (!s || !cpu || (s->native_ready && !native))
		return -EINVAL;
	start = now_ns();
	if (s->native_ready) {
		uint64_t begin = now_ns();

		rc = native(context);
		c.native_ns = now_ns() - begin;
		if (!rc) {
			c.backend = ANE_INFER_NATIVE;
		} else {
			s->native_ready = 0;
			s->reason = ANE_FALLBACK_NATIVE_FAILED;
			s->native_error = rc < 0 ? rc : -EIO;
		}
	}
	c.reason = s->reason;
	c.native_error = s->native_error;
	if (!c.backend)
		rc = cpu(context);
	c.status = rc > 0 ? -EIO : rc;
	c.latency_ns = now_ns() - start;
	if (!c.status) {
		s->calls++;
		s->native_calls += c.backend == ANE_INFER_NATIVE;
		s->cpu_calls += c.backend == ANE_INFER_CPU;
		s->last_ns = c.latency_ns;
		s->total_ns += c.latency_ns;
		if (!s->min_ns || c.latency_ns < s->min_ns)
			s->min_ns = c.latency_ns;
		if (c.latency_ns > s->max_ns)
			s->max_ns = c.latency_ns;
	} else {
		s->failed_calls++;
	}
	if (call)
		*call = c;
	return c.status;
}
