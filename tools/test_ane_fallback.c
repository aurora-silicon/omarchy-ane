// SPDX-License-Identifier: MIT
#include "ane.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>

struct work { int native_calls, cpu_calls, native_status, cpu_status, value; };
static int native(void *context)
{
	struct work *w = context;
	w->native_calls++;
	if (!w->native_status)
		w->value = 42;
	return w->native_status;
}
static int cpu(void *context)
{
	struct work *w = context;
	w->cpu_calls++;
	if (!w->cpu_status)
		w->value = 42;
	return w->cpu_status;
}
int main(void)
{
	struct ane_fallback_state s;
	struct ane_fallback_call c;
	struct work w = {0};

	ane_fallback_init(&s, 0);
	assert(!ane_fallback_run(&s, NULL, cpu, &w, &c));
	assert(c.backend == ANE_INFER_CPU && c.reason == ANE_FALLBACK_UNAVAILABLE);
	assert(w.value == 42 && w.native_calls == 0 && s.cpu_calls == 1);
	ane_fallback_init(&s, 1);
	assert(!ane_fallback_run(&s, native, cpu, &w, &c));
	assert(c.backend == ANE_INFER_NATIVE && s.native_calls == 1);
	w.native_status = -EBUSY;
	assert(!ane_fallback_run(&s, native, cpu, &w, &c));
	assert(c.backend == ANE_INFER_CPU && c.native_error == -EBUSY);
	assert(s.reason == ANE_FALLBACK_NATIVE_FAILED && !s.native_ready);
	assert(!ane_fallback_run(&s, native, cpu, &w, &c));
	assert(w.native_calls == 2 && s.calls == 3 && s.cpu_calls == 2);
	assert(s.min_ns <= s.max_ns && s.total_ns >= s.last_ns);
	w.cpu_status = -EINVAL;
	assert(ane_fallback_run(&s, native, cpu, &w, &c) == -EINVAL);
	assert(c.status == -EINVAL && s.failed_calls == 1 && s.calls == 3);
	assert(ane_fallback_run(NULL, native, cpu, &w, &c) == -EINVAL);
	ane_fallback_init(&s, 1);
	assert(ane_fallback_run(&s, NULL, cpu, &w, &c) == -EINVAL);
	assert(ane_fallback_run(&s, native, NULL, &w, &c) == -EINVAL);
	puts("libane fallback: PASS");
	return 0;
}
