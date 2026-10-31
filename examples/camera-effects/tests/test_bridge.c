/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include "../camera_effects.h"
#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv)
{
	assert(argc==2);
	/* Match the filter: the executable does not link the plugin/libpython. */
	void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE);
	assert(library);
	const struct ce_ops *(*entry)(void);
	*(void **)(&entry)=dlsym(library,CE_ENTRY);
	assert(entry);
	const struct ce_ops *ops=entry();
	assert(ops && ops->abi==CE_ABI);
	struct ce_config cfg={.abi=CE_ABI,.width=64,.height=32,.matrix=601,
						  .zoom_min=1,.zoom_max=6,.zoom_default=1.5};
	size_t bytes=64*32*3/2;
	unsigned char *in=malloc(bytes),*out=malloc(bytes);
	assert(in && out);
	for (size_t i=0;i<bytes;i++) in[i]=(unsigned char)(i%251);
	void *off=ops->create(&cfg);
	assert(off && ops->process(off,in,out,1000000)==0 && !memcmp(in,out,bytes));
	assert(ops->process(off,out,out,2000000)==0 && !memcmp(in,out,bytes));
	ops->destroy(off);
	cfg.background=1;cfg.framing=1;
	void *effects=ops->create(&cfg);
	assert(effects && !ops->process(effects,in,out,1000000000));
	assert(memcmp(in,out,bytes));
	ops->reset(effects);
	ops->destroy(effects);
	cfg.zoom=NAN; assert(!ops->create(&cfg));
	cfg.zoom=0;cfg.width=63;assert(!ops->create(&cfg));
	free(in);free(out);
	return 0;
}
