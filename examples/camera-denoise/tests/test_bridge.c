/* SPDX-License-Identifier: MIT */
#include "camera_ane_backend.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <string.h>

int main(void)
{
    unsigned char input[50], output[60];
    const struct camera_ane_backend *backend = &camera_ane_backend_v1;
    struct camera_ane_plane source = {input, 10, sizeof input};
    struct camera_ane_plane destination = {output, 12, sizeof output};
    void *context;
    memset(input, 127, sizeof input);
    memset(output, 73, sizeof output);
    assert(!backend->init(7, 5, 0));
    context = backend->init(7, 5, CAMERA_ANE_GRAY8);
    assert(context);
    assert(backend->process(context, &source, NULL, NAN, &destination) == -EINVAL);
    for (unsigned i = 0; i < sizeof output; ++i)
        assert(output[i] == 73);
    assert(!backend->process(context, &source, NULL, 0, &destination));
    for (unsigned y = 0; y < 5; ++y) {
        assert(!memcmp(input + y * 10, output + y * 12, 7));
        for (unsigned x = 7; x < 12; ++x)
            assert(output[y * 12 + x] == 73);
    }
    assert(!backend->process(context, &source, NULL, 1, &destination));
    assert(!backend->process(context, &source, NULL, 1, &source));
    source.bytes = 46;
    assert(backend->process(context, &source, NULL, 1, &destination) == -EINVAL);
    source.bytes = sizeof input;
    source.stride = SIZE_MAX;
    assert(backend->process(context, &source, NULL, 1, &destination) == -EINVAL);
    backend->fini(context);
    backend->fini(NULL);
    return 0;
}
