/* SPDX-License-Identifier: MIT */
#ifndef CAMERA_ANE_BACKEND_H
#define CAMERA_ANE_BACKEND_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Provisional integration seam until the filter publishes its backend ABI.
 * GRAY8 means one full-range luma plane; the filter retains chroma ownership.
 * One context per stream. Use a worker thread, never PipeWire's RT callback.
 */
enum camera_ane_format { CAMERA_ANE_GRAY8 = 1 };
struct camera_ane_plane {
    uint8_t *data;
    size_t stride;
    size_t bytes;
};
struct camera_ane_backend {
    void *(*init)(unsigned width, unsigned height, enum camera_ane_format format);
    /* prev_state is reserved, never read or changed by this spatial model.
     * strength is finite in [0,1]. Returns 0 or negative errno. Error leaves
     * output untouched; in-place and padded rows are supported. */
    int (*process)(void *context, const struct camera_ane_plane *in_frame,
                   const void *prev_state, float strength,
                   struct camera_ane_plane *out_frame);
    void (*fini)(void *context);
};
extern const struct camera_ane_backend camera_ane_backend_v1;

#ifdef __cplusplus
}
#endif
#endif
