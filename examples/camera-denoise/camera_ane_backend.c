/* SPDX-License-Identifier: MIT */
/* Thin CPython embedding bridge. Model execution stays in the existing
 * libane binding and ONNX Runtime. The process owns the Python interpreter.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "camera_ane_backend.h"

#ifndef CAMERA_ANE_PYTHON_DIR
#define CAMERA_ANE_PYTHON_DIR "/usr/lib/aurora-camera-ane"
#endif
#ifndef CAMERA_ANE_PYTHON_EXECUTABLE
#define CAMERA_ANE_PYTHON_EXECUTABLE "/usr/bin/python3"
#endif

struct context {
    unsigned width, height;
    PyObject *backend;
    pthread_mutex_t lock;
};
static pthread_once_t interpreter_once = PTHREAD_ONCE_INIT;
static int interpreter_error;

static void start_interpreter(void)
{
    if (!Py_IsInitialized()) {
        PyConfig config;
        PyStatus status;
        PyConfig_InitPythonConfig(&config);
        config.install_signal_handlers = 0;
        status = PyConfig_SetBytesString(&config, &config.program_name,
                                        CAMERA_ANE_PYTHON_EXECUTABLE);
        if (!PyStatus_Exception(status))
            status = Py_InitializeFromConfig(&config);
        PyConfig_Clear(&config);
        if (PyStatus_Exception(status)) {
            interpreter_error = 1;
            return;
        }
        PyEval_SaveThread();
    }
}

static void *backend_init(unsigned width, unsigned height, enum camera_ane_format format)
{
    struct context *context;
    PyGILState_STATE gil;
    PyObject *path, *directory, *module = NULL;
    if (!width || !height || width > 4096 || height > 2160 || format != CAMERA_ANE_GRAY8) {
        errno = EINVAL;
        return NULL;
    }
    context = calloc(1, sizeof(*context));
    if (!context)
        return NULL;
    if (pthread_mutex_init(&context->lock, NULL)) {
        free(context);
        errno = ENOMEM;
        return NULL;
    }
    pthread_once(&interpreter_once, start_interpreter);
    if (interpreter_error) {
        pthread_mutex_destroy(&context->lock);
        free(context);
        errno = EIO;
        return NULL;
    }
    gil = PyGILState_Ensure();
    path = PySys_GetObject("path"); /* borrowed */
    directory = PyUnicode_FromString(CAMERA_ANE_PYTHON_DIR);
    if (!path || !directory || PyList_Insert(path, 0, directory))
        goto done;
    module = PyImport_ImportModule("aurora_camera_ane.backend");
    if (module)
        context->backend = PyObject_CallMethod(module, "init", "IIs", width, height, "GRAY8");
done:
    if (!context->backend && PyErr_Occurred())
        PyErr_Print();
    Py_XDECREF(module);
    Py_XDECREF(directory);
    PyGILState_Release(gil);
    if (!context->backend) {
        pthread_mutex_destroy(&context->lock);
        free(context);
        errno = EIO;
        return NULL;
    }
    context->width = width;
    context->height = height;
    return context;
}

static int valid_plane(const struct context *context, const struct camera_ane_plane *plane)
{
    size_t rows = context->height - 1;
    return plane && plane->data && plane->stride >= context->width &&
        plane->stride <= (SIZE_MAX - context->width) / (rows ? rows : 1) &&
        plane->bytes >= rows * plane->stride + context->width;
}

static int backend_process(void *opaque, const struct camera_ane_plane *source,
                           const void *prev_state, float strength,
                           struct camera_ane_plane *destination)
{
    struct context *context = opaque;
    PyGILState_STATE gil;
    PyObject *input = NULL, *output = NULL;
    Py_ssize_t size;
    char *packed;
    int rc = -EIO;
    (void)prev_state;
    if (!context || !isfinite(strength) || strength < 0 || strength > 1 ||
        !valid_plane(context, source) || !valid_plane(context, destination))
        return -EINVAL;
    size = (Py_ssize_t)context->width * context->height;
    /* Copy all input rows before committing any output, including overlapping
     * source/destination planes. No borrowed buffer can escape this call. */
    packed = malloc((size_t)size);
    if (!packed)
        return -ENOMEM;
    pthread_mutex_lock(&context->lock);
    for (unsigned y = 0; y < context->height; ++y)
        memcpy(packed + (size_t)y * context->width, source->data + y * source->stride, context->width);
    if (strength == 0) {
        for (unsigned y = 0; y < context->height; ++y)
            memcpy(destination->data + y * destination->stride, packed + (size_t)y * context->width, context->width);
        rc = 0;
        goto unlocked;
    }
    gil = PyGILState_Ensure();
    input = PyBytes_FromStringAndSize(packed, size);
    if (input)
        output = PyObject_CallMethod(context->backend, "process_bytes", "Od", input, (double)strength);
    if (output && PyBytes_Check(output) && PyBytes_GET_SIZE(output) == size) {
        const char *result = PyBytes_AS_STRING(output);
        for (unsigned y = 0; y < context->height; ++y)
            memcpy(destination->data + y * destination->stride, result + (size_t)y * context->width, context->width);
        rc = 0;
    }
    if (PyErr_Occurred())
        PyErr_Print();
    Py_XDECREF(input);
    Py_XDECREF(output);
    PyGILState_Release(gil);
unlocked:
    pthread_mutex_unlock(&context->lock);
    free(packed);
    return rc;
}

static void backend_fini(void *opaque)
{
    struct context *context = opaque;
    PyGILState_STATE gil;
    PyObject *result;
    if (!context)
        return;
    /* Caller has stopped and joined all stream processing before fini. */
    gil = PyGILState_Ensure();
    result = PyObject_CallMethod(context->backend, "fini", NULL);
    if (!result)
        PyErr_Print();
    Py_XDECREF(result);
    Py_DECREF(context->backend);
    PyGILState_Release(gil);
    pthread_mutex_destroy(&context->lock);
    free(context);
    /* Never Py_Finalize a process-owned interpreter or unload its extensions. */
}

const struct camera_ane_backend camera_ane_backend_v1 = {
    .init = backend_init, .process = backend_process, .fini = backend_fini,
};
