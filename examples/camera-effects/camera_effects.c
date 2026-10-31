/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "camera_effects.h"
#ifndef CE_PYTHON_DIR
#define CE_PYTHON_DIR "/usr/lib/aurora-camera-effects"
#endif
#ifndef CE_PYTHON_EXECUTABLE
#define CE_PYTHON_EXECUTABLE "/usr/bin/python3"
#endif
#ifndef CE_PYTHON_LIBRARY
#error CE_PYTHON_LIBRARY must identify the linked shared CPython library
#endif
struct state { PyObject *object; size_t bytes; };
static pthread_once_t once = PTHREAD_ONCE_INIT;
static int init_error;
static void init_python(void)
{
	/* The filter loads this plugin RTLD_LOCAL. CPython's extension modules
	 * expect its C API in the global namespace. Promote the already-linked
	 * interpreter, never load a second interpreter or finalize the owner. */
	if (!dlopen(CE_PYTHON_LIBRARY, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD)) {
		fprintf(stderr, "camera-effects: cannot expose linked CPython API: %s\n", dlerror());
		init_error = 1;
		return;
	}
	if (!Py_IsInitialized()) {
		PyConfig cfg;
		PyStatus status;
		PyConfig_InitPythonConfig(&cfg);
		cfg.install_signal_handlers = 0;
		status = PyConfig_SetBytesString(&cfg, &cfg.program_name, CE_PYTHON_EXECUTABLE);
		if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&cfg);
		PyConfig_Clear(&cfg);
		if (PyStatus_Exception(status)) { init_error = 1; return; }
		PyEval_SaveThread();
	}
}
static void *create(const struct ce_config *c)
{
	struct state *s;
	PyObject *module = NULL, *directory;
	PyGILState_STATE gil;
	if (!c || c->abi != CE_ABI || c->width < 2 || c->height < 2 ||
		c->width > 4096 || c->height > 2160 || ((c->width | c->height) & 1) ||
		c->background > 2 || c->framing > 1 || (c->matrix != 601 && c->matrix != 709) ||
		c->full_range > 1 || !memchr(c->replacement, 0, sizeof(c->replacement)) ||
		!isfinite(c->zoom) || !isfinite(c->zoom_min) || !isfinite(c->zoom_max) ||
		!isfinite(c->zoom_default) || (c->framing &&
		!(1 <= c->zoom_min && c->zoom_min <= c->zoom_default && c->zoom_default <= c->zoom_max &&
		  c->zoom_max <= 6 && (c->zoom == 0 || (c->zoom >= c->zoom_min && c->zoom <= c->zoom_max)))))
		return NULL;
	s = calloc(1, sizeof(*s));
	if (!s) return NULL;
	pthread_once(&once, init_python);
	if (init_error) { free(s); return NULL; }
	gil = PyGILState_Ensure();
	directory = PyUnicode_FromString(CE_PYTHON_DIR);
	if (directory && !PyList_Insert(PySys_GetObject("path"), 0, directory))
		module = PyImport_ImportModule("aurora_camera_effects.backend");
	if (module)
		s->object = PyObject_CallMethod(module, "create", "iiiiddddsii",
			(int)c->width, (int)c->height, (int)c->background, (int)c->framing,
			(double)c->zoom_min, (double)c->zoom_max, (double)c->zoom_default,
			(double)c->zoom, c->replacement, (int)c->matrix, (int)c->full_range);
	if (PyErr_Occurred()) PyErr_Print();
	Py_XDECREF(directory);
	Py_XDECREF(module);
	PyGILState_Release(gil);
	if (!s->object) { free(s); return NULL; }
	s->bytes = (size_t)c->width * c->height * 3 / 2;
	return s;
}
static int process(void *opaque, const uint8_t *in, uint8_t *out, uint64_t pts)
{
	struct state *s = opaque;
	PyGILState_STATE gil;
	PyObject *input, *result = NULL;
	int rc = -EIO;
	if (!s || !in || !out) return -EINVAL;
	gil = PyGILState_Ensure();
	input = PyBytes_FromStringAndSize((const char *)in, (Py_ssize_t)s->bytes);
	if (input) result = PyObject_CallMethod(s->object, "process_bytes", "OK", input, (unsigned long long)pts);
	if (result && PyBytes_Check(result) && PyBytes_GET_SIZE(result) == (Py_ssize_t)s->bytes) {
		memcpy(out, PyBytes_AS_STRING(result), s->bytes);
		rc = 0;
	}
	if (PyErr_Occurred()) PyErr_Print();
	Py_XDECREF(input);
	Py_XDECREF(result);
	PyGILState_Release(gil);
	return rc;
}
static void reset(void *opaque)
{
	struct state *s = opaque;
	if (!s) return;
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *r = PyObject_CallMethod(s->object, "reset", NULL);
	if (!r) PyErr_Print();
	Py_XDECREF(r);
	PyGILState_Release(gil);
}
static void destroy(void *opaque)
{
	struct state *s = opaque;
	if (!s) return;
	PyGILState_STATE gil = PyGILState_Ensure();
	PyObject *r = PyObject_CallMethod(s->object, "close", NULL);
	if (!r) PyErr_Print();
	Py_XDECREF(r);
	Py_DECREF(s->object);
	PyGILState_Release(gil);
	free(s);
	/* Process-owned interpreter/extensions stay loaded until process exit. */
}
const struct ce_ops *camera_effects_entry(void)
{
	static const struct ce_ops ops = { CE_ABI, create, process, reset, destroy };
	return &ops;
}
