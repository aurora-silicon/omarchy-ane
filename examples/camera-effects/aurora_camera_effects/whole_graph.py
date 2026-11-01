# SPDX-License-Identifier: MIT
"""Resident H13 camera programs, with qualification before device access.

One PreparedProgram owns the LOAD for the lifetime of this object. A frame
does one refill/EXECUTE/readback. The shared Fallback retires a failed model.
CPU simulation is deliberately unable to authorize device execution.
"""
import ctypes as C
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .hybrid import local, sha


SCHEMA = 'aurora.camera-linked.v1'
RECEIPT_SCHEMA = 'aurora.camera-linked-qualification.v1'
RUNTIME_FILES = ('tools/h13_run_linux.py', 'research/inspect_anec.py', 'research/h13_td.py')


def runtime_identity(compiler_root):
    return {name:sha(Path(compiler_root)/name) for name in RUNTIME_FILES}


def checked_package(root, metadata, package=None):
    root = Path(root).resolve()
    directory = Path(package).resolve() if package else root/'compiled/H13/linked'
    plan_path = local(directory, 'camera-linked.json')
    plan = json.loads(plan_path.read_text())
    if (plan.get('schema') != SCHEMA or plan.get('target') != 'H13' or
            plan.get('model') != metadata['model'] or
            plan.get('source_sha256') != metadata['source_sha256'] or
            plan.get('onnx_sha256') != sha(root/'model.onnx') or
            plan.get('input_shape') != metadata['input'] or
            plan.get('output_shapes') != metadata['output_shapes'] or
            plan.get('output_names') != metadata['output_names'] or
            plan.get('mil_outputs') != metadata['mil_outputs']):
        raise ValueError('linked package model identity mismatch')
    files = plan.get('files', {})
    if not {'manifest.json', 'model.mil', 'weights.bin'} <= files.keys():
        raise ValueError('incomplete linked package identities')
    for name, digest in files.items():
        if sha(local(directory, name)) != digest:
            raise ValueError(f'linked artifact identity mismatch: {name}')
    manifest = json.loads(local(directory, 'manifest.json').read_text())
    if (manifest.get('target') != 'H13' or len(manifest.get('programs', [])) != 1 or
            manifest.get('dispatchPlan') != [0]):
        raise ValueError('whole graph requires exactly one program and one dispatch')
    program = manifest['programs'][0]
    if (program.get('operation') != 'linked' or
            program.get('encoder') not in ('h13-linked-conv', 'h13-linked-native-graph') or
            program.get('file') not in files or
            len(program.get('inputs', [])) != 1 or
            program['inputs'][0].get('shape') != metadata['input'] or
            [b.get('name') for b in program.get('outputs', [])] != metadata['mil_outputs'] or
            [b.get('shape') for b in program['outputs']] != metadata['output_shapes']):
        raise ValueError('unsupported linked program boundary contract')
    # Validate every physical view before allocating buffers or opening libane.
    for binding in program['inputs'] + program['outputs']:
        _check_binding(binding)
    return directory, plan, manifest, program


def _check_binding(binding):
    dims = binding.get('nchw', [])
    if (len(dims) != 6 or any(type(v) is not int or v < 1 for v in dims) or
            type(binding.get('allocationBytes')) is not int):
        raise ValueError('invalid linked surface dimensions')
    n, c, h, w, plane, row = dims
    size = binding['allocationBytes']
    if (n != 1 or binding.get('dtype') != 'float16' or
            not isinstance(binding.get('shape'), list) or
            any(type(v) is not int or v < 1 for v in binding['shape']) or
            math.prod(binding['shape']) != n*c*h*w or
            binding.get('logicalBytes') != n*c*h*w*2 or
            row < w*2 or plane < row*h or size < c*plane or size > 512*1024*1024):
        raise ValueError('invalid linked physical surface')


def surface(buffer, binding):
    _check_binding(binding)
    n, c, h, w, plane, row = binding['nchw']
    if len(buffer) != binding['allocationBytes']:
        raise ValueError('linked buffer allocation mismatch')
    return np.ndarray((n, c, h, w), dtype='<f2', buffer=buffer,
                      strides=(c*plane, plane, row, 2))


def pack_surface(value, binding):
    if list(value.shape) != binding['shape']:
        raise ValueError('linked input shape mismatch')
    buffer = bytearray(binding['allocationBytes'])
    view = surface(buffer, binding)
    view[:] = value.reshape(view.shape)
    return bytes(buffer)


def unpack_surface(buffer, binding):
    return surface(buffer, binding).astype(np.float32).reshape(binding['shape'])


def check_receipt(receipt, directory, *, simulation=False):
    records = receipt.get('records', [])
    mode = 'cpu-simulation' if simulation else 'libane-device'
    valid = (receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('status') == 'pass' and
             receipt.get('mode') == mode and receipt.get('target') == 'H13' and
             receipt.get('package_sha256') == sha(directory/'camera-linked.json') and
             receipt.get('images') == 30 and len(records) == 30 and
             len({r.get('file') for r in records}) == 30 and
             all(r.get('status') == 'pass' for r in records))
    if not simulation:
        timing = receipt.get('timing', {})
        numbers = [timing.get(key, np.nan) for key in ('ane_p50_ms', 'ane_p99_ms', 'cpu_p50_ms')]
        valid = (valid and (receipt.get('abi'), receipt.get('chip')) == (1, 13) and
                 receipt.get('cold_passes') == 30 and receipt.get('wedged_before') == '0' and
                 receipt.get('wedged_after') == '0' and
                 timing.get('frames', 0) >= 100 and
                 all(isinstance(v, (float, int)) and np.isfinite(v) and v > 0 for v in numbers) and
                 0 < timing.get('ane_p50_ms', 0) < timing.get('cpu_p50_ms', 0) and
                 timing.get('ane_p99_ms', 0) >= timing.get('ane_p50_ms', 0))
    plan = json.loads((directory/'camera-linked.json').read_text())
    if plan['model'] == 'selfie':
        valid = valid and all(r.get('mask_iou', 0) >= .98 and
                             r.get('mask_disagreement', 1) <= .005 and
                             r.get('mask_mean_abs', 1) <= .005 for r in records)
    else:
        valid = valid and all(r.get('yunet_tensor_gate') is True for r in records)
    if not valid:
        raise OSError('no matching whole-graph qualification')


class WholeGraph:
    def __init__(self, root, metadata, library, *, package=None, simulate=False,
                 qualification=False, compiler_root=Path('/usr/lib/mil-hwx-compiler'),
                 runner=None):
        self.closed = False
        self.prepared = None
        self.simulate = simulate
        self.loads = self.executes = 0
        self.directory, self.plan, self.manifest, self.program = checked_package(root, metadata, package)
        if not simulate and not qualification:
            receipt = json.loads(local(self.directory, 'qualification.json').read_text())
            check_receipt(receipt, self.directory)
            if (receipt.get('library_sha256') != sha(library) or
                    receipt.get('compiler_files') != runtime_identity(compiler_root)):
                raise OSError('whole-graph qualification runtime identity changed')
        sys.path[:0] = [str(Path(compiler_root)/'tools'), str(compiler_root)]
        # All package and qualification checks above precede caps/device access.
        device_runner = runner is None
        if device_runner:
            runner = importlib.import_module('h13_run_linux')
            if Path(runner.__file__).resolve() != (Path(compiler_root)/'tools/h13_run_linux.py').resolve():
                raise OSError('a different compiler reader is already imported')
        # Validate ANEC structure and descriptor/manifest agreement before caps.
        parsed, _ = runner.inspect_anec.load_package(self.directory)
        if parsed != self.manifest:
            raise ValueError('linked manifest changed during preparation')
        if simulate:
            from .linked_sim import MILSimulation
            self.simulator = MILSimulation(self.directory/'model.mil', self.directory)
            return
        if device_runner:
            lib = C.CDLL(str(library))
            lib.pyane_query_caps.argtypes = [C.c_int, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]
            lib.pyane_query_caps.restype = C.c_int
            abi, chip = C.c_uint32(), C.c_uint32()
            if lib.pyane_query_caps(0, C.byref(abi), C.byref(chip)) or (abi.value, chip.value) != (1, 13):
                raise OSError('whole-graph camera requires H13 ABI 1')
        adapter = runner.LibANEAdapter(library)
        start = time.perf_counter_ns()
        self.prepared = runner.PreparedProgram(adapter,
            local(self.directory, self.program['file']).read_bytes(),
            [bytes(b['allocationBytes']) for b in self.program['inputs']],
            [b['allocationBytes'] for b in self.program['outputs']])
        self.setup_ms = (time.perf_counter_ns()-start)/1e6
        self.loads = 1

    def __call__(self, x):
        if self.closed:
            raise RuntimeError('whole graph closed')
        if list(x.shape) != self.plan['input_shape'] or not np.isfinite(x).all():
            raise ValueError('invalid whole-graph input')
        if self.simulate:
            values = self.simulator(x, self.plan['mil_outputs'])
        else:
            self.prepared.refill([pack_surface(x, self.program['inputs'][0])])
            self.executes += 1
            raw = self.prepared.run_once()
            if len(raw) != len(self.program['outputs']):
                raise ValueError('missing linked outputs')
            values = [unpack_surface(data, b) for data, b in zip(raw, self.program['outputs'])]
        if len(values) != len(self.plan['output_shapes']) or any(list(v.shape) != shape or not np.isfinite(v).all()
               for v, shape in zip(values, self.plan['output_shapes'])):
            raise ValueError('invalid whole-graph output')
        return values

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.prepared is not None:
            self.prepared.close()
            self.prepared = None
