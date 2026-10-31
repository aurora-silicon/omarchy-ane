# SPDX-License-Identifier: MIT
"""CPU fallback policy in the existing libane binding.
Native callbacks use existing libane/graph runners; no model loader is copied.
"""
import ctypes as C
import errno
import threading

class _State(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('calls','native_calls','cpu_calls','failed_calls',
                'total_ns','last_ns','min_ns','max_ns')] + [
                ('native_ready',C.c_uint32),('reason',C.c_uint32),
                ('native_error',C.c_int32),('reserved',C.c_uint32)]

class Call(C.Structure):
    _fields_ = [('latency_ns',C.c_uint64),('native_ns',C.c_uint64),
                ('backend',C.c_uint32),('reason',C.c_uint32),
                ('native_error',C.c_int32),('status',C.c_int32)]

_Callback = C.CFUNCTYPE(C.c_int, C.c_void_p)

class Fallback:
    """Run native(inputs), or cpu(inputs) when unavailable or failed.

    Callbacks return outputs. CPU must use the same model/weights. Native state
    stays owned by its existing runner, including cleanup and fault quarantine.
    A native exception retires it permanently for this Fallback instance.
    """
    def __init__(self, cpu, native=None, library='/usr/lib/libane_python.so', *, native_factory=None):
        if not callable(cpu) or (native is not None and not callable(native)):
            raise TypeError('callbacks must be callable')
        if native_factory is not None and (native is not None or not callable(native_factory)):
            raise TypeError('use either native or a callable native_factory')
        self._lib = C.CDLL(str(library))
        self._lib.pyane_fallback_init.argtypes = [C.POINTER(_State), C.c_int]
        self._lib.pyane_fallback_init.restype = None
        self._lib.pyane_fallback_run.argtypes = [C.POINTER(_State),_Callback,_Callback,C.c_void_p,C.POINTER(Call)]
        self._lib.pyane_fallback_run.restype = C.c_int
        self.native_failure = None
        if native_factory is not None:
            try:
                native = native_factory()
            except Exception as error:
                self.native_failure = error
            if native is not None and not callable(native):
                raise TypeError('native_factory must return a callable or None')
        self._state = _State()
        self._lib.pyane_fallback_init(C.byref(self._state), native is not None)
        self.cpu, self.native = cpu, native
        self._lock = threading.Lock()

    def run(self, inputs):
        with self._lock:
            result = None
            cpu_failure = None
            cancellation = None
            def invoke(function, is_cpu):
                nonlocal result, cpu_failure, cancellation
                if cancellation is not None:
                    return -errno.ECANCELED
                try:
                    result = function(inputs)
                    return 0
                except BaseException as error:
                    if not isinstance(error, Exception):
                        cancellation = error
                        return -errno.ECANCELED
                    if is_cpu: cpu_failure = error
                    else: self.native_failure = error
                    number = getattr(error, 'errno', None)
                    return -number if isinstance(number, int) and number > 0 else -errno.EIO
            # Keep callback objects alive until the synchronous C call returns.
            native = _Callback(lambda _: invoke(self.native, False))
            cpu = _Callback(lambda _: invoke(self.cpu, True))
            call = Call()
            rc = self._lib.pyane_fallback_run(C.byref(self._state),native,cpu,None,C.byref(call))
            if cancellation is not None:
                raise cancellation
            if rc:
                raise RuntimeError(f'CPU inference failed: {rc}') from cpu_failure
            return result, call

    def stats(self):
        with self._lock:
            result = {n:getattr(self._state,n) for n,_ in _State._fields_ if n != 'reserved'}
            calls, ns = result['calls'], result['total_ns']
            result['calls_per_second'] = calls*1e9/ns if ns else 0
            return result

class OnnxCPU:
    """Optional ONNX Runtime CPU adapter, keeping fp16 at the caller boundary.

    The ONNX graph must contain the same fp16-rounded weights as the native
    export. Exporters can insert fp16/fp32 casts for activation-rounding parity.
    """
    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        for item in self.inputs + self.outputs:
            if item.type != 'tensor(float)' or any(type(d) is not int or d <= 0 for d in item.shape):
                raise ValueError('CPU graph requires static float32 interfaces')

    def __call__(self, inputs):
        import numpy as np
        if set(inputs) != {i.name for i in self.inputs}:
            raise ValueError('CPU model input names differ')
        arrays = {}
        for item in self.inputs:
            value = np.asarray(inputs[item.name])
            if value.dtype != np.float16 or tuple(value.shape) != tuple(item.shape):
                raise ValueError(f'{item.name}: expected fp16 {item.shape}')
            arrays[item.name] = np.ascontiguousarray(value, dtype=np.float32)
        values = self.session.run([o.name for o in self.outputs], arrays)
        return {o.name:v.astype(np.float16) for o,v in zip(self.outputs, values)}
