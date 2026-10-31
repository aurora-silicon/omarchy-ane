# SPDX-License-Identifier: MIT
import errno
import ctypes as C
import importlib.util
from importlib.machinery import SourceFileLoader
from unittest.mock import patch, Mock
from types import SimpleNamespace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bindings/python/python'))
from ane import Fallback, OnnxCPU, model
LIB = os.environ.get('LIBANE_PYTHON',str(ROOT/'bindings/python/dylib/libane_python.so'))

class FallbackTests(unittest.TestCase):
    def test_missing_native(self):
        f = Fallback(lambda x:x+1,library=LIB)
        value, call = f.run(4)
        self.assertEqual(value,5); self.assertEqual(call.backend,0)
        self.assertEqual(f.stats()['cpu_calls'],1)
        self.assertGreater(f.stats()['calls_per_second'],0)

    def test_native_load_refusal_uses_cpu(self):
        attempts = []
        def factory():
            attempts.append(1)
            raise OSError(errno.ENODEV,'no ANE')
        f = Fallback(lambda x:x+1,library=LIB,native_factory=factory)
        self.assertEqual(f.run(4)[0],5)
        self.assertEqual(f.run(5)[0],6)
        self.assertEqual(attempts,[1])
        self.assertEqual(f.native_failure.errno,errno.ENODEV)
        self.assertEqual(f.stats()['native_calls'],0)

    def test_native_success_and_busy_retirement(self):
        count = [0]
        def native(x):
            count[0] += 1
            if count[0] > 1: raise OSError(errno.EBUSY,'busy')
            return x+1
        f = Fallback(lambda x:x+1,native,LIB)
        self.assertEqual(f.run(1)[1].backend,1)
        value, call = f.run(2)
        self.assertEqual(value,3); self.assertEqual(call.backend,0)
        self.assertEqual(call.native_error,-errno.EBUSY)
        f.run(3)
        self.assertEqual(count[0],2); self.assertEqual(f.stats()['native_calls'],1)
        self.assertEqual(f.stats()['cpu_calls'],2)

    def test_cancellation_never_escapes_ctypes_or_runs_cpu(self):
        calls = []
        def native(_): raise KeyboardInterrupt()
        f = Fallback(lambda x:calls.append(x),native,LIB)
        with self.assertRaises(KeyboardInterrupt): f.run(0)
        self.assertEqual(calls,[])
        self.assertEqual(f.stats()['failed_calls'],1)

    def test_existing_predict_raises_native_errors(self):
        from types import SimpleNamespace
        for failed in ('pyane_send','pyane_exec','pyane_read'):
            calls = []
            def function(name):
                def invoke(*args):
                    calls.append(name)
                    return -errno.EBUSY if name == failed else 0
                return invoke
            lib = SimpleNamespace(**{n:function(n) for n in ('pyane_send','pyane_exec','pyane_read')})
            m = model.__new__(model)
            m.driver = SimpleNamespace(lib=lib)
            m.handle, m.src_count, m.dst_count = None,0,0
            m.src_nchw,m.dst_nchw,m.outputs = [],[],[]
            m.inputs_pad,m.outputs_pad = [],[]
            with self.assertRaises(OSError): m.predict([])
            self.assertEqual(calls[-1],failed)

    def test_cpu_failure_propagates(self):
        def cpu(_): raise ValueError('bad model')
        f = Fallback(cpu,library=LIB)
        with self.assertRaises(RuntimeError) as c: f.run(0)
        self.assertIsInstance(c.exception.__cause__,ValueError)
        self.assertEqual(f.stats()['failed_calls'],1)

    def test_concurrent_calls(self):
        f = Fallback(lambda x:x+1,library=LIB)
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda i:f.run(i)[0],range(16)))
        self.assertEqual(values,list(range(1,17)))
        self.assertEqual(f.stats()['calls'],16)

    def test_onnx_same_fp16_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.onnx'
            weights = np.array([.125,-.25],dtype=np.float16)
            graph = h.make_graph([h.make_node('Add',['x','w'],['y'])],'weights',
                                 [h.make_tensor_value_info('x',T.FLOAT,[1,2])],
                                 [h.make_tensor_value_info('y',T.FLOAT,[1,2])],
                                 [nh.from_array(weights.astype(np.float32).reshape(1,2),'w')])
            onnx.save(h.make_model(graph,opset_imports=[h.make_opsetid('',13)],ir_version=10),path)
            cpu = OnnxCPU(path)
            f = Fallback(cpu,library=LIB)
            x = np.array([[1,2]],dtype=np.float16)
            value, _ = f.run({'x':x})
            np.testing.assert_array_equal(value['y'],x+weights)
            with self.assertRaises(ValueError): cpu({'x':x.astype(np.float32)})
            with self.assertRaises(ValueError): cpu({'wrong':x})

class NativeBenchGates(unittest.TestCase):
    def module(self):
        loader = SourceFileLoader('ane_bench_test',str(ROOT/'tools/ane-bench'))
        spec = importlib.util.spec_from_loader(loader.name,loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    def caps(self, abi, chip):
        def query(index, a, c):
            C.cast(a,C.POINTER(C.c_uint32))[0] = abi
            C.cast(c,C.POINTER(C.c_uint32))[0] = chip
            return 0
        return SimpleNamespace(pyane_query_caps=query)

    def test_h14_never_enters_h13_runner(self):
        bench = self.module()
        with patch.object(bench.C,'CDLL',return_value=self.caps(2,14)), patch.object(bench.subprocess,'Popen') as run:
            with self.assertRaises(OSError):
                bench.native_benchmark(Path('/model'),Path('/compiler'),{},Path('/libane.so'),1,10)
            run.assert_not_called()

    def test_native_failure_preserves_receipt_without_killing_runner(self):
        bench = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root/'receipt'; receipt.mkdir()
            failed = SimpleNamespace(returncode=65,communicate=Mock(return_value=('runner output','runner refused')),kill=Mock())
            with patch.object(bench.C,'CDLL',return_value=self.caps(1,13)), \
                 patch.object(bench.tempfile,'mkdtemp',return_value=str(receipt)), \
                 patch.object(bench.subprocess,'Popen',return_value=failed) as run:
                with self.assertRaises(OSError):
                    bench.native_benchmark(root,root,{'x':np.zeros((1,1),dtype=np.float16)},root/'libane.so',1,10)
                self.assertNotIn('start_new_session',run.call_args.kwargs)
                self.assertEqual(run.call_args.args[0][:2],['/bin/sh','-c'])
                failed.kill.assert_not_called()
                self.assertEqual((receipt/'stderr.log').read_text(),'runner refused')
                self.assertTrue((receipt/'x.fp16').is_file())
                self.assertTrue((receipt/'command.json').is_file())

    def test_interrupt_waits_for_owner_and_preserves_evidence(self):
        bench = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); receipt = root/'receipt'; receipt.mkdir()
            process = SimpleNamespace(returncode=0,kill=Mock(),
                communicate=Mock(side_effect=[KeyboardInterrupt(),('finished','cleaned up')]))
            with patch.object(bench.C,'CDLL',return_value=self.caps(1,13)), \
                 patch.object(bench.tempfile,'mkdtemp',return_value=str(receipt)), \
                 patch.object(bench.subprocess,'Popen',return_value=process):
                with self.assertRaises(KeyboardInterrupt):
                    bench.native_benchmark(root,root,{},root/'libane.so',1,10)
                process.kill.assert_not_called()
                self.assertEqual(process.communicate.call_count,2)
                self.assertTrue((receipt/'interrupted.txt').is_file())
                self.assertEqual((receipt/'stderr.log').read_text(),'cleaned up')

if __name__ == '__main__': unittest.main()
