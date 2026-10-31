# SPDX-License-Identifier: MIT
"""Dependency-free policy checks: no NumPy, C build, CPU engine or device."""
import ctypes as C
import errno
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    loader = SourceFileLoader(name,str(path))
    spec = importlib.util.spec_from_loader(loader.name,loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

class NativeBenchGates(unittest.TestCase):
    def module(self):
        # native_benchmark only needs the supplied input's tobytes contract.
        # The CPU/array dependencies belong to separate Linux integration tests.
        with patch.dict('sys.modules',{'numpy':SimpleNamespace(),
                                      'ane':SimpleNamespace(Fallback=object,OnnxCPU=object)}):
            return load('ane_bench_owner_test',ROOT/'tools/ane-bench')

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
            root = Path(directory); receipt = root/'receipt'; receipt.mkdir()
            failed = SimpleNamespace(returncode=65,communicate=Mock(return_value=('runner output','runner refused')),kill=Mock())
            with patch.object(bench.C,'CDLL',return_value=self.caps(1,13)), \
                 patch.object(bench.tempfile,'mkdtemp',return_value=str(receipt)), \
                 patch.object(bench.subprocess,'Popen',return_value=failed) as run:
                with self.assertRaises(OSError):
                    bench.native_benchmark(root,root,{'x':SimpleNamespace(tobytes=lambda:b'\x00\x00')},root/'libane.so',1,10)
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

class FactoryPolicy(unittest.TestCase):
    def module(self):
        return load('ane_fallback_factory_test',ROOT/'bindings/python/python/ane/fallback.py')

    def library(self):
        return SimpleNamespace(pyane_fallback_init=Mock(),pyane_fallback_run=Mock())

    def test_load_refusal_initializes_existing_policy_for_cpu(self):
        m = self.module(); lib = self.library()
        factory = Mock(side_effect=OSError(errno.ENODEV,'no ANE'))
        with patch.object(m.C,'CDLL',return_value=lib):
            fallback = m.Fallback(lambda x:x,native_factory=factory)
        factory.assert_called_once_with()
        self.assertEqual(fallback.native_failure.errno,errno.ENODEV)
        self.assertFalse(lib.pyane_fallback_init.call_args.args[1])

    def test_native_factory_runs_once_and_initializes_native_policy(self):
        m = self.module(); lib = self.library(); native = lambda x:x
        factory = Mock(return_value=native)
        with patch.object(m.C,'CDLL',return_value=lib):
            fallback = m.Fallback(lambda x:x,native_factory=factory)
        factory.assert_called_once_with()
        self.assertIs(fallback.native,native)
        self.assertTrue(lib.pyane_fallback_init.call_args.args[1])

    def test_failed_library_load_never_opens_native_owner(self):
        m = self.module(); factory = Mock()
        with patch.object(m.C,'CDLL',side_effect=OSError('missing binding')):
            with self.assertRaises(OSError): m.Fallback(lambda x:x,native_factory=factory)
        factory.assert_not_called()

if __name__ == '__main__': unittest.main()
