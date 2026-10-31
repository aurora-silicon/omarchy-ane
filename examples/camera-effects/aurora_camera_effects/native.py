# SPDX-License-Identifier: MIT
"""Dormant native adapter: reuse t-3650's existing graph owner. A compiled
and independently qualified whole graph is required before opening a device.
"""
import ctypes as C
import hashlib
import importlib
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np


class NativeGraph:
    def __init__(self, root, metadata, library):
        self.root=Path(root)
        self.records={}
        self.failed=False
        self.directory=None
        # Export/compile success alone never authorizes device execution.
        if 'H13' not in metadata.get('native_targets',[]) or 'H13' not in metadata.get('qualified_targets',[]):
            raise OSError('no qualified full-model H13 package; using shared CPU fallback')
        package=self.root/'compiled/H13/full-model'
        if not (package/'manifest.json').is_file():
            raise OSError('missing native manifest')
        for file in package.rglob('*'):
            if file.is_file() and file.suffix!='.log':
                key=str(file.relative_to(self.root))
                if metadata['artifacts'].get(key)!=hashlib.sha256(file.read_bytes()).hexdigest():
                    raise ValueError('unidentified native artifact')
        lib=C.CDLL(str(library))
        lib.pyane_query_caps.argtypes=[C.c_int,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
        lib.pyane_query_caps.restype=C.c_int
        abi,chip=C.c_uint32(),C.c_uint32()
        rc=lib.pyane_query_caps(0,C.byref(abi),C.byref(chip))
        if rc or (abi.value,chip.value)!=(1,13):
            raise OSError('generation/ABI not supported by the H13 graph owner')
        compiler=Path('/usr/lib/mil-hwx-compiler')
        sys.path[:0]=[str(compiler/'tools'),str(compiler)]
        self.runner=importlib.import_module('h13_run_linux')
        self.adapter=self.runner.LibANEAdapter(library)
        self.adapter.execute=self.execute
        self.metadata=metadata
        self.directory=Path(tempfile.mkdtemp(prefix='aurora-camera-effects-'))
        try:
            self.package=self.directory/'package'
            shutil.copytree(package,self.package,ignore=shutil.ignore_patterns('*.log'))
        except BaseException:
            shutil.rmtree(self.directory)
            raise

    def execute(self, anec, inputs, sizes):
        key=hashlib.sha256(anec).digest()
        if key not in self.records:
            self.records[key]=self.runner.PreparedProgram(self.adapter,anec,inputs,sizes)
        self.records[key].refill(inputs)
        return self.records[key].run_once()

    def __call__(self, x):
        try:
            path=self.directory/'x.fp16'
            path.write_bytes(x.astype('<f2').tobytes())
            outputs={n:self.directory/(str(i)+'.fp16') for i,n in enumerate(self.metadata['mil_outputs'])}
            _,actual,_=self.runner.run_package(self.package,self.root/'model.mil',self.root,
                                               {'x':path},outputs,self.adapter,deadline_seconds=2)
            # Public ONNX and MIL output order is recorded by the exporter.
            result = [np.frombuffer(actual[n],'<f2').reshape(shape).astype(np.float32)
                      for n,shape in zip(self.metadata['mil_outputs'],self.metadata['output_shapes'])]
            if any(not np.isfinite(value).all() for value in result):
                raise ValueError('nonfinite native output')
            return result
        except Exception as error:
            self.failed=True
            (self.directory/'failure.txt').write_text(str(error)+'\n')
            raise

    def close(self):
        if self.directory is None:
            return
        try:
            self.runner._close_all(list(self.records.values()),None)
        except Exception as error:
            self.failed=True
            (self.directory/'cleanup-failure.txt').write_text(str(error)+'\n')
            raise
        finally:
            self.records.clear()
            if not self.failed:
                shutil.rmtree(self.directory)
