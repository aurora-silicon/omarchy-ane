#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record direct H13 lowering of every exported operator, with exact shapes."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile


def probe(models,compiler,output):
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in ('selfie','yunet'):
        root=models/name
        metadata=json.loads((root/'model.json').read_text())
        with zipfile.ZipFile(root/'probes.zip') as archive:
            for op in [dict(name='full-model',op='full-model',mil='model.mil'),*metadata['ops']]:
                dest=output/name/op['name']
                dest.mkdir(parents=True)
                source=dest/'source.mil'
                source.write_bytes((root/'model.mil').read_bytes() if op['op']=='full-model' else archive.read(op['mil']))
                run=subprocess.run([str(compiler),'--target','H13','--mil',str(source),
                    '--model-root',str(root),'--output',str(dest/'bundle')],capture_output=True,
                    text=True,timeout=120,env=dict(os.environ,MILHWXC_H13_CONV_GENERIC='1'))
                (dest/'compile.log').write_text(run.stdout+run.stderr)
                records.append(dict(model=name,**op,rc=run.returncode,diagnostic=run.stderr[-4000:]))
    (output/'results.json').write_text(json.dumps(dict(host=platform.system()+'/'+platform.machine()+' build host',platform=platform.platform(),
        compiler=str(compiler),records=records),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('models',type=Path);p.add_argument('compiler',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();probe(a.models.resolve(),a.compiler.resolve(),a.output.resolve())
