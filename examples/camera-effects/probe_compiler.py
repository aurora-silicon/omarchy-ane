#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Capture exact full-graph and one-per-op lowering results. Never publishes
native_targets: compilation is only the first gate before qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('models',type=Path);p.add_argument('compiler',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();records=[]
for name in ('selfie','yunet'):
    root=(a.models/name).resolve();metadata=json.loads((root/'model.json').read_text())
    # Distinct movement and convolution shapes are recorded, not extrapolated
    # from an unrelated geometry. Keep the sweep bounded to first per op type.
    probes=[dict(name='full-model',mil='model.mil')]
    seen=set()
    for op in metadata['ops']:
        if op['op'] not in seen:
            seen.add(op['op']);probes.append(op)
    for target in ('H13','H14','H16G'):
        for probe in probes:
            if not probe['name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in probe['name']):
                raise ValueError('invalid probe name')
            dest=a.output.parent/'compiler-output'/name/target/probe['name']
            dest.mkdir(parents=True,exist_ok=False)
            source=root/'model.mil'
            if probe['name']!='full-model':
                candidate=(root/probe['mil']).resolve()
                if not candidate.is_relative_to(root):
                    raise ValueError('probe path escapes model root')
                if candidate.is_file():
                    source=candidate  # Also accept a fresh unpacked export.
                else:
                    source=dest/'source.mil'
                    with zipfile.ZipFile(root/'probes.zip') as archive:
                        source.write_bytes(archive.read(probe['mil']))
            command=[str(a.compiler.resolve()),'--mil',str(source),'--model-root',str(root),
                     '--target',target,'--output',str(dest)]
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=120)
                rc,out,err=result.returncode,result.stdout,result.stderr
            except subprocess.TimeoutExpired as error:
                rc,out,err=124,str(error.stdout or ''),str(error.stderr or '')
            (dest/'stdout.log').write_text(out);(dest/'stderr.log').write_text(err)
            records.append(dict(model=name,target=target,name=probe['name'],op=probe.get('op','full-model'),
                                rc=rc,diagnostic=err[-4000:],mil=probe['mil']))
a.output.write_text(json.dumps(records,indent=2)+'\n')
