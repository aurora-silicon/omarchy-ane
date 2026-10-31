#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Try separate scalar-type/channel rewrites for the remaining host ops."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def main(models,compiler,output):
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in ('selfie','yunet'):
        root=models/name
        meta=json.loads((root/'model.json').read_text())
        with zipfile.ZipFile(root/'probes.zip') as archive:
            for op in meta['ops']:
                if op['op'] not in ('clip','sigmoid'):
                    continue
                text=archive.read(op['mil']).decode()
                rewrite='fp32-exact-clip-scalars' if op['op']=='clip' else 'pad-channels-to-64-or-512'
                if op['op']=='clip':
                    text='\n'.join(line.replace('fp16','fp32') if re.match(r'\s*fp16 \w+ = const',line) else line for line in text.splitlines())+'\n'
                else:
                    shape=op['output']
                    if len(shape)!=4:
                        continue
                    new=list(shape);new[1]=64 if shape[1]<=64 else 512
                    text=text.replace(str(shape),str(new))
                dest=output/name/op['name'];dest.mkdir(parents=True)
                (dest/'source.mil').write_text(text)
                run=subprocess.run([str(compiler),'--target','H13','--mil',str(dest/'source.mil'),
                    '--model-root',str(root),'--output',str(dest/'bundle')],capture_output=True,text=True,
                    timeout=120,env=dict(os.environ,MILHWXC_H13_CONV_GENERIC='1'))
                records.append(dict(model=name,name=op['name'],op=op['op'],original_shape=op['output'],
                                    rewrite=rewrite,rc=run.returncode,diagnostic=run.stderr[-4000:]))
    (output/'results.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('models',type=Path);p.add_argument('compiler',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.models.resolve(),a.compiler.resolve(),a.output.resolve())
