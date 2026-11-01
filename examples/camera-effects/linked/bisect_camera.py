#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare and run intermediate probes in a controller-approved H13 slot.

This diagnostic never writes production qualification. A numerical mismatch
is recorded and probing may continue; a device error aborts the ladder and
preserves the exact active task range. No recovery or machine settings change.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from aurora_camera_effects.hybrid import sha
from aurora_camera_effects.whole_graph import pack_surface, unpack_surface, runtime_identity
from aurora_camera_effects.linked_sim import MILSimulation
from aurora_camera_effects.yunet_boundary import prepare as phase_input
from qualify import portraits


def write(path, value):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def check_index(probes, source):
    index=json.loads((probes/'index.json').read_text())
    if index.get('schema')!='aurora.camera-prefixes.v1' or index.get('diagnostic_only') is not True:
        raise ValueError('not a diagnostic prefix index')
    if index['source_files'] != {n:sha(source/n) for n in index['source_files']}:
        raise ValueError('prefix source identities differ')
    for record in index['records']:
        if record['status'] != 'pass': continue
        directory=probes/record['directory']
        if directory.parent.resolve()!=probes.resolve(): raise ValueError('unsafe diagnostic directory')
        for name,digest in record['files'].items():
            path=directory/name
            if not path.resolve().is_relative_to(directory.resolve()) or sha(path)!=digest:
                raise ValueError('diagnostic file identity differs')
    return index


def prepare(probes, source, images, compiler, output, portrait=0):
    index=check_index(probes,source)
    sys.path[:0]=[str(compiler/'tools'),str(compiler/'research')]
    record,x=portraits(images,'yunet')[portrait]
    simulation=MILSimulation(source/'model.mil',source)
    wanted={r['result'] for r in index['records']}|{r['boundary'] for r in index['records']}
    values={}
    simulation(phase_input(x),simulation.returns,trace=lambda n,v:
               values.update({n:np.asarray(v,np.float16).copy()}) if n in wanted else None)
    if set(values)!=wanted: raise ValueError('missing diagnostic oracle tensor')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'oracle.npz',**values)
    metadata=dict(schema='aurora.camera-prefix-oracle.v1',index_sha256=sha(probes/'index.json'),
                  source_files=index['source_files'],portraits_sha256=sha(images/'images.json'),
                  portrait=record,oracle_sha256=sha(output/'oracle.npz'),
                  simulator_sha256=sha(Path(__file__).resolve().parents[1]/'aurora_camera_effects/linked_sim.py'))
    write(output/'oracle.json',metadata)
    return metadata


def preflight(probes,source,oracle,compiler,library,approval,device):
    index=check_index(probes,source)
    meta=json.loads((oracle/'oracle.json').read_text())
    pins=json.loads(approval.read_text())
    if (device!='m1pro' or pins.get('device')!=device or
            not str(pins.get('request','')).startswith('h-') or
            pins.get('library_sha256')!=sha(library) or
            pins.get('compiler_files')!=runtime_identity(compiler) or
            pins.get('script_sha256')!=sha(Path(__file__)) or
            pins.get('index_sha256')!=sha(probes/'index.json') or
            pins.get('oracle_metadata_sha256')!=sha(oracle/'oracle.json') or
            meta['index_sha256']!=sha(probes/'index.json') or
            meta['source_files']!=index['source_files'] or
            meta['oracle_sha256']!=sha(oracle/'oracle.npz')):
        raise ValueError('controller-approved prefix/source/oracle identity mismatch')
    compatible=Path('/proc/device-tree/compatible').read_bytes().split(b'\0')
    if not {b'apple,t6000',b'apple,t6001'}&set(compatible): raise OSError('requires M1 Pro/Max H13')
    # Validate all package descriptors before opening the device.
    sys.path[:0]=[str(compiler/'tools'),str(compiler/'research')]
    reader=importlib.import_module('inspect_anec')
    for r in index['records']:
        if r['status']=='pass':
            d=probes/r['directory'];reader.load_package(d/'package' if (d/'package').is_dir() else d)
    return index,pins


def metrics(actual,expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all():
        raise ValueError('invalid device intermediate')
    expected=expected.astype(np.float32)
    delta=np.abs(actual-expected)
    coord=np.unravel_index(np.argmax(delta),delta.shape)
    return dict(max_abs=float(delta[coord]),mean_abs=float(delta.mean()),
                rel_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(expected.ravel()),1e-20)),
                fraction_over_gate=float(np.mean(delta>0.03+0.02*np.abs(expected))),
                worst_coordinate=list(map(int,coord)),expected=float(expected[coord]),actual=float(actual[coord]))


def run(probes,source,oracle,compiler,library,approval,device,output,*,layers=None,modes=None,cold_runs=3,
        wedged=Path('/sys/class/accel/accel0/device/wedged')):
    if cold_runs<1: raise ValueError('cold_runs must be positive')
    index,pins=preflight(probes,source,oracle,compiler,library,approval,device)
    from h13_run_linux import LibANEAdapter, PreparedProgram
    import inspect_anec
    output.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema='aurora.camera-prefix-diagnostic.v1',status='incomplete',device=device,
                 request=pins['request'],kernel=__import__('platform').release(),records=[],
                 index_sha256=sha(probes/'index.json'),oracle_metadata_sha256=sha(oracle/'oracle.json'),
                 wedged_before=wedged.read_text().strip(),active=None)
    def save(): write(output/'diagnostic.json',receipt)
    save()
    if receipt['wedged_before']!='0': raise OSError('ANE is wedged before ladder')
    selected=[r for r in index['records'] if r['status']=='pass' and
              (not layers or r['layer'] in layers) and (not modes or r['mode'] in modes)]
    if not selected: raise ValueError('no matching diagnostic probes')
    adapter=LibANEAdapter(library)
    with np.load(oracle/'oracle.npz',allow_pickle=False) as values:
        for rec in selected:
            case=probes/rec['directory'];directory=case/'package' if (case/'package').is_dir() else case
            manifest,_=inspect_anec.load_package(directory);program=manifest['programs'][0]
            for cold in range(cold_runs):
                receipt['active']=dict(layer=rec['layer'],result=rec['result'],mode=rec['mode'],cold=cold,
                    source_first_task=rec['source_first_task'],source_tasks=rec['source_tasks'],
                    submitted_tasks=program['taskDescriptors'])
                save()
                prepared=None
                try:
                    if wedged.read_text().strip()!='0': raise OSError('ANE became wedged')
                    payload=pack_surface(values[rec['boundary']],program['inputs'][0])
                    prepared=PreparedProgram(adapter,(directory/program['file']).read_bytes(),
                                             [payload],[program['outputs'][0]['allocationBytes']])
                    begin=time.perf_counter_ns()
                    actual=unpack_surface(prepared.run_once()[0],program['outputs'][0])
                    elapsed=(time.perf_counter_ns()-begin)/1e6
                    score=metrics(actual,values[rec['result']])
                    np.save(output/f'{rec["directory"]}-cold{cold}.npy',actual,allow_pickle=False)
                    health=wedged.read_text().strip()
                    receipt['records'].append(dict(**receipt['active'],ms=elapsed,wedged=health,**score))
                    print(json.dumps(receipt['records'][-1]),flush=True)
                    save()
                    if health!='0': raise OSError('ANE wedged after prefix')
                except BaseException as error:
                    receipt.update(status='device-error',error=f'{type(error).__name__}: {error}')
                    save()
                    raise
                finally:
                    if prepared is not None:
                        try:
                            prepared.close()
                        except BaseException as error:
                            receipt.update(status='device-error',close_error=f'{type(error).__name__}: {error}')
                            save()
                            raise
    receipt.update(status='diagnostic-complete',active=None,wedged_after=wedged.read_text().strip())
    save()
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--run',action='store_true')
    for name in ('probes','source','compiler-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('images','oracle','library','approval'): p.add_argument('--'+name,type=Path)
    p.add_argument('--portrait',type=int,default=0);p.add_argument('--device')
    p.add_argument('--layer',type=int,action='append');p.add_argument('--mode',action='append')
    p.add_argument('--cold-runs',type=int,default=3)
    a=p.parse_args()
    if a.prepare:
        if not a.images: p.error('--prepare requires --images')
        result=prepare(a.probes,a.source,a.images,a.compiler_root,a.output,a.portrait)
        print(json.dumps(result,indent=2))
    else:
        if not all((a.oracle,a.library,a.approval,a.device)):
            p.error('--run requires --oracle, --library, --approval and --device')
        run(a.probes,a.source,a.oracle,a.compiler_root,a.library,a.approval,a.device,a.output,
            layers=a.layer,modes=a.mode,cold_runs=a.cold_runs)
