#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify linked camera packages: 30 cold portraits, parity and p50/p99.

Use --simulate on a host. --qualify runs only in the lead's controller-approved
M1 Pro execute request, with external approved source identities. No camera,
audio, boot, module, power or session settings are changed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora_camera_effects.backend import verified
from aurora_camera_effects.hybrid import session, sha
from aurora_camera_effects.whole_graph import WholeGraph, RECEIPT_SCHEMA, checked_package, runtime_identity
from h13.verify import output_metrics, write_receipt


def portraits(images, name):
    files = sorted(images.glob('*.jpg'))
    provenance = json.loads((images/'images.json').read_text())
    by_file = {v['file_name']:v for v in provenance}
    if (len(files) != 30 or len(provenance) != 30 or len(by_file) != 30 or
            set(by_file) != {p.name for p in files} or
            any(v['license'] != 4 or v['height'] < v['width'] for v in provenance)):
        raise ValueError('expected the 30 audited CC BY 2.0 COCO portraits')
    result = []
    for path in files:
        if sha(path) != by_file[path.name]['sha256']:
            raise ValueError('portrait identity mismatch')
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise ValueError('cannot decode portrait')
        size = 256 if name == 'selfie' else 320
        pixels = cv2.resize(pixels, (size, size), interpolation=cv2.INTER_AREA)
        if name == 'selfie':
            pixels = pixels[..., ::-1].astype(np.float32)/255
        x = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float16)
        result.append((dict(file=path.name, id=by_file[path.name]['id'], sha256=sha(path)), x))
    return result


def preflight(root, images, compiler, library, simulate, approval, device):
    metadata = verified(root, root.name)
    directory, _, _, _ = checked_package(root, metadata)
    inputs = portraits(images, root.name)
    if not simulate:
        # Concrete device identities and policy approvals stay outside source.
        pins = json.loads(approval.read_text())
        if (device != 'm1pro' or pins.get('device') != device or
                not str(pins.get('request', '')).startswith('h-') or
                pins.get('library_sha256') != sha(library) or
                pins.get('reader_sha256') != sha(compiler/'tools/h13_run_linux.py') or
                pins.get('compiler_files') != runtime_identity(compiler) or
                pins.get('portraits_sha256') != sha(images/'images.json') or
                pins.get('packages', {}).get(root.name) != sha(directory/'camera-linked.json')):
            raise ValueError('controller-approved device/source identity mismatch')
        compatible = Path('/proc/device-tree/compatible').read_bytes().split(b'\0')
        if not {b'apple,t6000', b'apple,t6001'} & set(compatible):
            raise OSError('qualification requires M1 Pro/Max H13')
    return metadata, directory, inputs


def qualify(root, images, compiler, library, output, *, simulate, frames=300,
            approval=None, device=None, wedged=Path('/sys/class/accel/accel0/device/wedged')):
    package = root/'compiled/H13/linked'
    receipt = package/'qualification.json'
    if not simulate and receipt.exists():
        receipt.rename(receipt.with_name(f'qualification.previous-{time.time_ns()}.json'))
    result = dict(schema=RECEIPT_SCHEMA, status='fail', target='H13', model=root.name,
                  mode='cpu-simulation' if simulate else 'libane-device', images=30,
                  cold_passes=0, records=[], active=None, wedged_checks=[])

    def save():
        write_receipt(output, result)
        if not simulate and package.is_dir():
            write_receipt(receipt, result)

    def check_wedged(phase, frame):
        value=wedged.read_text().strip()
        result['wedged_checks'].append(dict(phase=phase,frame=frame,value=value))
        if value!='0': raise OSError('ANE wedged during '+phase)

    save()
    try:
        metadata, package, inputs = preflight(root, images, compiler, library, simulate, approval, device)
        result['package_sha256'] = sha(package/'camera-linked.json')
        reference = session(root/'model.onnx')
        if not simulate:
            result.update(abi=1, chip=13, device=device,
                          library_sha256=sha(library), reader_sha256=sha(compiler/'tools/h13_run_linux.py'),
                          compiler_files=runtime_identity(compiler),
                          wedged_before=wedged.read_text().strip())
            if result['wedged_before'] != '0':
                raise OSError('ANE wedged before qualification')
        for record, x in inputs:
            record['status']='running'
            result['records'].append(record)
            result['active']=dict(phase='cold-load',portrait=record['file'])
            save()
            graph = WholeGraph(root, metadata, library, simulate=simulate,
                               qualification=True, compiler_root=compiler)
            if simulate:
                result['simulation_method'] = getattr(graph,'simulation_method','test-fixture')
            execution_failed=False
            stage='execute'
            result['active'].update(phase='cold-execute',
                task_count=getattr(graph,'program',{}).get('taskDescriptors'))
            save()
            try:
                # Fresh handle, new input, first EXECUTE, no warmup. Reference
                # evaluation is outside the measured device frame.
                start = time.perf_counter_ns()
                actual = graph(x)
                record['cold_ms'] = (time.perf_counter_ns()-start)/1e6
                stage='parity'
                cpu = reference.run(None, {'x':x.astype(np.float32)})
                output_metrics(root.name, actual, cpu, record=record)
                if root.name == 'yunet':
                    record['yunet_tensor_gate'] = True
                if not simulate and (graph.loads, graph.executes) != (1, 1):
                    raise ValueError('cold run was not one LOAD/one EXECUTE')
                record['status'] = 'pass'
                result['cold_passes'] += 1
            except Exception as error:
                record.update(status='fail', error=f'{type(error).__name__}: {error}')
                execution_failed=not simulate and stage=='execute'
            finally:
                try:
                    graph.close()
                except Exception as error:
                    record.update(status='fail',close_error=f'{type(error).__name__}: {error}')
                    raise
            save()
            print(root.name, record['file'], record['status'], record.get('mask_iou', ''), flush=True)
            if execution_failed:
                raise OSError('device execution failed; aborting before another LOAD')
            if not simulate:
                check_wedged('cold',len(result['records'])-1)
        result['active']=None
        if any(r['status'] != 'pass' for r in result['records']):
            raise ValueError('30-portrait parity gate failed')
        if root.name == 'selfie':
            result['min_mask_iou'] = min(r['mask_iou'] for r in result['records'])
        if not simulate:
            result['active']=dict(phase='timing-load')
            save()
            graph = WholeGraph(root, metadata, library, qualification=True, compiler_root=compiler)
            result['active'].update(phase='timing-execute',frame=0,
                task_count=getattr(graph,'program',{}).get('taskDescriptors'))
            save()
            ane, cpu_times = [], []
            try:
                for index in range(frames):
                    x = inputs[index % 30][1]
                    result['active'].update(frame=index,portrait=inputs[index % 30][0]['file'])
                    check_wedged('timing-before',index)
                    # Alternate order to reduce ordering bias. Both inputs are
                    # already normalized; both timed calls include conversion
                    # to their native layout and readback to dense outputs.
                    calls = [('ane', lambda:graph(x)),
                             ('cpu', lambda:reference.run(None, {'x':x.astype(np.float32)}))]
                    if index % 2:
                        calls.reverse()
                    values = {}
                    for engine, call in calls:
                        start = time.perf_counter_ns()
                        values[engine] = call()
                        (ane if engine == 'ane' else cpu_times).append((time.perf_counter_ns()-start)/1e6)
                    output_metrics(root.name, values['ane'], values['cpu'])
                    check_wedged('timing-after',index)
                if (graph.loads, graph.executes) != (1, frames):
                    raise ValueError('resident timing violated one LOAD/one EXECUTE per frame')
                result['timing'] = dict(frames=frames, loads=graph.loads, executes=graph.executes,
                    setup_ms=graph.setup_ms, ane_p50_ms=float(np.percentile(ane, 50)),
                    ane_p99_ms=float(np.percentile(ane, 99)), cpu_p50_ms=float(np.percentile(cpu_times, 50)),
                    cpu_p99_ms=float(np.percentile(cpu_times, 99)),
                    scope='normalized tensor to dense outputs; includes physical packing, EXECUTE and readback')
            finally:
                graph.close()
            result['wedged_after'] = wedged.read_text().strip()
            if result['wedged_after'] != '0':
                raise OSError('ANE wedged after timing')
            if result['timing']['ane_p50_ms'] >= result['timing']['cpu_p50_ms']:
                raise ValueError('ANE p50 is not faster than measured CPU p50')
        result['status'] = 'pass'
        result['active']=None
    except Exception as error:
        if result['records'] and result['records'][-1].get('status')=='running':
            result['records'][-1].update(status='fail',error=f'{type(error).__name__}: {error}')
        result['error'] = f'{type(error).__name__}: {error}'
        if not simulate and result.get('active') is not None:
            result['device_error']=True
    finally:
        if not simulate:
            try:
                result['wedged_after'] = wedged.read_text().strip()
                if result['wedged_after'] != '0':
                    result.update(status='fail', error='ANE wedged at final check')
            except OSError as error:
                result.update(status='fail', wedged_error=str(error))
        save()
    print(root.name,'qualification',result['status'],result.get('error',''),flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--model', choices=['both','selfie','yunet'], default='both')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--compiler-root', type=Path, required=True)
    p.add_argument('--library', type=Path, default=Path('/usr/lib/libane_python.so'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=300)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--simulate', action='store_true')
    mode.add_argument('--qualify', action='store_true')
    p.add_argument('--approval', type=Path, help='external controller-approved identity manifest')
    p.add_argument('--device', choices=['m1pro'])
    a = p.parse_args()
    if a.frames < 100 or (a.qualify and (a.approval is None or a.device is None)):
        p.error('device qualification requires --approval, --device m1pro and --frames >= 100')
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in (('selfie','yunet') if a.model=='both' else (a.model,)):
        results.append(qualify(a.models/name, a.images, a.compiler_root, a.library,
            a.output/(name+'.json'), simulate=a.simulate, frames=a.frames,
            approval=a.approval, device=a.device))
        if results[-1].get('device_error'):
            break  # Never LOAD the next model after a device failure.
    return 0 if all(r['status'] == 'pass' for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
