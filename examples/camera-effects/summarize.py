#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Print bounded numeric receipts to the normal build log as well as files."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
a=p.parse_args()
path=a.output/'measurements.json'
if path.exists():
    data=json.loads(path.read_text());q=data['quality']
    quality={k:v for k,v in q.items() if k!='images'}
    quality['image_ids']=[r['id'] for r in q['images']]
    rows=[]
    for record in data['timings']:
        row={k:v for k,v in record.items() if k!='runtime'}
        row['inference']={k:{n:s[n] for n in ('cpu_calls','native_calls','failed_calls')} for k,s in record['runtime'].items()}
        rows.append(row)
    print('HOST_MEASUREMENT='+json.dumps(dict(cpu=data['cpu'],platform=data['platform'],
          ort_threads=data['ort_threads'],opencv_threads=data['opencv_threads'],quality=quality,timings=rows),separators=(',',':')))
path=a.output/'source-reference.json'
if path.exists():
    print('SOURCE_REFERENCE='+json.dumps(json.loads(path.read_text()),separators=(',',':')))
