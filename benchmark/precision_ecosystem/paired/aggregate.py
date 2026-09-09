#!/usr/bin/env python3
"""Audit >=3 paired process artifacts; emit local observations, not qualification.
Usage: aggregate.py run-directory ...
Each directory must contain run.jl output plus a launcher receipt.json binding
its actual subprocess pid/exit. No files are modified by this reader.
"""
from pathlib import Path
import hashlib, json, statistics, sys, tomllib

def load(path):
    return tomllib.loads(path.read_text())

def aggregate(paths):
    if len(paths)<3:
        raise ValueError('at least three independent processes required')
    baseline=None;seen_pids=set();seen_repetitions=set();rows=[]
    for raw in paths:
        p=Path(raw);before=load(p/'before.toml');after=load(p/'after.toml')
        assert before==after,'source/environment changed'
        data=load(p/'result.toml');receipt=json.loads((p/'receipt.json').read_text())
        assert receipt['exit']==0 and not receipt.get('timeout',False)
        assert receipt['result_sha256']==hashlib.sha256((p/'result.toml').read_bytes()).hexdigest()
        assert receipt['command'][-2]==data['input_file_hashes']['plan.toml']
        assert int(receipt['command'][-1])==data['repetition']
        assert receipt['pid']==data['pid'] and data['pid'] not in seen_pids
        assert data['repetition'] not in seen_repetitions
        seen_pids.add(data['pid']);seen_repetitions.add(data['repetition'])
        assert data['all_measured_outputs_pass'] and data['mfa_serial_parallel_bits_identical']
        assert not data['zero_input_control'],'zero-input cells are controls only'
        assert data['input_hashes_before']==data['input_hashes_after']==[data['input_sha256']]*2
        assert all(x['metrics']['pass'] for x in data['probes']+data['allocations'])
        binding=(before,data['input_file_hashes'],data['input_sha256'],data['configs'],data['threads'])
        if baseline is None:baseline=binding
        else:assert binding==baseline,'different actual source/input/type/goal/thread cell'
        samples=data['samples'];assert len(samples)==16
        expected=[]
        for block in (1,2):
            orders=((2,1,1,2),(1,2,2,1)) if (data['repetition']+block)%2 else ((1,2,2,1),(2,1,1,2))
            for quad,order in enumerate(orders,1):
                for position,variant in enumerate(order,1):expected.append((block,quad,position,variant))
        durations={1:[],2:[]}
        for i,(sample,want) in enumerate(zip(samples,expected),1):
            assert sample['sequence']==i
            assert tuple(sample[k] for k in ('block','quad','position','variant'))==want
            assert sample['metrics']['pass'] and sample['nanoseconds']>0
            durations[sample['variant']].append(sample['nanoseconds'])
        a=statistics.median(durations[1]);b=statistics.median(durations[2])
        rows.append(dict(path=str(p),pid=data['pid'],repetition=data['repetition'],
            direct_median_ns=a,mfa_median_ns=b,direct_over_mfa=a/b,
            artifact_sha256=hashlib.sha256((p/'result.toml').read_bytes()).hexdigest()))
    assert {r%2 for r in seen_repetitions}=={0,1},'starting-order counterbalance missing'
    return dict(processes=rows,process_ratio_median=statistics.median(r['direct_over_mfa'] for r in rows),
        common_input_sha256=baseline[2],performance_qualification=False,
        scope='bounded local observations; independent review and broader repetition remain required')

if __name__=='__main__':
    print(json.dumps(aggregate(sys.argv[1:]),indent=2))
