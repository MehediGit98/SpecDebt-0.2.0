from pathlib import Path
import json, math
def compare_results(study, expected, actual):
    files={'streetlux':['modes.json','designs.json'],
           'shadeparadox':['e1.json','e2.json','e3.json'],
           'specdebt':['specdebt_results.json']}[study]
    checked=0
    skip={'wall_clock_s','reference_dir','platform','python','numpy'}
    def walk(a,b,path=''):
        nonlocal checked
        if isinstance(a,dict):
            if not isinstance(b,dict):raise AssertionError(path)
            for k,v in a.items():
                if k not in skip:
                    if k not in b:raise AssertionError('Missing '+path+'/'+k)
                    walk(v,b[k],path+'/'+k)
        elif isinstance(a,list):
            if len(a)!=len(b):raise AssertionError('Length '+path)
            for i,(v,w) in enumerate(zip(a,b)):walk(v,w,path+'/'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8):raise AssertionError(f'{path}: {a} != {b}')
            checked+=1
        elif a!=b:raise AssertionError(f'{path}: unequal values')
    for f in files:walk(json.loads((Path(expected)/f).read_text()),json.loads((Path(actual)/f).read_text()),f)
    return checked
