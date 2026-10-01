"""Reproduce one study from bundled local source and official reference tables."""
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'master' if (ROOT/'master').exists() else ROOT / 'source'
for name in ['streetlux','shadeparadox','specdebt']:
    if (SOURCE/name).exists(): sys.path.insert(0,str(SOURCE/name))

from streetlux.reference_data import load_publication_references
from streetlux import self_check, spectra, photometry
from streetlux.jsonio import dumps

def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(dumps(data),encoding='utf-8')

def csvsave(path, rows):
    rows=list(rows)
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,keys); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})

def reproduce(study, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    ref=load_publication_references(SOURCE/'reference_data')
    assert spectra.provenance()['official_action_spectra']
    assert spectra.provenance()['official_solar_spectrum']
    check=self_check(); save(out/'metrology_checks.json',check)
    if study=='streetlux':
        from streetlux.route import load_route
        from streetlux.scenarios import compare_modes, compare_designs
        from streetlux.report import write_html
        route=load_route(SOURCE/'streetlux/examples/dhaka_mirpur_motijheel.json')
        route.name='Dhaka-inspired five-section route, morning'
        save(out/'route.json',route.to_dict())
        for name, fn in [('modes',compare_modes),('designs',compare_designs)]:
            result=fn(route,n_headings=24,n_directions=768,step_seconds=60.0,objective='dose_weighted')
            save(out/(name+'.json'),result)
            write_html(result,out/(name+'.html'))
            csvsave(out/(name+'_summary.csv'),({'key':r['key'],'label':r['label'],
                **{k:v for k,v in r['summary'].items() if not isinstance(v,(dict,list))},
                'heading_ratio':r['summary']['heading_spread']['ratio_max_min'],
                'delta_vs_baseline_pct':r.get('delta_vs_baseline_pct'),
                'decomposition':r.get('decomposition')} for r in result['rows']))
            csvsave(out/(name+'_time_series.csv'),({'key':r['key'],**st} for r in result['rows'] for st in r['series']))
    elif study=='shadeparadox':
        from shadeparadox.experiment import e1_behaviour,e2_encoding_vs_algorithm,e3_view_direction
        from shadeparadox.report import write_html
        from shadeparadox.sim import REPRESENTATIVE_DAYS
        e1=e1_behaviour(orientations=(0,90,180,270),depths=(2.5,4.5,6.5),n_directions=256,dt_h=.5,seeds=tuple(range(5)))
        save(out/'e1.json',e1); print('E1 complete',flush=True)
        e2=e2_encoding_vs_algorithm(budget=16,n_directions=192,dt_h=.5)
        save(out/'e2.json',e2); print('E2 complete',flush=True)
        e3=e3_view_direction(depth=2.5,n_headings=12,n_directions=256,dt_h=.5)
        save(out/'e3.json',e3); print('E3 complete',flush=True)
        prov={'version':'0.2.0','spectral_inputs':spectra.provenance(),
              'k_mel_v_d65_derived_W_per_lm':photometry.calibration()['k_mel'],
              'k_mel_v_d65_official_W_per_lm':photometry.K_MEL_V_D65_OFFICIAL,
              'official_action_spectra':True, 'official_solar_spectrum':True,
              'k_mel_deviation_from_CIE_S026_pct':photometry.calibration()['deviation_pct'],
              'representative_days':REPRESENTATIVE_DAYS,'known_biases':spectra.active_biases(),
              'state_rule':'One controller across Jan-Apr-Jul-Oct synthetic representative-day sequence; not a chronological annual simulation.'}
        save(out/'shadeparadox_results.json',{'e1':e1,'e2':e2,'e3':e3,'provenance':prov})
        write_html(e1,e2,e3,prov,out/'shadeparadox_report.html')
        csvsave(out/'e1_per_seed.csv',({'orientation_deg':r['orientation_deg'],'depth_m':r['depth_m'],
            'controller':r['controller'],**v} for r in e1['rows'] for v in r['per_seed']))
        csvsave(out/'e2_cells.csv',({'encoding':c['encoding'],'optimiser':c['optimiser'],**c} for c in e2['cells']))
        csvsave(out/'e3_headings.csv',({'controller':r['controller'],**v} for r in e3['rows'] for v in r['values']))
    elif study=='specdebt':
        from specdebt import run_study,melanopic_metric,melanopic_specification,MELANOPIC_DECISION,benchmark_metric,benchmark_specification,BENCHMARK_DECISION
        from specdebt.cases import make_corpus
        mel=run_study(melanopic_metric(270,3,192,require_official_baseline=True),melanopic_specification(),MELANOPIC_DECISION,'melanopic EDI at the eye','lx')
        ben=run_study(benchmark_metric(600,seed=0),benchmark_specification(),BENCHMARK_DECISION,'accuracy(A) minus accuracy(B)','accuracy points')
        save(out/'specdebt_results.json',{'melanopic':mel.debt(.01),'benchmark':ben.debt(.01),
            'melanopic_specification':mel.spec.describe(),'benchmark_specification':ben.spec.describe(),
            'provenance':{'version':'0.2.0','spectral_inputs':spectra.provenance(),'corpus_seed':0,'n_items':600}})
        for name,r in [('melanopic',mel),('benchmark',ben)]:
            csvsave(out/(name+'_specifications.csv'),({'specification_id':i,**s,'value':float(v),'decision':bool(d),
                'outcome':('A_wins' if v>0 else 'B_wins' if v<0 else 'tie') if name=='benchmark' else ('pass' if d else 'fail')}
                for i,(s,v,d) in enumerate(zip(r.specs,r.values,r.decisions))))
            csvsave(out/(name+'_declaration_curve.csv'),r.sufficiency_curve(.01))
        a,b=make_corpus(600,0)
        csvsave(out/'synthetic_corpus.csv',({'model':c['name'],'item_id':i,**row} for c in [a,b] for i,row in enumerate(c['rows'])))
        save(out/'synthetic_corpus.json',{'A':a,'B':b})
    else: raise ValueError(study)
    save(out/'run_metadata.json',{'study':study,'package_version':'0.2.0','python':sys.version.split()[0],
        'numpy':np.__version__,'platform':platform.system(),'wall_clock_s':round(time.time()-t0,2),
        'parameters_source':'pipeline.py','reference_data':ref,
        'reference_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'reference_data').glob('*.csv')}})
    print(study,'completed in',round(time.time()-t0,2),'s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',choices=['streetlux','shadeparadox','specdebt']);p.add_argument('--out',required=True)
    a=p.parse_args();reproduce(a.study,a.out)
