"""Independent regression checks; standard library unittest, no pytest required."""
from pathlib import Path
import itertools, json, math, sys, unittest
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'master' if (ROOT/'master').exists() else ROOT/'source'
for name in ['streetlux','shadeparadox','specdebt']:
    if (SOURCE/name).exists():sys.path.insert(0,str(SOURCE/name))

class SharedChecks(unittest.TestCase):
    def test_spectral_reference(self):
        from streetlux.reference_data import load_publication_references
        from streetlux import photometry,self_check,spectra
        load_publication_references(SOURCE/'reference_data')
        self.assertEqual(photometry.K_MEL_V_D65_OFFICIAL,.0013262)
        self.assertTrue(spectra.provenance()['official_action_spectra'])
        self.assertLess(abs(self_check()['der_d65']-1),.001)

    def test_strict_json(self):
        from streetlux.jsonio import dumps
        self.assertEqual(json.loads(dumps({'undefined':float('nan'),'infinite':float('inf')})),{'undefined':None,'infinite':None})

class StreetChecks(unittest.TestCase):
    def route(self):
        from streetlux.route import load_route
        return load_route(SOURCE/'streetlux/examples/dhaka_mirpur_motijheel.json')

    def test_design_invariants(self):
        from streetlux.scenarios import STANDARD_SCENARIOS
        route=self.route(); scenarios={s.key:s for s in STANDARD_SCENARIOS}
        for old,new in zip(route.sections,scenarios['widen_18'].apply(route).sections):
            self.assertGreaterEqual(new.width_m,old.width_m); new.validate()
            self.assertEqual(new.eye_offset_m,old.eye_offset_m)
        under=route.sections[3]
        for key in ['canopy_40','canopy_70']:
            new=scenarios[key].apply(route).sections[3]
            self.assertEqual(new.canopy,under.canopy)
        new=scenarios['pale_paving'].apply(route)
        self.assertEqual(new.sections[0].ground_material,'brick_pavement')
        self.assertEqual(new.sections[1].ground_material,'concrete_pavement')

    def test_invalid_eye_location(self):
        r=self.route(); r.sections[0].eye_offset_m=100
        with self.assertRaises(ValueError):r.sections[0].validate()

    def test_pareto_indices(self):
        from streetlux.scenarios import pareto_front
        rows=[{'objectives':{'x':1,'y':2}},{'objectives':{'x':4,'y':4}},{'objectives':{'x':2,'y':1}}]
        self.assertEqual(pareto_front(rows,('x','y')),[1])

@unittest.skipUnless((SOURCE/'specdebt').exists(),'SpecDebt not included')
class SpecDebtChecks(unittest.TestCase):
    def example(self):
        from specdebt.spec import Specification
        from specdebt.study import run_study,Threshold
        sp=Specification(name='interaction');sp.add('a',(0,1),'Binary levels for interaction regression');sp.add('b',(0,1),'Binary levels for interaction regression');sp.add('inert',(0,1),'Binary levels for interaction regression')
        return run_study(lambda s:s['a']^s['b'],sp,Threshold(.5),'XOR')

    def test_interactions_require_joint_declaration(self):
        r=self.example();self.assertEqual(r.flip_rate(),.5)
        self.assertEqual(len(r.minimum_declaration_set(0)['set']),2)
        self.assertEqual([x['residual_flip_rate'] for x in r.sufficiency_curve()],[.5,.5,0,0])

    def test_fail_fast(self):
        from specdebt.study import run_study,Threshold
        sp=self.example().spec
        with self.assertRaises(ValueError):run_study(lambda s:float('nan'),sp,Threshold(.5),'bad')
        with self.assertRaises(RuntimeError):run_study(lambda s:1/0,sp,Threshold(.5),'bad')

    def test_ties_are_distinct(self):
        from specdebt.spec import Specification
        from specdebt.study import run_study,Comparison
        sp=Specification(name='ties');sp.add('x',(-1,0,1),'Negative, zero and positive comparison scores')
        r=run_study(lambda s:s['x'],sp,Comparison(),'comparison')
        self.assertEqual(r.debt()['comparison_outcomes'],{'A_wins':1,'B_wins':1,'ties':1})
        self.assertAlmostEqual(r.flip_rate(),1/3)

    def test_weighted_residual_is_not_cell_guarantee(self):
        r=self.example()
        self.assertEqual(r.cell_residuals(('a',))['max_cell_residual'],.5)
        self.assertEqual(r.cell_residuals(('a','b'))['max_cell_residual'],0)

if __name__=='__main__':unittest.main(verbosity=2)
