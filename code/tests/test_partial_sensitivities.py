import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from mt.iterative_sensitivity import iterative_sensitivity_optimize

class PartialTests(unittest.TestCase):
    def run_case(self, recover):
        calls=[];screens=[]
        def jac(evaluate,x,r,lo,hi,active,*args):
            screens.append(list(active))
            stable=recover and len(screens)>=3
            return np.array([[1.,1.]]),[dict(stable=True),dict(stable=stable)]
        def inner(evaluate,x,lo,hi,active,*args,**kwargs):
            calls.append(list(active));new=x.copy()
            if len(calls)>1:new[active]-=.1
            return SimpleNamespace(x=new,residual=np.array([new.sum()]),status='maximum_iterations',converged=False,iterations=1,diagnostics={})
        with patch('mt.iterative_sensitivity.adaptive_jacobian',side_effect=jac),patch('mt.iterative_sensitivity.sensitivity_trust_region',side_effect=inner):
            result=iterative_sensitivity_optimize(lambda ps:[np.array([p.sum()]) for p in ps],np.array([1.,1.]),[-10.,-10.],[10.,10.],[1,2],max_rounds=3,sensitivity_retries=0,top_reactions=2)
        return result,calls,screens
    def test_retry_after_no_progress_and_recover(self):
        result,calls,screens=self.run_case(True)
        self.assertEqual(calls,[[0],[0],[0,1]])
        self.assertTrue(all(s==[0,1] for s in screens))
        self.assertEqual(result['rounds'][0]['skipped_coordinates'],[1])
        self.assertEqual(result['rounds'][0]['status'],'iterating')
    def test_missing_coordinate_stays_fixed_without_false_convergence(self):
        result,calls,_=self.run_case(False)
        self.assertTrue(all(c==[0] for c in calls))
        self.assertEqual(result['bestx'][1],1.)
        self.assertFalse(result['converged'])
