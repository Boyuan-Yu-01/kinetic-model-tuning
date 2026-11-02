"""Convex constrained-step regressions; no chemical solvers are invoked."""
import unittest
import numpy as np
from mt.constrained_steps import least_squares_branches, gradient_mapping


class RedundantConstraintTests(unittest.TestCase):
    @staticmethod
    def branch(repeats):
        # A narrow coupled kinetic envelope with redundant temperature cuts.
        # For residual [g,g], symmetry gives the exact optimum (-.1,-.1).
        rows = np.array([[1.,1.],[-1.,-1.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        bounds = np.array([-.2,-.2,-1.,-1.,-1.,-1.])
        return np.tile(rows,(repeats,1)),np.tile(bounds,repeats)

    def test_large_residual_redundant_envelope_has_known_feasible_optimum(self):
        for repeats in (1,33):
            with self.subTest(repeats=repeats):
                c,b = self.branch(repeats)
                steps = least_squares_branches(np.eye(2),np.array([1e4,1e4]),
                    np.zeros(2),[0,1],np.array([-2.,-2.]),np.array([2.,2.]),[(c,b)])
                self.assertEqual(len(steps),1)
                np.testing.assert_allclose(steps[0],[-.1,-.1],rtol=0,atol=1e-7)
                self.assertTrue(np.all(c@steps[0] >= b-1e-10))

    def test_slsqp_failure_uses_feasible_quadratic_fallback(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from scipy.optimize import minimize as real_minimize
        def solver(*args, **kwargs):
            if kwargs.get('method') == 'SLSQP':
                return SimpleNamespace(success=False, message='Positive directional derivative for linesearch')
            return real_minimize(*args, **kwargs)
        c,b=self.branch(3)
        with patch('mt.constrained_steps.minimize', side_effect=solver):
            steps=least_squares_branches(np.eye(2),np.ones(2),np.zeros(2),[0,1],
                np.array([-2.,-2.]),np.array([2.,2.]),[(c,b)])
        np.testing.assert_allclose(steps[0],[-.1,-.1],atol=1e-6)
        self.assertTrue(np.all(c@steps[0]>=b-1e-10))

    def test_gradient_mapping_does_not_report_false_stationarity(self):
        c,b = self.branch(33)
        projected = gradient_mapping(np.zeros(2),[0,1],np.array([1e4,1e4]),
            np.array([-2.,-2.]),np.array([2.,2.]),lambda:[(c,b)])
        np.testing.assert_allclose(projected,[.1,.1],rtol=0,atol=1e-7)

    def test_infeasible_branch_is_skipped_without_losing_feasible_branch(self):
        c,b = self.branch(33)
        impossible=(np.array([[1.,0.],[-1.,0.]]),np.array([3.,3.]))
        steps=least_squares_branches(np.eye(2),np.ones(2),np.zeros(2),[0,1],
            np.array([-2.,-2.]),np.array([2.,2.]),[impossible,(c,b)])
        self.assertEqual(len(steps),1)
        np.testing.assert_allclose(steps[0],[-.1,-.1],rtol=0,atol=1e-7)


if __name__ == '__main__':
    unittest.main()

class FeasibilityRepairTests(unittest.TestCase):
    def test_captured_lfs_step(self):
        from pathlib import Path
        d=np.load(Path(__file__).parent/'fixtures/lfs_failed_step.npz')
        c=np.zeros((len(d['rhs']),len(d['x'])))
        c[:,d['active']]=d['a']
        steps=least_squares_branches(d['model'],d['residual'],d['x'],d['active'],
            d['low'],d['high'],[(c,d['rhs']+c@d['x'])],float(d['damping']))
        self.assertTrue(steps)
        step=steps[0]
        self.assertTrue(np.all(d['a']@step>=d['rhs']-1e-12))
        self.assertTrue(np.all(step>=d['low']-1e-12))
        self.assertTrue(np.all(step<=d['high']+1e-12))
        self.assertGreater(float(d['residual']@d['residual']-np.sum((d['residual']+d['model']@step)**2)),0)

    def test_projection_failure_is_not_stationarity(self):
        from unittest.mock import patch
        from mt.constrained_steps import ConstrainedStepError
        with patch('mt.constrained_steps.least_squares_branches',side_effect=ConstrainedStepError('test')):
            g=gradient_mapping(np.zeros(1),[0],np.ones(1),-np.ones(1),np.ones(1),lambda:[])
        self.assertTrue(np.isinf(g).all())

    def test_failed_update_stalls_without_crashing(self):
        from unittest.mock import patch
        from mt.constrained_steps import ConstrainedStepError
        from mt.sensitivity_search import sensitivity_trust_region
        evaluate=lambda points:[np.array([x[0]-1.]) for x in points]
        with patch('mt.sensitivity_search.least_squares_branches',side_effect=ConstrainedStepError('test')):
            result=sensitivity_trust_region(evaluate,np.zeros(1),-np.ones(1)*2,np.ones(1)*2,[0],np.array([.01]),
                constraints=lambda:[(np.ones((1,1)),np.array([-2.]))],max_iterations=2)
        self.assertEqual(result.status,'trust_region_stalled')
        self.assertFalse(result.converged)
