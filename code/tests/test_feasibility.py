"""Mocked search checks: no chemical integration is performed."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from mt.feasibility import find_feasible


class FeasibilityTests(unittest.TestCase):
    def make_space(self):
        origin = np.array([.1, .2, .2, .2, -.3])
        return SimpleNamespace(base=SimpleNamespace(reaction_numbers=(1,2,2,2,4), source_coordinates=origin),
            x0=np.r_[origin,0.,0.], na=5, lower=np.r_[np.full(5,-10.),0.,0.],
            upper=np.r_[np.full(5,10.),0.,0.], a_low=np.full(5,-10.), a_high=np.full(5,10.),
            validate=lambda x:x)

    def run_search(self, space, *, fine_valid=True):
        config = dict(stage='A', plog_direction='either', steps_per_end=4000,
            rel_tol=1e-10, abs_tol=1e-14, max_rhs_calls=100000,
            initial_search=dict(a_ratio={2:[.01,100.]}, samples=9, batch_size=4, seed=42, finalists=3))
        before = deepcopy(config)
        calls = []
        records = []
        def batch(vectors, full=False, settings=None):
            calls.append((np.array(vectors).copy(),full,dict(settings)))
            return [[dict(value=float(np.linalg.norm(vector)), valid=fine_valid if full else True)] for vector in vectors]
        def scorer(rows, refs, cfg):
            return (np.array([rows[0]['value']]),[]) if rows[0]['valid'] else (None,[])
        with TemporaryDirectory() as folder, patch('mt.iterative_znd._score',side_effect=scorer), \
             patch('mt.iterative_znd._interval',return_value=[300.,4000.]), \
             patch('mt.iterative_znd._candidate_monotonicity',return_value={'valid':True}), \
             patch('mt.iterative_znd._save',side_effect=lambda p,r:records.append(deepcopy(r))):
            result = find_feasible(batch,space,[{}],config,Path(folder))
        self.assertEqual(config,before)
        return result,calls,records

    def test_shared_plog_multiplier_bounds_exclusions_and_immutable_source(self):
        space = self.make_space()
        # Narrow different nodes: shared source-relative interval is [-.1,.1].
        space.a_low[1] = .1
        space.a_high[3] = .3
        before = {key:getattr(space,key).copy() for key in ('x0','lower','upper','a_low','a_high')}
        origin = space.base.source_coordinates.copy()
        (chosen,_),calls,records = self.run_search(space)
        samples = np.concatenate([v for v,full,_ in calls if not full])
        shifts = samples[:,1:4]-origin[1:4]
        np.testing.assert_allclose(shifts[:,0],shifts[:,1],atol=0)
        np.testing.assert_allclose(shifts[:,0],shifts[:,2],atol=0)
        self.assertTrue(np.all(shifts >= -.1-1e-14))
        self.assertTrue(np.all(shifts <= .1+1e-14))
        np.testing.assert_array_equal(samples[:,[0,4,5,6]],np.tile(space.x0[[0,4,5,6]],(9,1)))
        for key,value in before.items(): np.testing.assert_array_equal(getattr(space,key),value)
        np.testing.assert_array_equal(space.base.source_coordinates,origin)
        self.assertTrue(any(full for _,full,_ in calls))
        self.assertTrue(all(settings['rel_tol']==1e-10 for _,full,settings in calls if full))
        self.assertEqual(records[-1]['status'],'feasible')
        second,_,_ = self.run_search(space)
        np.testing.assert_array_equal(chosen,second[0])

    def test_disjoint_plog_node_bounds_rejected(self):
        space = self.make_space()
        space.a_low[1] = 1.
        space.a_high[3] = 0.
        with self.assertRaisesRegex(ValueError,'shared search interval'):
            self.run_search(space)

    def test_non_source_center_rejected(self):
        space = self.make_space(); space.x0[2] += .1
        with self.assertRaisesRegex(ValueError,'original source center'):
            self.run_search(space)

    def test_no_accurate_candidate_is_not_accepted(self):
        with self.assertRaisesRegex(RuntimeError,'No feasible A-only'):
            self.run_search(self.make_space(),fine_valid=False)


if __name__ == '__main__':
    unittest.main()
