"""Pure aggregation checks; no Cantera solver is run."""
import unittest
import numpy as np
from mt.config import Conditions, FlameConditions, RelativeSquaredError, SquaredError
from mt.workflow import normalized_residuals, objective_score


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.conditions = {
            'znd': Conditions([1, 1], [300, 300], [1, 1], [1, 1], weights=[1, 3]),
            'flame': FlameConditions([1], [300], [1]),
            'ignition': Conditions([1, 1, 1], [1000]*3, [1]*3, [1]*3),
        }
        self.reference = {'znd': [{'T_ZND': 100}, {'T_ZND': 100}],
                          'flame': [{'laminar_flame_speed': 1}],
                          'ignition': [{'ignition_delay_time': 1}]*3}
        self.actual = {'znd': [{'T_ZND': 200}, {'T_ZND': 400}],
                       'flame': [{'laminar_flame_speed': 3}],
                       'ignition': [{'ignition_delay_time': 2}]*3}
        self.objectives = {key: RelativeSquaredError() for key in
                           ('T_ZND', 'laminar_flame_speed', 'ignition_delay_time')}

    def score(self, **kwargs):
        return objective_score(self.actual, self.reference, conditions=self.conditions,
                               objectives=self.objectives, **kwargs)

    def test_independent_counts_and_hierarchical_weights(self):
        self.assertAlmostEqual(self.score(), (7 + 4 + 1) / 3)
        self.assertAlmostEqual(self.score(branch_weights={'znd': 2, 'flame': 1, 'ignition': 1}), 19/4)
        residuals = normalized_residuals(self.actual, self.reference,
                                        conditions=self.conditions, objectives=self.objectives)
        self.assertEqual(residuals.shape, (6,))
        self.assertAlmostEqual(float(residuals @ residuals), self.score())

    def test_objective_weights_normalized_within_branch(self):
        self.objectives['T_flame'] = SquaredError(100, weight=3)
        self.reference['flame'][0]['T_flame'] = 1000
        self.actual['flame'][0]['T_flame'] = 1100
        self.assertAlmostEqual(self.score(), (7 + (4 + 3)/4 + 1)/3)

    def test_disabled_branch_needs_no_simulations(self):
        del self.actual['ignition']
        self.assertAlmostEqual(self.score(branch_weights={'znd': 1, 'flame': 0, 'ignition': 0}), 7)

    def test_bad_weights_and_counts_rejected(self):
        for weights in ({'znd': 1}, {'znd': 0, 'flame': 0, 'ignition': 0},
                        {'znd': float('nan'), 'flame': 1, 'ignition': 1}):
            with self.assertRaises(ValueError):
                self.score(branch_weights=weights)
        self.actual['znd'].pop()
        with self.assertRaises(ValueError):
            self.score()

    def test_unknown_or_inactive_objectives_rejected(self):
        self.objectives = {'unknown': RelativeSquaredError()}
        with self.assertRaises(ValueError):
            self.score()
        self.objectives = {'T_ZND': RelativeSquaredError(0)}
        with self.assertRaises(ValueError):
            self.score()

class BackendRoutingTests(unittest.TestCase):
    def test_openmp_lazy_initializer_and_future_protocol(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        from mt.search import evaluation_pool, worker_state
        import mt.search as search_module
        previous = search_module._WORKER_STATE, search_module._WORKER_FAILURES
        try:
            with patch('mt.executors.create_executor', return_value=ThreadPoolExecutor(1)) as factory:
                with evaluation_pool(1, abs, [[0, 1]], backend='openmp') as (pool, key):
                    function, bounds, compact = pool.submit(worker_state, key).result()
                    self.assertIs(function, abs)
                    np.testing.assert_array_equal(bounds, [[0, 1]])
                    self.assertFalse(compact)
                factory.assert_called_once_with('openmp', 1)
        finally:
            search_module._WORKER_STATE, search_module._WORKER_FAILURES = previous

    def test_unknown_backend_and_mpi_conflict(self):
        from mt.search import execution_backend, mpi_context
        with self.assertRaises(ValueError):
            execution_backend('threads')
        with mpi_context(object(), 1):
            self.assertEqual(execution_backend(), 'mpi')
            with self.assertRaises(ValueError):
                execution_backend('openmp')


if __name__ == '__main__':
    unittest.main()
