"""Stopping-policy checks with deterministic, inexpensive optimizer doubles."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from mt.iterative_sensitivity import iterative_sensitivity_optimize


class StagnationTests(unittest.TestCase):
    def optimize(self, reductions, *, stable=True, inner_status='maximum_iterations', move=.1, **kwargs):
        loss = 1.
        reductions = iter(reductions)

        def inner(evaluate, x, *args, **kw):
            nonlocal loss
            loss *= 1. - next(reductions)
            return SimpleNamespace(x=x + move, residual=np.array([np.sqrt(loss)]),
                                   status=inner_status, converged=False, iterations=1, diagnostics=[])

        def jac(*args, **kw):
            return np.array([[1.]]), [{'stable': stable}]

        with patch('mt.iterative_sensitivity.sensitivity_trust_region', side_effect=inner), \
             patch('mt.iterative_sensitivity.adaptive_jacobian', side_effect=jac):
            return iterative_sensitivity_optimize(lambda points: [np.array([1.]) for _ in points],
                [1.], [-10.], [10.], ['A'], sensitivity_retries=0, **kwargs)

    def test_default_stops_after_three_negligible_reductions(self):
        result = self.optimize([1e-6] * 3)
        self.assertEqual(result['status'], 'loss_plateau')
        self.assertFalse(result['converged'])
        self.assertEqual(len(result['rounds']), 3)
        record = result['rounds'][-1]
        self.assertEqual(record['stagnation_count'], 3)
        self.assertAlmostEqual(record['relative_loss_reduction'], 1e-6)
        self.assertEqual(record['inner_status'], 'maximum_iterations')
        self.assertEqual(record['convergence_blockers'][0]['parameter_indices'], [0])

    def test_material_improvement_resets_counter(self):
        result = self.optimize([1e-6, .1, 1e-6, 1e-6, 1e-6])
        self.assertEqual([r['stagnation_count'] for r in result['rounds']], [1, 0, 1, 2, 3])

    def test_disabled_and_custom_threshold(self):
        result = self.optimize([1e-6] * 4, stagnation_rounds=0, max_rounds=4)
        self.assertEqual(result['status'], 'maximum_rounds')
        result = self.optimize([.001] * 2, stagnation_rounds=2, stagnation_relative_tolerance=.002)
        self.assertEqual(result['status'], 'loss_plateau')

    def test_convergence_takes_precedence(self):
        result = self.optimize([1e-6], move=.001, stagnation_rounds=1)
        self.assertTrue(result['converged'])
        self.assertEqual(result['status'], 'parameter_change')
        self.assertEqual(result['rounds'][0]['convergence_blockers'], [])

    def test_unresolved_sensitivities_are_not_hidden(self):
        result = self.optimize([1e-6], stable=False, stagnation_rounds=1)
        self.assertEqual(result['status'], 'unresolved_sensitivities')
        self.assertIn('unresolved_sensitivities', [b['reason'] for b in result['rounds'][0]['convergence_blockers']])

    def test_inner_failure_remains_explicit(self):
        result = self.optimize([1e-6], move=.001, inner_status='loss_stalled', stagnation_rounds=1)
        self.assertFalse(result['converged'])
        self.assertEqual(result['rounds'][0]['convergence_blockers'],
                         [{'reason': 'inner_solver_status', 'status': 'loss_stalled'}])

    def test_zero_update_requires_stationarity(self):
        result = self.optimize([0.], move=0.)
        self.assertEqual(result['status'], 'stalled')
        self.assertIn('zero_update_requires_stationarity', [b['reason'] for b in result['rounds'][0]['convergence_blockers']])

    def test_invalid_settings(self):
        for name, values in [('stagnation_rounds', [-1, 1.5, True]),
                             ('stagnation_relative_tolerance', [-1., float('nan'), float('inf'), 1., True])]:
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.optimize([], **{name: value})


if __name__ == '__main__':
    unittest.main()
