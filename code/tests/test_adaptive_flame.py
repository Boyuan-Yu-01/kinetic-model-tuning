"""Structural trials retain accepted anchors and reject unhelpful changes."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import unittest
from mt.adaptive_flame import run_adaptive_flame

class AdaptiveFlameTests(unittest.TestCase):
    def test_trial_acceptance_and_rollback(self):
        for trial_loss, expected_model in [(5.,'plog'),(12.,'base')]:
            with self.subTest(trial_loss=trial_loss), TemporaryDirectory() as tmp:
                cfg=dict(output_dir=str(Path(tmp)/'out'),model='base',reference='reference',artificial=[2],
                    uncertainty_factors={},a_ratio=[.01,100],ea_delta=1000,b_delta=.2,max_rounds=2,
                    stages=[dict(parameters=s,a_ratio=[.5,2]) for s in ['A','AE','AEB']],plog_candidates=[2])
                calls=[]
                def runner(settings):
                    calls.append(settings)
                    p=Path(settings['output_dir']);p.mkdir(parents=True)
                    (p/'best_model.yaml').write_text('model')
                    trial='plog_trial' in p.name
                    return dict(space=Mock(),coordinates=[1.],loss=trial_loss if trial else 10.,
                        validation_passed=True,optimization=dict(status='stalled',screen={}))
                with patch('mt.adaptive_flame.rank_plog_candidates',return_value=[2]),patch('mt.adaptive_flame.add_plog',return_value='plog'),patch('mt.adaptive_flame.StagedParameters'),patch('mt.adaptive_flame.transfer_coordinates',return_value=[1.]),patch('mt.adaptive_flame.check_rate_equivalence',return_value={'valid':True}):
                    run_adaptive_flame(cfg,runner)
                self.assertEqual([c['stages'][0]['parameters'] for c in calls],['A','A','AE','AEB'])
                self.assertEqual(calls[-1]['model'],expected_model)
                self.assertEqual(calls[-1]['reference'],'reference')
                self.assertFalse(calls[-1]['enable_plog'])
