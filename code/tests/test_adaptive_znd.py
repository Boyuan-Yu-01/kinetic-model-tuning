"""Mock the chemical solves; exercise adaptive orchestration and public artifacts."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
import runpy
import sys
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from mt.adaptive_znd import run_adaptive_znd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE=ROOT/'ethylene_ZND'
if not EXAMPLE.exists():EXAMPLE=ROOT.parent/'ethylene_ZND'


class AdaptiveTests(unittest.TestCase):
    def run_case(self, *, trial_loss=5., trial_passed=True, enabled_candidate=True):
        calls, insertions = [], []
        original = ROOT/'ethylene_ZND/inputs/model_populated.yaml'
        config = dict(model=original, reference=ROOT/'ethylene_ZND/inputs/FFCM2.yaml',
                      enable_plog=True, plog_free=None, unbounded_reactions=[2], plog_candidates=[2],
                      plog_min_improvement=.001, initial_search={'samples':10},
                      stages=[dict(parameters=p,a_ratio=(.5,2.)) for p in ('A','AE','AEB')])
        def make_cfg(active, settings, center, output):
            return dict(deepcopy(active), stage=settings['parameters'],stage_center=center,output_dir=str(output))
        def registry(cfg):
            split = bool(cfg.get('plog_free'))
            pressures = [1e5,5e6,1e7] if split else [None]
            labels = [dict(reaction=2,parameter='log_k_pivot',pressure_pa=p) for p in pressures]
            labels += [dict(reaction=2,parameter=p,pressure_pa=None,scale=1.) for p in ('scaled_ea','scaled_b')]
            return SimpleNamespace(coordinate_labels=labels,pivot=1800.,scheme=lambda x:
                {'reactions':[{}, {'equation':'A => B','rate-constant':{'A':1.,'b':0.,'Ea':1.}}]})
        def insert(source, output, entries):
            self.assertEqual(len(calls),1, 'PLOG must be proposed only after the first fit.')
            self.assertEqual(Path(source),original, 'New anchors must come from the human original.')
            insertions.append((source,entries))
            Path(output).parent.mkdir(parents=True,exist_ok=True)
            Path(output).write_text('expanded anchor')
            return Path(output)
        def solve(cfg):
            # Reproduce the worker's truth-value expression; ndarray centers break it.
            tuple(cfg.get('stage_center') or [])
            calls.append(deepcopy(cfg))
            folder = Path(cfg['output_dir']);folder.mkdir(parents=True)
            (folder/'run_info').mkdir()
            (folder/'run_info/initial_monotonicity.json').write_text('{"temperature_range_K":[300,3000]}')
            trial = folder.name.startswith('plog_trial')
            center = cfg.get('stage_center')
            coordinates = [.2,0.,0.] if center is None else list(center)
            if trial:
                self.assertEqual(coordinates,[.2,.2,.2,0.,0.])
                coordinates[0] += .01
            passed = trial_passed if trial else True
            (folder/'best_model.yaml').write_text(folder.name)
            (folder/'results.yaml').write_text(yaml.safe_dump({'validation':{'passed':passed},'cases':[]}))
            return dict(status='validation_failed' if passed is False else 'not_converged',
                        coordinates=coordinates,baseline_loss=10.,best_loss=trial_loss if trial else 10.,
                        validation={'performed':passed is not None,'passed':passed},
                        optimization=dict(converged=False,status='max_rounds',rounds=[],screen={'ranking':[
                            dict(group=2,score=1.,all_resolved=enabled_candidate,reliable_coordinates=[0])]}))
        with TemporaryDirectory() as folder:
            config['output_dir']=str(Path(folder)/'output')
            with patch('mt.adaptive_znd.stage_config',side_effect=make_cfg), \
                 patch('mt.adaptive_znd._registry',side_effect=registry), \
                 patch('mt.adaptive_znd.add_plog',side_effect=insert), \
                 patch('mt.adaptive_znd.check_rate_equivalence',return_value={'valid':True}), \
                 patch('mt.adaptive_znd.run_iterative_znd',side_effect=solve):
                result=run_adaptive_znd(config)
            self.assertEqual((Path(config['output_dir'])/'best_model.yaml').read_text(),'03_AEB')
            decision=yaml.safe_load((Path(config['output_dir'])/'plog_decision.json').read_text())
        return calls,insertions,result,decision

    def test_retains_improving_validated_trial_and_transfers_stages(self):
        calls,insertions,result,decision=self.run_case()
        self.assertEqual(len(insertions),1)
        self.assertTrue(decision['accepted'])
        self.assertEqual([c['stage'] for c in calls],['A','A','AE','AEB'])
        np.testing.assert_allclose(calls[2]['stage_center'],[.21,.2,.2,0.,0.])
        self.assertEqual(calls[2]['model'],calls[1]['model'])
        self.assertEqual(calls[3]['model'],calls[1]['model'])
        self.assertNotIn('initial_search',calls[1])
        self.assertEqual(result['status'],'completed')

    def test_rejects_no_gain_or_failed_or_missing_validation(self):
        for loss,passed in ((10.,True),(5.,False),(5.,None)):
            with self.subTest(loss=loss,passed=passed):
                calls,_,_,decision=self.run_case(trial_loss=loss,trial_passed=passed)
                self.assertFalse(decision['accepted'])
                self.assertEqual(calls[2]['model'],calls[0]['model'])
                self.assertEqual(calls[2]['stage_center'],[.2,0.,0.])

    def test_unresolved_sensitivity_skips_structure_change(self):
        calls,insertions,_,decision=self.run_case(enabled_candidate=False)
        self.assertEqual(insertions,[])
        self.assertEqual([c['stage'] for c in calls],['A','AE','AEB'])
        self.assertFalse(decision['accepted'])

    def test_example_entrypoint_does_not_eagerly_insert_plog(self):
        settings=ModuleType('settings');settings.problem={'model':'human-original'};settings.tuning={'plog':{'enable_plog':True}};settings.execution={}
        saved_path=list(sys.path)
        try:
            with patch.dict(sys.modules,{'settings':settings}), \
                 patch('mt.multistart_znd.run_tuning') as launch, \
                 patch('mt.initialization.add_plog',side_effect=AssertionError('Eager insertion')):
                runpy.run_path(str(EXAMPLE/'run.py'))
                launch.assert_called_once()
                problem,tuning,execution=launch.call_args.args
                self.assertEqual(Path(problem['model']).name,'model_populated.yaml')
                self.assertTrue(tuning['plog'].get('enabled', tuning['plog'].get('enable_plog')))
                self.assertGreaterEqual(tuning['multistart']['starts'], 1)
                self.assertLessEqual(tuning['multistart']['starts'], tuning['exploration']['keep_candidates'])
        finally:
            sys.path[:]=saved_path


if __name__=='__main__':
    unittest.main()
