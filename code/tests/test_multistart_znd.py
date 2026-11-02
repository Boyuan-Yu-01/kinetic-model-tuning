"""Workflow checks use synthetic candidates; no combustion jobs are launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from mt.multistart_znd import run_multistart, resolve_settings
from mt.iterative_znd import _score

class WorkflowTests(unittest.TestCase):
    def test_unscaled_lengths(self):
        ref=dict(grid_T=np.array([100.]),profile_scale=5.,T_final=100.,ind_len_ZND=1.,exo_len_ZND=1.)
        row=dict(grid_T=[100.],T_final=100.,ind_len_ZND=np.e,exo_len_ZND=np.e**2)
        residual,_=_score([row],[ref],dict(length_factor=None,temperature_scale=25.))
        self.assertAlmostEqual(float(residual@residual),1.25)

    def test_separate_starts_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            def explore(cfg):
                folder=Path(cfg['output_dir']);folder.mkdir()
                (folder/'verified_candidates.json').write_text(json.dumps([
                    dict(loss=5.,coordinates=[1]),dict(loss=6.,coordinates=[2]),dict(loss=7.,coordinates=[3])]))
            def optimize(cfg):
                folder=Path(cfg['output_dir']);folder.mkdir()
                x=cfg['initial_coordinates'][0]
                if x==2: raise RuntimeError('synthetic failure')
                (folder/'best_model.yaml').write_text(str(x))
                return dict(status='completed',validation={'passed':True},
                    stages=[dict(best_loss=4./x,stop_reason='loss_plateau')])
            cfg=dict(output_dir=str(root),multistart={'starts':3},workers=8,stages=[{}])
            with patch('mt.multistart_znd.stage_config',side_effect=lambda c,s,x,o:dict(output_dir=str(o))), \
                 patch('mt.multistart_znd.run_iterative_znd',side_effect=explore), \
                 patch('mt.multistart_znd.run_staged_iterative_znd',side_effect=optimize):
                result=run_multistart(cfg)
            self.assertEqual(result['best_start'],'start_03')
            self.assertEqual(result['starts'][1]['status'],'failed')
            self.assertEqual((root/'best_model.yaml').read_text(),'3')
            self.assertTrue((root/'start_01/best_model.yaml').exists())

if __name__=='__main__': unittest.main()

class ObjectiveSelectionTests(unittest.TestCase):
    def test_selected_losses(self):
        ref=dict(grid_T=np.array([100.,100.]),profile_scale=25.,T_final=100.,ind_len_ZND=1.,exo_len_ZND=1.)
        row=dict(grid_T=[125.,125.],T_final=150.,ind_len_ZND=np.e,exo_len_ZND=np.e**2)
        for names,expected in [(['T_profile','T_ZND'],2.5),(['T_ZND','ind_len_ZND','exo_len_ZND'],3.)]:
            r,_=_score([row],[ref],dict(objective_names=names,length_factor=None,temperature_scale=25.))
            self.assertAlmostEqual(float(r@r),expected)
