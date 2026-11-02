from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from mt.run_diagnostics import reference_key,fit_quality
from mt.iterative_znd import _configuration
from test_plog_adaptation import MODEL,REFERENCE

class DiagnosticsTests(unittest.TestCase):
    def test_cache_invalidates_on_reference_cases_solver_but_not_candidate(self):
        cfg=_configuration(dict(model=str(MODEL),reference=str(REFERENCE),output_dir='/tmp/cache-test',
            P_bar=[1],T0=298,phi=[1],t_end=[1e-5],fuel='C2H4',oxidizer='O2:1,N2:3.76'))
        key=reference_key(cfg)
        for field,val in [('P_bar',[5]),('t_end',[2e-5]),('steps_per_end',100),('rel_tol',1e-6)]:
            other=deepcopy(cfg);other[field]=val
            self.assertNotEqual(key,reference_key(other))
        other=dict(cfg,model='different-candidate.yaml',profile_points=50)
        self.assertEqual(key,reference_key(other))
        with TemporaryDirectory() as d:
            p=Path(d)/'ref.yaml';p.write_bytes(REFERENCE.read_bytes()+b'\n# changed\n')
            self.assertNotEqual(key,reference_key(dict(cfg,reference=str(p))))
    def test_fit_targets_separate_from_loss(self):
        rows=[dict(T_final=1020,ind_len_ZND=2,exo_len_ZND=.5,profile_rmse_K=10,components=[4,1,9,16])]
        refs=[dict(T_final=1000,ind_len_ZND=1,exo_len_ZND=1)]
        r=fit_quality(rows,refs,dict(induction_percent=10))
        self.assertFalse(r['passed']);self.assertAlmostEqual(sum(r['loss_contributions'].values()),7.5)
        self.assertEqual(r['mean_absolute_errors']['induction_percent'],100)
        self.assertIsNone(fit_quality(rows,refs)['passed'])
