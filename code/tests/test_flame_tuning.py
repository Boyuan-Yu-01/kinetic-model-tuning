"""Flame objective and example configuration checks without running optimization."""
import unittest
import numpy as np
from mt.flame_tuning import flame_residual
class FlameTuningTests(unittest.TestCase):
    def test_equal_case_log_loss(self):
        r=flame_residual([{'speed_m_s':np.e},{'speed_m_s':np.e**2}], [{'speed_m_s':1.}]*2)
        self.assertAlmostEqual(float(r@r),2.5)
        self.assertIsNone(flame_residual([{'error':'failed'}],[{'speed_m_s':1.}]))

    def test_speed_and_temperature_normalization(self):
        refs=[dict(speed_m_s=1.,burnt_temperature_K=2000.)]
        rows=[dict(speed_m_s=np.e,burnt_temperature_K=2025.)]
        residual=flame_residual(rows,refs,temperature_scale_K=25.)
        self.assertAlmostEqual(float(residual@residual),1.)
        rows[0]['speed_m_s']=1.
        residual=flame_residual(rows,refs,temperature_scale_K=25.,speed_weight=1.,temperature_weight=3.)
        self.assertAlmostEqual(float(residual@residual),.75)

    def test_module_launcher_dispatches_importable_worker(self):
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        root=Path(__file__).resolve().parents[1]
        backends=['process']
        if (root/'native/openmp_runner').is_file(): backends.append('openmp')
        for backend in backends:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'output'
                request=Path(tmp)/'request.json'
                # A deliberately missing input should return a structured solver
                # error through the worker, not fail while unpickling its callable.
                cfg=dict(output_dir=str(output),backend=backend,workers=2,
                    reference=str(Path(tmp)/'missing.yaml'),
                    solver=dict(timeout_seconds=10),
                    cases=[dict(T=298.,P_atm=1.,phi=1.)])
                request.write_text(json.dumps(cfg))
                env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
                result=subprocess.run([sys.executable,'-m','mt.flame_tuning',str(request)],
                    capture_output=True,text=True,env=env,timeout=60)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Reference flames failed',result.stderr)
                self.assertNotIn("Can't get attribute",result.stderr)
                rows=json.loads((output/'FFCM2_results.json').read_text())
                self.assertIn('CanteraError',rows[0]['error'])
