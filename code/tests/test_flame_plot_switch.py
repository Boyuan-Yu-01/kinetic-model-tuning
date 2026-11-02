"""The comparison flag affects figure output, not numerical result files."""
from pathlib import Path
import tempfile
import unittest
from mt.flame_tuning import save_flame_comparison

class PlotSwitchTests(unittest.TestCase):
    def test_enabled_and_disabled(self):
        rows=[dict(P_atm=p,phi=1.,reference_cm_s=30.,model_cm_s=29.,reference_temperature_K=2200.,model_temperature_K=2199.) for p in (1,10,50)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            result=root/'results.json';result.write_text('unchanged')
            save_flame_comparison(root,rows,rows,False)
            self.assertFalse((root/'comparison.png').exists())
            save_flame_comparison(root,rows,rows,True)
            self.assertGreater((root/'comparison.png').stat().st_size,0)
            self.assertEqual(result.read_text(),'unchanged')
