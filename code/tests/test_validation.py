"""Validation API contracts: units, case grids, partial failures and cached plotting."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from mt.validation import Validation, ValidationResult, _cases
from mt.validation_worker import _run


class ValidationTests(unittest.TestCase):
    def test_cartesian_grid(self):
        rows = _cases(dict(pressure_atm=[1,10],temperature_K=[1400,1600],phi=[.6,1,1.3]),'ignition')
        self.assertEqual(len(rows),12)
        self.assertEqual(rows[-1]['P_atm'],10)

    def test_znd_pairing_and_bar_conversion(self):
        rows = _cases(dict(pressure_bar=[1,5],temperature_K=298,phi=[.5,1],end_time_s=[1e-4,1e-5]),'znd')
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(rows[1]['P_atm']*101325,5e5)
        self.assertEqual(rows[1]['T'],298)

    def test_mismatched_znd_rejected(self):
        with self.assertRaises(ValueError):
            _cases(dict(pressure_bar=[1,5,20],temperature_K=298,phi=[.5,1],end_time_s=1e-4),'znd')

    def test_bad_units_or_values(self):
        for values in (dict(pressure_atm=[1],pressure_bar=[1],temperature_K=298,phi=1),
                       dict(pressure_atm=[],temperature_K=298,phi=1),
                       dict(pressure_atm=[1],temperature_K=np.nan,phi=1)):
            with self.assertRaises(ValueError): _cases(values,'flame')

    def test_model_labels_and_worker_count(self):
        for models,workers in [({},1),({'Reference':'x'},1),({'Tuned':'x'},0)]:
            with self.assertRaises(ValueError): Validation('x',models,'H2',workers=workers)
        self.assertEqual(len(Validation('x',{str(i):'x' for i in range(5)},'H2').models),6)

    def test_partial_failure_retained(self):
        cfg=dict(kind='ignition',cases=[{},{}],models={'Reference':'x','Tuned':'y'},workers=1)
        with patch('mt.validation_worker._solve',side_effect=[{'a':1},{'a':2},{'error':'bad'},{'a':4}]),redirect_stdout(StringIO()):
            result = _run(cfg)
        self.assertEqual(result['Tuned'][1],{'a':4})
        self.assertEqual(result['Tuned'][0]['error'],'bad')

    def test_summary_and_cached_plot_with_gap(self):
        cases = _cases(dict(pressure_atm=[1],temperature_K=298,phi=[.7,1,1.4]),'flame')
        row = dict(laminar_flame_speed=1.,T_flame=2000.)
        rows = {'Reference':[row]*3,'Tuned':[dict(laminar_flame_speed=2.,T_flame=2010.),{'error':'bad'},row]}
        with tempfile.TemporaryDirectory() as tmp:
            result = ValidationResult('flame',cases,rows,Path(tmp))
            stream = StringIO()
            with redirect_stdout(stream): result.print_summary()
            self.assertIn('0.693147',stream.getvalue())
            self.assertIn('FAILED',stream.getvalue())
            self.assertEqual(len(result.failures),1)
            from matplotlib.axes import Axes
            original = Axes.plot
            series=[]
            def record(ax,*args,**kwargs):
                series.append(args)
                return original(ax,*args,**kwargs)
            with patch.object(Axes,'plot',record): paths=result.plot()
            self.assertTrue(paths[0].is_file())
            self.assertTrue(np.isnan(series[1][1][1]))
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_znd_profile_plot(self):
        cases = _cases(dict(pressure_bar=1,temperature_K=298,phi=1,end_time_s=1e-4),'znd')
        row = dict(distance_m=[0,.001,.002],temperature_K=[1500,2000,2500],T_ZND=2500,ind_len_ZND=.001,exo_len_ZND=.0002)
        with tempfile.TemporaryDirectory() as tmp:
            result=ValidationResult('znd',cases,{'Reference':[row],'Tuned':[row]},Path(tmp))
            self.assertTrue(result.plot()[0].is_file())


if __name__=='__main__': unittest.main()
