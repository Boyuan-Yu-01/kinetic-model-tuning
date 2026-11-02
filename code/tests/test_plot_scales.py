"""Check configurable display scales without solving combustion cases."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from mt.plot_settings import yscale
from mt.flame_tuning import save_flame_comparison
from mt.ignition_cases import save_ignition_plot
from mt.iterative_znd import _plots

class PlotScaleTests(unittest.TestCase):
    def test_defaults_and_override(self):
        self.assertEqual(yscale({},'T_profile'),'linear')
        self.assertEqual(yscale({},'ignition_delay_time','log'),'log')
        self.assertEqual(yscale({'T_profile':'log'},'T_profile'),'log')
        with self.assertRaises(ValueError):yscale({'T_profile':'bad'},'T_profile')

    def test_plotters_use_scales(self):
        import matplotlib.axes
        original=matplotlib.axes.Axes.set_yscale
        calls=[]
        def record(ax,value,*args,**kwargs):
            calls.append(value);return original(ax,value,*args,**kwargs)
        with TemporaryDirectory() as tmp,patch.object(matplotlib.axes.Axes,'set_yscale',record):
            root=Path(tmp)
            save_ignition_plot([dict(pressure_atm=1,temperature_K=1500,phi=1,ignition_delay_s=1e-4)],root/'idt.png')
            self.assertIn('log',calls);calls.clear()
            row=dict(distance=[0.,1.,2.,3.],T=[300.,1000.,2000.,2000.])
            _plots(root,[row],[row],[row],dict(P_bar=[1],phi=[1],plot_yscales={'T_profile':'log'}))
            self.assertTrue(calls and all(v=='log' for v in calls));calls.clear()
            rows=[dict(P_atm=1,phi=1,reference_cm_s=30,model_cm_s=29,reference_temperature_K=2000,model_temperature_K=2001)]
            save_flame_comparison(root,rows,rows,scales={'laminar_flame_speed':'log','T_flame':'linear'})
            self.assertEqual(calls,['log','linear'])
