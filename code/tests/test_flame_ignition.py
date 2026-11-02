"""Joint residual weighting and fixed-horizon case mapping."""
import unittest
from unittest.mock import patch
import numpy as np
from mt.flame_ignition import combined_residual, resolve_combined_settings

class JointTests(unittest.TestCase):
    def test_each_branch_has_equal_weight(self):
        refs=[dict(speed_m_s=1.,burnt_temperature_K=2000.)]*15+[dict(ignition_delay_s=1.,burnt_temperature_K=2000.)]*27
        rows=[dict(speed_m_s=np.e,burnt_temperature_K=2025.)]*15+[dict(ignition_delay_s=np.e**2,burnt_temperature_K=2050.)]*27
        cfg=dict(cases=[{}]*15+[{'kind':'ignition'}]*27,objectives=dict(temperature_scale_K=25.),ignition_objectives=dict(delay_weight=1.,temperature_weight=1.,temperature_scale_K=25.),branch_weights=dict(flame=1.,ignition=1.))
        r=combined_residual(rows,refs,cfg)
        self.assertAlmostEqual(r@r,2.5)
        rows[-1]={'error':'failed'}
        self.assertIsNone(combined_residual(rows,refs,cfg))

    def test_fixed_horizons(self):
        p=dict(conditions={'ignition':dict(pressure_atm=[1,10],temperature_K=[1400],phi=[.6,1.])},objectives=dict(ignition_delay_time={'loss':'log_squared'},T_ignition={'loss':'squared','scale':25}),ignition_solver={"end_time":.003})
        with patch('mt.flame_settings.resolve_flame_settings',return_value={'cases':[{}]}):c=resolve_combined_settings(p,{}, {})
        self.assertEqual([(x['P_atm'],x['phi']) for x in c['cases'][1:]],[(1,.6),(1,1.),(10,.6),(10,1.)])
        np.testing.assert_allclose([x['end_time_s'] for x in c['cases'][1:]],[.003]*4)
